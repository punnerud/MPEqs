//! A real LRU over expert slabs, replayed against a real trace with real uncached reads.
//!
//! This is where layout and caching meet. On a miss the loader fetches the whole contiguous
//! *run* of selected experts, not one expert — so a clustered layout drags in neighbours the
//! token is about to want anyway. That prefetch-for-free effect exists only if the layout put
//! them together, and it is not measured anywhere in the MoE-offload literature.
//!
//! It also hosts the cache-aware routing policy. Given the cache state at the moment the
//! router fires, a cold expert with a small gate weight may not be worth a disk round trip.
//! That decision needs no prediction of the future — which matters, because we measured
//! cross-layer mutual information at 0.66 bits out of 5.96 and there is almost no future to
//! predict. It is a Lagrangian: fetch a cold expert when `w · lambda > fetch cost`. Sweeping
//! lambda traces the quality-versus-IO frontier.

use std::collections::HashMap;

/// Identifies one expert slab: which MoE layer, which physical slot.
pub type Slab = (u32, u32);

/// What decides which experts occupy a fixed memory budget.
///
/// LRU is the reflex, and for MoE experts it is close to the worst possible choice. LRU
/// assumes recency predicts reuse; load-balanced routing makes access nearly uniform, which
/// is the classic LRU pathology — every admission evicts something that is about to be needed
/// just as much as the newcomer. Under uniform access a cache holding fraction *f* of the data
/// should hit about *f* of the time, and LRU delivers far less than that.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Least recently used.
    Lru,
    /// Pin the globally most-frequent experts, measured on an earlier slice of the trace, and
    /// never evict. Nothing adapts, which is the point.
    StaticPinned,
    /// Random replacement — the baseline LRU has to beat to justify itself.
    Random,
    /// Pin the hottest experts into part of the budget and run random replacement over the
    /// rest. The natural compromise: keep what is globally hot, still adapt to whatever this
    /// particular session happens to use.
    Hybrid,
    /// Frequency with ageing. Every access adds to a slab's score, and every `halflife`
    /// accesses all scores are halved, so an expert that was hot and goes cold does fall out.
    /// This is what static pinning should become once the workload is not stationary: it
    /// keeps the frequency signal that makes pinning work, without freezing yesterday's
    /// answer.
    Decayed,
}

impl std::str::FromStr for Policy {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "lru" => Ok(Policy::Lru),
            "static" => Ok(Policy::StaticPinned),
            "random" => Ok(Policy::Random),
            "hybrid" => Ok(Policy::Hybrid),
            "decayed" => Ok(Policy::Decayed),
            other => Err(format!(
                "unknown policy '{other}' (lru, static, random, hybrid, decayed)"
            )),
        }
    }
}

pub struct Lru {
    cap_bytes: u64,
    used: u64,
    tick: u64,
    /// slab -> (last use tick, bytes)
    entries: HashMap<Slab, (u64, u64)>,
    /// Slabs that must never be evicted, whatever the policy.
    pinned: std::collections::HashSet<Slab>,
    /// Decayed access score per slab, for `Policy::Decayed`.
    score: HashMap<Slab, f64>,
    /// Accesses between halvings. 0 disables ageing.
    halflife: u64,
    since_decay: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl Lru {
    pub fn new(cap_bytes: u64) -> Self {
        Lru {
            cap_bytes,
            used: 0,
            tick: 0,
            entries: HashMap::new(),
            pinned: std::collections::HashSet::new(),
            score: HashMap::new(),
            halflife: 0,
            since_decay: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Enables frequency ageing with the given number of accesses per halving.
    pub fn with_halflife(mut self, halflife: u64) -> Self {
        self.halflife = halflife;
        self
    }

    /// Records an access for the ageing policy, whether or not it was a hit.
    pub fn observe(&mut self, s: Slab) {
        *self.score.entry(s).or_insert(0.0) += 1.0;
        if self.halflife == 0 {
            return;
        }
        self.since_decay += 1;
        if self.since_decay >= self.halflife {
            self.since_decay = 0;
            self.score.retain(|_, v| {
                *v *= 0.5;
                *v > 1e-3
            });
        }
    }

    /// Admits a slab, evicting the lowest-scoring entry. Used by `Policy::Decayed`.
    pub fn insert_by_score(&mut self, s: Slab, bytes: u64) {
        if self.entries.contains_key(&s) || bytes > self.cap_bytes {
            return;
        }
        while self.used + bytes > self.cap_bytes {
            let victim = self
                .entries
                .keys()
                .filter(|k| !self.pinned.contains(*k))
                .min_by(|a, b| {
                    let sa = self.score.get(*a).copied().unwrap_or(0.0);
                    let sb = self.score.get(*b).copied().unwrap_or(0.0);
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied();
            let Some(victim) = victim else { break };
            // Only admit if the newcomer is actually hotter than what it would displace.
            if self.score.get(&s).copied().unwrap_or(0.0)
                <= self.score.get(&victim).copied().unwrap_or(0.0)
            {
                return;
            }
            if let Some((_, b)) = self.entries.remove(&victim) {
                self.used -= b;
                self.evictions += 1;
            }
        }
        self.tick += 1;
        self.entries.insert(s, (self.tick, bytes));
        self.used += bytes;
    }

    #[cfg(test)]
    pub fn resident(&self, s: Slab) -> bool {
        self.entries.contains_key(&s)
    }

    /// Marks a slab as used now, counting a hit. Returns false if it was not resident.
    pub fn touch(&mut self, s: Slab) -> bool {
        self.tick += 1;
        let t = self.tick;
        match self.entries.get_mut(&s) {
            Some(e) => {
                e.0 = t;
                self.hits += 1;
                true
            }
            None => {
                self.misses += 1;
                false
            }
        }
    }

    /// Admits a slab, evicting until it fits. `victim_tick` picks what goes.
    pub fn insert_with(&mut self, s: Slab, bytes: u64, random: bool) {
        if self.entries.contains_key(&s) || bytes > self.cap_bytes {
            return;
        }
        while self.used + bytes > self.cap_bytes {
            // Pinned slabs are never candidates, so a hybrid cache cannot cannibalise its
            // own hot set to make room for a one-off.
            let victim = if random {
                let evictable: Vec<Slab> = self
                    .entries
                    .keys()
                    .filter(|k| !self.pinned.contains(*k))
                    .copied()
                    .collect();
                if evictable.is_empty() {
                    break;
                }
                // Deterministic pseudo-random choice, enough to characterise the policy.
                let idx =
                    (self.tick.wrapping_mul(6364136223846793005) >> 33) as usize % evictable.len();
                evictable[idx]
            } else {
                match self
                    .entries
                    .iter()
                    .filter(|(k, _)| !self.pinned.contains(*k))
                    .min_by_key(|(_, v)| v.0)
                {
                    Some((&k, _)) => k,
                    None => break,
                }
            };
            if let Some((_, b)) = self.entries.remove(&victim) {
                self.used -= b;
                self.evictions += 1;
            }
            self.tick += 1;
        }
        self.tick += 1;
        self.entries.insert(s, (self.tick, bytes));
        self.used += bytes;
    }

    /// Pins a slab so it is never evicted. Used to preload a static working set.
    pub fn pin(&mut self, s: Slab, bytes: u64) -> bool {
        if self.entries.contains_key(&s) || self.used + bytes > self.cap_bytes {
            return false;
        }
        self.tick += 1;
        self.entries.insert(s, (u64::MAX, bytes));
        self.pinned.insert(s);
        self.used += bytes;
        true
    }

    /// Admits a slab under strict LRU. Superseded by `insert_with`; kept for the tests.
    #[cfg(test)]
    pub fn insert(&mut self, s: Slab, bytes: u64) {
        if self.entries.contains_key(&s) {
            return;
        }
        if bytes > self.cap_bytes {
            // A single slab larger than the whole budget can never be cached; charging it
            // to `used` would evict everything forever.
            return;
        }
        while self.used + bytes > self.cap_bytes {
            let Some((&victim, _)) = self.entries.iter().min_by_key(|(_, v)| v.0) else {
                break;
            };
            if let Some((_, b)) = self.entries.remove(&victim) {
                self.used -= b;
                self.evictions += 1;
            }
        }
        self.tick += 1;
        self.entries.insert(s, (self.tick, bytes));
        self.used += bytes;
    }
}

/// How many distinct experts a sliding window of tokens touches, per layer.
///
/// The working set is what decides whether caching can work at all. If a 512-token document
/// touches 40 of 64 experts per layer there is nothing to cache; if it touches 15, residency
/// is the whole game.
pub fn working_set(tokens: &[&[u16]], window: usize) -> (f64, f64) {
    if tokens.is_empty() {
        return (0.0, 0.0);
    }
    let w = window.min(tokens.len()).max(1);
    let mut sizes = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let end = (i + w).min(tokens.len());
        let mut seen = std::collections::HashSet::new();
        for t in &tokens[i..end] {
            seen.extend(t.iter().copied());
        }
        sizes.push(seen.len() as f64);
        i += w;
    }
    let mean = sizes.iter().sum::<f64>() / sizes.len() as f64;
    let max = sizes.iter().cloned().fold(0.0f64, f64::max);
    (mean, max)
}

/// Splits sorted slots into contiguous runs, bridging gaps up to `max_gap`.
pub fn runs(slots: &[u32], max_gap: u64) -> Vec<(u32, u32)> {
    if slots.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = slots[0];
    let mut prev = slots[0];
    for &s in &slots[1..] {
        if (s - prev) as u64 - 1 > max_gap {
            out.push((start, prev));
            start = s;
        }
        prev = s;
    }
    out.push((start, prev));
    out
}

/// Caps a run so no single read crosses the size where the device falls off its curve.
///
/// Cold on this SSD, bandwidth still climbs to 32 MiB (1.82 GB/s at 1 MiB, 3.69 at 32 MiB),
/// but it climbs far slower than the byte count does, so merging past a few megabytes always
/// loses. The cap keeps a pathological layout from issuing one enormous read per layer.
pub fn split_at_limit(from: u32, to: u32, stride: u64, limit_bytes: u64) -> Vec<(u32, u32)> {
    if stride == 0 || limit_bytes == 0 {
        return vec![(from, to)];
    }
    let max_slots = (limit_bytes / stride).max(1) as u32;
    let mut out = Vec::new();
    let mut a = from;
    while a <= to {
        let b = (a + max_slots - 1).min(to);
        out.push((a, b));
        if b == u32::MAX {
            break;
        }
        a = b + 1;
    }
    out
}

/// Result of replaying one policy against one cache size.
#[derive(Debug, Clone, Default)]
pub struct CacheRun {
    pub tokens: u64,
    pub hits: u64,
    pub misses: u64,
    pub fetches: u64,
    pub bytes_from_disk: u64,
    /// Gate mass of experts the policy chose not to fetch, as a fraction of all gate mass.
    pub skipped_mass: f64,
    pub total_mass: f64,
    pub skipped_experts: u64,
    pub millis: f64,
}

impl CacheRun {
    pub fn hit_rate(&self) -> f64 {
        let t = self.hits + self.misses;
        if t == 0 {
            0.0
        } else {
            self.hits as f64 / t as f64
        }
    }
    pub fn mib_per_token(&self) -> f64 {
        self.bytes_from_disk as f64 / self.tokens.max(1) as f64 / (1 << 20) as f64
    }
    pub fn quality_cost_pct(&self) -> f64 {
        if self.total_mass <= 0.0 {
            0.0
        } else {
            100.0 * self.skipped_mass / self.total_mass
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_evicts_least_recently_used() {
        let mut c = Lru::new(300);
        c.insert((0, 0), 100);
        c.insert((0, 1), 100);
        c.insert((0, 2), 100);
        assert!(c.touch((0, 0))); // 1 is now the oldest
        c.insert((0, 3), 100);
        assert!(!c.resident((0, 1)));
        assert!(c.resident((0, 0)));
        assert!(c.resident((0, 3)));
        assert_eq!(c.evictions, 1);
    }

    #[test]
    fn oversized_slab_is_never_admitted() {
        let mut c = Lru::new(100);
        c.insert((0, 0), 50);
        c.insert((0, 1), 500);
        assert!(!c.resident((0, 1)));
        assert!(
            c.resident((0, 0)),
            "admitting the giant must not evict everything"
        );
    }

    #[test]
    fn pinned_entries_are_never_evicted() {
        let mut c = Lru::new(300);
        assert!(c.pin((0, 0), 100));
        c.insert((0, 1), 100);
        c.insert((0, 2), 100);
        c.insert((0, 3), 100);
        assert!(
            c.resident((0, 0)),
            "a pinned slab must survive eviction pressure"
        );
    }

    #[test]
    fn pinning_respects_the_budget() {
        let mut c = Lru::new(150);
        assert!(c.pin((0, 0), 100));
        assert!(
            !c.pin((0, 1), 100),
            "must refuse rather than overflow the cap"
        );
    }

    #[test]
    fn ageing_lets_a_formerly_hot_slab_fall_out() {
        let mut c = Lru::new(200).with_halflife(4);
        // (0,0) is hot early, then never touched again; (0,1) stays warm.
        for _ in 0..8 {
            c.observe((0, 0));
        }
        c.insert_by_score((0, 0), 100);
        for _ in 0..8 {
            c.observe((0, 1));
        }
        c.insert_by_score((0, 1), 100);
        // Both fit. Now a third slab, hotter than the decayed (0,0), must displace it.
        for _ in 0..40 {
            c.observe((0, 2));
        }
        c.insert_by_score((0, 2), 100);
        assert!(c.resident((0, 2)), "a hotter newcomer must get in");
        assert!(!c.resident((0, 0)), "the slab that went cold must fall out");
    }

    #[test]
    fn a_colder_newcomer_is_refused() {
        let mut c = Lru::new(100);
        c.observe((0, 0));
        c.observe((0, 0));
        c.insert_by_score((0, 0), 100);
        c.observe((0, 1));
        c.insert_by_score((0, 1), 100);
        assert!(c.resident((0, 0)), "the hotter incumbent keeps its place");
        assert!(!c.resident((0, 1)));
    }

    #[test]
    fn runs_merge_only_within_the_gap() {
        assert_eq!(runs(&[0, 1, 2], 0), vec![(0, 2)]);
        assert_eq!(runs(&[0, 2], 0), vec![(0, 0), (2, 2)]);
        assert_eq!(runs(&[0, 2], 1), vec![(0, 2)]);
        assert_eq!(runs(&[0, 5, 6, 20], 1), vec![(0, 0), (5, 6), (20, 20)]);
    }

    #[test]
    fn runs_are_capped_at_the_device_cliff() {
        // 2 MiB slabs, 8 MiB limit -> at most 4 slots per read.
        let parts = split_at_limit(0, 9, 2 << 20, 8 << 20);
        assert_eq!(parts, vec![(0, 3), (4, 7), (8, 9)]);
    }

    #[test]
    fn working_set_counts_distinct_experts() {
        let a: Vec<u16> = vec![0, 1, 2, 3];
        let b: Vec<u16> = vec![2, 3, 4, 5];
        let toks: Vec<&[u16]> = vec![&a, &b];
        let (mean, max) = working_set(&toks, 2);
        assert_eq!(mean, 6.0);
        assert_eq!(max, 6.0);
    }
}
