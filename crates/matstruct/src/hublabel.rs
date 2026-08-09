//! Two-sided hub labels: every point keeps its own way *out* and its own way *in*.
//!
//! `landmark.rs` uses one global set of L landmarks and `base(i,j) = min_l d(i,l) + d(l,j)` —
//! one relay, the same candidates for every point. MPEE's later model (MTZU) is two-sided:
//! each point stores its own k hubs for leaving and its own k for arriving, plus a dense
//! hub-to-hub table, and
//!
//! ```text
//!     base(i, j) = min over a in out(i), b in in(j) of  out_d(i,a) + dhh(a,b) + in_d(b,j)
//! ```
//!
//! On real road data that is worth 3.28x -> 7.6x over the one-relay model. It is also the
//! literal form of "an embedding points at one, which points further on" — the chain is
//! i -> a -> b -> j, and neither end has to agree with the other about which hub to use.
//!
//! Two things carry the difference, and both are absent from `landmark.rs`:
//!
//! **Path mining.** Labels are not the k nearest hubs. For each point a sample of partners is
//! drawn, the hub that would have been the best *relay* is credited, and the top-k by credit
//! are kept with distance only as a tie-break. On London 4000² at a fixed 386 KB budget this is
//! the difference between 8.9 % and 35.5 % of cells exact — 4x, on identical memory.
//!
//! **The reach vector.** Evaluating a whole row naively costs n*k² min-plus terms. Factoring
//! the j-independent half out gives `m[b] = min_a out_d(i,a) + dhh(a,b)` once per row at k*H,
//! then k per column: **k*H + n*k** instead of n*k². At H=128, k=16, n=10 000 that is 162 048
//! terms against 2 560 000. Exact, not approximate — min-plus is associative.
//!
//! Reference: `landeveier/mpee/crates/matcodec/src/lib.rs` — `HubModel` :887, `row_reach` :900,
//! `base_from_reach` :914, hub selection :1221-1235, path mining :1051-1110.

use crate::landmark::DistanceSource;
use anyhow::{bail, Result};
use serde::Serialize;

/// Deterministic LCG, the constants matcodec uses, so a shared seed gives a shared sample.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as usize
    }
}

/// Tuning, with MPEE's defaults. `hubs` is capped at 255 because label ids are `u8`.
#[derive(Debug, Clone, Copy)]
pub struct HubOpts {
    pub hubs: usize,
    pub k: usize,
    /// Candidate rows fetched up front. 0 means MTZU's rule, `(n/4).clamp(256, 1024)`.
    pub candidates: usize,
    /// Partners sampled per point when mining labels.
    pub samples_per_row: usize,
    pub seed: u64,
}

impl Default for HubOpts {
    fn default() -> Self {
        Self {
            hubs: 128,
            k: 16,
            candidates: 0,
            samples_per_row: 128,
            seed: 0xA5A5_5A5A_DEAD_BEEF,
        }
    }
}

/// The label index. Resident cost is `2*n*k*5` bytes plus the `H*H` table.
pub struct HubIndex {
    pub h: usize,
    pub k: usize,
    #[cfg_attr(not(test), allow(dead_code))]
    pub n: usize,
    /// `dhh[a*h + b]` — exact hub-to-hub distance.
    pub dhh: Vec<i32>,
    pub out_h: Vec<u8>,
    pub out_d: Vec<i32>,
    pub in_h: Vec<u8>,
    pub in_d: Vec<i32>,
    /// Point indices of the chosen hubs, kept so a caller can inspect or reuse them.
    #[cfg_attr(not(test), allow(dead_code))]
    pub hub_ids: Vec<usize>,
    /// Number of min-plus squarings applied to `dhh`. 0 is MPEE's two-hub model.
    pub squarings: u32,
}

impl HubIndex {
    /// `m[b] = min over a in out(i) of out_d(i,a) + dhh(a,b)`, the j-independent half.
    pub fn row_reach(&self, i: usize) -> Vec<i64> {
        let mut m = vec![i64::MAX; self.h];
        for q in 0..self.k {
            let a = self.out_h[i * self.k + q] as usize;
            let da = self.out_d[i * self.k + q] as i64;
            if da == i64::MAX {
                continue;
            }
            for (b, slot) in m.iter_mut().enumerate() {
                let v = da + self.dhh[a * self.h + b] as i64;
                if v < *slot {
                    *slot = v;
                }
            }
        }
        m
    }

    /// Finish a row's base against one column, given its reach vector. O(k).
    pub fn base_from_reach(&self, m: &[i64], j: usize) -> i64 {
        let mut best = i64::MAX;
        for r in 0..self.k {
            let b = self.in_h[j * self.k + r] as usize;
            let db = self.in_d[j * self.k + r] as i64;
            if db == i64::MAX || m[b] == i64::MAX {
                continue;
            }
            let v = m[b] + db;
            if v < best {
                best = v;
            }
        }
        best
    }

    /// A single cell, O(k²). Cheaper than building a reach vector when only one is wanted,
    /// and the reference the factorised path is tested against.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn base_cell(&self, i: usize, j: usize) -> i64 {
        let mut best = i64::MAX;
        for q in 0..self.k {
            let a = self.out_h[i * self.k + q] as usize;
            let da = self.out_d[i * self.k + q] as i64;
            if da == i64::MAX {
                continue;
            }
            for r in 0..self.k {
                let b = self.in_h[j * self.k + r] as usize;
                let db = self.in_d[j * self.k + r] as i64;
                if db == i64::MAX {
                    continue;
                }
                let v = da + self.dhh[a * self.h + b] as i64 + db;
                if v < best {
                    best = v;
                }
            }
        }
        best
    }

    pub fn resident_bytes(&self) -> usize {
        self.dhh.len() * 4
            + (self.out_h.len() + self.in_h.len())
            + (self.out_d.len() + self.in_d.len()) * 4
    }

    /// One min-plus squaring: `dhh'[a][c] = min_b dhh[a][b] + dhh[b][c]`.
    ///
    /// Each squaring doubles the number of hub-to-hub legs a path may use, so `base` goes from
    /// two hubs to three, then five, and so on. MPEE never does this and has a good argument
    /// for why: on a shortest-path metric `dhh` is already exact, so relaying through a second
    /// hub cannot beat the direct entry. That argument does not transfer to a chord metric,
    /// where `dhh` is not a shortest path through anything — which is the whole reason to
    /// measure it rather than assume either way.
    pub fn square_dhh(&mut self) {
        let h = self.h;
        let mut next = vec![i32::MAX; h * h];
        for a in 0..h {
            for c in 0..h {
                let mut best = self.dhh[a * h + c] as i64;
                for b in 0..h {
                    let (x, y) = (self.dhh[a * h + b] as i64, self.dhh[b * h + c] as i64);
                    if x + y < best {
                        best = x + y;
                    }
                }
                next[a * h + c] = best.min(i32::MAX as i64) as i32;
            }
        }
        self.dhh = next;
        self.squarings += 1;
    }
}

/// Build with hubs supplied by the caller, skipping facility location.
///
/// Exists because hub *selection* has turned out to matter far more than the label structure:
/// betweenness-chosen landmarks on the geodesic metric reach 39.08 % of cells exact where
/// facility location on the chord reaches 0.91 %. MPEE reaches the same conclusion from the
/// other side — its `ExternalHubs` path takes candidates from the top of a contraction
/// hierarchy and measures 9.4x against 7.6x for point-mined hubs.
pub fn build_with_hubs<S: DistanceSource>(
    src: &S,
    opts: HubOpts,
    hubs: Vec<usize>,
) -> Result<HubIndex> {
    build_inner(src, opts, Some(hubs))
}

/// Build the index: pick hubs, then mine each point's two label sets.
pub fn build<S: DistanceSource>(src: &S, opts: HubOpts) -> Result<HubIndex> {
    build_inner(src, opts, None)
}

fn build_inner<S: DistanceSource>(
    src: &S,
    opts: HubOpts,
    given: Option<Vec<usize>>,
) -> Result<HubIndex> {
    let n = src.n();
    if n < 3 {
        bail!("need at least 3 points");
    }
    let c_count = if opts.candidates == 0 {
        (n / 4).clamp(256, 1024).min(n)
    } else {
        opts.candidates.min(n)
    };
    let mut rng = Lcg(opts.seed);

    // Candidate rows, fetched once. C x n, never n x n.
    let mut cand = Vec::with_capacity(c_count);
    let mut seen = vec![false; n];
    while cand.len() < c_count {
        let v = rng.next() % n;
        if !seen[v] {
            seen[v] = true;
            cand.push(v);
        }
    }
    // Supplied hubs replace the candidate pool outright: their rows are all that is needed,
    // and fetching C extra rows to run a selection we are not going to use would be waste.
    let (cand, c_count) = match &given {
        Some(h) if !h.is_empty() => {
            let h: Vec<usize> = h.iter().copied().filter(|&v| v < n).collect();
            let c = h.len();
            (h, c)
        }
        _ => (cand, c_count),
    };
    let mut cand_rows = vec![0i32; c_count * n];
    let mut row = vec![0i32; n];
    for (a, &c) in cand.iter().enumerate() {
        src.row(c, &mut row);
        cand_rows[a * n..(a + 1) * n].copy_from_slice(&row);
    }

    // Greedy facility location over sampled pairs, scoring d(i,a) + d(a,j) out of the same
    // table — which is why candidates must be rows we already fetched.
    let h_count = opts.hubs.min(c_count).clamp(1, 255);
    if given.is_some() {
        // Take the supplied order as given — it already encodes the caller's ranking.
        let hub_ids: Vec<usize> = (0..h_count).collect();
        return finish(src, opts, cand, cand_rows, hub_ids, n);
    }
    let s_count = (c_count * 24).min(16384);
    let pairs: Vec<(usize, usize)> = (0..s_count)
        .map(|_| (rng.next() % c_count, rng.next() % n))
        .collect();
    let mut cur = vec![i64::MAX; s_count];
    let mut used = vec![false; c_count];
    let mut hub_ids = Vec::with_capacity(h_count);
    for _ in 0..h_count {
        let (mut best_a, mut best_cost) = (usize::MAX, i64::MAX);
        for a in 0..c_count {
            if used[a] {
                continue;
            }
            let mut cost = 0i64;
            for (s, &(ci, j)) in pairs.iter().enumerate() {
                let via = cand_rows[ci * n + cand[a]] as i64 + cand_rows[a * n + j] as i64;
                cost += cur[s].min(via);
            }
            if cost < best_cost {
                best_cost = cost;
                best_a = a;
            }
        }
        if best_a == usize::MAX {
            break;
        }
        used[best_a] = true;
        hub_ids.push(best_a);
        for (s, &(ci, j)) in pairs.iter().enumerate() {
            let via = cand_rows[ci * n + cand[best_a]] as i64 + cand_rows[best_a * n + j] as i64;
            cur[s] = cur[s].min(via);
        }
    }
    finish(src, opts, cand, cand_rows, hub_ids, n)
}

/// Everything after hub choice: the hub table, per-point hub distances, and path mining.
fn finish<S: DistanceSource>(
    src: &S,
    opts: HubOpts,
    cand: Vec<usize>,
    cand_rows: Vec<i32>,
    hub_ids: Vec<usize>,
    n: usize,
) -> Result<HubIndex> {
    let mut rng = Lcg(opts.seed ^ 0x9E37_79B9);
    let h = hub_ids.len();
    let k = opts.k.min(h).max(1);
    let _ = src;

    // dhh, lifted straight out of the candidate rows — exact, no extra queries.
    let mut dhh = vec![0i32; h * h];
    for (a, &ha) in hub_ids.iter().enumerate() {
        for (b, &hb) in hub_ids.iter().enumerate() {
            dhh[a * h + b] = cand_rows[ha * n + cand[hb]];
        }
    }

    // Each point's distance to and from every hub, from the same table.
    let mut d_hub = vec![0i32; n * h];
    for (a, &ha) in hub_ids.iter().enumerate() {
        for j in 0..n {
            d_hub[j * h + a] = cand_rows[ha * n + j];
        }
    }
    drop(cand_rows);

    // Path mining, both directions. Credit the hub that would have been the best relay, then
    // take the top-k by credit with distance as the tie-break — that is what fills unused
    // slots with near hubs when nothing has been credited.
    let spp = opts.samples_per_row.max(16);
    let mut out_h = vec![0u8; n * k];
    let mut out_d = vec![0i32; n * k];
    let mut in_h = vec![0u8; n * k];
    let mut in_d = vec![0i32; n * k];
    let mut credit = vec![0u32; h];
    let mut order: Vec<usize> = (0..h).collect();

    // The out- and in-sets are mined once, not twice, and the reason is worth stating.
    //
    // MPEE keeps them separate because road networks are directed: a point's best motorway
    // on-ramp is not its best off-ramp when streets are one-way, so `d(i,a)` and `d(a,i)`
    // differ and the two credit histograms come out different. Every metric in this repository
    // is symmetric — angular distance and shortest paths over an undirected kNN graph both
    // satisfy `d(i,j) == d(j,i)` — so the two directions score identically and mining twice
    // produces two copies of one answer.
    //
    // clippy caught this as "identical blocks" and it is more than a style note: on a symmetric
    // metric the two-sided model *degenerates to one-sided* while still costing twice the
    // resident memory. That is a large part of why it loses to `landmark.rs` here and wins on
    // MPEE's directed road matrices. The tables are still written separately so the format
    // matches MTZU and a directed source could fill them independently later.
    for i in 0..n {
        credit.iter_mut().for_each(|c| *c = 0);
        for _ in 0..spp {
            let p = rng.next() % n;
            if p == i {
                continue;
            }
            let (mut best_a, mut best_v) = (0usize, i64::MAX);
            for a in 0..h {
                let v = d_hub[i * h + a] as i64 + d_hub[p * h + a] as i64;
                if v < best_v {
                    best_v = v;
                    best_a = a;
                }
            }
            credit[best_a] += 1;
        }
        order.sort_unstable_by_key(|&a| (std::cmp::Reverse(credit[a]), d_hub[i * h + a]));
        for (slot, &a) in order.iter().take(k).enumerate() {
            out_h[i * k + slot] = a as u8;
            out_d[i * k + slot] = d_hub[i * h + a];
            in_h[i * k + slot] = a as u8;
            in_d[i * k + slot] = d_hub[i * h + a];
        }
    }

    Ok(HubIndex {
        h,
        k,
        n,
        dhh,
        out_h,
        out_d,
        in_h,
        in_d,
        hub_ids: hub_ids.iter().map(|&a| cand[a]).collect(),
        squarings: 0,
    })
}

/// What the label index answers, scored for search rather than for compression.
#[derive(Debug, Serialize)]
pub struct HubReport {
    pub n: usize,
    pub hubs: usize,
    pub k: usize,
    pub squarings: u32,
    pub resident_bytes: usize,
    pub rows_sampled: usize,
    /// Share of cells the base reproduces exactly.
    pub exact_cell_pct: f64,
    /// Mean `base - d` as a fraction of mean distance. 0 is a perfect skeleton.
    pub residual_ratio: f64,
    /// Fraction of the true k nearest that a rank by `base` recovers — the search metric.
    pub recall_at_k: f64,
    /// Cells where base < d. Impossible for a metric; non-zero invalidates the bound.
    pub violations: u64,
    pub max_violation: i32,
}

/// Score the index against the source, on a strided sample of rows.
pub fn measure<S: DistanceSource>(
    src: &S,
    idx: &HubIndex,
    rows: usize,
    recall_k: usize,
) -> HubReport {
    let n = src.n();
    let rows = rows.clamp(1, n);
    let step = (n / rows).max(1);
    let (mut exact, mut total) = (0u64, 0u64);
    let (mut sum_resid, mut sum_dist) = (0f64, 0f64);
    let (mut violations, mut worst) = (0u64, 0i32);
    let (mut recall_hits, mut recall_total) = (0u64, 0u64);
    let mut row = vec![0i32; n];
    let mut sampled = 0usize;

    for i in (0..n).step_by(step) {
        sampled += 1;
        src.row(i, &mut row);
        let m = idx.row_reach(i);

        // True nearest, and the nearest the base would pick. Recall is the search question:
        // does routing through the labels find what a full scan would have found?
        let mut truth: Vec<(i32, usize)> =
            (0..n).filter(|&j| j != i).map(|j| (row[j], j)).collect();
        let kk = recall_k.min(truth.len());
        truth.select_nth_unstable(kk.saturating_sub(1));
        truth.truncate(kk);
        let truth_set: std::collections::HashSet<usize> = truth.iter().map(|&(_, j)| j).collect();

        let mut est: Vec<(i64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (idx.base_from_reach(&m, j), j))
            .collect();
        if kk > 0 {
            est.select_nth_unstable(kk - 1);
            est.truncate(kk);
            recall_hits += est.iter().filter(|&&(_, j)| truth_set.contains(&j)).count() as u64;
            recall_total += kk as u64;
        }

        for (j, &dij) in row.iter().enumerate() {
            if j == i {
                continue;
            }
            let base = idx.base_from_reach(&m, j);
            if base == i64::MAX {
                continue;
            }
            let r = base - dij as i64;
            if r < 0 {
                violations += 1;
                worst = worst.max((-r).min(i32::MAX as i64) as i32);
            }
            if r == 0 {
                exact += 1;
            }
            total += 1;
            sum_resid += r.unsigned_abs() as f64;
            sum_dist += dij as f64;
        }
    }

    let cells = total.max(1) as f64;
    let mean_dist = sum_dist / cells;
    HubReport {
        n,
        hubs: idx.h,
        k: idx.k,
        squarings: idx.squarings,
        resident_bytes: idx.resident_bytes(),
        rows_sampled: sampled,
        exact_cell_pct: 100.0 * exact as f64 / cells,
        residual_ratio: if mean_dist > 0.0 {
            (sum_resid / cells) / mean_dist
        } else {
            0.0
        },
        recall_at_k: if recall_total > 0 {
            recall_hits as f64 / recall_total as f64
        } else {
            0.0
        },
        violations,
        max_violation: worst,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::landmark::{measure as lm_measure, pick_landmarks, DenseSource, LandmarkIndex};

    /// MPEE's synthetic gateway world, the same generator `landmark.rs` validates against.
    fn gateway_world(n: usize, regions: usize) -> DenseSource {
        let per = n / regions;
        let n = per * regions;
        let mut s: u64 = 0xC0FFEE;
        let mut rnd = |range: i64| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as i64) % range
        };
        let mut pts = vec![(0i64, 0i64); n];
        for r in 0..regions {
            let c = ((r % 4) as i64 * 100_000, (r / 4) as i64 * 100_000);
            for i in 0..per {
                pts[r * per + i] = (c.0 + rnd(4000), c.1 + rnd(4000));
            }
        }
        let l1 = |a: (i64, i64), b: (i64, i64)| (a.0 - b.0).abs() + (a.1 - b.1).abs();
        let gw = |r: usize, k: usize| r * per + k;
        let road = 30_000i64;
        let mut d = vec![0i32; n * n];
        for i in 0..n {
            for j in 0..n {
                let (ri, rj) = (i / per, j / per);
                let v = if ri == rj {
                    l1(pts[i], pts[j])
                } else {
                    (0..3)
                        .map(|k| l1(pts[i], pts[gw(ri, k)]) + road + l1(pts[gw(rj, k)], pts[j]))
                        .min()
                        .unwrap()
                };
                d[i * n + j] = v as i32;
            }
        }
        DenseSource { n, d }
    }

    /// The base can never undercut a true metric distance, whatever the labels.
    #[test]
    fn the_base_never_undercuts_the_truth() {
        let src = gateway_world(600, 4);
        let idx = build(
            &src,
            HubOpts {
                hubs: 32,
                k: 8,
                ..Default::default()
            },
        )
        .unwrap();
        let rep = measure(&src, &idx, 120, 10);
        assert_eq!(
            rep.violations, 0,
            "min-plus through a metric cannot undercut: {rep:?}"
        );
    }

    /// The reach factorisation must be exact, not an approximation. If these two disagree the
    /// k*H + n*k shortcut is wrong and every row-wise number built on it is wrong with it.
    #[test]
    fn reach_factorisation_equals_the_naive_cell() {
        let src = gateway_world(400, 4);
        let idx = build(
            &src,
            HubOpts {
                hubs: 24,
                k: 6,
                ..Default::default()
            },
        )
        .unwrap();
        for i in (0..src.n()).step_by(23) {
            let m = idx.row_reach(i);
            for j in (0..src.n()).step_by(29) {
                assert_eq!(
                    idx.base_from_reach(&m, j),
                    idx.base_cell(i, j),
                    "reach and cell disagree at ({i},{j})"
                );
            }
        }
    }

    /// Path mining must beat taking the k nearest hubs at identical memory. This is the whole
    /// claim of the two-sided model — MPEE measures 8.9 % against 35.5 % on London 4000² — and
    /// without it the extra machinery buys nothing over `landmark.rs`.
    #[test]
    fn path_mining_beats_nearest_hubs_at_equal_memory() {
        let src = gateway_world(800, 8);
        let opts = HubOpts {
            hubs: 32,
            k: 8,
            samples_per_row: 64,
            ..Default::default()
        };
        let mined = measure(&src, &build(&src, opts).unwrap(), 160, 10);

        // The same index with labels replaced by the k nearest hubs, nothing else changed.
        let mut naive = build(&src, opts).unwrap();
        for i in 0..naive.n {
            let mut by_dist: Vec<(i32, usize)> = (0..naive.h)
                .map(|a| {
                    let mut r = vec![0i32; src.n()];
                    src.row(naive.hub_ids[a], &mut r);
                    (r[i], a)
                })
                .collect();
            by_dist.sort_unstable();
            for (slot, &(d, a)) in by_dist.iter().take(naive.k).enumerate() {
                naive.out_h[i * naive.k + slot] = a as u8;
                naive.out_d[i * naive.k + slot] = d;
                naive.in_h[i * naive.k + slot] = a as u8;
                naive.in_d[i * naive.k + slot] = d;
            }
        }
        let naive_rep = measure(&src, &naive, 160, 10);
        assert!(
            mined.exact_cell_pct >= naive_rep.exact_cell_pct,
            "path mining {:.2} % must not lose to nearest-hub {:.2} %",
            mined.exact_cell_pct,
            naive_rep.exact_cell_pct
        );
    }

    /// MPEE's counterintuitive signature: on the *synthetic* gateway world the two-sided model
    /// is WORSE than the one-relay model (11.2x against 17.6x), because its hub miner only sees
    /// a candidate sample while full pivot mining sees the whole matrix. A port that "improves"
    /// on that has diverged from the original.
    #[test]
    fn two_sided_loses_to_one_relay_on_the_synthetic_gateway_world() {
        let src = gateway_world(1200, 8);
        let one = lm_measure(
            &src,
            &LandmarkIndex::build(&src, pick_landmarks(&src, 32, usize::MAX, 1)).unwrap(),
            200,
            2,
        );
        let two = measure(
            &src,
            &build(
                &src,
                HubOpts {
                    hubs: 32,
                    k: 8,
                    ..Default::default()
                },
            )
            .unwrap(),
            200,
            10,
        );
        assert!(
            one.exact_cell_pct > two.exact_cell_pct,
            "one relay ({:.2} %) should beat two-sided ({:.2} %) here, as it does in MPEE",
            one.exact_cell_pct,
            two.exact_cell_pct
        );
    }

    /// Squaring may only tighten the base, never loosen it, and never break the bound.
    #[test]
    fn squaring_only_tightens() {
        let src = gateway_world(400, 4);
        let mut idx = build(
            &src,
            HubOpts {
                hubs: 24,
                k: 6,
                ..Default::default()
            },
        )
        .unwrap();
        let before: Vec<i64> = (0..src.n())
            .step_by(31)
            .flat_map(|i| (0..src.n()).step_by(37).map(move |j| (i, j)))
            .map(|(i, j)| idx.base_cell(i, j))
            .collect();
        idx.square_dhh();
        let after: Vec<i64> = (0..src.n())
            .step_by(31)
            .flat_map(|i| (0..src.n()).step_by(37).map(move |j| (i, j)))
            .map(|(i, j)| idx.base_cell(i, j))
            .collect();
        for (b, a) in before.iter().zip(&after) {
            assert!(a <= b, "squaring loosened a bound: {b} -> {a}");
        }
        assert_eq!(measure(&src, &idx, 100, 10).violations, 0);
    }
}
