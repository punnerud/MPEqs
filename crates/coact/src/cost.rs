//! The fetch cost model, and layouts evaluated against it.
//!
//! This is MPEdb §8.5 transplanted: a fetch has a fixed cost regardless of size, so it can
//! be worth reading experts you do not need in order to turn two fetches into one. The
//! break-even gap is `n_tensors * C_fetch / (bytes_per_expert * C_byte)`, both constants
//! measured on the actual device by `fetchbench calibrate`.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// Where a cost constant came from, borrowed from MPEdb's MPEE solver (DESIGN-MPEE-SOLVER §2).
///
/// That solver refuses to invent a selectivity factor, because a plan that is optimal on a
/// made-up estimate can be catastrophic on the data. The same rule earns its keep here: an
/// early calibration of this project read pages that were still resident and reported
/// `C_fetch = 6.66 us` against the true 230.74 us. Every downstream number was scaled by that
/// 35x error, and — worse — the guard meant to catch cache contamination compares achieved
/// bandwidth against this very constant, so an invented value silently disabled its own check.
///
/// The rule: a constant is either measured on the device it is used to reason about, or it is
/// an explicit upper bound carried from elsewhere, or it does not exist. There is no fourth
/// case where a plausible-looking default gets quietly substituted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provenance {
    /// Measured by `fetchbench calibrate` on this device, cold.
    Measured,
    /// Carried from another device or model. Usable, but every result derived from it must say
    /// so, and it may be wrong in either direction. This is the default so that a cost model
    /// deserialised from an old file without the field is never mistaken for a measurement.
    #[default]
    Assumed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CostModel {
    /// Fixed cost of issuing one read, in nanoseconds.
    pub c_fetch_ns: f64,
    /// Marginal cost per byte read, in nanoseconds.
    pub c_byte_ns: f64,
    /// Whether these numbers were measured here or inherited. Never silently defaulted.
    #[serde(default)]
    pub provenance: Provenance,
}

impl CostModel {
    /// The last cold calibration on an Apple M3 Pro NVMe, offered only as an explicit
    /// fallback. It is never substituted automatically: callers must ask for it and say so.
    pub fn assumed_apple_nvme() -> Self {
        CostModel {
            c_fetch_ns: 230_740.0,
            c_byte_ns: 0.2856,
            provenance: Provenance::Assumed,
        }
    }

    /// True when every result derived from this model has to be labelled as unfounded.
    pub fn is_assumed(&self) -> bool {
        self.provenance == Provenance::Assumed
    }
}

impl CostModel {
    /// Largest run of unwanted experts worth reading through to avoid an extra fetch.
    pub fn max_gap(&self, bytes_per_expert: u64, n_tensors: usize) -> u64 {
        if self.c_byte_ns <= 0.0 || bytes_per_expert == 0 {
            return 0;
        }
        let g = (n_tensors as f64 * self.c_fetch_ns) / (bytes_per_expert as f64 * self.c_byte_ns);
        if g <= 0.0 {
            0
        } else {
            g.floor() as u64
        }
    }
}

/// Geometry of one MoE layer, as far as the cost model is concerned.
#[derive(Debug, Clone, Copy)]
pub struct LayerGeometry {
    /// Number of fused weight tensors an expert is split across (3 for gate/up/down).
    pub n_tensors: usize,
    /// Bytes for one expert, summed over those tensors.
    pub bytes_per_expert: u64,
}

/// `perm[slot] = original expert id`. Slot order is physical order in the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Permutation {
    perm: Vec<u16>,
    pos: Vec<u16>,
}

impl Permutation {
    pub fn identity(n: usize) -> Self {
        Self::from_perm((0..n as u16).collect()).expect("identity is a valid permutation")
    }

    pub fn from_perm(perm: Vec<u16>) -> Result<Self> {
        let n = perm.len();
        let mut pos = vec![u16::MAX; n];
        for (slot, &e) in perm.iter().enumerate() {
            let e = e as usize;
            if e >= n {
                bail!("permutation names expert {e} but has length {n}");
            }
            if pos[e] != u16::MAX {
                bail!("permutation is not a bijection: expert {e} appears twice");
            }
            pos[e] = slot as u16;
        }
        Ok(Permutation { perm, pos })
    }

    pub fn len(&self) -> usize {
        self.perm.len()
    }

    pub fn is_empty(&self) -> bool {
        self.perm.is_empty()
    }

    /// Original expert id stored in physical slot `slot`.
    pub fn expert_at(&self, slot: usize) -> u16 {
        self.perm[slot]
    }

    /// Physical slot holding original expert `e`.
    pub fn slot_of(&self, e: u16) -> usize {
        self.pos[e as usize] as usize
    }

    pub fn as_slice(&self) -> &[u16] {
        &self.perm
    }

    pub fn swap(&mut self, a: usize, b: usize) {
        self.perm.swap(a, b);
        self.pos[self.perm[a] as usize] = a as u16;
        self.pos[self.perm[b] as usize] = b as u16;
    }

    pub fn inverse(&self) -> Permutation {
        Permutation {
            perm: self.pos.clone(),
            pos: self.perm.clone(),
        }
    }
}

/// What one token's routing costs under a given layout.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FetchPlan {
    pub n_fetches: u64,
    pub bytes: u64,
}

/// Groups the selected slots into fetch ranges, bridging gaps up to `max_gap`.
///
/// `scratch` is reused across calls; the caller owns it purely to keep this allocation-free
/// in the inner search loop.
pub fn plan(
    selected: &[u16],
    p: &Permutation,
    geom: LayerGeometry,
    max_gap: u64,
    scratch: &mut Vec<u32>,
) -> FetchPlan {
    scratch.clear();
    for &e in selected {
        scratch.push(p.slot_of(e) as u32);
    }
    scratch.sort_unstable();
    scratch.dedup();
    if scratch.is_empty() {
        return FetchPlan::default();
    }

    let mut groups = 0u64;
    let mut span_slots = 0u64;
    let mut start = scratch[0];
    let mut prev = scratch[0];
    for &s in &scratch[1..] {
        if (s - prev) as u64 - 1 > max_gap {
            groups += 1;
            span_slots += (prev - start + 1) as u64;
            start = s;
        }
        prev = s;
    }
    groups += 1;
    span_slots += (prev - start + 1) as u64;

    FetchPlan {
        n_fetches: groups * geom.n_tensors as u64,
        bytes: span_slots * geom.bytes_per_expert,
    }
}

/// Aggregate statistics for one layer under one layout.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct LayerStats {
    pub n_tokens: u64,
    pub fetches: u64,
    pub bytes: u64,
    pub cost_ns: f64,
}

impl LayerStats {
    pub fn fetches_per_token(&self) -> f64 {
        self.fetches as f64 / self.n_tokens.max(1) as f64
    }
    pub fn bytes_per_token(&self) -> f64 {
        self.bytes as f64 / self.n_tokens.max(1) as f64
    }
    pub fn cost_per_token_ns(&self) -> f64 {
        self.cost_ns / self.n_tokens.max(1) as f64
    }
    pub fn add(&mut self, other: &LayerStats) {
        self.n_tokens = self.n_tokens.max(other.n_tokens);
        self.fetches += other.fetches;
        self.bytes += other.bytes;
        self.cost_ns += other.cost_ns;
    }
}

pub fn evaluate(
    tokens: impl Iterator<Item = impl AsRef<[u16]>>,
    p: &Permutation,
    geom: LayerGeometry,
    cm: &CostModel,
) -> LayerStats {
    let max_gap = cm.max_gap(geom.bytes_per_expert, geom.n_tensors);
    let mut scratch = Vec::with_capacity(64);
    let mut st = LayerStats::default();
    for tok in tokens {
        let fp = plan(tok.as_ref(), p, geom, max_gap, &mut scratch);
        st.n_tokens += 1;
        st.fetches += fp.n_fetches;
        st.bytes += fp.bytes;
    }
    st.cost_ns = st.fetches as f64 * cm.c_fetch_ns + st.bytes as f64 * cm.c_byte_ns;
    st
}

#[cfg(test)]
mod tests {
    use super::*;

    const GEOM: LayerGeometry = LayerGeometry {
        n_tensors: 3,
        bytes_per_expert: 4_079_616,
    };

    fn scratch() -> Vec<u32> {
        Vec::new()
    }

    #[test]
    fn contiguous_selection_is_one_group() {
        let p = Permutation::identity(64);
        let fp = plan(&[3, 0, 1, 2], &p, GEOM, 0, &mut scratch());
        assert_eq!(fp.n_fetches, 3);
        assert_eq!(fp.bytes, 4 * GEOM.bytes_per_expert);
    }

    #[test]
    fn scattered_selection_is_one_group_per_expert() {
        let p = Permutation::identity(64);
        let fp = plan(&[0, 10, 20, 30], &p, GEOM, 0, &mut scratch());
        assert_eq!(fp.n_fetches, 4 * 3);
        assert_eq!(fp.bytes, 4 * GEOM.bytes_per_expert);
    }

    #[test]
    fn gap_bridging_trades_bytes_for_fetches() {
        let p = Permutation::identity(64);
        // Slots 0 and 2 with one unwanted expert between them.
        let tight = plan(&[0, 2], &p, GEOM, 0, &mut scratch());
        assert_eq!(tight.n_fetches, 6);
        assert_eq!(tight.bytes, 2 * GEOM.bytes_per_expert);

        let bridged = plan(&[0, 2], &p, GEOM, 1, &mut scratch());
        assert_eq!(bridged.n_fetches, 3);
        assert_eq!(bridged.bytes, 3 * GEOM.bytes_per_expert);
    }

    #[test]
    fn permutation_makes_a_scattered_set_contiguous() {
        // Place experts 0,10,20,30 into slots 0..3.
        let mut perm: Vec<u16> = vec![0, 10, 20, 30];
        perm.extend((0..64u16).filter(|e| ![0, 10, 20, 30].contains(e)));
        let p = Permutation::from_perm(perm).unwrap();
        let fp = plan(&[0, 10, 20, 30], &p, GEOM, 0, &mut scratch());
        assert_eq!(fp.n_fetches, 3);
    }

    #[test]
    fn inverse_round_trips() {
        let mut p = Permutation::identity(8);
        p.swap(1, 5);
        p.swap(0, 3);
        let inv = p.inverse();
        for e in 0..8u16 {
            assert_eq!(inv.expert_at(p.slot_of(e)), e);
        }
    }

    #[test]
    fn duplicate_expert_is_rejected() {
        assert!(Permutation::from_perm(vec![0, 1, 1, 3]).is_err());
    }

    #[test]
    fn max_gap_matches_break_even() {
        let cm = CostModel {
            c_fetch_ns: 60_000.0,
            c_byte_ns: 0.35,
            provenance: Provenance::Assumed,
        };
        // 3 * 60000 / (4079616 * 0.35) = 0.126 -> never worth bridging at this size.
        assert_eq!(cm.max_gap(4_079_616, 3), 0);
        // Small experts flip the trade the other way.
        assert!(cm.max_gap(8_192, 3) > 0);
    }
}
