//! Reader for the binary router trace written by `moetrace`.
//!
//! Layout: a 32-byte header followed by fixed-size records.
//!
//! ```text
//! header: u32 magic "MOET" | u32 version | u32 n_layer | u32 n_expert | u32 top_k | u32 pad | u64 n_records
//! record: u16 layer | u16 pad | u32 token | u16 expert_id * top_k
//! ```

use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const MAGIC: u32 = 0x5445_4F4D; // "MOET"

/// Per-layer accumulator while parsing: expert ids, gate weights, output norms.
type LayerColumns = (Vec<u16>, Vec<f32>, Vec<f32>);

/// All routing decisions for one layer, flattened as `top_k` ids per token.
///
/// `weights` holds the router's softmax probability for each selected expert, in the same
/// order, when the trace was captured at version 2 or later. It is the magnitude with which
/// that expert enters the residual stream — the activation, not just the selection.
#[derive(Debug, Clone)]
pub struct LayerTrace {
    pub layer: u16,
    pub top_k: usize,
    pub experts: Vec<u16>,
    pub weights: Vec<f32>,
    /// L2 norm of each selected expert's pre-gate output (trace version 3).
    pub norms: Vec<f32>,
}

impl LayerTrace {
    pub fn n_tokens(&self) -> usize {
        self.experts.len() / self.top_k
    }

    pub fn token(&self, i: usize) -> &[u16] {
        &self.experts[i * self.top_k..(i + 1) * self.top_k]
    }

    /// Gate weights for token `i`, or an empty slice for a version-1 trace.
    pub fn token_weights(&self, i: usize) -> &[f32] {
        if self.weights.is_empty() {
            &[]
        } else {
            &self.weights[i * self.top_k..(i + 1) * self.top_k]
        }
    }

    pub fn has_weights(&self) -> bool {
        !self.weights.is_empty()
    }

    /// Per-expert output norms for token `i`, or an empty slice if not captured.
    pub fn token_norms(&self, i: usize) -> &[f32] {
        if self.norms.is_empty() {
            &[]
        } else {
            &self.norms[i * self.top_k..(i + 1) * self.top_k]
        }
    }

    pub fn has_norms(&self) -> bool {
        !self.norms.is_empty()
    }

    pub fn tokens(&self) -> impl Iterator<Item = &[u16]> {
        self.experts.chunks_exact(self.top_k)
    }

    /// Evenly-spaced subsample of at most `n` tokens.
    ///
    /// Local search cost is linear in the number of tokens scored, and the objective is an
    /// average over tokens — a few tens of thousands already pin it down. Spacing the
    /// sample rather than truncating keeps every part of the corpus represented.
    pub fn subsample(&self, n: usize) -> LayerTrace {
        let total = self.n_tokens();
        if n == 0 || total <= n {
            return self.clone();
        }
        let mut experts = Vec::with_capacity(n * self.top_k);
        let mut weights = Vec::with_capacity(if self.has_weights() {
            n * self.top_k
        } else {
            0
        });
        let mut norms = Vec::new();
        for i in 0..n {
            let idx = i * total / n;
            experts.extend_from_slice(self.token(idx));
            weights.extend_from_slice(self.token_weights(idx));
            norms.extend_from_slice(self.token_norms(idx));
        }
        LayerTrace {
            layer: self.layer,
            top_k: self.top_k,
            experts,
            weights,
            norms,
        }
    }

    /// Splits into (train, holdout) by taking every `1/holdout_every`-th token for holdout.
    ///
    /// Interleaving rather than slicing matters: a contiguous tail would be a different
    /// slice of the corpus, and any layout gain would be confounded with topic drift.
    pub fn split(&self, holdout_every: usize) -> (LayerTrace, LayerTrace) {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        let (mut wa, mut wb) = (Vec::new(), Vec::new());
        let (mut na, mut nb) = (Vec::new(), Vec::new());
        for i in 0..self.n_tokens() {
            let holdout = holdout_every > 0 && i.is_multiple_of(holdout_every);
            let (e, w, n) = if holdout {
                (&mut b, &mut wb, &mut nb)
            } else {
                (&mut a, &mut wa, &mut na)
            };
            e.extend_from_slice(self.token(i));
            w.extend_from_slice(self.token_weights(i));
            n.extend_from_slice(self.token_norms(i));
        }
        (
            LayerTrace {
                layer: self.layer,
                top_k: self.top_k,
                experts: a,
                weights: wa,
                norms: na,
            },
            LayerTrace {
                layer: self.layer,
                top_k: self.top_k,
                experts: b,
                weights: wb,
                norms: nb,
            },
        )
    }
}

/// One record in the order it was written, paired with its position in the vector stream.
#[derive(Debug, Clone)]
pub struct RawRecord {
    pub layer: u16,
    pub token: u32,
    pub experts: Vec<u16>,
    pub weights: Vec<f32>,
    pub norms: Vec<f32>,
}

/// Reads records in file order rather than grouped by layer.
///
/// `moetrace --vecs` writes the expert output vectors from the same loop as the records, so
/// the two streams line up one to one only in this order.
pub fn load_raw(path: impl AsRef<Path>) -> Result<(usize, Vec<RawRecord>)> {
    let path = path.as_ref();
    let raw = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if raw.len() < 32 {
        bail!("{} is too short to be a trace", path.display());
    }
    let u32_at = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
    if u32_at(0) != MAGIC {
        bail!("{} is not a MOET trace", path.display());
    }
    let version = u32_at(4);
    let top_k = u32_at(16) as usize;
    let n_records = u64::from_le_bytes(raw[24..32].try_into().unwrap()) as usize;
    let has_weights = version >= 2;
    let has_norms = version >= 3;
    let rec_len = 8
        + 2 * top_k
        + if has_weights { 4 * top_k } else { 0 }
        + if has_norms { 4 * top_k } else { 0 };

    let mut out = Vec::with_capacity(n_records);
    for i in 0..n_records {
        let off = 32 + i * rec_len;
        let f32_at = |o: usize| f32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
        out.push(RawRecord {
            layer: u16::from_le_bytes(raw[off..off + 2].try_into().unwrap()),
            token: u32::from_le_bytes(raw[off + 4..off + 8].try_into().unwrap()),
            experts: (0..top_k)
                .map(|j| {
                    u16::from_le_bytes(raw[off + 8 + 2 * j..off + 10 + 2 * j].try_into().unwrap())
                })
                .collect(),
            weights: if has_weights {
                (0..top_k)
                    .map(|j| f32_at(off + 8 + 2 * top_k + 4 * j))
                    .collect()
            } else {
                Vec::new()
            },
            norms: if has_norms {
                (0..top_k)
                    .map(|j| f32_at(off + 8 + 6 * top_k + 4 * j))
                    .collect()
            } else {
                Vec::new()
            },
        });
    }
    Ok((top_k, out))
}

#[derive(Debug, Clone)]
pub struct Trace {
    pub n_layer: u32,
    pub n_expert: u32,
    pub top_k: usize,
    pub layers: Vec<LayerTrace>,
}

impl Trace {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        if raw.len() < 32 {
            bail!("{} is too short to be a trace", path.display());
        }
        let u32_at = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
        if u32_at(0) != MAGIC {
            bail!("{} is not a MOET trace", path.display());
        }
        let version = u32_at(4);
        if !(1..=3).contains(&version) {
            bail!("unsupported trace version {version}");
        }
        let n_layer = u32_at(8);
        let n_expert = u32_at(12);
        let top_k = u32_at(16) as usize;
        let n_records = u64::from_le_bytes(raw[24..32].try_into().unwrap()) as usize;
        if top_k == 0 {
            bail!("trace declares top_k = 0");
        }

        // Version 2 appends one f32 gate weight per selected expert.
        let has_weights = version >= 2;
        let has_norms = version >= 3;
        let rec_len = 8
            + 2 * top_k
            + if has_weights { 4 * top_k } else { 0 }
            + if has_norms { 4 * top_k } else { 0 };
        let available = (raw.len() - 32) / rec_len;
        if available < n_records {
            bail!(
                "trace header claims {n_records} records but only {available} are present \
                 (truncated run?)"
            );
        }

        let mut by_layer: BTreeMap<u16, LayerColumns> = BTreeMap::new();
        for i in 0..n_records {
            let off = 32 + i * rec_len;
            let layer = u16::from_le_bytes(raw[off..off + 2].try_into().unwrap());
            let (ids, wts, nrm) = by_layer.entry(layer).or_default();
            for j in 0..top_k {
                let o = off + 8 + 2 * j;
                let e = u16::from_le_bytes(raw[o..o + 2].try_into().unwrap());
                if e as u32 >= n_expert {
                    bail!("record {i} names expert {e} but the model has {n_expert}");
                }
                ids.push(e);
            }
            if has_weights {
                for j in 0..top_k {
                    let o = off + 8 + 2 * top_k + 4 * j;
                    wts.push(f32::from_le_bytes(raw[o..o + 4].try_into().unwrap()));
                }
            }
            if has_norms {
                for j in 0..top_k {
                    let o = off + 8 + 6 * top_k + 4 * j;
                    nrm.push(f32::from_le_bytes(raw[o..o + 4].try_into().unwrap()));
                }
            }
        }

        let layers = by_layer
            .into_iter()
            .map(|(layer, (experts, weights, norms))| LayerTrace {
                layer,
                top_k,
                experts,
                weights,
                norms,
            })
            .collect();

        let tr = Trace {
            n_layer,
            n_expert,
            top_k,
            layers,
        };
        tr.reject_argsort_artefact()?;
        Ok(tr)
    }

    /// Refuse a trace whose "tokens" are slices of one token's full expert ranking.
    ///
    /// `ffn_moe_topk` is a view of the argsort node with the full row stride in `nb[1]`, so a
    /// flat backend read walks straight through the gaps and returns every expert in rank
    /// order. Chopped into records that looks like plausible routing — plausible ids, correct
    /// descending gate weights — but each `n_expert / top_k` consecutive records are then an
    /// exact partition of every expert, and the access pattern is uniform by construction.
    ///
    /// That artefact drove this project's central claim for its entire first half: it made
    /// routing look flat, which made caching look impossible, which made static pinning look
    /// like a 2.24x win over LRU. On the corrected trace LRU hits 78.6 % and wins. Real
    /// routing has 47x frequency skew.
    ///
    /// A partition is impossible for a real router — independent tokens collide — so this
    /// costs one pass and can only fire on a broken capture.
    pub fn reject_argsort_artefact(&self) -> Result<()> {
        let n_expert = self.n_expert as usize;
        let group = n_expert / self.top_k.max(1);
        if group < 2 {
            return Ok(()); // top_k covers the whole layer; a partition carries no information
        }
        for lt in &self.layers {
            let n = lt.n_tokens();
            let blocks = n / group;
            if blocks < 8 {
                continue;
            }
            let partitions = (0..blocks)
                .filter(|b| {
                    let mut seen = vec![false; n_expert];
                    let mut distinct = 0;
                    for t in b * group..(b + 1) * group {
                        for &e in lt.token(t) {
                            let e = e as usize;
                            if e < seen.len() && !seen[e] {
                                seen[e] = true;
                                distinct += 1;
                            }
                        }
                    }
                    distinct == n_expert
                })
                .count();
            if partitions * 2 > blocks {
                bail!(
                    "layer {}: {partitions} of {blocks} groups of {group} tokens are an exact \
                     partition of all {} experts. Real routing collides; this is the \
                     ffn_moe_topk view being read flat instead of row-wise at nb[1]. \
                     Re-capture with a moetrace built after that fix.",
                    lt.layer,
                    n_expert
                );
            }
        }
        Ok(())
    }

    pub fn n_tokens(&self) -> usize {
        self.layers.first().map_or(0, LayerTrace::n_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::{LayerTrace, Trace};

    fn trace(experts: Vec<u16>) -> Trace {
        Trace {
            n_layer: 1,
            n_expert: 64,
            top_k: 8,
            layers: vec![LayerTrace {
                layer: 0,
                top_k: 8,
                experts,
                weights: vec![],
                norms: vec![],
            }],
        }
    }

    /// The artefact that drove half this project's conclusions, in two lines of setup.
    ///
    /// Eight "tokens" whose ids together are exactly 0..63 — what a flat read of the
    /// `ffn_moe_topk` view produces from one token's full argsort.
    #[test]
    fn an_exact_partition_of_every_expert_is_rejected_as_an_argsort_artefact() {
        let artefact: Vec<u16> = (0..64 * 16).map(|i| (i % 64) as u16).collect();
        let err = trace(artefact)
            .reject_argsort_artefact()
            .unwrap_err()
            .to_string();
        assert!(err.contains("partition"), "unhelpful message: {err}");
    }

    /// Routing that never covers all 64 in a group of 8 must pass, or the guard is useless.
    #[test]
    fn ordinary_routing_is_not_rejected() {
        let real: Vec<u16> = (0..64 * 16).map(|i| ((i * 7) % 40) as u16).collect();
        assert!(trace(real).reject_argsort_artefact().is_ok());
    }
}
