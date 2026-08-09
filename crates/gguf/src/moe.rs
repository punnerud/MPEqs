//! MoE-specific view over a GGUF file.
//!
//! In llama.cpp the per-layer experts are stored as *fused* tensors with the expert axis
//! last, e.g. `blk.0.ffn_gate_exps.weight = [n_embd, n_ff_exp, n_expert]`. Because ggml
//! lays dimensions out with dim 0 fastest, one expert's slab is a single contiguous byte
//! range. Reordering experts is therefore a permutation of equal-sized blocks — and if the
//! router weight `ffn_gate_inp` is permuted along the same axis, the model's output is
//! bit-for-bit unchanged.

use crate::{Gguf, TensorInfo};
use anyhow::{bail, Result};

/// Tensors whose last axis indexes experts and must be permuted together.
///
/// `shexp` (shared-expert) tensors are excluded on purpose: they are not routed, so they
/// carry no expert axis to permute.
const EXPERT_AXIS_PATTERNS: &[&str] = &["_exps", "ffn_gate_inp", "exp_probs_b"];

#[derive(Debug, Clone)]
pub struct MoeLayer {
    pub index: u32,
    /// Every tensor in this layer whose last axis is the expert axis.
    pub tensors: Vec<TensorInfo>,
}

impl MoeLayer {
    /// The three big weight tensors, i.e. the ones that dominate the byte cost.
    pub fn weight_tensors(&self) -> impl Iterator<Item = &TensorInfo> {
        self.tensors.iter().filter(|t| t.name.contains("_exps"))
    }

    /// Bytes that must be fetched to evaluate one expert in this layer.
    pub fn bytes_per_expert(&self) -> Result<u64> {
        let mut total = 0;
        for t in self.weight_tensors() {
            total += t.last_axis_stride()?;
        }
        Ok(total)
    }

    /// Absolute byte ranges for one expert, one per fused weight tensor.
    pub fn expert_ranges(&self, expert: u32) -> Result<Vec<(u64, u64)>> {
        self.weight_tensors()
            .map(|t| t.slice_range(expert as u64))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct MoeModel {
    pub arch: String,
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub layers: Vec<MoeLayer>,
}

impl MoeModel {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.architecture()?.to_string();
        let n_expert = g.arch_u64("expert_count")? as u32;
        let n_expert_used = g.arch_u64("expert_used_count")? as u32;
        if n_expert == 0 {
            bail!("{arch} reports expert_count = 0; not a MoE model");
        }
        let n_layer = g.arch_u64("block_count")? as u32;

        let mut layers = Vec::new();
        for index in 0..n_layer {
            let prefix = format!("blk.{index}.");
            let tensors: Vec<TensorInfo> = g
                .tensors
                .iter()
                .filter(|t| t.name.starts_with(&prefix))
                .filter(|t| !t.name.contains("shexp"))
                .filter(|t| EXPERT_AXIS_PATTERNS.iter().any(|p| t.name.contains(p)))
                .filter(|t| t.last_dim() == n_expert as u64)
                .cloned()
                .collect();
            if tensors.is_empty() {
                // Hybrid models (Granite 4, Qwen3-Next) interleave dense layers; skipping
                // them is correct, not an error.
                continue;
            }
            layers.push(MoeLayer { index, tensors });
        }

        if layers.is_empty() {
            bail!("found no expert-axis tensors in {arch}");
        }
        Ok(MoeModel {
            arch,
            n_expert,
            n_expert_used,
            layers,
        })
    }

    pub fn layer(&self, index: u32) -> Result<&MoeLayer> {
        self.layers
            .iter()
            .find(|l| l.index == index)
            .ok_or_else(|| anyhow::anyhow!("layer {index} is not a MoE layer"))
    }

    /// Total bytes of routed-expert weights across the model.
    pub fn expert_bytes(&self) -> Result<u64> {
        let mut total = 0;
        for l in &self.layers {
            for t in l.weight_tensors() {
                total += t.nbytes;
            }
        }
        Ok(total)
    }

    /// Bytes touched by a single decode step: every MoE layer fetches `n_expert_used`
    /// experts, each spread across the fused weight tensors.
    pub fn bytes_per_token(&self) -> Result<u64> {
        let mut total = 0;
        for l in &self.layers {
            total += l.bytes_per_expert()? * self.n_expert_used as u64;
        }
        Ok(total)
    }

    /// Number of disjoint byte ranges a naive fetcher issues per decode step.
    pub fn ranges_per_token(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weight_tensors().count() * self.n_expert_used as usize)
            .sum()
    }
}
