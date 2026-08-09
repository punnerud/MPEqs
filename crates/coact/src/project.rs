//! What the layout is worth on a model far too large to hold in memory.
//!
//! The target case is a several-hundred-gigabyte MoE on a laptop: the resident fraction is a
//! few percent, so essentially every expert access is a disk fetch. That is exactly the
//! regime `fetchbench` measures — it reads uncached, at a 100 % miss rate — which means the
//! numbers transfer without needing the model present.
//!
//! What does *not* transfer automatically is the fetch-count reduction: how much a layout can
//! cluster a token's experts depends on that model's co-activation structure, which cannot be
//! known without tracing it. So the projection takes the reduction as an explicit input and
//! reports the arithmetic, rather than pretending one model's 14.4 % is universal.

use crate::cost::CostModel;
use anyhow::Result;
use gguf::{moe::MoeModel, Gguf};
use serde::Serialize;

#[derive(Serialize)]
pub struct Projection {
    pub model: String,
    pub arch: String,
    pub file_gb: f64,
    pub n_moe_layers: usize,
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub bytes_per_expert: u64,
    /// Disjoint byte ranges a naive fetcher issues per decode token.
    pub ranges_per_token: usize,
    pub mib_per_token: f64,
    /// Requests per mebibyte of per-token traffic. The single best predictor of whether
    /// layout can pay: it decides whether a token's cost is dominated by request latency or
    /// by transfer. Measured speedups track it monotonically — 0.83 gives 1.078x, 3.86 gives
    /// 1.140x — while total model size does not predict it at all.
    pub ranges_per_mib: f64,
    /// Experts worth reading through to merge two fetches, at the measured device cost.
    pub max_bridged_gap: u64,
    /// Share of the per-token cost that is per-request overhead rather than transfer.
    pub fetch_overhead_share_pct: f64,
    pub ms_per_token_shipped: f64,
    pub ms_per_token_optimised: f64,
    pub speedup: f64,
    /// True when the header came from one shard of a split model and the per-token figures
    /// were scaled up to the whole model.
    pub extrapolated_from_shard: bool,
    pub layers_seen: usize,
    pub layers_total: usize,
    /// Ceiling: what the same model would do if every token's experts were one contiguous run.
    pub ms_per_token_perfect: f64,
    pub perfect_speedup: f64,
    pub tok_per_s_shipped: f64,
    pub tok_per_s_optimised: f64,
}

/// Projects per-token fetch cost, given a measured device and an assumed clustering gain.
///
/// `fetch_reduction` is the fraction of fetches a layout removes, e.g. 0.144 for the value
/// measured on OLMoE.
pub fn project(g: &Gguf, name: &str, cm: &CostModel, fetch_reduction: f64) -> Result<Projection> {
    let moe = MoeModel::from_gguf(g)?;

    // Per-layer, because mixed quantisation makes bytes-per-expert vary within one file.
    let mut bytes = 0u64;
    let mut ranges = 0usize;
    let mut per_expert_sum = 0u64;
    for l in &moe.layers {
        let be = l.bytes_per_expert()?;
        per_expert_sum += be;
        bytes += be * moe.n_expert_used as u64;
        ranges += l.weight_tensors().count() * moe.n_expert_used as usize;
    }
    let bytes_per_expert = per_expert_sum / moe.layers.len().max(1) as u64;

    // A split GGUF puts a quarter of the layers in each shard, and each shard carries only
    // its own tensor index. Reporting shard 1's per-token cost as the model's would
    // understate a 480 B model fourfold, so scale by the block count the header declares.
    let layers_total = g.arch_u64("block_count").unwrap_or(moe.layers.len() as u64) as usize;
    let layers_seen = moe.layers.len();
    let shard_scale = if layers_seen > 0 && layers_total > layers_seen {
        layers_total as f64 / layers_seen as f64
    } else {
        1.0
    };
    let bytes = (bytes as f64 * shard_scale) as u64;
    let ranges = (ranges as f64 * shard_scale) as usize;

    let transfer_ns = bytes as f64 * cm.c_byte_ns;
    let shipped_ns = ranges as f64 * cm.c_fetch_ns + transfer_ns;
    let optimised_ns = ranges as f64 * (1.0 - fetch_reduction) * cm.c_fetch_ns + transfer_ns;
    // Perfect clustering: one run per layer, so one fetch per fused tensor per layer.
    let tensors_per_layer = moe.layers[0].weight_tensors().count();
    let perfect_ns =
        (layers_total.max(layers_seen) * tensors_per_layer) as f64 * cm.c_fetch_ns + transfer_ns;

    Ok(Projection {
        model: name.to_string(),
        arch: moe.arch.clone(),
        file_gb: g.file_size as f64 / 1e9,
        n_moe_layers: layers_total.max(layers_seen),
        n_expert: moe.n_expert,
        n_expert_used: moe.n_expert_used,
        bytes_per_expert,
        ranges_per_token: ranges,
        mib_per_token: bytes as f64 / (1 << 20) as f64,
        ranges_per_mib: ranges as f64 / (bytes as f64 / (1 << 20) as f64).max(1e-9),
        max_bridged_gap: cm.max_gap(bytes_per_expert, tensors_per_layer),
        fetch_overhead_share_pct: 100.0 * (shipped_ns - transfer_ns) / shipped_ns,
        ms_per_token_shipped: shipped_ns / 1e6,
        ms_per_token_optimised: optimised_ns / 1e6,
        speedup: shipped_ns / optimised_ns,
        extrapolated_from_shard: shard_scale > 1.0,
        layers_seen,
        layers_total,
        ms_per_token_perfect: perfect_ns / 1e6,
        perfect_speedup: shipped_ns / perfect_ns,
        tok_per_s_shipped: 1e9 / shipped_ns,
        tok_per_s_optimised: 1e9 / optimised_ns,
    })
}
