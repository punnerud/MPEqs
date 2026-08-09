//! Decomposing the damage from fetching fewer experts into *scale* and *direction*.
//!
//! OLMoE has `norm_topk_prob = false`, so the gate weights are raw softmax probabilities over
//! all 64 experts and the top-8 sum to only ~0.43. Dropping half of them therefore shrinks
//! the FFN output by roughly a quarter before it has lost any information at all. That part
//! of the damage is a magnitude error and a single scalar fixes it. The rest is genuine loss.
//!
//! Telling those apart needs the actual expert output vectors, not their norms — a truncated
//! sum's norm depends on how the vectors align. `moetrace --vecs` streams them as f16.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

const VEC_MAGIC: u32 = 0x564E_4F4D; // "MOEV"

pub struct VecStream {
    reader: BufReader<File>,
    pub n_embd: usize,
    pub top_k: usize,
    pub n_records: u64,
    scratch: Vec<u8>,
}

impl VecStream {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut reader = BufReader::with_capacity(1 << 22, f);
        let mut hdr = [0u8; 24];
        reader.read_exact(&mut hdr)?;
        let u32_at = |o: usize| u32::from_le_bytes(hdr[o..o + 4].try_into().unwrap());
        if u32_at(0) != VEC_MAGIC {
            bail!("{} is not a MOEV vector stream", path.display());
        }
        let n_embd = u32_at(8) as usize;
        let top_k = u32_at(12) as usize;
        let n_records = u64::from_le_bytes(hdr[16..24].try_into().unwrap());
        if n_embd == 0 || top_k == 0 {
            bail!(
                "{} has an empty header; the capture did not finish",
                path.display()
            );
        }
        let scratch = vec![0u8; top_k * n_embd * 2];
        Ok(VecStream {
            reader,
            n_embd,
            top_k,
            n_records,
            scratch,
        })
    }

    /// Reads the next record's `top_k * n_embd` values, converted to f32.
    pub fn next_into(&mut self, out: &mut Vec<f32>) -> Result<bool> {
        match self.reader.read_exact(&mut self.scratch) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(false),
            Err(e) => return Err(e.into()),
        }
        out.clear();
        out.reserve(self.top_k * self.n_embd);
        for c in self.scratch.chunks_exact(2) {
            out.push(f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
        }
        Ok(true)
    }
}

/// IEEE half to single. ggml writes plain f16, so no denormal shortcuts.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if frac == 0 => sign << 31,
        0 => {
            // Subnormal: renormalise into a single-precision exponent.
            let mut e = -1i32;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            let f = f & 0x3ff;
            (sign << 31) | (((127 - 15 + 1 + e) as u32) << 23) | (f << 13)
        }
        0x1f => (sign << 31) | (0xff << 23) | (frac << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

/// Error of one truncation policy against the untruncated FFN output.
#[derive(Serialize, Default, Clone)]
pub struct PolicyError {
    pub keep: usize,
    /// Keep the router's first `keep`, weights untouched.
    pub truncate_pct: f64,
    /// Same selection, weights rescaled so they sum to what all `top_k` summed to.
    pub renormalised_pct: f64,
    /// Keep the `keep` largest contributors, weights untouched.
    pub oracle_pct: f64,
    /// Both: oracle selection and rescaled weights.
    pub oracle_renormalised_pct: f64,
    /// How much of `truncate` is pure magnitude, i.e. what renormalising recovers.
    pub scale_share_pct: f64,
    /// Error after the least-squares-optimal single scalar. This is the floor for *any*
    /// rescaling scheme, so it separates "we picked the wrong scalar" from "no scalar helps".
    pub best_scalar_pct: f64,
    /// The optimal scalar itself. Renormalising uses `w_total / w_kept` instead; the gap
    /// between the two is why that heuristic overshoots.
    pub best_scalar: f64,
    /// Error after solving for the least-squares-optimal *per-expert* weights over the kept
    /// set. This is the ceiling for co-tuning the split and the combine together: no learned
    /// combiner can beat it, because it already knows the answer it is trying to reproduce.
    pub best_weights_pct: f64,
    /// Same, with the oracle choosing which experts to keep.
    pub oracle_best_weights_pct: f64,
}

/// Solves `G a = b` for a small symmetric positive-definite `G`, with a ridge term.
///
/// Expert outputs are near-orthogonal so `G` is well conditioned, but two experts that happen
/// to align on one token would make it singular; the ridge keeps that from blowing up.
fn solve_spd(mut g: Vec<f64>, mut b: Vec<f64>, n: usize) -> Vec<f64> {
    let trace: f64 = (0..n).map(|i| g[i * n + i]).sum();
    let ridge = 1e-9 * (trace / n.max(1) as f64).max(1e-30);
    for i in 0..n {
        g[i * n + i] += ridge;
    }
    // Gaussian elimination with partial pivoting; n is at most top_k.
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if g[r * n + col].abs() > g[piv * n + col].abs() {
                piv = r;
            }
        }
        if piv != col {
            for c in 0..n {
                g.swap(col * n + c, piv * n + c);
            }
            b.swap(col, piv);
        }
        let d = g[col * n + col];
        if d.abs() < 1e-30 {
            continue;
        }
        for r in col + 1..n {
            let f = g[r * n + col] / d;
            if f == 0.0 {
                continue;
            }
            for c in col..n {
                g[r * n + c] -= f * g[col * n + c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut acc = b[i];
        for j in i + 1..n {
            acc -= g[i * n + j] * x[j];
        }
        let d = g[i * n + i];
        x[i] = if d.abs() < 1e-30 { 0.0 } else { acc / d };
    }
    x
}

fn rel_err(approx: &[f32], full: &[f32], full_norm: f64) -> f64 {
    if full_norm <= 0.0 {
        return 0.0;
    }
    let d: f64 = approx
        .iter()
        .zip(full)
        .map(|(&a, &b)| {
            let d = a as f64 - b as f64;
            d * d
        })
        .sum();
    d.sqrt() / full_norm
}

/// Per-layer truncation error, and the best way to spend a global expert budget.
///
/// A uniform top-k spends the same number of fetches on every layer. If layers differ in how
/// much they lose when truncated, that is the wrong allocation: the budget should go where it
/// buys the most. This measures the per-layer curves and then allocates greedily against them,
/// which is optimal when each curve is convex — and they are, since each successive expert
/// contributes less than the last.
#[derive(Serialize)]
pub struct LayerBudget {
    pub layer: u32,
    /// Relative L2 error at each keep depth, index 0 = keep 1.
    pub error_by_keep: Vec<f64>,
    /// Experts this layer gets under the allocation.
    pub allocated_keep: usize,
}

#[derive(Serialize)]
pub struct BudgetPlan {
    pub total_experts: usize,
    pub uniform_keep: usize,
    pub uniform_mean_error_pct: f64,
    pub allocated_mean_error_pct: f64,
    pub layers: Vec<LayerBudget>,
}

/// Greedy allocation of `total` expert slots across layers, minimising mean error.
pub fn allocate(per_layer: &[LayerErrorCurve], total: usize) -> BudgetPlan {
    let n_layers = per_layer.len();
    let top_k = per_layer.first().map_or(0, |(_, v)| v.len());
    let mut keep = vec![1usize; n_layers];
    let mut spent = n_layers;

    // Each step, give one more expert to whichever layer's error drops the most.
    while spent < total {
        let mut best = (0.0f64, usize::MAX);
        for (i, (_, errs)) in per_layer.iter().enumerate() {
            if keep[i] >= top_k {
                continue;
            }
            let gain = errs[keep[i] - 1] - errs[keep[i]];
            if gain > best.0 {
                best = (gain, i);
            }
        }
        if best.1 == usize::MAX {
            break;
        }
        keep[best.1] += 1;
        spent += 1;
    }

    let uniform_keep = (total / n_layers.max(1)).clamp(1, top_k.max(1));
    let uniform_err = per_layer
        .iter()
        .map(|(_, e)| e[uniform_keep - 1])
        .sum::<f64>()
        / n_layers as f64;
    let alloc_err = per_layer
        .iter()
        .enumerate()
        .map(|(i, (_, e))| e[keep[i] - 1])
        .sum::<f64>()
        / n_layers as f64;

    BudgetPlan {
        total_experts: total,
        uniform_keep,
        uniform_mean_error_pct: 100.0 * uniform_err,
        allocated_mean_error_pct: 100.0 * alloc_err,
        layers: per_layer
            .iter()
            .enumerate()
            .map(|(i, (l, e))| LayerBudget {
                layer: *l,
                error_by_keep: e.iter().map(|x| 100.0 * x).collect(),
                allocated_keep: keep[i],
            })
            .collect(),
    }
}

/// Mean truncation error per keep depth, for one layer.
pub type LayerErrorCurve = (u32, Vec<f64>);

/// Mean relative L2 error of each policy, over every record in the streams.
pub fn evaluate(
    trace_path: &Path,
    vec_path: &Path,
) -> Result<(Vec<PolicyError>, u64, Vec<LayerErrorCurve>)> {
    let (top_k, records) = crate::trace::load_raw(trace_path)?;
    let mut vs = VecStream::open(vec_path)?;
    if vs.top_k != top_k {
        bail!(
            "trace has top_k {top_k} but the vector stream has {}",
            vs.top_k
        );
    }
    let n_embd = vs.n_embd;

    let mut acc = vec![PolicyError::default(); top_k];
    let mut by_layer: std::collections::BTreeMap<u32, (Vec<f64>, u64)> =
        std::collections::BTreeMap::new();
    let mut n = 0u64;
    let mut vecs: Vec<f32> = Vec::new();
    let mut full = vec![0f32; n_embd];
    let mut part = vec![0f32; n_embd];

    for rec in &records {
        if !vs.next_into(&mut vecs)? {
            break;
        }
        if rec.weights.is_empty() {
            bail!("trace has no gate weights; recapture with --contrib");
        }

        let w: Vec<f64> = rec.weights.iter().map(|&x| x as f64).collect();
        let w_total: f64 = w.iter().sum();

        full.iter_mut().for_each(|v| *v = 0.0);
        for j in 0..top_k {
            let v = &vecs[j * n_embd..(j + 1) * n_embd];
            for d in 0..n_embd {
                full[d] += (w[j] * v[d] as f64) as f32;
            }
        }
        let full_norm: f64 = full
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        if full_norm <= 0.0 {
            continue;
        }

        // Contribution order, for the oracle policies.
        let mut order: Vec<usize> = (0..top_k).collect();
        let norms: Vec<f64> = (0..top_k)
            .map(|j| {
                let v = &vecs[j * n_embd..(j + 1) * n_embd];
                w[j] * v
                    .iter()
                    .map(|&x| (x as f64) * (x as f64))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        order.sort_by(|&x, &y| norms[y].partial_cmp(&norms[x]).unwrap());

        for keep in 1..=top_k {
            // Least-squares-optimal per-expert weights over the kept set.
            let optimal_weights = |sel: &[usize]| -> f64 {
                let m = sel.len();
                let mut gram = vec![0.0f64; m * m];
                let mut rhs = vec![0.0f64; m];
                for (a, &i) in sel.iter().enumerate() {
                    let vi = &vecs[i * n_embd..(i + 1) * n_embd];
                    for (b2, &j) in sel.iter().enumerate().skip(a) {
                        let vj = &vecs[j * n_embd..(j + 1) * n_embd];
                        let d: f64 = (0..n_embd).map(|d| vi[d] as f64 * vj[d] as f64).sum();
                        gram[a * m + b2] = d;
                        gram[b2 * m + a] = d;
                    }
                    rhs[a] = (0..n_embd).map(|d| vi[d] as f64 * full[d] as f64).sum();
                }
                let alpha = solve_spd(gram, rhs, m);
                let mut rec = vec![0f32; n_embd];
                for (a, &i) in sel.iter().enumerate() {
                    let vi = &vecs[i * n_embd..(i + 1) * n_embd];
                    for d in 0..n_embd {
                        rec[d] += (alpha[a] * vi[d] as f64) as f32;
                    }
                }
                rel_err(&rec, &full, full_norm)
            };

            let mut eval = |sel: &[usize]| -> (f64, f64, f64, f64) {
                part.iter_mut().for_each(|v| *v = 0.0);
                let mut kept_w = 0.0;
                for &j in sel {
                    kept_w += w[j];
                    let v = &vecs[j * n_embd..(j + 1) * n_embd];
                    for d in 0..n_embd {
                        part[d] += (w[j] * v[d] as f64) as f32;
                    }
                }
                let plain = rel_err(&part, &full, full_norm);
                // Rescale so the kept weights carry the full gate mass.
                let s = if kept_w > 0.0 { w_total / kept_w } else { 1.0 };
                let scaled: Vec<f32> = part.iter().map(|&x| (x as f64 * s) as f32).collect();
                // The projection of `full` onto `part` — the scalar no other scalar beats.
                let dot: f64 = part
                    .iter()
                    .zip(&full)
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum();
                let pn: f64 = part.iter().map(|&x| (x as f64) * (x as f64)).sum();
                let alpha = if pn > 0.0 { dot / pn } else { 1.0 };
                let best: Vec<f32> = part.iter().map(|&x| (x as f64 * alpha) as f32).collect();
                (
                    plain,
                    rel_err(&scaled, &full, full_norm),
                    rel_err(&best, &full, full_norm),
                    alpha,
                )
            };

            let gate_sel: Vec<usize> = (0..keep).collect();
            let bw = optimal_weights(&gate_sel);
            let obw = optimal_weights(&order[..keep]);
            let (t, tr_, bs, alpha) = eval(&gate_sel);
            let (o, or, _, _) = eval(&order[..keep]);

            let a = &mut acc[keep - 1];
            a.keep = keep;
            a.truncate_pct += t;
            a.renormalised_pct += tr_;
            a.oracle_pct += o;
            a.oracle_renormalised_pct += or;
            a.best_scalar_pct += bs;
            a.best_scalar += alpha;
            a.best_weights_pct += bw;
            a.oracle_best_weights_pct += obw;

            let e = by_layer
                .entry(rec.layer as u32)
                .or_insert_with(|| (vec![0.0; top_k], 0));
            e.0[keep - 1] += t;
        }
        by_layer
            .entry(rec.layer as u32)
            .or_insert_with(|| (vec![0.0; top_k], 0))
            .1 += 1;
        n += 1;
    }

    if n == 0 {
        bail!("no records were paired between the trace and the vector stream");
    }
    let per_layer: Vec<LayerErrorCurve> = by_layer
        .into_iter()
        .map(|(l, (sums, cnt))| (l, sums.iter().map(|s| s / cnt as f64).collect()))
        .collect();
    for a in &mut acc {
        a.truncate_pct = 100.0 * a.truncate_pct / n as f64;
        a.renormalised_pct = 100.0 * a.renormalised_pct / n as f64;
        a.oracle_pct = 100.0 * a.oracle_pct / n as f64;
        a.oracle_renormalised_pct = 100.0 * a.oracle_renormalised_pct / n as f64;
        a.best_scalar_pct = 100.0 * a.best_scalar_pct / n as f64;
        a.best_scalar /= n as f64;
        a.best_weights_pct = 100.0 * a.best_weights_pct / n as f64;
        a.oracle_best_weights_pct = 100.0 * a.oracle_best_weights_pct / n as f64;
        a.scale_share_pct = if a.truncate_pct > 0.0 {
            100.0 * (a.truncate_pct - a.renormalised_pct) / a.truncate_pct
        } else {
            0.0
        };
    }
    Ok((acc, n, per_layer))
}

/// How the eight expert outputs relate to each other and to the sum they form.
///
/// If the layer is an ensemble vote — experts that broadly agree and reinforce — the
/// alignment ratio is near 1 and every expert points the same way as the total. If they are
/// complementary features it is near `1/sqrt(k)`. If some experts are *corrections* that
/// subtract, their cosine against the total is negative, and dropping them does not just
/// lose signal: it leaves an uncorrected error behind, which is exactly what renormalising
/// then amplifies.
#[derive(Serialize, Default, Clone)]
pub struct EnsembleStats {
    pub rank: usize,
    /// Mean cosine between this rank's output and the full weighted sum.
    pub cosine_with_total: f64,
    /// Mean cosine against the other seven experts of the same token.
    pub mean_pairwise_cosine: f64,
    /// Share of the total's norm this rank would add if it were perfectly aligned.
    pub projection_share_pct: f64,
}

#[derive(Serialize, Clone)]
pub struct EnsembleSummary {
    /// `‖Σ w v‖ / Σ w‖v‖`. 1.0 means perfect agreement, `1/sqrt(k)` means orthogonal.
    pub alignment_ratio: f64,
    /// The same ratio if the experts were exactly orthogonal, for reference.
    pub orthogonal_reference: f64,
    pub n_records: u64,
    pub per_rank: Vec<EnsembleStats>,
}

pub fn ensemble(trace_path: &Path, vec_path: &Path) -> Result<EnsembleSummary> {
    let (top_k, records) = crate::trace::load_raw(trace_path)?;
    let mut vs = VecStream::open(vec_path)?;
    let n_embd = vs.n_embd;

    let mut cos_total = vec![0.0f64; top_k];
    let mut cos_pair = vec![0.0f64; top_k];
    let mut proj = vec![0.0f64; top_k];
    let mut align_num = 0.0f64;
    let mut align_den = 0.0f64;
    let mut n = 0u64;

    let mut vecs: Vec<f32> = Vec::new();
    let mut total = vec![0f64; n_embd];

    for rec in &records {
        if !vs.next_into(&mut vecs)? {
            break;
        }
        if rec.weights.is_empty() {
            bail!("trace has no gate weights; recapture with --contrib");
        }
        let w: Vec<f64> = rec.weights.iter().map(|&x| x as f64).collect();

        total.iter_mut().for_each(|v| *v = 0.0);
        let mut sum_of_norms = 0.0;
        let mut wn = vec![0.0f64; top_k];
        for j in 0..top_k {
            let v = &vecs[j * n_embd..(j + 1) * n_embd];
            let mut nn = 0.0;
            for d in 0..n_embd {
                let x = w[j] * v[d] as f64;
                total[d] += x;
                nn += x * x;
            }
            wn[j] = nn.sqrt();
            sum_of_norms += wn[j];
        }
        let tn = total.iter().map(|x| x * x).sum::<f64>().sqrt();
        if tn <= 0.0 || sum_of_norms <= 0.0 {
            continue;
        }
        align_num += tn;
        align_den += sum_of_norms;

        for j in 0..top_k {
            if wn[j] <= 0.0 {
                continue;
            }
            let vj = &vecs[j * n_embd..(j + 1) * n_embd];
            let dot: f64 = (0..n_embd).map(|d| w[j] * vj[d] as f64 * total[d]).sum();
            cos_total[j] += dot / (wn[j] * tn);
            proj[j] += dot / (tn * tn);

            let mut pc = 0.0;
            let mut cnt = 0.0;
            for i in 0..top_k {
                if i == j || wn[i] <= 0.0 {
                    continue;
                }
                let vi = &vecs[i * n_embd..(i + 1) * n_embd];
                let d2: f64 = (0..n_embd)
                    .map(|d| w[j] * vj[d] as f64 * w[i] * vi[d] as f64)
                    .sum();
                pc += d2 / (wn[j] * wn[i]);
                cnt += 1.0;
            }
            if cnt > 0.0 {
                cos_pair[j] += pc / cnt;
            }
        }
        n += 1;
    }
    if n == 0 {
        bail!("no records were paired between the trace and the vector stream");
    }

    let per_rank = (0..top_k)
        .map(|j| EnsembleStats {
            rank: j + 1,
            cosine_with_total: cos_total[j] / n as f64,
            mean_pairwise_cosine: cos_pair[j] / n as f64,
            projection_share_pct: 100.0 * proj[j] / n as f64,
        })
        .collect();

    Ok(EnsembleSummary {
        alignment_ratio: align_num / align_den,
        orthogonal_reference: 1.0 / (top_k as f64).sqrt(),
        n_records: n,
        per_rank,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trips_known_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert!((f16_to_f32(0x3555) - 0.333).abs() < 1e-3);
    }

    #[test]
    fn relative_error_is_zero_for_an_exact_match() {
        let a = [1.0f32, 2.0, 3.0];
        let n = (14.0f64).sqrt();
        assert!(rel_err(&a, &a, n) < 1e-12);
    }
}
