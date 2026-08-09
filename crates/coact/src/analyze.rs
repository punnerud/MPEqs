//! Two questions about MoE structure that decide what a layout can and cannot buy.
//!
//! **Contribution.** The router's gate weight is its *opinion* of an expert. What reaches
//! the residual stream is `weight × ‖expert output‖`. If the bottom-ranked expert really
//! contributed a fraction of a percent, declining to fetch it would be nearly free. That is
//! an empirical question, and it is answered here rather than assumed.
//!
//! **Cross-layer predictability.** Every prefetch scheme in the MoE-offload literature bets
//! that seeing layer L's experts narrows down layer L+1's. If routing is close to
//! independent across layers, that bet does not pay, and layout — which needs no prediction
//! at all — is the only lever left. Measured here as the mutual information between
//! consecutive layers' expert sets, in bits, against the entropy of the layer on its own.

use crate::trace::Trace;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Serialize)]
pub struct RankContribution {
    pub rank: usize,
    pub mean_gate_weight: f64,
    pub mean_output_norm: f64,
    /// `gate × norm`, as a share of the token's total across all ranks.
    pub contribution_pct: f64,
    /// Contribution of this rank and every rank above it.
    pub cumulative_pct: f64,
}

#[derive(Serialize)]
pub struct CrossLayer {
    pub from_layer: u32,
    pub to_layer: u32,
    /// Entropy of the next layer's top-1 expert, in bits. log2(64) = 6 for OLMoE.
    pub entropy_bits: f64,
    /// Entropy remaining once the previous layer's top-1 expert is known.
    pub conditional_entropy_bits: f64,
    /// How much knowing the previous layer tells you. 0 = independent.
    pub mutual_information_bits: f64,
    /// Best achievable accuracy predicting the next top-1 from the previous top-1.
    pub top1_predictability_pct: f64,
}

/// Per-rank contribution, pooled across layers. Requires a trace captured with `--contrib`.
pub fn rank_contributions(tr: &Trace) -> Vec<RankContribution> {
    let k = tr.top_k;
    let mut gate = vec![0.0f64; k];
    let mut norm = vec![0.0f64; k];
    let mut contrib = vec![0.0f64; k];
    let mut n = 0u64;

    for lt in &tr.layers {
        if !lt.has_norms() || !lt.has_weights() {
            continue;
        }
        for t in 0..lt.n_tokens() {
            let w = lt.token_weights(t);
            let o = lt.token_norms(t);
            for j in 0..k {
                gate[j] += w[j] as f64;
                norm[j] += o[j] as f64;
                contrib[j] += (w[j] as f64) * (o[j] as f64);
            }
            n += 1;
        }
    }
    if n == 0 {
        return Vec::new();
    }

    let total: f64 = contrib.iter().sum();
    let mut cum = 0.0;
    (0..k)
        .map(|j| {
            cum += contrib[j];
            RankContribution {
                rank: j + 1,
                mean_gate_weight: gate[j] / n as f64,
                mean_output_norm: norm[j] / n as f64,
                contribution_pct: 100.0 * contrib[j] / total,
                cumulative_pct: 100.0 * cum / total,
            }
        })
        .collect()
}

/// How much better a *perfect* reranker could do, per truncation depth.
///
/// The router ranks by gate probability. What matters is contribution, `w · ‖out‖`. If those
/// two orders agree, the router is already extracting everything its inputs allow and no
/// amount of retraining helps. If they disagree, the gap between them is the ceiling on any
/// rerank-based scheme — including the one this project would otherwise go on to build.
#[derive(Serialize)]
pub struct Headroom {
    /// Experts kept out of the full top-k.
    pub keep: usize,
    /// Contribution captured keeping the first `keep` in the router's own order.
    pub gate_capture_pct: f64,
    /// Contribution captured keeping the `keep` largest contributors.
    pub oracle_capture_pct: f64,
    /// `oracle - gate`. The entire budget available to a better router.
    pub headroom_pct: f64,
    /// Mean overlap between the two selections, as a fraction of `keep`.
    pub selection_agreement_pct: f64,
}

/// Rank correlation between gate order and contribution order, per layer.
#[derive(Serialize)]
pub struct RankAgreement {
    pub layer: u32,
    pub spearman: f64,
    /// Fraction of tokens where the single largest contributor is also the router's top pick.
    pub argmax_agreement_pct: f64,
}

fn spearman_from_ranks(a: &[usize], b: &[usize]) -> f64 {
    // Both inputs are permutations of 0..n, so the shortcut formula is exact: no ties.
    let n = a.len() as f64;
    if n < 2.0 {
        return 1.0;
    }
    let d2: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum();
    1.0 - 6.0 * d2 / (n * (n * n - 1.0))
}

/// Contribution captured by gate order versus by an oracle, for every truncation depth.
pub fn headroom(tr: &Trace) -> (Vec<Headroom>, Vec<RankAgreement>) {
    let k = tr.top_k;
    let mut gate_cap = vec![0.0f64; k];
    let mut oracle_cap = vec![0.0f64; k];
    let mut agree = vec![0.0f64; k];
    let mut total = 0.0f64;
    let mut n = 0u64;

    let mut per_layer = Vec::new();
    let mut contrib = vec![0.0f64; k];
    let mut order: Vec<usize> = Vec::with_capacity(k);
    let mut rank_of: Vec<usize> = vec![0; k];

    for lt in &tr.layers {
        if !lt.has_norms() || !lt.has_weights() {
            continue;
        }
        let (mut sp_sum, mut argmax_hits, mut ln) = (0.0f64, 0u64, 0u64);

        for t in 0..lt.n_tokens() {
            let w = lt.token_weights(t);
            let o = lt.token_norms(t);
            let mut sum = 0.0;
            for j in 0..k {
                contrib[j] = (w[j] as f64) * (o[j] as f64);
                sum += contrib[j];
            }
            if sum <= 0.0 {
                continue;
            }

            // Descending by contribution. The trace already stores gate order, so index j
            // *is* the gate rank.
            order.clear();
            order.extend(0..k);
            order.sort_by(|&x, &y| contrib[y].partial_cmp(&contrib[x]).unwrap());
            for (r, &j) in order.iter().enumerate() {
                rank_of[j] = r;
            }

            let gate_ranks: Vec<usize> = (0..k).collect();
            let contrib_ranks: Vec<usize> = (0..k).map(|j| rank_of[j]).collect();
            sp_sum += spearman_from_ranks(&gate_ranks, &contrib_ranks);
            if order[0] == 0 {
                argmax_hits += 1;
            }
            ln += 1;

            let mut g = 0.0;
            let mut oc = 0.0;
            for keep in 1..=k {
                g += contrib[keep - 1];
                oc += contrib[order[keep - 1]];
                gate_cap[keep - 1] += g / sum;
                oracle_cap[keep - 1] += oc / sum;
                // Gate keeps ranks 0..keep; the oracle keeps `order[..keep]`.
                let hits = order[..keep].iter().filter(|&&j| j < keep).count();
                agree[keep - 1] += hits as f64 / keep as f64;
            }
            total += sum;
            n += 1;
        }

        if ln > 0 {
            per_layer.push(RankAgreement {
                layer: lt.layer as u32,
                spearman: sp_sum / ln as f64,
                argmax_agreement_pct: 100.0 * argmax_hits as f64 / ln as f64,
            });
        }
    }
    let _ = total;

    if n == 0 {
        return (Vec::new(), per_layer);
    }
    let rows = (1..=k)
        .map(|keep| {
            let g = 100.0 * gate_cap[keep - 1] / n as f64;
            let o = 100.0 * oracle_cap[keep - 1] / n as f64;
            Headroom {
                keep,
                gate_capture_pct: g,
                oracle_capture_pct: o,
                headroom_pct: o - g,
                selection_agreement_pct: 100.0 * agree[keep - 1] / n as f64,
            }
        })
        .collect();
    (rows, per_layer)
}

/// Does the co-activation graph contain tight groups, or is it flat?
///
/// The whole layout idea assumes experts form clusters that recur — "these three usually come
/// together, so fetch them as one read". Whether that is true is measurable: compare each
/// pair's observed co-selection count against what independence predicts,
/// `lift(i,j) = observed / (freq_i · freq_j / n_tokens)`. Lift near 1 everywhere means the
/// router picks its eight essentially independently and there is nothing to group.
#[derive(Serialize)]
pub struct ClusterStats {
    pub layer: u32,
    /// Lift at the 50th, 90th, 99th percentile, and the maximum.
    pub lift_p50: f64,
    pub lift_p90: f64,
    pub lift_p99: f64,
    pub lift_max: f64,
    /// Share of pairs co-selected at least twice as often as independence predicts.
    pub pairs_above_2x_pct: f64,
    /// Mean lift among the `n_expert` strongest pairs — the ones a layout can actually exploit,
    /// since each expert has only two neighbours on a line.
    pub top_pairs_mean_lift: f64,
}

pub fn clusters(tr: &Trace) -> Vec<ClusterStats> {
    let n = tr.n_expert as usize;
    let mut out = Vec::new();
    for lt in &tr.layers {
        let n_tok = lt.n_tokens();
        if n_tok == 0 {
            continue;
        }
        let mut w = vec![0.0f64; n * n];
        let mut freq = vec![0.0f64; n];
        for t in 0..n_tok {
            let tok = lt.token(t);
            for (a, &i) in tok.iter().enumerate() {
                freq[i as usize] += 1.0;
                for &j in &tok[a + 1..] {
                    w[i as usize * n + j as usize] += 1.0;
                    w[j as usize * n + i as usize] += 1.0;
                }
            }
        }
        let nt = n_tok as f64;
        let mut lifts = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in i + 1..n {
                let expected = freq[i] * freq[j] / nt;
                if expected <= 0.0 {
                    continue;
                }
                lifts.push(w[i * n + j] / expected);
            }
        }
        if lifts.is_empty() {
            continue;
        }
        lifts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |f: f64| lifts[((lifts.len() - 1) as f64 * f) as usize];
        let above2 = lifts.iter().filter(|&&x| x >= 2.0).count() as f64;
        // A linear layout gives every expert at most two neighbours, so only about `n` pairs
        // can ever be made adjacent. Those are the ones whose lift matters.
        let top: f64 = lifts.iter().rev().take(n).sum::<f64>() / n as f64;
        out.push(ClusterStats {
            layer: lt.layer as u32,
            lift_p50: q(0.5),
            lift_p90: q(0.9),
            lift_p99: q(0.99),
            lift_max: *lifts.last().unwrap(),
            pairs_above_2x_pct: 100.0 * above2 / lifts.len() as f64,
            top_pairs_mean_lift: top,
        });
    }
    out
}

/// Mutual information between the top-1 expert of consecutive MoE layers.
pub fn cross_layer(tr: &Trace) -> Vec<CrossLayer> {
    let mut out = Vec::new();
    for w in tr.layers.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        let n = a.n_tokens().min(b.n_tokens());
        if n == 0 {
            continue;
        }

        let mut joint: HashMap<(u16, u16), u64> = HashMap::new();
        let mut pa: HashMap<u16, u64> = HashMap::new();
        let mut pb: HashMap<u16, u64> = HashMap::new();
        for t in 0..n {
            // Rank 0 is the highest-probability expert; argsort puts it first.
            let (x, y) = (a.token(t)[0], b.token(t)[0]);
            *joint.entry((x, y)).or_default() += 1;
            *pa.entry(x).or_default() += 1;
            *pb.entry(y).or_default() += 1;
        }
        let nf = n as f64;
        let h = |m: &HashMap<u16, u64>| -> f64 {
            -m.values()
                .map(|&c| {
                    let p = c as f64 / nf;
                    p * p.log2()
                })
                .sum::<f64>()
        };
        let hb = h(&pb);
        let hjoint: f64 = -joint
            .values()
            .map(|&c| {
                let p = c as f64 / nf;
                p * p.log2()
            })
            .sum::<f64>();
        let ha = h(&pa);
        let mi = ha + hb - hjoint;

        // Best single-guess accuracy: for each preceding expert, always predict the most
        // common successor.
        let mut best: HashMap<u16, u64> = HashMap::new();
        for (&(x, _), &c) in &joint {
            let e = best.entry(x).or_default();
            *e = (*e).max(c);
        }
        let hits: u64 = best.values().sum();

        out.push(CrossLayer {
            from_layer: a.layer as u32,
            to_layer: b.layer as u32,
            entropy_bits: hb,
            conditional_entropy_bits: hb - mi,
            mutual_information_bits: mi,
            top1_predictability_pct: 100.0 * hits as f64 / nf,
        });
    }
    out
}
