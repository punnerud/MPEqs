//! Turning a co-activation graph into a physical expert ordering.
//!
//! This is MPEdb §8.7's min-cut, applied to a different graph: nodes are experts, edge
//! weight `w(i,j)` counts tokens that routed to both. Recursive spectral bisection with
//! Kernighan–Lin refinement lays tightly-coupled halves next to each other, so a token's
//! selected experts land in few contiguous runs. A final greedy pass optimises the actual
//! fetch objective rather than the cut proxy.

use crate::cost::{evaluate, CostModel, LayerGeometry, LayerStats, Permutation};
use crate::trace::LayerTrace;

/// Dense symmetric co-activation counts, zero diagonal.
#[derive(Debug, Clone)]
pub struct CoGraph {
    pub n: usize,
    pub w: Vec<f64>,
    /// Marginal selection count per expert.
    pub freq: Vec<u64>,
}

/// How much a co-activated pair contributes to an edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeWeight {
    /// Every co-selection counts the same.
    Count,
    /// Weight by the product of the two router gate probabilities.
    ///
    /// Selection alone overstates experts that are picked with a gate weight of 0.01 and
    /// barely touch the residual. Using the activation magnitude concentrates the layout on
    /// pairs that actually carry the token, which is the structure worth making contiguous.
    Gate,
}

impl CoGraph {
    pub fn build(t: &LayerTrace, n_expert: usize) -> Self {
        Self::build_with(t, n_expert, EdgeWeight::Count)
    }

    pub fn build_with(t: &LayerTrace, n_expert: usize, mode: EdgeWeight) -> Self {
        let use_gate = mode == EdgeWeight::Gate && t.has_weights();
        let mut w = vec![0.0f64; n_expert * n_expert];
        let mut freq = vec![0u64; n_expert];
        for ti in 0..t.n_tokens() {
            let tok = t.token(ti);
            let gw = t.token_weights(ti);
            for (a, &i) in tok.iter().enumerate() {
                freq[i as usize] += 1;
                for (b, &j) in tok.iter().enumerate().skip(a + 1) {
                    let contrib = if use_gate {
                        (gw[a] as f64) * (gw[b] as f64)
                    } else {
                        1.0
                    };
                    let (i, j) = (i as usize, j as usize);
                    w[i * n_expert + j] += contrib;
                    w[j * n_expert + i] += contrib;
                }
            }
        }
        CoGraph {
            n: n_expert,
            w,
            freq,
        }
    }

    #[inline]
    fn edge(&self, i: usize, j: usize) -> f64 {
        self.w[i * self.n + j]
    }

    /// Rich-get-richer sharpening: `w' = w · lift^alpha`, where lift is the observed
    /// co-selection over what independence predicts.
    ///
    /// This is what reinforcing successful routes would do to the graph, applied here as a
    /// pure analysis-side transform. Two readings, and the holdout decides between them: it
    /// either denoises — suppressing pairs that co-occur only by chance, so the layout
    /// optimises real structure — or it distorts, optimising a belief the access pattern does
    /// not share. Only the second would show up as worse holdout fetches.
    pub fn sharpen(&mut self, alpha: f64) {
        if alpha <= 0.0 {
            return;
        }
        let n = self.n;
        let total: f64 = self.freq.iter().map(|&f| f as f64).sum();
        if total <= 0.0 {
            return;
        }
        // Tokens, recovered from the marginals: each token contributes top_k selections.
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let expected = self.freq[i] as f64 * self.freq[j] as f64 / total;
                if expected <= 0.0 {
                    self.w[i * n + j] = 0.0;
                    continue;
                }
                let lift = self.w[i * n + j] / expected;
                if lift > 0.0 {
                    self.w[i * n + j] *= lift.powf(alpha);
                }
            }
        }
    }

    /// Sum of edge weights crossing between `part` members and non-members.
    pub fn cut(&self, nodes: &[usize], side: &[bool]) -> f64 {
        let mut c = 0.0;
        for (a, &i) in nodes.iter().enumerate() {
            for (b, &j) in nodes.iter().enumerate() {
                if b > a && side[a] != side[b] {
                    c += self.edge(i, j);
                }
            }
        }
        c
    }
}

/// Deterministic small-state PRNG — avoids a dependency and keeps runs reproducible.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

/// Fiedler vector of the sub-graph induced by `nodes`, via shifted power iteration.
///
/// The Laplacian's smallest eigenvalue is 0 with eigenvector 1. Iterating on
/// `(lambda_max * I - L)` while projecting out 1 converges to the eigenvector of the
/// smallest remaining eigenvalue, which is exactly the Fiedler vector.
fn fiedler(g: &CoGraph, nodes: &[usize]) -> Vec<f64> {
    let m = nodes.len();
    if m <= 2 {
        return vec![0.0; m];
    }

    let mut sub = vec![0.0f64; m * m];
    let mut deg = vec![0.0f64; m];
    for a in 0..m {
        for b in 0..m {
            if a != b {
                let v = g.edge(nodes[a], nodes[b]);
                sub[a * m + b] = v;
                deg[a] += v;
            }
        }
    }

    // Gershgorin bound on the largest Laplacian eigenvalue.
    let lambda_max = deg.iter().cloned().fold(0.0f64, f64::max) * 2.0 + 1.0;

    let mul = |x: &[f64], y: &mut [f64]| {
        for a in 0..m {
            // (lambda_max*I - L) x = lambda_max*x - deg[a]*x[a] + sum_b w[a][b]*x[b]
            let mut acc = (lambda_max - deg[a]) * x[a];
            for b in 0..m {
                if a != b {
                    acc += sub[a * m + b] * x[b];
                }
            }
            y[a] = acc;
        }
    };

    let mut x: Vec<f64> = (0..m)
        .map(|i| ((i * 2654435761) % 1000) as f64 / 500.0 - 1.0)
        .collect();
    let mut y = vec![0.0f64; m];
    for _ in 0..400 {
        // Project out the constant vector, the known eigenvector of eigenvalue 0.
        let mean = x.iter().sum::<f64>() / m as f64;
        for v in x.iter_mut() {
            *v -= mean;
        }
        let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < 1e-12 {
            return vec![0.0; m];
        }
        for v in x.iter_mut() {
            *v /= norm;
        }
        mul(&x, &mut y);
        std::mem::swap(&mut x, &mut y);
    }
    let mean = x.iter().sum::<f64>() / m as f64;
    for v in x.iter_mut() {
        *v -= mean;
    }
    x
}

/// One pass of Kernighan–Lin: repeatedly swap the cross-pair with the best gain.
fn kernighan_lin(g: &CoGraph, nodes: &[usize], side: &mut [bool], rounds: usize) {
    let m = nodes.len();
    for _ in 0..rounds {
        // D[a] = external - internal edge weight for node a.
        let mut d = vec![0.0f64; m];
        for a in 0..m {
            for b in 0..m {
                if a == b {
                    continue;
                }
                let w = g.edge(nodes[a], nodes[b]);
                if side[a] == side[b] {
                    d[a] -= w;
                } else {
                    d[a] += w;
                }
            }
        }

        let mut best = (0.0f64, usize::MAX, usize::MAX);
        for a in 0..m {
            for b in 0..m {
                if side[a] == side[b] {
                    continue;
                }
                let gain = d[a] + d[b] - 2.0 * g.edge(nodes[a], nodes[b]);
                if gain > best.0 {
                    best = (gain, a, b);
                }
            }
        }
        if best.1 == usize::MAX {
            break;
        }
        side.swap(best.1, best.2);
    }
}

/// Recursive balanced bisection; returns experts in physical slot order.
pub fn spectral_order(g: &CoGraph) -> Vec<u16> {
    let mut out = Vec::with_capacity(g.n);
    let all: Vec<usize> = (0..g.n).collect();
    bisect(g, &all, &mut out);
    out.into_iter().map(|i| i as u16).collect()
}

fn bisect(g: &CoGraph, nodes: &[usize], out: &mut Vec<usize>) {
    if nodes.len() <= 2 {
        out.extend_from_slice(nodes);
        return;
    }
    let f = fiedler(g, nodes);

    // Balanced split at the median of the Fiedler vector. Balance is not cosmetic: an
    // unbalanced recursion degenerates into a linear chain and loses the locality we are
    // after.
    let mut idx: Vec<usize> = (0..nodes.len()).collect();
    idx.sort_by(|&a, &b| f[a].partial_cmp(&f[b]).unwrap_or(std::cmp::Ordering::Equal));
    let half = nodes.len() / 2;
    let mut side = vec![false; nodes.len()];
    for &i in &idx[half..] {
        side[i] = true;
    }

    kernighan_lin(g, nodes, &mut side, nodes.len());

    let left: Vec<usize> = nodes
        .iter()
        .zip(&side)
        .filter(|(_, &s)| !s)
        .map(|(&n, _)| n)
        .collect();
    let right: Vec<usize> = nodes
        .iter()
        .zip(&side)
        .filter(|(_, &s)| s)
        .map(|(&n, _)| n)
        .collect();
    if left.is_empty() || right.is_empty() {
        out.extend_from_slice(nodes);
        return;
    }
    bisect(g, &left, out);
    bisect(g, &right, out);
}

/// Greedy nearest-neighbour chain over normalised affinity.
///
/// Recursive bisection minimises *cuts*, which is only a proxy: what actually merges two
/// fetches into one is two experts landing in adjacent slots. This builds the order by
/// repeatedly extending whichever end of the path has the strongest remaining neighbour,
/// which is that adjacency objective directly.
///
/// Affinity is normalised by expected co-occurrence, `w(i,j) / (freq_i · freq_j)`. Raw
/// counts would just chain the hottest experts together — they co-occur with everything,
/// so their raw edges are large without carrying any structure.
pub fn chain_order(g: &CoGraph) -> Vec<u16> {
    let n = g.n;
    if n < 2 {
        return (0..n as u16).collect();
    }
    let aff = |i: usize, j: usize| -> f64 {
        let d = (g.freq[i] as f64) * (g.freq[j] as f64);
        if d <= 0.0 {
            0.0
        } else {
            g.edge(i, j) / d
        }
    };

    let mut best = (f64::NEG_INFINITY, 0usize, 1usize);
    for i in 0..n {
        for j in i + 1..n {
            let a = aff(i, j);
            if a > best.0 {
                best = (a, i, j);
            }
        }
    }

    let mut placed = vec![false; n];
    let mut path = std::collections::VecDeque::new();
    path.push_back(best.1);
    path.push_back(best.2);
    placed[best.1] = true;
    placed[best.2] = true;

    while path.len() < n {
        let (front, back) = (*path.front().unwrap(), *path.back().unwrap());
        let mut pick = (f64::NEG_INFINITY, usize::MAX, false);
        for (c, &done) in placed.iter().enumerate() {
            if done {
                continue;
            }
            let af = aff(front, c);
            if af > pick.0 {
                pick = (af, c, true);
            }
            let ab = aff(back, c);
            if ab > pick.0 {
                pick = (ab, c, false);
            }
        }
        // Every expert is reachable because `aff` returns 0 rather than skipping, so a
        // disconnected expert still gets appended instead of stalling the loop.
        let (_, c, at_front) = pick;
        placed[c] = true;
        if at_front {
            path.push_front(c);
        } else {
            path.push_back(c);
        }
    }
    path.into_iter().map(|i| i as u16).collect()
}

pub fn frequency_order(g: &CoGraph) -> Vec<u16> {
    let mut idx: Vec<u16> = (0..g.n as u16).collect();
    idx.sort_by_key(|&e| std::cmp::Reverse(g.freq[e as usize]));
    idx
}

pub fn random_order(n: usize, seed: u64) -> Vec<u16> {
    let mut v: Vec<u16> = (0..n as u16).collect();
    Rng::new(seed).shuffle(&mut v);
    v
}

/// Moves an expert out of slot `from` and reinserts it at slot `to`, shifting the rest.
///
/// Swapping two slots is a poor neighbourhood for a linear arrangement: it disturbs two
/// positions at once and cannot slide a well-placed cluster along the axis. Relocation moves
/// one expert and shifts the block between, which is the standard escape from swap-local
/// optima in minimum-linear-arrangement problems.
fn relocate(p: &Permutation, from: usize, to: usize) -> Permutation {
    let mut v = p.as_slice().to_vec();
    let e = v.remove(from);
    v.insert(to, e);
    Permutation::from_perm(v).expect("relocation preserves the bijection")
}

/// Local search on the real objective, over both swap and relocation neighbourhoods.
///
/// The cut is a proxy; this optimises what `fetchbench` will actually time. Runs on the
/// training split only.
pub fn greedy_refine(
    start: Permutation,
    train: &LayerTrace,
    geom: LayerGeometry,
    cm: &CostModel,
    max_sweeps: usize,
    seed: u64,
) -> (Permutation, LayerStats) {
    let mut p = start;
    let mut best = evaluate(train.tokens(), &p, geom, cm);
    let n = p.len();
    let mut rng = Rng::new(seed);

    for _ in 0..max_sweeps {
        let mut improved = false;
        let mut pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
            .collect();
        rng.shuffle(&mut pairs);

        for (a, b) in pairs {
            p.swap(a, b);
            let cand = evaluate(train.tokens(), &p, geom, cm);
            if cand.cost_ns < best.cost_ns {
                best = cand;
                improved = true;
            } else {
                p.swap(a, b);
            }
        }

        let mut moves: Vec<(usize, usize)> = (0..n)
            .flat_map(|a| (0..n).filter(move |&b| b != a).map(move |b| (a, b)))
            .collect();
        rng.shuffle(&mut moves);
        for (from, to) in moves {
            let cand_p = relocate(&p, from, to);
            let cand = evaluate(train.tokens(), &cand_p, geom, cm);
            if cand.cost_ns < best.cost_ns {
                best = cand;
                p = cand_p;
                improved = true;
            }
        }

        if !improved {
            break;
        }
    }
    (p, best)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace_from(tokens: &[[u16; 4]]) -> LayerTrace {
        LayerTrace {
            layer: 0,
            top_k: 4,
            experts: tokens.iter().flatten().copied().collect(),
            weights: Vec::new(),
            norms: Vec::new(),
        }
    }

    fn weighted_trace(tokens: &[[u16; 4]], w: &[[f32; 4]]) -> LayerTrace {
        LayerTrace {
            layer: 0,
            top_k: 4,
            experts: tokens.iter().flatten().copied().collect(),
            weights: w.iter().flatten().copied().collect(),
            norms: Vec::new(),
        }
    }

    #[test]
    fn cograph_counts_pairs() {
        let t = trace_from(&[[0, 1, 2, 3], [0, 1, 4, 5]]);
        let g = CoGraph::build(&t, 8);
        assert_eq!(g.edge(0, 1), 2.0);
        assert_eq!(g.edge(2, 3), 1.0);
        assert_eq!(g.edge(2, 4), 0.0);
        assert_eq!(g.freq[0], 2);
        assert_eq!(g.freq[6], 0);
    }

    #[test]
    fn gate_weighting_discounts_low_probability_experts() {
        // Experts 0 and 1 always carry the token; 2 and 3 ride along at 1 % gate weight.
        let toks = [[0u16, 1, 2, 3]; 10];
        let w = [[0.49f32, 0.49, 0.01, 0.01]; 10];
        let counted = CoGraph::build_with(&weighted_trace(&toks, &w), 8, EdgeWeight::Count);
        let gated = CoGraph::build_with(&weighted_trace(&toks, &w), 8, EdgeWeight::Gate);
        assert_eq!(counted.edge(0, 1), counted.edge(2, 3));
        assert!(gated.edge(0, 1) > gated.edge(2, 3) * 100.0);
    }

    #[test]
    fn gate_mode_falls_back_when_the_trace_has_no_weights() {
        let t = trace_from(&[[0, 1, 2, 3]]);
        let g = CoGraph::build_with(&t, 8, EdgeWeight::Gate);
        assert_eq!(g.edge(0, 1), 1.0);
    }

    #[test]
    fn sharpening_widens_the_gap_between_strong_and_weak_pairs() {
        // Pair (0,1) co-occurs far more than chance; (4,5) only as often as chance.
        let mut tokens = Vec::new();
        for i in 0..200u16 {
            if i % 2 == 0 {
                tokens.push([0, 1, 2, 3]);
            } else {
                tokens.push([4, 5, 6, 7]);
            }
        }
        let mut g = CoGraph::build(&trace_from(&tokens), 8);
        let before = g.edge(0, 1) / g.edge(0, 2).max(1e-12);
        g.sharpen(2.0);
        let after = g.edge(0, 1) / g.edge(0, 2).max(1e-12);
        assert!(
            after >= before,
            "sharpening must not shrink the lead of the stronger pair"
        );
        assert!(g.edge(0, 1) > 0.0, "a real pair must survive sharpening");
    }

    #[test]
    fn sharpening_with_alpha_zero_changes_nothing() {
        let t = trace_from(&[[0, 1, 2, 3], [0, 1, 4, 5]]);
        let mut g = CoGraph::build(&t, 8);
        let before = g.w.clone();
        g.sharpen(0.0);
        assert_eq!(before, g.w);
    }

    #[test]
    fn spectral_order_is_a_permutation() {
        let mut tokens = Vec::new();
        for i in 0..200u16 {
            // Two disjoint communities that never co-activate.
            if i % 2 == 0 {
                tokens.push([0, 1, 2, 3]);
            } else {
                tokens.push([12, 13, 14, 15]);
            }
        }
        let g = CoGraph::build(&trace_from(&tokens), 16);
        let order = spectral_order(&g);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..16u16).collect::<Vec<_>>());

        // Each community must occupy contiguous slots.
        let p = Permutation::from_perm(order).unwrap();
        let mut a: Vec<usize> = [0u16, 1, 2, 3].iter().map(|&e| p.slot_of(e)).collect();
        a.sort_unstable();
        assert_eq!(
            a[3] - a[0],
            3,
            "community 0 should be contiguous, got {a:?}"
        );
    }

    #[test]
    fn relocation_preserves_the_permutation() {
        let p = Permutation::identity(8);
        let q = relocate(&p, 6, 1);
        let mut seen = q.as_slice().to_vec();
        seen.sort_unstable();
        assert_eq!(seen, (0..8u16).collect::<Vec<_>>());
        assert_eq!(q.expert_at(1), 6);
        assert_eq!(q.expert_at(2), 1);
    }

    #[test]
    fn greedy_refine_never_regresses() {
        let mut tokens = Vec::new();
        for i in 0..100u16 {
            tokens.push([i % 16, (i + 1) % 16, (i + 2) % 16, (i + 3) % 16]);
        }
        let t = trace_from(&tokens);
        let geom = LayerGeometry {
            n_tensors: 3,
            bytes_per_expert: 4096,
        };
        let cm = CostModel::assumed_apple_nvme();
        let start = Permutation::identity(16);
        let before = evaluate(t.tokens(), &start, geom, &cm);
        let (_, after) = greedy_refine(start, &t, geom, &cm, 3, 7);
        assert!(after.cost_ns <= before.cost_ns);
    }
}
