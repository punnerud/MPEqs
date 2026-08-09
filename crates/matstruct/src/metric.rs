//! The measurements matcodec's compression is conditional on.
//!
//! matcodec wins because road networks between separated regions connect through a few
//! gateways, so a cross-region block is `D[a][b] = d(a, gw) + HWY + d(gw, b)` — additive
//! rank-1 — and the exact residual then has near-zero entropy. It reports 6.4× on a real road
//! matrix, ~10× on single-gateway synthetic worlds, and **~1.8× on structureless points**.
//!
//! Whether a given matrix sits at the 10× end or the 1.8× end is an empirical question with
//! four parts, measured here in the order that can stop the work earliest:
//!
//! 1. Is it a metric at all? The rank-1 base *is* the triangle inequality written out, and
//!    the O(L) bounds are triangle bounds. Without it, half the machinery is meaningless.
//! 2. Does it cluster into regions? No regions, no gateways.
//! 3. Do cross-region blocks fit a rank-1 base? This is the mechanism itself.
//! 4. Is the residual low-entropy? This is what turns the fit into bytes saved.

use crate::matrix::Matrix;
use coact::cluster::Rng;
use serde::Serialize;

/// How badly, and how often, `d(a,c) > d(a,b) + d(b,c)`.
#[derive(Serialize, Debug, Default)]
pub struct TriangleReport {
    pub triples: u64,
    pub violations: u64,
    pub violation_pct: f64,
    /// Largest absolute violation, in the matrix's own units.
    pub max_violation: f64,
    /// Largest violation as a fraction of the shorter side, which is the scale-free version.
    pub max_relative: f64,
    pub mean_relative_violation: f64,
    /// A matrix is treated as metric when violations are rounding-scale only.
    pub is_metric: bool,
}

/// Samples triples and checks the triangle inequality.
///
/// `tol_rel` is the fraction of `d(a,c)` a violation may reach before it counts as real rather
/// than floating-point noise — matcodec's `TRIANGLE_TOL = 2` is the same idea on integer
/// travel times.
pub fn triangle(m: &Matrix, samples: u64, tol_rel: f64, seed: u64) -> TriangleReport {
    let n = m.n;
    let mut r = TriangleReport::default();
    if n < 3 {
        r.is_metric = true;
        return r;
    }
    let mut rng = Rng::new(seed);
    let mut rel_sum = 0.0f64;

    for _ in 0..samples {
        let a = rng.below(n);
        let b = rng.below(n);
        let c = rng.below(n);
        if a == b || b == c || a == c {
            continue;
        }
        r.triples += 1;
        let direct = m.get(a, c) as f64;
        let detour = m.get(a, b) as f64 + m.get(b, c) as f64;
        let excess = direct - detour;
        if excess <= 0.0 {
            continue;
        }
        // Relative to the direct edge: a 1 % overshoot on a long edge is noise, the same
        // absolute overshoot on a short one is not.
        let rel = if direct.abs() > 1e-12 {
            excess / direct.abs()
        } else {
            f64::INFINITY
        };
        if rel > tol_rel {
            r.violations += 1;
            rel_sum += rel;
            if excess > r.max_violation {
                r.max_violation = excess;
            }
            if rel > r.max_relative {
                r.max_relative = rel;
            }
        }
    }
    r.violation_pct = 100.0 * r.violations as f64 / r.triples.max(1) as f64;
    r.mean_relative_violation = if r.violations > 0 {
        rel_sum / r.violations as f64
    } else {
        0.0
    };
    // One in ten thousand triples off by a hair is arithmetic; one in twenty is not a metric.
    r.is_metric = r.violation_pct < 0.1;
    r
}

#[derive(Serialize, Debug)]
pub struct ClusterReport {
    pub k: usize,
    /// Mean distance within a cluster over mean distance between clusters. Below 1 means the
    /// clusters are real; near 1 means the partition is arbitrary.
    pub intra_over_inter: f64,
    pub mean_silhouette: f64,
    pub smallest_cluster: usize,
    pub largest_cluster: usize,
}

/// k-medoids on the (symmetrised) matrix, the same partitioning matcodec uses.
///
/// Plain alternating assignment/update — the partition only has to be good enough to expose
/// block structure, and a better clusterer would not change whether that structure exists.
pub fn kmedoids(m: &Matrix, k: usize, iters: usize, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let n = m.n;
    let k = k.clamp(1, n);
    let mut rng = Rng::new(seed);

    // k-means++ style seeding: spread the initial medoids out, so a bad random draw does not
    // decide the answer.
    let mut medoids = vec![rng.below(n)];
    while medoids.len() < k {
        let mut best = (f64::NEG_INFINITY, 0usize);
        for cand in 0..n {
            if medoids.contains(&cand) {
                continue;
            }
            let nearest = medoids
                .iter()
                .map(|&md| m.get(cand, md) as f64)
                .fold(f64::INFINITY, f64::min);
            if nearest > best.0 {
                best = (nearest, cand);
            }
        }
        medoids.push(best.1);
    }

    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        let mut changed = false;
        for (i, a) in assign.iter_mut().enumerate() {
            let mut best = (f64::INFINITY, 0usize);
            for (ci, &md) in medoids.iter().enumerate() {
                let d = m.get(i, md) as f64;
                if d < best.0 {
                    best = (d, ci);
                }
            }
            if *a != best.1 {
                *a = best.1;
                changed = true;
            }
        }
        // New medoid = the member minimising total distance to the rest of its cluster.
        for (ci, medoid) in medoids.iter_mut().enumerate() {
            let members: Vec<usize> = (0..n).filter(|&i| assign[i] == ci).collect();
            if members.is_empty() {
                continue;
            }
            let mut best = (f64::INFINITY, *medoid);
            for &cand in &members {
                let tot: f64 = members.iter().map(|&o| m.get(cand, o) as f64).sum();
                if tot < best.0 {
                    best = (tot, cand);
                }
            }
            *medoid = best.1;
        }
        if !changed {
            break;
        }
    }
    (assign, medoids)
}

/// Intra/inter ratio and silhouette for a partition, on a sample of points.
pub fn cluster_quality(m: &Matrix, assign: &[usize], k: usize, sample: usize) -> ClusterReport {
    let n = m.n;
    let step = (n / sample.max(1)).max(1);
    let (mut intra, mut intra_n, mut inter, mut inter_n) = (0.0f64, 0u64, 0.0f64, 0u64);
    let mut sil_sum = 0.0f64;
    let mut sil_n = 0u64;

    let mut sizes = vec![0usize; k];
    for &a in assign {
        sizes[a] += 1;
    }

    for i in (0..n).step_by(step) {
        let mut per_cluster = vec![(0.0f64, 0u64); k];
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = m.get(i, j) as f64;
            let c = assign[j];
            per_cluster[c].0 += d;
            per_cluster[c].1 += 1;
            if assign[i] == c {
                intra += d;
                intra_n += 1;
            } else {
                inter += d;
                inter_n += 1;
            }
        }
        let own = assign[i];
        let a_i = if per_cluster[own].1 > 0 {
            per_cluster[own].0 / per_cluster[own].1 as f64
        } else {
            0.0
        };
        let b_i = (0..k)
            .filter(|&c| c != own && per_cluster[c].1 > 0)
            .map(|c| per_cluster[c].0 / per_cluster[c].1 as f64)
            .fold(f64::INFINITY, f64::min);
        if b_i.is_finite() {
            let denom = a_i.max(b_i);
            if denom > 1e-12 {
                sil_sum += (b_i - a_i) / denom;
                sil_n += 1;
            }
        }
    }

    ClusterReport {
        k,
        intra_over_inter: (intra / intra_n.max(1) as f64)
            / (inter / inter_n.max(1) as f64).max(1e-12),
        mean_silhouette: sil_sum / sil_n.max(1) as f64,
        smallest_cluster: sizes.iter().copied().min().unwrap_or(0),
        largest_cluster: sizes.iter().copied().max().unwrap_or(0),
    }
}

/// How spread out the distances are.
///
/// This is the property that decides whether "regions connected by gateways" can exist at all.
/// A road network is embedded in two dimensions and has real bottlenecks, so distances span
/// orders of magnitude. In a high-dimensional space, distances concentrate — every pair ends
/// up roughly equidistant — and there is no far-apart structure for a gateway to bridge.
#[derive(Serialize, Debug, Default)]
pub struct SpreadReport {
    pub mean: f64,
    pub stddev: f64,
    /// `stddev / mean`. Road matrices run well above 0.5; concentrated high-dimensional
    /// spaces fall towards 0.
    pub coefficient_of_variation: f64,
    pub p01: f64,
    pub p99: f64,
    /// `p99 / p01`. How many times farther the far pairs are than the near ones.
    pub dynamic_range: f64,
}

pub fn spread(m: &Matrix) -> SpreadReport {
    let n = m.n;
    let mut v: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in i + 1..n {
            v.push(m.get(i, j) as f64);
        }
    }
    if v.is_empty() {
        return SpreadReport::default();
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / v.len() as f64;
    let sd = var.sqrt();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| v[((v.len() - 1) as f64 * f) as usize];
    let (p01, p99) = (q(0.01), q(0.99));
    SpreadReport {
        mean,
        stddev: sd,
        coefficient_of_variation: sd / mean.abs().max(1e-12),
        p01,
        p99,
        dynamic_range: p99 / p01.abs().max(1e-12),
    }
}

#[derive(Serialize, Debug, Default)]
pub struct Rank1Report {
    pub blocks: u64,
    /// Residual RMS as a fraction of the block's own RMS. 0 = the base is exact.
    pub mean_residual_ratio: f64,
    /// Cells the rank-1 base already reproduces within tolerance.
    pub cells_within_tol_pct: f64,
    /// Blocks where *every* cell is within tolerance. These are the ones a resident index
    /// answers with no decompression at all, so this number is the "read without decoding"
    /// claim, measured.
    ///
    /// Degenerate blocks are excluded. A block with a single row or column is reproduced
    /// exactly by the rank-1 base by construction — `col0` *is* the block — so counting it
    /// would report structure that is an arithmetic identity. This mattered: three separate
    /// runs reported 25 %, 46 % and 58 % "readable without decoding" purely because k-medoids
    /// had produced singleton clusters, and each time the number was believed before it was
    /// checked.
    pub blocks_fully_within_tol_pct: f64,
    /// Blocks skipped for being one row or one column wide.
    pub degenerate_blocks: u64,
    /// Bytes of zigzag-varint residual over bytes of raw f32, before any deflate.
    pub varint_ratio: f64,
    /// Bytes after deflate over raw. This is the compression ratio a codec would land near.
    pub deflate_ratio: f64,
    /// The null model: residual RMS if the base were just the block's mean, as a fraction of
    /// block RMS. A rank-1 base captures the mean for free, so it only earns its complexity if
    /// it beats this. Comparing the two separates "there is gateway structure" from "distances
    /// within a block are all about the same".
    pub mean_only_residual_ratio: f64,
    /// `1 - rank1/mean_only`. The share of the *remaining* variation that the gateway model
    /// explains. This, not the raw residual ratio, is matcodec's actual contribution.
    pub rank1_gain_over_mean_pct: f64,
}

fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

fn varint_len(mut v: u64) -> usize {
    let mut n = 1;
    while v >= 0x80 {
        v >>= 7;
        n += 1;
    }
    n
}

/// Fits matcodec's rank-1 base to every cross-cluster block and measures what is left.
///
/// The base is `base[p][q] = col0[p] + row0[q] - c00`, taken from the block's own first row
/// and column — exactly matcodec's construction. `quantum` converts the matrix to the integer
/// grid the codec would store on; the residual is then counted in those units, so the entropy
/// figures are the ones a real encoder would see.
pub fn rank1_fit(
    m: &Matrix,
    assign: &[usize],
    k: usize,
    quantum: f64,
    tol_units: i64,
) -> Rank1Report {
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &c) in assign.iter().enumerate() {
        members[c].push(i);
    }

    let mut rep = Rank1Report::default();
    let (mut ratio_sum, mut cells, mut within, mut full_blocks) = (0.0f64, 0u64, 0u64, 0u64);
    let mut mean_ratio_sum = 0.0f64;
    let mut nondegenerate = 0u64;
    let mut raw_bytes = 0usize;
    let mut varint_bytes = 0usize;
    let mut residual_stream: Vec<u8> = Vec::new();

    for a in 0..k {
        for b in 0..k {
            if a == b || members[a].is_empty() || members[b].is_empty() {
                continue;
            }
            let rows = &members[a];
            let cols = &members[b];
            let q = |i: usize, j: usize| (m.get(i, j) as f64 / quantum).round() as i64;

            let c00 = q(rows[0], cols[0]);
            let col0: Vec<i64> = rows.iter().map(|&p| q(p, cols[0])).collect();
            let row0: Vec<i64> = cols.iter().map(|&qq| q(rows[0], qq)).collect();

            let mut block_sq = 0.0f64;
            let mut resid_sq = 0.0f64;
            let mut all_within = true;
            // Null model: subtract only the block mean.
            let mut sum = 0.0f64;
            for &p in rows.iter() {
                for &qq in cols.iter() {
                    sum += q(p, qq) as f64;
                }
            }
            let blk_mean = sum / (rows.len() * cols.len()) as f64;
            let mut mean_resid_sq = 0.0f64;
            for (pi, &p) in rows.iter().enumerate() {
                for (qi, &qq) in cols.iter().enumerate() {
                    let actual = q(p, qq);
                    let base = col0[pi] + row0[qi] - c00;
                    let r = actual - base;
                    block_sq += (actual as f64) * (actual as f64);
                    resid_sq += (r as f64) * (r as f64);
                    let md = actual as f64 - blk_mean;
                    mean_resid_sq += md * md;
                    cells += 1;
                    if r.abs() <= tol_units {
                        within += 1;
                    } else {
                        all_within = false;
                    }
                    raw_bytes += 4;
                    let vl = varint_len(zigzag(r));
                    varint_bytes += vl;
                    let mut z = zigzag(r);
                    while z >= 0x80 {
                        residual_stream.push((z as u8) | 0x80);
                        z >>= 7;
                    }
                    residual_stream.push(z as u8);
                }
            }
            rep.blocks += 1;
            if rows.len() < 2 || cols.len() < 2 {
                rep.degenerate_blocks += 1;
            } else {
                nondegenerate += 1;
                if all_within {
                    full_blocks += 1;
                }
            }
            let cellsf = (rows.len() * cols.len()) as f64;
            let denom = (block_sq / cellsf).sqrt().max(1e-12);
            ratio_sum += (resid_sq / cellsf).sqrt() / denom;
            mean_ratio_sum += (mean_resid_sq / cellsf).sqrt() / denom;
        }
    }

    rep.mean_residual_ratio = ratio_sum / rep.blocks.max(1) as f64;
    rep.cells_within_tol_pct = 100.0 * within as f64 / cells.max(1) as f64;
    rep.blocks_fully_within_tol_pct = 100.0 * full_blocks as f64 / nondegenerate.max(1) as f64;
    rep.varint_ratio = varint_bytes as f64 / raw_bytes.max(1) as f64;
    rep.deflate_ratio = deflate_len(&residual_stream) as f64 / raw_bytes.max(1) as f64;
    rep.mean_only_residual_ratio = mean_ratio_sum / rep.blocks.max(1) as f64;
    rep.rank1_gain_over_mean_pct = if rep.mean_only_residual_ratio > 1e-12 {
        100.0 * (1.0 - rep.mean_residual_ratio / rep.mean_only_residual_ratio)
    } else {
        0.0
    };
    rep
}

/// Deflated size of a byte stream, without keeping the output.
fn deflate_len(raw: &[u8]) -> usize {
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;
    let mut e = ZlibEncoder::new(Vec::new(), Compression::best());
    e.write_all(raw).expect("deflate");
    e.finish().expect("deflate finish").len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A world where every cross-region trip goes through one gateway. This is the case
    /// matcodec is built for, and the probe must report it as such — if it does not, the probe
    /// is wrong rather than the data.
    fn single_gateway(n_per: usize) -> (Matrix, Vec<usize>) {
        let n = n_per * 2;
        let mut m = Matrix::zeros(n);
        // Distance to the gateway, per point, arbitrary but fixed.
        let to_gw: Vec<f32> = (0..n).map(|i| 1.0 + (i % 7) as f32).collect();
        const HWY: f32 = 100.0;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let same = (i < n_per) == (j < n_per);
                let d = if same {
                    (to_gw[i] - to_gw[j]).abs() + 1.0
                } else {
                    to_gw[i] + HWY + to_gw[j]
                };
                m.set(i, j, d);
            }
        }
        let assign = (0..n).map(|i| usize::from(i >= n_per)).collect();
        (m, assign)
    }

    #[test]
    fn rank1_base_is_exact_on_a_single_gateway_world() {
        let (m, assign) = single_gateway(16);
        let rep = rank1_fit(&m, &assign, 2, 1.0, 0);
        assert_eq!(rep.blocks, 2);
        assert!(
            rep.mean_residual_ratio < 1e-9,
            "gateway blocks must be exactly rank-1, got {}",
            rep.mean_residual_ratio
        );
        assert_eq!(rep.blocks_fully_within_tol_pct, 100.0);
        assert!(
            rep.deflate_ratio < 0.05,
            "an all-zero residual must compress to almost nothing, got {}",
            rep.deflate_ratio
        );
    }

    #[test]
    fn rank1_beats_the_mean_only_null_on_a_gateway_world() {
        let (m, assign) = single_gateway(16);
        let rep = rank1_fit(&m, &assign, 2, 1.0, 0);
        assert!(
            rep.mean_only_residual_ratio > 0.01,
            "the block mean alone must leave something for rank-1 to explain"
        );
        assert!(
            rep.rank1_gain_over_mean_pct > 99.0,
            "on a true gateway world rank-1 must explain essentially all of it, got {}",
            rep.rank1_gain_over_mean_pct
        );
    }

    #[test]
    fn singleton_clusters_do_not_count_as_readable_without_decoding() {
        // Structureless noise, but with one point alone in its own cluster. Its blocks are
        // one row or one column wide and so are exactly rank-1 by construction; reporting
        // them as gateway structure is the artefact this guard exists for.
        let n = 40;
        let mut m = Matrix::zeros(n);
        let mut rng = Rng::new(11);
        for i in 0..n {
            for j in i + 1..n {
                let v = 10.0 + rng.below(200) as f32;
                m.set(i, j, v);
                m.set(j, i, v);
            }
        }
        let assign: Vec<usize> = (0..n).map(|i| if i == 0 { 2 } else { i % 2 }).collect();
        let rep = rank1_fit(&m, &assign, 3, 1.0, 0);
        assert!(
            rep.degenerate_blocks >= 4,
            "the singleton must produce degenerate blocks"
        );
        assert_eq!(
            rep.blocks_fully_within_tol_pct, 0.0,
            "noise has no readable blocks once the singleton is excluded"
        );
    }

    #[test]
    fn structureless_noise_does_not_fit_rank1() {
        let n = 40;
        let mut m = Matrix::zeros(n);
        let mut rng = Rng::new(9);
        for i in 0..n {
            for j in i + 1..n {
                let v = 10.0 + (rng.below(200) as f32);
                m.set(i, j, v);
                m.set(j, i, v);
            }
        }
        let assign: Vec<usize> = (0..n).map(|i| i % 2).collect();
        let rep = rank1_fit(&m, &assign, 2, 1.0, 0);
        assert!(
            rep.mean_residual_ratio > 0.2,
            "noise must leave a large residual, got {}",
            rep.mean_residual_ratio
        );
    }

    #[test]
    fn triangle_probe_finds_a_planted_violation() {
        let n = 30;
        let mut m = Matrix::zeros(n);
        for i in 0..n {
            for j in i + 1..n {
                m.set(i, j, 10.0);
                m.set(j, i, 10.0);
            }
        }
        let clean = triangle(&m, 20_000, 0.01, 3);
        assert!(
            clean.is_metric,
            "a constant matrix satisfies the inequality"
        );

        // Make a few of node 0's edges far longer than any two-hop detour around them. Only
        // a few: if *every* edge from 0 were long, every detour would start with a long edge
        // too and nothing would violate.
        for j in 1..5 {
            m.set(0, j, 1000.0);
            m.set(j, 0, 1000.0);
        }
        let broken = triangle(&m, 20_000, 0.01, 3);
        assert!(!broken.is_metric, "a planted violation must be caught");
        assert!(
            broken.violation_pct > 0.5,
            "expected roughly 0.8 % of triples to violate, got {}",
            broken.violation_pct
        );
        assert!(broken.max_violation > 900.0);
    }

    #[test]
    fn kmedoids_recovers_two_planted_clusters() {
        let n = 40;
        let mut m = Matrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let same = (i < n / 2) == (j < n / 2);
                m.set(i, j, if same { 1.0 } else { 50.0 });
            }
        }
        let (assign, _) = kmedoids(&m, 2, 20, 1);
        let first = assign[0];
        assert!((0..n / 2).all(|i| assign[i] == first));
        assert!((n / 2..n).all(|i| assign[i] != first));

        let q = cluster_quality(&m, &assign, 2, 40);
        assert!(q.intra_over_inter < 0.1);
        assert!(q.mean_silhouette > 0.9);
    }
}
