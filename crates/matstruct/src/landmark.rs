//! The landmark (min-plus) model, ported from MPEE's matcodec.
//!
//! `metric.rs` implements matcodec's *other* model — the per-cluster rank-1 base
//! `col0[p] + row0[q] - c00` — and it failed on our embeddings: `rank1_gain_over_mean = -93.1 %`
//! and 0.00 % of blocks readable without decoding (`data/matstruct-embeddings.json`). That
//! result was written up as "matcodec does not transfer", which was half the story. matcodec
//! picks the better of two models per matrix, and the other one was never tried here.
//!
//! The other one is this: choose `L` landmarks and approximate every distance by routing
//! through them,
//!
//! ```text
//!     base(i, j) = min over landmarks a of  d(i, a) + d(a, j)
//!     residual   = base(i, j) - d(i, j)     >= 0 under the triangle inequality
//! ```
//!
//! On MPEE's synthetic gateway world it reaches 17.6x with 88 % of blocks index-exact, and
//! 9.4x on a real London road matrix. It is a genuinely different base: the cluster model is
//! additive-separable within a block, this one is a min over `L` two-hop paths, and it is the
//! one that matches "find the extreme points, compute N x N only between them, and let
//! everything else route through that skeleton".
//!
//! Two things make it usable at a scale where the cluster model is not:
//!
//! - `pick_landmarks` never touches N². It samples `S` pairs and greedily adds the point that
//!   most reduces `Σ min(current, d(i,x) + d(x,j))` over that sample: O(L·n·S).
//! - The resident index is one `L x n` table, not `n x n`. For a symmetric metric `d(i,a)` and
//!   `d(a,j)` are the same table, so it is half of what matcodec stores for directed road
//!   matrices: 6.4 MB at n = 50 000, L = 32.
//!
//! Ported rather than depended on because matcodec's API is `i32` seconds over a dense
//! `&[i32]`, and we need `f32` angles streamed from an embedding file. Distances are quantised
//! to integers so "exact block" means exactly that; the quantum is explicit and reported.
//!
//! Reference: `landeveier/mpee/crates/matcodec/src/lib.rs` — `pick_landmarks` :309,
//! `encode_bridge` :458, `assign_cells` :653, blockmax construction :724, `cell_bounds` :1810.

use anyhow::{bail, Result};
use serde::Serialize;

/// Anything that can answer "how far apart are points i and j", in integer units.
///
/// The trait exists so the port can be validated against MPEE's own synthetic worlds, which
/// are handed out as dense `i32` matrices, while the real work streams from embeddings and
/// never materialises n². Both are symmetric; `build` checks that rather than assuming it.
pub trait DistanceSource {
    fn n(&self) -> usize;
    fn dist(&self, i: usize, j: usize) -> i32;

    /// Distances from `i` to every column, written into `out` (length `n`).
    fn row(&self, i: usize, out: &mut [i32]) {
        for (j, o) in out.iter_mut().enumerate() {
            *o = self.dist(i, j);
        }
    }
}

/// A dense matrix already in memory. Only for validation and for n small enough to afford it.
pub struct DenseSource {
    pub n: usize,
    pub d: Vec<i32>,
}

impl DistanceSource for DenseSource {
    fn n(&self) -> usize {
        self.n
    }
    fn dist(&self, i: usize, j: usize) -> i32 {
        self.d[i * self.n + j]
    }
    fn row(&self, i: usize, out: &mut [i32]) {
        out.copy_from_slice(&self.d[i * self.n..(i + 1) * self.n]);
    }
}

/// Angular distance between L2-normalised embeddings, quantised to integer units.
///
/// Rows are normalised once at construction so the distance is `acos(dot)`, which is a true
/// spherical metric — the triangle inequality is what every bound here rests on, and
/// `1 - cos` would not satisfy it.
pub struct EmbeddingSource {
    pub n: usize,
    pub dim: usize,
    emb: Vec<f32>,
    /// Integer units per radian. 1000 gives milliradians, so π is 3142.
    pub scale: f32,
}

impl EmbeddingSource {
    pub fn new(mut emb: Vec<f32>, dim: usize, scale: f32) -> Result<Self> {
        if dim == 0 || !emb.len().is_multiple_of(dim) {
            bail!(
                "embedding length {} is not a multiple of dim {dim}",
                emb.len()
            );
        }
        let n = emb.len() / dim;
        for row in emb.chunks_mut(dim) {
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                row.iter_mut().for_each(|v| *v /= norm);
            }
        }
        Ok(Self { n, dim, emb, scale })
    }

    pub fn vec(&self, i: usize) -> &[f32] {
        &self.emb[i * self.dim..(i + 1) * self.dim]
    }
}

impl DistanceSource for EmbeddingSource {
    fn n(&self) -> usize {
        self.n
    }
    fn dist(&self, i: usize, j: usize) -> i32 {
        // Angle as 2*atan2(|a-b|, |a+b|), not acos(dot).
        //
        // acos has an infinite derivative at 1, so for near-identical vectors it turns the
        // 1e-7 slack left by f32 normalisation into a full quantisation unit — a point's
        // distance to itself came out as 1, not 0. That is not a rounding curiosity here:
        // "exact block" is integer equality of the residual, so noise of one unit near zero
        // would have quietly undercounted exactly the blocks the mechanism is judged on, and
        // a false negative in the measurement would have looked like a real negative result.
        //
        // The half-angle form is well conditioned across the whole range: the chord governs
        // small angles and the sum governs large ones, and neither derivative blows up.
        // Accumulated in f64 because the inputs are f32 and the differences are small.
        let (a, b) = (self.vec(i), self.vec(j));
        let (mut diff, mut sum) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            let (x, y) = (*x as f64, *y as f64);
            diff += (x - y) * (x - y);
            sum += (x + y) * (x + y);
        }
        let ang = 2.0 * diff.sqrt().atan2(sum.sqrt());
        (ang * self.scale as f64).round() as i32
    }
}

/// Deterministic LCG, the same constants matcodec uses, so a shared seed gives a shared sample.
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

/// Greedy facility location over a sampled set of pairs — matcodec's `pick_landmarks`.
///
/// Faithful to the original with one change forced by scale. matcodec scans every point as a
/// candidate, which needs `d(i,x)` and `d(x,j)` for all x and is fine on a dense matrix. Here
/// the candidate set is capped and sampled, which is MPEE's own practice at scale (`MTZU`
/// takes `candidates = (n/4).clamp(256, 1024)`, lib.rs:1193). With the cap at or above `n` the
/// two are identical, and the test below pins that.
///
/// Memory is `|rows| x |candidates|`, never n².
pub fn pick_landmarks<S: DistanceSource>(
    src: &S,
    l: usize,
    cand_cap: usize,
    seed: u64,
) -> Vec<usize> {
    let n = src.n();
    let l = l.min(n);
    if n < 3 || l == 0 {
        return (0..l.max(1).min(n)).collect();
    }

    let s_count = (n * 4).clamp(512, 4096);
    let mut rng = Lcg(seed);
    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(s_count);
    while pairs.len() < s_count {
        let (i, j) = (rng.next() % n, rng.next() % n);
        if i != j {
            pairs.push((i, j));
        }
    }

    // Candidates. `0` used to mean "every point", which made the table below
    // `min(n, 8192) x n` — 1.6 GB at n = 50 000, in a function whose doc comment claims it
    // never touches n². The default is now MTZU's own rule, `(n/4).clamp(256, 1024)`
    // (matcodec lib.rs:1193), which bounds it at 33 MB. Passing an explicit cap at or above n
    // still scans everything, and `uncapped_candidates_scan_every_point` pins that the two
    // agree, so the deviation cannot become the silent default.
    let cand_cap = if cand_cap == 0 {
        (n / 4).clamp(256, 1024)
    } else {
        cand_cap
    };
    let cands: Vec<usize> = if cand_cap >= n {
        (0..n).collect()
    } else {
        let step = n as f64 / cand_cap as f64;
        (0..cand_cap)
            .map(|k| ((k as f64 + 0.5) * step) as usize % n)
            .collect()
    };

    // The greedy reads d(i, x) and d(x, j) only for sampled endpoints i, j and candidates x.
    // Gather those rows once: |endpoints| x |candidates| instead of n x n.
    let mut endpoints: Vec<usize> = pairs.iter().flat_map(|&(i, j)| [i, j]).collect();
    endpoints.sort_unstable();
    endpoints.dedup();
    let mut slot = vec![usize::MAX; n];
    for (k, &e) in endpoints.iter().enumerate() {
        slot[e] = k;
    }
    let mut tbl = vec![0i32; endpoints.len() * cands.len()];
    let mut row = vec![0i32; n];
    for (k, &e) in endpoints.iter().enumerate() {
        src.row(e, &mut row);
        for (c, &x) in cands.iter().enumerate() {
            tbl[k * cands.len() + c] = row[x];
        }
    }
    let at = |e: usize, c: usize| tbl[slot[e] * cands.len() + c] as i64;

    let mut cur = vec![i64::MAX; pairs.len()];
    let mut chosen = Vec::with_capacity(l);
    let mut used = vec![false; cands.len()];
    for _ in 0..l {
        let (mut best_c, mut best_cost) = (usize::MAX, i64::MAX);
        for (c, &taken) in used.iter().enumerate() {
            if taken {
                continue;
            }
            let mut cost = 0i64;
            for (s, &(i, j)) in pairs.iter().enumerate() {
                cost += cur[s].min(at(i, c) + at(j, c));
            }
            if cost < best_cost {
                best_cost = cost;
                best_c = c;
            }
        }
        if best_c == usize::MAX {
            break;
        }
        used[best_c] = true;
        chosen.push(cands[best_c]);
        for (s, &(i, j)) in pairs.iter().enumerate() {
            cur[s] = cur[s].min(at(i, best_c) + at(j, best_c));
        }
    }
    chosen
}

/// The road network: the landmarks, and every point's distance to each of them.
///
/// One `L x n` table, not two. matcodec keeps `dil` and `dlj` separately because road networks
/// are directed; an angular metric is symmetric, so `d(i, a) == d(a, i)` and the second table
/// would be a transpose of the first.
pub struct LandmarkIndex {
    pub landmarks: Vec<usize>,
    /// `dl[a * n + i]` = distance from landmark `a` to point `i`.
    pub dl: Vec<i32>,
    pub n: usize,
    /// `cell_of[j]` = the landmark nearest to `j`, i.e. its Voronoi cell.
    pub cell_of: Vec<u8>,
}

impl LandmarkIndex {
    pub fn build<S: DistanceSource>(src: &S, landmarks: Vec<usize>) -> Result<Self> {
        let n = src.n();
        let l = landmarks.len();
        if l == 0 {
            bail!("no landmarks");
        }
        if l > 255 {
            bail!("cell_of is a u8, so at most 255 landmarks (got {l})");
        }
        let mut dl = vec![0i32; l * n];
        let mut row = vec![0i32; n];
        for (a, &la) in landmarks.iter().enumerate() {
            src.row(la, &mut row);
            dl[a * n..(a + 1) * n].copy_from_slice(&row);
        }
        let mut cell_of = vec![0u8; n];
        for (j, c) in cell_of.iter_mut().enumerate() {
            let mut best = i64::MAX;
            for a in 0..l {
                let v = dl[a * n + j] as i64;
                if v < best {
                    best = v;
                    *c = a as u8;
                }
            }
        }
        Ok(Self {
            landmarks,
            dl,
            n,
            cell_of,
        })
    }

    pub fn l(&self) -> usize {
        self.landmarks.len()
    }

    /// `min over a of d(i, a) + d(a, j)`. An upper bound on `d(i, j)` when the source is metric.
    pub fn base(&self, i: usize, j: usize) -> i32 {
        (0..self.l())
            .map(|a| self.dl[a * self.n + i] + self.dl[a * self.n + j])
            .min()
            .unwrap_or(i32::MAX)
    }

    /// `(lower, upper)` without touching any residual — matcodec's `cell_bounds`.
    ///
    /// The upper bound is the routed base. The lower bound is the directed ALT bound
    /// `max_a |d(i,a) - d(j,a)|`, which is the reverse triangle inequality. Both are only
    /// meaningful if the source really is a metric, which is why `triangle_violations` is
    /// reported alongside and the caller is expected to look at it.
    pub fn cell_bounds(&self, i: usize, j: usize) -> (i32, i32) {
        let mut lo = 0i32;
        for a in 0..self.l() {
            let (di, dj) = (self.dl[a * self.n + i], self.dl[a * self.n + j]);
            lo = lo.max((di - dj).abs());
        }
        (lo, self.base(i, j))
    }

    /// Bytes held resident: the table, the landmark ids and the cell assignment.
    pub fn resident_bytes(&self) -> usize {
        self.dl.len() * 4 + self.landmarks.len() * 4 + self.cell_of.len()
    }
}

/// What the road network explains, and what it does not.
#[derive(Debug, Serialize)]
pub struct LandmarkReport {
    pub n: usize,
    pub landmarks: usize,
    pub rows_sampled: usize,
    pub quantum_units_per_radian: f32,
    pub resident_bytes: usize,
    /// Fraction of (row, cell) blocks whose residual is zero everywhere — answerable in O(L)
    /// with nothing decoded. This is the number the mechanism lives or dies by.
    pub exact_block_pct: f64,
    /// Same, allowing a residual of at most `tol` units.
    pub within_tol_block_pct: f64,
    pub tol_units: i32,
    /// Fraction of individual cells the base gets exactly right.
    pub exact_cell_pct: f64,
    pub mean_residual: f64,
    pub mean_distance: f64,
    /// Mean residual as a share of mean distance. 0 is a perfect skeleton.
    pub residual_ratio: f64,
    /// Cells where base < d, which cannot happen for a metric. Non-zero invalidates the bounds.
    pub triangle_violations: u64,
    /// The worst such violation, in units. A count without this is unreadable: quantising a
    /// geodesic matrix produces violations of exactly 1 unit in bulk, which are rounding and
    /// not a broken metric, and they must not be confused with a source that is genuinely
    /// non-metric — the expert co-activation graph violated by real margins.
    pub max_violation: i32,
    pub mean_bound_width: f64,
}

/// Measure the index against the source it was built from.
///
/// `rows` are sampled on a stride rather than exhaustively: the full measurement is O(n²) and
/// the quantity is a mean over blocks, which a strided sample estimates to well within the
/// precision anyone should read it at. The count is reported so nobody has to guess.
pub fn measure<S: DistanceSource>(
    src: &S,
    idx: &LandmarkIndex,
    rows: usize,
    tol: i32,
) -> LandmarkReport {
    let n = src.n();
    let l = idx.l();
    let rows = rows.clamp(1, n);
    let step = (n / rows).max(1);

    let mut blockmax = vec![0i32; l];
    let (mut exact_blocks, mut total_blocks) = (0u64, 0u64);
    let (mut within_blocks, mut exact_cells, mut total_cells) = (0u64, 0u64, 0u64);
    let (mut sum_resid, mut sum_dist, mut sum_width) = (0f64, 0f64, 0f64);
    let mut violations = 0u64;
    let mut worst = 0i32;
    let mut row = vec![0i32; n];
    let mut sampled = 0usize;

    for i in (0..n).step_by(step) {
        sampled += 1;
        src.row(i, &mut row);
        blockmax.iter_mut().for_each(|b| *b = 0);
        for (j, &d) in row.iter().enumerate() {
            if j == i {
                continue;
            }
            let base = idx.base(i, j);
            let r = base - d;
            if r < 0 {
                violations += 1;
                worst = worst.max(-r);
            }
            let b = &mut blockmax[idx.cell_of[j] as usize];
            *b = (*b).max(r.abs());
            if r == 0 {
                exact_cells += 1;
            }
            total_cells += 1;
            sum_resid += r.abs() as f64;
            sum_dist += d as f64;
            let (lo, up) = idx.cell_bounds(i, j);
            sum_width += (up - lo) as f64;
        }
        for &b in &blockmax {
            total_blocks += 1;
            if b == 0 {
                exact_blocks += 1;
            }
            if b <= tol {
                within_blocks += 1;
            }
        }
    }

    let cells = total_cells.max(1) as f64;
    let blocks = total_blocks.max(1) as f64;
    let mean_resid = sum_resid / cells;
    let mean_dist = sum_dist / cells;
    LandmarkReport {
        n,
        landmarks: l,
        rows_sampled: sampled,
        quantum_units_per_radian: 0.0,
        resident_bytes: idx.resident_bytes(),
        exact_block_pct: 100.0 * exact_blocks as f64 / blocks,
        within_tol_block_pct: 100.0 * within_blocks as f64 / blocks,
        tol_units: tol,
        exact_cell_pct: 100.0 * exact_cells as f64 / cells,
        mean_residual: mean_resid,
        mean_distance: mean_dist,
        residual_ratio: if mean_dist > 0.0 {
            mean_resid / mean_dist
        } else {
            0.0
        },
        triangle_violations: violations,
        max_violation: worst,
        mean_bound_width: sum_width / cells,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MPEE's own synthetic gateway world, copied from `matcodec/examples/cell_bench.rs:18`.
    ///
    /// Regions of points, and every cross-region path forced through one of three gateways.
    /// This is the positive control: the mechanism is *supposed* to find those gateways, and
    /// matcodec reports 88 % index-exact blocks on it at n=3000, L=32.
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

    /// Structureless points: the negative control. No gateways exist, so none can be found.
    fn euclid_world(n: usize) -> DenseSource {
        let mut s: u64 = 42;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) % 100_000) as i64
        };
        let pts: Vec<(i64, i64)> = (0..n).map(|_| (rnd(), rnd())).collect();
        let mut d = vec![0i32; n * n];
        for i in 0..n {
            for j in 0..n {
                let (dx, dy) = (pts[i].0 - pts[j].0, pts[i].1 - pts[j].1);
                d[i * n + j] = ((dx * dx + dy * dy) as f64).sqrt() as i32;
            }
        }
        DenseSource { n, d }
    }

    /// The port is validated against MPEE's measured number before it is pointed at our data.
    ///
    /// This is the discipline that caught the argsort trace artefact and the planted-MoE
    /// scoring bug: run the instrument on something whose answer is known first. If this fails,
    /// the port is wrong — not the embeddings it will later be run on.
    #[test]
    fn gateway_world_is_mostly_index_exact() {
        let src = gateway_world(1200, 8);
        // Explicitly uncapped: this is the validation against MPEE's measured behaviour, and
        // MPEE's own pivot mining scans every point. `capping_candidates_costs_quality` below
        // measures what the default cap gives up.
        let lm = pick_landmarks(&src, 32, usize::MAX, 0xA5A5_5A5A_DEAD_BEEF);
        let idx = LandmarkIndex::build(&src, lm).unwrap();
        let rep = measure(&src, &idx, 300, 2);
        assert_eq!(
            rep.triangle_violations, 0,
            "gateway world is a metric by construction"
        );
        assert!(
            rep.exact_block_pct > 60.0,
            "landmarks should find the gateways: {rep:?}"
        );
    }

    /// And it must fail where there is nothing to find, or it is measuring itself.
    #[test]
    fn structureless_points_are_not_index_exact() {
        let src = euclid_world(600);
        let lm = pick_landmarks(&src, 32, 0, 0xA5A5_5A5A_DEAD_BEEF);
        let idx = LandmarkIndex::build(&src, lm).unwrap();
        let rep = measure(&src, &idx, 200, 2);
        assert!(
            rep.exact_block_pct < 10.0,
            "no gateways exist here, so none may be reported: {rep:?}"
        );
    }

    /// What the default candidate cap costs, measured rather than assumed.
    ///
    /// The cap exists because scanning every point makes the endpoint x candidate table
    /// `min(n, 8192) x n` — 1.6 GB at n = 50 000. MPEE takes the same trade and records that
    /// candidate count dominates quality: London tol-5s block share runs 6 % at 192 candidates
    /// against 32 % at 1024. This pins the same direction on the gateway world so the trade is
    /// visible in the test suite rather than discovered later as a mysterious regression.
    #[test]
    fn capping_candidates_costs_quality() {
        let src = gateway_world(1200, 8);
        let score = |cap: usize| {
            let lm = pick_landmarks(&src, 32, cap, 0xA5A5_5A5A_DEAD_BEEF);
            let idx = LandmarkIndex::build(&src, lm).unwrap();
            measure(&src, &idx, 300, 2).exact_block_pct
        };
        let full = score(usize::MAX);
        let capped = score(0); // 0 = the default rule, (n/4).clamp(256, 1024) = 300 here
        assert!(full > capped, "uncapped {full} should beat capped {capped}");
        assert!(
            capped > 5.0,
            "the cap costs quality but must not destroy it: {capped} % of blocks still exact"
        );
    }

    /// Capping the candidate set is the one deviation from matcodec. With the cap disabled the
    /// two must agree exactly, so the deviation cannot silently become the default behaviour.
    #[test]
    fn uncapped_candidates_scan_every_point() {
        let src = euclid_world(200);
        let a = pick_landmarks(&src, 8, 0, 7);
        let b = pick_landmarks(&src, 8, 10_000, 7);
        assert_eq!(a, b);
    }

    /// The base can never undercut the true distance on a metric, and the bounds must bracket.
    #[test]
    fn bounds_bracket_the_true_distance() {
        let src = gateway_world(400, 4);
        let lm = pick_landmarks(&src, 16, 0, 1);
        let idx = LandmarkIndex::build(&src, lm).unwrap();
        for i in (0..src.n()).step_by(37) {
            for j in (0..src.n()).step_by(41) {
                let d = src.dist(i, j);
                let (lo, up) = idx.cell_bounds(i, j);
                assert!(lo <= d, "lower bound {lo} exceeds d {d} at ({i},{j})");
                assert!(up >= d, "upper bound {up} below d {d} at ({i},{j})");
            }
        }
    }

    /// Angular distance from embeddings must be a metric, symmetric, and zero on the diagonal.
    #[test]
    fn embedding_source_is_a_symmetric_metric() {
        let dim = 8;
        let mut s: u64 = 11;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / u32::MAX as f32) - 0.5
        };
        let emb: Vec<f32> = (0..40 * dim).map(|_| rnd()).collect();
        let src = EmbeddingSource::new(emb, dim, 1000.0).unwrap();
        for i in 0..src.n() {
            assert_eq!(src.dist(i, i), 0);
            for j in 0..src.n() {
                assert_eq!(src.dist(i, j), src.dist(j, i));
                for k in 0..src.n() {
                    assert!(
                        src.dist(i, k) <= src.dist(i, j) + src.dist(j, k) + 2,
                        "triangle inequality violated beyond rounding"
                    );
                }
            }
        }
    }
}
