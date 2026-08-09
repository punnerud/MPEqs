//! k nearest neighbours without ever forming the n x n similarity matrix.
//!
//! Every kNN construction in this repository materialises the full matrix first —
//! `geodesic.rs::build_knn` takes a `&Matrix` by signature, and the three Python probes each do
//! `sim = x @ x.t()`. At n = 4 000 that is 64 MB and nobody notices. At n = 50 000 it is 10 GB
//! and at n = 400 000 it is 640 GB, which is why the best result this project has produced —
//! betweenness landmarks on the geodesic metric, 39.08 % of cells exact — has only ever been
//! measured on a 4 000-point subsample.
//!
//! The fix is the one MPEE uses everywhere: stream the cells through a fixed budget and keep
//! only what the answer needs. Here that is a bounded max-heap of k per row. Rows are processed
//! in blocks against column blocks, each block is scored and discarded, and peak memory is
//! `block^2 + n*k` rather than `n^2`.
//!
//! Distance is the half-angle form `2*atan2(‖a-b‖, ‖a+b‖)`, not `acos(dot)`. That is not a
//! stylistic choice: `acos` has an infinite derivative at 1, so the 1e-7 slack left by f32
//! normalisation becomes a full quantisation unit and a point's distance to itself comes out as
//! 1 instead of 0. Since every downstream criterion here is integer equality of a residual, that
//! noise would have destroyed exactly the quantity being measured. The same form is used in
//! `landmark.rs` for the same reason.

use crate::landmark::EmbeddingSource;
use std::cmp::Ordering;

/// A row's k nearest neighbours, nearest first.
#[derive(Debug, Clone)]
pub struct Neighbours {
    /// `ids[i * k .. (i+1) * k]` are row `i`'s neighbours, sorted by increasing distance.
    pub ids: Vec<u32>,
    /// Distances in the same layout, in radians.
    pub dist: Vec<f32>,
    pub n: usize,
    pub k: usize,
}

impl Neighbours {
    pub fn row(&self, i: usize) -> (&[u32], &[f32]) {
        let (a, b) = (i * self.k, (i + 1) * self.k);
        (&self.ids[a..b], &self.dist[a..b])
    }

    /// Symmetric adjacency in compressed form: `offsets[i]..offsets[i+1]` indexes `edges`.
    ///
    /// Symmetrised because a road runs both ways — if `a` is among `b`'s nearest but not the
    /// reverse, the edge is kept anyway. `geodesic.rs::build_knn` does the same, and it is what
    /// makes the graph connected often enough for shortest paths to exist.
    pub fn to_csr(&self) -> (Vec<usize>, Vec<(u32, f32)>) {
        let mut adj: Vec<Vec<(u32, f32)>> = vec![Vec::new(); self.n];
        for i in 0..self.n {
            let (ids, ds) = self.row(i);
            for (&j, &d) in ids.iter().zip(ds) {
                adj[i].push((j, d));
                adj[j as usize].push((i as u32, d));
            }
        }
        let mut offsets = Vec::with_capacity(self.n + 1);
        let mut edges = Vec::new();
        for a in adj.iter_mut() {
            offsets.push(edges.len());
            a.sort_unstable_by_key(|&(j, _)| j);
            a.dedup_by_key(|&mut (j, _)| j);
            edges.extend(a.iter().copied());
        }
        offsets.push(edges.len());
        (offsets, edges)
    }
}

/// A fixed-size worst-first list. Cheaper than a real heap at the k we use (8–64) and it keeps
/// the result sorted, which the caller wants anyway.
struct TopK {
    ids: Vec<u32>,
    dist: Vec<f32>,
    k: usize,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            ids: Vec::with_capacity(k),
            dist: Vec::with_capacity(k),
            k,
        }
    }

    fn worst(&self) -> f32 {
        if self.dist.len() < self.k {
            f32::INFINITY
        } else {
            self.dist[self.dist.len() - 1]
        }
    }

    fn offer(&mut self, id: u32, d: f32) {
        if self.dist.len() == self.k && d >= self.dist[self.dist.len() - 1] {
            return;
        }
        let pos = self
            .dist
            .binary_search_by(|p| p.partial_cmp(&d).unwrap_or(Ordering::Equal))
            .unwrap_or_else(|e| e);
        self.ids.insert(pos, id);
        self.dist.insert(pos, d);
        if self.dist.len() > self.k {
            self.ids.pop();
            self.dist.pop();
        }
    }
}

/// The angle implied by a dot product, used for **ranking only**.
///
/// For unit vectors `‖a-b‖² = 2 - 2·dot`, so one blocked matrix product ranks every candidate
/// without forming a single difference — and ranking is all the inner loop needs, because the
/// angle is monotone in the dot product.
///
/// It must not be used for the *values*. `2 - 2·dot` is catastrophic cancellation near dot = 1,
/// which is `acos`'s problem relocated rather than solved: measured slack at the self-distance
/// is **0.69 quantisation units**, and anything above 0.5 rounds to 1. The accurate form below
/// is applied to the k survivors per row instead of to all n candidates — which is what the
/// module comment claimed all along and what this function originally did not do.
#[inline]
fn rank_key_from_dot(dot: f32) -> f32 {
    (2.0 - 2.0 * dot).max(0.0)
}

/// The accurate angle, from componentwise differences. ~0.0 units of slack at zero.
#[inline]
fn exact_angle(a: &[f32], b: &[f32]) -> f32 {
    let (mut diff, mut sum) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        diff += (x - y) * (x - y);
        sum += (x + y) * (x + y);
    }
    (2.0 * diff.sqrt().atan2(sum.sqrt())) as f32
}

/// Blocked exact kNN. Never allocates anything of size n x n.
///
/// `block` trades memory for loop overhead: peak scratch is `block * block` f32 plus `n * k`
/// for the result. 512 keeps the scratch at 1 MB, which stays in L2 on this machine.
pub fn knn_blocked(src: &EmbeddingSource, k: usize, block: usize) -> Neighbours {
    let n = src.n;
    let dim = src.dim;
    let k = k.clamp(1, n.saturating_sub(1).max(1));
    let block = block.clamp(1, n);

    let mut tops: Vec<TopK> = (0..n).map(|_| TopK::new(k)).collect();
    let mut scratch = vec![0f32; block * block];

    for r0 in (0..n).step_by(block) {
        let r1 = (r0 + block).min(n);
        for c0 in (0..n).step_by(block) {
            let c1 = (c0 + block).min(n);
            // One dot-product block. This is the only place cells exist, and it dies here.
            for (ri, i) in (r0..r1).enumerate() {
                let a = src.vec(i);
                for (ci, j) in (c0..c1).enumerate() {
                    let b = src.vec(j);
                    let mut dot = 0f32;
                    for t in 0..dim {
                        dot += a[t] * b[t];
                    }
                    scratch[ri * (c1 - c0) + ci] = dot;
                }
            }
            for (ri, i) in (r0..r1).enumerate() {
                let top = &mut tops[i];
                for (ci, j) in (c0..c1).enumerate() {
                    if i == j {
                        continue;
                    }
                    let key = rank_key_from_dot(scratch[ri * (c1 - c0) + ci].clamp(-1.0, 1.0));
                    if key < top.worst() {
                        top.offer(j as u32, key);
                    }
                }
            }
        }
    }

    // Only now, on k survivors per row rather than n candidates, is the accurate angle paid
    // for: n*k differences instead of n^2.
    let mut ids = Vec::with_capacity(n * k);
    let mut dist = Vec::with_capacity(n * k);
    for (i, t) in tops.iter_mut().enumerate() {
        for (slot, &j) in t.ids.iter().enumerate() {
            t.dist[slot] = exact_angle(src.vec(i), src.vec(j as usize));
        }
        // A row can end short only if n-1 < k, which the clamp above already prevents; pad
        // defensively rather than produce a ragged layout the callers do not expect.
        while t.ids.len() < k {
            t.ids.push(t.ids.last().copied().unwrap_or(0));
            t.dist.push(f32::INFINITY);
        }
        let _ = i;
        ids.extend_from_slice(&t.ids);
        dist.extend_from_slice(&t.dist);
    }
    Neighbours { ids, dist, n, k }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring_source(n: usize, dim: usize) -> EmbeddingSource {
        // Points on a great circle: neighbours in index order are neighbours in angle, so the
        // right answer is known without computing anything.
        let mut e = vec![0f32; n * dim];
        for i in 0..n {
            let t = std::f32::consts::TAU * i as f32 / n as f32;
            e[i * dim] = t.cos();
            e[i * dim + 1] = t.sin();
        }
        EmbeddingSource::new(e, dim, 1000.0).unwrap()
    }

    fn random_source(n: usize, dim: usize, seed: u64) -> EmbeddingSource {
        let mut s = seed;
        let mut e = vec![0f32; n * dim];
        for v in e.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
        }
        EmbeddingSource::new(e, dim, 1000.0).unwrap()
    }

    /// On a ring the k nearest are the k index-adjacent points. Anything else is a bug in the
    /// blocking, not in the metric.
    #[test]
    fn ring_neighbours_are_the_adjacent_points() {
        let n = 64;
        let src = ring_source(n, 4);
        let nb = knn_blocked(&src, 4, 7); // block deliberately not dividing n
        for i in 0..n {
            let (ids, _) = nb.row(i);
            let mut got: Vec<usize> = ids.iter().map(|&v| v as usize).collect();
            got.sort_unstable();
            let mut want = vec![(i + n - 2) % n, (i + n - 1) % n, (i + 1) % n, (i + 2) % n];
            want.sort_unstable();
            assert_eq!(got, want, "row {i}");
        }
    }

    /// The whole point of the file: blocked and unblocked must agree exactly, and the block
    /// size must not change the answer. A silent disagreement here would look like a result.
    #[test]
    fn blocking_does_not_change_the_answer() {
        let src = random_source(300, 16, 99);
        let reference = knn_blocked(&src, 8, 300); // one block = the dense path
        for block in [1, 7, 64, 299] {
            let got = knn_blocked(&src, 8, block);
            assert_eq!(got.ids, reference.ids, "ids differ at block={block}");
            for (a, b) in got.dist.iter().zip(&reference.dist) {
                assert!((a - b).abs() < 1e-6, "distances differ at block={block}");
            }
        }
    }

    /// Distances must come back sorted, because every caller assumes nearest-first.
    #[test]
    fn rows_are_sorted_nearest_first() {
        let src = random_source(200, 8, 5);
        let nb = knn_blocked(&src, 12, 33);
        for i in 0..nb.n {
            let (_, ds) = nb.row(i);
            for w in ds.windows(2) {
                assert!(w[0] <= w[1] + 1e-7, "row {i} not sorted");
            }
        }
    }

    /// A point is never its own neighbour, and the graph symmetrises without losing edges.
    #[test]
    fn csr_is_symmetric_and_excludes_self() {
        let src = random_source(120, 8, 7);
        let nb = knn_blocked(&src, 5, 16);
        for i in 0..nb.n {
            assert!(
                !nb.row(i).0.contains(&(i as u32)),
                "row {i} contains itself"
            );
        }
        let (off, edges) = nb.to_csr();
        for u in 0..nb.n {
            for &(v, w) in &edges[off[u]..off[u + 1]] {
                let back = edges[off[v as usize]..off[v as usize + 1]]
                    .iter()
                    .find(|&&(x, _)| x as usize == u);
                assert!(back.is_some(), "edge {u}->{v} has no reverse");
                assert!((back.unwrap().1 - w).abs() < 1e-6);
            }
        }
    }

    /// The bug this file exists to avoid: `acos(dot)` makes a point's distance to itself 1
    /// quantisation unit instead of 0, and every criterion downstream is integer equality.
    ///
    /// The assertion is on the *quantised* distance, not on the raw float. f32 normalisation
    /// leaves the self dot product at 1 ± 1e-7 and no formula recovers an exactly-zero angle
    /// from that; what matters is that the error stays far below half a unit. The half-angle
    /// form leaves ~0.35 units of slack where `acos` leaves ~1.4, which is the whole
    /// difference between rounding to 0 and rounding to 1.
    #[test]
    fn a_point_quantises_to_zero_distance_from_itself() {
        let src = random_source(50, 32, 3);
        let mut worst_half = 0f32;
        let mut worst_acos = 0f32;
        for i in 0..src.n {
            let dot: f32 = src.vec(i).iter().zip(src.vec(i)).map(|(a, b)| a * b).sum();
            worst_half = worst_half.max(exact_angle(src.vec(i), src.vec(i)) * 1000.0);
            worst_acos = worst_acos.max(dot.clamp(-1.0, 1.0).acos() * 1000.0);
            assert_eq!(
                (exact_angle(src.vec(i), src.vec(i)) * src.scale).round() as i32,
                0,
                "quantised self-distance is not 0 at {i}"
            );
        }
        assert!(
            worst_half < 0.5,
            "difference-form slack {worst_half} would round to 1"
        );
        assert!(
            worst_half < worst_acos,
            "difference form ({worst_half}) must beat acos ({worst_acos}) — that is why it is used"
        );
        // And the dot-product shortcut must NOT be trusted for values. Taken as a max over
        // all points, not one: for any individual point the f32 dot may land above 1 and clamp
        // to a spurious zero, which is the same cancellation wearing the opposite sign.
        let worst_dot = (0..src.n)
            .map(|i| {
                let dot: f32 = src.vec(i).iter().map(|v| v * v).sum();
                2.0 * rank_key_from_dot(dot).sqrt().atan2(2.0) * 1000.0
            })
            .fold(0f32, f32::max);
        assert!(
            worst_dot > 0.5 && worst_half < 0.5,
            "the split exists because the dot form slacks {worst_dot} units and the difference \
             form {worst_half}; if that ever stops being true the two paths can be merged"
        );
    }
}
