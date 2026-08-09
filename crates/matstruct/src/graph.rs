//! Distances through the kNN graph, and the junctions everything routes through.
//!
//! Two findings from this thread meet here. Landmarks chosen by **betweenness** and scored
//! against the **geodesic** metric reproduce 39.08 % of pairwise distances exactly, against
//! 0.91 % for facility location on the chord — a factor of 43, and the largest single effect in
//! the whole landmark line of work. And the two-sided label model, which is worth 2.3x on real
//! road data, is worth *less* than one relay here: 27.92 % against 39.08 %.
//!
//! Put together those say the lever is not the label structure but which points are chosen, and
//! that is exactly the "motorway junction" idea: MPEE's `ExternalHubs` takes its candidates from
//! the top of a contraction hierarchy, and measures 9.4x against 7.6x for doing so. The
//! knowledge-graph reading is the same shape — Python and Java are both programming languages,
//! and the abstraction is what every path between them passes through.
//!
//! So this module supplies the two pieces neither `landmark.rs` nor `hublabel.rs` has: a
//! `DistanceSource` whose distances are shortest paths through the kNN graph, and a betweenness
//! score to pick the junctions with. Both work off the streaming kNN, so neither needs n².

use crate::landmark::DistanceSource;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// f32 with a total order, so it can go in a heap. Graph weights are never NaN.
#[derive(PartialEq, PartialOrd, Debug, Clone, Copy)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other)
            .expect("graph weights are never NaN")
    }
}

/// Shortest-path distances over a symmetric kNN graph, computed on demand.
///
/// A row costs one Dijkstra, O(E log V), and nothing of size n² is ever formed — which is what
/// lets the geodesic metric run at 50 000 points where `geodesic::all_pairs` would need a
/// second dense matrix.
pub struct GraphSource {
    pub n: usize,
    offsets: Vec<usize>,
    edges: Vec<(u32, f32)>,
    /// Integer units per radian, matching the other sources.
    pub scale: f32,
    /// What an unreachable pair is reported as. A disconnected component is a real property of
    /// the graph, but an infinite cell makes every downstream statistic meaningless, so it is
    /// capped and the cap is visible rather than silent.
    pub unreachable: i32,
}

impl GraphSource {
    pub fn new(offsets: Vec<usize>, edges: Vec<(u32, f32)>, n: usize, scale: f32) -> Self {
        // Cap at 1.1x the widest single edge times a generous diameter guess, the same spirit
        // as `geodesic::all_pairs`. Any pair that hits it is flagged by `reachable_pct`.
        let widest = edges.iter().map(|&(_, w)| w).fold(0.0f32, f32::max);
        let unreachable = ((widest * scale) as i32).saturating_mul(64).max(1);
        Self {
            n,
            offsets,
            edges,
            scale,
            unreachable,
        }
    }

    /// Single-source shortest paths, in graph units (radians).
    pub fn sssp(&self, src: usize, dist: &mut [f32]) {
        dist.iter_mut().for_each(|d| *d = f32::INFINITY);
        dist[src] = 0.0;
        let mut heap = BinaryHeap::new();
        heap.push(Reverse((OrdF32(0.0), src as u32)));
        while let Some(Reverse((OrdF32(d), u))) = heap.pop() {
            let u = u as usize;
            if d > dist[u] {
                continue;
            }
            for &(v, w) in &self.edges[self.offsets[u]..self.offsets[u + 1]] {
                let nd = d + w;
                if nd < dist[v as usize] {
                    dist[v as usize] = nd;
                    heap.push(Reverse((OrdF32(nd), v)));
                }
            }
        }
    }

    /// Shortest paths from `src` to `targets`, returning the interior nodes of each route.
    fn routes(
        &self,
        src: usize,
        targets: &[usize],
        dist: &mut [f32],
        prev: &mut [u32],
    ) -> Vec<Vec<u32>> {
        dist.iter_mut().for_each(|d| *d = f32::INFINITY);
        prev.iter_mut().for_each(|p| *p = u32::MAX);
        dist[src] = 0.0;
        let mut heap = BinaryHeap::new();
        heap.push(Reverse((OrdF32(0.0), src as u32)));
        while let Some(Reverse((OrdF32(d), u))) = heap.pop() {
            let u = u as usize;
            if d > dist[u] {
                continue;
            }
            for &(v, w) in &self.edges[self.offsets[u]..self.offsets[u + 1]] {
                let nd = d + w;
                if nd < dist[v as usize] {
                    dist[v as usize] = nd;
                    prev[v as usize] = u as u32;
                    heap.push(Reverse((OrdF32(nd), v)));
                }
            }
        }
        targets
            .iter()
            .filter(|&&t| t != src && dist[t].is_finite())
            .map(|&t| {
                let mut path = Vec::new();
                let mut cur = t as u32;
                while prev[cur as usize] != u32::MAX {
                    cur = prev[cur as usize];
                    if cur as usize != src {
                        path.push(cur);
                    }
                }
                path
            })
            .collect()
    }

    /// The junctions: nodes that lie on the most shortest paths between sampled pairs.
    ///
    /// Returns `(hub ids by descending betweenness, concentration)`, where concentration is the
    /// share of traversals carried by the top 1 % of nodes divided by the 1 % a flat graph
    /// would give. A road network concentrates hard; an expander sits near 1.
    pub fn betweenness(&self, sources: usize, targets: usize, seed: u64) -> (Vec<usize>, f64) {
        let n = self.n;
        let mut s = seed | 1;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as usize % n
        };
        let srcs: Vec<usize> = (0..sources.min(n)).map(|_| next()).collect();
        let tgts: Vec<usize> = (0..targets.min(n)).map(|_| next()).collect();

        let mut counts = vec![0u64; n];
        let mut dist = vec![0f32; n];
        let mut prev = vec![u32::MAX; n];
        for &src in &srcs {
            for path in self.routes(src, &tgts, &mut dist, &mut prev) {
                for node in path {
                    counts[node as usize] += 1;
                }
            }
        }

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by_key(|&i| Reverse(counts[i]));
        let total: u64 = counts.iter().sum();
        let top1 = (n / 100).max(1);
        let share = if total > 0 {
            order[..top1].iter().map(|&i| counts[i]).sum::<u64>() as f64 / total as f64
        } else {
            0.0
        };
        (order, share / (top1 as f64 / n as f64))
    }

    /// Share of pairs that are reachable at all, from a sample. Below 100 % the graph is
    /// disconnected and every capped cell is a fiction the caller should know about.
    pub fn reachable_pct(&self, samples: usize, seed: u64) -> f64 {
        let mut s = seed | 1;
        let mut dist = vec![0f32; self.n];
        let (mut ok, mut tot) = (0u64, 0u64);
        for _ in 0..samples.min(self.n) {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.sssp((s >> 33) as usize % self.n, &mut dist);
            ok += dist.iter().filter(|d| d.is_finite()).count() as u64;
            tot += self.n as u64;
        }
        100.0 * ok as f64 / tot.max(1) as f64
    }
}

impl DistanceSource for GraphSource {
    fn n(&self) -> usize {
        self.n
    }

    fn dist(&self, i: usize, j: usize) -> i32 {
        let mut d = vec![0f32; self.n];
        self.sssp(i, &mut d);
        if d[j].is_finite() {
            (d[j] * self.scale).round() as i32
        } else {
            self.unreachable
        }
    }

    fn row(&self, i: usize, out: &mut [i32]) {
        let mut d = vec![0f32; self.n];
        self.sssp(i, &mut d);
        for (j, o) in out.iter_mut().enumerate() {
            *o = if d[j].is_finite() {
                (d[j] * self.scale).round() as i32
            } else {
                self.unreachable
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knn::knn_blocked;
    use crate::landmark::EmbeddingSource;

    fn ring(n: usize) -> EmbeddingSource {
        let mut e = vec![0f32; n * 4];
        for i in 0..n {
            let t = std::f32::consts::TAU * i as f32 / n as f32;
            e[i * 4] = t.cos();
            e[i * 4 + 1] = t.sin();
        }
        EmbeddingSource::new(e, 4, 1000.0).unwrap()
    }

    /// A ring: the geodesic is the arc, not the chord. Opposite points are 2 apart as the crow
    /// flies and pi apart along the ring, and that inflation is the whole reason the geodesic
    /// metric behaves differently from the angular one.
    #[test]
    fn the_graph_metric_is_the_arc_not_the_chord() {
        let n = 64;
        let src = ring(n);
        let (off, edges) = knn_blocked(&src, 2, 16).to_csr();
        let g = GraphSource::new(off, edges, n, 1000.0);
        let chord = src.dist(0, n / 2) as f64 / 1000.0;
        let arc = g.dist(0, n / 2) as f64 / 1000.0;
        assert!((chord - std::f64::consts::PI).abs() < 0.01, "chord {chord}");
        assert!(
            arc > chord * 0.99,
            "the geodesic can never be shorter than the direct metric"
        );
        assert!(g.reachable_pct(8, 1) > 99.9, "a ring at k=2 is connected");
    }

    /// Two dense clusters joined by one bridge: the bridge nodes must dominate betweenness.
    /// If a graph with a deliberate bottleneck does not concentrate, the measure is broken.
    #[test]
    fn a_bottleneck_shows_up_as_concentration() {
        let dim = 8;
        let n = 120;
        let mut e = vec![0f32; n * dim];
        for i in 0..n {
            // Two tight clusters at opposite poles, with a few points strung between them.
            let far = if i < n / 2 { 1.0 } else { -1.0 };
            e[i * dim] = far;
            e[i * dim + 1] = (i as f32 * 0.017).sin() * 0.05;
            if i % 40 == 0 {
                // the bridge: pull a few points onto the equator
                e[i * dim] = 0.0;
                e[i * dim + 1] = 1.0;
            }
        }
        let src = EmbeddingSource::new(e, dim, 1000.0).unwrap();
        let (off, edges) = knn_blocked(&src, 4, 32).to_csr();
        let g = GraphSource::new(off, edges, n, 1000.0);
        let (_, conc) = g.betweenness(40, 20, 7);
        assert!(
            conc > 2.0,
            "a bridged graph should concentrate, got {conc}x"
        );
    }
}
