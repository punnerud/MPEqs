//! Turning an embedding cloud into a road network.
//!
//! Raw angular distance between embeddings is the "fly straight there" metric, and in 384
//! dimensions it concentrates: every pair sits at roughly π/2, so there are no far-apart
//! regions and nothing for a gateway to bridge. A road network is not like that — you cannot
//! fly, you must follow roads, and *that* is what creates bottlenecks.
//!
//! So build the roads. Connect each point to its k nearest neighbours and define distance as
//! the shortest path through that graph. Crossing between two topic clusters then has to pass
//! through whichever few points bridge them — which is a gateway, manufactured rather than
//! hoped for. Shortest-path distance also satisfies the triangle inequality by construction,
//! so the metric requirement is met for free.

use crate::matrix::Matrix;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// An undirected k-nearest-neighbour graph in compressed adjacency form.
pub struct KnnGraph {
    pub n: usize,
    /// `offsets[i]..offsets[i+1]` indexes into `edges`.
    pub offsets: Vec<usize>,
    pub edges: Vec<(u32, f32)>,
}

/// Builds a symmetric kNN graph from a dense distance matrix.
///
/// Symmetric because a road is traversable both ways: if `a` is among `b`'s nearest but not
/// the reverse, the edge is kept anyway. That also makes the graph much more likely to be
/// connected, which matters because an unreachable pair has no geodesic at all.
pub fn build_knn(m: &Matrix, k: usize) -> KnnGraph {
    let n = m.n;
    let k = k.clamp(1, n.saturating_sub(1));
    let mut adj: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];

    let mut cand: Vec<(f32, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        cand.clear();
        for j in 0..n {
            if i != j {
                cand.push((m.get(i, j), j as u32));
            }
        }
        cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for &(d, j) in cand.iter().take(k) {
            adj[i].push((j, d));
            adj[j as usize].push((i as u32, d));
        }
    }

    let mut offsets = Vec::with_capacity(n + 1);
    let mut edges = Vec::new();
    for a in adj.iter_mut() {
        offsets.push(edges.len());
        a.sort_by_key(|&(j, _)| j);
        a.dedup_by_key(|&mut (j, _)| j);
        edges.extend(a.iter().copied());
    }
    offsets.push(edges.len());
    KnnGraph { n, offsets, edges }
}

/// All-pairs shortest paths by Dijkstra from every source.
///
/// Unreachable pairs are filled with `1.1 ×` the largest finite distance rather than infinity:
/// a disconnected component is a real property of the graph, but an infinite cell would make
/// every downstream statistic meaningless. The returned flag says whether that happened, so
/// the caller can say so rather than quietly reporting a number.
pub fn all_pairs(g: &KnnGraph) -> (Matrix, bool) {
    let n = g.n;
    let mut m = Matrix::zeros(n);
    let mut dist = vec![f32::INFINITY; n];
    let mut disconnected = false;
    let mut max_finite = 0.0f32;

    for src in 0..n {
        dist.iter_mut().for_each(|d| *d = f32::INFINITY);
        dist[src] = 0.0;
        let mut heap: BinaryHeap<Reverse<(ordered::OrdF32, u32)>> = BinaryHeap::new();
        heap.push(Reverse((ordered::OrdF32(0.0), src as u32)));

        while let Some(Reverse((ordered::OrdF32(d), u))) = heap.pop() {
            let u = u as usize;
            if d > dist[u] {
                continue;
            }
            for &(v, w) in &g.edges[g.offsets[u]..g.offsets[u + 1]] {
                let nd = d + w;
                if nd < dist[v as usize] {
                    dist[v as usize] = nd;
                    heap.push(Reverse((ordered::OrdF32(nd), v)));
                }
            }
        }
        for (j, &d) in dist.iter().enumerate() {
            if d.is_finite() {
                if d > max_finite {
                    max_finite = d;
                }
            } else if j != src {
                disconnected = true;
            }
            m.set(src, j, d);
        }
    }

    if disconnected {
        let cap = max_finite * 1.1;
        for v in m.d.iter_mut() {
            if !v.is_finite() {
                *v = cap;
            }
        }
    }
    (m, disconnected)
}

/// `f32` with a total order, so it can go in a heap. Distances here are never NaN.
mod ordered {
    #[derive(PartialEq, PartialOrd, Debug, Clone, Copy)]
    pub struct OrdF32(pub f32);
    impl Eq for OrdF32 {}
    #[allow(clippy::derive_ord_xor_partial_ord)]
    impl Ord for OrdF32 {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.partial_cmp(other).expect("distances are never NaN")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Points evenly spaced on a circle. Straight-line distance is the chord; travelling only
    /// between near neighbours forces you along the arc. Opposite points are 2 apart as the
    /// crow flies and π apart along the ring, so the inflation is exactly the mechanism this
    /// transform exists for — and it is what a road network does to a map.
    fn ring(n: usize) -> Matrix {
        let mut m = Matrix::zeros(n);
        let pts: Vec<(f32, f32)> = (0..n)
            .map(|i| {
                let t = std::f32::consts::TAU * i as f32 / n as f32;
                (t.cos(), t.sin())
            })
            .collect();
        for i in 0..n {
            for j in 0..n {
                let (dx, dy) = (pts[i].0 - pts[j].0, pts[i].1 - pts[j].1);
                m.set(i, j, (dx * dx + dy * dy).sqrt());
            }
        }
        m
    }

    #[test]
    fn knn_graph_is_symmetric_and_connected_at_a_sane_k() {
        let m = ring(24);
        let g = build_knn(&m, 2);
        for u in 0..g.n {
            for &(v, w) in &g.edges[g.offsets[u]..g.offsets[u + 1]] {
                let back = g.edges[g.offsets[v as usize]..g.offsets[v as usize + 1]]
                    .iter()
                    .find(|&&(x, _)| x as usize == u);
                assert!(back.is_some(), "edge {u}->{v} has no reverse");
                assert!((back.unwrap().1 - w).abs() < 1e-6);
            }
        }
        let (_, disconnected) = all_pairs(&g);
        assert!(!disconnected, "a ring with k=2 is connected");
    }

    #[test]
    fn geodesic_inflates_the_chord_into_the_arc() {
        let n = 24;
        let m = ring(n);
        let g = build_knn(&m, 2);
        let (geo, disconnected) = all_pairs(&g);
        assert!(!disconnected);

        // Opposite points: chord 2.0, arc π. Following near-neighbour hops must find the arc.
        let chord = m.get(0, n / 2);
        let arc = geo.get(0, n / 2);
        assert!(
            (chord - 2.0).abs() < 1e-3,
            "chord should be the diameter, got {chord}"
        );
        assert!(
            arc > 3.0 && arc < 3.2,
            "geodesic should approach pi, got {arc}"
        );
        assert!(
            arc / chord > 1.5,
            "the point is that the geodesic inflates the chord"
        );

        // A geodesic can never be shorter than the direct metric distance.
        for a in 0..n {
            for b in 0..n {
                assert!(geo.get(a, b) >= m.get(a, b) - 1e-4);
            }
        }
    }

    #[test]
    fn geodesic_distance_satisfies_the_triangle_inequality_by_construction() {
        let m = ring(16);
        let g = build_knn(&m, 2);
        let (geo, _) = all_pairs(&g);
        for a in 0..geo.n {
            for b in 0..geo.n {
                for c in 0..geo.n {
                    assert!(
                        geo.get(a, c) <= geo.get(a, b) + geo.get(b, c) + 1e-4,
                        "shortest paths cannot violate the triangle inequality"
                    );
                }
            }
        }
    }
}
