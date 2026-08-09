#!/usr/bin/env python3
"""Are there waypoints — points that many routes must pass through?

Everything measured so far scored matcodec's *compression* criterion: can a block be reproduced
exactly from the landmark index. That is one consequence of gateway structure, not the structure
itself. The structure itself is about routing: if a few points lie on most paths between
regions, they are gateways, and MPEE's whole approach is calculation between embeddings to find
those hops.

Betweenness centrality measures exactly that. Build the kNN graph, take shortest paths between
sampled pairs, and count how often each node appears as an interior point. A road network has a
handful of bridges with enormous betweenness; an expander spreads it evenly and has none.

The diagnostic is the *shape* of the distribution, not its mean, which is fixed by the number of
paths. Reported as the share of all traversals carried by the top 1 % of nodes, against the 1 %
a perfectly flat graph would give. That ratio is the gateway concentration, and it is the number
this project should have measured first.

Scored on the 39-class labelled embeddings, where clustering already recovers classes at 8.21x,
so any absence of waypoints cannot be blamed on the data having no structure at all.

Run with a torch venv, e.g. /Users/punnerud/Downloads/ainmt/venv/bin/python3.
"""
import collections
import heapq
import json
import struct
import sys
from pathlib import Path

import torch


def load(path, dim, limit=0):
    raw = Path(path).read_bytes()
    n = len(raw) // 4 // dim
    if limit:
        n = min(n, limit)
    x = torch.tensor(struct.unpack(f"<{n * dim}f", raw[: n * dim * 4])).view(n, dim)
    return torch.nn.functional.normalize(x, dim=-1)


def knn_graph(x, k):
    sim = x @ x.t()
    sim.fill_diagonal_(-2)
    nbr = sim.topk(k, dim=-1)
    adj = [dict() for _ in range(x.shape[0])]
    for i in range(x.shape[0]):
        for j, s in zip(nbr.indices[i].tolist(), nbr.values[i].tolist()):
            w = float(2.0 * torch.atan2((x[i] - x[j]).norm(), (x[i] + x[j]).norm()))
            adj[i][j] = w
            adj[j][i] = w          # a road runs both ways
    return adj


def dijkstra_paths(adj, src, targets):
    """Shortest paths from src, returning the interior nodes of each route to `targets`."""
    dist = {src: 0.0}
    prev = {}
    pq = [(0.0, src)]
    want = set(targets)
    found = {}
    while pq and len(found) < len(want):
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        if u in want:
            found[u] = d
        for v, w in adj[u].items():
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    routes = []
    for t in found:
        path, cur = [], t
        while cur in prev:
            cur = prev[cur]
            if cur != src:
                path.append(cur)
        routes.append(path)
    return routes


def main(emb="data/labelled/emb.f32", dim=384, k=16, sources=150, targets=60,
         limit=4000, out="data/custom/waypoints.json", n_lm=32):
    dim, k, n_lm = int(dim), int(k), int(n_lm)
    sources, targets, limit = int(sources), int(targets), int(limit)
    x = load(emb, dim, limit)
    n = x.shape[0]
    print(f"{n} points, {dim}-dim, kNN graph at k={k}")
    adj = knn_graph(x, k)

    g = torch.Generator().manual_seed(0)
    srcs = torch.randperm(n, generator=g)[:sources].tolist()
    tgts = torch.randperm(n, generator=g)[:targets].tolist()

    between = collections.Counter()
    hops, n_routes = [], 0
    for s in srcs:
        for path in dijkstra_paths(adj, s, tgts):
            n_routes += 1
            hops.append(len(path) + 1)
            between.update(path)

    counts = torch.zeros(n)
    for node, c in between.items():
        counts[node] = c
    total = counts.sum()
    order = counts.argsort(descending=True)
    top1 = max(1, n // 100)
    share = float(counts[order[:top1]].sum() / total) if total > 0 else 0.0
    flat = top1 / n
    hops_t = torch.tensor(hops, dtype=torch.float)

    rep = {
        "n": n, "k": k, "routes": n_routes,
        "mean_hops": round(float(hops_t.mean()), 2),
        "max_hops": int(hops_t.max()),
        "top1pct_traversal_share": round(share, 4),
        "flat_graph_share": round(flat, 4),
        "gateway_concentration": round(share / flat, 2) if flat else 0.0,
        "nodes_carrying_half": int((counts[order].cumsum(0) < total / 2).sum()) + 1,
        "nodes_touched_pct": round(100.0 * float((counts > 0).sum()) / n, 1),
    }
    print(f"  routes {rep['routes']}, mean {rep['mean_hops']} hops, max {rep['max_hops']}")
    print(f"  top 1 % of nodes carry {100 * share:.1f}% of traversals "
          f"(flat graph: {100 * flat:.1f}%)")
    print(f"  gateway concentration {rep['gateway_concentration']}x")
    print(f"  half of all traversals go through {rep['nodes_carrying_half']} of {n} nodes")
    # The correction this measurement forced. Landmarks were being chosen by facility
    # location on the ANGULAR metric, where straight-line distance passes through nothing, so
    # no choice of landmark can make a block exact. Betweenness names the nodes that routes
    # actually use; scoring them against the GEODESIC metric is the pairing never tried, and
    # it is the only one where "gateway" means the same thing on both sides.
    lm = order[:n_lm].tolist()
    d_lm = torch.zeros(n, len(lm))
    for a, node in enumerate(lm):
        dist = {node: 0.0}
        pq = [(0.0, node)]
        while pq:
            dd, u = heapq.heappop(pq)
            if dd > dist.get(u, float("inf")):
                continue
            for v, w in adj[u].items():
                nd = dd + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        far = max(dist.values()) * 1.1
        for i in range(n):
            d_lm[i, a] = dist.get(i, far)

    samp = torch.randperm(n, generator=g)[:400]
    exact = tot = 0
    for i in samp.tolist():
        dist = {i: 0.0}
        pq = [(0.0, i)]
        while pq:
            dd, u = heapq.heappop(pq)
            if dd > dist.get(u, float("inf")):
                continue
            for v, w in adj[u].items():
                nd = dd + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        for j in samp.tolist():
            if j == i or j not in dist:
                continue
            base = float((d_lm[i] + d_lm[j]).min())
            tot += 1
            if abs(base - dist[j]) < 1e-4:
                exact += 1
    rep["geodesic_betweenness_exact_cell_pct"] = round(100.0 * exact / max(tot, 1), 2)
    rep["landmarks_by_betweenness"] = len(lm)
    print(f"\n  betweenness landmarks on the GEODESIC metric: "
          f"{rep['geodesic_betweenness_exact_cell_pct']:.2f}% of cells exact")
    print(f"  (facility-location landmarks on the ANGULAR metric gave 0.91%)")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {out}")
    print("A road network concentrates hard — a few bridges carry most traffic. Concentration")
    print("near 1x means every node is as good a waypoint as any other, which is what an")
    print("expander looks like and why no gateway codec can work on it.")


if __name__ == "__main__":
    main(*sys.argv[1:])
