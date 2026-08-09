#!/usr/bin/env python3
"""What a graph walk over an on-disk embedding index actually costs in fetches.

`hopcount.py` established that the kNN graph has short paths — max 6 hops at k=32, 100 %
reachable. That is a statement about the graph, not about the disk. This is the disk half.

The framing changed once, and it matters. The obvious objection to a large index is that the
keys have to be resident: at N = 10M and 384 dimensions that is 15 GB, which defeats the
purpose on a 36 GB machine. But the index does not have to be resident, and then the binding
cost is not bytes held but **fetch rounds issued** — which is the quantity the hop count already
measures.

An earlier version of this docstring attributed the on-disk index to MPEdb. That was wrong:
MPEdb has no vector or embedding index, in code or design. Putting the index on disk is a design
proposal, and the measurements below neither depend on it nor test it.

So the comparison is:

  full scan     read every key, decide in one pass    -> N x (dim x 4 + k x 4) bytes, sequential
  graph walk    fetch only the nodes the walk visits  -> visited x one node record, scattered

with the node record laid out DiskANN-style: the vector and its neighbour ids adjacent, so
arriving at a node yields both what it is and where to go next in a single read.

Everything here is measured on the real graph, including the layout effect. The one number that
is a projection rather than a measurement is the hop growth beyond N = 3893, and it is labelled.

Run with a torch venv, e.g. /Users/punnerud/Downloads/ainmt/venv/bin/python3.
"""
import collections
import json
import struct
import sys
from pathlib import Path

import torch


def load(path, dim):
    raw = open(path, "rb").read()
    n = len(raw) // 4 // dim
    x = torch.tensor(struct.unpack(f"<{len(raw) // 4}f", raw)).view(n, dim)
    return torch.nn.functional.normalize(x, dim=-1)


def knn(x, k):
    """Symmetric kNN adjacency, as a sorted id list per node."""
    sim = x @ x.t()
    sim.fill_diagonal_(-2)
    nbr = sim.topk(k, dim=-1).indices
    adj = [set() for _ in range(x.shape[0])]
    for i in range(x.shape[0]):
        for j in nbr[i].tolist():
            adj[i].add(j)
            adj[j].add(i)
    return [sorted(a) for a in adj], sim


def beam_search(adj, sim, src, target, beam=1, budget=64):
    """Best-first walk from `src` towards `target`, counting node records fetched.

    Every neighbour whose distance the walk evaluates must have its vector read, so it counts
    as a fetch. Caching within one query is modelled — a node already read is not read again —
    because that is free and any real implementation does it.

    `beam` is why this is not a one-liner. Pure greedy (beam=1) stops at the first node with no
    better neighbour, and in 384 dimensions that happens constantly: it reached the target in
    8 % of queries at k=8. A local minimum is a property of the descent, not of the graph — the
    hop counts say a path exists — so reporting beam=1 alone would have blamed the index for
    the searcher's failure. Keeping the best `beam` frontier nodes fixes it at a known cost in
    fetches, which is exactly the trade this file exists to price.
    """
    frontier = [src]
    seen, fetched = {src}, 1
    for _ in range(budget):
        cands = sorted({v for u in frontier for v in adj[u] if v not in seen})
        if not cands:
            return fetched, False
        fetched += len(cands)
        seen.update(cands)
        if target in cands:
            return fetched, True
        scored = sorted(cands, key=lambda v: -sim[target, v].item())
        nxt = scored[:beam]
        best_new = sim[target, nxt[0]].item()
        best_old = max(sim[target, u].item() for u in frontier)
        if best_new <= best_old and beam == 1:
            return fetched, False          # local minimum, greedy is stuck
        frontier = nxt
    return fetched, False


def bfs_order(adj, n):
    """A locality-improving disk order: breadth-first from the highest-degree node.

    Not the greedy affinity chain used for experts — that solver is in Rust and works on a
    dense co-activation matrix. BFS is the cheap graph-native equivalent and is enough to show
    whether neighbour sets coalesce at all.
    """
    start = max(range(n), key=lambda i: len(adj[i]))
    order, seen, q = [], {start}, collections.deque([start])
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    order += [i for i in range(n) if i not in seen]    # any disconnected remainder
    pos = [0] * n
    for rank, node in enumerate(order):
        pos[node] = rank
    return pos


def runs(ids, pos):
    """Contiguous runs the ids form once mapped through a disk order.

    One run is one pread. This is the same quantity the expert layout work minimises, applied
    to index nodes instead of expert slabs.
    """
    p = sorted(pos[i] for i in ids)
    return 1 + sum(1 for a, b in zip(p, p[1:]) if b != a + 1)


def main(path="data/embeddings.f32", dim=384, cost="data/costmodel.json",
         out="data/traversal-cost.json", queries=200):
    x = load(path, dim)
    n = x.shape[0]
    cm = json.load(open(cost))
    c_fetch, c_byte = cm["c_fetch_ns"], cm["c_byte_ns"]
    prov = cm.get("provenance", "assumed")
    if prov != "measured":
        print(f"WARNING: cost model provenance is '{prov}' — times below are not grounded",
              file=sys.stderr)

    print(f"{n} embeddings, {dim}-dim, index on disk")
    print(f"cost model: C_fetch = {c_fetch / 1000:.2f} us, C_byte = {c_byte:.4f} ns/B "
          f"({prov})\n")

    ms = lambda fetches, byts: (fetches * c_fetch + byts * c_byte) / 1e6

    rows = []
    for k in (8, 16, 32):
        adj, sim = knn(x, k)
        deg = sum(len(a) for a in adj) / n
        rec = dim * 4 + int(deg) * 4                    # vector + neighbour ids, adjacent
        scan_bytes = n * rec

        # --- how many node records does a walk actually read? ---
        g = torch.Generator().manual_seed(7)
        srcs = torch.randint(0, n, (queries,), generator=g).tolist()
        tgts = torch.randint(0, n, (queries,), generator=g).tolist()
        by_beam = {}
        for beam in (1, 4):
            got, hits = [], 0
            for s, t in zip(srcs, tgts):
                nf, ok = beam_search(adj, sim, s, t, beam=beam)
                got.append(nf)
                hits += ok
            by_beam[beam] = (float(torch.tensor(got, dtype=torch.float).mean()),
                             100.0 * hits / queries)
        (greedy_visits, greedy_reach) = by_beam[1]
        (hops_visited, reach_pct) = by_beam[4]

        # --- does disk order coalesce the reads? ---
        ident = list(range(n))
        pos = bfs_order(adj, n)
        r_id = sum(runs(adj[i], ident) for i in range(n)) / n
        r_bfs = sum(runs(adj[i], pos) for i in range(n)) / n

        # A walk's fetches are neighbour sets; if the set is contiguous on disk it costs
        # `runs` preads instead of |set|, transferring the same bytes either way. Both orders
        # are reported because the ordering does not necessarily win — and here it does not.
        best_runs = min(r_id, r_bfs)
        laid_out = hops_visited * (best_runs / deg)

        t_scan = ms(1, scan_bytes)
        t_walk = ms(hops_visited, hops_visited * rec)
        t_laid = ms(laid_out, hops_visited * rec)

        # Where does the walk start winning? One fetch costs C_fetch/C_byte bytes of
        # sequential transfer, so a walk of `laid_out` fetches is worth it once the index
        # exceeds that many fetch-equivalents of bytes.
        breakeven_n = (laid_out * c_fetch / c_byte) / rec

        rows.append({
            "k": k, "degree": round(deg, 1), "record_bytes": rec,
            "scan_mib": round(scan_bytes / 2**20, 1),
            "visited_mean": round(hops_visited, 1),
            "reached_pct": round(reach_pct, 1),
            "greedy_visited": round(greedy_visits, 1),
            "greedy_reached_pct": round(greedy_reach, 1),
            "runs_identity": round(r_id, 1), "runs_bfs": round(r_bfs, 1),
            "breakeven_n": int(breakeven_n),
            "fetches_laid_out": round(laid_out, 1),
            "ms_scan": round(t_scan, 3), "ms_walk": round(t_walk, 3),
            "ms_laid_out": round(t_laid, 3),
            "speedup_walk": round(t_scan / t_walk, 2),
            "speedup_laid_out": round(t_scan / t_laid, 2),
        })
        print(f"k={k:<3} record {rec} B   full scan {scan_bytes / 2**20:6.1f} MiB "
              f"{t_scan:7.2f} ms")
        print(f"       greedy      visits {greedy_visits:5.1f} nodes, reaches "
              f"{greedy_reach:5.1f}%")
        print(f"       beam=4      visits {hops_visited:5.1f} nodes, reaches {reach_pct:5.1f}%"
              f"  {t_walk:7.2f} ms  {t_scan / t_walk:5.2f}x")
        print(f"       neighbour set of {deg:.0f} = {r_id:.1f} runs in id order, {r_bfs:.1f} "
              f"after BFS  -> {t_laid:6.2f} ms  {t_scan / t_laid:5.2f}x")
        print(f"       walk overtakes the scan at N > {breakeven_n:,.0f}\n")

    Path(out).write_text(json.dumps(
        {"n": n, "dim": dim, "provenance": prov, "by_k": rows}, indent=2))
    print(f"wrote {out}")

    # --- projection, explicitly not a measurement ---
    k = 32
    row = next(r for r in rows if r["k"] == k)
    print(f"\nPROJECTION (not measured): a random-ish graph of degree {row['degree']:.0f} has "
          f"diameter ~log N / log degree,")
    print("so visited count grows logarithmically while the full scan grows linearly.")
    import math
    print(f"\n{'N':>12} {'scan MiB':>10} {'scan ms':>10} {'visited':>9} {'walk ms':>9} "
          f"{'ratio':>8}")
    print(f"It also assumes the walk keeps its {row['reached_pct']:.1f} % success rate, which "
          f"is not established at larger N.")
    base_hops = math.log(n) / math.log(row["degree"])
    for big in (n, 100_000, 1_000_000, 10_000_000):
        sb = big * row["record_bytes"]
        vis = row["visited_mean"] * (math.log(big) / math.log(row["degree"])) / base_hops
        print(f"{big:>12,} {sb / 2**20:>10.1f} {ms(1, sb):>10.1f} {vis:>9.0f} "
              f"{ms(vis, vis * row['record_bytes']):>9.2f} "
              f"{ms(1, sb) / ms(vis, vis * row['record_bytes']):>7.0f}x")


if __name__ == "__main__":
    main(*sys.argv[1:])
