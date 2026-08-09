#!/usr/bin/env python3
"""Hop counts in the kNN graph over stored embeddings.

Phase 3 found that graph to be an expander — richly connected, no narrow cuts — which was
fatal for matcodec, since a gateway codec needs bottlenecks. This measures the same property
from the other side: an expander has short paths, so reaching any stored embedding from any
first hop should take only a few steps. If so, the measurement that killed the codec is the
one that supports traversal, and the architecture splits along that line: the weight product
gives the first hop, the graph walk finds the rest.

Run with a torch venv, e.g. /Users/punnerud/Downloads/ainmt/venv/bin/python3.
"""
import collections
import json
import struct
import sys
from pathlib import Path

import torch


def main(path="data/embeddings.f32", dim=384, sources=60,
         out="data/hopcount.json"):
    raw = open(path, "rb").read()
    n = len(raw) // 4 // dim
    x = torch.tensor(struct.unpack(f"<{len(raw) // 4}f", raw)).view(n, dim)
    x = torch.nn.functional.normalize(x, dim=-1)
    print(f"{n} stored embeddings, {dim}-dim\n")
    print(f"{'k':>4} {'mean hops':>11} {'median':>8} {'p99':>6} {'max':>5} {'reachable':>11}")

    sim = x @ x.t()
    sim.fill_diagonal_(-2)
    step = max(1, n // sources)

    rows = []
    for k in (4, 8, 16, 32):
        nbr = sim.topk(k, dim=-1).indices
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in nbr[i].tolist():
                adj[i].add(j)
                adj[j].add(i)          # symmetric: a road runs both ways

        hops, reach, n_src = [], 0, 0
        for src in range(0, n, step):
            n_src += 1
            d = {src: 0}
            q = collections.deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in d:
                        d[v] = d[u] + 1
                        q.append(v)
            hops += [v for key, v in d.items() if key != src]
            reach += len(d)
        h = torch.tensor(hops, dtype=torch.float)
        rows.append({"k": k, "mean_hops": round(float(h.mean()), 2),
                     "median_hops": int(h.median()), "p99_hops": int(h.quantile(0.99)),
                     "max_hops": int(h.max()),
                     "reachable_pct": round(100.0 * reach / (n_src * n), 1)})
        print(f"{k:>4} {h.mean():>11.2f} {h.median():>8.0f} "
              f"{h.quantile(0.99):>6.0f} {h.max():>5.0f} {100.0 * reach / (n_src * n):>10.1f}%")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"n": n, "dim": dim, "by_k": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
