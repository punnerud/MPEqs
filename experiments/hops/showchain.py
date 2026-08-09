#!/usr/bin/env python3
"""Print the actual six-hop chains, step by step, with the text at every node.

The aggregate said step-to-step correlation stays 0.60-0.90 while end-to-end goes negative, and
that chains rescue 33-39 items but break more than they rescue. Both are averages over millions
of pairs and neither shows what a chain *is*. This prints them: the text at each node, the
relative strength of each link, and the running product.

Two are chosen deliberately and the contrast is the point:

  MATCHING     endpoints in domains that genuinely relate — arithmetic and physics formulas,
               which the earlier geometry put 0.168 closer together than either is to prose
  NOT MATCHING endpoints with no relation, reached anyway because every individual step was
               strong enough to pass the relative floor

The second is the failure mode the aggregate could only count. Seeing it is the difference
between "chains break more than they rescue" and knowing why.

Path reconstruction is best-first over (max, x): the frontier is ordered by chain strength, and
a node is settled when no stronger chain to it can remain. That is Dijkstra with multiplication
in place of addition and a maximum in place of a minimum, which is valid because damping and
similarity are both in [0, 1] so a chain can only ever get weaker.

Run with a torch venv, e.g. /Users/punnerud/Downloads/ainmt/venv/bin/python3.
"""
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


def relative_graph(x, knn_k, rel_floor):
    """Same construction as chains.py: kNN, then each node's links relative to its own best."""
    sim = (x @ x.t()).clamp(-1.0, 1.0)
    sim = 1.0 - torch.arccos(sim) / torch.pi
    sim.fill_diagonal_(0.0)
    keep = sim.topk(knn_k, dim=-1).indices
    sparse = torch.zeros_like(sim)
    sparse.scatter_(1, keep, sim.gather(1, keep))
    best = sparse.max(dim=1, keepdim=True).values.clamp(min=1e-9)
    rel = sparse / best
    rel[rel < rel_floor] = 0.0
    rel = torch.maximum(rel, rel.t())
    rel.fill_diagonal_(0.0)
    return rel


def best_chains(rel, src, hops, damping):
    """Strongest chain from `src` to every node within `hops`, with the path kept."""
    n = rel.shape[0]
    best = [0.0] * n
    path = [None] * n
    best[src] = 1.0
    path[src] = [src]
    # (-strength, hops used, node) — negated because heapq is a min-heap.
    pq = [(-1.0, 0, src)]
    while pq:
        neg, used, u = heapq.heappop(pq)
        s = -neg
        if s < best[u] or used >= hops:
            continue
        row = rel[u]
        for v in torch.nonzero(row, as_tuple=False).flatten().tolist():
            cand = s * float(row[v]) * damping
            if cand > best[v]:
                best[v] = cand
                path[v] = path[u] + [v]
                heapq.heappush(pq, (-cand, used + 1, v))
    return best, path


def show(path, rel, damping, text, lab, title):
    print(f"\n{title}")
    print("-" * 78)
    running = 1.0
    for step, (a, b) in enumerate(zip(path, path[1:]), start=1):
        link = float(rel[a][b])
        running *= link * damping
        print(f"  hop {step}  link {link:.3f}  running {running:.4f}   [{lab[b]:>10}] "
              f"{text[b][:52]}")
    print(f"  start [{lab[path[0]]:>10}] {text[path[0]][:52]}")
    print(f"  end   [{lab[path[-1]]:>10}] {text[path[-1]][:52]}")
    print(f"  {len(path) - 1} hops, chain strength {running:.4f}, "
          f"domains crossed: {len({lab[p] for p in path})}")
    return running


def main(emb="data/custom/domains.f32", dim=384, text="data/custom/domains.text",
         labels="data/custom/domains.labels", knn_k=8, hops=6, damping=0.9, rel_floor=0.9,
         out="data/custom/showchain.json"):
    dim, knn_k, hops = int(dim), int(knn_k), int(hops)
    damping, rel_floor = float(damping), float(rel_floor)
    x = load(emb, dim)
    txt = [l.rstrip("\n") for l in open(text)]
    lab = [l.strip() for l in open(labels)]
    n = x.shape[0]
    assert len(txt) == n == len(lab), f"{len(txt)} text, {n} rows, {len(lab)} labels"
    rel = relative_graph(x, knn_k, rel_floor)
    print(f"{n} points, k={knn_k}, relative floor {rel_floor}, damping {damping}, "
          f"up to {hops} hops")

    # Search from a few arithmetic sources for the two cases wanted: a long chain ending in
    # physics (related by construction — both are formulas over numbers) and one ending in
    # prose or Norwegian (unrelated, reached only because every step passed the floor).
    found = {}
    for src in range(0, 800, 7):
        best, path = best_chains(rel, src, hops, damping)
        for j in range(n):
            p = path[j]
            if p is None or len(p) - 1 < hops:
                continue
            kind = "matching" if lab[j] == "physics" else (
                "not matching" if lab[j] in ("prose", "norwegian") else None)
            if kind and (kind not in found or best[j] > found[kind][0]):
                found[kind] = (best[j], p)
        if len(found) == 2 and src > 200:
            break

    rows = {}
    for kind in ("matching", "not matching"):
        if kind not in found:
            print(f"\nno {hops}-hop chain of the '{kind}' kind exists at this floor")
            continue
        strength, p = found[kind]
        title = (f"{hops}-HOP CHAIN, {kind.upper()}: "
                 f"{lab[p[0]]} -> {lab[p[-1]]}")
        show(p, rel, damping, txt, lab, title)
        rows[kind] = {
            "strength": round(strength, 5),
            "hops": len(p) - 1,
            "nodes": p,
            "labels": [lab[i] for i in p],
            "links": [round(float(rel[a][b]), 4) for a, b in zip(p, p[1:])],
            "text": [txt[i] for i in p],
        }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
