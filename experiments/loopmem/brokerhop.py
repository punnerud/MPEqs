#!/usr/bin/env python3
"""Pay for the neighbour graph once, or buy the cells you read and keep them.

The hop in phase 32 recovered real near misses and cost 4 x 8,920 dot products to do it, which
is four times a flat scan. That is not an argument against the hop, it is an argument against
computing it the way I did — materialising a full similarity row per candidate and throwing it
away. Both of the fixes are already written down.

`crates/matstruct/src/knn.rs` streams the graph under a memory bound and never forms n x n:
8,920 points, k=16, 13.4 seconds, 2.1 MiB of scratch against 303.5 MiB for a dense matrix, 1.1
MiB on disk. Against that stored graph a hop is 4 x 16 lookups and no arithmetic at all.

`DESIGN-MPEE-OPT.md` in mpedb calls the other half "buy once": pay for a cell exactly once, cache
it, share it. A query workload reuses candidates heavily, so a broker that computes a neighbour
row on first request and keeps it converges on the prebuilt graph without the up-front pass.

Three costings of the same rescue, measured rather than reasoned about:

  ON DEMAND    recompute every candidate's row per query           4 x 8,920 dots
  PREBUILT     stream the graph once, then look up                 one pass, then 64 lookups
  BROKER       compute a row the first time it is asked for, keep it

The question the numbers have to answer is not which is cheapest in the limit — prebuilt
obviously is — but where the crossover sits, because below it the up-front pass is waste.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from beamwide import descend, load  # noqa: E402

KNN_BIN = Path("data/custom/wikitext-knn.bin")


def read_knn(path):
    """The KNN1 block matstruct writes: header, then ids, then distances."""
    raw = path.read_bytes()
    magic, n, k, _ = struct.unpack_from("<4sIII", raw, 0)
    if magic != b"KNN1":
        raise SystemExit(f"not a KNN1 file: {magic!r}")
    ids = np.frombuffer(raw, dtype=np.uint32, count=n * k, offset=16).reshape(n, k)
    dist = np.frombuffer(raw, dtype=np.float32, count=n * k, offset=16 + n * k * 4).reshape(n, k)
    return ids, dist


class Broker:
    """Compute a neighbour row on first request, keep it. Counts what it actually buys."""

    def __init__(self, leaves, k=16):
        self.leaves = leaves
        self.k = k
        self.cache = {}
        self.dots = 0
        self.hits = 0

    def neighbours(self, i):
        if i in self.cache:
            self.hits += 1
            return self.cache[i]
        sims = self.leaves @ self.leaves[i]
        self.dots += len(self.leaves)
        row = np.argpartition(sims, -self.k - 1)[-self.k - 1:]
        row = [int(j) for j in row if int(j) != i][: self.k]
        self.cache[i] = row
        return row


def main(n_queries=40, seed=3, out="data/custom/brokerhop.json"):
    n_queries, seed = int(n_queries), int(seed)
    sents, leaves, levels, picks = load(n_queries, seed)
    if not KNN_BIN.exists():
        raise SystemExit(f"{KNN_BIN} missing — run: matstruct knn -e "
                         f"data/custom/wikitext-sentences.f32 --dim 384 -k 16 "
                         f"-o {KNN_BIN}")
    ids, _ = read_knn(KNN_BIN)
    n, k = ids.shape
    print(f"{n:,} leaves, stored graph {n} x {k}, {KNN_BIN.stat().st_size / 1e6:.1f} MB\n")

    from embednav import embed
    kinds = {"exact": [sents[i] for i in picks],
             "half": [sents[i][:len(sents[i]) // 2] for i in picks]}
    qv = {kk: np.array(embed(v), dtype=np.float32) for kk, v in kinds.items()}

    print(f"{'beam':>5}{'kind':>7}{'in beam':>9}{'+ stored hop':>14}{'lookups':>9}"
          f"{'on-demand dots':>16}")
    rows = []
    broker = Broker(leaves, k=k)
    for beam in (4, 16):
        for kind in kinds:
            inbeam = after = lookups = 0
            for q_i, target in enumerate(picks):
                cand, _ = descend(levels, qv[kind][q_i], beam)
                inbeam += target in cand
                # The stored graph: no arithmetic, just k ids per candidate.
                reach = set(cand)
                for c in cand:
                    reach.update(int(j) for j in ids[c])
                    lookups += k
                after += target in reach
                # The same rescue through the broker, so its cache warms on a real workload.
                for c in cand:
                    broker.neighbours(c)
            rows.append({"beam": beam, "kind": kind, "in_beam": inbeam, "after_hop": after,
                         "lookups": lookups / len(picks),
                         "on_demand_dots": beam * n})
            print(f"{beam:>5}{kind:>7}{inbeam:>7}/{len(picks):<2}{after:>12}/{len(picks):<2}"
                  f"{lookups / len(picks):>9.0f}{beam * n:>16,}")

    # What the broker actually paid, against what recomputing every time would have.
    served = broker.hits + len(broker.cache)
    naive = served * n
    print(f"\nbroker: {served:,} neighbour requests, {len(broker.cache):,} distinct, "
          f"{broker.hits:,} served from cache")
    print(f"        {broker.dots:,} dot products against {naive:,} recomputing every time "
          f"({naive / max(broker.dots, 1):.1f}x)")

    # Where the up-front pass starts paying. The build is the full pairwise pass, streamed.
    build_dots = n * (n - 1) // 2
    per_query_ondemand = 4 * n
    breakeven = build_dots / per_query_ondemand
    print(f"\nbuilding the graph costs {build_dots:,} pair evaluations, streamed in 2.1 MiB")
    print(f"on-demand costs {per_query_ondemand:,} per query at beam 4, so the prebuilt graph")
    print(f"overtakes after {breakeven:,.0f} queries — below that the up-front pass is waste.")
    print("\nThe broker needs no such decision: it pays only for what is asked and converges")
    print("on the prebuilt graph as the workload repeats, which is the point of buying once.")

    summary = {"leaves": n, "k": k, "queries": len(picks), "rows": rows,
               "broker_requests": served, "broker_distinct": len(broker.cache),
               "broker_hits": broker.hits, "broker_dots": broker.dots,
               "naive_dots": naive, "build_pair_evaluations": build_dots,
               "breakeven_queries": breakeven,
               "graph_bytes": KNN_BIN.stat().st_size}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
