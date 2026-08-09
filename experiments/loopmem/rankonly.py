#!/usr/bin/env python3
"""Throw the embeddings away. Keep the centroids, and a rank inside the group.

Every codec so far kept a vector per item and argued about how many bits it needed. The other
end of that argument is to keep no vector per item at all: store the C cluster centroids, and
describe each item by which cluster it is in and where it sits inside that cluster. The
embeddings become a property of the group, and an item becomes an address.

    cluster id   log2(C) bits
    rank         log2(cluster size) bits, position along the cluster's own leading direction

The rank is the split direction from phases 38 and 39 used as a coordinate rather than as a cut:
one number saying how far along it an item lies. That makes the whole per-item payload about
fourteen bits against the 12,288 an f32 vector costs.

Two variants, because the second costs another C x 384 floats and has to earn them:

  CLUSTER ONLY      an item is its centroid, and members of a cluster are indistinguishable
  CLUSTER + RANK    an item is its centroid displaced along the cluster's direction by an
                    amount its rank encodes

Swept over C, because the trade is obvious in shape and not in where it lands: more clusters
means better resolution and more centroids to store, and the centroids are the only real cost.
Measured the same way as everything else here — bytes, and whether beam-16 retrieval survives.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analogy import leading  # noqa: E402
from beamwide import descend  # noqa: E402
from clustercodec import kmeans, pack  # noqa: E402
from embcodec import levels_from  # noqa: E402
from embednav import CACHE, embed  # noqa: E402
from longdoc import DOC, sentences  # noqa: E402


def bits_for(x):
    return max(1, int(np.ceil(np.log2(max(x, 2)))))


def main(seed=3, n_queries=40, out="data/custom/rankonly.json"):
    seed, n_queries = int(seed), int(n_queries)
    sents = sentences(Path(DOC).read_text())
    X = np.array(embed(sents, CACHE), dtype=np.float32)
    n, dim = X.shape
    rng = np.random.default_rng(seed)
    idx = [i for i, s in enumerate(sents) if 60 < len(s) < 220]
    picks = [int(i) for i in rng.choice(idx, size=n_queries, replace=False)]
    qv = np.array(embed([sents[i] for i in picks]), dtype=np.float32)

    def retrieval(vecs):
        lv = levels_from(np.ascontiguousarray(vecs, np.float32))
        return sum(descend(lv, qv[k], 16)[0][0] == i for k, i in enumerate(picks))

    print(f"{n:,} x {dim}, {X.nbytes / 1e6:.2f} MB raw, uncompressed retrieval "
          f"{retrieval(X)}/{len(picks)}\n")
    print(f"{'C':>6}{'variant':<18}{'bytes':>11}{'ratio':>8}{'bits/item':>11}"
          f"{'in cluster':>12}{'beam-16':>9}")

    rows = []
    for C in (256, 1024, 2048, 4096):
        assign, cent = kmeans(X, C, seed=seed)
        members = [np.where(assign == j)[0] for j in range(C)]
        biggest = max(len(m) for m in members)
        id_bits, rank_bits = bits_for(C), bits_for(biggest)

        # How often the target is even in the cluster the query lands in. This is the ceiling
        # for anything that only stores a cluster address.
        in_cluster = sum(int(assign[i] == int(np.argmax(cent @ qv[k])))
                         for k, i in enumerate(picks))

        rec_flat = cent[assign].copy()
        # The per-item payload: an id, plus a rank for the second variant.
        payload_flat = (n * id_bits + 7) // 8
        payload_rank = (n * (id_bits + rank_bits) + 7) // 8

        dirs = np.zeros((C, dim), np.float32)
        rec_rank = cent[assign].copy()
        for j in range(C):
            m = members[j]
            if len(m) < 2:
                continue
            d = leading(X[m] - cent[j])
            if d is None:
                continue
            dirs[j] = d
            proj = (X[m] - cent[j]) @ d
            order = np.argsort(proj)
            # Rank becomes a coordinate: the item's position in the sorted order, mapped back
            # onto the direction by the spread the cluster actually has. Nothing per item is
            # stored except that integer.
            span = float(proj.max() - proj.min())
            if span <= 0:
                continue
            steps = np.linspace(proj.min(), proj.max(), len(m))
            rec_rank[m[order]] = cent[j] + np.outer(steps, d)

        for name, rec, payload, extra in (
                ("cluster only", rec_flat, payload_flat, [cent.tobytes()]),
                ("cluster + rank", rec_rank, payload_rank,
                 [cent.tobytes(), dirs.tobytes()])):
            size = pack([b"\0" * payload, *extra])
            score = retrieval(rec)
            per_item = 8 * size / n
            rows.append({"C": C, "variant": name, "bytes": size, "bits_per_item": per_item,
                         "in_cluster": in_cluster, "retrieval": score,
                         "id_bits": id_bits, "rank_bits": rank_bits})
            print(f"{C:>6}{name:<18}{size:>11,}{X.nbytes / size:>8.2f}{per_item:>11.1f}"
                  f"{in_cluster:>9}/{len(picks):<2}{score:>7}/{len(picks)}")

    best = max((r for r in rows), key=lambda r: (r["retrieval"], -r["bytes"]))
    print(f"\nbest: {best['variant']} at C={best['C']}, {best['bytes']:,} bytes, "
          f"{X.nbytes / best['bytes']:.1f}x, retrieval {best['retrieval']}/{len(picks)}")
    print("The per-item cost is an address of a dozen bits. Everything else is centroids, so")
    print("this is a bet that the group carries the meaning and the item only carries a place.")
    summary = {"n": n, "dim": dim, "raw_bytes": int(X.nbytes), "queries": len(picks),
               "rows": rows, "best": best}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
