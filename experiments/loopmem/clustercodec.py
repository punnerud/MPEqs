#!/usr/bin/env python3
"""Cluster first, then predict from the cluster. One base per neighbourhood, not one per corpus.

Phase 36 found the clever bases nearly worthless: a global centroid added 0.2% and a global
rank-8 base LOST 8%. That is the expected result once stated properly — a single base for 8,920
sentences about actors, battleships and Tang poetry predicts none of them, so the residual is
almost the original and the base is pure overhead.

Clustering fixes exactly that. Within a cluster the vectors are close, so the residual from the
cluster's own centroid is small, and a small residual needs fewer bits for the same fidelity.
The bases cost C x 384 floats, which is nothing when C is a few hundred against 8,920 rows.

The measurement is set up so it cannot flatter itself. Rather than fixing the bit width and
comparing sizes, the bit width is SWEPT DOWNWARD and each scheme is asked how low it can go
while beam-16 retrieval still answers 40/40. Bytes at a quality nobody checked are not a result;
the smallest encoding that still works is.

  GLOBAL     one scale for everything, which is phase 36's 4.76x baseline
  CLUSTER-C  k-means into C clusters, each vector coded as its centroid plus a residual

Phase 34 already showed this corpus has real cluster structure — mutual-kNN at k=2 gives 0.953
article purity against a 0.056 baseline — so the neighbourhoods being exploited here are known
to exist rather than assumed.
"""
import gzip
import json
import lzma
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from beamwide import descend  # noqa: E402
from embcodec import levels_from  # noqa: E402
from embednav import CACHE, embed  # noqa: E402
from longdoc import DOC, sentences  # noqa: E402


def kmeans(X, c, iters=12, seed=0):
    """Plain Lloyd from a random sample. Enough for a codec — the clusters only have to be
    tight, not optimal, and a better clusterer would only improve the numbers below."""
    rng = np.random.default_rng(seed)
    cent = X[rng.choice(len(X), size=c, replace=False)].copy()
    for _ in range(iters):
        # Assign by nearest centroid, in blocks so the n x c matrix never materialises whole.
        assign = np.empty(len(X), np.int32)
        for s in range(0, len(X), 2048):
            blk = X[s:s + 2048]
            assign[s:s + 2048] = np.argmax(blk @ cent.T, axis=1)
        for j in range(c):
            members = X[assign == j]
            if len(members):
                cent[j] = members.mean(0)
        cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
    return assign, cent.astype(np.float32)


def bitpack(vals, bits):
    """Pack unsigned values of `bits` width, vectorised.

    The loop version took longer than every other measurement in this file put together —
    3.4 million Python iterations per width per scheme — and produced the same bytes.
    """
    flat = np.ascontiguousarray(vals.ravel().astype(np.uint16))
    as_bits = np.unpackbits(flat.view(np.uint8).reshape(-1, 2)[:, ::-1], axis=1)
    return np.packbits(as_bits[:, -bits:].ravel()).tobytes()


def pack(payloads):
    blob = b"".join(payloads)
    return min(len(blob), len(gzip.compress(blob, 9)), len(lzma.compress(blob, preset=9)))


def main(n_queries=40, seed=3, out="data/custom/clustercodec.json"):
    n_queries, seed = int(n_queries), int(seed)
    sents = sentences(Path(DOC).read_text())
    X = np.array(embed(sents, CACHE), dtype=np.float32)
    n, dim = X.shape

    rng = np.random.default_rng(seed)
    idx = [i for i, s in enumerate(sents) if 60 < len(s) < 220]
    picks = [int(i) for i in rng.choice(idx, size=min(n_queries, len(idx)), replace=False)]
    qv = np.array(embed([sents[i] for i in picks]), dtype=np.float32)

    def retrieval(vecs):
        lv = levels_from(np.ascontiguousarray(vecs, dtype=np.float32))
        return sum(descend(lv, qv[k], 16)[0][0] == i for k, i in enumerate(picks))

    target = retrieval(X)
    print(f"{n:,} x {dim} embeddings, {X.nbytes / 1e6:.2f} MB raw, "
          f"uncompressed retrieval {target}/{len(picks)}\n")

    schemes = {"global": (None, None)}
    for c in (64, 256, 1024):
        print(f"clustering into {c}...", flush=True)
        schemes[f"cluster-{c}"] = kmeans(X, c, seed=seed)

    print(f"\n{'scheme':<12}{'bits':>6}{'bytes':>12}{'ratio':>8}{'max err':>10}{'beam-16':>9}")
    rows, best = [], {}
    for name, (assign, cent) in schemes.items():
        for bits in (8, 6, 5, 4, 3, 2):
            if assign is None:
                res = X
                extra = []
            else:
                res = X - cent[assign]
                # The bases and the assignment are part of the encoding and are counted.
                extra = [cent.tobytes(), assign.astype(np.int16).tobytes()]
            lim = float(np.abs(res).max()) or 1.0
            step = lim / (2 ** (bits - 1) - 1)
            qi = np.clip(np.rint(res / step), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            qi = qi.astype(np.int16)
            payload = bitpack(qi + (1 << (bits - 1)), bits)
            rec = qi.astype(np.float32) * step
            if assign is not None:
                rec = rec + cent[assign]
            size = pack([payload, *extra])
            err = float(np.abs(rec - X).max())
            score = retrieval(rec)
            rows.append({"scheme": name, "bits": bits, "bytes": size,
                         "max_err": err, "retrieval": score})
            print(f"{name:<12}{bits:>6}{size:>12,}{X.nbytes / size:>8.2f}"
                  f"{err:>10.4f}{score:>7}/{len(picks)}")
            if score >= target:
                cur_best = best.get(name)
                if cur_best is None or size < cur_best["bytes"]:
                    best[name] = {"bits": bits, "bytes": size, "max_err": err}

    print(f"\n{'scheme':<12}{'lowest bits that still answers':>32}{'bytes':>12}{'ratio':>8}")
    for name, b in best.items():
        print(f"{name:<12}{b['bits']:>32}{b['bytes']:>12,}{X.nbytes / b['bytes']:>8.2f}")

    winner = min(best.items(), key=lambda kv: kv[1]["bytes"]) if best else None
    if winner:
        print(f"\nsmallest encoding that still retrieves {target}/{len(picks)}: "
              f"{winner[0]} at {winner[1]['bits']} bits, {winner[1]['bytes']:,} bytes, "
              f"{X.nbytes / winner[1]['bytes']:.1f}x")
    print("A base is only worth its storage if it predicts. One centroid per corpus predicts")
    print("nothing; one per neighbourhood shrinks the residual, and the bit width follows.")

    summary = {"n": n, "dim": dim, "raw_bytes": int(X.nbytes), "target": target,
               "queries": len(picks), "rows": rows, "best_per_scheme": best,
               "winner": winner[0] if winner else None,
               "winner_bytes": winner[1]["bytes"] if winner else None,
               "winner_bits": winner[1]["bits"] if winner else None}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
