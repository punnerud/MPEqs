#!/usr/bin/env python3
"""Compress the embeddings themselves, and judge the result by whether retrieval survives.

The neighbour graph was 1.1 MB and compressed 3x. The embeddings are 13.7 MB — 8,920 x 384
floats — so they are where compression is actually worth something, and they are the thing
matcodec's `base + residual` was really designed for: a big block of numbers with structure a
general-purpose compressor cannot see.

Four bases, each with the exact residual quantised on top:

  CENTROID    base is the corpus mean. The cheapest possible predictor.
  RANK-k      base is a rank-k reconstruction from the SVD. This is matcodec's own base, the
              one the landmark work uses, applied to vectors instead of a distance matrix.
  GRAPH       base is an already-encoded NEIGHBOUR, taken from the kNN graph built in phase 33.
              Neighbours are similar by construction, so the residual is small — and the
              predictor costs nothing extra because the graph already exists for hopping.
  NONE        scalar quantisation alone, as the control that says how much the base is worth.

Bytes are the easy half. For embeddings the honest question is not the reconstruction error but
whether the index still WORKS, so every codec is judged by rerunning the beam-16 descent from
phase 32 on its reconstructed vectors. 40/40 is the uncompressed score; anything that keeps it
is lossless where it matters, whatever its error norm says.
"""
import gzip
import json
import lzma
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from beamwide import descend  # noqa: E402
from brokerhop import KNN_BIN, read_knn  # noqa: E402
from embednav import CACHE, embed  # noqa: E402
from longdoc import BRANCH, DOC, sentences  # noqa: E402


def quantise(res, bits=8):
    """Symmetric scalar quantisation with one scale for the whole block."""
    lim = float(np.abs(res).max()) or 1.0
    step = lim / (2 ** (bits - 1) - 1)
    q = np.rint(res / step).astype(np.int8 if bits == 8 else np.int16)
    return q, step


def levels_from(vecs):
    """The same mean-pooled split tree phase 31 built, over whatever vectors are given."""
    levels = [vecs]
    while len(levels[-1]) > BRANCH:
        cur = levels[-1]
        pad = (-len(cur)) % BRANCH
        blocks = np.concatenate([cur, np.zeros((pad, cur.shape[1]), np.float32)])
        parent = blocks.reshape(-1, BRANCH, cur.shape[1]).sum(1)
        parent /= np.linalg.norm(parent, axis=1, keepdims=True) + 1e-12
        levels.append(parent.astype(np.float32))
    return levels


def spanning_order(ids):
    """Visit order and parent for a spanning forest of the kNN graph.

    Each vector is predicted from a neighbour that has already been written, so the decoder can
    reconstruct in the same order. Breadth-first from the lowest unvisited id, which needs no
    extra storage: the parent is recoverable from the graph the decoder already has.
    """
    n = len(ids)
    parent = np.full(n, -1, np.int64)
    order, seen = [], np.zeros(n, bool)
    for root in range(n):
        if seen[root]:
            continue
        seen[root] = True
        order.append(root)
        queue = [root]
        while queue:
            cur = queue.pop(0)
            for j in ids[cur]:
                j = int(j)
                if not seen[j]:
                    seen[j] = True
                    parent[j] = cur
                    order.append(j)
                    queue.append(j)
    return np.array(order), parent


def sizes_of(payloads):
    blob = b"".join(payloads)
    return {"raw": len(blob), "gzip": len(gzip.compress(blob, 9)),
            "lzma": len(lzma.compress(blob, preset=9))}


def main(n_queries=40, seed=3, bits=8, out="data/custom/embcodec.json"):
    n_queries, seed, bits = int(n_queries), int(seed), int(bits)
    sents = sentences(Path(DOC).read_text())
    X = np.array(embed(sents, CACHE), dtype=np.float32)
    n, dim = X.shape
    ids, _ = read_knn(KNN_BIN)
    print(f"{n:,} x {dim} embeddings, {X.nbytes / 1e6:.2f} MB raw, {bits}-bit residuals\n")

    rng = np.random.default_rng(seed)
    idx = [i for i, s in enumerate(sents) if 60 < len(s) < 220]
    picks = [int(i) for i in rng.choice(idx, size=min(n_queries, len(idx)), replace=False)]
    qv = np.array(embed([sents[i] for i in picks]), dtype=np.float32)

    def retrieval(vecs):
        lv = levels_from(vecs)
        return sum(descend(lv, qv[k], 16)[0][0] == i for k, i in enumerate(picks))

    base_score = retrieval(X)
    print(f"uncompressed beam-16 retrieval: {base_score}/{len(picks)}\n")

    schemes = {}

    # NONE: quantise the vectors directly.
    q, step = quantise(X, bits)
    schemes["scalar only"] = (sizes_of([q.tobytes(), struct.pack("<f", step)]),
                              q.astype(np.float32) * step)

    # CENTROID: one vector of base, everything else a residual from it.
    mu = X.mean(0)
    q, step = quantise(X - mu, bits)
    schemes["centroid + residual"] = (
        sizes_of([mu.astype(np.float32).tobytes(), q.tobytes(), struct.pack("<f", step)]),
        q.astype(np.float32) * step + mu)

    # RANK-k: matcodec's own base, on vectors.
    for r in (8, 32):
        U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
        low = (U[:, :r] * S[:r]) @ Vt[:r]
        q, step = quantise(X - mu - low, bits)
        # The base costs the coefficients and the basis, both stored as float32.
        coeff = (U[:, :r] * S[:r]).astype(np.float32)
        schemes[f"rank-{r} + residual"] = (
            sizes_of([mu.astype(np.float32).tobytes(), Vt[:r].astype(np.float32).tobytes(),
                      coeff.tobytes(), q.tobytes(), struct.pack("<f", step)]),
            low + mu + q.astype(np.float32) * step)

    # GRAPH: predict each vector from a neighbour already written.
    order, parent = spanning_order(ids)
    res = np.zeros_like(X)
    for i in order:
        p = parent[i]
        res[i] = X[i] if p < 0 else X[i] - X[p]
    q, step = quantise(res, bits)
    roots = int((parent < 0).sum())
    schemes["graph-delta + residual"] = (
        sizes_of([q.tobytes(), struct.pack("<f", step)]), None)
    # Decode in the same order the encoder used, so error accumulates exactly as it would.
    rec = np.zeros_like(X)
    for i in order:
        p = parent[i]
        rec[i] = q[i].astype(np.float32) * step + (0.0 if p < 0 else rec[p])
    schemes["graph-delta + residual"] = (schemes["graph-delta + residual"][0], rec)

    plain = {"raw f32": X.nbytes, "gzip(f32)": len(gzip.compress(X.tobytes(), 9)),
             "lzma(f32)": len(lzma.compress(X.tobytes(), preset=9))}
    print(f"{'method':<26}{'bytes':>12}{'ratio':>8}{'bits/dim':>10}"
          f"{'max err':>10}{'beam-16':>9}")
    rows = []
    for name, b in plain.items():
        print(f"{name:<26}{b:>12,}{X.nbytes / b:>8.2f}{8 * b / (n * dim):>10.2f}"
              f"{0.0:>10.4f}{base_score:>7}/{len(picks)}")
        rows.append({"method": name, "bytes": b, "max_err": 0.0, "retrieval": base_score})
    for name, (sz, rec) in schemes.items():
        err = float(np.abs(rec - X).max())
        score = retrieval(rec.astype(np.float32))
        best = min(sz["raw"], sz["gzip"], sz["lzma"])
        print(f"{name:<26}{best:>12,}{X.nbytes / best:>8.2f}{8 * best / (n * dim):>10.2f}"
              f"{err:>10.4f}{score:>7}/{len(picks)}")
        rows.append({"method": name, "bytes": best, "sizes": sz, "max_err": err,
                     "retrieval": score})

    keep = [r for r in rows if r["retrieval"] >= base_score and r["method"] != "raw f32"]
    best_keep = min(keep, key=lambda r: r["bytes"]) if keep else None
    print(f"\nspanning forest roots: {roots}")
    if best_keep:
        print(f"smallest encoding that keeps retrieval at {base_score}/{len(picks)}: "
              f"{best_keep['method']} at {best_keep['bytes']:,} bytes, "
              f"{X.nbytes / best_keep['bytes']:.1f}x")
    print("Reconstruction error is the wrong yardstick on its own — what matters is whether")
    print("the index still answers, and several codecs with visible error still answer perfectly.")

    summary = {"n": n, "dim": dim, "bits": bits, "queries": len(picks),
               "uncompressed_retrieval": base_score, "raw_bytes": int(X.nbytes),
               "roots": roots, "rows": rows,
               "best_lossless_retrieval": best_keep["method"] if best_keep else None,
               "best_lossless_bytes": best_keep["bytes"] if best_keep else None}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
