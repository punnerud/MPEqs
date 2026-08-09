#!/usr/bin/env python3
"""Does a borrowed split pay on small clusters, where a cluster cannot estimate its own?

Phase 38 showed a split transfers if the donor is chosen by similarity — 0.0942 captured against
0.0499 for a universal direction, matching an oracle. That was measured on the same members the
direction was fitted to, which is exactly the condition under which a cluster's OWN direction
looks unbeatable: fitted on n points and scored on the same n points, it cannot lose.

The case for borrowing is entirely about small clusters. A direction estimated from six members
is mostly noise; a direction borrowed from a large similar cluster is estimated from hundreds. So
the test has to be held out — fit on half the members, score on the other half — and it has to be
broken down by cluster size, because that is the variable the whole claim is about.

Two questions, and the second is the one that decides anything:

  HELD-OUT VARIANCE   does a borrowed direction capture more of the unseen half than the
                      cluster's own direction does, and below what size?

  ACTUAL BYTES        does splitting on it produce a smaller encoding at equal retrieval? A
                      direction that explains more variance still has to pay for the extra
                      sub-centroid it introduces, and this project has already seen a base cost
                      more than it saved twice.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analogy import captured, leading  # noqa: E402
from beamwide import descend  # noqa: E402
from clustercodec import bitpack, kmeans, pack  # noqa: E402
from embcodec import levels_from  # noqa: E402
from embednav import CACHE, embed  # noqa: E402
from longdoc import DOC, sentences  # noqa: E402

BUCKETS = [(4, 8), (8, 16), (16, 32), (32, 64), (64, 10 ** 9)]


def main(n_clusters=512, seed=3, bits=4, n_queries=40, out="data/custom/borrowsplit.json"):
    n_clusters, seed, bits, n_queries = int(n_clusters), int(seed), int(bits), int(n_queries)
    sents = sentences(Path(DOC).read_text())
    X = np.array(embed(sents, CACHE), dtype=np.float32)
    n, dim = X.shape
    assign, cent = kmeans(X, n_clusters, seed=seed)
    members = [np.where(assign == j)[0] for j in range(n_clusters)]
    sizes = np.array([len(m) for m in members])
    print(f"{n:,} vectors into {n_clusters} clusters: "
          f"median size {int(np.median(sizes))}, smallest {sizes.min()}, largest {sizes.max()}")

    # Every cluster's direction, fitted on ALL its members. This is what a donor offers.
    donor = {}
    for j in range(n_clusters):
        if len(members[j]) >= 2:
            d = leading(X[members[j]] - cent[j])
            if d is not None:
                donor[j] = d

    rng = np.random.default_rng(seed)
    rows = []
    for j in range(n_clusters):
        m = members[j]
        if len(m) < 4:
            continue
        perm = rng.permutation(len(m))
        fit, test = m[perm[: len(m) // 2]], m[perm[len(m) // 2:]]
        if len(fit) < 2 or len(test) < 2:
            continue
        # Fitted on half, scored on the other half. The cluster's centroid is used for both so
        # only the direction differs between arms.
        own = leading(X[fit] - cent[j])
        if own is None:
            continue
        # The donor is the most similar OTHER cluster, chosen the way phase 38 showed works.
        sims = cent @ cent[j]
        sims[j] = -2
        order = np.argsort(sims)[::-1]
        nearest = next((int(k) for k in order if int(k) in donor), None)
        if nearest is None:
            continue
        block = X[test] - cent[j]
        rows.append({
            "cluster": j, "size": int(len(m)),
            "own": captured(block, own),
            "borrowed": captured(block, donor[nearest]),
            "random": float(np.mean([
                captured(block, r / np.linalg.norm(r))
                for r in rng.normal(size=(4, dim))])),
        })

    print(f"\nheld-out variance captured, {len(rows)} clusters\n")
    print(f"{'cluster size':<16}{'count':>7}{'own':>9}{'borrowed':>11}{'random':>9}"
          f"{'borrowed wins':>15}")
    buckets = []
    for lo, hi in BUCKETS:
        sel = [r for r in rows if lo <= r["size"] < hi]
        if not sel:
            continue
        o = float(np.mean([r["own"] for r in sel]))
        b = float(np.mean([r["borrowed"] for r in sel]))
        rd = float(np.mean([r["random"] for r in sel]))
        wins = sum(1 for r in sel if r["borrowed"] > r["own"])
        label = f"{lo}-{hi - 1}" if hi < 10 ** 9 else f"{lo}+"
        buckets.append({"lo": lo, "hi": hi, "count": len(sel), "own": o, "borrowed": b,
                        "random": rd, "borrowed_wins": wins})
        print(f"{label:<16}{len(sel):>7}{o:>9.4f}{b:>11.4f}{rd:>9.4f}"
              f"{wins:>10}/{len(sel):<4}")

    # ---- and does it actually shrink the file, at equal retrieval
    idx = [i for i, s in enumerate(sents) if 60 < len(s) < 220]
    picks = [int(i) for i in rng.choice(idx, size=n_queries, replace=False)]
    qv = np.array(embed([sents[i] for i in picks]), dtype=np.float32)

    def retrieval(vecs):
        lv = levels_from(np.ascontiguousarray(vecs, np.float32))
        return sum(descend(lv, qv[k], 16)[0][0] == i for k, i in enumerate(picks))

    def encode(direction_for):
        """Split each cluster in two along a direction, store both sub-centroids, quantise."""
        rec = np.zeros_like(X)
        subs = []
        for j in range(n_clusters):
            m = members[j]
            if len(m) < 4 or direction_for(j) is None:
                rec[m] = cent[j]
                subs.append(cent[j])
                continue
            d = direction_for(j)
            side = (X[m] - cent[j]) @ d >= 0
            for mask in (side, ~side):
                part = m[mask]
                if len(part) == 0:
                    continue
                c = X[part].mean(0)
                c /= np.linalg.norm(c) + 1e-12
                rec[part] = c
                subs.append(c.astype(np.float32))
        res = X - rec
        lim = float(np.abs(res).max()) or 1.0
        step = lim / (2 ** (bits - 1) - 1)
        qi = np.clip(np.rint(res / step), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1).astype(np.int16)
        payload = bitpack(qi + (1 << (bits - 1)), bits)
        centroids = np.array(subs, np.float32).tobytes()
        size = pack([payload, centroids, assign.astype(np.int16).tobytes()])
        return size, rec + qi.astype(np.float32) * step, len(subs)

    print(f"\nencoding at {bits} bits, {len(picks)} queries\n")
    print(f"{'split direction':<22}{'bytes':>12}{'ratio':>8}{'sub-centroids':>15}{'beam-16':>9}")
    fitted = {j: leading(X[members[j]] - cent[j]) if len(members[j]) >= 4 else None
              for j in range(n_clusters)}
    nearest_of = {}
    for j in range(n_clusters):
        sims = cent @ cent[j]
        sims[j] = -2
        nearest_of[j] = next((int(k) for k in np.argsort(sims)[::-1] if int(k) in donor), None)
    variants = {
        "none (cluster only)": lambda j: None,
        "own": lambda j: fitted[j],
        "borrowed from nearest": lambda j: donor.get(nearest_of[j]),
    }
    enc = []
    for name, fn in variants.items():
        size, rec, nsub = encode(fn)
        score = retrieval(rec)
        enc.append({"variant": name, "bytes": size, "sub_centroids": nsub, "retrieval": score})
        print(f"{name:<22}{size:>12,}{X.nbytes / size:>8.2f}{nsub:>15,}{score:>7}/{len(picks)}")

    print("\nHeld out, a borrowed direction is fitted on hundreds of points and a small")
    print("cluster's own is fitted on three. Whether that advantage survives paying for the")
    print("extra sub-centroid is the only question that decides anything.")
    summary = {"n": n, "clusters": n_clusters, "bits": bits, "queries": len(picks),
               "median_size": int(np.median(sizes)), "buckets": buckets,
               "encodings": enc, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
