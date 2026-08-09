#!/usr/bin/env python3
"""Does the way one cluster splits tell you how another one splits?

The architecture this thread arrived at compresses into clusters and reasons between them, and
the strongest claim in it is transfer: a split discovered in one neighbourhood should help split
a different one, even when the two are about unrelated things. That is what would make the
decomposition general rather than per-problem.

The split of a cluster is the direction along which it divides — the leading direction of its
members once its own centroid is removed. Borrowing means taking cluster A's direction and using
it on cluster B, and the question is how much of B's internal variation A's direction captures.

Four things are compared, and the third is what makes this a test rather than a demonstration:

  OWN        B's own leading direction. The ceiling — nothing borrowed can beat it.
  BORROWED   A's direction applied to B, averaged over every other cluster A.
  UNIVERSAL  the leading direction of ALL residuals pooled. If borrowed is no better than this,
             there is no analogy being transferred — only the global anisotropy of the embedding
             space, which every cluster shares because every cluster is in it.
  RANDOM     a random unit direction. The floor.

Without the universal control a positive result would be unreadable: borrowed directions would
score well simply because embedding spaces are not isotropic, and that has nothing to do with
one neighbourhood resembling another.

Reported two ways: variance captured, which is the geometric question, and article purity of the
resulting halves, which asks whether the borrowed split cuts along anything meaningful.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from clustercodec import kmeans  # noqa: E402
from crosscheck import article_of  # noqa: E402
from embednav import CACHE, embed  # noqa: E402
from longdoc import DOC, sentences  # noqa: E402


def leading(M):
    """Top right-singular vector of a centred block: the direction it varies most along."""
    if len(M) < 2:
        return None
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[0]


def captured(M, d):
    """Share of a block's variance that lies along direction d."""
    total = float((M ** 2).sum())
    if total <= 0:
        return 0.0
    return float(((M @ d) ** 2).sum() / total)


def split_purity(members, d, labels, X, centroid):
    """Cut the cluster by the sign of the projection and score both halves by article."""
    proj = (X[members] - centroid) @ d
    out = []
    for side in (proj >= 0, proj < 0):
        part = members[side]
        if len(part) < 2:
            continue
        counts = Counter(labels[i] for i in part)
        out.append(counts.most_common(1)[0][1] / len(part))
    return float(np.mean(out)) if out else 0.0


def main(n_clusters=64, min_size=30, seed=3, out="data/custom/analogy.json"):
    n_clusters, min_size, seed = int(n_clusters), int(min_size), int(seed)
    sents = sentences(Path(DOC).read_text())
    labels = article_of(sents)
    X = np.array(embed(sents, CACHE), dtype=np.float32)
    assign, cent = kmeans(X, n_clusters, seed=seed)

    groups = []
    for j in range(n_clusters):
        members = np.where(assign == j)[0]
        if len(members) >= min_size:
            block = X[members] - cent[j]
            d = leading(block)
            if d is not None:
                groups.append({"j": j, "members": members, "block": block, "dir": d})
    print(f"{len(groups)} clusters of {min_size}+ members out of {n_clusters}, "
          f"{sum(len(g['members']) for g in groups):,} sentences\n")

    # The universal direction: every cluster's residuals pooled, so it carries no information
    # about any particular neighbourhood.
    pooled = np.concatenate([g["block"] for g in groups])
    universal = leading(pooled)
    rng = np.random.default_rng(seed)

    rows = []
    for g in groups:
        B, d_own = g["block"], g["dir"]
        own = captured(B, d_own)
        borrowed = [captured(B, h["dir"]) for h in groups if h is not g]
        rnd = []
        for _ in range(8):
            r = rng.normal(size=X.shape[1])
            rnd.append(captured(B, r / np.linalg.norm(r)))
        rows.append({
            "cluster": int(g["j"]), "size": int(len(g["members"])),
            "own": own, "borrowed_mean": float(np.mean(borrowed)),
            "borrowed_best": float(np.max(borrowed)),
            "universal": captured(B, universal), "random": float(np.mean(rnd)),
            "purity_own": split_purity(g["members"], d_own, labels, X, cent[g["j"]]),
            "purity_borrowed": float(np.mean([
                split_purity(g["members"], h["dir"], labels, X, cent[g["j"]])
                for h in groups if h is not g])),
            "purity_random": float(np.mean([
                split_purity(g["members"], r / np.linalg.norm(r), labels, X, cent[g["j"]])
                for r in rng.normal(size=(4, X.shape[1]))])),
        })

    def col(k):
        return float(np.mean([r[k] for r in rows]))

    print(f"{'direction':<26}{'variance captured':>19}{'x random':>10}")
    for name, key in (("its own", "own"), ("borrowed, best other", "borrowed_best"),
                      ("borrowed, mean over all", "borrowed_mean"),
                      ("universal (pooled)", "universal"), ("random", "random")):
        print(f"{name:<26}{col(key):>19.4f}{col(key) / col('random'):>10.2f}")

    print(f"\n{'split quality':<26}{'article purity of halves':>26}")
    for name, key in (("its own", "purity_own"), ("borrowed", "purity_borrowed"),
                      ("random", "purity_random")):
        print(f"{name:<26}{col(key):>26.4f}")

    # The comparison the whole thing turns on.
    lift = (col("borrowed_mean") - col("random")) / max(col("universal") - col("random"), 1e-9)
    print(f"\nborrowed sits at {lift:.2f} of the way from random to the universal direction")
    if col("borrowed_mean") <= col("universal") * 1.02:
        print("which is at or below it — so what transfers is the shape of the space, not an")
        print("analogy between neighbourhoods. A borrowed split is a global direction in")
        print("disguise.")
    else:
        print("which is above it — clusters carry split structure that is genuinely shared and")
        print("not explained by the embedding space being anisotropic.")

    summary = {"clusters": len(groups), "min_size": min_size,
               "own": col("own"), "borrowed_mean": col("borrowed_mean"),
               "borrowed_best": col("borrowed_best"), "universal": col("universal"),
               "random": col("random"), "purity_own": col("purity_own"),
               "purity_borrowed": col("purity_borrowed"),
               "purity_random": col("purity_random"),
               "lift_toward_universal": lift, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
