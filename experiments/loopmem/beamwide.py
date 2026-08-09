#!/usr/bin/env python3
"""Widen the beam, score nodes by their loud dimensions, and rescue with an N x N hop.

Beam of 4 took tree routing from 18/40 to 31/40 against a flat scan's 40/40, at 116 comparisons
instead of 8,920. Three ways of closing what is left, and they are different kinds of fix:

  WIDER BEAM      keep more alternatives per level. Costs comparisons linearly and converges on
                  the flat scan by definition once the beam reaches the level width, so the
                  question is not whether it closes but where it stops being worth it.

  LOUD DIMENSIONS instead of the cosine over all 384 dimensions, score on the ones firing
                  unusually hard. This project measured `max_d min(a[d], b[d])` — shared extreme
                  — at +0.311 over cosine on cross-domain relatedness, 2.6x, so it is a real
                  candidate and not a guess. A mean-pooled parent washes out exactly the peaks
                  that make a leaf distinctive, which is the failure the greedy arm showed.

  N x N HOP       take what the beam did return and step to ITS nearest neighbours. If the true
                  target sits one hop from a retrieved leaf, the tree got close and stopped
                  short, and a hop is far cheaper than a wider beam. Computed streaming against
                  the leaf block rather than from a materialised 8,920 x 8,920 matrix.

The three answer different questions. The beam sweep says what brute force costs. The dimension
score says whether the summary can be improved rather than the search. The hop says whether the
remaining errors are near misses or genuine ones — and only the last of those tells you which of
the other two is worth paying for.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from embednav import CACHE, embed  # noqa: E402
from longdoc import BRANCH, DOC, build_tree, sentences  # noqa: E402


def load(n_queries, seed):
    text = Path(DOC).read_text()
    sents = sentences(text)
    leaves = np.array(embed(sents, CACHE), dtype=np.float32)
    levels = [leaves]
    while len(levels[-1]) > BRANCH:
        cur = levels[-1]
        pad = (-len(cur)) % BRANCH
        blocks = np.concatenate([cur, np.zeros((pad, cur.shape[1]), np.float32)])
        parent = blocks.reshape(-1, BRANCH, cur.shape[1]).sum(1)
        # The padding rows are zero, so they add nothing to the sum; only the divisor would be
        # wrong, and normalising immediately makes the divisor irrelevant.
        parent /= np.linalg.norm(parent, axis=1, keepdims=True) + 1e-12
        levels.append(parent.astype(np.float32))
    rng = np.random.default_rng(seed)
    idx = [i for i, s in enumerate(sents) if 60 < len(s) < 220]
    picks = list(rng.choice(idx, size=min(n_queries, len(idx)), replace=False))
    return sents, leaves, levels, [int(i) for i in picks]


def score(level, nodes, q, mode, topdims=None):
    """How well each candidate node matches the query, under one of three scores."""
    v = level[nodes]
    if mode == "cosine":
        return v @ q
    if mode == "topdims":
        # Only the dimensions the QUERY fires hardest on. A parent averaged over a thousand
        # sentences keeps its peaks better than it keeps its overall direction.
        return v[:, topdims] @ q[topdims]
    # shared extreme: the single dimension on which both are most strongly positive together
    return np.minimum(v, q).max(axis=1)


def descend(levels, q, beam, mode="cosine", top_k=24):
    topdims = np.argsort(q)[-top_k:] if mode == "topdims" else None
    frontier = np.arange(len(levels[-1]))
    comparisons = 0
    for depth in range(len(levels) - 1, -1, -1):
        s = score(levels[depth], frontier, q, mode, topdims)
        comparisons += len(frontier)
        keep = frontier[np.argsort(s)[-beam:][::-1]]
        if depth == 0:
            return [int(x) for x in keep], comparisons
        frontier = np.concatenate([
            np.arange(i * BRANCH, min((i + 1) * BRANCH, len(levels[depth - 1]))) for i in keep])
    return [], comparisons


def hop(leaves, candidates, q, k=8):
    """One N x N step: the nearest neighbours of what the beam returned, streamed not stored.

    Only the candidate rows are compared against the leaf block, so this is 32 x 8,920 dot
    products rather than the 8,920 x 8,920 matrix the whole streaming-kNN work exists to avoid.
    """
    out = set(candidates)
    for c in candidates:
        sims = leaves @ leaves[c]
        out.update(int(j) for j in np.argpartition(sims, -k)[-k:])
    return out


def main(n_queries=40, seed=3, out="data/custom/beamwide.json"):
    n_queries, seed = int(n_queries), int(seed)
    sents, leaves, levels, picks = load(n_queries, seed)
    print(f"{len(sents):,} leaves, levels {[len(x) for x in levels]}, "
          f"{leaves.shape[1]} dimensions, {len(picks)} queries\n")

    kinds = {"exact": [sents[i] for i in picks],
             "half": [sents[i][:len(sents[i]) // 2] for i in picks]}
    qv = {k: np.array(embed(v), dtype=np.float32) for k, v in kinds.items()}

    result = {"leaves": len(sents), "queries": len(picks)}

    print("beam sweep, cosine\n")
    print(f"{'beam':>5}{'exact':>8}{'half':>7}{'cmp':>8}   top-1 hit rate")
    sweep = []
    for beam in (1, 2, 4, 8, 16, 32, 64, 128):
        row = {"beam": beam}
        for kind in kinds:
            hits = cost = 0
            for k, i in enumerate(picks):
                cand, c = descend(levels, qv[kind][k], beam)
                cost += c
                hits += cand[0] == i
            row[kind] = hits
            row[f"{kind}_cmp"] = cost / len(picks)
        sweep.append(row)
        print(f"{beam:>5}{row['exact']:>6}/{len(picks):<2}{row['half']:>5}/{len(picks):<2}"
              f"{row['exact_cmp']:>8.0f}")
    result["sweep"] = sweep

    print("\nscoring the nodes differently, at beam 4 and 16\n")
    print(f"{'score':<14}{'beam':>5}{'exact':>8}{'half':>7}")
    scores = []
    for mode in ("cosine", "topdims", "shared"):
        for beam in (4, 16):
            row = {"mode": mode, "beam": beam}
            for kind in kinds:
                hits = 0
                for k, i in enumerate(picks):
                    cand, _ = descend(levels, qv[kind][k], beam, mode)
                    hits += cand[0] == i
                row[kind] = hits
            scores.append(row)
            print(f"{mode:<14}{beam:>5}{row['exact']:>6}/{len(picks):<2}"
                  f"{row['half']:>5}/{len(picks):<2}")
    result["scores"] = scores

    print("\none N x N hop out from what the beam returned\n")
    print(f"{'beam':>5}{'kind':>7}{'in beam':>9}{'after hop':>11}")
    hops = []
    for beam in (4, 16):
        for kind in kinds:
            inbeam = after = 0
            for k, i in enumerate(picks):
                cand, _ = descend(levels, qv[kind][k], beam)
                inbeam += i in cand
                after += i in hop(leaves, cand, qv[kind][k])
            hops.append({"beam": beam, "kind": kind, "in_beam": inbeam, "after_hop": after})
            print(f"{beam:>5}{kind:>7}{inbeam:>7}/{len(picks):<2}{after:>9}/{len(picks):<2}")
    result["hops"] = hops

    print("\nA hop that rescues a lot means the tree was landing next door and a wider beam is")
    print("the wrong medicine. A hop that rescues nothing means the misses are real misses.")
    Path(out).write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
