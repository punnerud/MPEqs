#!/usr/bin/env python3
"""Clusters and links found by starting from several places and seeing what agrees.

One descent gives one answer and no way to tell a good one from a bad one. Starting from several
different places and cross-checking is the way to get both a cluster (things that keep being
found together) and a confidence (an answer several starts agree on) without a grader — which is
what phase 15 wanted, phase 23 could not get from inversion, and phase 26 could not get from
repetition, because sampling the same question again is not a different starting point.

Three things over the stored neighbour graph, and the document supplies its own ground truth: the
64 article boundaries in the wikitext are real groupings nobody told the index about.

  CLUSTERS    mutual k-nearest neighbours, connected. Two sentences are linked only if each is in
              the other's neighbour list, which is a much stronger claim than one-way similarity
              and is what stops everything collapsing into one component. Scored against the
              article each sentence actually came from.

  CROSS-CHECK the same target approached from several query variants. Where the descents agree,
              is the answer more likely to be right? That is the confidence signal.

  LINKS       mutual pairs whose two sentences come from DIFFERENT articles. Those are the
              cross-domain connections — the thing the hop work was for — and unlike a cluster
              they cannot be found by reading one article closely.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from beamwide import descend, load  # noqa: E402
from brokerhop import KNN_BIN, read_knn  # noqa: E402
from longdoc import DOC, sentences  # noqa: E402


def article_of(sents):
    """Which article each sentence belongs to, from the ` = Title = ` headers."""
    labels, current = [], "(front matter)"
    for s in sents:
        m = re.search(r"\n = ([^=\n]+?) = \n", s)
        if m:
            current = m.group(1).strip()
        labels.append(current)
    return labels


def mutual_components(ids):
    """Connected components of the mutual-kNN graph, via union-find."""
    n = len(ids)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sets = [set(int(j) for j in row) for row in ids]
    edges = 0
    for i in range(n):
        for j in sets[i]:
            if j != i and i in sets[j]:      # mutual only
                edges += 1
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    comp = defaultdict(list)
    for i in range(n):
        comp[find(i)].append(i)
    return list(comp.values()), edges // 2


def purity(components, labels, min_size=3):
    """Of the components worth calling clusters, what share of members share an article."""
    kept = [c for c in components if len(c) >= min_size]
    if not kept:
        return 0.0, 0, 0
    total = right = 0
    for c in kept:
        counts = Counter(labels[i] for i in c)
        right += counts.most_common(1)[0][1]
        total += len(c)
    return right / total, len(kept), total


def main(n_queries=40, seed=3, out="data/custom/crosscheck.json"):
    n_queries, seed = int(n_queries), int(seed)
    sents_raw = sentences(Path(DOC).read_text())
    labels = article_of(sents_raw)
    sents, leaves, levels, picks = load(n_queries, seed)
    ids, _ = read_knn(KNN_BIN)
    print(f"{len(sents):,} sentences across {len(set(labels))} articles, "
          f"stored graph {ids.shape[0]} x {ids.shape[1]}\n")

    comps, medges = mutual_components(ids)
    sizes = sorted((len(c) for c in comps), reverse=True)
    pur, kept, covered = purity(comps, labels)
    # The baseline a clustering has to beat: guessing the commonest article every time.
    base = Counter(labels).most_common(1)[0][1] / len(labels)
    print(f"mutual edges {medges:,}, components {len(comps):,}, "
          f"largest {sizes[0]}, singletons {sum(1 for s in sizes if s == 1):,}")
    print(f"clusters of 3+: {kept}, covering {covered:,} sentences, "
          f"article purity {pur:.3f} against a {base:.3f} baseline\n")

    # Cross-check: four ways into the same target, and what agreement is worth.
    from embednav import embed
    variants = {
        "whole": [sents[i] for i in picks],
        "first half": [sents[i][:len(sents[i]) // 2] for i in picks],
        "second half": [sents[i][len(sents[i]) // 2:] for i in picks],
        "middle": [sents[i][len(sents[i]) // 4: 3 * len(sents[i]) // 4] for i in picks],
    }
    qv = {k: np.array(embed(v), dtype=np.float32) for k, v in variants.items()}
    print(f"{'agreeing starts':>16}{'cases':>7}{'top-1 right':>13}")
    buckets = defaultdict(lambda: [0, 0])
    per_variant = Counter()
    for q_i, target in enumerate(picks):
        tops = {}
        for name in variants:
            cand, _ = descend(levels, qv[name][q_i], 4)
            tops[name] = cand[0]
            per_variant[name] += cand[0] == target
        # How many starts landed on the majority answer, and was the majority right?
        counts = Counter(tops.values())
        winner, agree = counts.most_common(1)[0]
        buckets[agree][0] += 1
        buckets[agree][1] += winner == target
    for agree in sorted(buckets):
        cases, right = buckets[agree]
        print(f"{agree:>13}/4{cases:>7}{right:>9}/{cases:<3}")
    print("\nper start: " + ", ".join(f"{k} {v}/{len(picks)}" for k, v in per_variant.items()))

    # Links: mutual pairs that cross an article boundary.
    sets = [set(int(j) for j in row) for row in ids]
    cross, within = [], 0
    for i in range(len(ids)):
        for j in sets[i]:
            if j > i and i in sets[j]:
                if labels[i] != labels[j]:
                    cross.append((i, j))
                else:
                    within += 1
    print(f"\nmutual pairs: {within:,} inside one article, {len(cross):,} across two "
          f"({100 * len(cross) / max(within + len(cross), 1):.1f}% cross-article)")
    for i, j in cross[:4]:
        print(f"  {labels[i][:26]:<26} <-> {labels[j][:26]:<26}")
        print(f"    {' '.join(sents[i].split())[:78]}")
        print(f"    {' '.join(sents[j].split())[:78]}")

    summary = {
        "sentences": len(sents), "articles": len(set(labels)),
        "mutual_edges": medges, "components": len(comps), "largest": sizes[0],
        "clusters_3plus": kept, "covered": covered, "purity": pur, "baseline": base,
        "agreement": {str(a): {"cases": v[0], "right": v[1]} for a, v in buckets.items()},
        "per_variant": dict(per_variant),
        "cross_article_pairs": len(cross), "within_article_pairs": within,
    }
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
