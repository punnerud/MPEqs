#!/usr/bin/env python3
"""How much CONTEXT does a word need before its routing becomes predictable?

Phase 139 keyed on the token alone and found it worth less than nothing, and the objection
to that is correct: a token does not mean anything by itself. What a word IS, to the
router, is the word together with what came before it — so the key should be the context,
not the symbol.

The systems point survives the correction completely, which is why this is worth
measuring. Every n-gram of a prompt is known at time zero, in parallel, with no forward
pass: if the last two or three tokens predict the experts, the fetch can start before any
layer runs, exactly as the single token was supposed to.

So the same corrected trace is keyed at four context lengths, each backing off to the
shorter one and finally to frequency, and two things are reported for each:

    COVERAGE       of the eight experts a position needs, how many were prefetched
    SELF-OVERLAP   when the SAME context recurs, how much do its expert sets agree —
                   against two random selections at the same layer, which is the floor

Predictability that only appears at length three would say the routing is contextual but
reachable; predictability that never appears says it is not reachable from the surface at
all, and the honest report is the curve either way.
"""
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations, islice
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).parent))
from tokenprefetch import read_trace, tokenise_corpus  # noqa: E402

TRACE = Path("data/trace-contrib.bin")
LENGTHS = (1, 2, 3, 4)


def key(toks, pos, n):
    """The last n tokens ending at pos — all of it known before any compute."""
    if pos + 1 < n:
        return None
    return tuple(toks[pos - n + 1:pos + 1])


def main(budget=16, out="data/custom/ctxprefetch.json"):
    budget = int(budget)
    meta, recs = read_trace(TRACE)
    toks = tokenise_corpus(max(t for _l, t, _i in recs) + 1)
    usable = min(len(toks), max(t for _l, t, _i in recs) + 1)
    split = usable // 2
    print(f"{meta['layers']} layers, {meta['experts']} experts, top-{meta['top_k']}; "
          f"{usable} positions, table on the first {split}\n")

    tables = {n: defaultdict(Counter) for n in LENGTHS}
    freq = defaultdict(Counter)
    test = []
    for layer, pos, ids in recs:
        if pos >= usable:
            continue
        if pos < split:
            for n in LENGTHS:
                k = key(toks, pos, n)
                if k is not None:
                    for e in ids:
                        tables[n][(layer, k)][e] += 1
            for e in ids:
                freq[layer][e] += 1
        else:
            test.append((layer, pos, ids))

    # coverage per context length, each backing off through shorter keys to frequency
    cov = {n: [0, 0, 0] for n in LENGTHS}          # hits, total, keyed
    base_hits = 0
    for layer, pos, ids in test:
        base = {e for e, _c in freq[layer].most_common(budget)}
        base_hits += len(set(ids) & base)
        for n in LENGTHS:
            pre, keyed = set(), False
            for m in range(n, 0, -1):              # back off to shorter contexts
                k = key(toks, pos, m)
                tab = tables[m].get((layer, k)) if k is not None else None
                if tab:
                    keyed = keyed or m == n
                    for e, _c in tab.most_common(budget):
                        if len(pre) >= budget:
                            break
                        pre.add(e)
                if len(pre) >= budget:
                    break
            for e, _c in freq[layer].most_common():
                if len(pre) >= budget:
                    break
                pre.add(e)
            cov[n][0] += len(set(ids) & pre)
            cov[n][1] += len(ids)
            cov[n][2] += keyed

    print(f"prefetch budget {budget} of {meta['experts']} per layer, "
          f"{budget / meta['experts']:.0%} residency\n")
    print(f"{'context':>9}{'coverage':>11}{'vs frequency':>15}{'keys that hit':>16}")
    total = cov[1][1]
    print(f"{'frequency':>9}{base_hits / total:>11.1%}{'—':>15}{'—':>16}")
    for n in LENGTHS:
        h, t, keyed = cov[n]
        print(f"{n:>9}{h / t:>11.1%}{(h - base_hits) / t:>+15.1%}"
              f"{keyed / t:>16.1%}")

    # self-overlap: when the same context recurs, does it route the same way?
    print(f"\n{'context':>9}{'recurring':>11}{'same-key overlap':>19}"
          f"{'random pair':>14}")
    rnd = Random(0)
    overlaps = {}
    for n in LENGTHS:
        for layer in (0, 8, 15):
            groups = defaultdict(list)
            for lay, pos, ids in recs:
                if lay != layer or pos >= usable:
                    continue
                k = key(toks, pos, n)
                if k is not None:
                    groups[k].append(frozenset(ids))
            rep = [v for v in groups.values() if len(v) >= 3]
            if not rep:
                continue
            same = [len(a & b) / meta["top_k"]
                    for v in rep for a, b in islice(combinations(v, 2), 10)]
            pool = [s for v in groups.values() for s in v]
            diff = [len(rnd.choice(pool) & rnd.choice(pool)) / meta["top_k"]
                    for _ in range(2000)]
            overlaps[f"n{n}_layer{layer}"] = [round(sum(same) / len(same), 3),
                                              round(sum(diff) / len(diff), 3),
                                              len(rep)]
            if layer == 8:
                print(f"{n:>9}{len(rep):>11}{sum(same) / len(same):>19.2f}"
                      f"{sum(diff) / len(diff):>14.2f}")

    print("\nEvery key above is available before a single layer runs, in parallel for the")
    print("whole prompt. The curve says how much of the sentence you have to hold before")
    print("the router becomes guessable from the surface — and if it never becomes")
    print("guessable, that is the same answer arrived at honestly.")
    summary = {"budget": budget, "positions": usable, "top_k": meta["top_k"],
               "frequency_coverage": round(base_hits / total, 4),
               "coverage": {str(n): round(cov[n][0] / cov[n][1], 4) for n in LENGTHS},
               "keyed_rate": {str(n): round(cov[n][2] / cov[n][1], 4) for n in LENGTHS},
               "self_overlap": overlaps}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
