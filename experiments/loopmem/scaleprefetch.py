#!/usr/bin/env python3
"""Eight times the trace: which prefetch coverage was data, which was structure?

Phases 140-141 split their conclusion into two columns and admitted one of them was
starved: only 8.6% of test positions had even their unigram key seen in a 4,096-position
training half, so the coverage numbers measured the corpus while the overlap numbers
measured the hypothesis. The trace has now been regenerated at 65,536 tokens — eight
times the data through the same corrected shim, same model, same corpus.

PREDICTIONS, WRITTEN BEFORE THE RUN so they cannot be adjusted afterwards:

    unigram coverage    rises clearly — fewer unseen tokens is pure data
    exact-context       rises more than unigram — its tables were the starved ones
    composed-vs-unigram stays a small gain — non-additivity is a property of routing,
                        not an artifact of table size, and more data should not
                        manufacture additivity
    same-key overlap    roughly unchanged — it was never starved, 0.58/0.72 already
                        measured on recurring keys only

Both traces run through identical code below; the only variable is the data.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tokenprefetch import read_trace, tokenise_corpus  # noqa: E402

BUDGET = 16
OFFSETS = (0, 1, 2, 3)
STEEP = (1.0, 0.3, 0.1, 0.03)


def measure(trace_path, toks, interleave=False):
    """interleave=True splits by alternating 1024-token blocks, so train and test
    cover the same registers. The half-split run exposed why that matters: the corpus
    is several registers concatenated, the 32k half-split trained on prose and tested
    on code and Norwegian, and every table looked like it got worse with more data.
    A fallen frequency baseline (52.7 -> 51.9) was the tell — no lookup table can be
    hurt by more rows, but every table is hurt by a moved test set."""
    meta, recs = read_trace(Path(trace_path))
    usable = min(len(toks), max(t for _l, t, _i in recs) + 1)
    split = usable // 2

    def is_train(pos):
        if interleave:
            return (pos // 1024) % 2 == 0
        return pos < split

    uni = defaultdict(Counter)
    bi = defaultdict(Counter)
    contrib = defaultdict(Counter)
    freq = defaultdict(Counter)
    bigram_seen = set()
    test = []
    for layer, pos, ids in recs:
        if pos >= usable:
            continue
        if is_train(pos):
            uni[(layer, toks[pos])].update(ids)
            if pos >= 1:
                bi[(layer, toks[pos - 1], toks[pos])].update(ids)
                bigram_seen.add((toks[pos - 1], toks[pos]))
            for d in OFFSETS:
                if pos - d >= 0:
                    contrib[(layer, d, toks[pos - d])].update(ids)
            freq[layer].update(ids)
        else:
            test.append((layer, pos, ids))

    totals = {k: sum(c.values()) for k, c in contrib.items()}
    frank = {la: [e for e, _c in c.most_common()] for la, c in freq.items()}

    def pad(pre, layer):
        pre = list(pre)
        for e in frank[layer]:
            if len(pre) >= BUDGET:
                break
            if e not in pre:
                pre.append(e)
        return set(pre)

    scores = Counter()
    tot = 0
    starved = [0, 0]
    for layer, pos, ids in test:
        idset = set(ids)
        tot += len(ids)
        scores["freq"] += len(idset & set(frank[layer][:BUDGET]))
        t = uni.get((layer, toks[pos]))
        scores["unigram"] += len(idset & pad(
            [e for e, _c in t.most_common(BUDGET)] if t else [], layer))
        b = bi.get((layer, toks[pos - 1], toks[pos])) if pos >= 1 else None
        pre = [e for e, _c in b.most_common(BUDGET)] if b else \
            ([e for e, _c in t.most_common(BUDGET)] if t else [])
        scores["exact"] += len(idset & pad(pre, layer))
        sc = Counter()
        for d, w in zip(OFFSETS, STEEP):
            if pos - d >= 0:
                tab = contrib.get((layer, d, toks[pos - d]))
                if tab:
                    tt = totals[(layer, d, toks[pos - d])]
                    for e, c in tab.items():
                        sc[e] += w * c / tt
        comp_hit = len(idset & pad([e for e, _s in sc.most_common(BUDGET)], layer))
        scores["composed"] += comp_hit
        is_starved = pos >= 1 and (toks[pos - 1], toks[pos]) not in bigram_seen
        starved[0] += is_starved * len(ids)
        starved[1] += len(ids)

    return {"positions": usable, "train": split,
            "coverage": {k: round(v / tot, 4) for k, v in scores.items()},
            "starved_share": round(starved[0] / starved[1], 4)}


def measure_fixed_test(trace_path, toks, train_limit):
    """ONE trace, ONE test set, only the training volume varies.

    The two-trace comparison had two confounds the numbers themselves exposed: the test
    halves were different text, and 23% of overlapping records differ between the two
    traces anyway (Metal flips boundary experts; the sets that differ still share 6.7 of
    8). Test = odd 1024-blocks within the first 8192 positions; training = even blocks
    there, plus — when train_limit allows — everything beyond 8192. Same trace, same
    test rows, no confounds left except the thing being measured."""
    meta, recs = read_trace(Path(trace_path))
    usable = min(len(toks), max(t for _l, t, _i in recs) + 1)

    def role(pos):
        if pos < 8192:
            return "test" if (pos // 1024) % 2 else "train"
        return "train" if pos < train_limit else "drop"

    uni, bi, contrib, freq = (defaultdict(Counter) for _ in range(4))
    bigram_seen, test = set(), []
    n_train = 0
    for layer, pos, ids in recs:
        if pos >= usable:
            continue
        r = role(pos)
        if r == "train":
            n_train += layer == 0
            uni[(layer, toks[pos])].update(ids)
            if pos >= 1:
                bi[(layer, toks[pos - 1], toks[pos])].update(ids)
                bigram_seen.add((toks[pos - 1], toks[pos]))
            for d in OFFSETS:
                if pos - d >= 0:
                    contrib[(layer, d, toks[pos - d])].update(ids)
            freq[layer].update(ids)
        elif r == "test":
            test.append((layer, pos, ids))

    totals = {k: sum(c.values()) for k, c in contrib.items()}
    frank = {la: [e for e, _c in c.most_common()] for la, c in freq.items()}

    def pad(pre, layer):
        pre = list(pre)
        for e in frank[layer]:
            if len(pre) >= BUDGET:
                break
            if e not in pre:
                pre.append(e)
        return set(pre)

    scores, tot, starved = Counter(), 0, [0, 0]
    for layer, pos, ids in test:
        idset = set(ids)
        tot += len(ids)
        scores["freq"] += len(idset & set(frank[layer][:BUDGET]))
        t = uni.get((layer, toks[pos]))
        scores["unigram"] += len(idset & pad(
            [e for e, _c in t.most_common(BUDGET)] if t else [], layer))
        b = bi.get((layer, toks[pos - 1], toks[pos])) if pos >= 1 else None
        pre = [e for e, _c in b.most_common(BUDGET)] if b else             ([e for e, _c in t.most_common(BUDGET)] if t else [])
        scores["exact"] += len(idset & pad(pre, layer))
        sc = Counter()
        for d, w in zip(OFFSETS, STEEP):
            if pos - d >= 0:
                tab = contrib.get((layer, d, toks[pos - d]))
                if tab:
                    tt = totals[(layer, d, toks[pos - d])]
                    for e, c in tab.items():
                        sc[e] += w * c / tt
        scores["composed"] += len(idset & pad(
            [e for e, _s in sc.most_common(BUDGET)], layer))
        st = pos >= 1 and (toks[pos - 1], toks[pos]) not in bigram_seen
        starved[0] += st * len(ids)
        starved[1] += len(ids)
    return {"train_positions": n_train,
            "coverage": {k: round(v / tot, 4) for k, v in scores.items()},
            "starved_share": round(starved[0] / starved[1], 4)}


def main(out="data/custom/scaleprefetch.json"):
    toks = tokenise_corpus(70000)
    small = measure_fixed_test("data/trace-big.bin", toks, train_limit=8192)
    big = measure_fixed_test("data/trace-big.bin", toks, train_limit=65536)

    print(f"fixed test set; training {small['train_positions']} vs "
          f"{big['train_positions']} positions\n")
    print(f"{'':<12}{'4k train':>10}{'61k train':>11}{'delta':>8}")
    for k in ("freq", "unigram", "exact", "composed"):
        a, b = small["coverage"][k], big["coverage"][k]
        print(f"{k:<12}{a:>10.1%}{b:>11.1%}{b - a:>+8.1%}")
    print(f"{'starved %':<12}{small['starved_share']:>10.1%}"
          f"{big['starved_share']:>11.1%}")
    ex_gain_small = small["coverage"]["exact"] - small["coverage"]["unigram"]
    ex_gain_big = big["coverage"]["exact"] - big["coverage"]["unigram"]
    co_gain_small = small["coverage"]["composed"] - small["coverage"]["unigram"]
    co_gain_big = big["coverage"]["composed"] - big["coverage"]["unigram"]
    print(f"\nexact-over-unigram gain : {ex_gain_small:+.1%} -> {ex_gain_big:+.1%}")
    print(f"composed-over-unigram   : {co_gain_small:+.1%} -> {co_gain_big:+.1%}")
    print("\nThe predictions were written in the docstring before the big trace was")
    print("read. Whichever way each number moved, that is the verdict on whether it")
    print("was measuring the corpus or the routing.")
    summary = {"small": small, "big": big,
               "exact_gain": [round(ex_gain_small, 4), round(ex_gain_big, 4)],
               "composed_gain": [round(co_gain_small, 4), round(co_gain_big, 4)]}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
