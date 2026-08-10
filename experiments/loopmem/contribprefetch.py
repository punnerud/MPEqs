#!/usr/bin/env python3
"""Composing WORD CONTRIBUTIONS instead of matching exact contexts: the starved fix.

Phase 140 measured two things that point in opposite directions. Keyed on exact n-grams,
a context's expert set agrees with itself more and more as the key grows (0.58 at one
token, 0.72 at four) — the signal is there. But the coverage column starved: only 0.9% of
four-gram keys had ever been seen, because exact-match tables grow as the product of
vocabularies while the corpus grows linearly.

The proposal this measures is the composable alternative: store a table PER WORD PER
RELATIVE POSITION — what does this word, standing d tokens back, contribute to the
routing here — and score a new position by ADDING the contributions of its last few
words. The tables are per-word, so they never starve on unseen combinations; a sentence
nobody wrote is still the sum of words everybody wrote. An index lookup for a sentence
becomes n small lookups and an addition, all available at time zero, in parallel.

    FREQUENCY      the layer's global favourites (the standing baseline, 52.7%)
    UNIGRAM        the word's own table (phase 140: 62.4%)
    EXACT BACKOFF  n-gram keys backing off to shorter ones (62.8%, starved)
    COMPOSED       sum of per-offset word contributions, weights chosen on a
                   validation slice of the training half, never on test

And the split that decides whether composition earns its keep: coverage on exactly the
positions whose bigram key was NEVER seen — the region where exact matching has nothing
and composition claims to generalise.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tokenprefetch import read_trace, tokenise_corpus  # noqa: E402

TRACE = Path("data/trace-contrib.bin")
OFFSETS = (0, 1, 2, 3)
SCHEMES = {"self only": (1.0,), "half decay": (1.0, 0.5, 0.25, 0.125),
           "gentle": (1.0, 0.6, 0.4, 0.25), "flat": (1.0, 1.0, 1.0, 1.0),
           "steep": (1.0, 0.3, 0.1, 0.03)}


def main(budget=16, out="data/custom/contribprefetch.json"):
    budget = int(budget)
    meta, recs = read_trace(TRACE)
    toks = tokenise_corpus(max(t for _l, t, _i in recs) + 1)
    usable = min(len(toks), max(t for _l, t, _i in recs) + 1)
    split = usable // 2
    val_from = split - split // 5          # last fifth of training picks the weights

    # contribution tables: (layer, offset, word) -> expert counter, training half only
    contrib = defaultdict(Counter)
    freq = defaultdict(Counter)
    bigram_seen = set()
    train, val, test = [], [], []
    for layer, pos, ids in recs:
        if pos >= usable:
            continue
        if pos < val_from:
            train.append((layer, pos, ids))
            for d in OFFSETS:
                if pos - d >= 0:
                    contrib[(layer, d, toks[pos - d])].update(ids)
            freq[layer].update(ids)
            if pos >= 1:
                bigram_seen.add((toks[pos - 1], toks[pos]))
        elif pos < split:
            val.append((layer, pos, ids))
        else:
            test.append((layer, pos, ids))

    totals = {k: sum(c.values()) for k, c in contrib.items()}
    freq_rank = {layer: [e for e, _c in c.most_common()] for layer, c in freq.items()}

    def composed(layer, pos, weights):
        score = Counter()
        for d, w in zip(OFFSETS, weights):
            if pos - d < 0:
                continue
            key = (layer, d, toks[pos - d])
            tab = contrib.get(key)
            if tab:
                t = totals[key]
                for e, c in tab.items():
                    score[e] += w * c / t
        pre = [e for e, _s in score.most_common(budget)]
        for e in freq_rank[layer]:
            if len(pre) >= budget:
                break
            if e not in pre:
                pre.append(e)
        return set(pre)

    def coverage(records, weights):
        hit = tot = 0
        for layer, pos, ids in records:
            hit += len(set(ids) & composed(layer, pos, weights))
            tot += len(ids)
        return hit / tot

    best_name, best_w, best_cov = None, None, -1.0
    for name, w in SCHEMES.items():
        cov = coverage(val, w)
        print(f"validation  {name:<11} {cov:.1%}")
        if cov > best_cov:
            best_name, best_w, best_cov = name, w, cov
    print(f"chosen on validation: {best_name}\n")

    # test: overall, and split by whether the exact bigram was ever seen in training
    hit = tot = 0
    starved_hit = starved_tot = seen_hit = seen_tot = 0
    uni_hit = 0
    for layer, pos, ids in test:
        pre = composed(layer, pos, best_w)
        uni = composed(layer, pos, (1.0,))
        h = len(set(ids) & pre)
        hit += h
        tot += len(ids)
        uni_hit += len(set(ids) & uni)
        starved = pos >= 1 and (toks[pos - 1], toks[pos]) not in bigram_seen
        if starved:
            starved_hit += h
            starved_tot += len(ids)
        else:
            seen_hit += h
            seen_tot += len(ids)

    base_hit = sum(len(set(ids) & set(freq_rank[layer][:budget]))
                   for layer, _p, ids in test)
    print(f"prefetch budget {budget} of {meta['experts']}, test half only\n")
    print(f"{'frequency':<26}{base_hit / tot:.1%}")
    print(f"{'unigram (self only)':<26}{uni_hit / tot:.1%}")
    print(f"{'exact backoff (ph. 140)':<26}62.8%")
    print(f"{'COMPOSED (' + best_name + ')':<26}{hit / tot:.1%}")
    print(f"\nwhere the exact bigram was NEVER seen "
          f"({starved_tot // meta['top_k']} positions):")
    print(f"{'  composed':<26}{starved_hit / max(starved_tot, 1):.1%}")
    print(f"where it had been seen ({seen_tot // meta['top_k']} positions):")
    print(f"{'  composed':<26}{seen_hit / max(seen_tot, 1):.1%}")
    print(f"\ntables held: {len(contrib)} (word, offset) entries — no n-gram product,")
    print("and a sentence's lookup is the sum of its words' lookups, in parallel,")
    print("before any layer runs. The gap left to the oracle is the part of routing")
    print("that only the forward pass knows: what attention adds that addition cannot.")
    summary = {"budget": budget, "scheme": best_name,
               "frequency": round(base_hit / tot, 4),
               "unigram": round(uni_hit / tot, 4),
               "exact_backoff_ref": 0.628,
               "composed": round(hit / tot, 4),
               "starved_composed": round(starved_hit / max(starved_tot, 1), 4),
               "seen_composed": round(seen_hit / max(seen_tot, 1), 4),
               "starved_positions": starved_tot // meta["top_k"],
               "tables": len(contrib)}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
