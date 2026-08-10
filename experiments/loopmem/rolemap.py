#!/usr/bin/env python3
"""Role retrieval against text retrieval, on the three sets where the difference shows.

Phase 88's key was masked surface text and it has been failing in a shape the numbers
already named: Norwegian problems retrieve their own class 8 times in 18 where English
manages 32 in 34, and olympiad problems retrieve nothing useful at all. Both are lexical
failures of a lexical key.

This measures the alternative on the same sets, changing ONLY how the two exemplars are
chosen:

    NORWEGIAN   18 problems, six classes, English bank      (text key: 8/18)
    MIXED       34 problems, seventeen classes, English     (text key: 32/34) — the
                control, where the text key already works and the role key must not
                make things worse
    AIME        30 unseen olympiad problems                 (text key: no class to hit,
                so scored differently: does the retrieved shape match the one a human
                would pick for the five that are hand-mappable?)

One extra model call per problem builds the signature — the verb, the subject, the
relations, the scope — and retrieval is role agreement, no embeddings anywhere.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bands2 import build as build_b2  # noqa: E402
from bands4 import build as build_b4  # noqa: E402
from bands5 import build as build_b5  # noqa: E402
from hardarith import build as build_hard  # noqa: E402
from newbands import build as build_nb  # noqa: E402
from norsk import build as build_no  # noqa: E402
from roles import extract_signature, retrieve  # noqa: E402
from solve import bank  # noqa: E402

# What a human would reach for on the five hand-mappable fresh AIME problems
# (phase 130), by index in that battery.
AIME_EXPECTED = {11: "count", 15: "count", 18: "big", 19: "count", 21: "sequence"}


def mixed_battery(per_class=2):
    pool = {}
    for fam, story, truth in (build_hard(1) + build_nb() + build_b2() + build_b4()
                              + build_b5()):
        pool.setdefault(fam, []).append((story, str(truth)))
    return [(fam, s) for fam, items in sorted(pool.items()) for s, _t in
            items[:per_class]]


def fresh_aime():
    import random
    from olympiad import load_problems
    _, aime = load_problems()
    used = {p for p, _a in random.Random(5).sample(aime, 30)}
    return [(None, p) for p, _a in aime if p not in used]


def main(k=2, out="data/custom/rolemap.json"):
    k = int(k)
    b = bank()
    tags = [t for t, _p, _s in b]
    result = {}

    for name, items, baseline in (
            ("norwegian", [(f, s) for f, s, _t in build_no()], "8/18"),
            ("mixed", mixed_battery(), "32/34"),
            ("aime_fresh", fresh_aime(), "n/a")):
        hits = miss = nosig = 0
        rows = []
        for idx, (fam, story) in enumerate(items):
            sig = extract_signature(story)
            if sig is None:
                nosig += 1
                rows.append({"family": fam, "signature": None, "shown": []})
                continue
            shown = [tags[j] for j in retrieve(sig, b, k)]
            if fam is not None:
                hit = fam in shown
                hits += hit
                miss += not hit
            elif idx in AIME_EXPECTED:
                hit = AIME_EXPECTED[idx] in shown
                hits += hit
                miss += not hit
            rows.append({"family": fam, "signature": sig, "shown": shown})
        n = len(items)
        scored = hits + miss
        print(f"{name:<12} role retrieval {hits}/{scored} scored of {n} problems "
              f"(text key: {baseline}); signatures unread {nosig}")
        result[name] = {"hits": hits, "scored": scored, "n": n, "no_signature": nosig,
                        "rows": rows}

    print("\nsignature vocabulary in use:")
    for name, r in result.items():
        acts = Counter(x["signature"]["action"] for x in r["rows"] if x["signature"])
        print(f"  {name:<12}{dict(acts)}")
    print("\nA key that is the verb, the subject and the relations should not care what")
    print("language the sentence is in or whether the story is about apples — and where")
    print("it still misses, the miss is a structure the vocabulary genuinely lacks")
    print("rather than a word the encoder never saw.")
    Path(out).write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
