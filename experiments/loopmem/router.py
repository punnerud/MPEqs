#!/usr/bin/env python3
"""Who should do the work? A mechanical router over every problem measured on both arms.

Phase 110's law is a deployment instruction if anything can act on it: route by whether
the arithmetic will fit in the model's head, not by topic. Every battery stored its
per-problem outcomes, so the router can be built and scored with no model call at all —
219 problems, each with a text, a solo verdict and an MPEqs verdict already recorded.

The rule is written A PRIORI from the law rather than fitted to the outcomes, and it
reads only the problem text:

    send it to the machinery when the text demands exactness (the words "exact",
    "fraction", "remainder"), or carries a fraction, or names a number past what fits
    in a head (>= 10000), or spans a range of that size, or is calendar arithmetic;
    otherwise let the model answer

Scored against four references: always-solo, always-MPEqs, the oracle that picks the
right arm every time, and a coin. A fitted variant is reported separately, under
leave-one-out so it cannot memorise the problem it is scoring.
"""
import json
import re
import sys
from pathlib import Path

SOURCES = [
    ("gsmsolve.json", "solo_ok", "m35"),
    ("hardarith.json", "solo_ok", "mpeqs_ok"),
    ("hardarith_heldout.json", "solo_ok", "mpeqs_ok"),
    ("newbands.json", "solo_ok", "mpeqs_ok"),
    ("bands2.json", "solo_ok", "mpeqs_ok"),
    ("bands3.json", "solo_ok", "mpeqs_ok"),
    ("bands3_hard.json", "solo_ok", "mpeqs_ok"),
    ("mixedretr.json", "solo_ok", "mpeqs_ok"),
]

EXACT_WORDS = ("exact", "fraction", "remainder", "precisely")
CALENDAR = ("day", "days", "date", "weekday", "leap", "january", "february", "march",
            "april", "may", "june", "july", "august", "september", "october",
            "november", "december")


def features(text):
    t = text.lower()
    nums = [int(x) for x in re.findall(r"\d+", t)]
    return {
        "exact_word": any(w in t for w in EXACT_WORDS),
        "has_fraction": bool(re.search(r"\d+\s*/\s*\d+", t)),
        "big_number": max(nums) >= 10000 if nums else False,
        "wide_range": bool(re.search(r"from\s+\d+\s+to\s+(\d{5,})", t)),
        "calendar": sum(w in t for w in CALENDAR) >= 2,
        "many_numbers": len(nums) >= 6,
    }


def apriori(text):
    """The rule as written from the law, before looking at a single outcome."""
    f = features(text)
    return (f["exact_word"] or f["has_fraction"] or f["big_number"]
            or f["wide_range"] or f["calendar"])


def reconstruct(fname):
    """Some batteries stored verdicts without the text. Every one of them draws its
    problems deterministically, so the texts come back exactly by replaying the draw —
    which is cheaper and safer than re-running thirty model calls to recover a string.
    """
    import random as _r
    import sys as _s
    _s.path.insert(0, str(Path(__file__).parent))
    if fname == "gsmsolve.json":
        from olympiad import load_problems
        gsm, _ = load_problems()
        return [q for q, _a in _r.Random(3).sample(gsm, 30)]
    if fname == "mixedretr.json":
        from bands2 import build as b2
        from hardarith import build as bh
        from newbands import build as bn
        pool = {}
        for fam, story, _t in (bh(1) + bn() + b2()):
            pool.setdefault(fam, []).append(story)
        return [s for items in pool.values() for s in items[:3]]
    return None


def load(root):
    items = []
    for fname, solo_key, mp_key in SOURCES:
        p = root / fname
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        texts = reconstruct(fname)
        for idx, r in enumerate(d["rows"]):
            text = r.get("story") or (texts[idx] if texts and idx < len(texts) else "")
            if not text:
                continue
            solo = bool(r.get(solo_key))
            if mp_key == "m35":            # gsmsolve stores a string, not a flag
                mp = bool(r.get("m35")) and "!=" not in str(r.get("m35")) and \
                    "refused" not in str(r.get("m35")) and \
                    str(r.get("m35")) != "no spec"
            else:
                mp = bool(r.get(mp_key))
            # Did the record REFUSE, as opposed to answering wrongly? Only a refusal
            # is a safe hand-back to the model, and the two must never be conflated.
            raw = str(r.get("mpeqs") or r.get("m35") or "")
            refused = ("refused" in raw or "no spec" in raw or raw in ("", "None"))
            items.append({"text": text, "solo": solo, "mpeqs": mp,
                          "refused": refused, "source": fname})
    return items


def score(items, choose):
    right = sent = 0
    for it in items:
        to_machine = choose(it)
        sent += to_machine
        right += it["mpeqs"] if to_machine else it["solo"]
    return right, sent


def main(root="data/custom", out="data/custom/router.json"):
    root = Path(root)
    items = load(root)
    n = len(items)
    if not n:
        print("no stored outcomes found")
        return

    always_solo = sum(i["solo"] for i in items)
    always_mp = sum(i["mpeqs"] for i in items)
    oracle = sum(i["solo"] or i["mpeqs"] for i in items)
    coin = (always_solo + always_mp) / 2
    rule_right, rule_sent = score(items, lambda it: apriori(it["text"]))

    # A fitted variant, leave-one-out so it never scores a problem it learned from.
    keys = ["exact_word", "has_fraction", "big_number", "wide_range", "calendar",
            "many_numbers"]
    loo_right = 0
    for i, it in enumerate(items):
        train = items[:i] + items[i + 1:]
        best, best_gain = None, 0
        for k in keys:
            gain = 0
            for tr in train:
                use_m = features(tr["text"])[k]
                gain += (tr["mpeqs"] if use_m else tr["solo"])
            if gain > best_gain:
                best, best_gain = k, gain
        use_m = features(it["text"])[best] if best else True
        loo_right += it["mpeqs"] if use_m else it["solo"]

    print(f"{n} problems with both arms recorded, from {len(SOURCES)} batteries\n")
    print(f"{'policy':<34}{'right':>7}{'rate':>8}   sent to the machinery")
    print(f"{'always the model':<34}{always_solo:>7}{always_solo / n:>8.2f}   0")
    print(f"{'always the machinery':<34}{always_mp:>7}{always_mp / n:>8.2f}   {n}")
    print(f"{'coin flip (expected)':<34}{coin:>7.0f}{coin / n:>8.2f}   ~{n // 2}")
    print(f"{'the a priori rule':<34}{rule_right:>7}{rule_right / n:>8.2f}   "
          f"{rule_sent}")
    print(f"{'best single feature, leave-one-out':<34}{loo_right:>7}"
          f"{loo_right / n:>8.2f}   -")
    # Machinery first, model only where the RECORD ITSELF declined to answer.
    fallback = sum(i["solo"] if i["refused"] else i["mpeqs"] for i in items)
    handed = sum(1 for i in items if i["refused"])
    print(f"{'machinery, model on refusal':<34}{fallback:>7}{fallback / n:>8.2f}   "
          f"{n - handed}")
    print(f"{'oracle (either arm right)':<34}{oracle:>7}{oracle / n:>8.2f}   -")

    both_wrong = sum(1 for i in items if not i["solo"] and not i["mpeqs"])
    only_solo = sum(1 for i in items if i["solo"] and not i["mpeqs"])
    only_mp = sum(1 for i in items if i["mpeqs"] and not i["solo"])
    print(f"\nwhere they differ: machinery only {only_mp}, model only {only_solo}, "
          f"neither {both_wrong}")
    print("\nA rule written from the law before seeing an outcome, reading nothing but")
    print("the problem text, is the deployable form of everything measured here — and")
    print("what it is worth is the distance between always-one-arm and the oracle.")
    summary = {"n": n, "always_solo": always_solo, "always_mpeqs": always_mp,
               "fallback": fallback, "handed_back": handed,
               "rule_right": rule_right, "rule_sent": rule_sent, "oracle": oracle,
               "loo_right": loo_right, "only_machinery": only_mp,
               "only_model": only_solo, "neither": both_wrong}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
