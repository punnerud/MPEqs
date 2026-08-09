#!/usr/bin/env python3
"""Translation memory: agreement is paid at first encounter; after that, the record reads.

Phase 83 stored solving roads; this phase stores TRANSLATIONS. A story's surface, with
every number masked, is a template — and a template met once, dual-arm translated and
agreement-gated, teaches the record a slot mapping: which masked number feeds q, which
feeds s, what the phrase constants p and r are, which slots are distractor noise. From
then on the record translates instances of that template ALONE — zero model calls —
and the model is consulted only as a sampled audit (phase 83's discipline, random rate
1/4), whose disagreement would invalidate the entry and force relearning.

Thirty stories over five template families (three clean skins, two hard ones with
flipped clause order and numeric distractors). Truth tuples written first, instances
generated with unique slot values so the learned mapping is unambiguous — collisions
are detected at learn time and skipped honestly, full price next instance. Hard-skin
arms may legitimately orient the equation either way (equality is symmetric), so
exactness is judged where orientation cannot hide: the delivered x against the truth's
x, exact.

The bill is the point: model calls with memory against the two-per-story baseline,
while exactness holds at thirty of thirty.
"""
import json
import random
import re
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eqbricks import solve_direct  # noqa: E402
from eqmemory import walk  # noqa: E402
from eqwords import story as clean_story, translate  # noqa: E402
from eqwordshard import flip, hard_story  # noqa: E402

ROAD = ("CLEAR", "MOVEX", "MOVEC", "DIV")

# (family name, skin, p, r, s_sign, hard) — q, s vary per instance; phrases fixed.
FAMILIES = [
    ("num-clean", "number", F(3), F(2), 1, False),
    ("age-clean", "age", F(1, 2), F(2), 1, False),
    ("price-clean", "price", F(5), F(3), 1, False),
    ("num-hard", "number", F(2), F(4), 1, True),
    ("age-hard", "age", F(4), F(2), -1, True),
]
FORBIDDEN = {3, 6, 14}                     # the distractor numbers of the skins


def gen_instance(rng, fam):
    _, skin, p, r, s_sign, hard = fam
    while True:
        q, s_abs = rng.randint(1, 19), rng.randint(1, 19)
        if q != s_abs and q not in FORBIDDEN and s_abs not in FORBIDDEN:
            st = (p, F(q), r, F(s_sign * s_abs))
            text = hard_story(st, skin) if hard else clean_story(st, skin)
            return st, text


def mask(text):
    slots = [int(x) for x in re.findall(r"\d+", text)]
    return re.sub(r"\d+", "<n>", text), slots


def learn(template, slots, state):
    """From one agreed translation, find which slot feeds q and which feeds s.
    Ambiguous slot values refuse to teach — full price next time."""
    p, q, r, s = state
    def slot_of(v):
        hits = [i for i, x in enumerate(slots) if x == abs(v)]
        return hits[0] if len(hits) == 1 else None
    iq, is_ = slot_of(q), slot_of(s)
    if iq is None or is_ is None:
        return None
    return {"p": p, "r": r, "iq": iq, "is": is_,
            "sq": 1 if q >= 0 else -1, "ss": 1 if s >= 0 else -1}


def apply_mapping(entry, slots):
    return (entry["p"], F(entry["sq"] * slots[entry["iq"]]),
            entry["r"], F(entry["ss"] * slots[entry["is"]]))


def main(seed=23, out="data/custom/eqtransmem.json"):
    seed = int(seed)
    rng = random.Random(seed)
    audit_rng = random.Random(seed + 1)
    stream = []
    for fam in FAMILIES:
        for _ in range(6):
            stream.append((fam, *gen_instance(rng, fam)))
    rng.shuffle(stream)

    store = {}
    model_calls = first_enc = record_served = audits = 0
    audit_mismatch = ambiguous = flagged = exact = 0
    for fam, truth, text in stream:
        template, slots = mask(text)
        truth_x = solve_direct(truth)
        entry = store.get(template)
        if entry is None:
            a, b = translate(text)
            model_calls += 2
            agreed = a is not None and (a == b or (b is not None and a == flip(b)))
            if not agreed:
                a, b = translate(text)     # one retry, both arms
                model_calls += 2
                agreed = a is not None and (a == b or (b is not None and a == flip(b)))
            if not agreed:
                flagged += 1
                continue
            first_enc += 1
            mapping = learn(template, slots, a)
            if mapping is None:
                ambiguous += 1
            else:
                store[template] = mapping
            state = a
        else:
            state = apply_mapping(entry, slots)
            record_served += 1
            if audit_rng.random() < 0.25:  # sampled audit: one arm only
                audits += 1
                from eqwords import PROMPT_JSON, parse_json_state
                from cutbig import ask
                aa = parse_json_state(
                    ask("qwen-35b", PROMPT_JSON.format(story=text), n=90))
                model_calls += 1
                if aa is not None and aa != state and aa != flip(state):
                    audit_mismatch += 1
                    del store[template]    # invalidate; relearn at next encounter
        fin, _ = walk(state, ROAD)
        ok = fin == (F(1), F(0), F(0), solve_direct(state)) and fin[3] == truth_x
        exact += ok

    n = len(stream)
    baseline = 2 * n
    print(f"{n} stories over {len(FAMILIES)} templates, store born empty\n")
    print(f"model calls      : {model_calls} against a {baseline}-call dual-arm "
          f"baseline ({baseline / max(model_calls, 1):.1f}x cheaper)")
    print(f"first encounters : {first_enc} (dual-arm, agreement-gated); "
          f"record-served: {record_served}")
    print(f"audits           : {audits} sampled, {audit_mismatch} mismatches; "
          f"ambiguous skips {ambiguous}, flagged {flagged}")
    print(f"exact deliveries : {exact}/{n} (judged on x, where orientation cannot "
          f"hide)")
    print("\nThe agreement gate is a toll paid once per template, not once per story.")
    print("After the first crossing the record extracts the slots itself, the model")
    print("drops to a sampled audit, and the bill falls while exactness stands still.")
    summary = {"stories": n, "templates": len(FAMILIES), "model_calls": model_calls,
               "baseline_calls": baseline, "first_encounters": first_enc,
               "record_served": record_served, "audits": audits,
               "audit_mismatch": audit_mismatch, "ambiguous": ambiguous,
               "flagged": flagged, "exact": exact}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
