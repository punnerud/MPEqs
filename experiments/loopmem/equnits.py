#!/usr/bin/env python3
"""Units join the text pipeline: the model names the quantity, the bricks do the rest.

The user's founding example — "Per kjører i tre miles per sekund, hva blir det i km/t" —
run as the full stated shape. Ten conversion questions are GENERATED from ten truth
triplets (value, from-unit, to-unit), every story opening with a numeric-and-unit
distractor sentence (the phase 85 lesson: hard skin from the start). The model's only
job is to read the language into a typed query, through TWO entry points — a JSON
object and a bare arrow line — and the agreement gate delivers only when both name the
identical exact triplet.

After the gate, no model: phase 73's brick router streams a frontier through type
space (sixteen facts, lifted into compound types on demand), the chain's factor is
exact, and two hand anchors — derived independently in comments below — pin the
arithmetic to the textbook. Errors can enter at the reading alone; the reading is
double-entried; and how often the two entries agree on the same wrong triplet is the
phase's risk number.
"""
import json
import random
import re
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bricks import build_registry, route  # noqa: E402
from cutbig import ask  # noqa: E402

# The truths, before any model call: (value, from, to) — dims as "num/den" strings.
BATTERY = [
    (F(3), "mile/second", "km/hour",
     "Per is 42 years old. Per runs 3 miles per second. How fast is that in "
     "kilometers per hour?"),
    (F(100), "km/hour", "foot/second",
     "The car cost 250000 kroner. It drives at 100 kilometers per hour. How fast is "
     "that in feet per second?"),
    (F(5), "yard/minute", "mm/second",
     "The garden has 12 trees. A snail crawls 5 yards per minute. How fast is that "
     "in millimeters per second?"),
    (F(2), "g/ml", "pound/gallon",
     "The lab is on floor 3. A syrup has a density of 2 grams per milliliter. What "
     "is that in pounds per gallon?"),
    (F(7), "km", "mile",
     "Kari owns 2 bicycles. The trail is 7 kilometers long. How long is that in "
     "miles?"),
    (F(12), "inch", "cm",
     "The shop opens at 9. A screen is 12 inches wide. How wide is that in "
     "centimeters?"),
    (F(90), "minute", "hour",
     "The cinema has 180 seats. The film lasts 90 minutes. How long is that in "
     "hours?"),
    (F(4), "litre", "ml",
     "The kitchen has 6 chairs. A jug holds 4 litres. How much is that in "
     "milliliters?"),
    (F(60), "mile/hour", "foot/second",
     "The road has 4 lanes. The speed limit is 60 miles per hour. What is that in "
     "feet per second?"),
    (F(250), "ml", "gallon",
     "The recipe serves 8 people. It needs 250 milliliters of milk. How much is "
     "that in gallons?"),
]

# Hand anchors, derived independently of the router:
#   3 mile/second: 3 * 1609.344 m/s = 4828.032 m/s; times 3.6 -> 17380.9152 km/h,
#   exactly 10863072/625.
#   100 km/hour: 100000 m / 3600 s; a foot is 0.3048 m -> 100000/(3600*0.3048)
#   = 1000000000/10972800 = 312500/3429 ft/s.
ANCHORS = {0: F(10863072, 625), 1: F(312500, 3429)}

VOCAB = ("km, m, cm, mm, inch, foot, yard, mile, second, minute, hour, day, "
         "g, kg, pound, ounce, litre, ml, gallon")

PROMPT_JSON = """Read this problem. Name the quantity to convert and the target unit.

Problem: {story}

Unit tokens you may use: {vocab}. Write a rate as token/token.
Reply with ONLY this JSON: {{"value": "<number>", "from": "<unit>", "to": "<unit>"}}"""

PROMPT_ARROW = """Read this problem. Name the quantity to convert and the target unit.

Problem: {story}

Unit tokens you may use: {vocab}. Write a rate as token/token.
Reply with ONLY one line: first the number being converted, then its unit, then an
arrow, then the target unit — shaped like: 8 kg -> g"""


def parse_dims(tok):
    parts = tok.strip().split("/")
    if len(parts) == 1:
        return {parts[0].strip(): 1}
    if len(parts) == 2:
        return {parts[0].strip(): 1, parts[1].strip(): -1}
    return None


def parse_json_q(reply):
    m = re.search(r"\{[^{}]*\}", reply, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return (F(str(d["value"]).strip()), d["from"].strip().lower(),
                d["to"].strip().lower())
    except Exception:  # noqa: BLE001
        return None


def parse_arrow_q(reply):
    for line in reply.strip().splitlines():
        m = re.match(r"\s*([0-9./]+)\s+([a-z/]+)\s*->\s*([a-z/]+)\s*$",
                     line.strip().lower())
        if m:
            try:
                return F(m.group(1)), m.group(2), m.group(3)
            except ValueError:
                return None
    return None


def main(n_inspect=0, out="data/custom/equnits.json"):
    n_inspect = int(n_inspect)
    bricks = build_registry()
    if n_inspect:
        for idx in (0, 7):
            _, _, _, story = BATTERY[idx]
            ra = ask("qwen-35b", PROMPT_JSON.format(story=story, vocab=VOCAB), n=80)
            rb = ask("qwen-35b", PROMPT_ARROW.format(story=story, vocab=VOCAB), n=40)
            print(f"story: {story}\n  RAW json:  {ra!r}\n  RAW arrow: {rb!r}\n")
        return

    a_right = b_right = agree = agree_true = agree_wrong = 0
    retries = flagged = routed = anchor_ok = e2e = 0
    rows = []
    for i, (val, src, dst, story) in enumerate(BATTERY):
        truth = (val, src, dst)
        a = parse_json_q(ask("qwen-35b", PROMPT_JSON.format(story=story, vocab=VOCAB),
                             n=80))
        b = parse_arrow_q(ask("qwen-35b", PROMPT_ARROW.format(story=story,
                                                              vocab=VOCAB), n=40))
        a_right += a == truth
        b_right += b == truth
        if a != b or a is None:
            retries += 1
            hint = (f"\n\nTwo readings disagreed: {a} versus {b}. Read again; ignore "
                    f"numbers that are not the quantity being converted.")
            a = parse_json_q(ask("qwen-35b",
                                 PROMPT_JSON.format(story=story, vocab=VOCAB) + hint,
                                 n=80))
            b = parse_arrow_q(ask("qwen-35b",
                                  PROMPT_ARROW.format(story=story, vocab=VOCAB) + hint,
                                  n=40))
        status = "flagged"
        if a == b and a is not None:
            agree += 1
            agree_true += a == truth
            agree_wrong += a != truth
            sd, dd = parse_dims(a[1]), parse_dims(a[2])
            res, _ = route(bricks, sd, dd) if sd and dd else (None, 0)
            if res:
                routed += 1
                ans = a[0] * res[0]
                if i in ANCHORS:
                    anchor_ok += ans == ANCHORS[i]
                truth_res, _ = route(bricks, parse_dims(src), parse_dims(dst))
                e2e += ans == val * truth_res[0]
                status = f"= {ans} ({float(ans):.4f})"
            else:
                status = "agreed, NO ROUTE"
        else:
            flagged += 1
        rows.append({"story": story, "truth": [str(val), src, dst],
                     "a": a and [str(a[0]), a[1], a[2]],
                     "b": b and [str(b[0]), b[1], b[2]], "status": status})
        print(f"{story[:58]:<60} {status}")

    n = len(BATTERY)
    print(f"\ntranslation: JSON arm {a_right}/{n}, arrow arm {b_right}/{n} (first try)")
    print(f"agreement gate: {agree}/{n} delivered ({retries} retries, {flagged} "
          f"flagged); agreed on truth {agree_true}/{agree}, agreed WRONG "
          f"{agree_wrong}/{agree}")
    print(f"routing after the gate: {routed}/{agree} found a chain; hand anchors exact "
          f"{anchor_ok}/{len(ANCHORS)}; end-to-end exact {e2e}/{routed}")
    print("\nThe founding example runs whole: language names the quantity twice, the")
    print("gate compares exact triplets, sixteen facts lift and chain through type")
    print("space, and the number that lands is anchored to the textbook by hand.")
    summary = {"problems": n, "a_right": a_right, "b_right": b_right, "agree": agree,
               "agree_true": agree_true, "agree_wrong": agree_wrong,
               "retries": retries, "flagged": flagged, "routed": routed,
               "anchor_ok": anchor_ok, "e2e": e2e, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
