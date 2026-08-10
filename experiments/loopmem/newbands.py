#!/usr/bin/env python3
"""Two new classes in the winning band: exact probability and unit conversion.

Phase 104 drew MPEqs' domain — the model wins on grade-school sums, the machinery wins
on hard exact arithmetic — and the obvious way to make the tool solve more problems is
to widen the winning band rather than to keep pushing at competition mathematics, where
the mapper is the measured bottleneck.

Two classes go in, both chosen because a model fails them for the SAME reason it fails
the hard-arithmetic battery (the answer is an exact ratio it cannot hold) while the
mapping stays one sentence long:

  PROBABILITY  a uniform space described by variables and constraints, an event
               described by more, and the answer as a Fraction — never a decimal,
               because the asked-for answer is m/n or "one in seven"
  CONVERT      the phase 73 brick router, finally a library member: sixteen conversion
               facts lifted through compound type space, exact, refusing when no path
               exists

Twenty problems, ten of each, truths computed before any model call, same two arms as
phase 102: the model answering alone against the model mapping onto exact machines.
"""
import json
import math
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aimecover import EXPR_FUNCS_HELP, SCHEMAS2  # noqa: E402
from cutbig import ask  # noqa: E402
from gsmsolve import ARITH_SCHEMA, ask_spec_model, equal  # noqa: E402
from olympiad import SOLO, last_number  # noqa: E402
from solvemap import PREDICATE_HELP, SCHEMAS, answer_of, parse_spec  # noqa: E402
from solvers2 import parse_units, run2  # noqa: E402

SCHEMA_NEW = {
    "probability": '{"solver":"probability","variables":[{"name":"a","from":<int>,'
                   '"to":<int>},...],"conditions":["<constraints defining the sample '
                   'space, optional>"],"event":["<constraints defining the event>"],'
                   '"report":"m_plus_n" (optional)}',
    "convert": '{"solver":"convert","value":<number>,"from":"<unit>","to":"<unit>"}  '
               'units like m, km, cm, mm, inch, foot, yard, mile, second, minute, '
               'hour, day, week, g, kg, pound, ounce, litre, ml, gallon; rates as '
               'unit/unit, powers as unit^2',
}

FEWSHOT = """Map the problem onto ONE solver and fill its slots. Do NOT compute the
answer — an exact executor computes it from your spec.

Example: "Two fair dice are rolled. What is the probability the sum is 9?"
Spec: {{"solver":"probability","variables":[{{"name":"a","from":1,"to":6}},
{{"name":"b","from":1,"to":6}}],"event":["a + b == 9"]}}

Example: "A car travels 3 miles per second. How fast is that in kilometers per hour?"
Spec: {{"solver":"convert","value":3,"from":"mile/second","to":"km/hour"}}

Example: "Start with 5. Twelve times in a row, multiply by 2/3 and add 1/4."
Spec: {{"solver":"iterate","init":"5","step":"acc * 2/3 + 1/4","from":1,"to":12}}

Catalogue:
{catalogue}

For the search solver, conditions use these ops: {preds}
Expressions may call: {funcs}   (^ is a power; / is exact rational division)

Problem: {story}
Spec:"""


def build():
    """Truths first, computed here."""
    out = []

    def prob(nvars, lo, hi, event, text, ordering=None):
        import itertools
        space = [t for t in itertools.product(range(lo, hi + 1), repeat=nvars)
                 if not ordering or all(t[i] < t[i + 1] for i in range(len(t) - 1))]
        fav = [t for t in space if event(*t)]
        return ("probability", text, str(F(len(fav), len(space))))

    out.append(prob(2, 1, 6, lambda a, b: a + b == 9,
                    "Two fair six-sided dice are rolled. What is the exact probability "
                    "that the sum is 9? Give the answer as a fraction."))
    out.append(prob(3, 1, 6, lambda a, b, c: a + b + c == 10,
                    "Three fair six-sided dice are rolled. What is the exact "
                    "probability that the sum is 10? Give the answer as a fraction."))
    out.append(prob(2, 1, 12, lambda a, b: math.gcd(a, b) == 1,
                    "Two numbers are picked independently and uniformly from 1 to 12. "
                    "What is the exact probability they are coprime? Give a fraction."))
    out.append(prob(2, 1, 20, lambda a, b: (a * b) % 4 == 0,
                    "Two numbers are picked independently and uniformly from 1 to 20. "
                    "What is the exact probability their product is divisible by 4? "
                    "Give a fraction."))
    out.append(prob(3, 1, 5, lambda a, b, c: a < b < c,
                    "Three numbers are picked independently and uniformly from 1 to 5. "
                    "What is the exact probability they come out strictly increasing? "
                    "Give a fraction."))
    out.append(prob(2, 1, 10, lambda a, b: abs(a - b) <= 2,
                    "Two numbers are picked independently and uniformly from 1 to 10. "
                    "What is the exact probability they differ by at most 2? Give a "
                    "fraction."))
    out.append(prob(4, 0, 1, lambda a, b, c, d: a + b + c + d == 2,
                    "Four fair coins are flipped, each 0 or 1. What is the exact "
                    "probability that exactly two show 1? Give a fraction."))
    out.append(prob(2, 1, 8, lambda a, b: (a + b) % 3 == 0,
                    "Two numbers are picked independently and uniformly from 1 to 8. "
                    "What is the exact probability their sum is divisible by 3? Give a "
                    "fraction."))
    out.append(prob(3, 1, 4, lambda a, b, c: a * b * c % 2 == 1,
                    "Three numbers are picked independently and uniformly from 1 to 4. "
                    "What is the exact probability their product is odd? Give a "
                    "fraction."))
    out.append(prob(2, 1, 15, lambda a, b: a * a + b * b > 200,
                    "Two numbers are picked independently and uniformly from 1 to 15. "
                    "What is the exact probability that the sum of their squares "
                    "exceeds 200? Give a fraction."))

    from bricks import build_registry, route
    reg = build_registry()

    def conv(value, src, dst, text):
        res, _ = route(reg, parse_units(src), parse_units(dst))
        return ("convert", text, str(F(str(value)) * res[0]))

    out.append(conv(3, "mile/second", "km/hour",
                    "A probe moves at 3 miles per second. Exactly how fast is that in "
                    "kilometres per hour? Give the exact value as a fraction."))
    out.append(conv(100, "km/hour", "foot/second",
                    "A car drives at 100 kilometres per hour. Exactly how fast is that "
                    "in feet per second? Give the exact value as a fraction."))
    out.append(conv(7, "gallon", "litre",
                    "A tank holds 7 gallons. Exactly how many litres is that? Give the "
                    "exact value as a fraction."))
    out.append(conv(5, "mile/hour", "m/second",
                    "A runner moves at 5 miles per hour. Exactly how many metres per "
                    "second is that? Give the exact value as a fraction."))
    out.append(conv(12, "ounce", "g",
                    "A parcel weighs 12 ounces. Exactly how many grams is that? Give "
                    "the exact value as a fraction."))
    out.append(conv(3, "yard/minute", "mm/second",
                    "A snail crawls 3 yards per minute. Exactly how many millimetres "
                    "per second is that? Give the exact value as a fraction."))
    out.append(conv(2, "g/ml", "pound/gallon",
                    "A syrup has density 2 grams per millilitre. Exactly what is that "
                    "in pounds per gallon? Give the exact value as a fraction."))
    out.append(conv(40, "mile/hour^2", "m/second^2",
                    "A car accelerates at 40 miles per hour squared. Exactly what is "
                    "that in metres per second squared? Give the exact value as a "
                    "fraction."))
    out.append(conv(6, "week", "hour",
                    "A voyage lasts 6 weeks. Exactly how many hours is that?"))
    out.append(conv(9, "inch", "mm",
                    "A screen is 9 inches wide. Exactly how many millimetres is that? "
                    "Give the exact value as a fraction."))
    return out


def main(out="data/custom/newbands.json"):
    battery = build()
    catalogue = "\n".join(f"- {v}" for v in
                          {"arith": ARITH_SCHEMA, **SCHEMA_NEW, **SCHEMAS,
                           **SCHEMAS2}.values())
    t = {k: 0 for k in ("solo", "mpeqs", "parsed", "ran", "wrong")}
    byfam = {}
    rows = []
    for fam, story, truth in battery:
        a_solo = last_number(ask("qwen-35b", SOLO.format(problem=story), n=420))
        solo_ok = a_solo is not None and equal(a_solo, truth)
        t["solo"] += solo_ok

        spec = parse_spec(ask_spec_model(
            "qwen-35b", FEWSHOT.format(story=story, catalogue=catalogue,
                                       preds=PREDICATE_HELP,
                                       funcs=EXPR_FUNCS_HELP), n=420))
        got, mp_ok = None, False
        if isinstance(spec, dict) and "solver" in spec:
            t["parsed"] += 1
            res, why = run2(spec)
            if res is not None:
                t["ran"] += 1
                got = answer_of(res, spec)
                mp_ok = equal(got, truth)
                t["mpeqs"] += mp_ok
                t["wrong"] += not mp_ok
            else:
                got = why[:40]
        f = byfam.setdefault(fam, [0, 0, 0])
        f[0] += 1
        f[1] += solo_ok
        f[2] += mp_ok
        rows.append({"family": fam, "story": story[:70], "truth": truth,
                     "solo": str(a_solo), "solo_ok": bool(solo_ok),
                     "mpeqs": str(got), "mpeqs_ok": bool(mp_ok), "spec": spec})
        print(f"{fam:<12}{'solo ok' if solo_ok else 'solo X '} "
              f"{'mpeqs ok' if mp_ok else 'mpeqs X '}  {truth[:20]:<22}{story[:40]}")

    n = len(battery)
    print(f"\nSOLO-35B : {t['solo']}/{n}")
    print(f"MPEqs    : {t['mpeqs']}/{n}  (parsed {t['parsed']}, ran {t['ran']}, "
          f"wrong {t['wrong']})")
    for fam, (cnt, so, mp) in byfam.items():
        print(f"  {fam:<12}{cnt:>3}{so:>6}{mp:>7}")
    print("\nWidening the band is the cheap direction: both classes fail a model for the")
    print("reason the hard-arithmetic battery does — the answer is an exact ratio it")
    print("cannot hold — while the mapping stays one sentence, which is the side of the")
    print("pipeline that works.")
    summary = {"n": n, **t, "byfam": byfam, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
