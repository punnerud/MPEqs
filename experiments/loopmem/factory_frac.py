#!/usr/bin/env python3
"""The fractional families through the Fraction side of the factory.

Phase 78 drew the line by measurement: integer-factor PySpell bricks mount four for four,
fractional ones die on f64's promises — `/` yields Float, Int(1) does not round-trip. The
continuation is not to fight f64 but to change what the model authors: a scale brick IS
its rational factor, so the model names the exact fraction — {"num": 1852, "den": 1000} —
and the record builds the brick in the Fraction registry where the routing already lives.
The phase 45 shape again: the model fills a parameter, the record owns the machinery.

The judge with perfect ground truth is embedded in each task's own text: "forward(2) =
3.704" is a verification sample the record checks EXACTLY (2 * num/den == 3704/1000, as
fractions). The pre-written truth table stays as the experiment's meta-judge, reported
separately — a brick can satisfy its example and still not be the intended transform, and
only the table sees that.

The seven families f64 refused, plus growth accounting on the merged registry: sek and dkk
join the currency block, so the roads-per-pair audit capacity of phase 74 finally moves —
counted, as always, not claimed.
"""
import json
import re
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bricks2 import ABrick, all_roads, registry2  # noqa: E402
from cutbig import ask  # noqa: E402
from factory_scale import count_reach  # noqa: E402

# (name, task text with its embedded example, src, dst, pre-written truth factor)
FAMILIES = [
    ("tax", "adds 25% tax: forward(400) = 500", "net", "gross", F(5, 4)),
    # Truth corrected: the first table held 1852/500 = 3.704 — the EXAMPLE'S y-value,
    # not the factor. The factor is 1.852 = 463/250, which the model named correctly and
    # my table flagged as a mismatch. The mechanical example-judge beat the hand-written
    # truth table, twice in one run (see the demo anchor below for the other).
    ("nmi", "nautical miles to km: forward(2) = 3.704", "nmi", "km", F(463, 250)),
    ("stone", "stone to pounds: forward(2) = 28", "stone", "pound", F(14)),
    ("pint", "UK pints to litres: forward(1000) = 568.26125", "pint", "litre",
     F(56826125, 100000000)),
    ("sek", "Swedish kronor to NOK at rate 0.97: forward(100) = 97", "sekr", "nok",
     F(97, 100)),
    ("dkk", "Danish kroner to NOK at rate 1.52: forward(100) = 152", "dkk", "nok",
     F(152, 100)),
    ("markup", "adds 40% markup: forward(10) = 14", "cost", "price", F(7, 5)),
]

NAME_THE_FRACTION = """A conversion multiplies by one exact fraction. Name it.

Task: {task}

Reply with only JSON: {{"num": <integer>, "den": <integer>}} so that the conversion is
exactly multiplication by num/den. Make the example in the task come out exactly.
{hint}"""

EXAMPLE = re.compile(r"forward\((\d+(?:\.\d+)?)\)\s*=\s*(\d+(?:\.\d+)?)")


def main(out="data/custom/factory_frac.json"):
    base = registry2()
    reach_before = count_reach(base)[1]
    roads_before = len(all_roads(base, {"nok": 1}, {"eur": 1}, max_len=3))

    mounted = []
    rows = []
    example_pass = truth_pass = 0
    for name, task, src_u, dst_u, truth in FAMILIES:
        m = EXAMPLE.search(task)
        x, y = F(m.group(1)), F(m.group(2).replace(",", ""))
        hint = ""
        row = {"family": name, "truth": str(truth)}
        for attempt in range(2):
            reply = ask("qwen-35b", NAME_THE_FRACTION.format(task=task, hint=hint), n=120)
            jm = re.search(r'\{[^{}]*"num"[^{}]*\}', reply)
            if not jm:
                hint = "\nYour last reply was not the JSON asked for."
                continue
            try:
                d = json.loads(jm.group(0))
                factor = F(int(d["num"]), int(d["den"]))
            except Exception:  # noqa: BLE001
                hint = "\nnum and den must be plain integers."
                continue
            # The task's own example is the judge, exact: x * factor must equal y.
            if x * factor != y:
                hint = (f"\nRefused: {d['num']}/{d['den']} times {x} is {x * factor}, "
                        f"but the task says {y}. Name the fraction that fits exactly.")
                continue
            example_pass += 1
            ok_truth = factor == truth
            truth_pass += ok_truth
            row.update({"stage": "mounted", "factor": str(factor),
                        "attempt": attempt, "matches_truth": ok_truth})
            mounted.append(ABrick(f"{src_u}->{dst_u}", {src_u: 1}, {dst_u: 1}, factor))
            mounted.append(ABrick(f"{dst_u}->{src_u}", {dst_u: 1}, {src_u: 1},
                                  1 / factor))
            break
        else:
            row["stage"] = "refused twice"
        rows.append(row)
        print(f"{name:<8} {row['stage']:<14} "
              + (f"factor {row.get('factor')}, truth match {row.get('matches_truth')}"
                 if row["stage"] == "mounted" else ""))

    merged = base + mounted
    reach_after = count_reach(merged)[1]
    roads_after = len(all_roads(merged, {"nok": 1}, {"eur": 1}, max_len=3))
    # And one compound route through a fresh fractional brick, exact against hand truth.
    demo = None
    if any(b.name == "nmi->km" for b in mounted):
        f_nmi = next(b.t[0] for b in mounted if b.name == "nmi->km")
        # Anchor corrected: 10 nmi is 18.52 km = 1852/100; the first anchor said 1852/50,
        # which is twenty nautical miles.
        demo = {"ten_nmi_km": str(F(10) * f_nmi), "exact": F(10) * f_nmi == F(1852, 100)}
        print(f"\n10 nmi = {F(10) * f_nmi} km, exact: {demo['exact']}")

    print(f"\nexample-judge passed {example_pass}/{len(FAMILIES)}, "
          f"truth table agrees {truth_pass}/{example_pass}")
    print(f"registry: {len(base)} -> {len(merged)} bricks; routable pairs "
          f"{reach_before} -> {reach_after}; roads nok->eur {roads_before} -> "
          f"{roads_after}")
    print("\nThe model names the fraction, the task's own example judges it exactly, and")
    print("the Fraction registry does what f64 could not promise. The line phase 78 drew")
    print("was never about what the model can author — it was about which engine should")
    print("hold the result.")
    summary = {"families": len(FAMILIES), "example_pass": example_pass,
               "truth_pass": truth_pass, "bricks_before": len(base),
               "bricks_after": len(merged), "reach_before": reach_before,
               "reach_after": reach_after, "roads_before": roads_before,
               "roads_after": roads_after, "demo": demo, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
