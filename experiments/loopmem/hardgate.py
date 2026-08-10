#!/usr/bin/env python3
"""The gate in the band where MPEqs wins: what can be delivered without ever lying?

Phase 102's scoreboard has a detail worth staring at: on the hard-arithmetic battery the
model alone scores 0 of 20, and it does not abstain once — twenty confident wrong
numbers. MPEqs scores 12, which is better, but six of its answers are also wrong and
nothing in that arm says which. In a band where the user cannot check the arithmetic
themselves (that is what makes it the band), an unmarked wrong answer is the whole
problem.

So the phase 97 gate is applied where it matters most: the model maps each problem
TWICE with different solvers, the record runs both, and delivery requires value-echo
plus agreement between two different machines. Cross-kind agreement with a solo answer
is not available here — the solo answers are all wrong — so this is the two-machine
gate alone, which phase 97 measured as the weaker layer on easy problems and which has
a much better substrate here: on exact arithmetic, two different machines agreeing is
two different computations of the same number.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aimecover import EXPR_FUNCS_HELP, SCHEMAS2, ask_spec  # noqa: E402
from gsmsolve import ARITH_SCHEMA, equal  # noqa: E402
from hardarith import build  # noqa: E402
from solvemap import PREDICATE_HELP, SCHEMAS, answer_of, parse_spec  # noqa: E402
from solvers2 import run2  # noqa: E402
from specgate import norm, value_echo  # noqa: E402

TWO_ROADS = """Map this problem onto the solver catalogue TWICE, using a DIFFERENT
solver each time, so two different machines compute the same answer. Never compute
anything yourself.

Problem: {story}

Catalogue:
{catalogue}

For the search solver, conditions use these ops: {preds}
Expressions may call: {funcs}   (^ is a power; / is exact rational division)

Reply with ONLY: {{"spec_a": <spec>, "spec_b": <spec>}}
If a second, genuinely different mapping is impossible, use {{"spec_a": <spec>,
"spec_b": null}}"""


def main(out="data/custom/hardgate.json"):
    battery = build()
    catalogue = "\n".join(f"- {v}" for v in
                          {"arith": ARITH_SCHEMA, **SCHEMAS, **SCHEMAS2}.values())
    t = {k: 0 for k in ("both_roads", "a_ran", "b_ran", "delivered", "right",
                        "wrong", "flagged", "echo_flag", "disagree", "one_road",
                        "a_right_ungated")}
    rows = []
    for fam, story, truth in battery:
        reply = ask_spec(TWO_ROADS.format(story=story, catalogue=catalogue,
                                          preds=PREDICATE_HELP,
                                          funcs=EXPR_FUNCS_HELP), n=700)
        outer = parse_spec(reply)
        sa = outer.get("spec_a") if isinstance(outer, dict) else None
        sb = outer.get("spec_b") if isinstance(outer, dict) else None
        if isinstance(outer, dict) and sa is None and "solver" in outer:
            sa = outer
        verdict = "flagged"
        if not isinstance(sa, dict):
            t["flagged"] += 1
            verdict = "no spec_a"
        else:
            ra = run2(sa)[0]
            t["a_ran"] += ra is not None
            if ra is not None and equal(answer_of(ra, sa), truth):
                t["a_right_ungated"] += 1
            rb = run2(sb)[0] if isinstance(sb, dict) else None
            t["both_roads"] += isinstance(sb, dict)
            t["b_ran"] += rb is not None
            bad = value_echo(sa, story) + (value_echo(sb, story)
                                           if isinstance(sb, dict) else [])
            if bad:
                t["echo_flag"] += 1
                t["flagged"] += 1
                verdict = f"echo {bad[:2]}"
            elif ra is None or rb is None:
                t["one_road"] += 1
                t["flagged"] += 1
                verdict = "only one road ran"
            elif norm(answer_of(ra, sa)) != norm(answer_of(rb, sb)):
                t["disagree"] += 1
                t["flagged"] += 1
                verdict = "roads disagree"
            else:
                t["delivered"] += 1
                ok = equal(answer_of(ra, sa), truth)
                t["right" if ok else "wrong"] += 1
                verdict = f"DELIVERED {str(answer_of(ra, sa))[:18]}" + \
                    ("" if ok else " WRONG")
        rows.append({"family": fam, "truth": truth, "verdict": verdict,
                     "story": story, "spec_a": sa, "spec_b": sb})
        print(f"{fam:<10}{verdict[:44]:<46}{story[:38]}")

    n = len(battery)
    print(f"\ntwo roads offered {t['both_roads']}/{n}; road A ran {t['a_ran']}, "
          f"road B ran {t['b_ran']}")
    print(f"gate: delivered {t['delivered']} (right {t['right']}, WRONG {t['wrong']}), "
          f"flagged {t['flagged']}")
    print(f"  flags: echo {t['echo_flag']}, only one road {t['one_road']}, "
          f"disagree {t['disagree']}")
    print(f"ungated road A alone: {t['a_right_ungated']} right of {t['a_ran']} that ran")
    print(f"the model answering alone on this battery (phase 102): 0 right, 20 "
          f"confident wrong, 0 abstentions")
    print("\nIn a band where the reader cannot check the arithmetic, an unmarked wrong")
    print("answer is the entire problem. What a gate is worth is measured against that,")
    print("not against a scoreboard.")
    summary = {"n": n, **t, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
