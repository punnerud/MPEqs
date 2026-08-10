#!/usr/bin/env python3
"""Teaching the mapping: three worked hard mappings, then the same fifteen AIME problems.

Phase 99 named the bottleneck by separating it from everything else: five of these
fifteen problems ARE expressible in the library, and the model mapped none of them. The
question left is whether that is a ceiling of the model or a gap in what it was shown.
So it is shown — three worked mappings of the KIND these problems need, none of them
drawn from the test set (they are invented for the purpose, so nothing leaks):

  a bit-pattern enumeration  colourings as the bits of one integer
  a two-variable Diophantine  minimise an objective subject to an exact equation
  an objective sum with a post-op  aggregate an expression over a range, then mod

This is not a retry — no problem is asked twice, nothing is repaired after feedback.
It is the same single call with a better prompt, which is the only lever left once
choosing, filling and executing have each been measured separately.

Scored against phase 99's hand-mapped ceiling of 5/15 and phase 94's measured 0.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aimecover import EXPR_FUNCS_HELP, SCHEMAS2, ask_spec  # noqa: E402
from aimeceiling import HAND_SPECS  # noqa: E402
from olympiad import load_problems  # noqa: E402
from solvemap import PREDICATE_HELP, SCHEMAS, answer_of, parse_spec  # noqa: E402
from solvers2 import run2  # noqa: E402

TEACH = """Map this competition problem onto ONE solver, or say what is missing. An
exact executor runs your spec — never compute the answer yourself.

Three worked mappings of the kind these problems need:

1. "Each vertex of a regular hexagon is painted black or white. How many paintings have
no two opposite vertices both black?" — a painting is the bits of one integer, and each
condition is arithmetic on those bits:
{{"solver":"multisearch","variables":[{{"name":"c","from":0,"to":63}}],
"conditions":["not ((c//2**0)%2 == 1 and (c//2**3)%2 == 1)",
"not ((c//2**1)%2 == 1 and (c//2**4)%2 == 1)",
"not ((c//2**2)%2 == 1 and (c//2**5)%2 == 1)"],"aggregate":"count"}}

2. "Positive integers x and y satisfy 3x + 5y = 100. What is the least possible value of
x + y?" — unknowns become variables with bounds, the relation becomes a condition, the
question becomes the objective:
{{"solver":"multisearch","variables":[{{"name":"x","from":1,"to":100}},
{{"name":"y","from":1,"to":100}}],"conditions":["3*x + 5*y == 100"],
"objective":"x + y","aggregate":"min"}}

3. "Find the remainder when the sum of k(k+1)/2 for k = 1 to 50 is divided by 1000." —
an expression aggregated over a range, with the remainder as a post-op:
{{"solver":"multisearch","variables":[{{"name":"k","from":1,"to":50}}],
"objective":"(k*(k+1))//2","aggregate":"sum","post":{{"op":"mod","arg":1000}}}}

Catalogue:
{catalogue}

For the search solver, conditions use these ops: {preds}
Expressions may call: {funcs}

Problem: {problem}

Reply with ONLY the JSON spec, or ONLY {{"none":"<the one capability that is missing>"}}"""


def main(out="data/custom/aimefewshot.json"):
    _, aime = load_problems()
    picks = random.Random(5).sample(aime, 30)[:15]
    catalogue = "\n".join(f"- {v}" for v in {**SCHEMAS, **SCHEMAS2}.values())
    reachable = set(HAND_SPECS)

    t = Counter()
    rows = []
    for i, (problem, truth) in enumerate(picks):
        spec = parse_spec(ask_spec(TEACH.format(problem=problem, catalogue=catalogue,
                                                preds=PREDICATE_HELP,
                                                funcs=EXPR_FUNCS_HELP), n=900))
        row = {"i": i, "truth": str(truth), "reachable": i in reachable}
        if spec is None:
            t["no_reply"] += 1
            row["outcome"] = "unparseable"
        elif "none" in spec and "solver" not in spec:
            t["declined"] += 1
            t["declined_reachable" if i in reachable else "declined_correctly"] += 1
            row["outcome"] = f"declined: {str(spec['none'])[:60]}"
        else:
            t["claimed"] += 1
            res, why = run2(spec)
            if res is None:
                t["refused"] += 1
                row["outcome"] = f"refused: {why[:50]}"
            else:
                got = answer_of(res)
                ok = str(got) == str(truth)
                t["ran"] += 1
                t["exact" if ok else "wrong"] += 1
                if ok and i in reachable:
                    t["exact_reachable"] += 1
                row["outcome"] = f"{got}" + (" EXACT" if ok else f" != {truth}")
            row["solver"] = spec.get("solver")
            row["spec"] = spec
        rows.append(row)
        mark = "*" if i in reachable else " "
        print(f"{mark}[{i:>2}] truth {str(truth):>5}  {row['outcome'][:70]}")

    n = len(picks)
    print(f"\nclaimed {t['claimed']}/{n}, ran {t['ran']}, EXACT {t['exact']} "
          f"(wrong {t['wrong']}, refused {t['refused']}, declined {t['declined']}, "
          f"unparseable {t['no_reply']})")
    print(f"of the {len(reachable)} problems phase 99 proved reachable: "
          f"{t['exact_reachable']} solved, {t['declined_reachable']} declined")
    print(f"of the {n - len(reachable)} genuinely out of reach: "
          f"{t['declined_correctly']} correctly declined")
    print(f"\nbaselines: phase 94 (no examples) 0 exact; phase 99 hand-mapped ceiling "
          f"{len(reachable)}")
    print("Three worked mappings are the whole intervention — no retry, no feedback,")
    print("no second look at any problem. Whatever moved, moved because the model was")
    print("shown the SHAPE of the work rather than told the vocabulary of it.")
    summary = {"n": n, **dict(t), "reachable": sorted(reachable), "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
