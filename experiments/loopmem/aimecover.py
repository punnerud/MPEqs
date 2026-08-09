#!/usr/bin/env python3
"""How much of AIME does the solver vocabulary actually reach — and what is missing?

The library grows by measurement, not by taste. Thirty real AIME problems are shown to
the model with the whole catalogue, and it answers in exactly one of two ways: a spec
that maps the problem onto a solver, or a refusal that NAMES the capability the
catalogue lacks. The record then runs every spec it is given and compares to the
published answer.

Three numbers matter, and they mean different things:

  CLAIMED   the model says the problem maps                — reach of the vocabulary
            as the model perceives it
  RAN       the spec validates and executes                — reach as the record
            can honour it
  EXACT     the executed answer equals the published one   — reach that is real

The gap between CLAIMED and EXACT is the mapping's error, not the machines'; the gap
between the catalogue and the named-missing list is the growth plan for the next phase.
Nothing here is a delivery gate — with the published answers in hand this is an oracle
measurement, declared as such, whose purpose is to point at what to build.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aimebudget import ask_work  # noqa: E402
from olympiad import load_problems  # noqa: E402
from solvemap import PREDICATE_HELP, SCHEMAS, answer_of, parse_spec  # noqa: E402
from solvers import run  # noqa: E402
from solvers2 import WORDINGS2, run2  # noqa: E402

# The grown catalogue's schemas — phase 95's additions, written as the model sees them.
SCHEMAS2 = {
    "multisearch": '{"solver":"multisearch","variables":[{"name":"a","from":<int>,'
                   '"to":<int>},...up to 4],"ordering":"strict_increasing" (optional),'
                   '"conditions":["<arithmetic expression in the variables, e.g. '
                   'a*a + b*b == c*c or gcd(a,b) == 1 or digit_sum(n) == 10>",...],'
                   '"objective":"<expression to aggregate, optional>",'
                   '"aggregate":"count"|"sum"|"min"|"max"|"list",'
                   '"post":{"op":"mod","arg":<int>} (optional)}',
    "polynomial": '{"solver":"polynomial","coefficients":[<highest degree first>,...],'
                  '"at":<number, optional: evaluate instead of finding roots>}',
    "geometry": '{"solver":"geometry","kind":"distance_squared"|"polygon_area"|'
                '"collinear"|"circle_through","points":[[x,y],...]} | '
                '{"solver":"geometry","kind":"line_intersection","line1":[a,b,c],'
                '"line2":[a,b,c]}  (a x + b y = c)',
    "modular": '{"solver":"modular","kind":"power","base":<int>,"exponent":<int>,'
               '"modulus":<int>} | {"kind":"inverse","a":<int>,"modulus":<int>} | '
               '{"kind":"order","a":<int>,"modulus":<int>} | {"kind":"totient",'
               '"n":<int>}',
}
EXPR_FUNCS_HELP = ("abs, min, max, gcd, lcm, isqrt, digit_sum, digit_count, "
                   "is_square, is_prime, num_divisors")

MAP_OR_NAME = """Map this competition problem onto ONE solver from the catalogue, or
say what is missing. Do not compute the answer yourself — the executor computes it from
your spec.

Problem: {problem}

Catalogue:
{catalogue}

For the search solver, conditions use these ops: {preds}

If the problem CAN be expressed with one of these solvers, reply with ONLY the JSON
spec. If it cannot, reply with ONLY: {{"none": "<the one capability a solver would
need>"}}"""


def main(n_problems=30, library=1, seed=5, out=None):
    """library 1 = the phase 92 catalogue; 2 = grown with phase 95's additions."""
    n_problems, library, seed = int(n_problems), int(library), int(seed)
    out = out or f"data/custom/aimecover{'' if library == 1 else '2'}.json"
    _, aime = load_problems()
    picks = random.Random(seed).sample(aime, min(n_problems, len(aime)))
    schemas = dict(SCHEMAS) if library == 1 else {**SCHEMAS, **SCHEMAS2}
    execute = run if library == 1 else run2
    catalogue = "\n".join(f"- {v}" for v in schemas.values())
    if library == 2:
        catalogue += ("\n\nExpressions in multisearch may call: "
                      + EXPR_FUNCS_HELP)

    tally = Counter()
    missing, rows = [], []
    for i, (problem, truth) in enumerate(picks):
        reply = ask_work(MAP_OR_NAME.format(problem=problem, catalogue=catalogue,
                                            preds=PREDICATE_HELP), n=1200)
        spec = parse_spec(reply)
        row = {"truth": str(truth), "i": i}
        if spec is None:
            tally["no_reply"] += 1
            row["outcome"] = "unparseable reply"
        elif "none" in spec and "solver" not in spec:
            tally["declined"] += 1
            missing.append(str(spec["none"])[:120])
            row["outcome"] = f"declined: {str(spec['none'])[:70]}"
        else:
            tally["claimed"] += 1
            res, why = execute(spec)
            if res is None:
                tally["refused"] += 1
                row["outcome"] = f"spec refused: {why[:60]}"
            else:
                tally["ran"] += 1
                got = answer_of(res)
                ok = got == str(truth)
                tally["exact" if ok else "wrong"] += 1
                row["outcome"] = f"{got}" + ("  EXACT" if ok else f" != {truth}")
            row["solver"] = spec.get("solver")
        rows.append(row)
        print(f"{i:>3} truth {str(truth):>5}  {row['outcome'][:78]}")

    n = len(picks)
    print(f"\nclaimed a mapping : {tally['claimed']}/{n}  "
          f"(declined {tally['declined']}, unparseable {tally['no_reply']})")
    print(f"spec ran          : {tally['ran']}  (record refused {tally['refused']})")
    print(f"exact vs published: {tally['exact']}  (wrong {tally['wrong']})")
    used = Counter(r.get("solver") for r in rows if r.get("solver"))
    print(f"solvers reached for: {dict(used)}")
    print("\nnamed as missing:")
    for m in missing:
        print(f"  - {m}")
    print("\nThe gap between claimed and exact is the mapping's error; the named-missing")
    print("list is the growth plan. A library grows by what the problems asked for and")
    print("did not get, which is a measurement, not a taste.")
    summary = {"n": n, "library": library, **dict(tally),
               "solvers_used": dict(used),
               "missing": missing, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
