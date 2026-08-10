#!/usr/bin/env python3
"""Does the catalogue scale? Twenty-seven problems, nine classes, one prompt.

Every class so far was measured in its own run, with its own worked examples and a
catalogue that happened to contain the machine it needed. That is the friendly case. The
question this asks is the deployment one: with TWENTY-SIX solvers in the prompt and
three GENERIC examples that match no class in particular, can the model still pick the
right machine — or does a growing library eat its own gains?

Twenty-seven problems, three from each of nine classes, all drawn from batteries whose
truths were computed before any model saw them. Per-class accuracy is compared against
the focused runs, and the confound is declared: those runs showed class-matched worked
examples and this one does not, so a drop mixes two causes and only a NON-drop is clean
evidence.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aimecover import EXPR_FUNCS_HELP, SCHEMAS2  # noqa: E402
from bands2 import SCHEMA_MORE  # noqa: E402
from bands2 import build as build_b2  # noqa: E402
from cutbig import ask  # noqa: E402
from gsmsolve import ARITH_SCHEMA, ask_spec_model, equal  # noqa: E402
from hardarith import build as build_hard  # noqa: E402
from newbands import SCHEMA_NEW  # noqa: E402
from newbands import build as build_nb  # noqa: E402
from olympiad import SOLO, last_number  # noqa: E402
from solvemap import PREDICATE_HELP, SCHEMAS, answer_of, parse_spec  # noqa: E402
from solvers2 import run2  # noqa: E402

GENERIC = """Map the problem onto ONE solver from the catalogue and fill its slots. Do
NOT compute the answer — an exact executor computes it from your spec.

Example: "A shop buys 14 crates of 24 bottles and breaks 17. How many are left?"
Spec: {{"solver":"arith","let":{{"total":"14*24"}},"answer":"total - 17"}}

Example: "How many integers from 1 to 900 are divisible by 11?"
Spec: {{"solver":"search","domain":{{"kind":"range","from":1,"to":900}},
"conditions":[{{"op":"divisible_by","arg":11}}],"aggregate":"count"}}

Example: "Two numbers x and y satisfy x + y = 30 and x - y = 4. Find x and y."
Spec: {{"solver":"linear_system","rows":[[1,1],[1,-1]],"rhs":[30,4]}}

Catalogue:
{catalogue}

For the search solver, conditions use these ops: {preds}
Expressions may call: {funcs}   (^ is a power; / is exact rational division)

Problem: {story}
Spec:"""

# Per-class results from the focused runs, for the comparison.
FOCUSED = {"fractions": 4, "big": 4, "count": 4, "divisors": 5, "probability": 9,
           "convert": 8, "statistics": 6, "datetime": 6, "finance": 6}
FOCUSED_N = {"fractions": 5, "big": 5, "count": 5, "divisors": 5, "probability": 10,
             "convert": 10, "statistics": 6, "datetime": 6, "finance": 6}


def main(per_class=3, out="data/custom/mixedband.json"):
    per_class = int(per_class)
    pool = {}
    for fam, story, truth in (build_hard(1) + build_nb() + build_b2()):
        pool.setdefault(fam, []).append((story, str(truth)))
    battery = [(fam, s, t) for fam, items in pool.items()
               for s, t in items[:per_class]]

    catalogue = "\n".join(f"- {v}" for v in
                          {"arith": ARITH_SCHEMA, **SCHEMA_MORE, **SCHEMA_NEW,
                           **SCHEMAS, **SCHEMAS2}.values())
    n_solvers = len(catalogue.split("\n"))

    t = {k: 0 for k in ("solo", "mpeqs", "parsed", "ran", "wrong")}
    byfam = {}
    rows = []
    for fam, story, truth in battery:
        raw = ask("qwen-35b", SOLO.format(problem=story), n=420)
        num = last_number(raw)
        solo_ok = (equal(num, truth) if num is not None else False) or \
            (not str(truth).replace("-", "").isdigit() and str(truth) in raw)
        t["solo"] += solo_ok

        spec = parse_spec(ask_spec_model(
            "qwen-35b", GENERIC.format(story=story, catalogue=catalogue,
                                       preds=PREDICATE_HELP,
                                       funcs=EXPR_FUNCS_HELP), n=420))
        got, ok = None, False
        if isinstance(spec, dict) and "solver" in spec:
            t["parsed"] += 1
            res, why = run2(spec)
            if res is not None:
                t["ran"] += 1
                got = answer_of(res, spec)
                ok = equal(got, truth) or str(got) == str(truth)
                t["mpeqs"] += ok
                t["wrong"] += not ok
            else:
                got = why[:36]
        f = byfam.setdefault(fam, [0, 0, 0])
        f[0] += 1
        f[1] += solo_ok
        f[2] += ok
        rows.append({"family": fam, "truth": truth, "solo_ok": bool(solo_ok),
                     "mpeqs": str(got), "mpeqs_ok": bool(ok),
                     "solver": (spec or {}).get("solver")})
        print(f"{fam:<12}{'solo ok' if solo_ok else 'solo X '} "
              f"{'mpeqs ok' if ok else 'mpeqs X '} {str((spec or {}).get('solver')):<14}"
              f"{story[:38]}")

    n = len(battery)
    print(f"\ncatalogue shown: {n_solvers} solver lines, examples matched no class")
    print(f"SOLO-35B : {t['solo']}/{n}")
    print(f"MPEqs    : {t['mpeqs']}/{n}  (parsed {t['parsed']}, ran {t['ran']}, "
          f"wrong {t['wrong']})")
    print(f"\n{'class':<13}{'n':>2}{'solo':>6}{'mixed':>7}{'focused rate':>15}")
    for fam, (cnt, so, mp) in sorted(byfam.items()):
        rate = f"{FOCUSED.get(fam, 0)}/{FOCUSED_N.get(fam, 0)}"
        print(f"{fam:<13}{cnt:>2}{so:>6}{mp:>7}{rate:>15}")
    print("\nA library that cannot be addressed at scale is a library that shrinks as it")
    print("grows. What this measures is whether the twenty-sixth machine costs the first")
    print("twenty-five anything — and the honest reading is asymmetric, since a drop")
    print("here mixes catalogue size with the loss of class-matched examples.")
    summary = {"n": n, "solver_lines": n_solvers, **t, "byfam": byfam,
               "focused": FOCUSED, "focused_n": FOCUSED_N, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
