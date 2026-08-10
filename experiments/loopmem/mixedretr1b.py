#!/usr/bin/env python3
"""Can the 1B map when the exemplar is fetched for it? The same twenty-seven, at 1B.

Phase 84 measured OLMoE-1B at the parrot line on one-bit reads and phase 98 saw it parse
sixteen specs of thirty while only three executed. Both were before retrieval existed.
The 1B's failure was never arithmetic — the record does that — it was knowing what to
write, and a retrieved exemplar is precisely a template of what to write, in the shape
of this problem, with a machine that fits.

So the question is asked once more with everything the pipeline now has: same twenty-
seven problems, same catalogue, same bank, same two nearest exemplars, the JSON brace
prefilled, and OLMoE-1B in the seat where the 35B scored 25 of 27.

The mixed run showed a growing catalogue holds for most classes and loses exactly
the ones whose worked example disappeared — fractions 0 of 3 without the fold
example, probability 1 of 3 without the probability one. A catalogue LINE is not
enough for a machine the model has never seen used.

Phase 101 measured the fix on AIME and found retrieval perfect (the right shape
reached the model 5 times of 5) even where the mapper could not use it. Here the
mapper CAN use it, so this is the test that matters: one exemplar per class in a
bank, embedded with numbers masked, the two nearest retrieved per problem, and
everything else identical to the mixed run — same problems, same catalogue, same
model, same generic instruction.
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
from embednav import embed  # noqa: E402
from mapmemory import mask  # noqa: E402
from solvers2 import run2  # noqa: E402

# One invented exemplar per class — the library's memory of how each machine gets used.
BANK = [
    ("fractions", "Start with 4/9. Seven times in a row, multiply by 3/5 and add 1/6. "
     "Give the exact result.",
     '{"solver":"iterate","init":"4/9","step":"acc * 3/5 + 1/6","from":1,"to":7}'),
    ("big", "What is 314159265 times 271828182?",
     '{"solver":"arith","let":{"p":"314159265 * 271828182"},"answer":"p"}'),
    ("count", "How many integers from 1 to 90000 are divisible by 19 and leave "
     "remainder 2 when divided by 7?",
     '{"solver":"search","domain":{"kind":"range","from":1,"to":90000},'
     '"conditions":[{"op":"divisible_by","arg":19},{"op":"mod_eq","arg":[7,2]}],'
     '"aggregate":"count"}'),
    ("divisors", "How many positive divisors does 14 factorial have?",
     '{"solver":"factor","k":14,"report":"divisor_count"}'),
    ("probability", "Two fair eight-sided dice are rolled. What is the exact "
     "probability that the sum is 7?",
     '{"solver":"probability","variables":[{"name":"a","from":1,"to":8},'
     '{"name":"b","from":1,"to":8}],"event":["a + b == 7"]}'),
    ("convert", "A boat moves at 12 yards per minute. Exactly how many centimetres "
     "per second is that?",
     '{"solver":"convert","value":12,"from":"yard/minute","to":"cm/second"}'),
    ("statistics", "What is the exact sample variance of 6, 11, 3, 14 and 9?",
     '{"solver":"statistics","values":[6,11,3,14,9],"report":"sample_variance"}'),
    ("datetime", "How many days are there from 3 May 1985 to 19 November 2011?",
     '{"solver":"datetime","kind":"days_between","from":"1985-05-03",'
     '"to":"2011-11-19"}'),
    ("finance", "A price of 640 falls 15 percent twice and then rises 5 percent. What "
     "is it exactly now?",
     '{"solver":"finance","kind":"percent_chain","start":640,'
     '"changes":[-15,-15,5]}'),
]

GENERIC = """Map the problem onto ONE solver from the catalogue and fill its slots. Do
NOT compute the answer — an exact executor computes it from your spec.

{examples}

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


def main(per_class=3, k=2, model="olmoe-1b",
         out="data/custom/mixedretr_1b.json"):
    per_class, k = int(per_class), int(k)
    pool = {}
    for fam, story, truth in (build_hard(1) + build_nb() + build_b2()):
        pool.setdefault(fam, []).append((story, str(truth)))
    battery = [(fam, s, t) for fam, items in pool.items()
               for s, t in items[:per_class]]

    catalogue = "\n".join(f"- {v}" for v in
                          {"arith": ARITH_SCHEMA, **SCHEMA_MORE, **SCHEMA_NEW,
                           **SCHEMAS, **SCHEMAS2}.values())
    n_solvers = len(catalogue.split("\n"))

    bvecs = embed([mask(p) for _tag, p, _s in BANK])
    pvecs = embed([mask(st) for _f, st, _t in battery])
    t = {key: 0 for key in ("solo", "mpeqs", "parsed", "ran", "wrong",
                            "retrieval_hit")}
    byfam = {}
    rows = []
    for idx, (fam, story, truth) in enumerate(battery):
        sims = [sum(a * b for a, b in zip(pvecs[idx], bv)) for bv in bvecs]
        order = sorted(range(len(BANK)), key=lambda j: -sims[j])[:k]
        t["retrieval_hit"] += fam in [BANK[j][0] for j in order]
        examples = "\n\n".join('Example: "' + BANK[j][1] + '"\nSpec: ' + BANK[j][2]
                                for j in order)
        raw = ask(model, SOLO.format(problem=story), n=420)
        num = last_number(raw)
        solo_ok = (equal(num, truth) if num is not None else False) or \
            (not str(truth).replace("-", "").isdigit() and str(truth) in raw)
        t["solo"] += solo_ok

        spec = parse_spec(ask_spec_model(
            model, GENERIC.format(story=story, catalogue=catalogue,
                                       examples=examples, preds=PREDICATE_HELP,
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
    print(f"\ncatalogue shown: {n_solvers} solver lines; the needed class was "
          f"retrieved for {t['retrieval_hit']}/{n} problems")
    print(f"SOLO {model} : {t['solo']}/{n}")
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
    summary = {"n": n, "model": model, "solver_lines": n_solvers, **t,
               "byfam": byfam, "qwen_retrieved": 25,
               "focused": FOCUSED, "focused_n": FOCUSED_N, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
