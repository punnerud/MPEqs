#!/usr/bin/env python3
"""Thirty AIME problems this study has never touched, through the finished pipeline.

Every olympiad measurement so far used one sample — thirty problems drawn with seed 5,
and mostly the first fifteen of those — which the library, the schemas and the exemplar
bank have all been shaped around at one remove. The remaining thirty problems in the
cached AIME set have never been read by anything here.

So they get the finished machine exactly as it stands: 41 solvers, the full catalogue,
the retrieval bank, slot-friendly schemas, one spec per problem, executed exactly. Both
arms, so the comparison is against the same model answering with visible working rather
than against nothing.

The expectation, stated before the run so it cannot be adjusted afterwards: phase 99
measured a hand-mapped ceiling of five in fifteen and phase 124 got two of those five,
so something in the region of two to five of thirty is what the previous evidence
predicts. Anything above that is the growth since; anything below is the sample.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aimebudget import ask_work  # noqa: E402
from aimecover import EXPR_FUNCS_HELP, ask_spec  # noqa: E402
from embednav import embed  # noqa: E402
from mapmemory import mask  # noqa: E402
from olympiad import last_number, load_problems  # noqa: E402
from solve import bank, catalogue  # noqa: E402
from solvemap import PREDICATE_HELP, answer_of, parse_spec  # noqa: E402
from solvers2 import run2  # noqa: E402

PROMPT = """Map this competition problem onto ONE solver and fill its slots, or say what
is missing. An exact executor runs your spec — never compute the answer yourself.

{examples}

Catalogue:
{catalogue}

For the search solver, conditions use these ops: {preds}
Expressions may call: {funcs}   (^ is a power; / is exact rational division)

Problem: {problem}

Reply with ONLY the JSON spec, or ONLY {{"none":"<the one capability that is missing>"}}"""

SOLO = """{problem}

Work it out, then write the final line exactly as:
Answer: <number>
"""


def main(k=2, out="data/custom/aimefresh.json"):
    k = int(k)
    _, aime = load_problems()
    used = {p for p, _a in random.Random(5).sample(aime, 30)}
    fresh = [(p, a) for p, a in aime if p not in used]
    print(f"{len(fresh)} AIME problems never used in this study\n")

    b = bank()
    bvecs = embed([mask(p) for _t, p, _s in b])
    pvecs = embed([mask(p) for p, _a in fresh])
    cat = catalogue()

    t = Counter()
    rows = []
    for i, (problem, truth) in enumerate(fresh):
        sims = [sum(x * y for x, y in zip(pvecs[i], bv)) for bv in bvecs]
        order = sorted(range(len(b)), key=lambda j: -sims[j])[:k]
        examples = "\n\n".join('Example: "' + b[j][1] + '"\nSpec: ' + b[j][2]
                               for j in order)
        spec = parse_spec(ask_spec(PROMPT.format(
            problem=problem, catalogue=cat, examples=examples,
            preds=PREDICATE_HELP, funcs=EXPR_FUNCS_HELP), n=900))

        got, ok, why = None, False, ""
        if spec is None:
            t["unparseable"] += 1
            why = "unparseable"
        elif ("none" in spec and "solver" not in spec) or spec.get("solver") == "none":
            t["declined"] += 1
            why = f"declined: {str(spec.get('none') or spec.get('missing'))[:44]}"
        else:
            t["claimed"] += 1
            res, refusal = run2(spec)
            if res is None:
                t["refused"] += 1
                why = f"refused: {refusal[:44]}"
            else:
                t["ran"] += 1
                got = answer_of(res, spec)
                ok = str(got) == str(truth)
                t["exact" if ok else "wrong"] += 1

        solo = last_number(ask_work(SOLO.format(problem=problem), n=1400))
        solo_ok = solo is not None and str(solo) == str(truth)
        t["solo"] += solo_ok

        rows.append({"truth": str(truth), "mpeqs": str(got), "mpeqs_ok": bool(ok),
                     "solo": str(solo), "solo_ok": bool(solo_ok),
                     "solver": (spec or {}).get("solver"), "why": why,
                     "problem": " ".join(problem.split())[:300]})
        print(f"{i:>3} truth {str(truth):>5}  MPEqs "
              f"{('EXACT ' + str(got)) if ok else (str(got) if got else why)[:34]:<38}"
              f"solo {'ok' if solo_ok else str(solo)[:8]}")

    n = len(fresh)
    print(f"\nSOLO-35B (visible working) : {t['solo']}/{n}")
    print(f"MPEqs                      : {t['exact']}/{n}  "
          f"(claimed {t['claimed']}, ran {t['ran']}, wrong {t['wrong']}, "
          f"refused {t['refused']}, declined {t['declined']}, "
          f"unparseable {t['unparseable']})")
    either = sum(1 for r in rows if r["mpeqs_ok"] or r["solo_ok"])
    both = sum(1 for r in rows if r["mpeqs_ok"] and r["solo_ok"])
    print(f"either arm right {either}/{n}, both right {both}")
    used_solvers = Counter(r["solver"] for r in rows if r["solver"])
    print(f"solvers reached for: {dict(used_solvers)}")
    print("\nThe honest reading of an olympiad number is which problems it is, not how")
    print("many: the machines are exact, so anything solved here was expressible, and")
    print("anything missed was either out of vocabulary or out of the mapper's reach.")
    summary = {"n": n, **dict(t), "either": either, "both": both,
               "solvers": dict(used_solvers), "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
