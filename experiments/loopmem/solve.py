#!/usr/bin/env python3
"""MPEqs as one command: a problem in, an exact answer or an honest refusal out.

    ./solve.py "How many days are there from 1 January 1970 to 10 August 2026?"
    ./solve.py --model olmoe-1b "What is the exact population variance of 3, 8, 15?"
    ./solve.py --explain "A price of 100 rises 20 percent then falls 20 percent."

Everything the study measured, wired together in the order it was measured in: retrieve
two exemplars for this problem, show the catalogue, let the model write ONE spec, execute
it exactly, and — per phase 112 — fall back to the model's own answer only where the
record REFUSED, never where it answered. The exit code is 0 for an answer, 2 for a
refusal that nothing rescued, so this composes with other tools.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aimecover import EXPR_FUNCS_HELP, SCHEMAS2  # noqa: E402
from bands2 import SCHEMA_MORE  # noqa: E402
from bands3 import SCHEMA_3  # noqa: E402
from bands4 import SCHEMA_4  # noqa: E402
from bands5 import SCHEMA_5  # noqa: E402
from bands6 import SCHEMA_6  # noqa: E402
from bands7 import SCHEMA_7  # noqa: E402
from cutbig import ask  # noqa: E402
from embednav import embed  # noqa: E402
from gsmsolve import ARITH_SCHEMA, ask_spec_model  # noqa: E402
from mapmemory import mask  # noqa: E402
from mixedretr import BANK as BANK9  # noqa: E402
from bands4 import BANK as BANK4  # noqa: E402
from bands5 import BANK as BANK5  # noqa: E402
from bands6 import BANK as BANK6  # noqa: E402
from bands7 import BANK as BANK7  # noqa: E402
from newbands import SCHEMA_NEW  # noqa: E402
from olympiad import SOLO, last_number  # noqa: E402
from solvemap import PREDICATE_HELP, SCHEMAS, answer_of, parse_spec  # noqa: E402
from solvers2 import run2  # noqa: E402

PROMPT = """Map the problem onto ONE solver and fill its slots. Do NOT compute the
answer — an exact executor computes it from your spec.

{examples}

Catalogue:
{catalogue}

For the search solver, conditions use these ops: {preds}
Expressions may call: {funcs}   (^ is a power; / is exact rational division)

Problem: {story}
Spec:"""


def catalogue():
    return "\n".join(f"- {v}" for v in
                     {"arith": ARITH_SCHEMA, **SCHEMA_7, **SCHEMA_6, **SCHEMA_5,
                      **SCHEMA_4, **SCHEMA_3, **SCHEMA_MORE, **SCHEMA_NEW,
                      **SCHEMAS, **SCHEMAS2}.values())


def bank():
    out, seen = [], set()
    for entry in list(BANK9) + list(BANK4) + list(BANK5) + list(BANK6) + list(BANK7):
        if entry[0] in seen:
            continue
        seen.add(entry[0])
        out.append(entry)
    return out


def solve(story, model="qwen-35b", k=2, explain=False):
    b = bank()
    bvecs = embed([mask(p) for _t, p, _s in b])
    qvec = embed([mask(story)])[0]
    sims = [sum(x * y for x, y in zip(qvec, bv)) for bv in bvecs]
    order = sorted(range(len(b)), key=lambda j: -sims[j])[:k]
    examples = "\n\n".join('Example: "' + b[j][1] + '"\nSpec: ' + b[j][2]
                           for j in order)
    spec = parse_spec(ask_spec_model(model, PROMPT.format(
        story=story, catalogue=catalogue(), examples=examples,
        preds=PREDICATE_HELP, funcs=EXPR_FUNCS_HELP), n=500))

    report = {"problem": story, "retrieved": [b[j][0] for j in order], "spec": spec}
    if isinstance(spec, dict) and "solver" in spec:
        res, why = run2(spec)
        if res is not None:
            report.update({"answer": answer_of(res, spec), "source": "record",
                           "detail": {k2: v for k2, v in res.items()
                                      if k2 != "value"}})
            return report
        report["refusal"] = why
    else:
        report["refusal"] = "no spec"

    # Phase 112: the model answers only what the record declined.
    raw = ask(model, SOLO.format(problem=story), n=420)
    guess = last_number(raw)
    report.update({"answer": None if guess is None else str(guess),
                   "source": "model fallback (the record refused)"})
    return report


def main(*args):
    argv = list(args)
    model, k, explain = "qwen-35b", 2, False
    while argv and argv[0].startswith("--"):
        flag = argv.pop(0)
        if flag == "--model":
            model = argv.pop(0)
        elif flag == "--explain":
            explain = True
        elif flag == "--k":
            k = int(argv.pop(0))
        else:
            print(f"unknown flag {flag}")
            return 2
    if not argv:
        print(__doc__)
        return 2
    report = solve(" ".join(argv), model=model, k=k, explain=explain)
    if explain:
        print(json.dumps(report, indent=2, default=str))
    else:
        if report.get("answer") is None:
            print(f"REFUSED: {report.get('refusal', 'no answer')}")
            return 2
        print(report["answer"])
        if report["source"] != "record":
            print(f"  ({report['source']})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
