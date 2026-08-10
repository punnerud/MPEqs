#!/usr/bin/env python3
"""Better exemplars, same model: is the 1B's ceiling the example or the model?

Phase 118 drew the small model's boundary precisely — it solves the classes whose
retrieved exemplar is nearly a template and misses the ones where the example has to be
ADAPTED. That is a claim about the exemplars, not about the model, and it is testable by
rewriting them.

So five near-template exemplars are added for the classes the 1B missed (big products,
unit conversion, factorial divisors, fraction folds, word arrangements), each phrased
the way those problems are usually phrased and each on DIFFERENT numbers and a different
question from anything in the battery. Nothing else changes: same model, same catalogue,
same retrieval, same battery. If the score moves, the ceiling was the example.

Phase 109 measured the small model driving twenty-six solvers: 12 of 27, up from 1 alone,
with every spec parsing. The library has since grown by eleven machines and the catalogue
by a third, which is exactly the direction that should hurt a model whose failure mode
was never arithmetic but instruction-following. Same battery as the 35B run, same bank,
same retrieval, same prefill — only the seat changes.

Phase 108 measured the recipe at twenty-six solvers and found it held — retrieval put
the right exemplar up and the mixed battery went 25 of 27 with nothing wrong. The
library has since grown to thirty-seven machines across thirteen classes, and a scaling
law that holds once is a coincidence until it holds again at a size that could have
broken it.

Same construction, bigger everything: two problems from each class (all drawn from
batteries whose truths were computed before any model saw them), the whole catalogue in
the prompt, one exemplar per class in the bank, the two nearest retrieved. Nothing here
is tuned for this run.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aimecover import EXPR_FUNCS_HELP, SCHEMAS2  # noqa: E402
from bands2 import SCHEMA_MORE  # noqa: E402
from bands2 import build as build_b2  # noqa: E402
from bands3 import SCHEMA_3  # noqa: E402
from bands4 import SCHEMA_4  # noqa: E402
from bands4 import BANK as BANK4  # noqa: E402
from bands4 import build as build_b4  # noqa: E402
from bands5 import SCHEMA_5  # noqa: E402
from bands5 import BANK as BANK5  # noqa: E402
from bands5 import build as build_b5  # noqa: E402
from cutbig import ask  # noqa: E402
from embednav import embed  # noqa: E402
from gsmsolve import ARITH_SCHEMA, ask_spec_model, equal  # noqa: E402
from hardarith import build as build_hard  # noqa: E402
from mapmemory import mask  # noqa: E402
from mixedretr import BANK as BANK9  # noqa: E402
from newbands import SCHEMA_NEW  # noqa: E402
from newbands import build as build_nb  # noqa: E402
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


def main(per_class=2, k=2, model="olmoe-1b",
         out="data/custom/grandmix_1b_tpl.json"):
    per_class, k = int(per_class), int(k)
    pool = {}
    for fam, story, truth in (build_hard(1) + build_nb() + build_b2() + build_b4()
                              + build_b5()):
        pool.setdefault(fam, []).append((story, str(truth)))
    battery = [(fam, s, t) for fam, items in sorted(pool.items())
               for s, t in items[:per_class]]

    # Near-template exemplars for the classes phase 118 measured as misses. Different
    # numbers, different questions, same sentence shape — the shape is what a small
    # model copies, and that is the hypothesis under test.
    TEMPLATES = [
        ("big", "What is 731942856 times 419273865?",
         '{"solver":"arith","let":{"p":"731942856 * 419273865"},"answer":"p"}'),
        ("convert", "A train moves at 29 kilometres per hour. Exactly how many "
         "millimetres per second is that? Give a fraction.",
         '{"solver":"convert","value":29,"from":"km/hour","to":"mm/second"}'),
        ("divisors", "What is the sum of the exponents in the prime factorisation of "
         "19 factorial?",
         '{"solver":"factor","k":19,"report":"exponent_sum"}'),
        ("fractions", "Add up 1/(k(k+7)) for every integer k from 1 to 25. Give the "
         "exact sum as a fraction.",
         '{"solver":"multisearch","variables":[{"name":"k","from":1,"to":25}],'
         '"objective":"1/(k*(k+7))","aggregate":"sum"}'),
        ("strcount", "How many distinct arrangements are there of the letters of "
         "TENNESSEE?",
         '{"solver":"strcount","word":"TENNESSEE","kind":"arrangements"}'),
    ]
    bank = list(BANK9) + [b for b in BANK4 if b[0] not in {x[0] for x in BANK9}] + \
        [b for b in BANK5 if b[0] not in {x[0] for x in BANK9}] + TEMPLATES
    # bands4 and bands5 carry the shape/inclusion/formula entries too; keep one each.
    # Deduplication is deliberately NOT applied to the templates: a class may now hold
    # two exemplars, and retrieval picks whichever is closer to the problem at hand.

    catalogue = "\n".join(f"- {v}" for v in
                          {"arith": ARITH_SCHEMA, **SCHEMA_5, **SCHEMA_4, **SCHEMA_3,
                           **SCHEMA_MORE, **SCHEMA_NEW, **SCHEMAS,
                           **SCHEMAS2}.values())
    bvecs = embed([mask(p) for _t, p, _s in bank])
    pvecs = embed([mask(s) for _f, s, _t in battery])

    t = {key: 0 for key in ("solo", "mpeqs", "parsed", "ran", "wrong",
                            "retrieval_hit")}
    byfam = {}
    rows = []
    for i, (fam, story, truth) in enumerate(battery):
        sims = [sum(a * b for a, b in zip(pvecs[i], bv)) for bv in bvecs]
        order = sorted(range(len(bank)), key=lambda j: -sims[j])[:k]
        shown = [bank[j][0] for j in order]
        t["retrieval_hit"] += fam in shown
        examples = "\n\n".join('Example: "' + bank[j][1] + '"\nSpec: ' + bank[j][2]
                               for j in order)
        raw = ask(model, SOLO.format(problem=story), n=420)
        num = last_number(raw)
        solo_ok = (equal(num, truth) if num is not None else False) or \
            (not str(truth).replace("-", "").isdigit() and str(truth) in raw)
        t["solo"] += solo_ok

        spec = parse_spec(ask_spec_model(
            model, PROMPT.format(story=story, catalogue=catalogue,
                                      examples=examples, preds=PREDICATE_HELP,
                                      funcs=EXPR_FUNCS_HELP), n=420))
        got, ok = None, False
        if isinstance(spec, dict) and "solver" in spec:
            t["parsed"] += 1
            res, why = run2(spec)
            if res is not None:
                t["ran"] += 1
                got = answer_of(res, spec)
                ok = str(got) == str(truth) or equal(got, truth)
                t["mpeqs"] += ok
                t["wrong"] += not ok
            else:
                got = why[:34]
        f = byfam.setdefault(fam, [0, 0, 0])
        f[0] += 1
        f[1] += solo_ok
        f[2] += ok
        rows.append({"family": fam, "truth": truth, "solo_ok": bool(solo_ok),
                     "mpeqs": str(got), "mpeqs_ok": bool(ok),
                     "solver": (spec or {}).get("solver"), "shown": shown})
        print(f"{fam:<12}{'solo ok' if solo_ok else 'solo X '} "
              f"{'mpeqs ok' if ok else 'mpeqs X '} {str((spec or {}).get('solver')):<13}"
              f"{story[:34]}")

    n = len(battery)
    print(f"\ncatalogue {len(catalogue.splitlines())} lines, bank {len(bank)} exemplars;"
          f" the needed class was retrieved for {t['retrieval_hit']}/{n}")
    print(f"SOLO {model} : {t['solo']}/{n}")
    print(f"MPEqs    : {t['mpeqs']}/{n}  (parsed {t['parsed']}, ran {t['ran']}, "
          f"wrong {t['wrong']})")
    print(f"\n{'class':<13}{'n':>2}{'solo':>6}{'MPEqs':>7}")
    for fam, (cnt, so, mp) in sorted(byfam.items()):
        print(f"{fam:<13}{cnt:>2}{so:>6}{mp:>7}")
    print("\nAt twenty-six machines the recipe held; this asks the same question at")
    print("thirty-seven, where a prompt is long enough and a bank crowded enough that")
    print("failing would have been the ordinary outcome.")
    summary = {"n": n, "model": model, "qwen_reference": 33, "baseline_1b": 18,
               "bank": len(bank),
               "catalogue_lines": len(catalogue.splitlines()), **t, "byfam": byfam,
               "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
