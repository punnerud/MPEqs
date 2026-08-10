#!/usr/bin/env python3
"""MPEqs end to end where the vocabulary actually reaches: GSM8K, three arms.

AIME said no (phase 94: 22 mappings claimed, 0 right), and the reason was never the
machines — competition problems need mathematics the catalogue does not contain. Grade
school word problems are the other thing entirely: a handful of quantities combined in
a few steps, which is exactly the arith solver added in this phase, and occasionally a
search or a ratio split.

So the question this phase asks is the one the whole study exists for: WHERE THE
MAPPING IS POSSIBLE, does routing everything through exact machinery beat the model
answering by itself — and can a 1B do the mapping when it cannot do the arithmetic?

  SOLO-35B   the model answers, brief working, last number taken (the baseline)
  MPEqs-35B  the model emits a spec against the catalogue, few-shot with three worked
             mappings; the record executes exactly; nothing the model computes is used
  MPEqs-1B   the same spec prompt at OLMoE-1B — the model that scored at the parrot
             line on one-bit reads (phase 84) and 16/20 answering GSM8K alone

Specs are recorded so the value-echo gate can be applied offline, and every arm is
compared on exact equality with the dataset's own answer.
"""
import json
import random
import re
import subprocess
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aimecover import EXPR_FUNCS_HELP, SCHEMAS2  # noqa: E402
from cutbig import BIN, MODELS, ask  # noqa: E402
from olympiad import SOLO, last_number, load_problems  # noqa: E402
from solvemap import PREDICATE_HELP, SCHEMAS, answer_of, parse_spec  # noqa: E402
from solvers2 import REPAIRS, run2  # noqa: E402
from specgate import value_echo  # noqa: E402

ARITH_SCHEMA = ('{"solver":"arith","let":{"<name>":"<expression using the problem\'s '
                'numbers and earlier names>",...},"answer":"<expression>"}')

FEWSHOT = """Map the problem onto ONE solver and fill its slots. Do NOT compute the
answer — an exact executor computes it from your spec.

Example 1.
Problem: Ann buys 3 boxes of 12 pens and gives away 5 pens. How many pens are left?
Spec: {{"solver":"arith","let":{{"total":"3*12"}},"answer":"total - 5"}}

Example 2.
Problem: A shop sells shirts for 40 kr each. Bo buys 7 and pays with 500 kr. How much
change does he get?
Spec: {{"solver":"arith","let":{{"cost":"40*7"}},"answer":"500 - cost"}}

Example 3.
Problem: How many numbers from 1 to 60 are divisible by 4?
Spec: {{"solver":"search","domain":{{"kind":"range","from":1,"to":60}},
"conditions":[{{"op":"divisible_by","arg":4}}],"aggregate":"count"}}

Catalogue:
{catalogue}

For the search solver, conditions use these ops: {preds}
Expressions may call: {funcs}

Problem: {story}
Spec:"""


def ask_spec_model(model, prompt, n=400):
    """Prefill the opening brace — the format lever from phase 94 — for either model."""
    path, tpl = MODELS[model]
    src = Path(f"/tmp/gsmspec-{model}.txt")
    src.write_text(tpl.format(p=prompt) + "{")
    out = subprocess.run(
        [BIN, "-m", path, "-f", str(src), "-n", str(n), "--temp", "0",
         "-no-cnv", "-st", "-ngl", "99"], capture_output=True, text=True).stdout
    marks = list(re.finditer(r"<\|im_start\|>assistant|<\|assistant\|>|\bassistant\b",
                             out))
    if marks:
        out = out[marks[-1].end():]
    out = re.sub(r"<think>.*?</think>", " ", out, flags=re.S)
    out = out.split("[end of text]")[0]
    return "{" + out.split("{", 1)[-1] if "{" in out else out


def equal(got, truth):
    try:
        return F(str(got)) == F(str(truth))
    except (ValueError, ZeroDivisionError):
        return str(got).strip() == str(truth).strip()


def main(n_problems=30, seed=3, out="data/custom/gsmsolve.json"):
    n_problems, seed = int(n_problems), int(seed)
    gsm, _ = load_problems()
    picks = random.Random(seed).sample(gsm, n_problems)
    catalogue = "\n".join(f"- {v}" for v in
                          {"arith": ARITH_SCHEMA, **SCHEMAS, **SCHEMAS2}.values())

    t = {k: 0 for k in ("solo", "m35_parsed", "m35_ran", "m35_exact", "m35_wrong",
                        "m1b_parsed", "m1b_ran", "m1b_exact", "m1b_wrong",
                        "m35_echo_clean", "m35_gated_right", "m35_gated_wrong",
                        "m35_gated_flag", "agree_solo_mpeqs", "agree_and_right")}
    rows = []
    for i, (story, truth) in enumerate(picks):
        a_solo = last_number(ask("qwen-35b", SOLO.format(problem=story), n=380))
        solo_ok = a_solo is not None and equal(a_solo, truth)
        t["solo"] += solo_ok

        prompt = FEWSHOT.format(story=story, catalogue=catalogue,
                                preds=PREDICATE_HELP, funcs=EXPR_FUNCS_HELP)
        row = {"truth": str(truth), "solo": str(a_solo), "solo_ok": bool(solo_ok)}
        for tag, model in (("m35", "qwen-35b"), ("m1b", "olmoe-1b")):
            spec = parse_spec(ask_spec_model(model, prompt))
            if not isinstance(spec, dict) or "solver" not in spec:
                row[tag] = "no spec"
                continue
            t[f"{tag}_parsed"] += 1
            res, why = run2(spec)
            if res is None:
                row[tag] = f"refused: {why[:34]}"
                continue
            t[f"{tag}_ran"] += 1
            got = answer_of(res)
            ok = equal(got, truth)
            t[f"{tag}_exact" if ok else f"{tag}_wrong"] += 1
            row[tag] = f"{got}" + ("" if ok else f" != {truth}")
            row[f"{tag}_spec"] = spec
            if tag == "m35":
                clean = not value_echo(spec, story)
                t["m35_echo_clean"] += clean
                if clean:
                    t["m35_gated_right" if ok else "m35_gated_wrong"] += 1
                else:
                    t["m35_gated_flag"] += 1
                if a_solo is not None and equal(got, a_solo):
                    t["agree_solo_mpeqs"] += 1
                    t["agree_and_right"] += ok
        rows.append(row)
        print(f"{i:>3} truth {str(truth):>8}  solo {str(a_solo)[:9]:<10}"
              f"{'ok' if solo_ok else '  '}   35B {str(row.get('m35'))[:26]:<28}"
              f"1B {str(row.get('m1b'))[:22]}")

    n = len(picks)
    print(f"\nSOLO-35B      : {t['solo']}/{n}")
    print(f"MPEqs-35B     : spec parsed {t['m35_parsed']}, ran {t['m35_ran']} -> "
          f"EXACT {t['m35_exact']}, wrong {t['m35_wrong']}")
    print(f"MPEqs-1B      : spec parsed {t['m1b_parsed']}, ran {t['m1b_ran']} -> "
          f"EXACT {t['m1b_exact']}, wrong {t['m1b_wrong']}")
    print(f"value-echo gate on 35B specs: clean {t['m35_echo_clean']} -> right "
          f"{t['m35_gated_right']}, WRONG {t['m35_gated_wrong']}, flagged "
          f"{t['m35_gated_flag']}")
    print(f"solo and MPEqs agree on {t['agree_solo_mpeqs']}, right when they agree "
          f"{t['agree_and_right']}")
    print(f"shape disposed over name on {REPAIRS['count']} specs: "
          f"{REPAIRS['by'][:6]}")
    print("\nTwo different kinds of work — one model answering, one model mapping onto")
    print("exact machinery — on problems the machinery can actually express. Where the")
    print("mapping is possible the comparison is honest; where it is not, phase 94")
    print("already said so in the plainest possible numbers.")
    summary = {"n": n, **t, "shape_repairs": REPAIRS["count"],
               "repairs": [list(x) for x in REPAIRS["by"]], "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
