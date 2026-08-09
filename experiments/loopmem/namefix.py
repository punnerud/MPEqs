#!/usr/bin/env python3
"""The record hands out the node names: does the namespace residual die?

Phase 64's autopsy: 7 of 20 fan-out failures were parts inventing different names for the
same thing — `april` against `april_clips`. That was never a tagging problem; it was two
parts coining strings independently. The record is not context-limited the way the parts
are: it reads the WHOLE problem, extracts every noun lemma through the lexicon, and issues
ONE shared name list that every part must draw from. Consistency by construction, whatever
the tagger's precision — a mediocre noun is still the SAME mediocre noun in every part.

The re-run is phase 64's exactly — same 20 problems, same seed, same byte-exact sentence
split — with two changes and only two:

    the prompt lists the allowed names (text lemmas plus 'answer'), with the numbers the
    record attached to them;
    a definition using any other name is refused STRUCTURALLY (perfect ground truth, the
    phase 55 rule), the reason names the foreign word, one retry per part.

Anchors: single-pass record-cut 2/20 with 7 'does not reduce' failures; fixpoint 1/20;
one-shot 16/20. If the name failures collapse, the residual was names; whatever gap to
one-shot remains is the relations themselves.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402
from lexicon import Lexicon  # noqa: E402
from mapstore import NUM, norm  # noqa: E402
from olympiad import load_problems  # noqa: E402
from relgraph import solve_system  # noqa: E402
from rretlfan import split_parts  # noqa: E402
from stagedabs import extract_json  # noqa: E402
from textgraph import units_of_v2  # noqa: E402

WORD = re.compile(r"[A-Za-z]+")
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

TRANSLATE = """You are absorbing a word problem into named quantities. You may ONLY use
these names, chosen from the problem's own words:
  {names}
Definitions so far: {defs}

This fragment (part {i} of {n}):
{part}

Give the definitions this fragment adds, as JSON like {{"clip": "48 / 2"}}. Use the names
above on both sides; "answer" is for the final asked quantity. If the fragment adds
nothing, reply {{}}.
{hint}"""


def allowed_names(problem, lex):
    """Every lexicon noun-lemma in the problem, plus the numbers the record attached."""
    names = []
    for w in WORD.findall(norm(problem)):
        lemma = lex.noun_lemma(w)
        if lemma and " " not in lemma:
            lemma = re.sub(r"[^a-z0-9_]", "", lemma.lower())
            if lemma and lemma not in names:
                names.append(lemma)
    values = [norm(m) for m in NUM.findall(norm(problem))]
    units = units_of_v2(problem, lex)
    attached = {}
    for v, u in zip(values, units):
        if u:
            u2 = re.sub(r"[^a-z0-9_]", "", u.lower())
            if u2:
                attached.setdefault(u2, v)
    return names + ["answer"], attached


def main(n_test=20, seed=5, model="qwen-35b", out="data/custom/namefix.json"):
    import random
    n_test, seed = int(n_test), int(seed)
    lex = Lexicon()
    gsm, _ = load_problems()
    tests = random.Random(seed).sample(gsm, n_test)

    tally = {"solved": 0, "foreign_refusals": 0, "retried_ok": 0, "not_reduce": 0,
             "missing_numbers": 0, "calls": 0}
    rows = []
    for problem, truth in tests:
        parts = split_parts(problem)
        names, attached = allowed_names(problem, lex)
        listed = ", ".join(f"{n}({attached[n]})" if n in attached else n for n in names)
        defs = {}
        for i, part in enumerate(parts):
            hint = ""
            for attempt in range(2):
                d = extract_json(ask(model, TRANSLATE.format(
                    names=listed, defs=json.dumps(defs) or "{}",
                    i=i + 1, n=len(parts), part=part.strip(), hint=hint), n=300))
                tally["calls"] += 1
                if not isinstance(d, dict):
                    break
                foreign = [w for k, v in d.items()
                           for w in [k] + NAME.findall(str(v))
                           if w not in names and w not in defs
                           and not re.fullmatch(r"\d+", w)]
                if foreign:
                    tally["foreign_refusals"] += 1
                    hint = (f"\nRefused: {', '.join(sorted(set(foreign))[:3])} is not an "
                            f"allowed name. Use only the listed names.")
                    continue
                for k, v in d.items():
                    if isinstance(v, (str, int, float)) and str(k) not in defs:
                        defs[str(k)] = str(v)
                if attempt > 0:
                    tally["retried_ok"] += 1
                break
        # The sink rule; 'answer' wins when defined.
        asked = "answer" if "answer" in defs else None
        if asked is None:
            refs = set()
            for body in defs.values():
                refs.update(NAME.findall(str(body)))
            sinks = [k for k in defs if k not in refs]
            asked = sinks[-1] if sinks else (list(defs)[-1] if defs else "")
        ans, why = solve_system({"defs": defs, "asked": asked}, problem)
        ok = ans == truth
        tally["solved"] += ok
        if why:
            tally["not_reduce"] += "does not reduce" in why
            tally["missing_numbers"] += why.startswith("problem numbers missing")
        rows.append({"truth": str(truth), "answer": str(ans), "ok": ok,
                     "why": why, "n_names": len(names), "defs": len(defs)})

    n = n_test
    print(f"{model}, {n} problems, the record issuing the names:\n")
    print(f"  solved                      : {tally['solved']}/{n}   "
          f"(anchors: single-pass 2/20, fixpoint 1/20, one-shot 16/20)")
    print(f"  'does not reduce' failures  : {tally['not_reduce']}/{n}   (was 7/20)")
    print(f"  missing-number failures     : {tally['missing_numbers']}/{n}")
    print(f"  foreign-name refusals fired : {tally['foreign_refusals']} "
          f"({tally['retried_ok']} parts recovered on retry)")
    print(f"  calls                       : {tally['calls']} "
          f"({tally['calls'] / n:.1f}/problem)")
    print("\nIf the name failures collapsed, the hidden residual was the namespace and the")
    print("record's whole-problem view pays it. Whatever gap to one-shot remains is the")
    print("relations — the part of the story only a whole-problem READER can hold.")
    Path(out).write_text(json.dumps({"model": model, "n": n, **tally, "rows": rows},
                                    indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
