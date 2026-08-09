#!/usr/bin/env python3
"""Fan-out as rRETL: the record splits the text reversibly, and the parts ARE the subtasks.

The model-authored fan-out (fanout.py) writes new subquestions — paraphrases, which cannot
reconstruct the problem and carry phase 57's paraphrase risk. This is the reversible form:
the RECORD splits the problem at sentence boundaries into its parts, keeping them whole, and
concatenating the parts must reproduce the problem byte for byte — verified, the same
contract phase 29 held on 1.29 MB and phase 59 held on 109 spans. Cutting text into smaller
tasks is a fan-out, and cutting is the record's job.

Each part fans out to one subtask: translate THIS fragment into named definitions, given the
graph so far. That is staged absorption with the cutting moved from the model to the record,
which deletes a measured cost outright — phase 59 spent 86 refused span misses on the model
quoting text the record could have cut itself — and guarantees 100% coverage by construction
instead of 97% by behaviour.

Same 20 problems as phases 58-59, both models. Anchors: one-shot absorption 16/20 (35B),
model-cut staged 9/20, and the calls per problem drop from ~10 (steps plus misses) to
exactly the sentence count plus none.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402
from olympiad import load_problems  # noqa: E402
from relgraph import solve_system  # noqa: E402
from stagedabs import extract_json  # noqa: E402

TRANSLATE_PART = """You are absorbing a word problem into named quantities, one fragment at
a time. Definitions so far: {defs}

This fragment (part {i} of {n}):
{part}

Give the definitions this fragment adds, using earlier names where it refers to them. If it
adds none (pure question text), reply {{}}.

Reply with only JSON, like: {{"may": "april / 2"}}
"""


def split_parts(text):
    """Sentence-boundary split, separators kept WITH their parts, rejoin byte-exact.

    The residue of this fan-out is nothing at all: each part carries its own delimiter, so
    concatenation is the inverse and the empty residual is the proof the split lost nothing —
    the same empty-residual shape as the unit graph's reciprocal routes.
    """
    parts = [p for p in re.findall(r"[^.!?]*[.!?]['\"]?\s*|[^.!?]+$", text) if p]
    assert "".join(parts) == text, "split must rejoin byte-exact"
    return parts


def main(n_test=20, seed=5, out="data/custom/rretlfan.json"):
    import random
    n_test, seed = int(n_test), int(seed)
    gsm, _ = load_problems()
    tests = random.Random(seed).sample(gsm, n_test)

    results = {}
    for model in ("qwen-35b", "olmoe-1b"):
        solved = rejoined = calls = 0
        fan = []
        rows = []
        for problem, truth in tests:
            parts = split_parts(problem)
            rejoined += "".join(parts) == problem
            fan.append(len(parts))
            defs = {}
            for i, part in enumerate(parts):
                d = extract_json(ask(model, TRANSLATE_PART.format(
                    defs=json.dumps(defs) or "{}", i=i + 1, n=len(parts),
                    part=part.strip()), n=300))
                calls += 1
                if isinstance(d, dict):
                    defs.update({str(k): str(v) for k, v in d.items()
                                 if isinstance(v, (str, int, float))})
            # The sink rule from phase 59: the asked node is the one nothing references.
            refs = set()
            for body in defs.values():
                refs.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(body)))
            sinks = [k for k in defs if k not in refs]
            ans = None
            why = "no definitions"
            for cand in (sinks[-1:] if sinks else []) + list(defs)[-1:]:
                ans, why = solve_system({"defs": defs, "asked": cand}, problem)
                if ans is not None:
                    break
            ok = ans == truth
            solved += ok
            rows.append({"truth": str(truth), "answer": str(ans), "parts": len(parts),
                         "ok": ok, "why": why if ans is None else "ok"})
        results[model] = {"n": n_test, "solved": solved, "rejoined_exact": rejoined,
                          "mean_fanout": sum(fan) / len(fan), "calls": calls,
                          "rows": rows}
        print(f"{model}: {solved}/{n_test} solved, {rejoined}/{n_test} splits rejoin exact, "
              f"mean fan-out {sum(fan) / len(fan):.1f}, {calls} calls "
              f"({calls / n_test:.1f}/problem)")

    print("\nAnchors: one-shot absorption 16/20 (35B), model-cut staged 9/20 at ~10 calls per")
    print("problem with 86 span misses. The record cutting is the rRETL form of fan-out —")
    print("the parts are the subtasks, the join is the inverse, and the residual is empty.")
    Path(out).write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
