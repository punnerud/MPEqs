#!/usr/bin/env python3
"""Fan-out: one problem in, two or three self-contained subproblems out.

Phase 19 measured the JOIN direction — how many things one step may combine — and found
arity was not the lever. This is the dual, proposed as the missing piece: DECOMPOSITION
arity. One task in, two or three subtasks out, each a self-contained little problem a small
model can answer on its own, plus a combining formula the record computes exactly.

    {"subs": ["How many eggs are in 3 boxes of 12?",
              "..."],
     "combine": "S1 - 5"}

The division of labour follows the cascade: the DECOMPOSER writes questions and a formula
(never an answer); the 1B model answers each subquestion solo — they are sized for exactly
the 37/60 capability it has; the record substitutes the answers into the formula and
computes. Both decomposer sizes run, because phase 60 showed the 1B cannot plan and this
asks whether SPLITTING is cheaper than planning:

    A)  1B decomposes,  1B answers subs, record combines
    B) 35B decomposes,  1B answers subs, record combines

Against the measured anchors: 1B solo 37/60, 35B solo 51/60. Arm B costs one 35B call like
35B solo does — its interest is whether a split's structure survives handing the arithmetic
legs to a model 10x cheaper, which is what would let the expensive model serve many cheap
ones. Wall time per arm is recorded, since speed is half the proposal.
"""
import json
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402
from mapstore import TEST, norm  # noqa: E402
from olympiad import SOLO, last_number  # noqa: E402

DECOMPOSE = """Split the problem into 2 or 3 smaller questions that can each be answered on
their own, and give the formula that combines their answers. Each small question must repeat
the numbers and context it needs — it will be answered by someone who cannot see the
original problem. Do not answer anything yourself.

Reply with only JSON:
{{"subs": ["How many eggs are in 3 boxes of 12 eggs each?",
           "If Tom starts with that many eggs and eats 5, how many is that minus?"],
 "combine": "S1 - 5"}}
(S1 is the answer to the first question, S2 the second, S3 the third.)

Problem: {problem}
"""


def decompose(model, problem):
    reply = ask(model, DECOMPOSE.format(problem=problem), n=400)
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        subs = [str(s) for s in d["subs"]][:3]
        return {"subs": subs, "combine": str(d["combine"])} if 2 <= len(subs) else None
    except Exception:  # noqa: BLE001
        return None


def run_fanout(dec_model, problem):
    d = decompose(dec_model, problem)
    if d is None:
        return None, 0, "no decomposition"
    answers = {}
    for i, sub in enumerate(d["subs"]):
        a = last_number(ask("olmoe-1b", SOLO.format(problem=sub), n=256))
        if a is None:
            return None, len(d["subs"]), f"sub {i + 1} unanswered"
        answers[f"S{i + 1}"] = a
    expr = d["combine"]
    for k in sorted(answers, reverse=True):
        expr = re.sub(rf"\b{k}\b", f"({answers[k]})", expr)
    if not re.fullmatch(r"[\d\s+*/().-]+", expr):
        return None, len(d["subs"]), "combine not arithmetic"
    try:
        return Fraction(eval(expr)), len(d["subs"]), "ok"  # noqa: S307 - digits and ops
    except Exception:  # noqa: BLE001
        return None, len(d["subs"]), "combine failed"


def main(n_test=60, seed=5, out="data/custom/fanout.json"):
    import random
    n_test, seed = int(n_test), int(seed)
    tests = []
    for line in TEST.read_text().splitlines():
        d = json.loads(line)
        tests.append((d["question"],
                      Fraction(norm(d["answer"].rsplit("#### ", 1)[-1].strip()))))
    tests = random.Random(seed).sample(tests, n_test)

    results = {}
    for label, dec_model in (("1b_decomposes", "olmoe-1b"), ("35b_decomposes", "qwen-35b")):
        t0 = time.monotonic()
        right = ran = 0
        fanouts = []
        rows = []
        for q, truth in tests:
            ans, n_subs, stage = run_fanout(dec_model, q)
            ok = ans == truth
            right += ok
            ran += stage == "ok"
            if n_subs:
                fanouts.append(n_subs)
            rows.append({"truth": str(truth), "answer": str(ans), "stage": stage,
                         "subs": n_subs, "ok": ok})
        secs = time.monotonic() - t0
        results[label] = {"right": right, "ran": ran, "n": n_test,
                          "mean_fanout": sum(fanouts) / max(len(fanouts), 1),
                          "seconds": round(secs, 1), "rows": rows}
        print(f"{label:<16}: {right}/{n_test} right, {ran} complete runs, "
              f"mean fan-out {results[label]['mean_fanout']:.1f}, {secs:.0f}s")

    print("\nAnchors: 1B solo 37/60, 35B solo 51/60. Arm A asks whether the small model can")
    print("SPLIT even though it cannot plan; arm B whether a split survives handing its legs")
    print("to a model ten times cheaper — the shape that would let one expensive decomposer")
    print("serve many cheap solvers.")
    Path(out).write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
