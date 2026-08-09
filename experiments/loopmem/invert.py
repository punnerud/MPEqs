#!/usr/bin/env python3
"""Can it go the other way? Forward and backward over the SAME arithmetic facts.

The round trip rests on an assumption worth testing on its own: a model that can produce the
residual should be able to consume it. `39 * 8 = 312` forward and `39 * ? = 312` backward are the
same fact, so if the second is much harder than the first the round trip is measuring the
model's weakness at inversion rather than the correctness of its work.

The first attempt suggested exactly that — `? / 2 = 175` came back as 175 — but a bare question
is not a fair comparison when the forward arm gets a clean one. So each direction is asked twice,
plainly and as JSON, since JSON is the notation these models are trained on and the one that took
the graph arm from 3/20 to 11/20.

Four cells over the same facts:

              plain     json
  forward     a + b = ?
  backward    a + ? = c

Whatever the answer, it is a property of the model and not of this project's machinery, which is
why it is worth pinning separately.
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from twoway import ask_num  # noqa: E402

FWD = """<|endoftext|><|user|>
What is {a} {op} {b}? Reply with only the number.
<|assistant|>
"""

BWD = """<|endoftext|><|user|>
{left} = {c}

What number goes in the blank? Reply with only the number.
<|assistant|>
"""

FWD_JSON = """<|endoftext|><|user|>
Fill in the value.

Example:
{{"expression": "6 * 5", "value": ?}}
{{"value": 30}}

{{"expression": "{a} {op} {b}", "value": ?}}
<|assistant|>
"""

BWD_JSON = """<|endoftext|><|user|>
Fill in the missing number.

Example:
{{"from": "6 * ?", "to": 30}}
{{"missing": 5}}

{{"from": "{left}", "to": {c}}}
<|assistant|>
"""


# The inverse of each operation, posed as an ordinary forward computation rather than as a blank
# to fill. Phase 18's backward pass worked (12/12 errors detected) and it was written this way —
# `s - b`, a sum — while this file's backward arm asks `b + ? = s`, which is a different question
# about the same fact. If the rephrasing recovers the accuracy, then reversal is usable and only
# blank-filling is not.
INVOP = {"+": "-", "-": "+", "*": "/", "/": "*"}

FWD_INV = """<|endoftext|><|user|>
What is {c} {iop} {a}? Reply with only the number.
<|assistant|>
"""


def facts(n, rng):
    """Arithmetic of the same shape the graph arm actually produces."""
    out = []
    while len(out) < n:
        op = rng.choice("+-*/")
        if op == "*":
            a, b = rng.randint(11, 49), rng.randint(2, 12)
        elif op == "/":
            b = rng.randint(2, 29)
            a = b * rng.randint(2, 40)
        else:
            a, b = rng.randint(11, 400), rng.randint(2, 99)
        c = {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[op]
        if c == int(c):
            out.append((a, op, b, int(c)))
    return out


def main(n_facts=24, seed=11, out="data/custom/invert.json"):
    rng = random.Random(int(seed))
    fs = facts(int(n_facts), rng)
    print(f"{len(fs)} facts, each asked forwards and backwards, plainly and as JSON\n")
    print(f"{'fact':>18}{'fwd':>6}{'fwdJ':>6}{'bwd':>6}{'bwdJ':>6}{'bwdF':>6}")
    tally = Counter()
    rows = []
    for a, op, b, c in fs:
        # Backward blanks the SECOND operand, so the arithmetic content is identical and only
        # the direction of the question changes.
        left = f"{a} {op} ?"
        got = {
            "fwd": ask_num(FWD.format(a=a, op=op, b=b)),
            "fwdJ": ask_num(FWD_JSON.format(a=a, op=op, b=b)),
            "bwd": ask_num(BWD.format(left=left, c=c)),
            "bwdJ": ask_num(BWD_JSON.format(left=left, c=c)),
            # Same inversion, asked as arithmetic: `a + b = c` becomes "what is c - a".
            "bwdF": ask_num(FWD_INV.format(c=c, iop=INVOP[op], a=a)),
        }
        want = {"fwd": c, "fwdJ": c, "bwd": b, "bwdJ": b,
                "bwdF": b if op in "+*" else (a - c if op == "-" else a / c)}
        for k, v in got.items():
            tally[k] += v is not None and abs(v - want[k]) < 1e-9
        rows.append({"a": a, "op": op, "b": b, "c": c, **got})
        mark = {k: ("ok" if got[k] is not None and abs(got[k] - want[k]) < 1e-9 else ".")
                for k in got}
        print(f"{f'{a} {op} {b} = {c}':>18}{mark['fwd']:>6}{mark['fwdJ']:>6}"
              f"{mark['bwd']:>6}{mark['bwdJ']:>6}{mark['bwdF']:>6}")

    n = len(fs)
    print(f"\nforward, plain  : {tally['fwd']}/{n}")
    print(f"forward, JSON   : {tally['fwdJ']}/{n}")
    print(f"backward, plain : {tally['bwd']}/{n}")
    print(f"backward, JSON  : {tally['bwdJ']}/{n}")
    print(f"backward as plain arithmetic : {tally['bwdF']}/{n}")
    best_f = max(tally["fwd"], tally["fwdJ"])
    best_b = max(tally["bwd"], tally["bwdJ"], tally["bwdF"])
    print(f"\nbest forward {best_f}/{n} against best backward {best_b}/{n}")
    print("If backward is far worse, the round trip cannot be used as a correctness check —")
    print("it would be measuring the model's inversion, not its work.")
    summary = {"facts": n, **{f"{k}_correct": tally[k] for k in ("fwd", "fwdJ", "bwd", "bwdJ", "bwdF")},
               "best_forward": best_f, "best_backward": best_b}
    Path(out).write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
