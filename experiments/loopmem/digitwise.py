#!/usr/bin/env python3
"""Split to single digits with an explicit carry — is even `13 + 8` too big a step?

The scratchpad used one binary operation on two multi-digit numbers as the atom, and a 1B model
solved 0 of 8 with it. That granularity is still coarse: `13 + 8` is not one operation, it is
`3 + 8 = 11, write 1, carry 1` then `1 + 1 = 2`. The carry is the join, and the carry is exactly
where this project already measured arithmetic to stop being decomposable — held-out accuracy
fell 0.647 to 0.437 as carries went 0 to 3, and grouping training data by carry count was worth
+0.157 to +0.226, the largest verified effect in the whole project.

So the hypothesis is specific: cut the step to ONE COLUMN and the model never performs
multi-digit arithmetic at all, only single-digit sums it has memorised, and the record carries
the state between columns instead of the model.

Three arms on the same additions, same model:

  WHOLE      ask for `a + b` in one go
  DIGITWISE  ask for one column at a time, we supply the carry, we assemble the answer
  ORACLE     the columns computed correctly, to show what the record contributes when the
             single-digit answers are right — separating "the model cannot add digits" from
             "the decomposition does not help"
"""
import json
import random
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from measure_loops import BIN, MODEL  # noqa: E402

WHOLE = """<|endoftext|><|user|>
What is {a} + {b}? Reply with only the number.
<|assistant|>
"""

COLUMN = """<|endoftext|><|user|>
Add three single digits: {x} + {y} + {c}

Reply with only the total as a number.
<|assistant|>
"""


def ask(prompt, n=48):
    Path("/tmp/dw.txt").write_text(prompt)
    out = subprocess.run(
        [BIN, "-m", MODEL, "-f", "/tmp/dw.txt", "-n", str(n), "--temp", "0",
         "-no-cnv", "-st", "-ngl", "99"],
        capture_output=True, text=True).stdout
    i = out.rfind("<|assistant|>")
    tail = out[i + 13:] if i >= 0 else out
    tail = re.sub(r"<think>.*?</think>", " ", tail, flags=re.S).split("[end of text]")[0]
    # The LAST number, not the first. At n=12 the model was cut off mid-restatement — "The
    # total of the three digits 9, 7, and" — and taking the first match returned 9, the digit
    # it was asked about. Every digitwise answer then came back as the first operand, which
    # looked exactly like a model that cannot add and was a token budget of twelve.
    nums = re.findall(r"-?\d+", tail.replace(",", ""))
    return int(nums[-1]) if nums else None


def digitwise(a, b, oracle=False):
    """One column at a time. The record holds the carry and the digits; the model adds three
    single digits or nothing at all."""
    da = [int(c) for c in str(a)][::-1]
    db = [int(c) for c in str(b)][::-1]
    width = max(len(da), len(db))
    da += [0] * (width - len(da))
    db += [0] * (width - len(db))
    carry, out, asked = 0, [], 0
    for k in range(width):
        if oracle:
            total = da[k] + db[k] + carry
        else:
            total = ask(COLUMN.format(x=da[k], y=db[k], c=carry))
            asked += 1
            if total is None:
                return None, asked
        # The record does the splitting, not the model: it never has to know what a carry is.
        out.append(total % 10)
        carry = total // 10
    if carry:
        out.append(carry)
    return int("".join(str(d) for d in out[::-1])), asked


def main(n_tasks=12, seed=5, out="data/custom/digitwise.json"):
    n_tasks, seed = int(n_tasks), int(seed)
    rng = random.Random(seed)
    # Deliberately carry-heavy: pairs whose columns mostly overflow, which is where the whole
    # difficulty was measured to live.
    tasks = []
    while len(tasks) < n_tasks:
        a = rng.randint(100, 999)
        b = rng.randint(100, 999)
        carries = sum(1 for k in range(3)
                      if (a // 10**k) % 10 + (b // 10**k) % 10 >= 10)
        if carries >= 2:
            tasks.append((a, b))

    print(f"{n_tasks} carry-heavy three-digit additions, OLMoE 1B-7B\n")
    print(f"{'problem':>14}{'truth':>8}{'whole':>8}{'digitwise':>11}{'oracle':>8}{'calls':>7}")
    rows, w_ok, d_ok, o_ok = [], 0, 0, 0
    for a, b in tasks:
        truth = a + b
        whole = ask(WHOLE.format(a=a, b=b))
        dw, calls = digitwise(a, b)
        orc, _ = digitwise(a, b, oracle=True)
        w_ok += whole == truth
        d_ok += dw == truth
        o_ok += orc == truth
        rows.append({"a": a, "b": b, "truth": truth, "whole": whole,
                     "digitwise": dw, "oracle": orc, "calls": calls})
        print(f"{f'{a} + {b}':>14}{truth:>8}{str(whole):>8}{str(dw):>11}{str(orc):>8}"
              f"{calls:>7}")

    summary = {"tasks": n_tasks, "whole_correct": w_ok, "digitwise_correct": d_ok,
               "oracle_correct": o_ok}
    print(f"\nwhole number in one go : {w_ok}/{n_tasks}")
    print(f"one column at a time   : {d_ok}/{n_tasks}")
    print(f"columns done correctly : {o_ok}/{n_tasks}   (the record's own contribution)")
    print("\nIf digitwise beats whole, the step was too big. If oracle is perfect and digitwise")
    print("is not, the model cannot add three single digits and no decomposition rescues that.")
    Path(out).write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
