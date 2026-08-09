#!/usr/bin/env python3
"""`228 - 110` is still too big. Break the rephrased question down again.

Letting the model write its own question took the backward direction from 5/24 to 19/24, and it
wrote `What is 228 - 110?` — which is a three-digit subtraction, and phase 17 measured exactly
that atom as too large: whole-number addition was 6/12 where one column at a time was 11/12. So
the rephrasing is not the end of the chain, it is the start of another one.

Three ways to finish it, on the 24 inversions the model rephrased for itself:

  WHOLE     answer the rephrased question as it stands                    (19/24 established)
  COLUMNS   the record splits + and - into single-digit columns and holds the carry or borrow,
            which is phase 17's mechanism applied to whatever the model asked for
  AGAIN     the model breaks its own question into smaller steps as JSON, and the record checks
            the decomposition by inlining it — the same reversibility rule, one level deeper

COLUMNS only applies where the operation has columns. `10206 / 27` does not, and pretending
otherwise would hide the limit rather than measure it, so those fall through to WHOLE and the
count of how many were actually decomposable is reported alongside.
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from invert import INVOP, facts  # noqa: E402
from jsongraph import KEY, OPERAND, check_graph, parse_graph  # noqa: E402
from rephrase import ANSWER, REWRITE, ask_text, first_question  # noqa: E402
from repeat import ask  # noqa: E402
from substitute import EVAL  # noqa: E402

SPLIT = """<|endoftext|><|user|>
Break the sum into small steps and write them as JSON.

Rules:
- each key is one step using at most three numbers or earlier keys
- a key may be used in later steps
- the last key is the answer

Example:
Sum: 228 - 110
{{"A": "228 - 100", "B": "A - 10"}}

Sum: {expr}
<|assistant|>
"""

# The wording phases 17 and 18 verified. `8 - 0 = ? Reply with only the number.` was answered
# "8" and `2 - 1 = ?` was answered "2" — the model completes the pattern by echoing the first
# operand rather than computing, and a column-wise subtraction built on it returns its own input.
PAIR = {
    "+": """<|endoftext|><|user|>
Add two single digits: {x} + {y}

Reply with only the total as a number.
<|assistant|>
""",
    "-": """<|endoftext|><|user|>
Subtract two single digits: {x} - {y}

Reply with only the number.
<|assistant|>
""",
}

ARITH = re.compile(r"(-?\d+)\s*([-+*/])\s*(-?\d+)")


def extract_sum(question):
    """The arithmetic inside the model's own question, if it is plain arithmetic."""
    q = (question or "").replace("divided by", "/").replace("multiplied by", "*")
    q = q.replace("plus", "+").replace("minus", "-")
    m = ARITH.search(q)
    return (int(m.group(1)), m.group(2), int(m.group(3))) if m else None


def columns(a, op, b):
    """One digit column at a time, with the record holding the carry or the borrow.

    The model is never asked to subtract or add anything wider than a single digit; everything
    about position, carrying and borrowing is the record's job. That division of labour is the
    one this project has verified — 6/12 to 11/12 on carry-heavy addition.
    """
    width = max(len(str(abs(a))), len(str(abs(b))))
    da = [int(c) for c in str(abs(a)).zfill(width)][::-1]
    db = [int(c) for c in str(abs(b)).zfill(width)][::-1]
    carry, out, calls = 0, [], 0
    for k in range(width):
        if op == "+":
            v = ask(PAIR["+"].format(x=da[k], y=db[k]))
            calls += 1
            if v is None:
                return None, calls
            v += carry
            out.append(int(v) % 10)
            carry = int(v) // 10
        else:
            top = da[k] - carry
            borrow = 0
            if top < db[k]:
                top += 10
                borrow = 1
            v = ask(PAIR["-"].format(x=top, y=db[k]))
            calls += 1
            if v is None:
                return None, calls
            out.append(int(v) % 10)
            carry = borrow
    if op == "+" and carry:
        out.append(carry)
    return int("".join(str(d) for d in out[::-1])), calls


def again(expr):
    """The model splits its own question further; the record checks it by inlining."""
    g, why = parse_graph(ask_text(SPLIT.format(expr=expr), n=256))
    if g is None:
        return None, f"parse: {why}"
    inlined, why = check_graph(g, expr)
    if inlined is None:
        return None, f"check: {why}"
    values = {}
    for k in g:
        body = g[k]
        for k2, v2 in values.items():
            body = re.sub(rf"\b{k2}\b", f"({v2})", body)
        v = ask(EVAL.format(expr=body))
        if v is None:
            return None, f"no number for {k}"
        values[k] = v
    return values[list(g)[-1]], "ok"


def main(n_facts=24, seed=11, out="data/custom/deepen.json"):
    rng = random.Random(int(seed))
    fs = facts(int(n_facts), rng)
    print(f"{len(fs)} inversions the model rephrased itself, then broken down again\n")
    print(f"{'the model asked':>26}{'want':>8}{'whole':>7}{'cols':>6}{'again':>7}  note")
    tally, rows = Counter(), []
    for a, op, b, c in fs:
        left = f"{a} {op} ?"
        want = b if op in "+*" else (a - c if op == "-" else a / c)
        q = first_question(ask_text(REWRITE.format(left=left, c=c)))
        v_whole = ask(ANSWER.format(question=q)) if q else None

        parsed = extract_sum(q)
        v_cols, note = v_whole, "not a column operation"
        if parsed and parsed[1] in "+-":
            x, o, y = parsed
            if x >= 0 and y >= 0 and (o == "+" or x >= y):
                v_cols, _ = columns(x, o, y)
                note = f"{x} {o} {y} by column"
                tally["decomposable"] += 1
        v_again, why = (again(f"{parsed[0]} {parsed[1]} {parsed[2]}") if parsed
                        else (None, "no arithmetic found"))

        got = {"whole": v_whole, "cols": v_cols, "again": v_again}
        for k, v in got.items():
            tally[k] += v is not None and abs(v - want) < 1e-9
        tally["rephrased"] += q is not None
        rows.append({"left": left, "c": c, "want": want, "question": q,
                     "note": note, "again_stage": why, **got})
        mark = {k: ("ok" if got[k] is not None and abs(got[k] - want) < 1e-9 else ".")
                for k in got}
        print(f"{(q or '-')[:26]:>26}{want:>8g}{mark['whole']:>7}{mark['cols']:>6}"
              f"{mark['again']:>7}  {note}")

    n = len(fs)
    print(f"\nthe model's own question, answered whole : {tally['whole']}/{n}")
    print(f"split into digit columns by the record   : {tally['cols']}/{n}"
          f"   ({tally['decomposable']}/{n} had columns to split)")
    print(f"the model splits it again as JSON        : {tally['again']}/{n}")
    print("\nThe chain is now: blank -> the model's own question -> smaller steps. Each link was")
    print("worth measuring separately because each one is a different thing going wrong.")
    summary = {"facts": n, **{f"{k}_correct": tally[k] for k in ("whole", "cols", "again")},
               "decomposable": tally["decomposable"], "rephrased": tally["rephrased"]}
    Path(out).write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
