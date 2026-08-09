#!/usr/bin/env python3
"""Shown a worked example, does it generalise the principle or copy the answer?

The learning loop failed and the reason turned out to be specific and measurable. Given one
retrieved worked example alongside the fixed one, the 1B model emitted the FIXED example's graph
verbatim — `{"A": "2 * 5", "B": "7 + A", "C": "B / 3"}` for an expression containing none of
those numbers. Every one was refused by the record for inlining to a different value, which is
why its store never filled with rubbish, and also why it never filled at all.

That is the distinction worth measuring, so it is measured directly:

  COPIED       the graph is one of the shown examples, unchanged
  ADAPTED      the graph uses the new expression's own numbers
  VERIFIED     the record accepts it, whichever of those it is

And provenance is kept, which is the part that makes a failure useful: each attempt records
which example it was given. An example that is followed by refusals is one the model is
generalising from wrongly, and knowing which one that was is the precondition for repairing it —
you cannot look for a rule covering the old case and the new one without knowing what the old
case was.

Two models, because copying is a capacity failure and the 35B one should not do it.
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import MODELS, ask  # noqa: E402
from jsongraph import check_graph, parse_graph  # noqa: E402
from substitute import gen_expr  # noqa: E402

FIXED = ("(7 + 2 * 5) / 3", {"A": "2 * 5", "B": "7 + A", "C": "B / 3"})

PROMPT = """Break the expression into small steps and write them as JSON.

Rules:
- each key is one step using at most three numbers or earlier keys
- a key may be used in later steps
- the last key is the answer
- use the numbers from the expression you are given, not from the examples

{examples}
Expression: {expr}
"""


def block(pairs):
    return "\n".join(f"Example:\nExpression: {e}\n{json.dumps(g)}" for e, g in pairs) + "\n"


def numbers(x):
    return set(re.findall(r"\d+", x if isinstance(x, str) else json.dumps(x)))


def main(n_tasks=20, seed=11, out="data/custom/reuse.json"):
    n_tasks, seed = int(n_tasks), int(seed)
    rng = random.Random(seed)
    tasks = [gen_expr(rng) for _ in range(n_tasks)]

    # A second example, close in form to the fixed one, so there are two things to copy from.
    extra = ("(26 + 6 * 3) / 4", {"A": "6 * 3", "B": "26 + A", "C": "B / 4"})
    shown = [FIXED, extra]

    print(f"{n_tasks} expressions, {len(shown)} worked examples shown\n")
    print(f"{'model':<10}{'copied':>8}{'adapted':>9}{'verified':>10}{'correct':>9}")
    summary, rows = {}, []
    for model in MODELS:
        tally = Counter()
        provenance = Counter()
        for expr, truth in tasks:
            reply = ask(model, PROMPT.format(examples=block(shown), expr=expr), n=320)
            g, why = parse_graph(reply)
            if g is None:
                tally["unparsed"] += 1
                rows.append({"model": model, "expr": expr, "graph": None, "why": why})
                continue
            copied = next((e for e, gg in shown if gg == g), None)
            # "Adapted" is judged on whether the graph mentions the expression's own numbers
            # rather than only the examples' — the copy failure is exactly a numbers failure.
            own = numbers(expr) & numbers(g)
            tally["copied"] += copied is not None
            tally["adapted"] += bool(own) and copied is None
            inlined, reason = check_graph(g, expr)
            tally["verified"] += inlined is not None
            val = None
            if inlined is not None:
                try:
                    val = eval(inlined)  # noqa: S307 - our own generated arithmetic
                except Exception:  # noqa: BLE001
                    val = None
            tally["correct"] += val is not None and abs(val - truth) < 1e-9
            if copied is not None:
                provenance[copied] += 1
            rows.append({"model": model, "expr": expr, "graph": g, "copied_from": copied,
                         "verified": inlined is not None, "reason": reason,
                         "structurally_right": val is not None and abs(val - truth) < 1e-9})
        summary[model] = {**tally, "provenance": dict(provenance), "tasks": n_tasks}
        print(f"{model:<10}{tally['copied']:>6}/{n_tasks:<2}{tally['adapted']:>7}/{n_tasks:<2}"
              f"{tally['verified']:>8}/{n_tasks:<2}{tally['correct']:>7}/{n_tasks}")
        if provenance:
            for src, c in provenance.most_common():
                print(f"           copied from {src} {c} times")

    print("\nA copied graph is refused by the record every time, because it inlines to a")
    print("different value than the expression it was given. The check does not need to know")
    print("what copying is — reproducing the wrong expression is already disqualifying.")
    Path(out).write_text(json.dumps({"tasks": n_tasks, "shown": [e for e, _ in shown],
                                     "summary": summary, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
