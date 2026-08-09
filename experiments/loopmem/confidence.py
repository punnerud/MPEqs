#!/usr/bin/env python3
"""A safety measure per step: is this value actually strongly supported?

Phase 15 could not test whether iterating longer helps, because a random driver never succeeds
and there was no success rate to compare. The way past that is not a better driver — it is to
stop accepting steps blindly and score them, which is the N x N idea applied as a confidence
measure rather than as a retrieval index.

The signal available without a model is MULTI-PATH SUPPORT. Enumerate the derivations a set of
givens admits, and count how many *independent* ones reach each value. A value that only one
route produces is a coincidence of that route; a value several unrelated routes agree on is
supported. This is the same bottleneck reasoning the chain work arrived at — a claim is as
strong as the weakest thing it depends on, and independent agreement is what removes that
dependence.

Two things are measured, and the second is the one that decides whether any of it is usable:

  DOES SUPPORT PREDICT TRUTH   over all reachable values, is the true answer among the
                               best-supported? If it is not, support is not a safety signal.
  DOES IT RAISE THE BASE RATE  a driver that only accepts well-supported steps should solve
                               more than one that accepts anything, which is exactly what
                               phase 15 lacked the power to see.

Exhaustive on these problems: five givens admit a few thousand complete derivations, so the
question can be answered by counting rather than by sampling.
"""
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from measure_loops import TASKS  # noqa: E402

OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
       "*": lambda a, b: a * b, "/": lambda a, b: a / b if b else None}


def derivations(values, depth=0, seen=None):
    """Every complete derivation over a multiset of values, yielding (result, path).

    A path is the sequence of operations taken, so two derivations that differ only in the
    order of independent steps still count as different routes to the same value — which is
    the point: agreement across routes is the evidence.
    """
    if len(values) == 1:
        yield values[0], ()
        return
    n = len(values)
    for i, j in combinations(range(n), 2):
        rest = [values[k] for k in range(n) if k not in (i, j)]
        a, b = values[i], values[j]
        for op in "+-*/":
            for x, y in ((a, b), (b, a)):
                if op in "+*" and (x, y) != (a, b):
                    continue                      # commutative: one order is enough
                v = OPS[op](x, y)
                if v is None or abs(v) > 1e9:
                    continue
                for res, path in derivations(rest + [v], depth + 1):
                    yield res, ((x, op, y, v),) + path


def support(problem, answer, cap=200000):
    """How many distinct routes reach each value, and where the true answer ranks."""
    givens = [float(t) for t in re.findall(r"\d+(?:\.\d+)?", problem)]
    counts = defaultdict(int)
    routes = 0
    for res, _ in derivations(givens):
        counts[round(res, 6)] += 1
        routes += 1
        if routes >= cap:
            break
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    pos = next((i for i, (v, _) in enumerate(ranked) if abs(v - answer) < 1e-6), None)
    return {
        "problem": problem, "answer": answer, "routes": routes,
        "distinct_values": len(counts),
        "answer_support": counts.get(round(float(answer), 6), 0),
        "top_support": ranked[0][1] if ranked else 0,
        "top_value": ranked[0][0] if ranked else None,
        "answer_rank": pos,
        "top_is_answer": pos == 0,
        "answer_in_top5": pos is not None and pos < 5,
    }


def main(out="data/custom/confidence.json"):
    print("Exhaustive derivations. Does multi-path support point at the true answer?\n")
    print(f"{'problem':<28}{'routes':>9}{'values':>8}{'ans supp':>10}{'top supp':>10}"
          f"{'rank':>6}{'top?':>6}")
    rows = []
    for problem, answer in TASKS:
        r = support(problem, answer)
        rows.append(r)
        print(f"{problem:<28}{r['routes']:>9}{r['distinct_values']:>8}"
              f"{r['answer_support']:>10}{r['top_support']:>10}"
              f"{str(r['answer_rank']):>6}{'yes' if r['top_is_answer'] else 'no':>6}")

    n = len(rows)
    top = sum(1 for r in rows if r["top_is_answer"])
    top5 = sum(1 for r in rows if r["answer_in_top5"])
    # The baseline a support measure has to beat: picking uniformly among distinct values.
    chance = sum(1.0 / r["distinct_values"] for r in rows) / n
    summary = {"tasks": n, "top_is_answer": top, "answer_in_top5": top5,
               "mean_distinct_values": sum(r["distinct_values"] for r in rows) / n,
               "chance_of_picking_answer": chance}
    print(f"\nbest-supported value IS the answer: {top}/{n}")
    print(f"answer among the five best supported: {top5}/{n}")
    print(f"picking at random among distinct values would give {chance:.3f} per problem")
    print("\nIf support ranks the answer first, a driver can accept only well-supported steps")
    print("and the base rate rises. If it does not, support is not a safety signal here and")
    print("no amount of iteration is protected by it.")
    Path(out).write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
