#!/usr/bin/env python3
"""The actual question: does storing what it solved make it solve more?

Everything in this thread was built for one loop. Break a problem into steps small enough to
hold; check each one with something that does not need the answer; keep what survived; and when
a new problem arrives, find the most similar thing already solved and work from that. This
measures the loop end to end, and the measurement is the one that matters — how much the model
solves on its own, and how much more it solves once it has a memory.

The memory is honest, which is the whole difficulty. A solution enters the store only if the
RECORD verified it, using the check from phase 21: inline the graph the model wrote and see
whether it reproduces the original expression's value. No ground truth is consulted at any
point during the run — the truth is used afterwards, to score, and never to decide what to
remember. A store filled by an oracle would prove nothing.

Three arms over the same 60 problems in the same order:

  SOLO           evaluate the whole expression in one call, the phase 20 baseline
  GRAPH          write a JSON graph and evaluate its steps, the phase 21 method
  GRAPH+MEMORY   the same, with the two most similar VERIFIED past solutions shown as worked
                 examples, retrieved by embedding

The learning claim is not that the memory arm scores higher overall — that could be the examples
acting as better prompting. It is that it improves AS THE STORE FILLS, so the first third and the
last third are reported separately.
"""
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from embednav import embed  # noqa: E402
from jsongraph import GRAPH, check_graph, parse_graph  # noqa: E402
from substitute import EVAL, WHOLE, gen_expr  # noqa: E402
from general import ask as gen_ask  # noqa: E402
from twoway import ask_num  # noqa: E402

# Two prompt defects, both of which were reported as facts about memory before they were
# found. Dropping the fixed worked example made the model emit prose and unquoted JSON. And
# without an explicit line forbidding it, a retrieved example makes the model reproduce an
# example's graph verbatim — every one refused by the record, which is why the store stayed at
# one entry for sixty tasks. With the line, copying falls to 1 in 20 (see `reuse.py`).
#
# The fixed worked example is not optional and dropping it was a bug. Without it the model
# writes `"step_1": 21 + 11` with unquoted values, or prose steps like "Add 30 and 39" — the
# memory arm was being scored on malformed JSON for every task before its store filled, which
# is a prompt defect and not a fact about memory. Retrieved examples now come IN ADDITION.
RECALL = """<|endoftext|><|user|>
Break the expression into small steps and write them as JSON.

Rules:
- each key is one step using at most three numbers or earlier keys
- a key may be used in later steps
- the last key is the answer
- use the numbers from the expression you are given, not from the examples

Example:
Expression: (7 + 2 * 5) / 3
{{"A": "2 * 5", "B": "7 + A", "C": "B / 3"}}

{examples}Expression: {expr}
<|assistant|>
"""


def solve_graph(expr, examples=""):
    """Write a graph, verify it against the expression, evaluate its steps. Returns
    (answer, verified, graph)."""
    # Always RECALL, so the only difference between the arms is whether examples were
    # retrieved — routing to a different prompt when the store is empty confounded the two.
    prompt = RECALL.format(examples=examples, expr=expr)
    g, _ = parse_graph(gen_ask(prompt, n=320))
    if g is None:
        return None, False, None
    inlined, _ = check_graph(g, expr)
    if inlined is None:
        # Refused by the record: the graph does not reproduce the expression, so whatever it
        # would have evaluated to is not an answer and must not be remembered.
        return None, False, g
    values = {}
    for k in g:
        body = g[k]
        for k2, v2 in values.items():
            body = re.sub(rf"\b{k2}\b", f"({v2})", body)
        v = ask_num(EVAL.format(expr=body))
        if v is None:
            return None, False, g
        values[k] = v
    return values[list(g)[-1]], True, g


def format_examples(store):
    if not store:
        return ""
    out = ["Worked examples:"]
    for expr, g in store:
        out.append(f"Expression: {expr}\n{json.dumps(g)}")
    return "\n".join(out) + "\n"


def main(n_tasks=60, seed=11, k=2, out="data/custom/learnloop.json"):
    n_tasks, seed, k = int(n_tasks), int(seed), int(k)
    rng = random.Random(seed)
    tasks = [gen_expr(rng) for _ in range(n_tasks)]
    print(f"{n_tasks} expressions, {k} retrieved examples, memory filled only by verified "
          f"solutions\n")

    store, store_vecs = [], []
    rows = []
    tally = {"solo": 0, "graph": 0, "memory": 0}
    verified_wrong = 0
    for t, (expr, truth) in enumerate(tasks):
        solo = ask_num(WHOLE.format(expr=expr))
        g_ans, _, _ = solve_graph(expr)

        # Retrieve before solving, from what has been verified so far and nothing else.
        examples = ""
        if store:
            qv = np.array(embed([expr])[0], dtype=np.float32)
            sims = np.array(store_vecs, dtype=np.float32) @ qv
            top = np.argsort(sims)[-k:][::-1]
            examples = format_examples([store[int(i)] for i in top])
        m_ans, verified, graph = solve_graph(expr, examples)

        # The store is written from the RECORD's verdict, never from the truth.
        if verified and graph is not None:
            store.append((expr, graph))
            store_vecs.append(embed([expr])[0])
            if m_ans is None or abs(m_ans - truth) > 1e-9:
                verified_wrong += 1

        tally["solo"] += solo == truth
        tally["graph"] += g_ans is not None and abs(g_ans - truth) < 1e-9
        tally["memory"] += m_ans is not None and abs(m_ans - truth) < 1e-9
        rows.append({"i": t, "expr": expr, "truth": truth, "solo": solo, "graph": g_ans,
                     "memory": m_ans, "verified": verified, "store_size": len(store)})
        if (t + 1) % 15 == 0:
            print(f"  after {t + 1:>3}: solo {tally['solo']:>2}  graph {tally['graph']:>2}  "
                  f"memory {tally['memory']:>2}   store {len(store)}")

    third = n_tasks // 3
    def rate(key, lo, hi):
        sel = rows[lo:hi]
        return sum(1 for r in sel
                   if r[key] is not None and abs(r[key] - r["truth"]) < 1e-9) / len(sel)

    print(f"\n{'arm':<16}{'total':>8}{'first third':>14}{'last third':>13}{'change':>9}")
    for key, label in (("solo", "whole expression"), ("graph", "graph, no memory"),
                       ("memory", "graph + memory")):
        a, b = rate(key, 0, third), rate(key, n_tasks - third, n_tasks)
        print(f"{label:<16}{tally[key]:>5}/{n_tasks:<2}{a:>14.2f}{b:>13.2f}{b - a:>+9.2f}")

    print(f"\nstore holds {len(store)} verified solutions, of which {verified_wrong} are "
          f"actually wrong ({1 - verified_wrong / max(len(store), 1):.0%} precision)")
    print("The store is written by the record's check, not by the answer key, so a wrong")
    print("solution that passes the check gets remembered and taught. That is the honest cost")
    print("of learning without a grader, and it is measured here rather than assumed away.")
    summary = {"tasks": n_tasks, "k": k, **{f"{a}_correct": tally[a] for a in tally},
               "store_size": len(store), "verified_wrong": verified_wrong,
               "first_third": {a: rate(a, 0, third) for a in tally},
               "last_third": {a: rate(a, n_tasks - third, n_tasks) for a in tally},
               "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
