#!/usr/bin/env python3
"""Maybe the model prefers f(v1, v2, v3): the fill as a Python call instead of a JSON graph.

Phase 45's model-fill arm asked for the whole graph rewritten with values — and on the small
model it lost to record binding 9/60 against 14/60, with one template failing 4 of its 7 fills.
The proposal: ask for less. The template is a function; the model supplies the argument tuple,
`solve(43, 12, 11, 5)`, which is Python-call notation it has seen constantly, and the record
does the binding exactly as it would from its own positional extraction.

Three fill interfaces on identical template hits, small model, the record checking each result
by inlining:

  GRAPH    rewrite the whole JSON graph with values        (phase 45's arm)
  CALL     reply with solve(v1, v2, ...) only
  RECORD   the record binds positionally                   (the ceiling, no model)

The store is pre-seeded with the canonical template for every shape, so every task is a hit and
the interfaces are compared on the same footing rather than through the noise of what happened
to get stored first.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402
from jsongraph import check_graph, parse_graph  # noqa: E402
from repair import canonical_graph, inlines_ok  # noqa: E402
from substitute import gen_expr  # noqa: E402
from template import FILL, NUM, generalise, instantiate, skeleton  # noqa: E402

# "Reply with only the call" is load-bearing: without it the model restates the signature —
# `solve(v1, v2, v3, v4)` — and spends its whole budget explaining what it is about to do.
CALL = """A template solves this kind of expression:

def solve({params}):
    return {texpr}

Reply with only the call, using the numbers from the expression in the order they appear.
Nothing else.

Example:
Expression: (7 + 2 * 5) / 3
solve(7, 2, 5, 3)

Expression: {expr}
"""


def main(n_tasks=30, seed=13, model="olmoe-1b", out="data/custom/calltuple.json"):
    n_tasks, seed = int(n_tasks), int(seed)
    rng = random.Random(seed)
    tasks = [gen_expr(rng) for _ in range(n_tasks)]

    store = {}
    for expr, _ in tasks:                       # pre-seed: every shape known up front
        key = skeleton(expr)
        if key not in store:
            tpl = generalise(expr, canonical_graph(expr))
            if tpl:
                store[key] = tpl

    tally = {"graph": 0, "call": 0, "record": 0,
             "graph_verified": 0, "call_parsed": 0}
    rows = []
    for expr, truth in tasks:
        tpl = store.get(skeleton(expr))
        if tpl is None:
            continue
        values = NUM.findall(expr)

        # RECORD: positional, deterministic, the ceiling.
        _, g_rec = instantiate(tpl, values)
        rec_ok = inlines_ok(g_rec, expr)

        # GRAPH: phase 45's fill, the whole JSON rewritten.
        mg, _ = parse_graph(ask(model, FILL.format(
            texpr=tpl["expr"], tgraph=json.dumps(tpl["graph"]), expr=expr), n=320))
        graph_ok = mg is not None and check_graph(mg, expr)[0] is not None
        tally["graph_verified"] += graph_ok

        # CALL: only the argument tuple, bound by the record. The reply's LAST call is taken —
        # the model reproduces the worked example's call first, the same echo phase 27 hit.
        params = ", ".join(f"v{i + 1}" for i in range(tpl["nvars"]))
        reply = ask(model, CALL.format(params=params, texpr=tpl["expr"], expr=expr), n=160)
        # The last call whose arguments are numbers, EXCLUDING the worked example's own tuple.
        # Seven of eight failures in the first run were `7, 2, 5, 3` — the model echoed the
        # example and ran out of budget before writing its call, and "take the last" took the
        # echo. The example is mine, so its bytes are known and excludable; a reply that only
        # echoes now counts as unparsed rather than as a wrong answer.
        calls = [c for c in re.findall(r"solve\(([^)]*)\)", reply)
                 if NUM.search(c) and NUM.findall(c) != ["7", "2", "5", "3"]]
        got = NUM.findall(calls[-1]) if calls else []
        tally["call_parsed"] += len(got) == tpl["nvars"]
        call_ok = False
        if len(got) == tpl["nvars"]:
            _, g_call = instantiate(tpl, got)
            call_ok = inlines_ok(g_call, expr)

        tally["record"] += rec_ok
        tally["graph"] += graph_ok
        tally["call"] += call_ok
        rows.append({"expr": expr, "record": rec_ok, "graph": graph_ok, "call": call_ok,
                     "call_args": got})

    n = len(rows)
    print(f"{n} template hits, {model}, three fill interfaces\n")
    print(f"record binds positionally : {tally['record']}/{n}   (no model, the ceiling)")
    print(f"model rewrites the graph  : {tally['graph']}/{n}   "
          f"({tally['graph_verified']} verified)")
    print(f"model calls solve(...)    : {tally['call']}/{n}   "
          f"({tally['call_parsed']} parsed as the right arity)")
    print("\nThe call asks for strictly less: the numbers, in order, in a notation the model")
    print("has seen more of than JSON. Whether less to write means less to get wrong is the")
    print("measurement.")
    Path(out).write_text(json.dumps({"model": model, "hits": n, **tally, "rows": rows},
                                    indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
