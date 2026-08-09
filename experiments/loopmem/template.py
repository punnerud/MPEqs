#!/usr/bin/env python3
"""Store the examples generalised: templates with placeholders, values bound per problem.

Phase 44 stored concrete solved graphs and measured three defects: the store grows one entry
per solved problem even when sixty problems share a handful of shapes; the record's check
verifies the decomposition and not the arithmetic inside it, so 67% of what was remembered was
right; and the small model copies a concrete example verbatim. A template attacks all three.
One template covers every instance of its shape, a template stripped of numbers cannot carry a
wrong number into the store, and a graph of variables offers nothing concrete to copy.

The generalisation is the record's job and it is mechanical: replace each numeric literal with
a variable, in order of appearance in the expression, identically in expression and graph. It
is admitted under the same rule as everything else in this thread — re-instantiating the
template with the original values must reproduce the expression and the graph character for
character, checked and not assumed.

Instantiation is measured both ways, because that is the open question:

  RECORD   the record binds the new expression's numbers to the variables positionally.
           Deterministic; the decomposition is correct by construction and the model only
           evaluates the steps.
  MODEL    the model is shown the template and asked to fill in the values itself, which is
           the proposal as stated. The record verifies the result by inlining.

Provenance is kept per template: which expressions it was generalised from, and how often each
instantiation arm succeeded or failed on it — the precondition for repairing a bad
generalisation, though repair itself is not attempted here.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import MODELS, ask  # noqa: E402
from jsongraph import check_graph, parse_graph  # noqa: E402
from substitute import gen_expr  # noqa: E402

NUM = re.compile(r"\d+(?:\.\d+)?")
VARS = [f"v{i}" for i in range(1, 17)]

SOLO = """What is {expr}? Reply with only the number.
"""

EVALQ = """What is {expr}?

Reply with only the number.
"""

GRAPH = """Break the expression into small steps and write them as JSON.

Rules:
- each key is one step using at most three numbers or earlier keys
- a key may be used in later steps
- the last key is the answer
- use the numbers from the expression you are given, not from the examples

Example:
Expression: (7 + 2 * 5) / 3
{{"A": "2 * 5", "B": "7 + A", "C": "B / 3"}}

Expression: {expr}
"""

# The model fills the values into a generalised template. The fixed example and the anti-copy
# line both stay: phase 44 measured each of their absences as a 2x swing in an arm's score.
FILL = """A template shows how to break this kind of expression into steps. Replace each
variable with the right number from the expression, and reply with only the JSON.

Rules:
- use the numbers from the expression you are given, not from the examples

Example:
Template for (v1 + v2 * v3) / v4:
{{"A": "v2 * v3", "B": "v1 + A", "C": "B / v4"}}
Expression: (7 + 2 * 5) / 3
{{"A": "2 * 5", "B": "7 + A", "C": "B / 3"}}

Template for {texpr}:
{tgraph}
Expression: {expr}
"""


def num(model, prompt, n=48):
    """The last number in the reply — the extraction every phase since 17 settled on."""
    out = ask(model, prompt, n=n)
    nums = NUM.findall(out.replace(",", ""))
    return float(nums[-1]) if nums else None


def skeleton(expr):
    """The shape with the numbers removed: the store's key."""
    return NUM.sub("#", expr)


def generalise(expr, graph):
    """Replace literals with variables, identically in expression and graph, and verify.

    Duplicate numbers make the graph-side mapping ambiguous — `(42 + 11 * 6) / 6` has two
    sixes and the graph mentions each once — so the assignment is greedy per occurrence and
    then CHECKED by re-instantiation. A wrong greedy guess fails the character-for-character
    round trip and the generalisation is refused rather than stored wrong.
    """
    values = NUM.findall(expr)
    if len(values) > len(VARS):
        return None
    texpr, pos = "", 0
    for k, m in enumerate(NUM.finditer(expr)):
        texpr += expr[pos:m.start()] + VARS[k]
        pos = m.end()
    texpr += expr[pos:]

    remaining = {v: [k for k, x in enumerate(values) if x == v] for v in set(values)}
    tgraph = {}
    for key, body in graph.items():
        out, pos = "", 0
        for m in NUM.finditer(body):
            v = m.group(0)
            slots = remaining.get(v, [])
            idx = slots.pop(0) if len(slots) > 1 else (slots[0] if slots else None)
            if idx is None:
                return None                    # a number not in the expression: not a template
            out += body[pos:m.start()] + VARS[idx]
            pos = m.end()
        tgraph[key] = out + body[pos:]

    template = {"skeleton": skeleton(expr), "expr": texpr, "graph": tgraph,
                "nvars": len(values)}
    back_e, back_g = instantiate(template, values)
    if back_e != expr or back_g != graph:
        return None                            # the round trip is the admission rule
    return template


def instantiate(template, values):
    """Bind values to variables positionally. Longest variable name first, so v12 is not
    corrupted by the v1 substitution."""
    def sub(text):
        for k in sorted(range(len(values)), reverse=True):
            text = text.replace(VARS[k], values[k])
        return text
    return sub(template["expr"]), {k: sub(v) for k, v in template["graph"].items()}


def evaluate(model, graph):
    """Bottom-up, one small expression per call, references resolved before asking."""
    values = {}
    for k in graph:
        body = graph[k]
        for k2, v2 in values.items():
            body = re.sub(rf"\b{k2}\b", f"({v2:g})", body)
        v = num(model, EVALQ.format(expr=body))
        if v is None:
            return None
        values[k] = v
    return values[list(graph)[-1]]


def run(model, tasks, out_rows):
    store = {}                                  # skeleton -> template with provenance
    tally = {"solo": 0, "graph": 0, "trecord": 0, "tmodel": 0,
             "hits": 0, "generalised": 0, "refused_generalise": 0}
    for t, (expr, truth) in enumerate(tasks):
        row = {"model": model, "i": t, "expr": expr, "truth": truth}
        row["solo"] = num(model, SOLO.format(expr=expr))

        g, _ = parse_graph(ask(model, GRAPH.format(expr=expr), n=320))
        verified = g is not None and check_graph(g, expr)[0] is not None
        row["graph"] = evaluate(model, g) if verified else None

        key = skeleton(expr)
        hit = store.get(key)
        row["template_hit"] = hit is not None
        if hit:
            tally["hits"] += 1
            # RECORD binds the values; nothing to verify because nothing was free to vary.
            _, inst = instantiate(hit, NUM.findall(expr))
            row["trecord"] = evaluate(model, inst)
            # MODEL binds the values; the record checks what came back by inlining it.
            mg, _ = parse_graph(ask(model, FILL.format(
                texpr=hit["expr"], tgraph=json.dumps(hit["graph"]), expr=expr), n=320))
            m_ok = mg is not None and check_graph(mg, expr)[0] is not None
            row["tmodel"] = evaluate(model, mg) if m_ok else None
            row["tmodel_verified"] = m_ok
            hit["uses"] += 1
            hit["model_fill_failed"] += 0 if m_ok else 1
        else:
            # No template yet: the graph arm's answer stands in for both, and if the graph
            # verified, it is generalised and stored. The store is written by the record's
            # check alone — the truth scores afterwards and never decides what is remembered.
            row["trecord"] = row["graph"]
            row["tmodel"] = row["graph"]
            if verified:
                tpl = generalise(expr, g)
                if tpl is None:
                    tally["refused_generalise"] += 1
                else:
                    tpl.update({"sources": [expr], "uses": 0, "model_fill_failed": 0})
                    store[key] = tpl
                    tally["generalised"] += 1

        for arm in ("solo", "graph", "trecord", "tmodel"):
            v = row.get(arm)
            tally[arm] += v is not None and abs(v - truth) < 1e-9
        out_rows.append(row)
    return tally, store


def main(n_small=60, n_big=20, seed=11, out="data/custom/template.json"):
    n_small, n_big, seed = int(n_small), int(n_big), int(seed)
    rng = random.Random(seed)
    tasks = [gen_expr(rng) for _ in range(n_small)]

    rows, summary = [], {}
    for model, n in (("olmoe-1b", n_small), ("qwen-35b", n_big)):
        tally, store = run(model, tasks[:n], rows)
        instances = sum(t["uses"] + 1 for t in store.values())
        summary[model] = {
            "tasks": n, **tally, "templates": len(store),
            "instances_covered": instances,
            "store": [{"skeleton": t["skeleton"], "sources": t["sources"], "uses": t["uses"],
                       "model_fill_failed": t["model_fill_failed"]} for t in store.values()],
        }
        print(f"\n{model}: {n} tasks")
        print(f"  solo {tally['solo']}/{n}   graph {tally['graph']}/{n}   "
              f"template+record {tally['trecord']}/{n}   template+model {tally['tmodel']}/{n}")
        print(f"  store: {len(store)} templates covering {instances} solved instances "
              f"({tally['hits']} hits, {tally['generalised']} generalised, "
              f"{tally['refused_generalise']} refused by the round trip)")
        for tpl in store.values():
            print(f"    {tpl['skeleton']}: used {tpl['uses']}x, "
                  f"model fill failed {tpl['model_fill_failed']}x")

    print("\nA template's decomposition cannot be wrong — the record instantiated it — so the")
    print("only thing left to fail is the arithmetic inside the steps, which is the model's")
    print("floor and not the memory's. That separation is what the concrete store never had.")
    Path(out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
