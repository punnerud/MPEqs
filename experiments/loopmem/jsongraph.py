#!/usr/bin/env python3
"""Let the model write only the links, as JSON. The record owns the rules.

Phase 20 asked the model to name a substring to defer, and it did that well and it did not pay:
the substitutions it chose were things like `Y = 3`, perfectly reversible and compressing nothing.
The interface was the problem. Naming substrings is not what these models are good at; emitting
JSON is.

So the model writes a graph and nothing else. `38 + 39 * 8` is A, and A can be written in the
next step. Every rule that makes this safe lives in the record, not in the prompt:

  single assignment   a key is written once and never rewritten, so no step can invalidate an
                      earlier one — the property the work record was built around, now free
  defined before use  a key must exist before it is referenced, and the graph must be acyclic
  at most three in    a step combines at most three numbers or earlier keys, one result out
  reversible          inline every key back and the value must equal the original expression's

That last check is the same rule as phase 20 and it is the only one that can reject a graph the
model was confident about. Nothing else has to be trusted: not the arithmetic, not the ordering,
not the choice of what to name.

Same twenty expressions and the same seed as phase 20, so the three arms are directly comparable:
whole expression (3/20), substring substitution (2/20), and this.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from general import ask as gen_ask  # noqa: E402
from substitute import EVAL, WHOLE, gen_expr  # noqa: E402
from twoway import ask_num  # noqa: E402

GRAPH = """<|endoftext|><|user|>
Break the expression into small steps and write them as JSON.

Rules:
- each key is one step using at most three numbers or earlier keys
- a key may be used in later steps
- the last key is the answer

Example:
Expression: (7 + 2 * 5) / 3
{{"A": "2 * 5", "B": "7 + A", "C": "B / 3"}}

Expression: {expr}
<|assistant|>
"""

KEY = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\b")
NUM = re.compile(r"\d+(?:\.\d+)?")
OPERAND = re.compile(r"[A-Z][A-Za-z0-9_]*|\d+(?:\.\d+)?")


def parse_graph(reply):
    """First JSON object in the reply, with duplicate keys rejected rather than collapsed.

    `json.loads` silently keeps the last of a repeated key, which would turn a violation of
    single assignment into a valid-looking graph. The record has to see the violation.
    """
    m = re.search(r"\{[^{}]*\}", reply, re.S)
    if not m:
        return None, "no JSON object"
    raw = m.group(0)
    names = re.findall(r'"([^"]+)"\s*:', raw)
    if len(names) != len(set(names)):
        return None, "a key was written twice"
    try:
        g = json.loads(raw)
    except Exception:  # noqa: BLE001 - malformed JSON is a rejection, not a crash
        return None, "malformed JSON"
    if not g or not all(isinstance(v, str) for v in g.values()):
        return None, "values must be expressions"
    return g, None


def check_graph(g, expr):
    """Every structural rule, plus the reversibility rule. Returns (inlined, reason)."""
    order = list(g)
    seen, fanin = set(), []
    for k in order:
        body = g[k]
        refs = {t for t in KEY.findall(body)}
        if k in refs:
            return None, f"{k} refers to itself"
        undefined = refs - seen
        if undefined:
            return None, f"{k} uses {sorted(undefined)[0]} before it is written"
        n_in = len(OPERAND.findall(body))
        if n_in > 3:
            return None, f"{k} takes {n_in} inputs"
        fanin.append(n_in)
        seen.add(k)

    # Inline everything back. This is the residue rule: the graph is a legal decomposition only
    # if putting every key back reproduces the original value.
    inlined = g[order[-1]]
    for k in reversed(order[:-1]):
        inlined = re.sub(rf"\b{k}\b", f"({g[k]})", inlined)
    for _ in range(len(order)):
        for k in reversed(order):
            inlined = re.sub(rf"\b{k}\b", f"({g[k]})", inlined)
    if KEY.search(inlined):
        return None, "unresolved key after inlining"
    try:
        if eval(inlined) != eval(expr):  # noqa: S307 - our own generated arithmetic
            return None, "inlines to a different value"
    except Exception:  # noqa: BLE001 - unevaluable means unverifiable
        return None, "inlined form does not evaluate"
    return inlined, None


def fanout(g):
    """How many later steps use each key. One onward is the shape being aimed at."""
    counts = {k: 0 for k in g}
    for body in g.values():
        for t in set(KEY.findall(body)):
            if t in counts:
                counts[t] += 1
    return counts


def run_graph(expr):
    """Ask for the graph, check it, then have the model evaluate one step at a time."""
    # 160 tokens ran out before the JSON on the wordier replies — the model prefaces with
    # "To break down the given expression step-by-step..." and the object never arrives. A
    # truncated preamble is indistinguishable from a model that cannot produce JSON.
    reply = gen_ask(GRAPH.format(expr=expr), n=320)
    g, why = parse_graph(reply)
    if g is None:
        return None, {"stage": "parse", "reason": why, "calls": 1}
    inlined, why = check_graph(g, expr)
    if inlined is None:
        return None, {"stage": "check", "reason": why, "graph": g, "calls": 1}

    # Bottom-up. Every step is one small expression with its references already resolved to
    # numbers, which is the whole point of the decomposition.
    values, calls = {}, 1
    for k in g:
        body = g[k]
        for k2, v2 in values.items():
            body = re.sub(rf"\b{k2}\b", f"({v2})", body)
        v = ask_num(EVAL.format(expr=body))
        calls += 1
        if v is None:
            return None, {"stage": "evaluate", "reason": f"no number for {k}",
                          "graph": g, "calls": calls}
        values[k] = v
    fi = [len(OPERAND.findall(b)) for b in g.values()]
    fo = fanout(g)
    return values[list(g)[-1]], {"stage": "ok", "graph": g, "calls": calls,
                                 "steps": len(g), "max_fanin": max(fi),
                                 "max_fanout": max(fo.values()),
                                 "fanout_one": sum(1 for k in list(g)[:-1] if fo[k] == 1),
                                 "inner": len(g) - 1}


def main(n_tasks=20, seed=7, out="data/custom/jsongraph.json"):
    rng = random.Random(int(seed))
    tasks = [gen_expr(rng) for _ in range(int(n_tasks))]
    print(f"{len(tasks)} expressions. The model writes only the links; the record checks them.\n")
    print(f"{'expression':<24}{'truth':>8}{'whole':>8}{'graph':>8}{'steps':>7}"
          f"{'in':>4}{'out':>4}  stage")
    rows, w_ok, g_ok = [], 0, 0
    parsed = checked = 0
    for expr, truth in tasks:
        w = ask_num(WHOLE.format(expr=expr))
        v, info = run_graph(expr)
        w_ok += w == truth
        g_ok += v == truth
        parsed += info["stage"] != "parse"
        checked += info["stage"] in ("evaluate", "ok")
        rows.append({"expr": expr, "truth": truth, "whole": w, "graph_value": v, **info})
        print(f"{expr:<24}{truth:>8}{str(w):>8}{str(v):>8}"
              f"{info.get('steps', ''):>7}{info.get('max_fanin', ''):>4}"
              f"{info.get('max_fanout', ''):>4}  {info['stage']}"
              f"{'' if info['stage'] == 'ok' else ': ' + info.get('reason', '')}")

    n = len(tasks)
    ok_rows = [r for r in rows if r["stage"] == "ok"]
    summary = {
        "tasks": n, "whole_correct": w_ok, "graph_correct": g_ok,
        "parsed": parsed, "structurally_valid": checked,
        "mean_steps": sum(r["steps"] for r in ok_rows) / len(ok_rows) if ok_rows else 0,
        "max_fanin": max((r["max_fanin"] for r in ok_rows), default=0),
        "fanout_one_pct": (100.0 * sum(r["fanout_one"] for r in ok_rows)
                           / max(sum(r["inner"] for r in ok_rows), 1)),
    }
    print(f"\nwhole expression in one go   : {w_ok}/{n}")
    print(f"model wrote the links as JSON: {g_ok}/{n}")
    print(f"parsed as a single-assignment object : {parsed}/{n}")
    print(f"passed every rule including reversal : {checked}/{n}")
    print(f"mean steps {summary['mean_steps']:.1f}, widest step {summary['max_fanin']} inputs, "
          f"{summary['fanout_one_pct']:.0f}% of inner keys used exactly once onward")
    Path(out).write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
