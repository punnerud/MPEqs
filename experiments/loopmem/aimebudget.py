#!/usr/bin/env python3
"""AIME at a real working budget, and the agreement gate where its guarantee can break.

The earlier olympiad runs were floors by construction: nothink solo 0/15, and the first
thinking run starved — n=1600 left four of six think blocks unclosed. The declared next
step was a real budget. This phase runs it: ten problems (the same six as before, same
seed, extended to ten), three arms, every call at n=6000 of visible working.

  A  think solo, standard phrasing
  B  think solo, re-read phrasing — a different surface, so temp-0 cannot copy A
  G  think, then emit ONLY the JSON arithmetic plan; the record executes every step
     with exact Fractions (the model never computes a number that lands in the answer)

And the measurement the equation phases cannot make: phases 85-88 ran the agreement
gate on translations where its risk number stayed zero across 72 deliveries — but
those were reads, and these are OLYMPIAD SOLVES, where two attempts can walk the same
wrong road for the same tempting reason. The gate delivers only when A and B agree;
how often agreement is WRONG here is the number this phase exists to find. A guarantee
is only understood when its breaking point is measured.
"""
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import re  # noqa: E402

from cutbig import BIN, MODELS  # noqa: E402
from jsongraph import parse_graph  # noqa: E402
from olympiad import GRAPH, RECHECK, SOLO, last_number, load_problems  # noqa: E402

BUDGET = 6000
QWEN = MODELS["qwen-35b"][0]

# Measured before this run, bisected at inspection: this llama.cpp build at temp 0 in
# raw mode either EOSes INSTANTLY on long problems (no prefill — which is what the
# earlier run's four "unclosed thoughts" almost certainly were) or closes the think
# block instantly when <think> is prefilled, then writes its working IN THE OPEN. So
# the budget below buys LONG VISIBLE WORKING, not hidden thinking — declared, and what
# this phase measures.
T2 = "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def ask_work(prompt, n=BUDGET):
    Path("/tmp/aimebudget.txt").write_text(T2.format(p=prompt))
    out = subprocess.run(
        [BIN, "-m", QWEN, "-f", "/tmp/aimebudget.txt", "-n", str(n), "--temp", "0",
         "-no-cnv", "-st", "-ngl", "99"], capture_output=True, text=True).stdout
    tail = out.rsplit("<think>", 1)[-1].split("[end of text]")[0]
    if "</think>" in tail:
        tail = tail.rsplit("</think>", 1)[-1]
    return tail.strip()


def graph_arm(problem):
    reply = ask_work(GRAPH.format(problem=problem))
    if len(reply) < 5:
        return None, "died at the prompt"
    g, why = parse_graph(reply)
    if g is None:
        return None, f"parse: {why}"
    values = {}
    for k, body in g.items():
        expr = body
        for k2, v2 in values.items():
            expr = re.sub(rf"\b{k2}\b", f"({v2})", expr)
        if not re.fullmatch(r"[\d\s+*/().-]+", expr):
            return None, f"{k} is not arithmetic"
        try:
            values[k] = Fraction(eval(expr))  # noqa: S307 - digits and operators only
        except Exception:  # noqa: BLE001
            return None, f"{k} does not evaluate"
    return values[list(g)[-1]], "ok"


def main(n_problems=10, seed=5, out="data/custom/aimebudget.json"):
    import random
    n_problems = int(n_problems)
    rng = random.Random(int(seed))
    _, aime = load_problems()
    picks = rng.sample(aime, min(15, len(aime)))[:n_problems]

    print(f"{len(picks)} AIME problems, Qwen 35B, visible working at n={BUDGET}\n")
    tally = {"a": 0, "b": 0, "g": 0, "unclosed": 0, "graphs_ran": 0,
             "gate_delivered": 0, "gate_right": 0, "gate_wrong": 0, "flagged": 0,
             "vote_right": 0, "agree3": 0}
    rows = []
    for i, (problem, truth) in enumerate(picks):
        ra = ask_work(SOLO.format(problem=problem))
        a = last_number(ra)
        rb = ask_work(RECHECK.format(problem=problem))
        b = last_number(rb)
        g, stage = graph_arm(problem)
        tally["unclosed"] += (len(ra) < 5) + (len(rb) < 5)
        tally["graphs_ran"] += stage == "ok"
        tally["a"] += a == truth
        tally["b"] += b == truth
        tally["g"] += g == truth

        if a is not None and a == b:                    # the two-arm agreement gate
            tally["gate_delivered"] += 1
            tally["gate_right"] += a == truth
            tally["gate_wrong"] += a != truth
        else:
            tally["flagged"] += 1
        votes = [x for x in (a, b, g) if x is not None]
        vote = max(set(votes), key=votes.count) if votes else None
        tally["vote_right"] += vote == truth
        tally["agree3"] += len(votes) == 3 and len(set(votes)) == 1

        rows.append({"truth": str(truth), "a": str(a), "b": str(b), "g": str(g),
                     "graph_stage": stage})
        print(f"{i:>3} truth {str(truth):>6}  A {str(a):>8}  B {str(b):>8}  "
              f"G {str(g):>8}  {stage}")

    n = len(picks)
    print(f"\nthink solo A      : {tally['a']}/{n}")
    print(f"think solo B      : {tally['b']}/{n}   "
          f"({tally['unclosed']} of {2 * n} replies died at the prompt)")
    print(f"think + graph     : {tally['g']}/{n}   ({tally['graphs_ran']} plans ran)")
    print(f"agreement gate    : delivered {tally['gate_delivered']}/{n}, right "
          f"{tally['gate_right']}, WRONG {tally['gate_wrong']}, flagged "
          f"{tally['flagged']}")
    print(f"three-arm vote    : {tally['vote_right']}/{n} (all three agree on "
          f"{tally['agree3']})")
    print("\nThe gate's zero was earned on reads; this is where solves can agree for")
    print("the wrong reason, and whatever the WRONG count is above, it is the honest")
    print("price of delivering olympiad answers without a grader.")
    summary = {"n": n, "budget": BUDGET, **tally, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
