#!/usr/bin/env python3
"""AIME with thinking ON: does graph+solver still add anything once the model may reason?

Every earlier arm prefilled an empty think block, so the comparison was between architectures
of work at equal budget — and olympiad scores were floors. This lets Qwen think, and asks the
only question that matters for the record: once the model can reason at length, is there still
a gap for the solver to close, or does thinking subsume the division of labour?

Sized for about twenty minutes: six problems, three arms.

  NOTHINK SOLO   the floor from the main experiment's setting (~25 s per problem)
  THINK SOLO     reason freely, answer at the end (~100 s)
  THINK GRAPH    reason freely, then emit ONLY the JSON arithmetic plan; the record executes
                 every step exactly. Thinking chooses the plan; it still never does the
                 arithmetic that lands in the answer.

The graph arm's failure mode is declared up front: if the think block does not close within
the token budget, no JSON ever arrives, and that counts against the arm — a plan that never
came is a miss, not a technicality.
"""
import json
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import BIN, MODELS  # noqa: E402
from olympiad import GRAPH, SOLO, last_number, load_problems, solve_graph  # noqa: E402

QWEN = MODELS["qwen-35b"][0]
THINK_TEMPLATE = "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"


def ask_think(prompt, n=1600):
    """Qwen with its thinking intact: no prefilled empty think block, generous budget."""
    Path("/tmp/aimethink.txt").write_text(THINK_TEMPLATE.format(p=prompt))
    out = subprocess.run(
        [BIN, "-m", QWEN, "-f", "/tmp/aimethink.txt", "-n", str(n), "--temp", "0",
         "-no-cnv", "-st", "-ngl", "99"], capture_output=True, text=True).stdout
    marks = list(re.finditer(r"<\|im_start\|>assistant|\bassistant\b", out))
    if marks:
        out = out[marks[-1].end():]
    closed = "</think>" in out
    out = re.sub(r"<think>.*?</think>", " ", out, flags=re.S)
    if not closed:
        out = re.sub(r"<think>.*", " ", out, flags=re.S)   # truncated thought: keep nothing
    return out.split("[end of text]")[0].strip(), closed


def solve_graph_think(problem):
    """Think, then plan. The record still executes; the model still never computes."""
    reply, closed = ask_think(GRAPH.format(problem=problem), n=1600)
    if not closed:
        return None, "think block never closed"
    from jsongraph import parse_graph
    g, why = parse_graph(reply)
    if g is None:
        return None, f"parse: {why}"
    # Reuse the exact executor from olympiad.py by round-tripping through its checks.
    import olympiad
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


def main(n_problems=6, seed=5, out="data/custom/aimethink.json"):
    import random
    n_problems = int(n_problems)
    rng = random.Random(int(seed))
    _, aime = load_problems()
    # The same sample the main run draws, truncated — comparable when that run resumes.
    picks = rng.sample(aime, min(15, len(aime)))[:n_problems]

    print(f"{len(picks)} AIME problems, Qwen 35B, thinking on\n")
    print(f"{'#':>3}{'truth':>7}{'nothink':>9}{'think':>8}{'think+graph':>13}  graph stage")
    from cutbig import ask as ask_nothink
    tally = {"nothink": 0, "think": 0, "tgraph": 0, "graphs_ran": 0, "unclosed": 0}
    rows = []
    for i, (problem, truth) in enumerate(picks):
        a0 = last_number(ask_nothink("qwen-35b", SOLO.format(problem=problem), n=512))
        t_reply, closed = ask_think(SOLO.format(problem=problem), n=1600)
        a1 = last_number(t_reply) if closed else None
        if not closed:
            tally["unclosed"] += 1
        a2, stage = solve_graph_think(problem)
        tally["nothink"] += a0 == truth
        tally["think"] += a1 == truth
        tally["tgraph"] += a2 == truth
        tally["graphs_ran"] += stage == "ok"
        rows.append({"truth": str(truth), "nothink": str(a0), "think": str(a1),
                     "think_graph": str(a2), "graph_stage": stage})
        print(f"{i:>3}{str(truth):>7}{str(a0):>9}{str(a1):>8}{str(a2):>13}  {stage}")

    n = len(picks)
    print(f"\nnothink solo   : {tally['nothink']}/{n}")
    print(f"think solo     : {tally['think']}/{n}   ({tally['unclosed']} thoughts truncated)")
    print(f"think + graph  : {tally['tgraph']}/{n}   ({tally['graphs_ran']} plans ran)")
    print("\nIf think+graph beats think, the solver still matters once reasoning is allowed —")
    print("the plan is what thinking is good for, and the arithmetic still is not.")
    Path(out).write_text(json.dumps({"n": n, **tally, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
