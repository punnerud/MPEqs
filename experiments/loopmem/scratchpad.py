#!/usr/bin/env python3
"""Pen and paper: one tiny operation at a time, with a graph checking the derivation.

The previous arm handed the model an evaluator by asking it for a Python expression. That is
not the design. The design is smaller and stricter: the model may perform ONE binary operation
on two things already written down, we evaluate that one operation, and both the result and the
TRANSITION are kept. The pad is what it can see; the graph is what checks it.

Three things the graph does that a value store alone does not:

  DERIVATION   every new value must come from operands already on the pad or literals in the
               problem. A step that introduces a number from nowhere is the model "suddenly
               thinking anew without rationalising it", and it is rejected with the pad shown
               back rather than silently accepted.
  COVERAGE     every literal in the problem must be consumed exactly as often as it appears.
               Finishing with an unused number means the answer cannot be right, and that is
               checkable before believing it.
  NON-DESTRUCTIVE  nothing is ever overwritten. A superseded value stays on the pad with its
               transition intact, so "check what you have done" is reading, not reconstructing.

Measured against the two baselines from the same model and the same problems: 0/8 working
free-form step by step, 6/8 writing one whole expression that we execute.
"""
import ast
import json
import operator
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from measure_loops import BIN, MODEL, TASKS  # noqa: E402

OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}

# The chat template, applied. Running this instruct model as raw completion produced four
# different malformed outputs in a row — markdown fences, a stray "1", a bare operator — and
# every one of them looked like a model failure. It was the missing template. A model asked
# outside the format it was trained in does not fail gracefully, it fails unrecognisably.
PROMPT = """<|endoftext|><|user|>
Problem: {problem}

Scratchpad so far:
{pad}

Numbers from the problem not yet used: {unused}
Values available to combine: {avail}
{refused}
Give exactly ONE arithmetic step combining two available values, written as `A + B` or `A - B`
or `A * B` or `A / B`. Use real numbers, not placeholders. Nothing else.
<|assistant|>
"""


class Pad:
    """The scratchpad and its derivation graph. Append-only."""

    def __init__(self, problem):
        self.problem = problem
        self.literals = [float(t) for t in re.findall(r"\d+(?:\.\d+)?", problem)]
        self.unused = list(self.literals)
        self.entries = []          # (name, a, op, b, value)
        self.values = {}           # name -> value

    def available(self):
        """Everything a next step may use: pad values and literals not yet consumed."""
        return [v for _, v in self.values.items()] + self.unused

    def take(self, x):
        """Consume x from the pad or from the unused literals. None if it is not available."""
        for name, v in self.values.items():
            if abs(v - x) < 1e-9:
                return ("pad", name)
        for i, v in enumerate(self.unused):
            if abs(v - x) < 1e-9:
                self.unused.pop(i)
                return ("literal", None)
        return None

    def apply(self, a, op, b):
        """Record one operation. Returns (name, value) or raises with the reason."""
        src_a = self.take(a)
        if src_a is None:
            raise ValueError(f"{a:g} is not on the pad and not an unused number")
        src_b = self.take(b)
        if src_b is None:
            if src_a[0] == "literal":
                self.unused.append(a)        # put it back; the step is rejected whole
            raise ValueError(f"{b:g} is not on the pad and not an unused number")
        if op == "/" and b == 0:
            raise ValueError("division by zero")
        val = OPS[op](a, b)
        name = f"v{len(self.entries) + 1}"
        self.entries.append((name, a, op, b, val))
        self.values[name] = val
        return name, val

    def render(self):
        if not self.entries:
            return "  (empty)"
        return "\n".join(f"  {n} = {a:g} {o} {b:g} = {v:g}" for n, a, o, b, v in self.entries)

    def covered(self):
        return not self.unused

    def truth(self):
        return safe_eval(ast.parse(self.problem, mode="eval"))


def safe_eval(node):
    m = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
         ast.Div: operator.truediv, ast.USub: operator.neg}
    if isinstance(node, ast.Expression):
        return safe_eval(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return m[type(node.op)](safe_eval(node.left), safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return m[type(node.op)](safe_eval(node.operand))
    raise ValueError("not arithmetic")


def ask(problem, pad, refused=()):
    unused = ", ".join(f"{v:g}" for v in pad.unused) or "(none)"
    avail = ", ".join(f"{v:g}" for v in pad.available()) or "(none)"
    # Feeding the refusals back is the whole "rationalise it" half. Without them the prompt is
    # identical every turn and a temperature-zero model returns the identical proposal: watched
    # step by step, it offered `15 + 25` four times running while the pad already held
    # `v1 = 15 + 25 = 40`. Detecting the repeat is useless if the detection is never shown.
    note = ""
    if refused:
        note = ("Already tried and rejected, do not repeat: "
                + "; ".join(f"{a:g} {o} {b:g}" for a, o, b in refused) + "\n")
    Path("/tmp/pad.txt").write_text(
        PROMPT.format(problem=problem, pad=pad.render(), unused=unused, avail=avail,
                      refused=note))
    out = subprocess.run(
        [BIN, "-m", MODEL, "-f", "/tmp/pad.txt", "-n", "24", "--temp", "0",
         "-no-cnv", "-st", "-ngl", "99"],
        capture_output=True, text=True).stdout
    i = out.rfind("<|assistant|>")
    tail = out[i + 13:] if i >= 0 else out
    tail = re.sub(r"<think>.*?</think>", " ", tail, flags=re.S).split("[end of text]")[0]
    tail = re.sub(r"```[a-zA-Z]*", " ", tail)
    # The one shape accepted, anywhere in the completion: number op number.
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([-+*/])\s*(-?\d+(?:\.\d+)?)", tail)
    return (float(m.group(1)), m.group(2), float(m.group(3))) if m else None


def solve(problem, answer, max_steps):
    pad = Pad(problem)
    rejected, unjustified, refused = 0, 0, []
    for _ in range(max_steps):
        prop = ask(problem, pad, refused)
        if prop is None:
            rejected += 1
            continue
        a, op, b = prop
        try:
            _, val = pad.apply(a, op, b)
        except ValueError:
            unjustified += 1        # the graph refused it: no derivation from what is written
            if (a, op, b) not in refused:
                refused.append((a, op, b))
            continue
        refused.clear()             # progress was made; the old refusals no longer apply
        # Done when every literal has been consumed and one value remains unused as an operand.
        if pad.covered() and abs(val - answer) < 1e-6:
            return True, pad, rejected, unjustified
    return False, pad, rejected, unjustified


def main(max_steps=8, out="data/custom/scratchpad-graph.json"):
    max_steps = int(max_steps)
    print(f"{len(TASKS)} problems, one binary operation per step, up to {max_steps} steps")
    print("the graph rejects any step not derivable from the pad\n")
    print(f"{'problem':<28}{'ok':>4}{'steps':>7}{'unjustified':>13}{'unparsed':>10}"
          f"{'covered':>9}  final")
    rows, n_ok = [], 0
    for problem, answer in TASKS:
        ok, pad, rejected, unjust = solve(problem, answer, max_steps)
        n_ok += ok
        final = pad.entries[-1][4] if pad.entries else None
        rows.append({"problem": problem, "answer": answer, "correct": bool(ok),
                     "steps": len(pad.entries), "unjustified": unjust,
                     "unparsed": rejected, "covered": pad.covered(),
                     "final": final,
                     "pad": [f"{n} = {a:g} {o} {b:g} = {v:g}" for n, a, o, b, v in pad.entries]})
        print(f"{problem:<28}{'ok' if ok else '.':>4}{len(pad.entries):>7}{unjust:>13}"
              f"{rejected:>10}{str(pad.covered()):>9}  "
              f"{('%g' % final) if final is not None else '-'}")

    summary = {"tasks": len(TASKS), "correct": n_ok,
               "baseline_stepwise": 0, "baseline_whole_expression": 6,
               "total_unjustified": sum(r["unjustified"] for r in rows),
               "total_steps": sum(r["steps"] for r in rows),
               "fully_covered": sum(1 for r in rows if r["covered"])}
    print(f"\ncorrect {n_ok}/{len(TASKS)}   "
          f"(free-form stepwise 0/{len(TASKS)}, whole expression executed 6/{len(TASKS)})")
    print(f"steps the graph refused as underivable: {summary['total_unjustified']}")
    print(f"runs that consumed every number in the problem: {summary['fully_covered']}"
          f"/{len(TASKS)}")
    Path(out).write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
