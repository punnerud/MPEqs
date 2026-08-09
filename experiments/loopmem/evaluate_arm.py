#!/usr/bin/env python3
"""The model writes correct code and hallucinates the result. Execute it instead.

Three harness bugs and two wrong hypotheses later, the actual behaviour is visible. Asked to
work one step at a time, the small model does not decompose at all: it writes a Python block
computing the WHOLE expression, then fabricates an `output` block with a number that did not
come from running anything, and states DONE with it.

On `(120 / 4 + 15) * 2 - 30` it wrote `result = (120 / 4 + 15) * 2 - 30` — exactly right — and
then claimed the output was 90.0. The true value is 60; it dropped the `- 30` while pretending
to evaluate.

That is not a looping failure, not a state-tracking failure, and not something an external
memory can fix. The model CAN express the computation and CANNOT evaluate it. So the tool it
needs is an evaluator, and this measures how much of the gap that closes.
"""
import ast, json, operator, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from measure_loops import BIN, MODEL, TASKS

PROMPT = """Write a single Python expression that computes the answer. No explanation.

Problem: {problem}
Expression:"""

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.USub: operator.neg, ast.Pow: operator.pow}


def safe_eval(node):
    """Arithmetic only. Executing text a model wrote needs a grammar, not exec()."""
    if isinstance(node, ast.Expression):
        return safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](safe_eval(node.left), safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](safe_eval(node.operand))
    raise ValueError(f"not arithmetic: {ast.dump(node)[:40]}")


def ask(problem):
    Path("/tmp/ev.txt").write_text(PROMPT.format(problem=problem))
    out = subprocess.run([BIN, "-m", MODEL, "-f", "/tmp/ev.txt", "-n", "60", "--temp", "0",
                          "-no-cnv", "-st", "-ngl", "99"],
                         capture_output=True, text=True).stdout
    i = out.rfind("Expression:")
    tail = out[i + 11:] if i >= 0 else out
    tail = re.sub(r"<think>.*?</think>", " ", tail, flags=re.S).split("[end of text]")[0]
    # The model's own claimed answer, wherever it states one — kept so the two can be compared.
    claim = None
    m = re.search(r"(?:DONE:|answer is|=)\s*\\?\(?\\?boxed\{?\s*(-?\d+(?:\.\d+)?)", tail)
    if m:
        claim = float(m.group(1))
    for line in tail.splitlines():
        line = line.strip().strip("`").strip()
        if not line or line.startswith(("#", "python", "output")):
            continue
        line = re.sub(r"^\w+\s*=\s*", "", line)          # drop `result = `
        if re.fullmatch(r"[-+*/(). \d]+", line):
            return line, claim
    return None, claim


def main(out="data/custom/evaluate.json"):
    print(f"{len(TASKS)} problems. The model writes the expression; we evaluate it.\n")
    print(f"{'problem':<28}{'expression written':<30}{'evaluated':>10}{'truth':>7}{'ok':>4}")
    rows, ok_n, wrote_n = [], 0, 0
    for problem, answer in TASKS:
        expr, claim = ask(problem)
        val = None
        if expr:
            wrote_n += 1
            try:
                val = safe_eval(ast.parse(expr, mode="eval"))
            except Exception:
                val = None
        good = val is not None and abs(val - answer) < 1e-6
        ok_n += good
        rows.append({"problem": problem, "answer": answer, "expression": expr,
                     "evaluated": val, "model_claim": claim, "correct": bool(good)})
        print(f"{problem:<28}{(expr or '(none)')[:29]:<30}"
              f"{('%.4g' % val) if val is not None else '-':>10}{answer:>7}"
              f"{'  ok' if good else '   .':>4}")
    summary = {"tasks": len(TASKS), "wrote_expression": wrote_n, "correct": ok_n,
               "baseline_stepwise_correct": 0}
    print(f"\nexpression written for {wrote_n}/{len(TASKS)}, evaluated correctly {ok_n}/"
          f"{len(TASKS)}")
    print(f"stepwise baseline, same model, same problems: 0/{len(TASKS)}")
    Path(out).write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
