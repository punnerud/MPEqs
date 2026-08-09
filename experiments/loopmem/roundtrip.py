#!/usr/bin/env python3
"""Give the model back its own answer and make it walk the graph the other way.

Everything so far needed the truth to say whether a run was right. This does not. The model
produces a decomposition and an answer; then it is handed the answer — not the problem — and
asked to invert each step. If `C = B / 2` and C is 175, then B is 350; if `B = 38 + A` and B is
350, then A is 312; if `A = 39 * 8` and A is 312, then the reconstructed operand must be 8, and 8
is in the original expression. Agreement between the two passes is the test.

It is a fair test for the same reason the residue rule is: a model that can produce the residual
should be able to consume it. The forward direction asks what a subexpression evaluates to; the
backward direction asks what an operand must have been. Neither is harder than the other, and
nothing about the second reveals the answer to the first — the backward pass never sees the
original expression, only the graph shape and the value the forward pass claimed.

What is being measured is not accuracy. It is whether AGREEMENT PREDICTS CORRECTNESS, because
that is what would make it usable: a run that agrees with itself can be trusted without a
grader, and a run that does not can be thrown away or retried.

Built on the JSON graph arm, which is the best decomposition measured here (11/20 against 3/20
for the whole expression), on the same twenty problems and the same seed.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from jsongraph import GRAPH, KEY, OPERAND, check_graph, parse_graph  # noqa: E402
from substitute import EVAL, WHOLE, gen_expr  # noqa: E402
from general import ask as gen_ask  # noqa: E402
from twoway import ask_num  # noqa: E402

INVERT = """<|endoftext|><|user|>
{left} = {result}

What is the missing number? Reply with only the number.
<|assistant|>
"""


def forward(expr):
    """The model writes the graph and evaluates it. Returns (answer, graph, values)."""
    g, why = parse_graph(gen_ask(GRAPH.format(expr=expr), n=320))
    if g is None:
        return None, None, None, f"parse: {why}"
    inlined, why = check_graph(g, expr)
    if inlined is None:
        return None, g, None, f"check: {why}"
    values = {}
    for k in g:
        body = g[k]
        for k2, v2 in values.items():
            body = re.sub(rf"\b{k2}\b", f"({v2})", body)
        v = ask_num(EVAL.format(expr=body))
        if v is None:
            return None, g, values, "evaluate: no number"
        values[k] = v
    return values[list(g)[-1]], g, values, "ok"


def backward(g, answer):
    """Walk the graph from the answer down, inverting one step at a time.

    The unknown in each step is its REFERENCE where it has one, and its last literal otherwise.
    That choice is what carries the chain: solving `? / 2 = 175` gives B, which is the result the
    next step down needs. Blanking the literal instead asks `B / ? = 175` with B not yet known,
    which is unanswerable — and the model duly answered it anyway, with 125.

    Every question has exactly one blank and is the same shape as the forward question. The
    backward pass sees the graph and the claimed answer and nothing else — not the original
    expression, not the truth.
    """
    recon, result = {}, {list(g)[-1]: answer}
    for k in reversed(list(g)):
        if k not in result:
            return recon, "a step's result was never derived"
        body = g[k]
        operands = OPERAND.findall(body)
        if not operands:
            return recon, f"{k} has no operands"
        refs = [o for o in operands if KEY.fullmatch(o)]
        unknown = refs[0] if refs else operands[-1]
        # Everything except the last operand is substituted with what the backward pass already
        # knows, so the question has one blank.
        left = body
        for o in operands:
            if o != unknown and KEY.fullmatch(o) and o in result:
                left = re.sub(rf"\b{o}\b", str(result[o]), left, count=1)
        left = re.sub(rf"(?<![\d.]){re.escape(unknown)}(?![\d.])", "?", left, count=1)
        v = ask_num(INVERT.format(left=left, result=result[k]))
        if v is None:
            return recon, f"no number inverting {k}"
        recon[k] = {"unknown": unknown, "reconstructed": v}
        if KEY.fullmatch(unknown):
            result[unknown] = v          # it was a reference, so that is its required value
        else:
            recon[k]["literal"] = float(unknown)
    return recon, "ok"


def agrees(recon, values, tol=1e-6):
    """Does the backward pass reproduce what the forward pass said, and the literals it used?"""
    for k, r in recon.items():
        u = r["unknown"]
        want = r.get("literal", values.get(u))
        if want is None or abs(want - r["reconstructed"]) > tol:
            return False, k
    return True, None


def main(n_tasks=20, seed=7, out="data/custom/roundtrip.json"):
    rng = random.Random(int(seed))
    tasks = [gen_expr(rng) for _ in range(int(n_tasks))]
    print(f"{len(tasks)} expressions. Forward, then the same model walking back from its own "
          f"answer.\n")
    print(f"{'expression':<24}{'truth':>8}{'whole':>7}{'forward':>9}{'right':>7}"
          f"{'agrees':>8}  first mismatch")
    rows, w_ok, f_ok = [], 0, 0
    agree_right = agree_wrong = dis_right = dis_wrong = 0
    for expr, truth in tasks:
        w = ask_num(WHOLE.format(expr=expr))
        w_ok += w == truth
        ans, g, values, stage = forward(expr)
        right = ans is not None and abs(ans - truth) < 1e-9
        f_ok += right
        ok, where, why = False, None, stage
        if g is not None and values is not None and ans is not None:
            recon, why = backward(g, ans)
            if why == "ok":
                ok, where = agrees(recon, values)
        # The four cells that decide whether agreement is worth anything.
        if ok and right:
            agree_right += 1
        elif ok and not right:
            agree_wrong += 1
        elif not ok and right:
            dis_right += 1
        else:
            dis_wrong += 1
        rows.append({"expr": expr, "truth": truth, "whole": w, "forward": ans,
                     "correct": right, "agrees": ok, "mismatch_at": where,
                     "graph": g, "values": values, "stage": why})
        print(f"{expr:<24}{truth:>8}{str(w):>7}{str(ans):>9}{'yes' if right else 'no':>7}"
              f"{'yes' if ok else 'no':>8}  {where or why}")

    n = len(tasks)
    agreed = agree_right + agree_wrong
    print(f"\nwhole expression in one go : {w_ok}/{n}")
    print(f"forward with a JSON graph  : {f_ok}/{n}")
    print(f"\n{'':<14}{'answer right':>14}{'answer wrong':>14}")
    print(f"{'agrees':<14}{agree_right:>14}{agree_wrong:>14}")
    print(f"{'disagrees':<14}{dis_right:>14}{dis_wrong:>14}")
    prec = agree_right / agreed if agreed else 0.0
    recall = agree_right / f_ok if f_ok else 0.0
    caught = dis_wrong / (n - f_ok) if n - f_ok else 0.0
    print(f"\nof the runs that agree with themselves, {prec:.0%} are right "
          f"(base rate {f_ok / n:.0%})")
    print(f"{recall:.0%} of correct runs are kept; {caught:.0%} of wrong runs are caught")
    print("\nThis is the number that matters: agreement is only useful if it beats the base")
    print("rate, because a check that passes everything is not a check.")
    summary = {"tasks": n, "whole_correct": w_ok, "forward_correct": f_ok,
               "agree_right": agree_right, "agree_wrong": agree_wrong,
               "disagree_right": dis_right, "disagree_wrong": dis_wrong,
               "precision_when_agreeing": prec, "base_rate": f_ok / n,
               "recall_of_correct": recall, "wrong_runs_caught": caught}
    Path(out).write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
