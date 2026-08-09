#!/usr/bin/env python3
"""Answer plus a record-executable check — the mechanical gate olympiad ground lacked.

Phase 89 measured the agreement gate's breaking point: on solves, two phrasings agree
on the same wrong number (2 of 3 deliveries wrong), because agreement is correlated
failure once the task is solving rather than reading. The replacement the session's
own law demands is a MECHANICAL verifier — and often, checking a number against the
problem's conditions is far easier to mechanise than finding it.

So the model now delivers two things in one reply: the answer, and a CHECK — a tiny
Python function in a whitelisted subset (arithmetic, for-over-range, comparisons; no
imports, no attributes, no while) that the record executes. And because the model
authors the check, the record AUDITS THE CHECK before trusting it, with decoys:

    deliver a  <=>  check parses in the subset
                AND check(a) is True
                AND check(d) is False for ALL six decoys (a±1, a±7, two seeded
                    randoms in 0..999)

A vacuous check (return True) dies on the first decoy. A circular check (recompute
the same wrong number, compare) survives decoys and delivers the wrong answer — that
is agreement in disguise, it cannot be excluded mechanically, and the phase counts a
mechanical suspicion marker for it: checks whose source embeds the answer literal.
The risk number is deliveries whose answer is wrong; phase 89's gate scored 1 right,
2 wrong on the same ten problems. This gate's job is to beat that, or to flag.
"""
import ast
import json
import math
import random
import re
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aimebudget import ask_work  # noqa: E402
from olympiad import load_problems  # noqa: E402

FORBIDDEN = (ast.Attribute, ast.Import, ast.ImportFrom, ast.While, ast.Lambda,
             ast.ClassDef, ast.With, ast.Try, ast.Raise, ast.Global, ast.Nonlocal,
             ast.Delete, ast.Yield, ast.YieldFrom, ast.Starred, ast.JoinedStr,
             ast.AsyncFunctionDef, ast.Await)
FUNCS = {"range", "len", "sum", "abs", "min", "max", "int", "gcd", "all", "any",
         "sorted", "set", "factorial", "comb", "divmod", "pow", "round", "enumerate"}


def _range(*a):
    r = range(*a)
    if len(r) > 2_000_000:
        raise ValueError("range too big")
    return r


SAFE_GLOBALS = {"__builtins__": {}, "range": _range, "len": len, "sum": sum,
                "abs": abs, "min": min, "max": max, "int": int, "gcd": math.gcd,
                "all": all, "any": any, "sorted": sorted, "set": set,
                "factorial": math.factorial, "comb": math.comb, "divmod": divmod,
                "pow": pow, "round": round, "enumerate": enumerate}


def subset_ok(src):
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return None, f"syntax: {e.msg}"
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN):
            return None, f"forbidden: {type(node).__name__}"
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCS:
                return None, "call outside the whitelist"
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return None, "dunder name"
    return tree, "ok"


def run_check(src, candidate, timeout=5):
    g = dict(SAFE_GLOBALS)
    try:
        signal.alarm(timeout)
        exec(compile(src, "<check>", "exec"), g)  # noqa: S102 - AST-whitelisted above
        result = g["check"](candidate)
        signal.alarm(0)
        return bool(result), "ok"
    except Exception as e:  # noqa: BLE001
        signal.alarm(0)
        return None, f"{type(e).__name__}: {str(e)[:60]}"


PROMPT = """{problem}

Work it out briefly. Then output exactly two things:

Answer: <number>

CHECK:
```python
def check(a):
    # Verify that a satisfies the problem's CONDITIONS. Use only arithmetic,
    # for-loops over range(...), and comparisons. No imports. Return True or False.
    ...
```

The check must test the problem's conditions directly — a check that just recomputes
your answer and compares proves nothing."""


def extract(reply):
    ans = None
    m = re.search(r"Answer:\s*(-?\d+)", reply)
    if m:
        ans = int(m.group(1))
    code = None
    cm = re.search(r"```(?:python)?\s*(def check\(.*?)```", reply, re.S)
    if cm:
        code = cm.group(1).strip()
    return ans, code


def main(n_inspect=0, out="data/custom/aimecheck.json"):
    n_inspect = int(n_inspect)
    rng = random.Random(5)
    _, aime = load_problems()
    picks = rng.sample(aime, 15)[:10]
    decoy_rng = random.Random(41)

    if n_inspect:
        for problem, truth in picks[:2]:
            reply = ask_work(PROMPT.format(problem=problem))
            ans, code = extract(reply)
            print(f"truth {truth}  answer {ans}\n--- check ---\n{code}\n")
        return

    tally = {"answer_right": 0, "check_parsed": 0, "check_ran": 0,
             "delivered": 0, "delivered_right": 0, "delivered_wrong": 0,
             "flagged": 0, "vacuous_killed": 0, "circular_suspect": 0}
    rows = []
    for i, (problem, truth) in enumerate(picks):
        reply = ask_work(PROMPT.format(problem=problem))
        ans, code = extract(reply)
        tally["answer_right"] += ans == int(truth)
        verdict = "flagged"
        why = "no answer" if ans is None else "no check block"
        if ans is not None and code:
            tree, why = subset_ok(code)
            if tree:
                tally["check_parsed"] += 1
                ok_self, why = run_check(code, ans)
                if ok_self is not None:
                    tally["check_ran"] += 1
                if ok_self:
                    decoys = [ans - 1, ans + 1, ans - 7, ans + 7]
                    while len(decoys) < 6:
                        d = decoy_rng.randint(0, 999)
                        if d != ans and d not in decoys:
                            decoys.append(d)
                    rejected = [run_check(code, d)[0] is False for d in decoys]
                    if all(rejected):
                        tally["delivered"] += 1
                        right = ans == int(truth)
                        tally["delivered_right"] += right
                        tally["delivered_wrong"] += not right
                        verdict = f"DELIVERED {ans}" + ("" if right else " (WRONG)")
                        if re.search(rf"\b{ans}\b", code):
                            tally["circular_suspect"] += 1
                    else:
                        tally["vacuous_killed"] += 1
                        why = f"accepted {rejected.count(False)} decoys"
                elif ok_self is False:
                    why = "check rejects its own answer"
        if verdict == "flagged":
            tally["flagged"] += 1
        rows.append({"truth": str(truth), "answer": ans, "verdict": verdict,
                     "why": why if verdict == "flagged" else "",
                     "check": (code or "")[:400]})
        print(f"{i:>3} truth {str(truth):>6}  ans {str(ans):>6}  "
              f"{verdict}{('  [' + why + ']') if verdict == 'flagged' else ''}")

    n = len(picks)
    print(f"\nanswers right outright     : {tally['answer_right']}/{n}")
    print(f"checks in subset / ran     : {tally['check_parsed']} / "
          f"{tally['check_ran']}")
    print(f"gate: delivered {tally['delivered']} (right {tally['delivered_right']}, "
          f"WRONG {tally['delivered_wrong']}), flagged {tally['flagged']}, "
          f"vacuous killed by decoys {tally['vacuous_killed']}")
    print(f"circular suspects among deliveries: {tally['circular_suspect']} "
          f"(answer literal inside the check)")
    print(f"\nphase 89's agreement gate on these problems: delivered 3, right 1, "
          f"WRONG 2")
    print("The check gate replaces a social signal with a mechanical one, and the")
    print("decoys audit the auditor. What survives all of that is the only thing")
    print("delivered — and the WRONG count above is what that discipline still lets")
    print("through: circularity, agreement in disguise, priced in the open.")
    summary = {"n": n, **tally, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
