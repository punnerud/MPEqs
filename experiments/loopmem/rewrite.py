#!/usr/bin/env python3
"""No keys at all. The model rewrites the expression; the record diffs it and the diff is the step.

The JSON graph worked because the model was asked for structure in a notation it can produce, but
it still had to invent names and keep a symbol table straight. Dropping the names entirely is
simpler and closer to what a person does on paper: write the expression again with one part
replaced by its value. Then

    from-text -> to-text

is the whole step, recovered by diffing consecutive states, and no key is needed because the new
number IS the pointer back to the part it replaced. Single assignment comes free — a state is
never edited, only succeeded — and so does error localisation, which the JSON arm did not have:
when a step is wrong the diff names the exact subexpression that was evaluated wrongly.

That last property is what makes rounds worth running. A rejected step can be re-asked with the
record telling the model precisely what was wrong, and a subexpression that keeps failing can be
broken down further, which is the thing phase 21 identified as the remaining floor.

Four arms on the same twenty expressions and the same seed as phases 20 and 21:

  WHOLE      evaluate in one call                                    (3/20 established)
  FIRST      rewrite until a number, accept whatever the model says
  CHECKED    the record verifies each step and re-asks a wrong one, naming the error
  SPLIT      as CHECKED, and a step that keeps failing is broken down further
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from substitute import WHOLE, gen_expr  # noqa: E402
from twoway import ask_num  # noqa: E402

STEP = """<|endoftext|><|user|>
Rewrite the expression with exactly one part replaced by its value. Change nothing else.
Reply with only the new expression.

Example:
Expression: (7 + 2 * 5) / 3
(7 + 10) / 3

Expression: {expr}
<|assistant|>
"""

RETRY = """<|endoftext|><|user|>
Rewrite the expression with exactly one part replaced by its value. Change nothing else.
Reply with only the new expression.

Example:
Expression: (7 + 2 * 5) / 3
(7 + 10) / 3

Your last attempt said {bad_from} is {bad_to}. That is wrong, so replace a different part
or give the right value.

Expression: {expr}
<|assistant|>
"""

EXPR_CHARS = re.compile(r"[-\d+*/(). ]+")
NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")


LATEX = ((r"\\cdot", "*"), (r"\\times", "*"), (r"\\div", "/"),
         (r"\\left", ""), (r"\\right", ""), (r"\\[\\[\\]()]", ""), (r"\$", ""))


def ask_expr(prompt, n=96):
    from general import ask
    raw = ask(prompt, n=n)
    # Without an example the model answers in prose and LaTeX — "we need to substitute
    # \(39 \cdot 8\)" — and the longest arithmetic span was then a truncated restatement
    # rather than the rewritten expression. The example fixes the format; this normalises what
    # still slips through rather than scoring a right answer as unparseable.
    for pat, rep in LATEX:
        raw = re.sub(pat, rep, raw)
    spans = [s.strip() for s in EXPR_CHARS.findall(raw)]
    spans = [s for s in spans if s and any(c.isdigit() for c in s)
             and s.count("(") == s.count(")")]
    return max(spans, key=len) if spans else None


def diff_span(old, new):
    """The single contiguous change between two states, as (from-text, to-text).

    A step is legal only if it IS a single contiguous change: common prefix, common suffix, one
    replaced middle. Anything else means the model rewrote more than one part, and the record
    cannot attribute a value to a part it cannot isolate.
    """
    i = 0
    while i < min(len(old), len(new)) and old[i] == new[i]:
        i += 1
    j = 0
    while j < min(len(old) - i, len(new) - i) and old[-1 - j] == new[-1 - j]:
        j += 1
    return old[i:len(old) - j].strip(), new[i:len(new) - j].strip()


def classify(frm):
    """What kind of step this was, so the failures can be counted by kind rather than guessed."""
    ops = [c for c in frm if c in "+-*/"]
    op = ops[-1] if ops else "?"
    nums = [abs(float(x)) for x in re.findall(r"\d+(?:\.\d+)?", frm)]
    try:
        val = eval(frm)  # noqa: S307 - substring of our own generated arithmetic
    except Exception:  # noqa: BLE001
        val = 0
    return {"op": op,
            "max_operand": max(nums) if nums else 0,
            "two_digit": any(x >= 10 for x in nums),
            "negative_result": val < 0,
            "result_over_100": abs(val) > 100}


def tokens(e):
    return len(re.findall(r"\d+(?:\.\d+)?|[-+*/()]", e))


def check_step(state, new):
    """Is this rewrite legal? The diff is the objective: smallest possible, and reversible.

    Requiring one contiguous replacement was too rigid and rejected correct work — the model
    writes `(38 + 312) / 2` as `(350 / 2)`, moving a bracket while it substitutes, which is a
    fine step. What actually has to hold is the residue rule stated over a transition:

        reversible   the new state evaluates to the same value as the old one
        progress     the new state is strictly shorter, so the rewrite is a reduction
        minimal      and the diff should be as small as it can be, which is measured

    Reversibility is the guard and it is total: `(38 + 39 * 8) / 2` rewritten as
    `(50 + 11 * 3) / 4` — the model copying the shape of the worked example — is 20.75 against
    175 and is refused on the spot, without any need to reason about what it was trying to do.
    Diff size is not a guard but a score, because a small diff is what makes the transition easy
    to check and easy to attribute when it is wrong.
    """
    if new is None or new == state:
        return None, None, "no rewrite", 0
    frm, to = diff_span(state, new)
    size = len(frm) + len(to)
    try:
        got = eval(new)  # noqa: S307 - our own generated arithmetic
        want = eval(state)  # noqa: S307
    except Exception:  # noqa: BLE001
        return frm, to, "not an expression", size
    if abs(got - want) > 1e-9:
        # Localised for free: this is the part that changed, and changing it broke the value.
        return frm, to, "changes the value", size
    if tokens(new) >= tokens(state):
        return frm, to, "no progress", size
    return frm, to, "ok", size


def solve(expr, retries=0, split=False, max_steps=8):
    """Rewrite until a number. `retries` re-asks a wrong step; `split` breaks a stuck one down."""
    state, trace, calls = expr, [], 0
    for _ in range(max_steps):
        if NUMBER.match(state.strip().strip("()")):
            break
        attempt, bad = 0, None
        while True:
            prompt = (STEP.format(expr=state) if bad is None
                      else RETRY.format(expr=state, bad_from=bad[0], bad_to=bad[1]))
            new = ask_expr(prompt)
            calls += 1
            frm, to, verdict, size = check_step(state, new)
            trace.append({"state": state, "from": frm, "to": to, "verdict": verdict,
                          "attempt": attempt, "diff_size": size,
                          **(classify(frm) if frm else {})})
            if verdict == "ok":
                state = new
                break
            attempt += 1
            if attempt > retries:
                if split and frm and len(re.findall(r"\d+(?:\.\d+)?", frm)) >= 2:
                    # Stuck on this part, so stop asking for a rewrite and ask for the part
                    # itself. This is the smaller-atom move that phase 17 measured: the model is
                    # no longer holding the surrounding expression at all.
                    v = ask_num(f"<|endoftext|><|user|>\nWhat is {frm}?\n\n"
                                f"Reply with only the number.\n<|assistant|>\n")
                    calls += 1
                    if v is not None and abs(eval(frm) - v) < 1e-9:  # noqa: S307
                        trace.append({"state": state, "from": frm, "to": str(v),
                                      "verdict": "ok after split", "attempt": attempt})
                        state = state.replace(frm, str(v), 1)
                        break
                # Give up on verifying and take what it said, so the arm still produces an
                # answer and the comparison is claim-rate against precision rather than blank.
                if new and NUMBER.match(str(to or "")):
                    state = new
                    break
                return None, trace, calls
            bad = (frm, to)
    try:
        return float(eval(state)), trace, calls  # noqa: S307
    except Exception:  # noqa: BLE001
        return None, trace, calls


def main(n_tasks=20, seed=7, out="data/custom/rewrite.json"):
    rng = random.Random(int(seed))
    tasks = [gen_expr(rng) for _ in range(int(n_tasks))]
    arms = [("first", dict(retries=0)), ("checked", dict(retries=2)),
            ("split", dict(retries=2, split=True))]
    print(f"{len(tasks)} expressions. No keys — the record diffs consecutive states.\n")
    print(f"{'expression':<24}{'truth':>8}{'whole':>8}"
          + "".join(f"{a:>9}" for a, _ in arms) + f"{'steps':>7}")
    rows, w_ok = [], 0
    tally = {a: 0 for a, _ in arms}
    all_trace = []
    for expr, truth in tasks:
        w = ask_num(WHOLE.format(expr=expr))
        w_ok += w == truth
        got, traces = {}, {}
        for name, kw in arms:
            v, tr, _ = solve(expr, **kw)
            got[name] = v
            traces[name] = tr
            tally[name] += v is not None and abs(v - truth) < 1e-9
            if name == "checked":
                all_trace.extend(tr)
        rows.append({"expr": expr, "truth": truth, "whole": w,
                     **{f"{a}_value": got[a] for a, _ in arms},
                     "trace": traces["checked"]})
        print(f"{expr:<24}{truth:>8}{str(w):>8}"
              + "".join(f"{str(got[a]):>9}" for a, _ in arms)
              + f"{len(traces['checked']):>7}")

    n = len(tasks)
    print(f"\nwhole expression in one go : {w_ok}/{n}")
    for a, _ in arms:
        print(f"{a:<26}: {tally[a]}/{n}")

    # Where it actually goes wrong, counted rather than guessed.
    bad = [t for t in all_trace if t.get("verdict") == "changes the value" and t.get("op")]
    good = [t for t in all_trace if t.get("verdict") == "ok" and t.get("op")]
    print(f"\nsteps verified {len(good)}, steps rejected {len(bad)}")
    by_op = Counter(t["op"] for t in bad)
    ok_op = Counter(t["op"] for t in good)
    print(f"{'operator':>10}{'wrong':>8}{'right':>8}{'error rate':>12}")
    for op in sorted(set(by_op) | set(ok_op)):
        tot = by_op[op] + ok_op[op]
        print(f"{op:>10}{by_op[op]:>8}{ok_op[op]:>8}{by_op[op] / tot:>12.2f}")
    for label, key in (("both operands two-digit", "two_digit"),
                       ("result negative", "negative_result"),
                       ("result over 100", "result_over_100")):
        b = sum(1 for t in bad if t.get(key))
        g = sum(1 for t in good if t.get(key))
        if b + g:
            print(f"{label:<26} error rate {b / (b + g):.2f}  ({b} of {b + g})")

    # Diff size as the objective, not just as a description. A correct transition replaces one
    # part and touches little; a wrong one is usually the model wandering off into a different
    # expression, which shows up as a large diff before anything is evaluated.
    d_ok = [t["diff_size"] for t in good if t.get("diff_size")]
    d_bad = [t["diff_size"] for t in bad if t.get("diff_size")]
    m_ok = sum(d_ok) / len(d_ok) if d_ok else 0
    m_bad = sum(d_bad) / len(d_bad) if d_bad else 0
    sep = (sum(1 for x in d_bad if x > m_ok) / len(d_bad)) if d_bad else 0
    print(f"\nmean diff size: {m_ok:.1f} chars when the step is right, "
          f"{m_bad:.1f} when it is wrong")
    print(f"{sep:.0%} of wrong steps have a diff bigger than the average right one")

    summary = {"tasks": n, "whole_correct": w_ok,
               "mean_diff_ok": m_ok, "mean_diff_bad": m_bad, "big_diff_when_wrong": sep,
               **{f"{a}_correct": tally[a] for a, _ in arms},
               "steps_verified": len(good), "steps_rejected": len(bad),
               "error_rate_by_op": {op: by_op[op] / (by_op[op] + ok_op[op])
                                    for op in set(by_op) | set(ok_op)}}
    Path(out).write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
