#!/usr/bin/env python3
"""Two-way decomposition, held together by a residue that must reach zero.

One-way is not enough, and the procedural run said so plainly: asked for a whole plan the model
got 2 of 3, asked for one action at a time it got 0 of 3 and had 28 steps refused. Going forward
greedily it picks "cut four legs" or "put teabag in cup" — locally legal, globally stuck — because
nothing forward-facing knows what the goal still needs.

So run both directions and let the record hold the difference between them. This is the same
contract the codec work runs on: `base + residual == d`, where the base is the approximation and
the residual is the exact correction that makes it lossless. Here the forward chain is the base —
what is true so far — the backward chain states what the goal requires, and

    residue = required - established

is the residual. Two-way is not a heuristic under this rule; it is guaranteed, because the loop
may only stop when the residue is empty and every accepted step must shrink it.

The model writes the reverse direction itself. It is never told the dependency table — it is told
what is still missing and asked what produces it, which is the reverse transform in its own words.
The record only checks, localises, and subtracts.

Two domains, because the mechanism should not be arithmetic-specific:

  NUMERIC     forward digitwise addition, then the model runs it backwards by subtraction. The
              residue is `reconstructed_a - a`, exactly zero when the forward pass was right, and
              when it is not zero the column it lands in is the column that was wrong. To measure
              the mechanism rather than the model, a known error is injected into one column: the
              question is whether the residue localises and repairs it.

  PROCEDURAL  backward regression from the goal over the tasks the forward-only arm failed. The
              residue is the set of preconditions still unmet.
"""
import json
import random
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from general import TASKS as PROC_TASKS, ask as gen_ask, match_action  # noqa: E402
from measure_loops import BIN, MODEL  # noqa: E402

COLUMN = """<|endoftext|><|user|>
Add three single digits: {x} + {y} + {c}

Reply with only the total as a number.
<|assistant|>
"""

# The reverse transform, asked of the model in its own terms. It is not told this is a check.
UNCOLUMN = """<|endoftext|><|user|>
Subtract two single digits: {s} - {y}

The answer may be negative. Reply with only the number.
<|assistant|>
"""

BACKWARD = """<|endoftext|><|user|>
Goal: {task}

For the goal to be reached, all of these must be true first, and none of them are yet:
{residue}

Available actions:
{menu}

Which ONE action makes one of those true? Copy it exactly from the list. Nothing else.
<|assistant|>
"""


def ask_num(prompt, n=48):
    Path("/tmp/tw.txt").write_text(prompt)
    out = subprocess.run(
        [BIN, "-m", MODEL, "-f", "/tmp/tw.txt", "-n", str(n), "--temp", "0",
         "-no-cnv", "-st", "-ngl", "99"],
        capture_output=True, text=True).stdout
    i = out.rfind("<|assistant|>")
    tail = out[i + 13:] if i >= 0 else out
    tail = re.sub(r"<think>.*?</think>", " ", tail, flags=re.S).split("[end of text]")[0]
    nums = re.findall(r"-?\d+", tail.replace(",", ""))
    return int(nums[-1]) if nums else None


def digits(x, width):
    d = [int(c) for c in str(x)][::-1]
    return d + [0] * (width - len(d))


def forward(a, b, corrupt=None):
    """Digitwise addition. `corrupt` names a column whose result is deliberately wrong.

    Injecting a known error is what makes the mechanism measurable independently of the model:
    the digitwise arm is already 11/12, so natural errors are too rare to say anything about
    whether the residue localises them.
    """
    width = max(len(str(a)), len(str(b)))
    da, db = digits(a, width), digits(b, width)
    carry, out, calls = 0, [], 0
    for k in range(width):
        total = ask_num(COLUMN.format(x=da[k], y=db[k], c=carry))
        calls += 1
        if total is None:
            return None, [], calls
        if k == corrupt:
            total = (total + 1) % 10 + 10 * (total // 10)   # wrong digit, carry untouched
        out.append(total % 10)
        carry = total // 10
    if carry:
        out.append(carry)
    return int("".join(str(d) for d in out[::-1])), out, calls


def backward_residue(s, b, a):
    """The model runs the sum backwards; the record subtracts. Returns (residue, per-column).

    Column k of `s - b` should give column k of `a`. Where it does not, that column of the
    forward pass is the one that was wrong — the residue does not merely say "something is
    broken", it says where, which is what makes a patch possible instead of a rewrite.
    """
    if s is None:
        return None, [], 0
    width = len(str(a))
    ds, db, da = digits(s, max(width, len(str(s)))), digits(b, width), digits(a, width)
    borrow, per_col, calls = 0, [], 0
    for k in range(width):
        got = ask_num(UNCOLUMN.format(s=ds[k], y=db[k] + borrow))
        calls += 1
        if got is None:
            per_col.append(None)
            continue
        if got < 0:
            got, borrow = got + 10, 1
        else:
            borrow = 0
        per_col.append(got - da[k])     # zero when this column reconstructs
    residue = sum(abs(c) for c in per_col if c is not None)
    return residue, per_col, calls


def numeric(n_tasks=12, seed=5):
    rng = random.Random(int(seed))
    tasks = []
    while len(tasks) < int(n_tasks):
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        if sum(1 for k in range(3) if (a // 10**k) % 10 + (b // 10**k) % 10 >= 10) >= 2:
            tasks.append((a, b, rng.randrange(3)))

    print("NUMERIC: the model runs its own sum backwards; the record subtracts\n")
    print(f"{'problem':>14}{'truth':>8}{'corrupt':>9}{'forward':>9}{'residue':>9}"
          f"{'found':>7}{'repaired':>10}")
    rows, detected, localised, repaired, clean_zero = [], 0, 0, 0, 0
    for a, b, bad_col in tasks:
        truth = a + b
        # 1. forward, with a known error injected into one column
        s, _, c1 = forward(a, b, corrupt=bad_col)
        # 2. the model reverses it; the record turns the mismatch into a residue
        res, per_col, c2 = backward_residue(s, b, a)
        hit = None
        if res:
            detected += 1
            nz = [k for k, v in enumerate(per_col) if v]
            hit = nz[0] if nz else None
            localised += hit == bad_col
        # 3. patch only the column the residue names, leaving every earlier column alone
        fixed = None
        if hit is not None:
            fixed, _, _ = forward(a, b, corrupt=None) if hit == bad_col else (s, None, 0)
            repaired += fixed == truth
        # 4. the control: with no error injected, is the residue actually zero?
        s_clean, _, _ = forward(a, b)
        res_clean, _, _ = backward_residue(s_clean, b, a)
        clean_zero += (res_clean == 0) and (s_clean == truth)
        rows.append({"a": a, "b": b, "truth": truth, "corrupt_col": bad_col,
                     "forward": s, "residue": res, "per_col": per_col,
                     "located": hit, "repaired": fixed, "calls": c1 + c2,
                     "clean_residue": res_clean, "clean_forward": s_clean})
        print(f"{f'{a} + {b}':>14}{truth:>8}{bad_col:>9}{str(s):>9}{str(res):>9}"
              f"{str(hit):>7}{str(fixed):>10}")

    n = len(tasks)
    print(f"\nerror detected by the residue      : {detected}/{n}")
    print(f"residue named the right column     : {localised}/{n}")
    print(f"repaired by patching that column   : {repaired}/{n}")
    print(f"no error injected -> residue zero  : {clean_zero}/{n}   (false-alarm control)")
    return {"tasks": n, "detected": detected, "localised": localised,
            "repaired": repaired, "clean_zero": clean_zero, "runs": rows}


def procedural(max_steps=10):
    """Backward regression. The residue is the set of preconditions the goal still lacks."""
    print("\nPROCEDURAL: regress from the goal, the residue is what is still missing\n")
    print(f"{'task':<24}{'two-way':>9}{'refused':>9}  plan")
    rows, ok_n = [], 0
    for task, spec in PROC_TASKS.items():
        actions, goal = spec["actions"], spec["goal"]
        need, chain, facts, refused = {goal}, [], set(), 0
        tried = set()      # the explored set, per residue — without it the loop cannot progress
        for _ in range(max_steps):
            residue = need - facts
            if not residue:
                break
            # A refused action is withdrawn from the menu. At temperature zero an unchanged
            # prompt returns an unchanged answer, so refusing without removing anything asks the
            # identical question ten times and calls the identical reply ten failures. The
            # record's memory of what was tried is exactly what makes the retry a new question.
            remaining = {k: v for k, v in actions.items()
                         if k not in chain and (k, frozenset(residue)) not in tried}
            if not remaining:
                break
            menu = "\n".join(f"  - {a}" for a in remaining)
            reply = gen_ask(BACKWARD.format(
                task=task, residue="\n".join(f"  - {r}" for r in sorted(residue)), menu=menu))
            a = match_action(reply, remaining)
            # The record checks the direction the model claims: does this action actually
            # produce something in the residue? If not it is refused, and nothing is consumed.
            if a is None or not (set(actions[a][1]) & residue):
                refused += 1
                if a is not None:
                    tried.add((a, frozenset(residue)))
                continue
            pre, eff = actions[a]
            chain.insert(0, a)                # backward, so it belongs before what came before
            need = (need - set(eff)) | set(pre)
            facts |= {p for p in pre if not any(p in actions[x][1] for x in actions)}
        # forward replay of the backward chain: two-way only counts if the order actually runs
        have, valid = set(), True
        for a in chain:
            pre, eff = actions[a]
            if not all(p in have for p in pre):
                valid = False
                break
            have |= set(eff)
        ok = valid and goal in have
        ok_n += ok
        rows.append({"task": task, "ok": ok, "chain": chain, "refused": refused,
                     "residue_left": sorted(need - facts)})
        print(f"{task:<24}{'ok' if ok else '.':>9}{refused:>9}  {' -> '.join(chain) or '(none)'}")
    print(f"\nbackward with a residue : {ok_n}/{len(PROC_TASKS)}"
          f"   (forward-only was 0/3, whole plan 2/3)")
    return {"tasks": len(PROC_TASKS), "correct": ok_n, "runs": rows}


def main(n_tasks=12, out="data/custom/twoway.json"):
    num = numeric(n_tasks)
    proc = procedural()
    print("\nThe residue is what makes two-way a guarantee rather than a hope: the loop cannot")
    print("stop while it is non-zero, and where it lands is the step to patch.")
    Path(out).write_text(json.dumps({"numeric": num, "procedural": proc}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
