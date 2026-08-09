#!/usr/bin/env python3
"""Two-into-one per step: never let the model hold more than two open subgoals.

The suggestion is a graph claim, not a prompting one — subtasks should join pairwise, and a step
that takes three or more things into one should be split until it does not. It has a clean reason
behind it, already measured twice in this project: the digit result was exactly this, an atom cut
until the model only ever combined things it knew, and the carry work found difficulty rising with
the number of things that must be held at once.

So this measures the arity of a step directly, in two places.

  NUMERIC     a column of long addition is x + y + carry, which is THREE into one. Split it into
              two pairwise sums — (x + y), then (+ carry) — and the model never adds more than two
              digits. Same twelve-to-twenty-four problems, only the arity differs.

  PROCEDURAL  the backward residue can hold several unmet preconditions at once. The MANY arm
              shows the model all of them; the BINARY arm shows at most two, so every step is a
              choice between two open subgoals and the joins happen pairwise up the graph.

Both arms of each pair see identical problems and an identical record. Only the arity changes.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from general import ask as gen_ask, match_action  # noqa: E402
from tasklib import TASKS, depth, widest_join  # noqa: E402
from twoway import BACKWARD, ask_num, digits  # noqa: E402

PAIR = """<|endoftext|><|user|>
Add two single digits: {x} + {y}

Reply with only the total as a number.
<|assistant|>
"""

TRIPLE = """<|endoftext|><|user|>
Add three single digits: {x} + {y} + {c}

Reply with only the total as a number.
<|assistant|>
"""


# ---------------------------------------------------------------- numeric: 3->1 versus 2->1

def add_ternary(a, b):
    """One column as a single three-input step, which is what the digitwise arm did."""
    width = max(len(str(a)), len(str(b)))
    da, db = digits(a, width), digits(b, width)
    carry, out, calls = 0, [], 0
    for k in range(width):
        t = ask_num(TRIPLE.format(x=da[k], y=db[k], c=carry))
        calls += 1
        if t is None:
            return None, calls
        out.append(t % 10)
        carry = t // 10
    if carry:
        out.append(carry)
    return int("".join(map(str, out[::-1]))), calls


def add_binary(a, b):
    """The same column as two pairwise steps. The record holds the partial between them.

    x + y + c becomes (x + y) then (that + c). The model never sees three inputs, and the
    intermediate is a value the record owns rather than something the model must remember.
    """
    width = max(len(str(a)), len(str(b)))
    da, db = digits(a, width), digits(b, width)
    carry, out, calls = 0, [], 0
    for k in range(width):
        s1 = ask_num(PAIR.format(x=da[k], y=db[k]))
        calls += 1
        if s1 is None:
            return None, calls
        if carry:
            # Only ask the second question when it is a real one; adding zero is not a step.
            lo, hi = s1 % 10, s1 // 10
            s2 = ask_num(PAIR.format(x=lo, y=carry))
            calls += 1
            if s2 is None:
                return None, calls
            t = 10 * hi + s2
        else:
            t = s1
        out.append(t % 10)
        carry = t // 10
    if carry:
        out.append(carry)
    return int("".join(map(str, out[::-1]))), calls


def numeric(n_tasks=24, seed=5):
    rng = random.Random(int(seed))
    tasks = []
    while len(tasks) < int(n_tasks):
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        if sum(1 for k in range(3) if (a // 10**k) % 10 + (b // 10**k) % 10 >= 10) >= 2:
            tasks.append((a, b))

    print(f"NUMERIC: {len(tasks)} carry-heavy additions, three-into-one against two-into-one\n")
    print(f"{'problem':>14}{'truth':>8}{'3->1':>8}{'2->1':>8}{'calls 3':>9}{'calls 2':>9}")
    rows, t_ok, b_ok, t_calls, b_calls = [], 0, 0, 0, 0
    for a, b in tasks:
        truth = a + b
        v3, c3 = add_ternary(a, b)
        v2, c2 = add_binary(a, b)
        t_ok += v3 == truth
        b_ok += v2 == truth
        t_calls += c3
        b_calls += c2
        rows.append({"a": a, "b": b, "truth": truth, "ternary": v3, "binary": v2,
                     "ternary_calls": c3, "binary_calls": c2})
        print(f"{f'{a} + {b}':>14}{truth:>8}{str(v3):>8}{str(v2):>8}{c3:>9}{c2:>9}")

    n = len(tasks)
    print(f"\nthree digits per step (x + y + carry) : {t_ok}/{n}   {t_calls} calls")
    print(f"two digits per step  (x + y, + carry) : {b_ok}/{n}   {b_calls} calls")
    return {"tasks": n, "ternary_correct": t_ok, "binary_correct": b_ok,
            "ternary_calls": t_calls, "binary_calls": b_calls, "runs": rows}


# ------------------------------------------------------- procedural: many open goals vs two

def plan_backward(task, spec, arity, max_steps=None):
    """Backward regression. `arity` caps how many open subgoals the model is shown at once."""
    actions, goal = spec["actions"], spec["goal"]
    # Proportional to the graph, not a flat cap. A fixed 24 was smaller than what a ten-action
    # depth-ten chain needs once refusals are counted, so the deep tasks were being scored as
    # planning failures when they had simply run out of steps.
    max_steps = max_steps or 4 * len(actions)
    need, chain, refused = {goal}, [], 0
    tried, window = set(), 0
    for _ in range(max_steps):
        residue = sorted(need)
        if not residue:
            break
        # The binary arm shows at most two open subgoals, so a wide join is worked pairwise
        # instead of being presented whole. On refusal the window slides to the next pair, or
        # the arm would give up on a join whose first two members happen to be exhausted.
        if arity:
            start = (window * arity) % max(len(residue), 1)
            shown = set(residue[start:start + arity]) or set(residue[:arity])
        else:
            shown = set(residue)
        remaining = {k: v for k, v in actions.items()
                     if k not in chain and (k, frozenset(shown)) not in tried}
        if not remaining:
            window += 1
            if arity and window * arity < len(residue) + arity:
                continue
            break
        menu = "\n".join(f"  - {a}" for a in remaining)
        reply = gen_ask(BACKWARD.format(
            task=task, residue="\n".join(f"  - {r}" for r in sorted(shown)), menu=menu))
        a = match_action(reply, remaining)
        if a is None or not (set(actions[a][1]) & shown):
            refused += 1
            if a is not None:
                tried.add((a, frozenset(shown)))
            window += 1
            continue
        pre, eff = actions[a]
        chain.insert(0, a)
        need = (need - set(eff)) | set(pre)
        window = 0

    have, valid = set(), True
    for a in chain:
        pre, eff = actions[a]
        if not all(p in have for p in pre):
            valid = False
            break
        have |= set(eff)
    return (valid and goal in have), chain, refused


def procedural():
    print(f"\nPROCEDURAL: {len(TASKS)} everyday procedures, all open goals against at most two\n")
    print(f"{'task':<26}{'acts':>5}{'depth':>6}{'join':>5}{'many':>6}{'two':>5}"
          f"{'ref many':>9}{'ref two':>8}")
    rows, m_ok, b_ok = [], 0, 0
    for name, spec in TASKS.items():
        ok_m, chain_m, ref_m = plan_backward(name, spec, arity=None)
        ok_b, chain_b, ref_b = plan_backward(name, spec, arity=2)
        m_ok += ok_m
        b_ok += ok_b
        rows.append({"task": name, "actions": len(spec["actions"]), "depth": depth(spec),
                     "widest_join": widest_join(spec), "many_ok": ok_m, "binary_ok": ok_b,
                     "many_chain": chain_m, "binary_chain": chain_b,
                     "many_refused": ref_m, "binary_refused": ref_b})
        print(f"{name:<26}{len(spec['actions']):>5}{depth(spec):>6}{widest_join(spec):>5}"
              f"{'ok' if ok_m else '.':>6}{'ok' if ok_b else '.':>5}{ref_m:>9}{ref_b:>8}")

    n = len(TASKS)
    print(f"\nall open subgoals shown  : {m_ok}/{n}")
    print(f"at most two shown        : {b_ok}/{n}")
    return {"tasks": n, "many_correct": m_ok, "binary_correct": b_ok, "runs": rows}


def main(n_tasks=24, out="data/custom/binary.json"):
    num = numeric(int(n_tasks))
    proc = procedural()
    print("\nIf two-into-one wins in both places the claim is about step arity and not about")
    print("arithmetic. If it wins in one only, the graph and the atom are separate questions.")
    Path(out).write_text(json.dumps({"numeric": num, "procedural": proc}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
