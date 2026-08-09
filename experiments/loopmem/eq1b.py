#!/usr/bin/env python3
"""The 1B protocol: the template thinks, the model reads — one yes/no per step.

The stated target: make the solution so simple that a 1B model can APPLY it without
thinking at any step. The machinery is already split the right way — phase 83's memory
holds the road (clear, gather x, gather constants, normalise), the record executes every
brick exactly, the goal test is mechanical. What remains for the model is one decision
per op, and this phase measures HOW SIMPLE that decision must be, with two arms on the
same twelve equations, same 1B model (OLMoE-1B):

  MENU  the model PLANS: sees the equation and five choices (remove fractions / move x
        over / move the number over / divide / done) and must pick the next step itself,
        round after round. This is small-scale planning — the thing the session's ladder
        says a 1B should not be asked to do.

  READ  the template DRIVES: the stored road supplies the ops in order, and before each
        op the record asks one reading question — is there a fraction in it? is there an
        x on the right? a plain number on the left? is the number in front of x equal to
        1? — and the yes/no decides run-or-skip. Reading, not planning. The record knows
        every true answer (lcm > 1, r != 0, q != 0, p != 1 on the CURRENT state), so the
        protocol's per-question reliability is measured against perfect ground truth.

Wrong answers degrade gracefully by construction — a wrong "yes" meets a self-skipping
op, and a missed CLEAR just makes the road expensive — except where they do not (a
missed MOVEX leaves x on the right), and the mechanical goal test catches exactly those:
any equation the 1B protocol leaves unsolved is escalated up the ladder, same questions,
35B model. Solve rate, per-question accuracy, wasted picks, loops and escalations are
all counted; the model never computes — every brick stays exact in the record's hands.
"""
import json
import random
import re
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402
from eqbricks import solve_direct  # noqa: E402
from eqmemory import op_move, walk  # noqa: E402
from eqbricks import apply_move  # noqa: E402

ROAD = ("CLEAR", "MOVEX", "MOVEC", "DIV")

QUESTIONS = {
    "CLEAR": "Is there any fraction (a number written with /) in the equation?",
    "MOVEX": "Is there an x term on the right side of the = sign?",
    "MOVEC": "Is there a plain number (without x) added or subtracted on the left side?",
    "DIV": "On the left side, is the number in front of x equal to 1?",
}
# DIV's mapping is inverted: "yes, it is already 1" means SKIP the divide.
RUN_ON_YES = {"CLEAR": True, "MOVEX": True, "MOVEC": True, "DIV": False}

MENU = """Equation: {eq}

Choose the next step to solve for x:
A) multiply both sides to remove fractions
B) move the x term from the right side to the left
C) move the plain number from the left side to the right
D) divide both sides by the number in front of x
E) nothing more is needed

Answer with only one letter."""

MENU_OPS = {"A": "CLEAR", "B": "MOVEX", "C": "MOVEC", "D": "DIV", "E": None}

# Free yes/no died at inspection (the 1B answered No to everything, 1/4 right on an
# equation full of fractions); the pilot ran the same questions as a two-letter choice
# and scored 10/12 — format IS capability at this scale. But the pilot's truths were 9
# yes of 12, so an always-A parrot scores 9/12 too: the letter->meaning mapping is
# COUNTERBALANCED per question below, dropping any letter parrot to chance, and the
# parrot's counterfactual score is computed and reported alongside.
READ = """Equation: {eq}

{q}
{opts}

Answer with only one letter."""


def fmt(c):
    return str(c) if c.denominator == 1 else f"{c.numerator}/{c.denominator}"


def render(st):
    p, q, r, s = st

    def side(a, b):
        tx = None if a == 0 else ("x" if a == 1 else f"{fmt(a)}x")
        if tx is None:
            return fmt(b)
        if b == 0:
            return tx
        return f"{tx} + {fmt(b)}" if b > 0 else f"{tx} - {fmt(-b)}"

    return f"{side(p, q)} = {side(r, s)}"


def truth(op, st):
    p, q, r, s = st
    if op == "CLEAR":
        return any(c.denominator > 1 for c in st)
    if op == "MOVEX":
        return r != 0
    if op == "MOVEC":
        return q != 0
    return p == 1                          # DIV question asks "is it already 1?"


def read_arm(st, model, inspect=False, flip_base=0):
    """Template drives; the model answers one two-letter reading question per op.
    Letter->meaning flips per question so a letter parrot scores chance; the parrot's
    counterfactual score is returned for the report."""
    cur = st
    q_total = q_right = invalid = parrot_right = 0
    for k, op in enumerate(ROAD):
        a_means_yes = (flip_base + k) % 2 == 0
        opts = "A) yes\nB) no" if a_means_yes else "A) no\nB) yes"
        prompt = READ.format(eq=render(cur), q=QUESTIONS[op], opts=opts)
        reply = ask(model, prompt, n=8)
        gt = truth(op, cur)
        if inspect:
            print(f"  RAW {op} [{'A=yes' if a_means_yes else 'A=no'}]: {reply!r}  "
                  f"(truth: {'yes' if gt else 'no'})")
        m = re.search(r"\b([AB])\b", reply)
        q_total += 1
        parrot_right += a_means_yes == gt  # what always-A would have scored
        if m is None:
            invalid += 1
            ans = False                    # unreadable counts wrong and skips the op
        else:
            ans = (m.group(1) == "A") == a_means_yes
        q_right += ans == gt
        if ans == RUN_ON_YES[op]:
            mv = op_move(op, cur)
            if mv:
                cur = apply_move(cur, mv)
    solved = cur == (F(1), F(0), F(0), solve_direct(st))
    return solved, q_right, q_total, invalid, parrot_right


def menu_arm(st, model, max_rounds=6, inspect=False):
    """The model plans: pick the next op itself, round after round."""
    cur = st
    goal = (F(1), F(0), F(0), solve_direct(st))
    seen = {cur}
    rounds = wasted = 0
    looped = False
    for _ in range(max_rounds):
        reply = ask(model, MENU.format(eq=render(cur)), n=8)
        if inspect:
            print(f"  RAW menu: {reply!r}  state {render(cur)}")
        m = re.search(r"\b([A-E])\b", reply)
        rounds += 1
        if not m or MENU_OPS[m.group(1)] is None:
            break                          # "done" (or unreadable) ends the attempt
        mv = op_move(MENU_OPS[m.group(1)], cur)
        if mv is None:
            wasted += 1
            continue
        cur = apply_move(cur, mv)
        if cur in seen:
            looped = True
            break
        seen.add(cur)
        if cur == goal:
            break
    return cur == goal, rounds, wasted, looped


def battery(rng):
    eqs = []
    for kind in ["frac"] * 6 + ["int"] * 3 + ["r0"] * 2 + ["q0"]:
        while True:
            dd = lambda: rng.randint(2, 6)  # noqa: E731
            if kind == "frac":
                st = (F(rng.randint(2, 9), dd()), F(rng.randint(1, 20), dd()),
                      F(rng.randint(1, 6), dd()), F(rng.randint(1, 20), dd()))
            elif kind == "int":
                r = F(rng.randint(1, 5))
                st = (r + rng.randint(1, 3), F(rng.randint(1, 9)), r,
                      F(rng.randint(1, 9)))
            elif kind == "r0":
                st = (F(rng.randint(2, 9), dd()), F(rng.randint(1, 20), dd()),
                      F(0), F(rng.randint(1, 20), dd()))
            else:
                st = (F(rng.randint(2, 9), dd()), F(0),
                      F(rng.randint(1, 6), dd()), F(rng.randint(1, 20), dd()))
            if st[0] != st[2] and st[0] != 0:
                eqs.append(st)
                break
    return eqs


def main(n_inspect=0, seed=19, out="data/custom/eq1b.json"):
    n_inspect, seed = int(n_inspect), int(seed)
    rng = random.Random(seed)
    eqs = battery(rng)
    if n_inspect:
        st = eqs[0]
        print(f"equation: {render(st)}")
        read_arm(st, "olmoe-1b", inspect=True)
        menu_arm(st, "olmoe-1b", max_rounds=2, inspect=True)
        return

    read_solved = q_right = q_total = invalid = parrot = 0
    esc_tried = esc_solved = 0
    menu_solved = menu_rounds = menu_wasted = menu_loops = 0
    rows = []
    for i, st in enumerate(eqs):
        okR, qr, qt, inv, pr = read_arm(st, "olmoe-1b", flip_base=i)
        read_solved += okR
        q_right += qr
        q_total += qt
        invalid += inv
        parrot += pr
        esc = None
        if not okR:                        # the ladder: same questions, bigger model
            esc_tried += 1
            esc, _, _, _, _ = read_arm(st, "qwen-35b", flip_base=i)
            esc_solved += esc
        okM, rounds, wasted, looped = menu_arm(st, "olmoe-1b")
        menu_solved += okM
        menu_rounds += rounds
        menu_wasted += wasted
        menu_loops += looped
        rows.append({"eq": render(st), "read": okR, "esc": esc, "menu": okM,
                     "q_right": qr, "rounds": rounds, "wasted": wasted})
        print(f"{render(st):<34} READ {'ok' if okR else 'X' + ('->35B ok' if esc else ' ->35B X' if esc is not None else '')}"
              f"  ({qr}/{qt} reads)   MENU {'ok' if okM else 'X'} "
              f"({rounds} rounds, {wasted} wasted{', loop' if looped else ''})")

    n = len(eqs)
    print(f"\nREAD arm (template drives, 1B answers a two-letter read):")
    print(f"  solved {read_solved}/{n}; reading accuracy {q_right}/{q_total} "
          f"({invalid} unreadable); a letter parrot would score {parrot}/{q_total}")
    print(f"  escalated {esc_tried}, 35B solved {esc_solved} of them")
    print(f"MENU arm (1B plans the next step itself):")
    print(f"  solved {menu_solved}/{n}; {menu_rounds} rounds, {menu_wasted} wasted "
          f"picks, {menu_loops} loops")
    print("\nThe model never computes and never chooses a road — the template thinks,")
    print("the record executes exact bricks, and the 1B is asked only to read. What the")
    print("goal test catches goes up the ladder unchanged. The gap between the arms is")
    print("the measurement: how much solve survives when planning is replaced by")
    print("reading, at the model size where planning is not on offer.")
    summary = {"equations": n, "read_solved": read_solved, "q_right": q_right,
               "q_total": q_total, "invalid": invalid, "parrot": parrot,
               "escalated": esc_tried, "esc_solved": esc_solved,
               "menu_solved": menu_solved, "menu_rounds": menu_rounds,
               "menu_wasted": menu_wasted, "menu_loops": menu_loops, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
