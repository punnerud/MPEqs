#!/usr/bin/env python3
"""The hard skin: flipped clause order and a numeric distractor in every story.

The clean battery swept 12/12 everywhere, and the session's rule is that a perfect
score must be attacked, not admired. Same twelve truth tuples, two hardenings:

  FLIP   the prose states the right-hand side first. A faithful reader may write the
         equation in either orientation — equality is symmetric — so arm accuracy
         accepts truth or flipped truth, and the agreement gate accepts a == b or
         a == flip(b). Declared, not fudged: both tuples encode the same equation.

  NOISE  every story opens with a sentence carrying an irrelevant number chosen to
         collide with plausible coefficients (cats, yesterday's tickets, the hour).
         The distractor test: does a number that does not belong leak into the state?

Same gates, same meta-judge, same blind road after the gate."""
import json
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eqbricks import solve_direct  # noqa: E402
from eqmemory import walk  # noqa: E402
from eqwords import BATTERY, NOUNS, XPH, translate  # noqa: E402

ROAD = ("CLEAR", "MOVEX", "MOVEC", "DIV")
NOISE = {"number": "Kari wrote 6 pages in her diary today.",
         "age": "Ada has 3 cats.",
         "price": "The shop sold 14 tickets yesterday."}


def flip(st):
    return (st[2], st[3], st[0], st[1])


def hard_story(st, skin):
    p, q, r, s = st
    n = NOUNS[skin]

    def side(a, b):
        if a == 0:
            return str(b)
        xp = XPH[a].format(n=n)
        if b == 0:
            return xp
        return f"{xp} plus {b}" if b > 0 else f"{xp} minus {-b}"

    # Right side spoken first: "<R> is the same as <L>."
    left, right = side(p, q), side(r, s)
    return (f"{NOISE[skin]} {right[0].upper() + right[1:]} is the same as {left}. "
            f"What is {n}?")


def main(out="data/custom/eqwords_hard.json"):
    a_right = b_right = agree = agree_true = agree_wrong = 0
    retries = retry_fixed = flagged = e2e = 0
    rows = []
    for st, skin in BATTERY:
        text = hard_story(st, skin)
        ok_states = (st, flip(st))
        a, b = translate(text)
        a_right += a in ok_states
        b_right += b in ok_states
        agreed = a is not None and (a == b or (b is not None and a == flip(b)))
        if not agreed:
            retries += 1
            def show(v):
                return "unreadable" if v is None else f"{v[0]}x + {v[1]} = {v[2]}x + {v[3]}"
            hint = (f"\n\nTwo readings of this problem disagreed: {show(a)} versus "
                    f"{show(b)}. Read the problem again carefully; ignore numbers that "
                    f"have nothing to do with the equation.")
            a, b = translate(text, hint)
            agreed = a is not None and (a == b or (b is not None and a == flip(b)))
            retry_fixed += agreed
        status = "flagged"
        if agreed:
            agree += 1
            agree_true += a in ok_states
            agree_wrong += a not in ok_states
            fin, _ = walk(a, ROAD)
            solved = fin == (F(1), F(0), F(0), solve_direct(a))
            truth_x = solve_direct(st)
            e2e += solved and fin[3] == truth_x
            status = f"x = {fin[3]}" + ("" if fin[3] == truth_x else " (WRONG)")
        else:
            flagged += 1
        rows.append({"story": text, "truth": [str(c) for c in st],
                     "a": a and [str(c) for c in a], "b": b and [str(c) for c in b],
                     "status": status})
        print(f"{text[:64]:<66} {status}")

    n = len(BATTERY)
    print(f"\nhard skin (flipped clause order + numeric distractor):")
    print(f"  JSON arm {a_right}/{n}, equation arm {b_right}/{n} (orientation-agnostic)")
    print(f"agreement gate: {agree}/{n} delivered ({retries} retries, {retry_fixed} "
          f"fixed, {flagged} flagged)")
    print(f"  agreed on truth {agree_true}/{agree}; agreed WRONG {agree_wrong}/{agree}")
    print(f"end-to-end exact: {e2e}/{agree}")
    summary = {"problems": n, "a_right": a_right, "b_right": b_right, "agree": agree,
               "agree_true": agree_true, "agree_wrong": agree_wrong,
               "retries": retries, "retry_fixed": retry_fixed, "flagged": flagged,
               "e2e": e2e, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
