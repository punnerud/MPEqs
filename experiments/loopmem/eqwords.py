#!/usr/bin/env python3
"""Word problems: the model translates language into state; everything after is blind.

Phase 84 ended with the division drawn sharp: for machine-readable states the record
needs no model at all, and the model earns its place only where the state is written in
language. This phase is that place. Twelve word problems are GENERATED from twelve
known states — the truth tuples written first, the stories rendered mechanically from
them in three surface skins (a number, an age, a price) — and the model's single job is
translation: read the story, name the state.

The gate is the session's agreement law, measured again on a new domain: the SAME story
is translated through TWO different entry points — once as coefficient JSON, once as a
written equation that the record parses — and the answer is delivered only when both
land the identical exact state. Disagreement gets one retry with both readings named;
still split means flagged, not guessed. The risk the gate carries is a number, not a
hope: how often do the two entry points agree on the WRONG state? (Phases 74 and 55
measured zero; this phase re-measures it where language can fool both readers the same
way.)

After the gate, no model: the universal road of phase 83 walks the agreed state to
solved form with exact bricks, and the goal test gates delivery. The meta-judge (the
pre-written truth) scores every stage separately: JSON arm right, equation arm right,
agreement rate, agreement-on-truth rate, end-to-end exact.
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
from eqmemory import walk  # noqa: E402

ROAD = ("CLEAR", "MOVEX", "MOVEC", "DIV")

# The truths, before any model call. p, r from the phrase table; q, s plain integers.
BATTERY = [
    ((F(3), F(4), F(2), F(9)), "number"),
    ((F(1, 2), F(7), F(2), F(1)), "age"),
    ((F(5), F(2), F(3), F(14)), "price"),
    ((F(1, 3), F(8), F(1), F(2)), "number"),
    ((F(4), F(-3), F(2), F(11)), "age"),
    ((F(2), F(5), F(1, 2), F(17)), "price"),
    ((F(3), F(0), F(1), F(12)), "number"),
    ((F(1, 2), F(9), F(1, 3), F(10)), "age"),
    ((F(4), F(6), F(0), F(18)), "price"),
    ((F(2), F(13), F(0), F(19)), "number"),
    ((F(5), F(-2), F(4), F(3)), "age"),
    ((F(3), F(10), F(2), F(16)), "price"),
]

NOUNS = {"number": "the number", "age": "Ada's age in years", "price": "the price"}
LEADS = {"number": "Think of a number.", "age": "Ada is some years old.",
         "price": "A ticket costs some amount."}
XPH = {F(1, 2): "half of {n}", F(1, 3): "a third of {n}", F(1): "{n}",
       F(2): "twice {n}", F(3): "three times {n}", F(4): "four times {n}",
       F(5): "five times {n}"}


def story(st, skin):
    p, q, r, s = st
    n = NOUNS[skin]

    def side(a, b):
        if a == 0:
            return str(b)
        xp = XPH[a].format(n=n)
        if b == 0:
            return xp
        return f"{xp} plus {b}" if b > 0 else f"{xp} minus {-b}"

    left, right = side(p, q), side(r, s)
    return (f"{LEADS[skin]} {left[0].upper() + left[1:]} equals {right}. "
            f"What is {n}?")


PROMPT_JSON = """Read this problem and write the equation it describes.

Problem: {story}

The equation has the form p*x + q = r*x + s. Reply with ONLY this JSON:
{{"p": "<number>", "q": "<number>", "r": "<number>", "s": "<number>"}}
Write fractions like "1/2". If a side has no x, its coefficient is "0". If a side has
no plain number, that entry is "0"."""

PROMPT_EQ = """Read this problem and write the equation it describes.

Problem: {story}

Reply with ONLY the equation on one line, in exactly this shape: <left> = <right>
where a side looks like 7x + 11 or 2/7x - 8 or 15 on its own. Use x as the unknown.
No other text."""

# The shape examples above use 7, 11, 2/7, 8, 15 — none appear as battery coefficients;
# translations equal to an example are counted as echoes and excluded, per the standing
# example-echo rule.
ECHOES = [(F(7), F(11), F(0), F(15)), (F(2, 7), F(-8), F(0), F(15))]


def parse_json_state(reply):
    m = re.search(r"\{[^{}]*\}", reply, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return tuple(F(str(d[k]).strip()) for k in ("p", "q", "r", "s"))
    except Exception:  # noqa: BLE001
        return None


def parse_side(t):
    t = t.replace(" ", "")
    if "x" in t:
        m = re.fullmatch(r"([0-9/]+)?\*?x([+-][0-9/]+)?", t)
        if not m:
            return None
        coef = F(m.group(1)) if m.group(1) else F(1)
        const = F(m.group(2)) if m.group(2) else F(0)
        return coef, const
    try:
        return F(0), F(t)
    except ValueError:
        return None


def parse_eq_state(reply):
    for line in reply.strip().splitlines():
        if "=" in line:
            halves = line.split("=")
            if len(halves) != 2:
                return None
            L, R = parse_side(halves[0]), parse_side(halves[1])
            if L and R:
                return (L[0], L[1], R[0], R[1])
            return None
    return None


def translate(text, hint=""):
    a = parse_json_state(ask("qwen-35b", PROMPT_JSON.format(story=text) + hint, n=90))
    b = parse_eq_state(ask("qwen-35b", PROMPT_EQ.format(story=text) + hint, n=60))
    return a, b


def main(n_inspect=0, out="data/custom/eqwords.json"):
    n_inspect = int(n_inspect)
    if n_inspect:
        for st, skin in (BATTERY[0], BATTERY[5]):
            text = story(st, skin)
            print(f"story: {text}")
            ra = ask("qwen-35b", PROMPT_JSON.format(story=text), n=90)
            rb = ask("qwen-35b", PROMPT_EQ.format(story=text), n=60)
            print(f"  RAW json: {ra!r}\n  RAW eq:   {rb!r}\n  truth: {st}\n")
        return

    a_right = b_right = agree = agree_true = agree_wrong = 0
    retries = retry_fixed = flagged = echoes = e2e = 0
    rows = []
    for st, skin in BATTERY:
        text = story(st, skin)
        a, b = translate(text)
        a_right += a == st
        b_right += b == st
        if (a and a in ECHOES) or (b and b in ECHOES):
            echoes += 1
        if a != b or a is None:
            retries += 1
            def show(v):
                return "unreadable" if v is None else f"{v[0]}x + {v[1]} = {v[2]}x + {v[3]}"
            hint = (f"\n\nTwo readings of this problem disagreed: {show(a)} versus "
                    f"{show(b)}. Read the problem again carefully.")
            a, b = translate(text, hint)
            retry_fixed += a == b and a is not None
        status = "flagged"
        if a == b and a is not None:
            agree += 1
            agree_true += a == st
            agree_wrong += a != st
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
    print(f"\ntranslation, meta-judged against pre-written truths:")
    print(f"  JSON arm {a_right}/{n}, equation arm {b_right}/{n} (first attempt)")
    print(f"  example echoes excluded: {echoes}")
    print(f"agreement gate: {agree}/{n} delivered ({retries} retries, "
          f"{retry_fixed} fixed by naming the split, {flagged} flagged unsolved)")
    print(f"  agreed on the TRUTH {agree_true}/{agree}; agreed on a WRONG state "
          f"{agree_wrong}/{agree} — the risk the gate carries")
    print(f"end-to-end exact through the blind road: {e2e}/{agree}")
    print("\nLanguage is the only step the record cannot do, so it is the only step the")
    print("model does — and two entry points that land the same exact state replace the")
    print("grader, on a domain where both could in principle be fooled the same way.")
    print("Whether they ever are is now a measured number, not an assumption.")
    summary = {"problems": n, "a_right": a_right, "b_right": b_right,
               "echoes": echoes, "agree": agree, "agree_true": agree_true,
               "agree_wrong": agree_wrong, "retries": retries,
               "retry_fixed": retry_fixed, "flagged": flagged, "e2e": e2e,
               "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
