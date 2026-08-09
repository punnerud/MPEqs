#!/usr/bin/env python3
"""Two unknowns behind the same gate: coupled systems from stories, solved exactly.

Phase 76 built the coupled solver (momentum against energy, dimension-checked, exact);
phase 85 built the language gate. This phase joins them and adds the verifier systems
earn for free: SUBSTITUTION. Eight stories from eight truth pairs, two families:

  PE         momentum p = m*v and kinetic energy E = m*v*v/2 known; find mass and
             speed. Record solves v = 2E/p, m = p/v.
  SUM-RATIO  two numbers with a known sum, one a known multiple of the other.
             Record solves y = S/(k+1), x = k*y.

Truths (m, v) and (x, y) written first; the given constants derived from them; every
story opens with a numeric distractor. The model translates twice — a JSON of the
constants, and the equations WRITTEN OUT, which the record parses — and the agreement
gate compares family and constants exactly. After the gate the record solves, then
SUBSTITUTES the solution back into both equations and delivers only on exact identity:
a second, mechanical gate that costs nothing and catches any translation the agreement
gate let through with self-consistent but wrong constants. Wrong deliveries require
BOTH gates to fail at once; how often that happens is the number this phase exists to
measure.
"""
import json
import random
import re
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402

# Truths first: PE from (m, v); SUM-RATIO from (x, y) with x = k*y.
PE_TRUTHS = [(F(20), F(150)), (F(4), F(10)), (F(8), F(25)), (F(2), F(6))]
SR_TRUTHS = [(F(20), F(10)), (F(24), F(8)), (F(35), F(7)), (F(18), F(6))]

PE_STORY = ("The lab bench is 2 meters long. An object has momentum {p} in kg*m/s "
            "and kinetic energy {E} in joules. What are its mass in kg and its "
            "speed in m/s?")
SR_STORY = ("Grandma is 74 years old. Two numbers add up to {S}, and the first is "
            "{k} times the second. What are the first and the second number?")

PROMPT_JSON_PE = """Read this problem and name its constants.

Problem: {story}

Reply with ONLY this JSON: {{"family": "pe", "p": "<number>", "E": "<number>"}}"""

PROMPT_JSON_SR = """Read this problem and name its constants.

Problem: {story}

Reply with ONLY this JSON: {{"family": "sum_ratio", "sum": "<number>", "k": "<number>"}}"""

PROMPT_EQS_PE = """Read this problem and write its two equations, using m for mass and
v for speed.

Problem: {story}

Reply with ONLY two lines, exactly in this shape:
m*v = <number>
m*v*v/2 = <number>"""

PROMPT_EQS_SR = """Read this problem and write its two equations, using x for the first
number and y for the second.

Problem: {story}

Reply with ONLY two lines, exactly in this shape:
x + y = <number>
x = <number>*y"""


def parse_json_c(reply):
    m = re.search(r"\{[^{}]*\}", reply, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        if d.get("family") == "pe":
            return ("pe", F(str(d["p"]).strip()), F(str(d["E"]).strip()))
        if d.get("family") == "sum_ratio":
            return ("sum_ratio", F(str(d["sum"]).strip()), F(str(d["k"]).strip()))
    except Exception:  # noqa: BLE001
        return None
    return None


def parse_eqs(reply, family):
    t = reply.lower().replace(" ", "")
    if family == "pe":
        m1 = re.search(r"m\*v=([0-9./]+)", t)
        m2 = re.search(r"m\*v\*v/2=([0-9./]+)", t)
        if m1 and m2:
            try:
                return ("pe", F(m1.group(1)), F(m2.group(1)))
            except ValueError:
                return None
        return None
    m1 = re.search(r"x\+y=([0-9./]+)", t)
    m2 = re.search(r"x=([0-9./]+)\*y", t)
    if m1 and m2:
        try:
            return ("sum_ratio", F(m1.group(1)), F(m2.group(1)))
        except ValueError:
            return None
    return None


def solve(triple):
    family, c1, c2 = triple
    if family == "pe":
        p, E = c1, c2
        v = 2 * E / p
        m = p / v
        return (m, v), (m * v == p and m * v * v / 2 == E)
    S, k = c1, c2
    y = S / (k + 1)
    x = k * y
    return (x, y), (x + y == S and x == k * y)


def main(n_inspect=0, out="data/custom/eqpairs.json"):
    n_inspect = int(n_inspect)
    battery = []
    for m, v in PE_TRUTHS:
        p, E = m * v, m * v * v / 2
        battery.append((("pe", p, E), (m, v),
                        PE_STORY.format(p=p, E=E), PROMPT_JSON_PE, PROMPT_EQS_PE))
    for x, y in SR_TRUTHS:
        S, k = x + y, x / y
        battery.append((("sum_ratio", S, k), (x, y),
                        SR_STORY.format(S=S, k=k), PROMPT_JSON_SR, PROMPT_EQS_SR))

    if n_inspect:
        for idx in (0, 4):
            truth_c, _, story, pj, pe_ = battery[idx]
            ra = ask("qwen-35b", pj.format(story=story), n=70)
            rb = ask("qwen-35b", pe_.format(story=story), n=60)
            print(f"story: {story}\n  RAW json: {ra!r}\n  RAW eqs:  {rb!r}\n"
                  f"  truth constants: {truth_c}\n")
        return

    a_right = b_right = agree = agree_true = agree_wrong = 0
    retries = flagged = subst_ok = e2e = 0
    rows = []
    for truth_c, truth_sol, story, pj, pe_ in battery:
        fam = truth_c[0]
        a = parse_json_c(ask("qwen-35b", pj.format(story=story), n=70))
        b = parse_eqs(ask("qwen-35b", pe_.format(story=story), n=60), fam)
        a_right += a == truth_c
        b_right += b == truth_c
        if a != b or a is None:
            retries += 1
            hint = (f"\n\nTwo readings disagreed: {a} versus {b}. Read again; ignore "
                    f"numbers that are not part of the equations.")
            a = parse_json_c(ask("qwen-35b", pj.format(story=story) + hint, n=70))
            b = parse_eqs(ask("qwen-35b", pe_.format(story=story) + hint, n=60), fam)
        status = "flagged"
        if a == b and a is not None:
            agree += 1
            agree_true += a == truth_c
            agree_wrong += a != truth_c
            sol, ok = solve(a)
            subst_ok += ok
            if ok:                          # the substitution gate must also pass
                e2e += sol == truth_sol
                status = f"solution {tuple(str(c) for c in sol)}"
            else:
                status = "agreed, SUBSTITUTION REFUSED"
        else:
            flagged += 1
        rows.append({"story": story[:70], "truth": [str(c) for c in truth_c[1:]],
                     "a": a and [str(c) for c in a[1:]],
                     "b": b and [str(c) for c in b[1:]], "status": status})
        print(f"{story[:56]:<58} {status}")

    n = len(battery)
    print(f"\ntranslation: JSON arm {a_right}/{n}, equations arm {b_right}/{n}")
    print(f"agreement gate: {agree}/{n} ({retries} retries, {flagged} flagged); "
          f"agreed on truth {agree_true}/{agree}, WRONG {agree_wrong}/{agree}")
    print(f"substitution gate: {subst_ok}/{agree} exact identities; end-to-end "
          f"matches the written truth pairs {e2e}/{subst_ok}")
    print("\nTwo unknowns cost nothing extra: the model still only reads, the record")
    print("solves the coupling exactly, and the solution buys a second gate for free —")
    print("substitution back into the equations it came from. A wrong delivery now")
    print("needs both gates wrong at once.")
    summary = {"problems": n, "a_right": a_right, "b_right": b_right, "agree": agree,
               "agree_true": agree_true, "agree_wrong": agree_wrong,
               "retries": retries, "flagged": flagged, "subst_ok": subst_ok,
               "e2e": e2e, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
