#!/usr/bin/env python3
"""The model picks the first step; the residue price judges the pick.

Phase 80 priced the roads: multiply both sides first is a factor five in carried residue
when there are denominators to clear. But "multiply first" is also a textbook slogan, and
a slogan is exactly what a model can hold without reading. So the battery splits:

  SIX FRACTIONAL equations where clearing denominators first IS the strict cheapest road,
  and SIX INTEGER equations where the multiply option (dressed in the same textbook
  surface: "multiply both sides by 2") is pure waste — the cheap road is subtracting the
  x-term at once, two steps, done. Strictness is verified mechanically at build time:
  an equation enters the battery only if its three roads have a unique argmin.

The model sees the equation and three concrete first steps, letter order shuffled per
equation. The record then walks all three roads to the solved form and prices them by
the phase 80 residue metric — sum over steps of the largest denominator alive. The
model's letter against the priced argmin. A parrot that always multiplies scores 6/12;
a coin scores 4; only reading the equation scores above the parrot line.

Every road of every equation must land the identical exact x — the verifier runs under
the whole battery (36 roads), model or no model.
"""
import json
import random
import re
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eqbricks import apply_move, solve_direct  # noqa: E402


def road(st, first):
    """One first step, then the canonical finish; returns final state and residues."""
    cur = st
    res = []

    def do(mv):
        nonlocal cur
        cur = apply_move(cur, mv)
        res.append(max(c.denominator for c in cur))

    if first is not None:
        do(first)
    if cur[2] != 0:
        do(("movex", None))
    if cur[1] != 0:
        do(("movec", None))
    if cur[0] not in (0, 1):
        do(("mul", 1 / cur[0]))
    return cur, res


def lcm4(st):
    import math
    out = 1
    for c in st:
        out = out * c.denominator // math.gcd(out, c.denominator)
    return out


def three_roads(st):
    """(label, first move, human text) for multiply / divide / subtract-x."""
    k = lcm4(st)
    k = F(k) if k > 1 else F(2)            # integer equations get the wasteful double
    p, _, r, _ = st
    return [("mul_lcm", ("mul", k), f"multiply both sides by {k}"),
            ("div_lead", ("mul", 1 / p), f"divide both sides by {fmt_c(p)}"),
            ("sub_x", ("movex", None), f"subtract {fmt_c(r)}x from both sides")]


def fmt_c(c):
    return str(c) if c.denominator == 1 else f"{c.numerator}/{c.denominator}"


def render(st):
    p, q, r, s = st
    def side(a, b):
        ax = "x" if a == 1 else f"{fmt_c(a)}x"
        return ax if b == 0 else f"{ax} + {fmt_c(b)}" if b > 0 else f"{ax} - {fmt_c(-b)}"
    return f"{side(p, q)} = {side(r, s)}"


def build_battery(rng, want=6):
    """Fractional equations with strict-cheapest multiply, integer ones with
    strict-cheapest subtract — membership is mechanical, never assumed."""
    frac, intg = [], []
    while len(frac) < want or len(intg) < want:
        if len(frac) < want:
            st = (F(rng.randint(2, 9), rng.randint(2, 5)),
                  F(rng.randint(1, 20), rng.randint(2, 6)),
                  F(rng.randint(1, 6), rng.randint(2, 5)),
                  F(rng.randint(1, 20), rng.randint(2, 6)))
            if st[0] != st[2] and strict_argmin(st) == "mul_lcm":
                frac.append(st)
        if len(intg) < want:
            r = F(rng.randint(1, 5))
            st = (r + 1, F(rng.randint(1, 9)), r, F(rng.randint(1, 9)))
            if strict_argmin(st) == "sub_x":
                intg.append(st)
    return frac + intg


def strict_argmin(st):
    prices = {}
    xstar = solve_direct(st)
    for label, first, _ in three_roads(st):
        fin, res = road(st, first)
        if not (fin[0] == 1 and fin[2] == 0 and fin[3] == xstar):
            return None                    # a road that fails to solve disqualifies
        prices[label] = sum(res)
    best = min(prices.values())
    winners = [k for k, v in prices.items() if v == best]
    return winners[0] if len(winners) == 1 else None


PROMPT = """Equation: {eq}

Possible first steps:
A) {a}
B) {b}
C) {c}

Which first step leads to the full solve with the LEAST fraction-carrying overall
(fewest and smallest denominators along the way)? Reply with only the letter."""


def main(n_inspect=0, seed=13, out="data/custom/eqplan.json"):
    n_inspect, seed = int(n_inspect), int(seed)
    rng = random.Random(seed)
    battery = build_battery(rng)
    if n_inspect:
        battery = battery[:1] + battery[-1:]

    from cutbig import ask
    correct = parrot_mul = roads_exact = 0
    rows = []
    for st in battery:
        opts = three_roads(st)
        order = list(range(3))
        rng.shuffle(order)
        letters = "ABC"
        by_letter = {letters[i]: opts[order[i]] for i in range(3)}
        xstar = solve_direct(st)
        prices = {}
        for label, first, _ in opts:
            fin, res = road(st, first)
            roads_exact += fin == (F(1), F(0), F(0), xstar)
            prices[label] = sum(res)
        best_label = min(prices, key=prices.get)
        best_letter = next(le for le, o in by_letter.items() if o[0] == best_label)

        reply = ask("qwen-35b", PROMPT.format(
            eq=render(st), a=by_letter["A"][2], b=by_letter["B"][2],
            c=by_letter["C"][2]), n=16)
        if n_inspect:
            print(f"RAW [{render(st)}] best={best_letter}({best_label}) "
                  f"prices={prices}\n  reply: {reply!r}\n")
            continue
        m = re.search(r"\b([ABC])\b", reply)
        pick = m.group(1) if m else "?"
        picked_label = by_letter.get(pick, (None,))[0]
        ok = pick == best_letter
        correct += ok
        parrot_mul += picked_label == "mul_lcm"
        rows.append({"eq": render(st), "prices": {k: str(v) for k, v in prices.items()},
                     "best": best_label, "pick": pick, "picked": picked_label, "ok": ok})
        print(f"{render(st):<38} best {best_label:<8} ({prices[best_label]:>3}) "
              f"pick {picked_label or '?':<8} {'OK' if ok else 'X'}")
    if n_inspect:
        return

    n = len(battery)
    print(f"\nmodel picked the priced-cheapest first step: {correct}/{n}")
    print(f"  baselines: coin 4/12 expected, always-multiply parrot 6/12")
    print(f"  multiply picked {parrot_mul}/{n} times (6 of {n} equations warrant it)")
    print(f"all roads land the exact x under the battery: {roads_exact}/{3 * n}")
    print("\nThe plan is now a priced object: the record walks every road the model could")
    print("have chosen and the residue bill separates reading from reciting — the slogan")
    print("scores six, the coin four, and only looking at the denominators scores higher.")
    summary = {"equations": n, "correct": correct, "parrot_mul": parrot_mul,
               "roads_exact": roads_exact, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
