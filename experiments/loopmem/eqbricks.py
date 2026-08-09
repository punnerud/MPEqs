#!/usr/bin/env python3
"""First steps in algebra: both-sides bricks, residue-aware roads, ping-pong to the goal.

An equation is a STATE — both sides linear in x, four exact coefficients — and the first
steps of school algebra are its bricks: add to both sides, subtract, multiply, divide,
move the x-terms over. Every brick is reversible (its inverse is the opposite operation),
so every solving road unwinds, and every road must land the same x — the many-roads
verifier of phase 74, now over algebra.

The proposal's sharp edge is RESIDUE: a bad first step drags complexity through the whole
solve. Divide early and every later coefficient carries the fraction; multiply both sides
by the denominators FIRST and the residue collapses at once. Three strategies race on the
same equations, and residue is counted, not felt: the largest denominator alive after
each step, summed along the road.

And since the goal is KNOWN — we have a solution to reach — the search ping-pongs:
a forward frontier from the equation, a backward frontier from x = value using the
inverse bricks, expanded alternately, deduplicated by canonical state, meeting in the
middle. Streamed: the move tree is never materialised, only the two rims, and the states
the meeting saves against forward-only search are counted.
"""
import json
import random
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# State: (p, q, r, s) meaning  p*x + q = r*x + s. Solved form: (1, 0, 0, x*).


def solve_direct(st):
    p, q, r, s = st
    return (s - q) / (p - r)


def apply_move(st, mv):
    p, q, r, s = st
    kind, k = mv
    if kind == "add":                      # both sides + k
        return (p, q + k, r, s + k)
    if kind == "mul":                      # both sides * k (k != 0)
        return (p * k, q * k, r * k, s * k)
    if kind == "movex":                    # subtract r*x from both sides
        return (p - r, q, F(0), s)
    if kind == "movec":                    # subtract q from both sides
        return (p, F(0), r, s - q)
    raise ValueError(kind)


def invert_move(mv):
    kind, k = mv
    if kind == "add":
        return ("add", -k)
    if kind == "mul":
        return ("mul", 1 / k)
    return None                            # movex/movec fold state; inverse needs the fold


def residue_of(st):
    """The complexity a step leaves behind: the largest denominator still alive."""
    return max(c.denominator for c in st)


def canon(st):
    p, q, r, s = st
    # Scale-invariant canonical form so forward and backward frontiers can meet.
    for lead in (p - r, p, r, q, s):
        if lead != 0:
            return tuple(c / lead for c in st)
    return st


def strategy_road(st, order):
    """One deterministic road: the given first-step ORDER, then finish canonically."""
    road, res = [], []
    cur = st

    def do(mv):
        nonlocal cur
        cur = apply_move(cur, mv)
        road.append(mv)
        res.append(residue_of(cur))

    if order == "clear_first":             # multiply by the lcm of denominators FIRST
        lcm = 1
        for c in cur:
            lcm = lcm * c.denominator // __import__("math").gcd(lcm, c.denominator)
        if lcm != 1:
            do(("mul", F(lcm)))
    elif order == "divide_early":          # normalise the leading coefficient at once
        if cur[0] not in (0, 1):
            do(("mul", 1 / cur[0]))
    # common finish: gather x, gather constants, normalise
    if cur[2] != 0:
        do(("movex", None))
    if cur[1] != 0:
        do(("movec", None))
    if cur[0] not in (0, 1):
        do(("mul", 1 / cur[0]))
    return cur, road, res


def pingpong(st, goal_x, max_rounds=6):
    """Bidirectional: forward from the equation, backward from (1,0,0,x*) with inverse
    bricks, alternating — the rims stream, the tree never exists."""
    goal = (F(1), F(0), F(0), goal_x)
    fwd = {canon(st): []}
    bwd = {canon(goal): []}
    explored = 2

    def moves_from(cur_states, forward):
        out = {}
        for cst, path in cur_states.items():
            for mv in (("movex", None), ("movec", None), ("mul", F(2)), ("mul", F(1, 2)),
                       ("add", F(1)), ("add", F(-1))):
                use = mv if forward else (invert_move(mv) or mv)
                try:
                    nst = canon(apply_move(cst, use))
                except ZeroDivisionError:
                    continue
                if nst not in cur_states and nst not in out:
                    out[nst] = path + [use]
        return out

    for rnd in range(max_rounds):
        side, other = (fwd, bwd) if rnd % 2 == 0 else (bwd, fwd)   # the ping pong
        new = moves_from(side, forward=(rnd % 2 == 0))
        explored += len(new)
        side.update(new)
        meet = set(side) & set(other)
        if meet:
            m = next(iter(meet))
            return True, len(fwd[m] if m in fwd else []) + len(bwd.get(m, [])), explored
    return False, 0, explored


def main(n_eq=20, seed=11, out="data/custom/eqbricks.json"):
    n_eq, seed = int(n_eq), int(seed)
    rng = random.Random(seed)
    eqs = []
    while len(eqs) < n_eq:
        p, r = F(rng.randint(2, 9), rng.randint(2, 5)), F(rng.randint(1, 6), rng.randint(2, 5))
        q, s = F(rng.randint(1, 30), rng.randint(2, 6)), F(rng.randint(1, 30), rng.randint(2, 6))
        if p != r:
            eqs.append((p, q, r, s))

    strategies = ("clear_first", "divide_early", "plain")
    agree = 0
    stats = {st: {"steps": 0, "residue": 0, "peak": 0} for st in strategies}
    pp_found = pp_states = fwd_only_states = 0
    for eq in eqs:
        xstar = solve_direct(eq)
        finals = []
        for strat in strategies:
            fin, road, res = strategy_road(eq, strat)
            finals.append(fin[3] if fin[0] == 1 and fin[2] == 0 else None)
            stats[strat]["steps"] += len(road)
            stats[strat]["residue"] += sum(res)
            stats[strat]["peak"] = max(stats[strat]["peak"], max(res) if res else 0)
        agree += all(f == xstar for f in finals)

        found, plen, explored = pingpong(eq, xstar)
        pp_found += found
        pp_states += explored
        # forward-only reference: rounds^2-ish growth; measure by running only fwd side.
        fwd = {canon(eq): 0}
        for _ in range(6):
            new = {}
            for cst in list(fwd):
                for mv in (("movex", None), ("movec", None), ("mul", F(2)),
                           ("mul", F(1, 2)), ("add", F(1)), ("add", F(-1))):
                    try:
                        nst = canon(apply_move(cst, mv))
                    except ZeroDivisionError:
                        continue
                    if nst not in fwd and nst not in new:
                        new[nst] = 0
            fwd.update(new)
            if canon((F(1), F(0), F(0), xstar)) in fwd:
                break
        fwd_only_states += len(fwd)

    n = n_eq
    print(f"{n} equations p*x + q = r*x + s with fractional coefficients\n")
    print(f"all roads land the same x (the verifier): {agree}/{n}\n")
    print(f"{'strategy':<14}{'steps':>7}{'total residue':>15}{'peak denominator':>18}")
    for strat in strategies:
        st = stats[strat]
        print(f"{strat:<14}{st['steps']:>7}{st['residue']:>15}{st['peak']:>18}")
    print(f"\nping-pong met in the middle on {pp_found}/{n}, exploring "
          f"{pp_states} states against {fwd_only_states} forward-only "
          f"({fwd_only_states / max(pp_states, 1):.1f}x)")
    print("\nThe first step decides the residue: clearing denominators before anything")
    print("else is the cheap road, dividing early is the expensive one, and every road —")
    print("priced differently — reaches the identical x, exactly. The goal being known is")
    print("what lets the search ping-pong, and meeting in the middle is what streaming")
    print("buys over growing one rim to the target.")
    summary = {"equations": n, "roads_agree": agree,
               **{f"{s}_residue": stats[s]["residue"] for s in strategies},
               **{f"{s}_steps": stats[s]["steps"] for s in strategies},
               "pingpong_met": pp_found, "pingpong_states": pp_states,
               "forward_only_states": fwd_only_states}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
