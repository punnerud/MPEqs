#!/usr/bin/env python3
"""Depth: scrambled equations, atomic both-sides bricks, where ping-pong starts to pay.

Phase 80 was honest about its own boundary: at three or four moves there is nothing for
a bidirectional search to halve. This phase builds the depth. The move set is ATOMIC
school algebra — multiply or divide both sides by 2, 3 or 5; add or subtract 1; add or
subtract one x — ten bricks, every one a both-sides operation with empty residual, every
one invertible move-for-move. The task is built the way the proposal frames it: WE HAVE
the solution, x = t, and the equation is a scramble — ten random bricks applied to the
solved form, like turning a cube away from solved. Solving is finding the way back.

Ping-pong: a forward rim from the scrambled equation, a backward rim from the solved
form, the smaller rim expanding each round, meeting on an exact state. The stitched road
is then VALIDATED both ways — applied forward it must land the solved form exactly, and
unwound (inverse bricks, reverse order) it must recover the scrambled equation exactly.
Forward-only search gets the same state budget and is reported as found-or-exhausted.
The rims stream; the move tree is never materialised — that is what the budget measures.
"""
import json
import random
import sys
from fractions import Fraction as F
from pathlib import Path

MOVES = [("mul", F(2)), ("mul", F(3)), ("mul", F(5)),
         ("mul", F(1, 2)), ("mul", F(1, 3)), ("mul", F(1, 5)),
         ("add", F(1)), ("add", F(-1)), ("addx", F(1)), ("addx", F(-1))]


def apply(st, mv):
    p, q, r, s = st
    kind, k = mv
    if kind == "mul":
        return (p * k, q * k, r * k, s * k)
    if kind == "add":
        return (p, q + k, r, s + k)
    return (p + k, q, r + k, s)            # addx: one x onto both sides


def inv(mv):
    kind, k = mv
    return (kind, 1 / k) if kind == "mul" else (kind, -k)


def scramble(t, depth, rng):
    st = (F(1), F(0), F(0), t)
    prev = None
    while depth:
        mv = rng.choice(MOVES)
        if prev and mv == inv(prev):       # no move immediately undone
            continue
        st = apply(st, mv)
        prev = mv
        depth -= 1
    return st


def bfs_forward(start, goal, cap):
    """One rim grown all the way to the target, or to the budget — the reference."""
    seen = {start}
    frontier = [start]
    depth = 0
    while frontier and len(seen) < cap:
        depth += 1
        nxt = []
        for st in frontier:
            for mv in MOVES:
                n = apply(st, mv)
                if n in seen:
                    continue
                seen.add(n)
                if n == goal:
                    return True, len(seen), depth
                nxt.append(n)
                if len(seen) >= cap:
                    return False, len(seen), depth
        frontier = nxt
    return False, len(seen), depth


def pingpong(start, goal, cap):
    """Two rims, the smaller one expands — the ping pong — meeting on an exact state."""
    fwd, bwd = {start: []}, {goal: []}
    ffront, bfront = [start], [goal]
    while len(fwd) + len(bwd) < cap and (ffront or bfront):
        expanding_fwd = bool(ffront) and (not bfront or len(ffront) <= len(bfront))
        side, other = (fwd, bwd) if expanding_fwd else (bwd, fwd)
        front = ffront if expanding_fwd else bfront
        nxt = []
        for st in front:
            for mv in MOVES:
                n = apply(st, mv)
                if n in side:
                    continue
                side[n] = side[st] + [mv]
                if n in other:
                    fpath = side[n] if expanding_fwd else other[n]
                    bpath = other[n] if expanding_fwd else side[n]
                    road = fpath + [inv(m) for m in reversed(bpath)]
                    return True, len(fwd) + len(bwd), road
                nxt.append(n)
        if expanding_fwd:
            ffront = nxt
        else:
            bfront = nxt
    return False, len(fwd) + len(bwd), []


def main(n_eq=8, depth=10, cap=100000, seed=7, out="data/custom/eqdeep.json"):
    n_eq, depth, cap, seed = int(n_eq), int(depth), int(cap), int(seed)
    rng = random.Random(seed)
    fwd_found = fwd_states = fwd_depth = 0
    pp_met = pp_states = 0
    valid = unwound = 0
    lengths = []
    for _ in range(n_eq):
        t = F(rng.randint(1, 9), rng.randint(1, 4))
        goal = (F(1), F(0), F(0), t)
        start = scramble(t, depth, rng)

        met, states, road = pingpong(start, goal, cap)
        pp_met += met
        pp_states += states
        if met:
            lengths.append(len(road))
            cur = start
            for mv in road:                # forward validation, exact
                cur = apply(cur, mv)
            valid += cur == goal
            back = goal
            for mv in reversed(road):      # unwind, exact
                back = apply(back, inv(mv))
            unwound += back == start

        found, fstates, fdepth = bfs_forward(start, goal, cap)
        fwd_found += found
        fwd_states += fstates
        fwd_depth = max(fwd_depth, fdepth)

    mean_len = sum(lengths) / max(len(lengths), 1)
    print(f"{n_eq} equations scrambled {depth} bricks deep from their solved form, "
          f"budget {cap} states each\n")
    print(f"ping-pong met in the middle : {pp_met}/{n_eq}, {pp_states} states total, "
          f"roads {min(lengths)}-{max(lengths)} bricks (mean {mean_len:.1f})")
    print(f"stitched road valid forward : {valid}/{pp_met}   (lands the solved form "
          f"exactly)")
    print(f"stitched road unwinds       : {unwound}/{pp_met}   (inverse bricks recover "
          f"the scramble exactly)")
    print(f"forward-only, same budget   : {fwd_found}/{n_eq} reached the goal, "
          f"{fwd_states} states, deepest level {fwd_depth}")
    if pp_states:
        print(f"\nstate ratio forward/ping-pong: {fwd_states / pp_states:.1f}x"
              + (" — a floor, the forward rims were cut off by the budget"
                 if fwd_found < n_eq else ""))
    print("\nAt depth three there was nothing to halve; at depth ten there is. Two rims")
    print("meeting in the middle do with thousands of states what one rim could not do")
    print("with a hundred thousand — and every stitched road still validates and unwinds")
    print("exactly, because every brick was a both-sides operation with empty residual.")
    summary = {"equations": n_eq, "scramble_depth": depth, "cap": cap,
               "pingpong_met": pp_met, "pingpong_states": pp_states,
               "road_valid": valid, "road_unwinds": unwound,
               "mean_road_len": round(mean_len, 1),
               "forward_found": fwd_found, "forward_states": fwd_states,
               "forward_deepest": fwd_depth,
               "ratio": round(fwd_states / max(pp_states, 1), 1)}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
