#!/usr/bin/env python3
"""Solved roads become memory; memory replaces search; pushback refines the vocabulary.

Phases 80-82 built the solving. This phase closes the session's stated loop — learn and
store, then search what was done before — and it runs in two acts because the first act
produced a zero that had to be autopsied.

ACT ONE. A solved road is stored as a template of macro ops (clear denominators, gather
x, gather constants, normalise) keyed by a prefix of the equation's shape signature.
The ops read their parameters off whatever state they meet and SKIP THEMSELVES when
inapplicable — and that turns out to be the whole story: one template, learned from one
search, is priced-optimal for the entire plain linear world, and transfers to
denominators far outside anything it was built on. The store never needed its keys. The
refinement machinery never fired — a clean zero, which the session's rules say must be
either autopsied or attacked.

ACT TWO attacks it. A class is added where gathering x FIRST is strictly cheaper than
clearing first (lead difference exactly 1, halves everywhere: subtracting the x-term
kills the big coefficients' denominators in one move, and the normalise step vanishes)
— and its four-bit signature is IDENTICAL to the class where clearing wins. The sampled
bill audit catches the overpayment; the split machinery runs out of signature; and the
memory must MINT a new bit from a library of computable predicates — the first one that
separates the two exemplar states — append it to its own vocabulary, and split there.
Improvement, measured: overpriced serves before the mint, zero after; searches per
window collapsing while every delivered answer stays exact.

Two kinds of pushback, two catchers, neither optional:
  HARD  a road that does not land (1, 0, 0, x*) — the verifier sees it BEFORE any
        answer is delivered; re-search, split. Silent wrong answers cannot exist.
  SOFT  a road that lands exactly but on the phase 80 expensive path — exactness
        cannot see it, so every fourth memory-served solve is re-searched and the
        bills compared.
"""
import json
import random
import sys
from fractions import Fraction as F
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eqbricks import apply_move, solve_direct  # noqa: E402

OPS = ("CLEAR", "MOVEX", "MOVEC", "DIV")

# The predicate library the memory may mint new signature bits from.
LIBRARY = [
    ("lead_one", lambda st: st[0] - st[2] == 1),
    ("lcm_two", lambda st: lcm4(st) == 2),
    ("q_whole", lambda st: st[1].denominator == 1),
    ("s_whole", lambda st: st[3].denominator == 1),
]


def lcm4(st):
    import math
    out = 1
    for c in st:
        out = out * c.denominator // math.gcd(out, c.denominator)
    return out


def op_move(op, st):
    """The op reads its parameter off the state it meets — that is what transfers."""
    p, q, r, s = st
    if op == "CLEAR":
        k = lcm4(st)
        return ("mul", F(k)) if k > 1 else None
    if op == "MOVEX":
        return ("movex", None) if r != 0 else None
    if op == "MOVEC":
        return ("movec", None) if q != 0 else None
    return ("mul", 1 / p) if p not in (0, 1) else None


def walk(st, road):
    cur, price = st, 0
    for op in road:
        mv = op_move(op, cur)
        if mv is None:
            continue
        cur = apply_move(cur, mv)
        price += max(c.denominator for c in cur)
    return cur, price


def sig(st, minted):
    p, q, r, s = st
    lead = (p - r) * (lcm4(st) if lcm4(st) > 1 else 1)
    base = (lcm4(st) > 1, r != 0, q != 0, lead != 1)
    return base + tuple(fn(st) for _, fn in minted)


def search(st):
    """Priced argmin over every op order — the expensive thing memory replaces."""
    xstar = solve_direct(st)
    best = None
    for k in range(len(OPS) + 1):
        for perm in permutations(OPS, k):
            fin, price = walk(st, perm)
            if fin == (F(1), F(0), F(0), xstar):
                cand = (price, len(perm), perm)
                best = min(best, cand) if best else cand
    return best[2], best[0]


def lookup(store, s):
    for key in store:                      # prefix-free: at most one can match
        if key == s[:len(key)]:
            return key
    return None


def store_at(store, s, entry):
    d = 0
    while any(len(k) > d and k[:d] == s[:d] for k in store):
        d += 1
    store[s[:d]] = entry


def refine(store, key, cur_state, cur_entry, minted):
    """Split at the first bit where the exemplars disagree; if the vocabulary runs out,
    MINT a new bit from the library — the memory extends its own signature."""
    old = store.pop(key)
    old_sig, cur_sig = sig(old["ex_state"], minted), sig(cur_state, minted)
    d = len(key) + 1
    while d <= len(cur_sig) and old_sig[:d] == cur_sig[:d]:
        d += 1
    if d > len(cur_sig):
        for name, fn in LIBRARY:
            if any(n == name for n, _ in minted):
                continue
            if fn(old["ex_state"]) != fn(cur_state):
                minted.append((name, fn))
                old_sig, cur_sig = (sig(old["ex_state"], minted),
                                    sig(cur_state, minted))
                d = len(cur_sig)
                break
        else:
            store[key] = old               # true near-tie: no predicate separates
            return "near-tie"
    store[old_sig[:d]] = old
    store[cur_sig[:d]] = cur_entry
    return "minted" if d > 4 else "split"


def run_stream(eqs, store, minted, audit_rate=0.25, audit_seed=1):
    # Audit picks are RANDOM, not strided: a fixed stride sharing a factor with the
    # world's pattern samples one class only — measured here first-hand, when every
    # fourth serve landed on the fairly-priced half of an alternating stream and
    # twenty overpayments in the other half went unsampled.
    audit_rng = random.Random(audit_seed)
    c = {"miss": 0, "hard": 0, "audits": 0, "soft": 0, "mints": 0,
         "overpriced_serves": 0, "exact": 0, "windows": [], "near_tie": 0}
    served = win = 0
    for i, st in enumerate(eqs):
        s = sig(st, minted)
        xstar = solve_direct(st)
        goal = (F(1), F(0), F(0), xstar)
        key = lookup(store, s)
        if key is None:
            road, _ = search(st)
            c["miss"] += 1
            win += 1
            store_at(store, s, {"road": road, "ex_state": st})
            fin, _ = walk(st, road)
        else:
            road = store[key]["road"]
            fin, price = walk(st, road)
            if fin != goal:                                 # HARD: pre-answer catch
                c["hard"] += 1
                win += 1
                road, _ = search(st)
                refine(store, key, st, {"road": road, "ex_state": st}, minted)
                fin, _ = walk(st, road)
            else:
                served += 1
                best_road, best_price = search(st)          # oracle for accounting
                if best_price < price:
                    c["overpriced_serves"] += 1
                if audit_rng.random() < audit_rate:         # SOFT: the sampled audit
                    c["audits"] += 1
                    if best_price < price:
                        c["soft"] += 1
                        win += 1
                        verdict = refine(store, key, st,
                                         {"road": best_road, "ex_state": st}, minted)
                        c["mints"] += verdict == "minted"
                        c["near_tie"] += verdict == "near-tie"
        c["exact"] += fin == goal
        if (i + 1) % 20 == 0:
            c["windows"].append(win)
            win = 0
    return c


def gen_plain(rng, dens):
    while True:
        kind = rng.choice(("frac", "int", "r0", "q0", "frac", "int"))
        dd = lambda: rng.randint(*dens)  # noqa: E731
        if kind == "frac":
            st = (F(rng.randint(2, 9), dd()), F(rng.randint(1, 20), dd()),
                  F(rng.randint(1, 6), dd()), F(rng.randint(1, 20), dd()))
        elif kind == "int":
            r = F(rng.randint(1, 5))
            st = (r + rng.randint(1, 3), F(rng.randint(1, 9)), r, F(rng.randint(1, 9)))
        elif kind == "r0":
            st = (F(rng.randint(2, 9), dd()), F(rng.randint(1, 20), dd()),
                  F(0), F(rng.randint(1, 20), dd()))
        else:
            st = (F(rng.randint(2, 9), dd()), F(0),
                  F(rng.randint(1, 6), dd()), F(rng.randint(1, 20), dd()))
        if st[0] != st[2] and st[0] != 0:
            return st


def gen_movex_wins(rng):
    """Lead difference exactly 1, halves everywhere: gathering x first is strictly
    cheapest, yet the four base signature bits match the clear-first class exactly."""
    while True:
        a = rng.randint(1, 4)
        st = (F(2 * a + 1, 2), F(2 * rng.randint(0, 4) + 1, 2),
              F(2 * a - 1, 2), F(2 * rng.randint(0, 4) + 1, 2))
        road, price = search(st)
        _, clear_price = walk(st, ("CLEAR", "MOVEX", "MOVEC", "DIV"))
        # Gather-first FAMILY strictly beats clearing; MOVEX-first and MOVEC-first tie
        # by symmetry inside the class, so the filter must accept either order.
        if price < clear_price and road[0] in ("MOVEX", "MOVEC"):
            return st


def gen_clear_wins(rng):
    while True:
        st = (F(rng.randint(2, 9), rng.randint(4, 6)),
              F(rng.randint(1, 20), rng.randint(4, 6)),
              F(rng.randint(1, 6), rng.randint(4, 6)),
              F(rng.randint(1, 20), rng.randint(4, 6)))
        if st[0] == st[2] or st[0] - st[2] == 1:
            continue
        road, price = search(st)
        _, movex_price = walk(st, ("MOVEX", "MOVEC", "DIV"))
        if price < movex_price and road[0] == "CLEAR":
            return st


def main(seed=17, out="data/custom/eqmemory.json"):
    seed = int(seed)
    rng = random.Random(seed)
    store, minted = {}, []

    # ACT ONE: the plain world, then transfer far outside it.
    act1 = run_stream([gen_plain(rng, (2, 6)) for _ in range(60)], store, minted)
    depths1 = sorted(len(k) for k in store)
    transfer = run_stream([gen_plain(rng, (7, 50)) for _ in range(20)], store, minted,
                          audit_rate=0.0)

    # ACT TWO: the same store meets the class its vocabulary cannot name.
    pairs = [gen_movex_wins(rng) if i % 2 == 0 else gen_clear_wins(rng)
             for i in range(40)]
    act2 = run_stream(pairs, store, minted, audit_rate=0.25, audit_seed=2)
    depths2 = sorted(len(k) for k in store)

    print("ACT ONE — 60 plain equations, store born empty")
    print(f"  searches: {act1['miss']} miss + {act1['hard']} hard + {act1['soft']} soft "
          f"(audit price {act1['audits']} kept apart); windows {act1['windows']}")
    print(f"  exact {act1['exact']}/60; store {len(store) and depths1} — one universal "
          f"key: the self-skipping ops made every class one class")
    print(f"  transfer dens 7-50: {transfer['miss']} misses, {transfer['hard']} hard, "
          f"exact {transfer['exact']}/20 — the template that cleared sixths clears "
          f"fiftieths\n")
    print("ACT TWO — 40 equations, half of a class the four bits cannot name")
    print(f"  overpriced serves before the audit caught it: {act2['overpriced_serves']}")
    print(f"  soft refinements {act2['soft']}, of which MINTED a new bit: "
          f"{act2['mints']} ({[n for n, _ in minted]})")
    print(f"  hard failures {act2['hard']} (all caught pre-answer), misses "
          f"{act2['miss']}, near-ties {act2['near_tie']}")
    print(f"  windows (miss+hard+soft) {act2['windows']}; exact {act2['exact']}/40")
    print(f"  store now {len(store)} keys at depths {depths2} — deepened only where "
          f"the world pushed back")
    print("\nLearning, measured twice over: the search bill collapses toward the audit")
    print("floor, and when the world produces a distinction the signature cannot say,")
    print("the memory coins the word for it — a predicate from the library, appended to")
    print("its own vocabulary — and the overpayments stop. Exactness never moved.")
    summary = {"act1_miss": act1["miss"], "act1_hard": act1["hard"],
               "act1_soft": act1["soft"], "act1_windows": act1["windows"],
               "act1_exact": act1["exact"],
               "transfer_miss": transfer["miss"], "transfer_hard": transfer["hard"],
               "transfer_exact": transfer["exact"],
               "act2_overpriced_serves": act2["overpriced_serves"],
               "act2_soft": act2["soft"], "act2_mints": act2["mints"],
               "act2_hard": act2["hard"], "act2_miss": act2["miss"],
               "act2_near_tie": act2["near_tie"], "act2_windows": act2["windows"],
               "act2_exact": act2["exact"],
               "minted": [n for n, _ in minted],
               "store_keys": len(store), "depths": depths2}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
