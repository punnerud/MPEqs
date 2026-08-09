#!/usr/bin/env python3
"""Combining formulas as routing: from the given quantities to the asked unit, mechanically.

The proposal: formulas COMBINE, so mapping input to solution is routing — the same move as the
unit graph, one level up. A problem's numbers arrive with units; the question names a target
unit; and the plan is a path through dimension space, built by combining quantities under the
type rules (add only like units, multiply and divide freely). Searched streaming, the N x N
discipline at micro scale: the frontier is deduplicated BY DIMENSION SIGNATURE, so the search
never materialises the tree of expressions — two ways of reaching "dollars" are one node, and
the state count is bounded by the distinct signatures, not the expression count.

The decisive property is UNIQUENESS. Where exactly one dimensional route reaches the asked
unit, the plan is derived — no model, no retrieval, no template. Where several routes type
correctly, dimensions alone cannot choose, and that ambiguity is precisely the role-binding
gap measured in phases 51 and 54. So the honest output is a triage of real problems:

    UNIQUE ROUTE    plan synthesised mechanically; is it right?
    AMBIGUOUS       dimensions reach the target several ways; someone must choose
    NO ROUTE        the units never combine into the target (tagger noise, or the problem
                    needs knowledge outside its own numbers)

GSM8K is count-heavy — many problems carry three numbers of the SAME unit, where dimensional
routing has nothing to grip. That is not a weakness of the test; it is the measurement of how
much of real problem-space this lever covers on its own.
"""
import json
import re
import sys
from fractions import Fraction
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dimcheck import asked_unit, units_of  # noqa: E402
from mapstore import NUM, TEST, norm  # noqa: E402

MAX_STEPS = 3
BEAM_PER_SIG = 4          # streamed: states kept per dimension signature, not per expression


def sig(dims):
    return tuple(sorted(dims.items()))


def combine(a, b):
    """Every typed combination of two dimensioned quantities. Additions must agree."""
    out = []
    if a[1] == b[1]:
        out.append((a[0] + b[0], a[1], "+"))
        out.append((a[0] - b[0], a[1], "-"))
    for x, y, op in ((a, b, "*"), (a, b, "/"), (b, a, "/")):
        if op == "*":
            d = dict(x[1])
            for k, e in y[1].items():
                d[k] = d.get(k, 0) + e
        else:
            if y[0] == 0:
                continue
            d = dict(x[1])
            for k, e in y[1].items():
                d[k] = d.get(k, 0) - e
        out.append(((x[0] * y[0]) if op == "*" else (x[0] / y[0]),
                    {k: e for k, e in d.items() if e}, op))
    return out


def route(quantities, target_dim, max_steps=MAX_STEPS):
    """All values reachable at the target signature within the step budget, streamed.

    Frontier states are (value, dims, sources) with sources the set of given quantities the
    state consumed; two states only combine when their sources are DISJOINT. Without that
    mask the search recombines a derived value with itself — x - x = 0, x + x = 2x — and
    every target signature drowns in fabricated candidates. It is the workpad's rule from
    phase 13, each given consumed once, resurfacing in dimension space: the first run
    reported 40 of 40 dimension-rich problems ambiguous, and all the extra candidates were
    self-combinations.

    Deduplicated by signature with a small beam each — the N x N discipline: the state space
    is signatures, never the expression tree.
    """
    frontier = [(Fraction(v), d, frozenset([i])) for i, (v, d) in enumerate(quantities)]
    reached = {}
    for _ in range(max_steps):
        by_sig = {}
        for a, b in combinations(frontier, 2):
            if a[2] & b[2]:
                continue
            for v, d, _ in combine((a[0], dict(a[1])), (b[0], dict(b[1]))):
                s = sig(d)
                bucket = by_sig.setdefault(s, [])
                key = (v, a[2] | b[2])
                if len(bucket) < BEAM_PER_SIG * 4 and all(
                        (x, m) != key for x, _, m in bucket):
                    bucket.append((v, d, a[2] | b[2]))
        nxt = list(frontier)
        for s, states in by_sig.items():
            for v, d, m in states[:BEAM_PER_SIG]:
                nxt.append((v, d, m))
                if s == sig(target_dim):
                    reached.setdefault(v, 0)
                    reached[v] += 1
        frontier = nxt
    return reached


def main(n_test=60, seed=5, out="data/custom/formularoute.json"):
    import random
    n_test, seed = int(n_test), int(seed)
    tests = []
    for line in TEST.read_text().splitlines():
        d = json.loads(line)
        tests.append((d["question"],
                      Fraction(norm(d["answer"].rsplit("#### ", 1)[-1].strip()))))
    tests = random.Random(seed).sample(tests, n_test)

    tally = {"unique": 0, "unique_right": 0, "ambiguous": 0, "ambig_contains": 0,
             "no_route": 0, "untaggable": 0}
    rows = []
    for q, truth in tests:
        values = [norm(m) for m in NUM.findall(norm(q))]
        units = units_of(q)
        want = asked_unit(q)
        if want is None or not values:
            tally["untaggable"] += 1
            rows.append({"stage": "untaggable"})
            continue
        quantities = [(v, {u: 1} if u else {}) for v, u in zip(values, units)]
        reached = route(quantities, {want: 1})
        # Plausibility filter, declared: GSM8K answers are non-negative integers.
        cands = sorted(v for v in reached if v.denominator == 1 and v >= 0)
        if not cands:
            tally["no_route"] += 1
            stage = "no route"
        elif len(cands) == 1:
            tally["unique"] += 1
            tally["unique_right"] += cands[0] == truth
            stage = f"unique -> {cands[0]} ({'right' if cands[0] == truth else 'wrong'})"
        else:
            tally["ambiguous"] += 1
            tally["ambig_contains"] += truth in cands
            stage = f"ambiguous ({len(cands)} candidates, truth present: {truth in cands})"
        rows.append({"q": q[:60], "stage": stage, "n_candidates": len(cands)})

    n = len(tests)
    print(f"{n} GSM8K problems triaged by dimensional routing, no model:\n")
    print(f"  untaggable (no asked unit)      : {tally['untaggable']}")
    print(f"  no dimensional route            : {tally['no_route']}")
    print(f"  UNIQUE route                    : {tally['unique']}, "
          f"right {tally['unique_right']}/{tally['unique']}")
    print(f"  ambiguous                       : {tally['ambiguous']}, "
          f"truth among candidates {tally['ambig_contains']}/{tally['ambiguous']}")
    print("\nWhere the route is unique the plan came from the dimensions alone. Where it is")
    print("ambiguous, the candidate set is what routing can offer a chooser — the model, or")
    print("agreement, or a retrieved template — and its size is the honest measure of how")
    print("much choosing is left. Count-heavy problems route poorly BY NATURE: three numbers")
    print("of the same unit give dimensions nothing to grip.")
    Path(out).write_text(json.dumps({"summary": tally, "tests": n, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
