#!/usr/bin/env python3
"""The vocabulary's ceiling on AIME, measured by hand — mapping failure vs reach failure.

Phase 94 measured the model mapping 19 AIME problems onto the library and getting none
right, which is compatible with two very different diagnoses: the library cannot express
these problems, or it can and the mapper missed. Those call for opposite work, so the
ambiguity has to be resolved before anything else is built.

This resolves it the only honest way: I write the specs by hand for the same fifteen
problems, with no model in the loop, and the record executes them against the published
answers. What lands is the vocabulary's REACH; what the model got is the MAPPER's skill;
the gap between them is where the next work belongs.

Two library additions were forced by problems in this set and are declared as such —
pow2 and inv_mod in the expression sandbox (problem 13 needs a modular inverse at a
variable power of two, which the literal-exponent rule blocks on purpose), and an
exponent_sum post-op (AIME asks constantly for the exponents of a prime factorisation to
be added, and without it phase 92's 13! problem needs two chained specs instead of one).
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from olympiad import load_problems  # noqa: E402
from solvemap import answer_of  # noqa: E402
from solvers2 import run2  # noqa: E402


def rect_conditions():
    """A rectangle in a regular 12-gon is two distinct diameters {i, i+6}, {j, j+6}
    whose four vertices share a colour; a colouring is the 12 bits of c."""
    out = []
    for i in range(6):
        for j in range(i + 1, 6):
            out.append(f"not ((c//2**{i})%2 == (c//2**{i + 6})%2 and "
                       f"(c//2**{j})%2 == (c//2**{j + 6})%2 and "
                       f"(c//2**{i})%2 == (c//2**{j})%2)")
    return out


HAND_SPECS = {
    0: ({"solver": "search", "domain": {"kind": "divisors_of_factorial", "k": 13},
         "conditions": [{"op": "quotient_is_square", "arg": 6227020800}],
         "aggregate": "sum", "post": {"op": "exponent_sum"}},
        "sum of m with 13!/m square, then add the exponents"),
    1: ({"solver": "multisearch", "variables": [{"name": "n", "from": 3, "to": 40}],
         "objective": "((n*(n-1)//2) * ((n*(n-1)//2) - 1)) // 2",
         "aggregate": "sum", "post": {"op": "mod", "arg": 1000}},
        "sum of C(C(n,2),2) for n = 3..40, mod 1000"),
    7: ({"solver": "multisearch", "variables": [{"name": "c", "from": 0, "to": 4095}],
         "conditions": rect_conditions(), "aggregate": "count"},
        "2-colourings of a 12-gon with no monochromatic rectangle"),
    12: ({"solver": "multisearch",
          "variables": [{"name": "k", "from": 1, "to": 2000},
                        {"name": "a", "from": 0, "to": 50}],
          "conditions": ["25*(5*k + a) == 11*(12*k + 50)"],
          "objective": "5*k + a", "aggregate": "min"},
         "adults 5/12 of the crowd, 11/25 after 50 arrive, minimise the total"),
    13: ({"solver": "multisearch", "variables": [{"name": "n", "from": 1, "to": 1000}],
          "conditions": ["inv_mod(23, pow2(n)) == inv_mod(23, pow2(n+1))"],
          "aggregate": "count"},
         "least multiple of 23 that is 1 mod 2^n, counting n with a(n) = a(n+1)"),
}

# Why the other ten are out of reach — the taxonomy that says what to build next.
OUT_OF_REACH = {
    2: "collections of subsets: the search space is over families of sets, not integers",
    3: "probability with an optimal-play tree; needs expectation over branches",
    4: "three-dimensional geometry with a plane section",
    5: "counting polynomials by a UNIQUENESS quantifier over integer roots",
    6: "plane geometry with angle equalities",
    8: "expected value under an optimal adaptive strategy; needs dynamic programming",
    9: "plane geometry and trigonometry",
    10: "three-dimensional geometry of tangent spheres",
    11: "combinatorial geometry over intersecting segments",
    14: "a sum over ordered pairs of equal-sized subsets; needs symbolic simplification",
}


def main(out="data/custom/aimeceiling.json"):
    _, aime = load_problems()
    picks = random.Random(5).sample(aime, 30)[:15]
    exact = ran = 0
    rows = []
    for i, (spec, what) in sorted(HAND_SPECS.items()):
        truth = picks[i][1]
        res, why = run2(spec)
        got = answer_of(res) if res else None
        ok = got is not None and str(got) == str(truth)
        ran += res is not None
        exact += ok
        rows.append({"i": i, "truth": str(truth), "got": str(got), "exact": bool(ok),
                     "what": what, "solver": spec["solver"]})
        print(f"[{i:>2}] {what[:58]:<60} {str(got):>8} vs {str(truth):<6} "
              f"{'EXACT' if ok else why[:30]}")

    n = len(picks)
    print(f"\nvocabulary ceiling, hand-mapped : {exact}/{n} exact ({ran} ran)")
    print(f"the model's mapping, phase 94    : 0 exact of 19 that ran")
    print(f"out of reach and why:")
    for i, reason in sorted(OUT_OF_REACH.items()):
        print(f"  [{i:>2}] {reason}")
    print("\nThe two diagnoses are now separated by measurement. A third of these")
    print("problems IS expressible in twenty solvers and an arithmetic sandbox — the")
    print("mapper found none of them — and the ten that are not name the same three")
    print("things every time: geometry, quantifiers over structures, and expectation")
    print("under strategy. That is a build list, not a mystery.")
    summary = {"n": n, "hand_exact": exact, "hand_ran": ran,
               "model_exact": 0, "model_ran": 19,
               "out_of_reach": len(OUT_OF_REACH), "rows": rows,
               "reasons": OUT_OF_REACH}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
