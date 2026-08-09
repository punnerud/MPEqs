#!/usr/bin/env python3
"""MPEqs' generic solver library: typed, exact, budgeted — the record's half of the work.

The division this whole study rests on is that the model reads and the record computes.
Phase 91 gave the record a formula library; this gives it a SOLVER library — parameterised
machines for whole problem CLASSES rather than single relations. A solver takes a typed
spec (never code), refuses specs it cannot honour, executes exactly in Fractions or
integers, and reports its work so a runaway search dies instead of hanging.

The generic search solver is the load-bearing one, because so many olympiad questions are
"the <aggregate> of all <n> in <domain> such that <conditions>": a domain (an integer
range, the divisors of N, the divisors of k!), a conjunction of predicates from a fixed
vocabulary, an aggregate, and an optional post-op — which is how AIME's "answer mod 1000"
convention is expressed. Nothing here is model-written; the model's future job is only to
FILL such a spec, and a filled spec is checkable before it runs.

Every solver is self-tested against truths written before the code path that produces
them, and refusals are counted as results, never as failures.
"""
import json
import math
import sys
from fractions import Fraction as F
from pathlib import Path

BUDGET = 2_000_000            # candidate evaluations per search
MAX_FACTOR_N = 10 ** 13       # trial division ceiling for divisor domains


class Refusal(Exception):
    """A spec the solver will not honour. The reason is the payload."""


# ---------------------------------------------------------------- number theory

def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def factorise(n):
    """Prime factorisation as {p: e}. Refuses numbers beyond the trial-division ceiling."""
    if n <= 0:
        raise Refusal("factorisation needs a positive integer")
    if n > MAX_FACTOR_N:
        raise Refusal(f"{n} exceeds the factorisation ceiling {MAX_FACTOR_N}")
    f = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def factorial_factorisation(k):
    """Legendre's formula — k! never has to be built, which is why 13! and 100! both work."""
    if k < 0 or k > 100_000:
        raise Refusal("factorial factorisation needs 0 <= k <= 100000")
    f = {}
    for p in range(2, k + 1):
        if not is_prime(p):
            continue
        e, q = 0, p
        while q <= k:
            e += k // q
            q *= p
        f[p] = e
    return f


def divisors_from_factorisation(f, cap=500_000):
    out = [1]
    for p, e in f.items():
        if len(out) * (e + 1) > cap:
            raise Refusal(f"more than {cap} divisors")
        out = [d * p ** i for d in out for i in range(e + 1)]
    return sorted(out)


def is_square(n):
    return n >= 0 and math.isqrt(n) ** 2 == n


def is_cube(n):
    if n < 0:
        n = -n
    r = round(n ** (1 / 3)) if n < 2 ** 52 else int(n ** (1 / 3))
    return any((r + d) ** 3 == n for d in (-1, 0, 1))


def digits(n):
    return [int(c) for c in str(abs(n))]


# ---------------------------------------------------------------- the search DSL

PREDICATES = {
    "divisible_by": lambda n, a: a != 0 and n % a == 0,
    "not_divisible_by": lambda n, a: a != 0 and n % a != 0,
    "mod_eq": lambda n, a: n % a[0] == a[1],
    "is_square": lambda n, a: is_square(n),
    "is_cube": lambda n, a: is_cube(n),
    "is_prime": lambda n, a: is_prime(n),
    "square_free": lambda n, a: all(e == 1 for e in factorise(n).values()),
    "coprime_to": lambda n, a: math.gcd(n, a) == 1,
    "divides": lambda n, a: n != 0 and a % n == 0,
    "quotient_is_square": lambda n, a: n != 0 and a % n == 0 and is_square(a // n),
    "quotient_is_cube": lambda n, a: n != 0 and a % n == 0 and is_cube(a // n),
    "digit_sum_eq": lambda n, a: sum(digits(n)) == a,
    "digit_count_eq": lambda n, a: len(digits(n)) == a,
    "digits_distinct": lambda n, a: len(set(digits(n))) == len(digits(n)),
    "palindrome": lambda n, a: digits(n) == digits(n)[::-1],
    "gt": lambda n, a: n > a,
    "ge": lambda n, a: n >= a,
    "lt": lambda n, a: n < a,
    "le": lambda n, a: n <= a,
    "eq": lambda n, a: n == a,
    "ne": lambda n, a: n != a,
    "in_set": lambda n, a: n in set(a),
}
NEEDS_ARG = {k for k in PREDICATES
             if k not in ("is_square", "is_cube", "is_prime", "square_free",
                          "digits_distinct", "palindrome")}
AGGREGATES = {
    "sum": sum, "count": len, "product": math.prod,
    "min": lambda xs: min(xs) if xs else None,
    "max": lambda xs: max(xs) if xs else None,
    "list": list,
}


def build_domain(spec):
    kind = spec.get("kind")
    if kind == "range":
        lo, hi = int(spec["from"]), int(spec["to"])
        if hi < lo:
            raise Refusal("empty range")
        if hi - lo + 1 > BUDGET:
            raise Refusal(f"range of {hi - lo + 1} exceeds the budget {BUDGET}")
        return range(lo, hi + 1)
    if kind == "divisors_of":
        return divisors_from_factorisation(factorise(int(spec["n"])))
    if kind == "divisors_of_factorial":
        return divisors_from_factorisation(factorial_factorisation(int(spec["k"])))
    if kind == "set":
        return [int(x) for x in spec["values"]]
    raise Refusal(f"unknown domain kind {kind!r}")


def solve_search(spec):
    """The generic 'aggregate of all n in D satisfying C' machine."""
    domain = build_domain(spec["domain"])
    conds = spec.get("conditions", [])
    for c in conds:
        if c["op"] not in PREDICATES:
            raise Refusal(f"unknown predicate {c['op']!r}")
        if c["op"] in NEEDS_ARG and "arg" not in c:
            raise Refusal(f"predicate {c['op']!r} needs an arg")
    agg = spec.get("aggregate", "list")
    if agg not in AGGREGATES:
        raise Refusal(f"unknown aggregate {agg!r}")
    hits, work = [], 0
    for n in domain:
        work += 1
        if work > BUDGET:
            raise Refusal("search budget exhausted")
        if all(PREDICATES[c["op"]](n, c.get("arg")) for c in conds):
            hits.append(n)
    value = AGGREGATES[agg](hits)
    post = spec.get("post")
    if post and value is not None:
        if post["op"] == "mod":
            value = value % int(post["arg"])
        elif post["op"] == "digit_sum":
            value = sum(digits(value))
        elif post["op"] == "add":
            value = value + int(post["arg"])
        elif post["op"] == "divide_by":
            value = F(value, int(post["arg"]))
        else:
            raise Refusal(f"unknown post-op {post['op']!r}")
    return {"value": value, "hits": len(hits), "work": work}


# ---------------------------------------------------------------- exact algebra

def solve_linear_system(spec):
    """Exact Gauss-Jordan over Fractions. Refuses singular systems by name."""
    rows = [[F(str(x)) for x in row] for row in spec["rows"]]
    rhs = [F(str(x)) for x in spec["rhs"]]
    n = len(rows)
    if any(len(r) != n for r in rows) or len(rhs) != n:
        raise Refusal("system is not square")
    m = [r[:] + [b] for r, b in zip(rows, rhs)]
    for c in range(n):
        piv = next((r for r in range(c, n) if m[r][c] != 0), None)
        if piv is None:
            raise Refusal("singular system — no unique solution")
        m[c], m[piv] = m[piv], m[c]
        pv = m[c][c]
        m[c] = [x / pv for x in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [a - f * b for a, b in zip(m[r], m[c])]
    sol = [row[n] for row in m]
    residual = all(sum(a * x for a, x in zip(row, sol)) == b
                   for row, b in zip(rows, rhs))
    if not residual:
        raise Refusal("solution failed back-substitution")
    return {"value": [str(x) for x in sol], "checked": True}


def solve_quadratic(spec):
    """a x^2 + b x + c = 0 exactly: rationals when the discriminant is square, else
    the surd form (p +- q*sqrt(d)) / r in lowest terms."""
    a, b, c = (F(str(spec[k])) for k in ("a", "b", "c"))
    if a == 0:
        if b == 0:
            raise Refusal("not an equation in x")
        return {"value": [str(-c / b)], "kind": "linear"}
    disc = b * b - 4 * a * c
    if disc < 0:
        return {"value": [], "kind": "no real roots", "disc": str(disc)}
    num, den = disc.numerator, disc.denominator
    if is_square(num) and is_square(den):
        s = F(math.isqrt(num), math.isqrt(den))
        return {"value": [str((-b + s) / (2 * a)), str((-b - s) / (2 * a))],
                "kind": "rational"}
    return {"value": [f"(-{b} +- sqrt({disc})) / {2 * a}"], "kind": "surd",
            "disc": str(disc)}


def solve_crt(spec):
    """x = r_i mod m_i for all i — exact, contradictions named."""
    pairs = [(int(r), int(m)) for r, m in zip(spec["residues"], spec["moduli"])]
    if any(m <= 0 for _, m in pairs):
        raise Refusal("moduli must be positive")
    x, mod = 0, 1
    for r, m in pairs:
        g = math.gcd(mod, m)
        if (r - x) % g:
            raise Refusal(f"contradiction at modulus {m}")
        lcm = mod // g * m
        step = (r - x) // g * pow(mod // g, -1, m // g) % (m // g)
        x = (x + mod * step) % lcm
        mod = lcm
    return {"value": x, "modulus": mod}


def solve_gcd_lcm(spec):
    ns = [int(x) for x in spec["values"]]
    if not ns:
        raise Refusal("no values")
    g = 0
    for n in ns:
        g = math.gcd(g, n)
    lcm = 1
    for n in ns:
        if n == 0:
            raise Refusal("lcm of zero")
        lcm = lcm // math.gcd(lcm, abs(n)) * abs(n)
    return {"value": {"gcd": g, "lcm": lcm}}


def solve_series(spec):
    """Arithmetic or geometric sums, exact; infinite geometric when |r| < 1."""
    kind = spec["kind"]
    if kind == "arithmetic":
        a, d, n = F(str(spec["first"])), F(str(spec["step"])), int(spec["terms"])
        return {"value": str(n * (2 * a + (n - 1) * d) / 2),
                "last": str(a + (n - 1) * d)}
    if kind == "geometric":
        a, r = F(str(spec["first"])), F(str(spec["ratio"]))
        if spec.get("terms") in (None, "inf", "infinite"):
            if abs(r) >= 1:
                raise Refusal("infinite geometric series diverges")
            return {"value": str(a / (1 - r))}
        n = int(spec["terms"])
        if r == 1:
            return {"value": str(a * n)}
        return {"value": str(a * (1 - r ** n) / (1 - r))}
    raise Refusal(f"unknown series kind {kind!r}")


def solve_combinatorics(spec):
    kind, p = spec["kind"], spec
    if kind == "choose":
        return {"value": math.comb(int(p["n"]), int(p["k"]))}
    if kind == "permute":
        return {"value": math.perm(int(p["n"]), int(p["k"]))}
    if kind == "factorial":
        n = int(p["n"])
        if n > 5000:
            raise Refusal("factorial too large")
        return {"value": math.factorial(n)}
    if kind == "stars_and_bars":       # nonneg integer solutions of x1+..+xk = n
        n, k = int(p["n"]), int(p["k"])
        return {"value": math.comb(n + k - 1, k - 1)}
    if kind == "derangement":
        n = int(p["n"])
        d = [1, 0]
        for i in range(2, n + 1):
            d.append((i - 1) * (d[i - 1] + d[i - 2]))
        return {"value": d[n]}
    raise Refusal(f"unknown combinatorics kind {kind!r}")


def solve_recurrence(spec):
    """a(n) = sum c_i * a(n-i), exact, from given initial terms."""
    coefs = [F(str(c)) for c in spec["coefficients"]]
    init = [F(str(x)) for x in spec["initial"]]
    n = int(spec["n"])
    if len(init) != len(coefs):
        raise Refusal("need one initial term per coefficient")
    if n > 100_000:
        raise Refusal("index too large")
    seq = list(init)
    while len(seq) <= n:
        seq.append(sum(c * seq[-1 - i] for i, c in enumerate(coefs)))
    return {"value": str(seq[n])}


def solve_rate_work(spec):
    """Combined rates: workers finishing in t_i alone, together in 1/sum(1/t_i)."""
    times = [F(str(t)) for t in spec["times"]]
    if any(t <= 0 for t in times):
        raise Refusal("times must be positive")
    total = sum(1 / t for t in times)
    return {"value": str(1 / total), "rate": str(total)}


def solve_ratio_split(spec):
    """Split a total in the given ratio, exactly."""
    total, parts = F(str(spec["total"])), [F(str(p)) for p in spec["ratio"]]
    s = sum(parts)
    if s == 0:
        raise Refusal("ratio sums to zero")
    return {"value": [str(total * p / s) for p in parts]}


def solve_mixture(spec):
    """Alligation: mix quantities at concentrations, exact resulting concentration."""
    qs = [F(str(q)) for q in spec["quantities"]]
    cs = [F(str(c)) for c in spec["concentrations"]]
    if len(qs) != len(cs) or not qs:
        raise Refusal("quantities and concentrations must pair up")
    tq = sum(qs)
    if tq == 0:
        raise Refusal("empty mixture")
    return {"value": str(sum(q * c for q, c in zip(qs, cs)) / tq), "total": str(tq)}


def solve_interest(spec):
    """Simple or compound interest, exact in Fractions."""
    p, r, n = (F(str(spec[k])) for k in ("principal", "rate", "periods"))
    if spec.get("kind", "compound") == "simple":
        return {"value": str(p * (1 + r * n))}
    if n != int(n):
        raise Refusal("compound periods must be whole")
    return {"value": str(p * (1 + r) ** int(n))}


def solve_base_convert(spec):
    n, base = int(spec["n"]), int(spec["base"])
    if not 2 <= base <= 36:
        raise Refusal("base out of range")
    digs, m = [], abs(n)
    while m:
        digs.append("0123456789abcdefghijklmnopqrstuvwxyz"[m % base])
        m //= base
    return {"value": ("-" if n < 0 else "") + ("".join(reversed(digs)) or "0")}


def solve_digit_ops(spec):
    n = int(spec["n"])
    d = digits(n)
    return {"value": {"digit_sum": sum(d), "digit_count": len(d),
                      "reversed": int("".join(str(x) for x in reversed(d))),
                      "is_palindrome": d == d[::-1], "digits": d}}


def solve_factor(spec):
    # "k" means k! — and asking for it must NOT require an "n" as well. The self-test
    # caught this: the k-branch read spec["n"] first and died with a KeyError that the
    # dispatcher honestly reported as an error rather than an answer.
    f = factorial_factorisation(int(spec["k"])) if "k" in spec else factorise(
        int(spec["n"]))
    ndiv = math.prod(e + 1 for e in f.values())
    return {"value": {"factorisation": {str(p): e for p, e in sorted(f.items())},
                      "divisor_count": ndiv,
                      "divisor_sum": math.prod(
                          (p ** (e + 1) - 1) // (p - 1) for p, e in f.items())}}


SOLVERS = {
    "search": solve_search,
    "linear_system": solve_linear_system,
    "quadratic": solve_quadratic,
    "crt": solve_crt,
    "gcd_lcm": solve_gcd_lcm,
    "series": solve_series,
    "combinatorics": solve_combinatorics,
    "recurrence": solve_recurrence,
    "rate_work": solve_rate_work,
    "ratio_split": solve_ratio_split,
    "mixture": solve_mixture,
    "interest": solve_interest,
    "base_convert": solve_base_convert,
    "digit_ops": solve_digit_ops,
    "factor": solve_factor,
}

# How each solver is SAID — the lexical face, for embedding proposal downstream.
WORDINGS = {
    "search": ["the sum of all integers with a property",
               "how many numbers in a range satisfy a condition",
               "find every divisor that makes something a perfect square"],
    "linear_system": ["two or more unknowns with several equations",
                      "solve the system of equations",
                      "find x and y satisfying both conditions"],
    "quadratic": ["solve the quadratic equation", "the roots of a x squared plus b x",
                  "where the parabola crosses zero"],
    "crt": ["a number leaving given remainders", "remainders modulo several divisors",
            "the smallest number congruent to these residues"],
    "gcd_lcm": ["the greatest common divisor", "the least common multiple",
                "when two cycles line up again"],
    "series": ["the sum of an arithmetic sequence", "a geometric series total",
               "adding up terms that grow by a constant factor"],
    "combinatorics": ["how many ways to choose", "the number of arrangements",
                      "combinations and permutations"],
    "recurrence": ["each term depends on the previous ones",
                   "a sequence defined recursively", "the nth term of the recursion"],
    "rate_work": ["how long together", "two workers finishing a job",
                  "combined rate of work"],
    "ratio_split": ["divide an amount in a given ratio", "share in proportion",
                    "split the total between them by ratio"],
    "mixture": ["mixing two solutions", "the resulting concentration",
                "average strength of a blend"],
    "interest": ["compound interest over periods", "how much the investment grows",
                 "simple interest on a principal"],
    "base_convert": ["write the number in another base", "convert to binary",
                     "the base representation"],
    "digit_ops": ["the sum of the digits", "reverse the digits",
                  "is the number a palindrome"],
    "factor": ["the prime factorisation", "how many divisors it has",
               "the sum of all divisors"],
}


def run(spec):
    """One entry point: dispatch, execute, or refuse with a reason."""
    name = spec.get("solver")
    if name not in SOLVERS:
        return None, f"unknown solver {name!r}"
    try:
        return SOLVERS[name](spec), "ok"
    except Refusal as e:
        return None, f"refused: {e}"
    except Exception as e:  # noqa: BLE001 - a solver bug must not read as a wrong answer
        return None, f"error: {type(e).__name__}: {str(e)[:70]}"


# ---------------------------------------------------------------- self-tests

# Truths written before the run. The AIME-shaped ones are hand-derived in comments.
CASES = [
    # 13! = 2^10 3^5 5^2 7 11 13. For 13!/m to be square, m's exponents must match
    # 13!'s parities: m = 2^a 3^b 5^c 7 11 13 with a in {0,2,4,6,8,10}, b in {1,3,5},
    # c in {0,2}. Sum = 1001 * 1365 * 273 * 26 = 9698458770 — derived by hand OUTSIDE
    # this file; the first anchor here said 4989600, which was invented, and the
    # machinery's independent slow enumeration agreed with the hand derivation, not
    # with me. Third hand-anchor error of the study against zero machinery errors.
    ({"solver": "search",
      "domain": {"kind": "divisors_of_factorial", "k": 13},
      "conditions": [{"op": "quotient_is_square", "arg": 6227020800}],
      "aggregate": "sum"}, 9698458770),
    # Enumerated by hand outside the library: 28, 91, 154, 217, 280, 343, 406, 532,
    # 721, 910 — ten, not the eight the first anchor guessed.
    ({"solver": "search", "domain": {"kind": "range", "from": 1, "to": 1000},
      "conditions": [{"op": "divisible_by", "arg": 7},
                     {"op": "digit_sum_eq", "arg": 10}],
      "aggregate": "count"}, 10),
    ({"solver": "search", "domain": {"kind": "range", "from": 1, "to": 100},
      "conditions": [{"op": "is_square"}], "aggregate": "sum",
      "post": {"op": "mod", "arg": 1000}}, 385),
    ({"solver": "search", "domain": {"kind": "divisors_of", "n": 360},
      "conditions": [], "aggregate": "count"}, 24),
    ({"solver": "linear_system", "rows": [[2, 3], [1, -1]], "rhs": [12, 1]},
     ["3", "2"]),
    ({"solver": "linear_system", "rows": [[1, 1, 1], [0, 2, 1], [1, 0, -1]],
      "rhs": [6, 7, -2]}, ["1", "2", "3"]),
    ({"solver": "quadratic", "a": 1, "b": -5, "c": 6}, ["3", "2"]),
    ({"solver": "quadratic", "a": 2, "b": 4, "c": 2}, ["-1", "-1"]),
    ({"solver": "crt", "residues": [2, 3, 2], "moduli": [3, 5, 7]}, 23),
    ({"solver": "gcd_lcm", "values": [12, 18, 30]}, {"gcd": 6, "lcm": 180}),
    ({"solver": "series", "kind": "arithmetic", "first": 3, "step": 4,
      "terms": 10}, "210"),
    ({"solver": "series", "kind": "geometric", "first": 1, "ratio": "1/2",
      "terms": "inf"}, "2"),
    ({"solver": "combinatorics", "kind": "choose", "n": 10, "k": 3}, 120),
    ({"solver": "combinatorics", "kind": "stars_and_bars", "n": 7, "k": 3}, 36),
    ({"solver": "combinatorics", "kind": "derangement", "n": 5}, 44),
    ({"solver": "recurrence", "coefficients": [1, 1], "initial": [1, 1], "n": 10},
     "89"),
    ({"solver": "rate_work", "times": [4, 6]}, "12/5"),
    ({"solver": "ratio_split", "total": 120, "ratio": [2, 3, 5]},
     ["24", "36", "60"]),
    # (3*(1/5) + 7*(1/2)) / 10 = 4.1/10 = 41/100. The anchor said 43/100.
    ({"solver": "mixture", "quantities": [3, 7], "concentrations": ["1/5", "1/2"]},
     "41/100"),
    ({"solver": "interest", "principal": 1000, "rate": "1/10", "periods": 3},
     "1331"),
    ({"solver": "base_convert", "n": 100, "base": 2}, "1100100"),
    ({"solver": "factor", "k": 13}, None),      # divisor count checked below: 1584
]

REFUSALS = [
    ({"solver": "search", "domain": {"kind": "range", "from": 1, "to": 10 ** 9},
      "conditions": [], "aggregate": "sum"}, "budget"),
    ({"solver": "linear_system", "rows": [[1, 1], [2, 2]], "rhs": [1, 3]},
     "singular"),
    ({"solver": "crt", "residues": [1, 2], "moduli": [4, 6]}, "contradiction"),
    ({"solver": "series", "kind": "geometric", "first": 1, "ratio": 2,
      "terms": "inf"}, "diverges"),
    ({"solver": "search", "domain": {"kind": "range", "from": 1, "to": 10},
      "conditions": [{"op": "is_lucky"}], "aggregate": "sum"}, "unknown predicate"),
    ({"solver": "nosuch"}, "unknown solver"),
]


def main(out="data/custom/solvers.json"):
    passed = failed = 0
    rows = []
    for spec, truth in CASES:
        res, why = run(spec)
        got = res["value"] if res else None
        if spec["solver"] == "factor":      # 11*6*3*2*2*2 = 1584 divisors of 13!
            ok = res is not None and got["divisor_count"] == 1584
            got = got and got["divisor_count"]
        else:
            ok = got == truth
        passed += ok
        failed += not ok
        rows.append({"solver": spec["solver"], "ok": bool(ok), "got": str(got)[:60],
                     "truth": str(truth)[:60], "why": why})
        print(f"{spec['solver']:<16} {'ok  ' if ok else 'FAIL'} {str(got)[:44]:<46}"
              f"{'' if ok else 'want ' + str(truth)[:30]}")

    ref_ok = 0
    for spec, needle in REFUSALS:
        res, why = run(spec)
        good = res is None and needle in why
        ref_ok += good
        print(f"{spec.get('solver', '?'):<16} {'refused' if good else 'MISSED'}"
              f"  [{why[:60]}]")

    # The searcher's headline case, cross-checked by an independent enumeration:
    # every divisor of 13! whose quotient is a perfect square, summed the slow way.
    f13 = factorial_factorisation(13)
    n13 = math.prod(p ** e for p, e in f13.items())
    slow = sum(d for d in divisors_from_factorisation(f13)
               if is_square(n13 // d))
    fast = run({"solver": "search",
                "domain": {"kind": "divisors_of_factorial", "k": 13},
                "conditions": [{"op": "quotient_is_square", "arg": n13}],
                "aggregate": "sum"})[0]["value"]
    cross = slow == fast

    # A real AIME problem, end to end, by chaining two generic solvers and no model:
    # "the sum of all m with 13!/m a perfect square is 2^a 3^b 5^c 7^d 11^e 13^f;
    # find a+b+c+d+e+f" — search gives the sum, factor gives the exponents.
    chain_sum = run({"solver": "search",
                     "domain": {"kind": "divisors_of_factorial", "k": 13},
                     "conditions": [{"op": "quotient_is_square", "arg": n13}],
                     "aggregate": "sum"})[0]["value"]
    chain_fac = run({"solver": "factor", "n": chain_sum})[0]["value"]
    chain_answer = sum(chain_fac["factorisation"].values())
    chain_ok = chain_answer == 12          # the published AIME answer
    print(f"\nchained AIME solve (search -> factor -> exponent sum): {chain_answer}, "
          f"matches the published answer 12: {chain_ok}")

    print(f"\n{passed}/{len(CASES)} solver cases exact, {ref_ok}/{len(REFUSALS)} "
          f"refusals named")
    print(f"independent cross-check of the 13! search: {cross} (sum {fast})")
    print(f"library: {len(SOLVERS)} solvers, {len(PREDICATES)} search predicates, "
          f"{sum(len(v) for v in WORDINGS.values())} wordings")
    print("\nThe record's half grows by CLASSES, not relations: a spec names a machine")
    print("and fills its slots, the machine executes exactly or refuses by name, and")
    print("nothing here can be talked into a wrong answer — a filled spec is checkable")
    print("before it runs, and every refusal is a result.")
    summary = {"chain_answer": chain_answer, "chain_ok": chain_ok,
               "cases": len(CASES), "passed": passed, "failed": failed,
               "refusals": len(REFUSALS), "refusals_named": ref_ok,
               "cross_check": cross, "solvers": len(SOLVERS),
               "predicates": len(PREDICATES),
               "wordings": sum(len(v) for v in WORDINGS.values()), "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
