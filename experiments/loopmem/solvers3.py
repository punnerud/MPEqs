#!/usr/bin/env python3
"""MPEqs' domain machines — the second half of the library, split out at 1500 lines.

solvers2.py keeps the ENGINE: the expression sandbox, the pruned tuple search, the fold,
the named-step arithmetic and the dispatcher that lets a shape overrule a name.
Everything here is a DOMAIN: probability, unit conversion, statistics, calendars,
finance, shapes, set counting, the formula library, sequences, matrices, partitions,
logarithms, base arithmetic, rational approximation, word counting and primes.

The split is mechanical, not conceptual: same Fractions, same Refusal, same discipline
of naming what a machine will not do. solvers2 imports this at the bottom, so run2 still
dispatches across the whole library and every caller keeps working unchanged.
"""
import math
import re
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from solvers import Refusal, digits, factorise, is_prime, is_square  # noqa: E402,F401
from solvers2 import compile_expr, eval_expr, solve_multisearch  # noqa: E402

SHAPES = {
    # name: (slots, area/volume as (rational, pi_power), perimeter/surface likewise)
    "circle": ("radius",),
    "rectangle": ("length", "width"),
    "triangle": ("base", "height"),
    "trapezium": ("a", "b", "height"),
    "cylinder": ("radius", "height"),
    "sphere": ("radius",),
    "cone": ("radius", "height"),
    "cube": ("side",),
    "box": ("length", "width", "height"),
}

def solve_probability(spec):
    """Exact probability as a ratio of two counts over the same tuple space.

    A uniform sample space described by variables and constraints, an event described
    by more constraints, and the answer is a Fraction — never a decimal, because the
    competition answer is m/n and the real-world answer is 'one in seven'. Both counts
    run through the multisearch machine, so pruning, budgets and the expression sandbox
    are inherited rather than rebuilt.
    """
    base = {"solver": "multisearch", "variables": spec["variables"],
            "conditions": list(spec.get("conditions", [])),
            "ordering": spec.get("ordering"), "aggregate": "count"}
    total = solve_multisearch(base)["value"]
    if total == 0:
        raise Refusal("the sample space is empty")
    ev = dict(base)
    ev["conditions"] = list(spec.get("conditions", [])) + list(spec["event"])
    favourable = solve_multisearch(ev)["value"]
    prob = F(favourable, total)
    out = {"value": str(prob), "favourable": favourable, "total": total}
    rep = spec.get("report")
    if rep == "m_plus_n":
        out["m_plus_n"] = prob.numerator + prob.denominator
    elif rep == "percent":
        out["percent"] = str(prob * 100)
    return out

# Written-out unit names, so a spec can be filled by COPYING the problem's words.
# Phase 119 measured the 1B failing every conversion because the exemplar's units are
# never the question's units and copying gets you "mile/second" where the text says
# kilometres per hour. A slot that accepts the words removes the adaptation entirely.
WORD_UNITS = {
    "metre": "m", "meter": "m", "metres": "m", "meters": "m",
    "kilometre": "km", "kilometer": "km", "kilometres": "km", "kilometers": "km",
    "centimetre": "cm", "centimeter": "cm", "centimetres": "cm", "centimeters": "cm",
    "millimetre": "mm", "millimeter": "mm", "millimetres": "mm", "millimeters": "mm",
    "inch": "inch", "inches": "inch", "foot": "foot", "feet": "foot",
    "yard": "yard", "yards": "yard", "mile": "mile", "miles": "mile",
    "second": "second", "seconds": "second", "minute": "minute", "minutes": "minute",
    "hour": "hour", "hours": "hour", "day": "day", "days": "day",
    "week": "week", "weeks": "week",
    "gram": "g", "grams": "g", "gramme": "g", "grammes": "g",
    "kilogram": "kg", "kilograms": "kg", "kilo": "kg", "kilos": "kg",
    "pound": "pound", "pounds": "pound", "ounce": "ounce", "ounces": "ounce",
    "litre": "litre", "liter": "litre", "litres": "litre", "liters": "litre",
    "millilitre": "ml", "milliliter": "ml", "millilitres": "ml", "milliliters": "ml",
    "gallon": "gallon", "gallons": "gallon",
}


def normalise_units(text):
    """Turn 'kilometres per hour' or 'metres per second squared' into the token form."""
    t = str(text).strip().lower()
    t = t.replace(" squared", "^2").replace(" cubed", "^3")
    t = re.sub(r"\s+per\s+", "/", t)
    parts = re.split(r"([/*])", t)
    out = []
    for part in parts:
        if part in ("/", "*"):
            out.append(part)
            continue
        token, _, power = part.strip().partition("^")
        token = token.strip()
        token = WORD_UNITS.get(token, token)
        out.append(token + ("^" + power if power else ""))
    return "".join(out)


def parse_units(text):
    """'km/hour', 'm/second^2', 'kg*m/second' or the written-out words."""
    text = normalise_units(text)
    dims = {}
    num, _, den = str(text).replace(" ", "").partition("/")
    for part, sign in ((num, 1), (den, -1)):
        for factor in filter(None, part.split("*")):
            unit, _, power = factor.partition("^")
            try:
                e = int(power) if power else 1
            except ValueError:
                raise Refusal(f"bad exponent in {factor!r}") from None
            dims[unit] = dims.get(unit, 0) + sign * e
    return {u: e for u, e in dims.items() if e}

def solve_convert(spec):
    """Unit conversion through the phase 73 brick router — sixteen facts, lifted.

    The routing machinery existed for twenty phases as its own experiment; making it a
    library member is what lets a mapped problem reach it. The chain is exact and the
    router refuses when no path exists, which is the honest answer to 'convert kroner
    into kilograms'.
    """
    from bricks import build_registry, route
    src, dst = parse_units(spec["from"]), parse_units(spec["to"])
    res, _explored = route(build_registry(), src, dst)
    if res is None:
        raise Refusal(f"no route from {spec['from']!r} to {spec['to']!r}")
    factor, path = res
    value = F(str(spec.get("value", 1))) * factor
    return {"value": str(value), "factor": str(factor), "steps": len(path)}

def solve_statistics(spec):
    """Mean, median, mode, range and both variances — exact, never rounded.

    A model asked for the variance of eleven numbers produces a plausible decimal; the
    exact answer is a fraction, and the difference is the entire reason to have a
    record. Standard deviation is reported only when it is rational, and named as
    irrational otherwise rather than quietly rounded.
    """
    xs = [F(str(x)) for x in spec["values"]]
    if not xs:
        raise Refusal("no values")
    n = len(xs)
    mean = sum(xs) / n
    srt = sorted(xs)
    median = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2
    pvar = sum((x - mean) ** 2 for x in xs) / n
    svar = sum((x - mean) ** 2 for x in xs) / (n - 1) if n > 1 else None
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    top = max(counts.values())
    modes = sorted(x for x, c in counts.items() if c == top)

    def root(v):
        if v is None:
            return None
        num, den = v.numerator, v.denominator
        if is_square(num) and is_square(den):
            return str(F(math.isqrt(num), math.isqrt(den)))
        return "irrational"

    return {"value": {"mean": str(mean), "median": str(median),
                      "population_variance": str(pvar),
                      "sample_variance": str(svar) if svar is not None else None,
                      "population_sd": root(pvar), "sample_sd": root(svar),
                      "range": str(srt[-1] - srt[0]), "sum": str(sum(xs)),
                      "mode": [str(m) for m in modes], "count": n}}

def solve_datetime(spec):
    """Calendar arithmetic — the classic thing a language model cannot do and a
    record does in one line: days between dates, the weekday of a date, a date shifted
    by days or weeks, leap years."""
    import datetime as _dt
    kind = spec["kind"]

    def parse(key):
        try:
            y, m, d = (int(x) for x in str(spec[key]).split("-"))
            return _dt.date(y, m, d)
        except (ValueError, KeyError) as e:
            raise Refusal(f"bad date in {key!r}: {e}") from None

    if kind == "days_between":
        return {"value": abs((parse("to") - parse("from")).days)}
    if kind == "weekday":
        d = parse("date")
        return {"value": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                          "Saturday", "Sunday"][d.weekday()],
                "iso_weekday": d.isoweekday()}
    if kind == "add_days":
        d = parse("date") + _dt.timedelta(days=int(spec.get("days", 0))
                                          + 7 * int(spec.get("weeks", 0)))
        return {"value": d.isoformat(), "weekday": d.isoweekday()}
    if kind == "leap_years":
        a, b = int(spec["from_year"]), int(spec["to_year"])
        return {"value": sum(1 for y in range(a, b + 1)
                             if y % 4 == 0 and (y % 100 or y % 400 == 0))}
    raise Refusal(f"unknown datetime kind {kind!r}")

def solve_finance(spec):
    """Percentage chains and annuities, exact in Fractions.

    Successive percentage changes do not add, and every model that answers "up 20 then
    down 20 is back where you started" is making the record's case for it.
    """
    kind = spec.get("kind", "percent_chain")
    if kind == "percent_chain":
        v = F(str(spec["start"]))
        for pct in spec["changes"]:
            v = v * (1 + F(str(pct)) / 100)
        return {"value": str(v)}
    if kind == "annuity":
        p, r = F(str(spec["principal"])), F(str(spec["rate"]))
        n = int(spec["periods"])
        if r == 0:
            return {"value": str(p / n)}
        pay = p * r / (1 - (1 + r) ** (-n))
        return {"value": str(pay), "total_paid": str(pay * n)}
    if kind == "vat":
        net = F(str(spec["net"]))
        if "percent" in spec:
            rate = F(str(spec["percent"])) / 100
        else:
            rate = F(str(spec["rate"]))
            if rate > 1:          # 25 can only mean 25 percent, never 2500 percent
                rate = rate / 100
        gross = net * (1 + rate)
        return {"value": str(gross), "tax": str(gross - net)}
    raise Refusal(f"unknown finance kind {kind!r}")

def solve_shape(spec):
    """Standard shapes, exact, with pi kept SYMBOLIC.

    The area of a circle of radius 5 is 25*pi, and 78.54 is a different answer to a
    different question. Results carry the rational coefficient and the power of pi, so
    nothing is ever silently decimalised, and "report" chooses area, perimeter, volume
    or surface.
    """
    name = str(spec.get("shape", "")).lower()
    if name not in SHAPES:
        raise Refusal(f"unknown shape {name!r}")
    g = {k: F(str(spec[k])) for k in SHAPES[name] if k in spec}
    missing = [k for k in SHAPES[name] if k not in g]
    if missing:
        raise Refusal(f"{name} needs {missing}")
    out = {}
    if name == "circle":
        r = g["radius"]
        out = {"area": (r * r, 1), "perimeter": (2 * r, 1)}
    elif name == "rectangle":
        a, b = g["length"], g["width"]
        out = {"area": (a * b, 0), "perimeter": (2 * (a + b), 0)}
    elif name == "triangle":
        out = {"area": (g["base"] * g["height"] / 2, 0)}
    elif name == "trapezium":
        out = {"area": ((g["a"] + g["b"]) * g["height"] / 2, 0)}
    elif name == "cylinder":
        r, h = g["radius"], g["height"]
        out = {"volume": (r * r * h, 1), "surface": (2 * r * (r + h), 1)}
    elif name == "sphere":
        r = g["radius"]
        out = {"volume": (F(4, 3) * r ** 3, 1), "surface": (4 * r * r, 1)}
    elif name == "cone":
        r, h = g["radius"], g["height"]
        out = {"volume": (F(1, 3) * r * r * h, 1)}
    elif name == "cube":
        a = g["side"]
        out = {"volume": (a ** 3, 0), "surface": (6 * a * a, 0)}
    else:
        a, b, c = g["length"], g["width"], g["height"]
        out = {"volume": (a * b * c, 0),
               "surface": (2 * (a * b + b * c + a * c), 0)}
    want = spec.get("report") or next(iter(out))
    if want not in out:
        raise Refusal(f"{name} has no {want!r}; it has {sorted(out)}")
    coef, ppow = out[want]
    text = str(coef) + ("" if ppow == 0 else ("*pi" if ppow == 1 else f"*pi^{ppow}"))
    return {"value": text, "coefficient": str(coef), "pi_power": ppow,
            "available": sorted(out)}

def solve_inclusion(spec):
    """Inclusion-exclusion over named set sizes — the survey word problem, exactly.

    Sizes are given by set name and by intersection name ("a&b"), and the union, the
    exactly-one count and the neither count all follow from the same signed sum.
    """
    sizes = {k.replace(" ", ""): F(str(v)) for k, v in spec["sizes"].items()}
    names = sorted({n for k in sizes for n in k.split("&")})
    if len(names) > 4:
        raise Refusal("inclusion-exclusion supports up to four sets")
    from itertools import combinations as _c
    union = F(0)
    for r in range(1, len(names) + 1):
        for combo in _c(names, r):
            key = "&".join(combo)
            if key not in sizes:
                raise Refusal(f"missing size for {key!r}")
            union += (-1) ** (r + 1) * sizes[key]
    total = F(str(spec["total"])) if "total" in spec else None
    out = {"union": str(union)}
    if total is not None:
        out["neither"] = str(total - union)
    want = spec.get("report", "union")
    if want not in out:
        raise Refusal(f"no {want!r}; available {sorted(out)}")
    return {"value": out[want], **out}

def solve_formula(spec):
    """Phase 91's formula library, addressable at last.

    Twelve named relations with typed slots — speed, momentum, kinetic energy, density,
    area, price and the rest — where the record checks that the given units type-match
    the formula before any number moves. The library existed for fifteen phases and
    could not be REACHED by a mapped problem; the same lesson as the unit router.
    """
    from formgraph import FORMULAS
    name = str(spec.get("name", "")).lower()
    entry = next((f for f in FORMULAS if f[0] == name), None)
    if entry is None:
        raise Refusal(f"unknown formula {name!r}; have "
                      f"{sorted(f[0] for f in FORMULAS)}")
    _n, ins, _out, fn, _w = entry
    args = spec.get("args") or spec.get("values")
    if not isinstance(args, list) or len(args) != len(ins):
        raise Refusal(f"{name} takes {len(ins)} arguments in order")
    return {"value": str(fn(*[F(str(a)) for a in args]))}

def solve_sequence(spec):
    """Identify the rule from the given terms, then extrapolate exactly.

    Arithmetic, geometric, quadratic and two-term linear recurrences are each FITTED and
    then VERIFIED against every given term before a single extrapolation is reported —
    a rule that explains four of five terms is not the rule, and saying so is the
    difference between a solver and a guess.
    """
    xs = [F(str(v)) for v in spec["terms"]]
    n = int(spec["n"])                      # index to report, 0-based
    if len(xs) < 3:
        raise Refusal("need at least three terms")
    if n > 100_000:
        raise Refusal("index too large")

    d = xs[1] - xs[0]
    if all(xs[i + 1] - xs[i] == d for i in range(len(xs) - 1)):
        return {"value": str(xs[0] + n * d), "rule": "arithmetic", "step": str(d)}
    if all(x != 0 for x in xs):
        r = xs[1] / xs[0]
        if all(xs[i + 1] == xs[i] * r for i in range(len(xs) - 1)):
            return {"value": str(xs[0] * r ** n), "rule": "geometric",
                    "ratio": str(r)}
    if len(xs) >= 4:                        # quadratic: constant second difference
        d1 = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        d2 = [d1[i + 1] - d1[i] for i in range(len(d1) - 1)]
        if all(v == d2[0] for v in d2):
            a = d2[0] / 2
            b = d1[0] - a
            c = xs[0]
            return {"value": str(a * n * n + b * n + c), "rule": "quadratic",
                    "coefficients": [str(a), str(b), str(c)]}
    if len(xs) >= 4:                        # a(k) = p a(k-1) + q a(k-2)
        det = xs[1] * xs[1] - xs[0] * xs[2]
        if det != 0:
            p_ = (xs[2] * xs[1] - xs[3] * xs[0]) / det
            q_ = (xs[1] * xs[3] - xs[2] * xs[2]) / det
            if all(xs[i] == p_ * xs[i - 1] + q_ * xs[i - 2]
                   for i in range(2, len(xs))):
                seq = list(xs)
                while len(seq) <= n:
                    seq.append(p_ * seq[-1] + q_ * seq[-2])
                return {"value": str(seq[n]), "rule": "linear recurrence",
                        "coefficients": [str(p_), str(q_)]}
    raise Refusal("no arithmetic, geometric, quadratic or two-term rule fits all terms")

def solve_matrix(spec):
    """Exact small-matrix work in Fractions: determinant, inverse, product, power."""
    def mat(key):
        m = [[F(str(x)) for x in row] for row in spec[key]]
        if any(len(r) != len(m) for r in m):
            raise Refusal(f"{key} is not square")
        if len(m) > 8:
            raise Refusal("matrices up to 8 by 8")
        return m

    def mul(a, b):
        return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
                 for j in range(len(b[0]))] for i in range(len(a))]

    def det(m):
        m = [row[:] for row in m]
        n, out = len(m), F(1)
        for c in range(n):
            piv = next((r for r in range(c, n) if m[r][c] != 0), None)
            if piv is None:
                return F(0)
            if piv != c:
                m[c], m[piv] = m[piv], m[c]
                out = -out
            out *= m[c][c]
            inv = 1 / m[c][c]
            m[c] = [x * inv for x in m[c]]
            for r in range(c + 1, n):
                if m[r][c]:
                    f = m[r][c]
                    m[r] = [x - f * y for x, y in zip(m[r], m[c])]
        return out

    kind = spec.get("kind", "determinant")
    if kind == "determinant":
        return {"value": str(det(mat("matrix")))}
    if kind == "multiply":
        a, b = mat("a"), mat("b")
        return {"value": [[str(x) for x in row] for row in mul(a, b)]}
    if kind == "power":
        a, k = mat("matrix"), int(spec["exponent"])
        if not 0 <= k <= 64:
            raise Refusal("exponent must be between 0 and 64")
        n = len(a)
        out = [[F(int(i == j)) for j in range(n)] for i in range(n)]
        for _ in range(k):
            out = mul(out, a)
        return {"value": [[str(x) for x in row] for row in out]}
    if kind == "inverse":
        a = mat("matrix")
        n = len(a)
        if det(a) == 0:
            raise Refusal("singular matrix has no inverse")
        aug = [a[i][:] + [F(int(i == j)) for j in range(n)] for i in range(n)]
        for c in range(n):
            piv = next(r for r in range(c, n) if aug[r][c] != 0)
            aug[c], aug[piv] = aug[piv], aug[c]
            pv = aug[c][c]
            aug[c] = [x / pv for x in aug[c]]
            for r in range(n):
                if r != c and aug[r][c]:
                    f = aug[r][c]
                    aug[r] = [x - f * y for x, y in zip(aug[r], aug[c])]
        return {"value": [[str(x) for x in row[n:]] for row in aug]}
    raise Refusal(f"unknown matrix kind {kind!r}")

def solve_partition(spec):
    """Counting ways to make a total — the DP a search cannot reach.

    Unordered partitions into parts from a set, ordered compositions, and change-making
    with unlimited or limited copies. multisearch handles up to six bounded variables;
    "how many ways to make 200 from coins" is none of those shapes.
    """
    total = int(spec["total"])
    if total > 100_000:
        raise Refusal("total too large")
    parts = [int(p) for p in spec.get("parts", [])] or list(
        range(1, int(spec.get("max_part", total)) + 1))
    if any(p <= 0 for p in parts):
        raise Refusal("parts must be positive")
    kind = spec.get("kind", "unordered")
    if kind == "unordered":                 # each part usable any number of times
        dp = [0] * (total + 1)
        dp[0] = 1
        for p in parts:
            for v in range(p, total + 1):
                dp[v] += dp[v - p]
        return {"value": dp[total]}
    if kind == "ordered":                   # compositions: order matters
        dp = [0] * (total + 1)
        dp[0] = 1
        for v in range(1, total + 1):
            dp[v] = sum(dp[v - p] for p in parts if p <= v)
        return {"value": dp[total]}
    if kind == "distinct":                  # each part at most once
        dp = [0] * (total + 1)
        dp[0] = 1
        for p in parts:
            for v in range(total, p - 1, -1):
                dp[v] += dp[v - p]
        return {"value": dp[total]}
    raise Refusal(f"unknown partition kind {kind!r}")

def solve_logexp(spec):
    """Integer logarithms, digit counts of huge powers, and exact exponential solves."""
    kind = spec["kind"]
    if kind == "digits_of_power":
        b, e = int(spec["base"]), int(spec["exponent"])
        if b <= 0 or e < 0 or e > 200_000:
            raise Refusal("base must be positive and the exponent at most 200000")
        return {"value": len(str(b ** e))}
    if kind == "integer_log":
        b, x = int(spec["base"]), int(spec["value"])
        if b < 2 or x < 1:
            raise Refusal("base at least 2 and value at least 1")
        k, v = 0, 1
        while v < x:
            v *= b
            k += 1
        if v != x:
            raise Refusal(f"{x} is not an exact power of {b}")
        return {"value": k}
    if kind == "solve_power":               # a^x = b with an exact integer solution
        a, b = int(spec["a"]), int(spec["b"])
        return solve_logexp({"kind": "integer_log", "base": a, "value": b})
    if kind == "trailing_zeros_factorial":
        n, z, p = int(spec["n"]), 0, 5
        while p <= n:
            z += n // p
            p *= 5
        return {"value": z}
    raise Refusal(f"unknown logexp kind {kind!r}")


DIGITS36 = "0123456789abcdefghijklmnopqrstuvwxyz"

def solve_basearith(spec):
    """Arithmetic performed IN a base, and conversions between two bases.

    Adding 4213 and 3654 in base 7 is a mechanical carry chain a model does by
    translating to decimal in its head and back, which is where it slips. The record
    converts exactly, computes exactly and renders exactly.
    """
    def to_int(text, base):
        t = str(text).strip().lower()
        neg = t.startswith("-")
        t = t.lstrip("-")
        if not t or any(c not in DIGITS36[:base] for c in t):
            raise Refusal(f"{text!r} is not a base-{base} numeral")
        v = 0
        for c in t:
            v = v * base + DIGITS36.index(c)
        return -v if neg else v

    def render(v, base):
        if v == 0:
            return "0"
        neg, v, out = v < 0, abs(v), []
        while v:
            out.append(DIGITS36[v % base])
            v //= base
        return ("-" if neg else "") + "".join(reversed(out))

    frm = int(spec.get("from_base", 10))
    to = int(spec.get("to_base", 10))
    if not (2 <= frm <= 36 and 2 <= to <= 36):
        raise Refusal("bases must be between 2 and 36")
    op = spec.get("op", "convert")
    vals = [to_int(v, frm) for v in (spec.get("values") or [spec.get("value")])]
    if op == "convert":
        if len(vals) != 1:
            raise Refusal("convert takes one value")
        result = vals[0]
    elif op == "add":
        result = sum(vals)
    elif op == "subtract":
        result = vals[0] - sum(vals[1:])
    elif op == "multiply":
        result = math.prod(vals)
    else:
        raise Refusal(f"unknown base operation {op!r}")
    return {"value": render(result, to), "decimal": result}

def solve_approx(spec):
    """Continued fractions and the best rational approximation under a denominator cap.

    "The closest fraction to 355/113 with denominator under 50" has one answer and no
    intuition finds it; the Stern-Brocot walk does, exactly, in a few dozen steps.
    """
    kind = spec.get("kind", "best_rational")
    if kind == "continued_fraction":
        x = F(str(spec["value"]))
        terms, limit = [], int(spec.get("terms", 12))
        for _ in range(limit):
            a = x.numerator // x.denominator
            terms.append(a)
            x -= a
            if x == 0:
                break
            x = 1 / x
        return {"value": terms}
    if kind == "best_rational":
        x = F(str(spec["value"]))
        cap = int(spec["max_denominator"])
        if cap < 1:
            raise Refusal("max_denominator must be at least 1")
        best = x.limit_denominator(cap)
        return {"value": str(best), "error": str(abs(best - x))}
    if kind == "convergents":
        x = F(str(spec["value"]))
        out, h1, h2, k1, k2 = [], 1, 0, 0, 1
        for _ in range(int(spec.get("terms", 8))):
            a = x.numerator // x.denominator
            h1, h2 = a * h1 + h2, h1
            k1, k2 = a * k1 + k2, k1
            out.append(f"{h1}/{k1}")
            x -= a
            if x == 0:
                break
            x = 1 / x
        return {"value": out}
    raise Refusal(f"unknown approximation kind {kind!r}")

def solve_strcount(spec):
    """Counting over a written word: letters, distinct arrangements, palindromes.

    Distinct arrangements of MISSISSIPPI is 34650 and a model will confidently say
    something else; it is a factorial over repeated-letter factorials, which is exactly
    the kind of thing to hand to a record.
    """
    word = str(spec.get("word", "")).strip()
    if not word or len(word) > 200:
        raise Refusal("word must be between 1 and 200 characters")
    letters = [c for c in word.lower() if c.isalnum()]
    counts = {}
    for c in letters:
        counts[c] = counts.get(c, 0) + 1
    kind = spec.get("kind", "arrangements")
    if kind == "arrangements":
        total = math.factorial(len(letters))
        for c in counts.values():
            total //= math.factorial(c)
        return {"value": total, "letters": len(letters),
                "repeats": {k: v for k, v in sorted(counts.items()) if v > 1}}
    if kind == "letter_count":
        which = str(spec.get("letter", "")).lower()
        return {"value": counts.get(which, 0)}
    if kind == "distinct_letters":
        return {"value": len(counts)}
    if kind == "is_palindrome":
        return {"value": letters == letters[::-1]}
    raise Refusal(f"unknown string kind {kind!r}")

def solve_primes(spec):
    """Prime counting, the nth prime, the next prime, and sums over a range."""
    kind = spec.get("kind", "count")
    if kind == "nth":
        n = int(spec["n"])
        if not 1 <= n <= 200_000:
            raise Refusal("n between 1 and 200000")
        found, cand = 0, 1
        while found < n:
            cand += 1
            if is_prime(cand):
                found += 1
        return {"value": cand}
    lo = int(spec.get("from", 1))
    hi = int(spec.get("to", 0))
    if kind in ("count", "sum"):
        if hi - lo > 5_000_000:
            raise Refusal("range too wide")
        sieve = bytearray([1]) * (hi + 1)
        sieve[0:2] = b"\x00\x00"
        for i in range(2, int(hi ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
        vals = [i for i in range(max(lo, 2), hi + 1) if sieve[i]]
        return {"value": len(vals) if kind == "count" else sum(vals)}
    if kind == "next":
        v = int(spec["value"]) + 1
        while not is_prime(v):
            v += 1
        return {"value": v}
    raise Refusal(f"unknown prime kind {kind!r}")


SOLVERS3 = {
    "probability": solve_probability,
    "convert": solve_convert,
    "statistics": solve_statistics,
    "datetime": solve_datetime,
    "finance": solve_finance,
    "shape": solve_shape,
    "inclusion_exclusion": solve_inclusion,
    "formula": solve_formula,
    "sequence": solve_sequence,
    "matrix": solve_matrix,
    "partition": solve_partition,
    "logexp": solve_logexp,
    "basearith": solve_basearith,
    "approx": solve_approx,
    "strcount": solve_strcount,
    "primes": solve_primes,
}
