#!/usr/bin/env python3
"""Library growth: expressions, tuples, polynomials, coordinates, modular arithmetic.

Phase 92's search solver has a fixed predicate list and one variable, which is exactly
where competition problems escape it: they ask about PAIRS and TRIPLES related by
arithmetic nobody enumerated in advance. The general move is to let conditions be
EXPRESSIONS — written as ordinary strings, because format is capability (phase 84) and
"a*a + b*b == c*c" is something any model can write while nested predicate JSON is not.

Safety comes from the same discipline as phase 90's checks: the string is parsed with
Python's own parser and walked against a whitelist of node types and function names, so
what executes is arithmetic over bound variables and nothing else. No eval of anything
unvetted, no attributes, no calls outside the list, integers and Fractions only.

Four additions, each self-tested against truths written first:

  multisearch   variables with ranges, expression conditions tested AS SOON AS their
                variables are bound (so triples are feasible), an objective expression,
                aggregates, and the mod-1000 post-op
  polynomial    exact rational roots, Vieta sums and products, evaluation, discriminant
  geometry      exact coordinate work: distance squared, shoelace area, collinearity,
                circle through three points, line intersection
  modular       modpow, inverse, multiplicative order, totient
"""
import ast
import json
import math
import re
import sys
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from solvers import Refusal, digits, factorise, is_prime, is_square  # noqa: E402

TUPLE_BUDGET = 4_000_000

ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
                 ast.Name, ast.Constant, ast.Load, ast.Call, ast.Tuple, ast.List,
                 ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
                 ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
                 ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.IfExp)


def _num_divisors(n):
    return math.prod(e + 1 for e in factorise(abs(int(n))).values()) if n else 0


EXPR_FUNCS = {
    "abs": abs, "min": min, "max": max, "gcd": math.gcd,
    "lcm": lambda *a: math.lcm(*(int(x) for x in a)),
    "isqrt": lambda x: math.isqrt(int(x)),
    "digit_sum": lambda x: sum(digits(int(x))),
    "digit_count": lambda x: len(digits(int(x))),
    "is_square": lambda x: is_square(int(x)),
    "is_prime": lambda x: is_prime(int(x)),
    "num_divisors": _num_divisors,
    "int": int, "len": len, "sum": sum,
    # Added in the ceiling phase, both driven by a specific unreachable problem:
    # a(n) = the least multiple of 23 congruent to 1 mod 2^n needs a modular inverse
    # and a variable power of two, and the literal-exponent rule blocks 2**n on purpose.
    "pow2": lambda k: 2 ** int(k) if 0 <= int(k) <= 4000 else _too_big(),
    "inv_mod": lambda a, m: pow(int(a), -1, int(m)),
}


def _too_big():
    raise Refusal("pow2 argument out of range")


def compile_expr(src, allowed_vars):
    """Parse and vet an expression string; return (code, variables it mentions)."""
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise Refusal(f"expression syntax: {e.msg}") from None
    used = set()
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise Refusal(f"expression uses {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in EXPR_FUNCS:
                raise Refusal("expression calls something outside the whitelist")
        if isinstance(node, ast.Name) and node.id not in EXPR_FUNCS:
            if node.id not in allowed_vars:
                raise Refusal(f"unknown variable {node.id!r}")
            used.add(node.id)
        if isinstance(node, ast.Pow):
            pass                      # guarded at evaluation by the exponent check
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if abs(node.value) > 10 ** 18:
                raise Refusal("constant too large")
    for node in ast.walk(tree):       # exponents must be small literals
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            e = node.right
            if not (isinstance(e, ast.Constant) and isinstance(e.value, int)
                    and abs(e.value) <= 64):
                raise Refusal("exponent must be a literal at most 64")
    return compile(tree, "<expr>", "eval"), used


def eval_expr(code, env):
    return eval(code, {"__builtins__": {}, **EXPR_FUNCS}, env)  # noqa: S307 - vetted AST


def is_linear_in(node, var):
    """Is this expression structurally degree-1 in var (or free of it)?

    The probe version of this test was UNSOUND and the self-tests caught it: sampling
    digit_sum(n) - 10 at n = 0, 1, 2 gives -10, -9, -8, which is a perfect straight
    line, so the solver skipped the scan and lost every hit. Three samples cannot
    certify linearity; the syntax can. A call containing the variable is never linear,
    a power of it never is, and a product is linear only when one side is free of it.
    """
    def free(n):
        return all(not (isinstance(x, ast.Name) and x.id == var) for x in ast.walk(n))

    if free(node):
        return True
    if isinstance(node, ast.Name):
        return node.id == var
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return is_linear_in(node.operand, var)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return is_linear_in(node.left, var) and is_linear_in(node.right, var)
        if isinstance(node.op, ast.Mult):
            return ((free(node.left) and is_linear_in(node.right, var))
                    or (free(node.right) and is_linear_in(node.left, var)))
        if isinstance(node.op, ast.Div):
            return free(node.right) and is_linear_in(node.left, var)
        return False                       # FloorDiv, Mod, Pow: not linear
    return False                           # calls, comparisons, anything else


def equality_residual(src, allowed_vars):
    """If the condition is a top-level equality, compile LHS - RHS as its residual.

    A residual that is AFFINE in the innermost variable can be SOLVED instead of
    scanned, which is phase 75's probe trick moved inside the search: two probes fix
    the line, a third refuses impostors, and a 10^8 scan becomes one candidate. The
    model's own formulation of an AIME problem died on the budget where this rescues
    it, so the growth is evidence-driven and the fallback (scan) is always available.
    """
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError:
        return None
    node = tree.body
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq)):
        return None
    diff = ast.BinOp(left=node.left, op=ast.Sub(), right=node.comparators[0])
    expr = ast.Expression(body=diff)
    ast.fix_missing_locations(expr)
    try:
        code, used = compile_expr(ast.unparse(expr), allowed_vars)
    except Refusal:
        return None
    linear = {v for v in used if is_linear_in(diff, v)}
    return code, used, linear


def solve_multisearch(spec):
    """Tuples of integers under expression constraints, pruned as variables bind."""
    vars_ = spec.get("variables")
    if not vars_ or len(vars_) > 4:
        raise Refusal("multisearch needs 1 to 4 variables")
    names = [v["name"] for v in vars_]
    if len(set(names)) != len(names):
        raise Refusal("duplicate variable names")
    ranges = []
    space = 1
    for v in vars_:
        lo, hi = int(v["from"]), int(v["to"])
        if hi < lo:
            raise Refusal(f"empty range for {v['name']}")
        ranges.append((lo, hi))
        space *= hi - lo + 1

    conds = []
    for c in spec.get("conditions", []):
        code, used = compile_expr(c, set(names))
        res = equality_residual(c, set(names))
        conds.append((code, used, c, res))
    ordering = spec.get("ordering")      # "increasing" | "strict_increasing" | None
    obj_src = spec.get("objective")
    obj = compile_expr(obj_src, set(names))[0] if obj_src else None

    # The budget is checked against what will actually be WALKED, which the affine
    # solve can shrink by the size of the innermost range.
    solved_dim = 1
    if len(names) > 1 and any(r and names[-1] in r[2] for *_x, r in conds):
        solved_dim = ranges[-1][1] - ranges[-1][0] + 1
    if space // max(solved_dim, 1) > TUPLE_BUDGET:
        raise Refusal(f"tuple space {space} exceeds the budget {TUPLE_BUDGET}")

    hits, work, solved = [], 0, 0

    def nonlocal_solved():
        nonlocal solved
        solved += 1

    def rec(i, env):
        nonlocal work
        if i == len(names):
            work += 1
            hits.append(eval_expr(obj, env) if obj else tuple(env[n] for n in names))
            return
        lo, hi = ranges[i]
        if ordering and i > 0:
            prev = env[names[i - 1]]
            lo = max(lo, prev + (1 if ordering == "strict_increasing" else 0))
        bound = set(names[:i + 1])
        ready = [(code, src) for code, used, src, _r in conds
                 if used <= bound and not used <= set(names[:i])]

        # Innermost variable with an affine equality: solve, do not scan.
        if i == len(names) - 1:
            for code, used, src, resid in conds:
                # Only a STRUCTURALLY linear residual may be solved instead of scanned.
                if not resid or names[i] not in resid[2] or not used <= bound:
                    continue
                rcode = resid[0]
                try:
                    probe = []
                    for v in (0, 1, 2):
                        env[names[i]] = v
                        probe.append(F(eval_expr(rcode, env)))
                except (ZeroDivisionError, TypeError, ValueError):
                    continue
                slope = probe[1] - probe[0]
                if slope == 0 or probe[2] - probe[1] != slope:
                    continue                       # not affine: the third probe says so
                root = -probe[0] / slope
                nonlocal_solved()
                cand = int(root) if root.denominator == 1 else None
                env.pop(names[i], None)
                if cand is None or not lo <= cand <= hi:
                    return
                env[names[i]] = cand
                try:
                    if all(eval_expr(c2, env) for c2, _s in ready):
                        rec(i + 1, env)
                except ZeroDivisionError:
                    pass
                env.pop(names[i], None)
                return

        for val in range(lo, hi + 1):
            work += 1
            if work > TUPLE_BUDGET:
                raise Refusal("multisearch budget exhausted")
            env[names[i]] = val
            try:
                if all(eval_expr(code, env) for code, _ in ready):
                    rec(i + 1, env)
            except ZeroDivisionError:
                continue
        env.pop(names[i], None)

    rec(0, {})
    agg = spec.get("aggregate", "count")
    if agg in ("eval", "only", "value"):
        if len(hits) != 1:
            raise Refusal(f"aggregate {agg!r} needs exactly one hit, found {len(hits)}")
        agg = "min"
    if agg == "count":
        value = len(hits)
    elif agg == "sum":
        value = sum(hits) if obj else sum(sum(t) for t in hits)
    elif agg == "min":
        value = min(hits) if hits else None
    elif agg == "max":
        value = max(hits) if hits else None
    elif agg == "list":
        value = [list(h) if isinstance(h, tuple) else h for h in hits[:200]]
    else:
        raise Refusal(f"unknown aggregate {agg!r}")
    post = spec.get("post")
    if post and value is not None:
        if post["op"] == "mod":
            value = value % int(post["arg"])
        elif post["op"] == "digit_sum":
            value = sum(digits(int(value)))
        elif post["op"] == "exponent_sum":
            # AIME asks this constantly: express the answer as a product of prime
            # powers and add the exponents. Without it phase 92's 13! problem needs
            # two specs chained; with it, one.
            value = sum(factorise(int(value)).values())
        else:
            raise Refusal(f"unknown post-op {post['op']!r}")
    return {"value": value, "hits": len(hits), "work": work, "solved": solved}


def solve_polynomial(spec):
    """Exact rational roots (rational root theorem, verified), Vieta, evaluation."""
    coeffs = [F(str(c)) for c in spec["coefficients"]]      # highest degree first
    while coeffs and coeffs[0] == 0:
        coeffs.pop(0)
    if len(coeffs) < 2:
        raise Refusal("not a polynomial in x")
    at = spec.get("at")
    if at is not None:
        x = F(str(at))
        val = sum(c * x ** (len(coeffs) - 1 - i) for i, c in enumerate(coeffs))
        return {"value": str(val)}
    den = math.lcm(*[c.denominator for c in coeffs])
    ints = [int(c * den) for c in coeffs]
    a0, an = ints[-1], ints[0]
    roots = []
    if a0 == 0:
        roots.append(F(0))
        while ints and ints[-1] == 0:
            ints.pop()
        a0 = ints[-1] if ints else 0
    if a0:
        ps = [d for d in range(1, min(abs(a0), 10 ** 6) + 1) if a0 % d == 0]
        qs = [d for d in range(1, min(abs(an), 10 ** 6) + 1) if an % d == 0]
        for p in ps:
            for q in qs:
                for r in (F(p, q), F(-p, q)):
                    if r in roots:
                        continue
                    if sum(F(c) * r ** (len(ints) - 1 - i)
                           for i, c in enumerate(ints)) == 0:
                        roots.append(r)
    n = len(coeffs) - 1
    vieta_sum = -coeffs[1] / coeffs[0]
    vieta_prod = (-1) ** n * coeffs[-1] / coeffs[0]
    return {"value": [str(r) for r in sorted(roots)], "degree": n,
            "sum_of_roots": str(vieta_sum), "product_of_roots": str(vieta_prod)}


def solve_geometry(spec):
    """Exact coordinate geometry: squared distances, shoelace area, circles, lines."""
    kind = spec["kind"]
    pts = [tuple(F(str(c)) for c in p) for p in spec.get("points", [])]
    if kind == "distance_squared":
        (x1, y1), (x2, y2) = pts
        return {"value": str((x2 - x1) ** 2 + (y2 - y1) ** 2)}
    if kind == "polygon_area":
        if len(pts) < 3:
            raise Refusal("a polygon needs at least three points")
        s = sum(pts[i][0] * pts[(i + 1) % len(pts)][1]
                - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))
        return {"value": str(abs(s) / 2)}
    if kind == "collinear":
        (x1, y1), (x2, y2), (x3, y3) = pts
        return {"value": (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)}
    if kind == "circle_through":
        (x1, y1), (x2, y2), (x3, y3) = pts
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if d == 0:
            raise Refusal("the three points are collinear")
        s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
        ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
        uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
        return {"value": {"center": [str(ux), str(uy)],
                          "radius_squared": str((x1 - ux) ** 2 + (y1 - uy) ** 2)}}
    if kind == "line_intersection":
        a1, b1, c1 = (F(str(v)) for v in spec["line1"])   # a x + b y = c
        a2, b2, c2 = (F(str(v)) for v in spec["line2"])
        det = a1 * b2 - a2 * b1
        if det == 0:
            raise Refusal("lines are parallel or identical")
        return {"value": [str((c1 * b2 - c2 * b1) / det),
                          str((a1 * c2 - a2 * c1) / det)]}
    if kind == "triangle_sides":       # side lengths squared -> area (Heron, exact^2)
        a2, b2, c2 = (F(str(v)) ** 2 for v in spec["sides"])
        sixteen = 4 * a2 * b2 - (a2 + b2 - c2) ** 2
        if sixteen <= 0:
            raise Refusal("not a valid triangle")
        return {"value_squared": str(sixteen / 16),
                "value": str(F(math.isqrt(int(sixteen / 16 * 10 ** 12)), 10 ** 6))
                if (sixteen / 16).denominator == 1 else "irrational"}
    raise Refusal(f"unknown geometry kind {kind!r}")


def solve_modular(spec):
    kind = spec["kind"]
    if kind == "power":
        return {"value": pow(int(spec["base"]), int(spec["exponent"]),
                             int(spec["modulus"]))}
    if kind == "inverse":
        a, m = int(spec["a"]), int(spec["modulus"])
        if math.gcd(a, m) != 1:
            raise Refusal("no inverse: not coprime")
        return {"value": pow(a, -1, m)}
    if kind == "order":
        a, m = int(spec["a"]), int(spec["modulus"])
        if math.gcd(a, m) != 1:
            raise Refusal("order undefined: not coprime")
        x, k = a % m, 1
        while x != 1:
            x = x * a % m
            k += 1
            if k > 10 ** 7:
                raise Refusal("order search exhausted")
        return {"value": k}
    if kind == "totient":
        n = int(spec["n"])
        t = n
        for p in factorise(n):
            t = t // p * (p - 1)
        return {"value": t}
    raise Refusal(f"unknown modular kind {kind!r}")


def solve_arith(spec):
    """Named arithmetic steps over the problem's numbers — phase 45's plan graph, now a
    library member with the expression sandbox around it.

    Word problems are mostly this: a handful of quantities combined in a few steps. The
    model writes the steps, never their results; the record evaluates each one exactly
    in Fractions, in order, with every earlier step available as a variable, and the
    same AST whitelist that guards multisearch guards these.
    """
    lets = spec.get("let", {})
    if not isinstance(lets, dict):
        raise Refusal("let must be an object of name -> expression")
    if len(lets) > 30:
        raise Refusal("too many steps")
    env, order = {}, []
    for name, src in lets.items():
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", str(name)):
            raise Refusal(f"bad step name {name!r}")
        code, _ = compile_expr(str(src), set(env))     # only EARLIER steps may be used
        try:
            env[name] = F(eval_expr(code, dict(env)))
        except ZeroDivisionError:
            raise Refusal(f"step {name} divides by zero") from None
        order.append(name)
    ans_src = spec.get("answer")
    if ans_src is None:
        if not order:
            raise Refusal("no steps and no answer expression")
        value = env[order[-1]]
    else:
        code, _ = compile_expr(str(ans_src), set(env))
        value = F(eval_expr(code, dict(env)))
    return {"value": str(value), "steps": {k: str(v) for k, v in env.items()}}


SOLVERS2 = {
    "arith": solve_arith,
    "multisearch": solve_multisearch,
    "polynomial": solve_polynomial,
    "geometry": solve_geometry,
    "modular": solve_modular,
}

WORDINGS2 = {
    "arith": ["work out the total step by step", "combine the given amounts",
              "how much is left after the costs"],
    "multisearch": ["how many pairs of integers satisfy the equation",
                    "count the ordered triples with a given property",
                    "find all pairs a and b with a condition relating them"],
    "polynomial": ["the roots of the polynomial", "the sum of the roots of the cubic",
                   "evaluate the polynomial at a value"],
    "geometry": ["the area of the triangle with these vertices",
                 "the distance between two points",
                 "the circle through three given points"],
    "modular": ["the remainder when the power is divided",
                "the multiplicative order modulo n",
                "the inverse of a modulo m"],
}


# Which slots identify which machine. Phase 91 measured that a typed shape should
# dispose over a lexical choice; a spec is the same situation one level up, and the
# 1B demonstrated it immediately by writing a perfect arith body under the name
# "search". Shape dispatch is tried ONLY after the named solver fails, and every
# repair is counted rather than performed silently.
SHAPE_KEYS = [
    ({"let"}, "arith"), ({"answer"}, "arith"),
    ({"variables"}, "multisearch"),
    ({"residues"}, "crt"), ({"moduli"}, "crt"),
    ({"rows"}, "linear_system"), ({"rhs"}, "linear_system"),
    ({"coefficients", "initial"}, "recurrence"),
    ({"coefficients"}, "polynomial"),
    ({"times"}, "rate_work"),
    ({"total", "ratio"}, "ratio_split"),
    ({"quantities", "concentrations"}, "mixture"),
    ({"principal"}, "interest"),
    ({"base"}, "base_convert"),
    ({"points"}, "geometry"), ({"line1"}, "geometry"),
    ({"domain"}, "search"),
    ({"values"}, "gcd_lcm"),
    ({"modulus"}, "modular"),
    ({"a", "b", "c"}, "quadratic"),
]
REPAIRS = {"count": 0, "by": []}


def dispatch_by_shape(spec):
    keys = set(spec) - {"solver"}
    for need, name in SHAPE_KEYS:
        if need <= keys and name != spec.get("solver"):
            return name
    return None


def _call(name, spec):
    from solvers import SOLVERS as SOLVERS1
    fn = SOLVERS2.get(name) or SOLVERS1.get(name)
    if fn is None:
        return None, f"unknown solver {name!r}"
    try:
        return fn(spec), "ok"
    except Refusal as e:
        return None, f"refused: {e}"
    except RecursionError:
        return None, "refused: recursion too deep"
    except Exception as e:  # noqa: BLE001
        return None, f"error: {type(e).__name__}: {str(e)[:70]}"


def run2(spec, repair=True):
    """Dispatch across BOTH libraries; if the NAME fails, let the SHAPE speak."""
    res, why = _call(spec.get("solver"), spec)
    if res is not None or not repair:
        return res, why
    alt = dispatch_by_shape(spec)
    if alt is None:
        return res, why
    res2, why2 = _call(alt, spec)
    if res2 is None:
        return res, why
    REPAIRS["count"] += 1
    REPAIRS["by"].append((spec.get("solver"), alt))
    return res2, f"ok (shape repaired {spec.get('solver')!r} -> {alt!r})"


# ---------------------------------------------------------------- self-tests
# Truths written before the code path that produces them; the non-obvious ones are
# derived by hand in comments.
CASES = [
    # Pythagorean triples with a < b < c <= 20: (3,4,5) (5,12,13) (6,8,10) (8,15,17)
    # (9,12,15) (12,16,20) — six.
    ({"solver": "multisearch",
      "variables": [{"name": "a", "from": 1, "to": 20}, {"name": "b", "from": 1,
                     "to": 20}, {"name": "c", "from": 1, "to": 20}],
      "ordering": "strict_increasing",
      "conditions": ["a*a + b*b == c*c"], "aggregate": "count"}, 6),
    # Ordered pairs with a + b = 20, gcd(a,b) = 1, a,b >= 1: phi-like count = 8
    # (1,19)(3,17)(7,13)(9,11)(11,9)(13,7)(17,3)(19,1).
    ({"solver": "multisearch",
      "variables": [{"name": "a", "from": 1, "to": 19}, {"name": "b", "from": 1,
                     "to": 19}],
      "conditions": ["a + b == 20", "gcd(a, b) == 1"], "aggregate": "count"}, 8),
    # Sum of a*b over pairs with a+b = 10, 1 <= a < b: 1*9+2*8+3*7+4*6 = 9+16+21+24 = 70
    ({"solver": "multisearch",
      "variables": [{"name": "a", "from": 1, "to": 9}, {"name": "b", "from": 1,
                     "to": 9}],
      "ordering": "strict_increasing", "conditions": ["a + b == 10"],
      "objective": "a*b", "aggregate": "sum"}, 70),
    ({"solver": "multisearch",
      "variables": [{"name": "n", "from": 1, "to": 1000}],
      "conditions": ["digit_sum(n) == 10", "n % 7 == 0"], "aggregate": "count"}, 10),
    ({"solver": "polynomial", "coefficients": [1, -6, 11, -6]},
     ["1", "2", "3"]),
    ({"solver": "polynomial", "coefficients": [2, -3, 1], "at": 5}, "36"),
    ({"solver": "geometry", "kind": "polygon_area",
      "points": [[0, 0], [4, 0], [4, 3], [0, 3]]}, "12"),
    ({"solver": "geometry", "kind": "distance_squared",
      "points": [[1, 2], [4, 6]]}, "25"),
    ({"solver": "geometry", "kind": "circle_through",
      "points": [[0, 0], [2, 0], [0, 2]]}, None),      # centre (1,1), r^2 = 2
    ({"solver": "geometry", "kind": "line_intersection", "line1": [1, 1, 10],
      "line2": [3, -1, 14]}, ["6", "4"]),
    ({"solver": "modular", "kind": "power", "base": 7, "exponent": 100,
      "modulus": 13}, 9),
    ({"solver": "modular", "kind": "order", "a": 2, "modulus": 7}, 3),
    ({"solver": "modular", "kind": "totient", "n": 36}, 12),
    ({"solver": "modular", "kind": "inverse", "a": 3, "modulus": 11}, 4),
    # arith: 3 crates of 12 minus 5 broken, split between 2 shops -> (36-5)/2 = 31/2
    ({"solver": "arith", "let": {"total": "3*12", "left": "total - 5"},
      "answer": "left / 2"}, "31/2"),
    ({"solver": "arith", "let": {"a": "15 + 27", "b": "a * 2", "c": "b - 9"}},
     "75"),
    # The 1B's actual failure mode: a perfect arith body under the name "search".
    ({"solver": "search", "let": {"a": "5 + (2 * 5)", "b": "8 / 2"},
      "answer": "a + b"}, "19"),
    # And the old library must still answer through the joint dispatcher.
    ({"solver": "crt", "residues": [2, 3, 2], "moduli": [3, 5, 7]}, 23),
]

REFUSALS = [
    # Nonlinear on purpose: with a linear condition the affine solve now makes this
    # space feasible, and the refusal correctly stops firing. The budget guard is for
    # what must still be WALKED.
    ({"solver": "multisearch",
      "variables": [{"name": "a", "from": 1, "to": 3000}, {"name": "b", "from": 1,
                     "to": 3000}], "conditions": ["a*a == b*b*b"]}, "budget"),
    ({"solver": "multisearch", "variables": [{"name": "a", "from": 1, "to": 5}],
      "conditions": ["__import__('os').system('ls')"]}, "outside the whitelist"),
    ({"solver": "multisearch", "variables": [{"name": "a", "from": 1, "to": 5}],
      "conditions": ["a.real == 1"]}, "Attribute"),
    ({"solver": "multisearch", "variables": [{"name": "a", "from": 1, "to": 5}],
      "conditions": ["b == 1"]}, "unknown variable"),
    ({"solver": "multisearch", "variables": [{"name": "a", "from": 1, "to": 5}],
      "conditions": ["a ** a == 4"]}, "exponent must be a literal"),
    ({"solver": "geometry", "kind": "circle_through",
      "points": [[0, 0], [1, 1], [2, 2]]}, "collinear"),
    ({"solver": "arith", "let": {"a": "1", "b": "c + 1"}}, "unknown variable"),
    ({"solver": "arith", "let": {"a": "5 / 0"}}, "divides by zero"),
]


def main(out="data/custom/solvers2.json"):
    passed = failed = 0
    rows = []
    for spec, truth in CASES:
        res, why = run2(spec)
        got = res["value"] if res else None
        if spec.get("kind") == "circle_through":
            ok = res is not None and got["center"] == ["1", "1"] and \
                got["radius_squared"] == "2"
            got = got and got["center"]
        else:
            ok = got == truth
        passed += ok
        failed += not ok
        rows.append({"solver": spec["solver"], "ok": bool(ok), "got": str(got)[:50],
                     "truth": str(truth)[:50], "why": why})
        print(f"{spec['solver']:<13} {'ok  ' if ok else 'FAIL'} {str(got)[:40]:<42}"
              f"{'' if ok else 'want ' + str(truth)[:28]}")

    ref_ok = 0
    for spec, needle in REFUSALS:
        res, why = run2(spec)
        good = res is None and needle in why
        ref_ok += good
        print(f"{spec['solver']:<13} {'refused' if good else 'MISSED '} [{why[:56]}]")

    total_solvers = len(SOLVERS2) + 15
    print(f"\nshape repaired a mis-named spec {REPAIRS['count']} times: "
          f"{REPAIRS['by']}")
    print(f"\n{passed}/{len(CASES)} cases exact, {ref_ok}/{len(REFUSALS)} refusals "
          f"named (three of them attempted escapes from the expression sandbox)")
    print(f"library now {total_solvers} solvers; expressions admit "
          f"{len(EXPR_FUNCS)} functions over up to 4 bound variables")
    print("\nConditions as EXPRESSIONS is the growth that matters: a fixed predicate")
    print("list can only meet problems someone foresaw, while a vetted arithmetic")
    print("string meets the ones nobody did — and the same AST discipline that made")
    print("model-written checks safe makes model-written constraints safe.")
    summary = {"shape_repairs": REPAIRS["count"],
               "cases": len(CASES), "passed": passed, "failed": failed,
               "refusals": len(REFUSALS), "refusals_named": ref_ok,
               "solvers_total": total_solvers, "expr_funcs": len(EXPR_FUNCS),
               "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
