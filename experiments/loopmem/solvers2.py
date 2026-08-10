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


class _Rationalise(ast.NodeTransformer):
    """Wrap integer literals as Fractions so `/` is exact.

    The hard-arithmetic battery caught this as a correctness bug, not a preference:
    "5/11 * 5/11 * 5/11" evaluated in floats and the record returned
    2774945224945457/225179981 — the exact binary expansion of a rounded double —
    where the truth is 125/1331. Only applied when the expression actually divides,
    so integer searches keep integer speed.
    """

    def visit_Constant(self, node):  # noqa: N802 - ast API
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return ast.Call(func=ast.Name(id="Frac", ctx=ast.Load()),
                            args=[node], keywords=[])
        return node


def compile_expr(src, allowed_vars):
    """Parse and vet an expression string; return (code, variables it mentions)."""
    # A spec language for mathematics: '^' is a power, not a bitwise xor. The model
    # wrote (-1)^1 and Python read BitXor; accommodating the notation costs nothing
    # because xor is not in the whitelist anyway.
    src = re.sub(r"\^", "**", str(src))
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
            if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub):
                e = e.operand
            # A variable exponent is safe when the base cannot grow: (-1)^k is the
            # alternating sign every second competition problem needs, and 1^k and
            # 0^k are bounded for the same reason. Anything else stays literal.
            b = node.left
            if isinstance(b, ast.UnaryOp) and isinstance(b.op, ast.USub):
                b = b.operand
            bounded_base = (isinstance(b, ast.Constant)
                            and isinstance(b.value, int) and abs(b.value) <= 1)
            if not bounded_base and not (isinstance(e, ast.Constant)
                                         and isinstance(e.value, int)
                                         and abs(e.value) <= 1024):
                raise Refusal("exponent must be a literal at most 1024")
    if any(isinstance(n, ast.Div) for n in ast.walk(tree)):
        tree = ast.fix_missing_locations(_Rationalise().visit(tree))
    return compile(tree, "<expr>", "eval"), used


def eval_expr(code, env):
    return eval(code, {"__builtins__": {}, "Frac": F, **EXPR_FUNCS},  # noqa: S307
                env)


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
    if not vars_ or len(vars_) > 6:
        raise Refusal("multisearch needs 1 to 6 variables")
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
    elif agg == "product":
        value = F(1)
        for h in hits:
            value *= F(h) if not isinstance(h, tuple) else F(math.prod(h))
        value = str(value)
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


def solve_iterate(spec):
    """acc = init; for k = from..to: acc = step(acc, k) — exact, budgeted.

    Five of the hard-arithmetic problems are folds ("nine times in a row, add 2/5 then
    multiply by 3/4"), which arith cannot express (no loop) and multisearch cannot
    either (a map-reduce is not a fold). The battery named the missing machine.
    """
    lo, hi = int(spec.get("from", 1)), int(spec.get("to", 1))
    if hi - lo + 1 > 200_000:
        raise Refusal("too many iterations")
    init_code, _ = compile_expr(str(spec.get("init", "0")), set())
    step_code, _ = compile_expr(str(spec["step"]), {"acc", "k"})
    acc = F(eval_expr(init_code, {}))
    for k in range(lo, hi + 1):
        acc = F(eval_expr(step_code, {"acc": acc, "k": k}))
    fin = spec.get("final")
    if fin:
        code, _ = compile_expr(str(fin), {"acc"})
        acc = F(eval_expr(code, {"acc": acc}))
    return {"value": str(acc), "iterations": hi - lo + 1}


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


def parse_units(text):
    """'km/hour', 'm/second^2', 'kg*m/second' -> the exponent map the router wants."""
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


SOLVERS2 = {
    "arith": solve_arith,
    "basearith": solve_basearith,
    "approx": solve_approx,
    "strcount": solve_strcount,
    "primes": solve_primes,
    "sequence": solve_sequence,
    "matrix": solve_matrix,
    "partition": solve_partition,
    "logexp": solve_logexp,
    "shape": solve_shape,
    "inclusion_exclusion": solve_inclusion,
    "formula": solve_formula,
    "statistics": solve_statistics,
    "datetime": solve_datetime,
    "finance": solve_finance,
    "probability": solve_probability,
    "convert": solve_convert,
    "iterate": solve_iterate,
    "multisearch": solve_multisearch,
    "polynomial": solve_polynomial,
    "geometry": solve_geometry,
    "modular": solve_modular,
}

WORDINGS2 = {
    "basearith": ["add the two numbers in base seven", "convert between bases",
                  "arithmetic written in another base"],
    "approx": ["the closest fraction with a small denominator",
               "the continued fraction expansion", "approximate the ratio"],
    "strcount": ["how many distinct arrangements of the letters",
                 "how many times does the letter appear", "anagrams of the word"],
    "primes": ["how many primes are there below", "the nth prime number",
               "the next prime after"],
    "sequence": ["what is the next term in the sequence",
                 "the nth term of this pattern", "continue the series"],
    "matrix": ["the determinant of the matrix", "multiply the two matrices",
               "the inverse of the matrix"],
    "partition": ["how many ways to make the total from these parts",
                  "in how many ways can the amount be paid",
                  "count the compositions of the number"],
    "logexp": ["how many digits does the power have",
               "how many trailing zeros in the factorial",
               "what power of the base gives this value"],
    "shape": ["the area of the circle", "the volume of the cylinder",
              "how much surface does the box have"],
    "inclusion_exclusion": ["how many take at least one of the subjects",
                            "how many like neither", "the union of the groups"],
    "formula": ["apply the standard formula", "the momentum from mass and velocity",
                "the density from mass and volume"],
    "statistics": ["the mean and the variance of these numbers",
                   "the median of the list", "how spread out the values are"],
    "datetime": ["how many days between the two dates",
                 "what weekday does it fall on", "the date some weeks later"],
    "finance": ["the price after successive percentage changes",
                "the monthly payment on a loan", "the amount including tax"],
    "probability": ["what is the probability that", "the chance of the event",
                    "how likely is it, as a fraction"],
    "convert": ["convert one unit into another", "how fast is that in other units",
                "express the length in a different measure"],
    "iterate": ["repeat the same operation many times over",
                "each round changes the amount by the same rule",
                "compound the value step after step"],
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


REPORT_HOME = {"gcd": "gcd_lcm", "lcm": "gcd_lcm",
               "divisor_count": "factor", "divisor_sum": "factor",
               "exponent_sum": "factor", "factorisation": "factor",
               "digit_sum": "digit_ops", "digit_count": "digit_ops",
               "reversed": "digit_ops", "is_palindrome": "digit_ops"}


def dispatch_by_shape(spec):
    want = spec.get("report") or spec.get("op")
    if want in REPORT_HOME and REPORT_HOME[want] != spec.get("solver"):
        return REPORT_HOME[want]
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
    """Dispatch across BOTH libraries; if the NAME fails, let the SHAPE speak.

    A named solver that RUNS can still be the wrong machine: asking digit_ops for a
    divisor sum succeeds and answers a different question. So a spec that names a
    report its result does not contain counts as a failure and gets the same repair.
    """
    res, why = _call(spec.get("solver"), spec)
    want = spec.get("report") or spec.get("op")
    if res is not None and want and isinstance(res.get("value"), dict) \
            and want not in res["value"] and repair:
        alt = dispatch_by_shape(spec)
        if alt:
            res2, why2 = _call(alt, spec)
            if res2 is not None and isinstance(res2.get("value"), dict) \
                    and want in res2["value"]:
                REPAIRS["count"] += 1
                REPAIRS["by"].append((spec.get("solver"), alt))
                return res2, f"ok (report repaired -> {alt!r})"
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
    # iterate: 3/7 then nine rounds of (+2/5)*3/4 — a fold, and exact in Fractions.
    ({"solver": "iterate", "init": "3/7", "step": "(acc + 2/5) * 3/4",
      "from": 1, "to": 9}, "10478607/9175040"),
    # alternating signs: a variable exponent on a bounded base
    ({"solver": "iterate", "init": "1000", "step": "acc * (1 + (-1)^k / (k+2))",
      "from": 1, "to": 12}, "5000/7"),
    ({"solver": "multisearch", "variables": [{"name": "k", "from": 2, "to": 15}],
      "objective": "k*k/(k*k-1)", "aggregate": "product"}, "15/8"),
    # exact rationals: the float bug the battery caught
    ({"solver": "arith", "let": {"a": "5/11 * 5/11 * 5/11", "b": "a * 121/25"},
      "answer": "b + 7/9"}, "122/99"),
    # '^' means power in a mathematics spec language
    ({"solver": "arith", "let": {"a": "(-1)^3 * 8"}, "answer": "a + 10"}, "2"),
    # 2^200 - 3^100: a large literal exponent is bounded, so the guard lets it pass.
    ({"solver": "arith", "let": {"a": "2^200 - 3^100"}}, str(2 ** 200 - 3 ** 100)),
    # probability: two dice, exact chance the sum is 9 -> 4/36 = 1/9
    ({"solver": "probability",
      "variables": [{"name": "a", "from": 1, "to": 6}, {"name": "b", "from": 1,
                     "to": 6}], "event": ["a + b == 9"]}, "1/9"),
    # and the m+n convention competitions ask for
    ({"solver": "probability",
      "variables": [{"name": "a", "from": 1, "to": 6}, {"name": "b", "from": 1,
                     "to": 6}], "event": ["a + b == 9"], "report": "m_plus_n"}, "1/9"),
    # convert: the founding example, exact
    ({"solver": "convert", "value": 3, "from": "mile/second", "to": "km/hour"},
     "10863072/625"),
    ({"solver": "convert", "value": 100, "from": "km/hour", "to": "foot/second"},
     "312500/3429"),
    # statistics: population variance of 2,4,4,4,5,5,7,9 is exactly 4
    ({"solver": "statistics", "values": [2, 4, 4, 4, 5, 5, 7, 9],
      "report": "population_variance"}, None),
    # datetime: 1999-12-31 to 2000-03-01 is 61 days (2000 is a leap year)
    ({"solver": "datetime", "kind": "days_between", "from": "1999-12-31",
      "to": "2000-03-01"}, 61),
    ({"solver": "datetime", "kind": "weekday", "date": "2026-08-10"}, "Monday"),
    ({"solver": "datetime", "kind": "leap_years", "from_year": 1896,
      "to_year": 1910}, 3),
    # finance: up 20% then down 20% is 96, not 100
    ({"solver": "finance", "kind": "percent_chain", "start": 100,
      "changes": [20, -20]}, "96"),
    # shape: a circle of radius 5 has area 25*pi, not 78.54
    ({"solver": "shape", "shape": "circle", "radius": 5, "report": "area"}, "25*pi"),
    ({"solver": "shape", "shape": "sphere", "radius": 3, "report": "volume"},
     "36*pi"),
    ({"solver": "shape", "shape": "box", "length": 3, "width": 4, "height": 5,
      "report": "surface"}, "94"),
    # inclusion-exclusion: 30 + 25 - 12 = 43 of 50, so 7 take neither
    ({"solver": "inclusion_exclusion", "sizes": {"a": 30, "b": 25, "a&b": 12},
      "total": 50, "report": "neither"}, "7"),
    # formula: momentum of 6 kg at 14 m/s
    ({"solver": "formula", "name": "momentum", "args": [6, 14]}, "84"),
    # 2, 6, 12, 20, 30 is k(k+1) with the FIRST term at index 0, so term 20 is
    # 21*22 = 462. The anchor said 420, which is the one-based reading — the ninth
    # hand-anchor error of the study against one machinery error.
    ({"solver": "sequence", "terms": [2, 6, 12, 20, 30], "n": 20}, "462"),
    ({"solver": "sequence", "terms": [3, 6, 12, 24], "n": 10}, "3072"),
    # 3(54-10) - 8(36-14) + (20-42) = 132 - 176 - 22 = -66, worked by hand outside
    # the library; the anchor said -124.
    ({"solver": "matrix", "kind": "determinant",
      "matrix": [[3, 8, 1], [4, 6, 2], [7, 5, 9]]}, "-66"),
    ({"solver": "matrix", "kind": "power", "matrix": [[1, 1], [1, 0]],
      "exponent": 10}, [["89", "55"], ["55", "34"]]),
    # 100 from 1, 5, 10, 25 coins: the classic 242
    ({"solver": "partition", "kind": "unordered", "total": 100,
      "parts": [1, 5, 10, 25]}, 242),
    ({"solver": "logexp", "kind": "digits_of_power", "base": 2, "exponent": 1000},
     302),
    ({"solver": "logexp", "kind": "trailing_zeros_factorial", "n": 1000}, 249),
    # base arithmetic: 4213 + 3654 in base 7 is 11200 (decimal 1483 + 1362 = 2845)
    ({"solver": "basearith", "op": "add", "values": ["4213", "3654"],
      "from_base": 7, "to_base": 7}, "11200"),
    ({"solver": "basearith", "op": "convert", "value": "beef", "from_base": 16,
      "to_base": 2}, "1011111011101111"),
    # MISSISSIPPI: 11! / (4! 4! 2!) = 34650
    ({"solver": "strcount", "word": "MISSISSIPPI", "kind": "arrangements"}, 34650),
    ({"solver": "approx", "kind": "best_rational", "value": "355/113",
      "max_denominator": 50}, "22/7"),
    ({"solver": "primes", "kind": "count", "from": 1, "to": 100}, 25),
    ({"solver": "primes", "kind": "nth", "n": 1000}, 7919),
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
    ({"solver": "arith", "let": {"a": "2 ** 5000"}}, "exponent must be a literal"),
    ({"solver": "geometry", "kind": "circle_through",
      "points": [[0, 0], [1, 1], [2, 2]]}, "collinear"),
    ({"solver": "convert", "value": 1, "from": "kr", "to": "kg"}, "no route"),
    ({"solver": "sequence", "terms": [1, 2, 4, 8, 17], "n": 9}, "no arithmetic"),
    ({"solver": "matrix", "kind": "inverse", "matrix": [[1, 2], [2, 4]]},
     "singular"),
    ({"solver": "basearith", "op": "convert", "value": "19", "from_base": 8,
      "to_base": 10}, "not a base-8 numeral"),
    ({"solver": "arith", "let": {"a": "1", "b": "c + 1"}}, "unknown variable"),
    ({"solver": "iterate", "init": "1", "step": "acc * m", "from": 1, "to": 3},
     "unknown variable"),
    ({"solver": "arith", "let": {"a": "5 / 0"}}, "divides by zero"),
]


def main(out="data/custom/solvers2.json"):
    passed = failed = 0
    rows = []
    for spec, truth in CASES:
        res, why = run2(spec)
        got = res["value"] if res else None
        if spec["solver"] == "statistics":
            ok = res is not None and got["population_variance"] == "4"
            got = got and got["population_variance"]
        elif spec.get("kind") == "circle_through":
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
