"""Exact calculus: differentiation, polynomial integration, and autograd.

Three ways to take a derivative live here, and the redundancy is the design:

* **Symbolic** -- ``differentiate`` transforms the expression tree by the sum,
  product, quotient and power rules, with the chain rule for powers of
  subexpressions. Readable output, exact evaluation at rational points.
* **Autograd** -- ``grad`` evaluates the same expression over dual numbers
  (a + b*eps with eps^2 = 0), where the chain rule falls out of the arithmetic
  instead of being implemented. A genuinely independent second implementation:
  the tests demand the two agree at random rational points, so a bug in either
  has to be a matching bug in both to survive.
* **Autograd of procedures** -- ``grad_pyspell`` interprets the PySpell subset
  (assignment, if/elif/else, while, break/continue, return) over dual numbers,
  so a function with loops and branches differentiates exactly at a point.
  PySpell itself cannot host dual numbers -- it has no way to build a pair --
  so the interpreter runs the same LANGUAGE outside the database. Where mpedb
  is installed, a test feeds each source to real PySpell to keep that claim
  honest.

Everything is exact over ``fractions.Fraction`` and the contract is mpeqs':
an exact answer or a ``Refusal`` naming what stopped it, never a float that
looks close and never a guess. Integration stops at expanded polynomials on
purpose -- the power rule is a theorem there, and everything past it is a
research field wearing a function signature.
"""

from __future__ import annotations

import ast
import math
from fractions import Fraction

from solvers import Refusal

# The same normalisation the arithmetic gate applies: what models write,
# mapped to what Python parses.
_SUBSTITUTIONS = {"×": "*", "·": "*", "÷": "/", "−": "-", "–": "-", "^": "**"}

_BUDGET = 250_000  # PySpell's own instruction budget, matched.


def _parse(expression: str, var: str) -> ast.expr:
    text = str(expression or "").strip()
    for wrong, right in _SUBSTITUTIONS.items():
        text = text.replace(wrong, right)
    try:
        tree = ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        raise Refusal(f"expression syntax: {exc.msg}") from None
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id != var:
            raise Refusal(f"unknown name {node.id!r}; the variable here is {var!r}")
        if isinstance(node, ast.Call):
            raise Refusal("function calls are not differentiable here")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exponent = node.right
            if isinstance(exponent, ast.UnaryOp) and isinstance(exponent.op, ast.USub):
                exponent = exponent.operand
            if not (isinstance(exponent, ast.Constant) and isinstance(exponent.value, int)):
                raise Refusal("exponents must be integer literals -- x**x is not "
                              "a polynomial and is refused rather than guessed at")
    return tree


# ---------------------------------------------------------------- symbolic

def differentiate(expression: str, var: str = "x") -> str:
    """The derivative as an expression, by the textbook rules.

    The chain rule lives in the power case: d/dx u(x)**n = n*u**(n-1) * u'.
    Output is lightly simplified -- zero terms dropped, unit factors folded --
    for readability, not canonical form; exactness is the evaluator's job.
    """
    return _render(_diff(_parse(expression, var), var))


def derivative_at(expression: str, var: str = "x", at=0) -> Fraction:
    """The derivative evaluated exactly at a rational point."""
    tree = _parse(expression, var)
    return _eval(_diff(tree, var), var, Fraction(at))


def evaluate_at(expression: str, var: str = "x", at=0) -> Fraction:
    """The expression itself, evaluated exactly at a rational point."""
    return _eval(_parse(expression, var), var, Fraction(at))


def _render(node: ast.expr) -> str:
    """The tree as text, with the noise folded away.

    Simplification here is cosmetic -- drop zero terms, fold unit factors --
    because the exact answers come from the evaluator, not from this string.
    A missed simplification is ugly; a wrong one would be a bug, so the folds
    are only the ones that are identities.
    """
    node = _simplify(node)
    text = ast.unparse(node)
    return text.replace("**", "**").strip()


def _simplify(node: ast.expr) -> ast.expr:
    if isinstance(node, ast.BinOp):
        left, right = _simplify(node.left), _simplify(node.right)
        lval = left.value if isinstance(left, ast.Constant) else None
        rval = right.value if isinstance(right, ast.Constant) else None
        if isinstance(node.op, ast.Add):
            if lval == 0:
                return right
            if rval == 0:
                return left
            if lval is not None and rval is not None:
                return ast.Constant(lval + rval)
        if isinstance(node.op, ast.Sub) and rval == 0:
            return left
        if isinstance(node.op, ast.Mult):
            if lval == 0 or rval == 0:
                return ast.Constant(0)
            if lval == 1:
                return right
            if rval == 1:
                return left
            if lval is not None and rval is not None:
                return ast.Constant(lval * rval)
        if isinstance(node.op, ast.Pow):
            if rval == 1:
                return left
            if rval == 0:
                return ast.Constant(1)
        return ast.BinOp(left, node.op, right)
    if isinstance(node, ast.UnaryOp):
        return ast.UnaryOp(node.op, _simplify(node.operand))
    return node


def _diff(node: ast.expr, var: str) -> ast.expr:
    if isinstance(node, ast.Constant):
        return ast.Constant(0)
    if isinstance(node, ast.Name):
        return ast.Constant(1)
    if isinstance(node, ast.UnaryOp):
        return ast.UnaryOp(node.op, _diff(node.operand, var))
    if isinstance(node, ast.BinOp):
        u, v = node.left, node.right
        du, dv = _diff(u, var), _diff(v, var)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return ast.BinOp(du, node.op, dv)
        if isinstance(node.op, ast.Mult):
            # (uv)' = u'v + uv'
            return ast.BinOp(ast.BinOp(du, ast.Mult(), v), ast.Add(),
                             ast.BinOp(u, ast.Mult(), dv))
        if isinstance(node.op, ast.Div):
            # (u/v)' = (u'v - uv') / v**2
            top = ast.BinOp(ast.BinOp(du, ast.Mult(), v), ast.Sub(),
                            ast.BinOp(u, ast.Mult(), dv))
            return ast.BinOp(top, ast.Div(), ast.BinOp(v, ast.Pow(), ast.Constant(2)))
        if isinstance(node.op, ast.Pow):
            # d u**n = n * u**(n-1) * u'  -- the chain rule, for integer n.
            n = _exponent_value(node.right)
            return ast.BinOp(
                ast.BinOp(ast.Constant(n), ast.Mult(),
                          ast.BinOp(u, ast.Pow(), ast.Constant(n - 1))),
                ast.Mult(), du)
    raise Refusal(f"cannot differentiate a {type(node).__name__}")


def _exponent_value(node: ast.expr) -> int:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_exponent_value(node.operand)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    raise Refusal("exponents must be integer literals")


# ------------------------------------------------------------- integration

def integrate(expression: str, var: str = "x", lower=None, upper=None):
    """Exact polynomial integration: the antiderivative, or a definite value.

    Term-wise power rule over an EXPANDED polynomial. A quotient with the
    variable underneath, a variable exponent, or an unexpanded composite like
    (x+1)**2 is refused with what stopped it -- integrating past polynomials
    exactly is a research field, and a partial answer that looks total is the
    kind of lie the rest of this library exists to avoid.

    With bounds, returns the exact ``Fraction``; without, the antiderivative
    as text (constant of integration omitted, and said so here rather than
    implied).
    """
    tree = _parse(expression, var)
    terms = _poly_terms(tree, var)

    anti: dict[int, Fraction] = {}
    for power, coefficient in terms.items():
        anti[power + 1] = anti.get(power + 1, Fraction(0)) + coefficient / (power + 1)

    if lower is None and upper is None:
        rendered = " + ".join(
            _term_text(c, p, var) for p, c in sorted(anti.items(), reverse=True) if c
        ) or "0"
        return rendered.replace("+ -", "- ")
    if lower is None or upper is None:
        raise Refusal("a definite integral needs both bounds")

    def value(point: Fraction) -> Fraction:
        return sum((c * point**p for p, c in anti.items()), Fraction(0))

    return value(Fraction(upper)) - value(Fraction(lower))


def _poly_terms(node: ast.expr, var: str) -> dict[int, Fraction]:
    """The polynomial as {power: coefficient}, or a Refusal naming the blocker."""
    if isinstance(node, ast.Constant):
        return {0: _fraction(node.value)}
    if isinstance(node, ast.Name):
        return {1: Fraction(1)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _poly_terms(node.operand, var)
        sign = -1 if isinstance(node.op, ast.USub) else 1
        return {p: sign * c for p, c in inner.items()}
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            left, right = _poly_terms(node.left, var), _poly_terms(node.right, var)
            sign = -1 if isinstance(node.op, ast.Sub) else 1
            for power, coefficient in right.items():
                left[power] = left.get(power, Fraction(0)) + sign * coefficient
            return left
        if isinstance(node.op, ast.Mult):
            left, right = _poly_terms(node.left, var), _poly_terms(node.right, var)
            out: dict[int, Fraction] = {}
            for p1, c1 in left.items():
                for p2, c2 in right.items():
                    out[p1 + p2] = out.get(p1 + p2, Fraction(0)) + c1 * c2
            return out
        if isinstance(node.op, ast.Div):
            right = _poly_terms(node.right, var)
            if set(right) - {0}:
                raise Refusal("the variable is in a denominator; that is not a "
                              "polynomial and 1/x integrates to a logarithm, "
                              "which is not exact here")
            divisor = right.get(0, Fraction(0))
            if divisor == 0:
                raise Refusal("division by zero in the integrand")
            return {p: c / divisor for p, c in _poly_terms(node.left, var).items()}
        if isinstance(node.op, ast.Pow):
            n = _exponent_value(node.right)
            if n < 0:
                raise Refusal("a negative power of the variable is not a polynomial")
            base = _poly_terms(node.left, var)
            out = {0: Fraction(1)}
            for _ in range(n):
                nxt: dict[int, Fraction] = {}
                for p1, c1 in out.items():
                    for p2, c2 in base.items():
                        nxt[p1 + p2] = nxt.get(p1 + p2, Fraction(0)) + c1 * c2
                out = nxt
            return out
    raise Refusal(f"cannot integrate a {type(node).__name__}")


def _term_text(coefficient: Fraction, power: int, var: str) -> str:
    c = str(coefficient) if coefficient.denominator != 1 else str(coefficient.numerator)
    if power == 0:
        return c
    x = var if power == 1 else f"{var}**{power}"
    return x if coefficient == 1 else (f"-{x}" if coefficient == -1 else f"{c}*{x}")


# ------------------------------------------------------------------ algebra

def solve_linear(a, b) -> Fraction:
    """The x with a*x + b = 0, exactly."""
    a, b = Fraction(a), Fraction(b)
    if a == 0:
        raise Refusal("a is zero: that is not a linear equation in x")
    return -b / a


def solve_quadratic(a, b, c) -> tuple[Fraction, Fraction]:
    """Both roots of a*x**2 + b*x + c = 0, exactly, largest first.

    Rational roots only. An irrational or complex discriminant is refused BY
    VALUE -- sqrt(discriminant) as a float would be the quiet inexactness this
    library refuses everywhere else.
    """
    a, b, c = Fraction(a), Fraction(b), Fraction(c)
    if a == 0:
        raise Refusal("a is zero: not a quadratic; solve_linear takes it")
    disc = b * b - 4 * a * c
    if disc < 0:
        raise Refusal(f"the discriminant is {disc}: the roots are complex")
    root = _fraction_sqrt(disc)
    if root is None:
        raise Refusal(f"the discriminant is {disc}, not a perfect square: the "
                      "roots are irrational and would not be exact")
    first, second = (-b + root) / (2 * a), (-b - root) / (2 * a)
    return (first, second) if first >= second else (second, first)


def _fraction_sqrt(value: Fraction) -> Fraction | None:
    """The exact square root of a Fraction, or None if it is irrational."""
    top, bottom = math.isqrt(value.numerator), math.isqrt(value.denominator)
    if top * top == value.numerator and bottom * bottom == value.denominator:
        return Fraction(top, bottom)
    return None


# ----------------------------------------------------------------- autograd

class Dual:
    """a + b*eps with eps**2 = 0: forward-mode autograd over exact Fractions.

    The chain rule is not implemented anywhere in this class -- it emerges from
    the multiplication rule, which is the point of dual numbers and the reason
    this is an independent check on the symbolic path.
    """

    __slots__ = ("eps", "real")

    def __init__(self, real, eps=0) -> None:
        self.real = Fraction(real)
        self.eps = Fraction(eps)

    def __add__(self, other):
        other = _dual(other)
        return Dual(self.real + other.real, self.eps + other.eps)

    __radd__ = __add__

    def __sub__(self, other):
        other = _dual(other)
        return Dual(self.real - other.real, self.eps - other.eps)

    def __rsub__(self, other):
        return _dual(other) - self

    def __mul__(self, other):
        other = _dual(other)
        return Dual(self.real * other.real,
                    self.real * other.eps + self.eps * other.real)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = _dual(other)
        if other.real == 0:
            raise Refusal("division by zero")
        return Dual(self.real / other.real,
                    (self.eps * other.real - self.real * other.eps)
                    / (other.real * other.real))

    def __rtruediv__(self, other):
        return _dual(other) / self

    def __pow__(self, n):
        if not isinstance(n, int):
            raise Refusal("dual powers take integer exponents")
        if n == 0:
            return Dual(1)
        if n < 0:
            return Dual(1) / (self ** (-n))
        out = Dual(1)
        base, k = self, n
        while k:
            if k & 1:
                out = out * base
            base = base * base
            k >>= 1
        return out

    def __neg__(self):
        return Dual(-self.real, -self.eps)


def _dual(value) -> Dual:
    return value if isinstance(value, Dual) else Dual(value)


def grad(expression: str, at, var: str = "x") -> Fraction:
    """d(expression)/d(var) at a rational point, by dual numbers alone."""
    tree = _parse(expression, var)
    seeded = Dual(Fraction(at), 1)
    return _eval_dual(tree, var, seeded).eps


# ------------------------------------------------------------- evaluation

def _fraction(value) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refusal(f"not a number: {value!r}")
    return Fraction(value)


def _eval(node: ast.expr, var: str, at: Fraction) -> Fraction:
    result = _eval_dual(node, var, Dual(at))
    return result.real


def _eval_dual(node: ast.expr, var: str, seeded: Dual) -> Dual:
    if isinstance(node, ast.Constant):
        return Dual(_fraction(node.value))
    if isinstance(node, ast.Name):
        return seeded
    if isinstance(node, ast.UnaryOp):
        inner = _eval_dual(node.operand, var, seeded)
        return -inner if isinstance(node.op, ast.USub) else inner
    if isinstance(node, ast.BinOp):
        left = _eval_dual(node.left, var, seeded)
        if isinstance(node.op, ast.Pow):
            return left ** _exponent_value(node.right)
        right = _eval_dual(node.right, var, seeded)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise Refusal(f"cannot evaluate a {type(node).__name__}")


# ------------------------------------------------- autograd of procedures

def grad_pyspell(source: str, at) -> Fraction:
    """The derivative of a PySpell function at a point, exactly.

    An interpreter for the PySpell subset -- assignment, if/elif/else, while,
    break/continue, return, arithmetic and comparison -- run over dual numbers,
    so a procedure with loops and branches differentiates without being an
    expression. Branches compare on the value part, which gives the derivative
    of the branch actually taken: exact at every point where the function is
    differentiable, and the discipline of stating the edge belongs here --
    AT a branch point (|x| at 0) the one-sided derivative of the chosen branch
    is returned, which is the standard forward-mode behaviour.

    Same instruction budget as PySpell itself, so a runaway loop fails here
    the way it fails there.
    """
    try:
        module = ast.parse(str(source or ""))
    except SyntaxError as exc:
        raise Refusal(f"source syntax: {exc.msg}") from None
    functions = [n for n in module.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1:
        raise Refusal(f"PySpell wants exactly one def; found {len(functions)}")
    fn = functions[0]
    if len(fn.args.args) != 1:
        raise Refusal("grad_pyspell differentiates single-argument functions")

    env: dict[str, Dual] = {fn.args.args[0].arg: Dual(Fraction(at), 1)}
    budget = [_BUDGET]
    result = _run_block(fn.body, env, budget)
    if result is None:
        raise Refusal("the function returned nothing")
    return result.eps


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


def _run_block(statements, env, budget) -> Dual | None:
    for statement in statements:
        budget[0] -= 1
        if budget[0] <= 0:
            raise Refusal(f"instruction budget of {_BUDGET} exhausted")
        if isinstance(statement, ast.Return):
            if statement.value is None:
                return None
            return _spell_eval(statement.value, env, budget)
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                raise Refusal("assignment targets one plain name")
            env[statement.targets[0].id] = _spell_eval(statement.value, env, budget)
            continue
        if isinstance(statement, ast.AugAssign):
            if not isinstance(statement.target, ast.Name):
                raise Refusal("assignment targets one plain name")
            current = env.get(statement.target.id)
            if current is None:
                raise Refusal(f"unknown variable {statement.target.id!r}")
            value = _spell_eval(statement.value, env, budget)
            env[statement.target.id] = _spell_binop(statement.op, current, value)
            continue
        if isinstance(statement, ast.If):
            branch = statement.body if _truthy(statement.test, env, budget) else statement.orelse
            result = _run_block(branch, env, budget)
            if result is not None:
                return result
            continue
        if isinstance(statement, ast.While):
            while _truthy(statement.test, env, budget):
                budget[0] -= 1
                if budget[0] <= 0:
                    raise Refusal(f"instruction budget of {_BUDGET} exhausted")
                try:
                    result = _run_block(statement.body, env, budget)
                except _Break:
                    break
                except _Continue:
                    continue
                if result is not None:
                    return result
            continue
        if isinstance(statement, ast.Break):
            raise _Break()
        if isinstance(statement, ast.Continue):
            raise _Continue()
        if isinstance(statement, ast.Pass):
            continue
        raise Refusal(f"{type(statement).__name__} is outside the PySpell subset")
    return None


def _truthy(node, env, budget) -> bool:
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise Refusal("one comparison at a time")
        left = _spell_eval(node.left, env, budget).real
        right = _spell_eval(node.comparators[0], env, budget).real
        op = node.ops[0]
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        raise Refusal(f"{type(op).__name__} is outside the PySpell subset")
    if isinstance(node, ast.BoolOp):
        parts = (_truthy(v, env, budget) for v in node.values)
        return all(parts) if isinstance(node.op, ast.And) else any(parts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _truthy(node.operand, env, budget)
    raise Refusal("conditions are comparisons, and/or, or not")


def _spell_eval(node, env, budget) -> Dual:
    budget[0] -= 1
    if budget[0] <= 0:
        raise Refusal(f"instruction budget of {_BUDGET} exhausted")
    if isinstance(node, ast.Constant):
        return Dual(_fraction(node.value))
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise Refusal(f"unknown variable {node.id!r}")
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _spell_eval(node.operand, env, budget)
        return -inner if isinstance(node.op, ast.USub) else inner
    if isinstance(node, ast.BinOp):
        left = _spell_eval(node.left, env, budget)
        if isinstance(node.op, ast.Pow):
            return left ** _exponent_value(node.right)
        right = _spell_eval(node.right, env, budget)
        return _spell_binop(node.op, left, right)
    raise Refusal(f"{type(node).__name__} is outside the differentiable subset")


def _spell_binop(op, left: Dual, right: Dual) -> Dual:
    if isinstance(op, ast.Add):
        return left + right
    if isinstance(op, ast.Sub):
        return left - right
    if isinstance(op, ast.Mult):
        return left * right
    if isinstance(op, ast.Div):
        return left / right
    raise Refusal(f"{type(op).__name__} is not differentiable here -- floor "
                  "division and modulo have zero derivative almost everywhere "
                  "and a lie at the jumps, so they are refused")
