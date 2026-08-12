"""Calculus, checked by disagreement-hunting rather than by trusting the author.

The module ships three independent derivative implementations -- symbolic
rules, dual-number autograd, and a PySpell-subset interpreter over duals --
and the central test here is that they AGREE at scores of random rational
points. A bug in any one of them has to be a matching bug in another to
survive, which is the same discipline the battery uses: no answer rests on
anybody's memory of the power rule.
"""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

from mpeqs import Refusal, calculus

EXPRESSIONS = [
    "x**2",
    "x**3 - 4*x + 7",
    "(3*x**2 + 5)**4",
    "(x + 1)*(x - 1)",
    "x/(x**2 + 1)",
    "(2*x - 3)**5 + x**2",
    "7",
    "-x**4 + x/3",
    "((x**2 + 1)**2 + x)**2",
]


class TestTheThreePathsAgree:
    @pytest.mark.parametrize("expression", EXPRESSIONS)
    def test_symbolic_and_dual_agree_at_random_rational_points(self, expression):
        rng = random.Random(20260812)
        for _ in range(8):
            at = Fraction(rng.randrange(-40, 40), rng.randrange(1, 12))
            symbolic = calculus.derivative_at(expression, at=at)
            dual = calculus.grad(expression, at)
            assert symbolic == dual, f"{expression} at {at}: {symbolic} != {dual}"

    def test_the_interpreter_agrees_with_both_on_a_loop(self):
        """(3x^2+5)^4 written as a loop must equal the expression's derivative."""
        source = (
            "def f(x):\n"
            "    y = 1\n"
            "    n = 0\n"
            "    while n < 4:\n"
            "        y = y * (3*x*x + 5)\n"
            "        n = n + 1\n"
            "    return y\n"
        )
        rng = random.Random(7)
        for _ in range(6):
            at = Fraction(rng.randrange(-20, 20), rng.randrange(1, 9))
            by_interpreter = calculus.grad_pyspell(source, at)
            by_expression = calculus.grad("(3*x**2 + 5)**4", at)
            assert by_interpreter == by_expression

    def test_hand_anchors(self):
        """A few results small enough to know absolutely."""
        assert calculus.derivative_at("x**2", at=3) == 6
        assert calculus.derivative_at("x**3", at=2) == 12
        assert calculus.grad("x**2 + x", Fraction(1, 2)) == 2
        assert calculus.integrate("x**2", lower=0, upper=1) == Fraction(1, 3)
        assert calculus.integrate("x", lower=0, upper=2) == 2


class TestSymbolicOutput:
    def test_the_rendered_derivative_reads_like_the_textbook(self):
        assert calculus.differentiate("x**2") == "2 * x"
        assert calculus.differentiate("7") == "0"

    def test_the_chain_rule_is_visible_in_the_output(self):
        text = calculus.differentiate("(3*x**2 + 5)**4")
        assert "** 3" in text, "the outer power must have come down by one"
        assert "2 * x" in text, "the inner derivative must be present"


class TestIntegration:
    def test_differentiating_the_antiderivative_gives_back_the_polynomial(self):
        """The round trip, on random polynomials -- the theorem as a test."""
        rng = random.Random(20260812)
        for _ in range(10):
            coefficients = [rng.randrange(-9, 10) for _ in range(rng.randrange(2, 5))]
            polynomial = " + ".join(
                f"{c}*x**{p}" for p, c in enumerate(coefficients) if c) or "0"
            antiderivative = calculus.integrate(polynomial)
            for _ in range(4):
                at = Fraction(rng.randrange(-15, 15), rng.randrange(1, 7))
                # d/dx of the antiderivative is the polynomial's VALUE. The
                # first version compared against its derivative -- the test
                # being wrong where the code was right, in the useful direction.
                assert calculus.derivative_at(antiderivative, at=at) == \
                    calculus.evaluate_at(polynomial, at=at)

    def test_definite_integrals_are_exact_fractions(self):
        value = calculus.integrate("x**3 - x", lower=Fraction(1, 2), upper=3)
        assert isinstance(value, Fraction)
        # F(x) = x^4/4 - x^2/2; F(3) - F(1/2) = (81/4 - 9/2) - (1/64 - 1/8)
        assert value == Fraction(81, 4) - Fraction(9, 2) - Fraction(1, 64) + Fraction(1, 8)

    def test_composites_integrate_by_expansion(self):
        # (x+1)**2 is a polynomial and the expander handles the power.
        assert calculus.integrate("(x + 1)**2", lower=0, upper=1) == \
            calculus.integrate("x**2 + 2*x + 1", lower=0, upper=1)

    @pytest.mark.parametrize("blocked, why", [
        ("1/x", "denominator"),
        ("x**-1", "negative power"),
        ("2**x", "integer literals"),
    ])
    def test_past_polynomials_is_refused_by_name(self, blocked, why):
        with pytest.raises(Refusal, match=why):
            calculus.integrate(blocked, lower=1, upper=2)


class TestAlgebra:
    def test_roots_substituted_back_give_zero(self):
        rng = random.Random(3)
        for _ in range(10):
            # Build a quadratic FROM its roots, so the truth is by construction.
            r1 = Fraction(rng.randrange(-12, 13), rng.randrange(1, 5))
            r2 = Fraction(rng.randrange(-12, 13), rng.randrange(1, 5))
            a = rng.randrange(1, 6)
            b, c = -a * (r1 + r2), a * r1 * r2
            top, bottom = calculus.solve_quadratic(a, b, c)
            assert {top, bottom} == {max(r1, r2), min(r1, r2)}
            assert a * top**2 + b * top + c == 0

    def test_linear_is_exact(self):
        assert calculus.solve_linear(3, -12) == 4
        assert calculus.solve_linear(Fraction(1, 3), 1) == -3

    def test_irrational_and_complex_are_refused_by_value(self):
        with pytest.raises(Refusal, match="not a perfect square"):
            calculus.solve_quadratic(1, 0, -2)  # roots ±sqrt(2)
        with pytest.raises(Refusal, match="complex"):
            calculus.solve_quadratic(1, 0, 1)

    def test_a_degenerate_quadratic_is_redirected(self):
        with pytest.raises(Refusal, match="solve_linear"):
            calculus.solve_quadratic(0, 2, 1)


class TestTheInterpreter:
    def test_branches_differentiate_the_branch_taken(self):
        source = (
            "def f(x):\n"
            "    if x < 0:\n"
            "        return x * x * x\n"
            "    return x * x\n"
        )
        assert calculus.grad_pyspell(source, 3) == 6        # d(x^2)
        assert calculus.grad_pyspell(source, -2) == 12      # d(x^3) = 3x^2

    def test_a_runaway_loop_hits_the_budget(self):
        source = "def f(x):\n    while x > -999999:\n        x = x + 1\n    return x\n"
        with pytest.raises(Refusal, match="budget"):
            calculus.grad_pyspell(source, 0)

    def test_floor_division_is_refused_not_zeroed(self):
        """// has zero derivative almost everywhere and a lie at the jumps."""
        with pytest.raises(Refusal, match="not differentiable"):
            calculus.grad_pyspell("def f(x):\n    return x // 2\n", 5)

    def test_the_source_must_be_one_single_argument_def(self):
        with pytest.raises(Refusal, match="exactly one def"):
            calculus.grad_pyspell("x = 1", 0)
        with pytest.raises(Refusal, match="single-argument"):
            calculus.grad_pyspell("def f(x, y):\n    return x\n", 0)

    def test_the_same_source_compiles_in_real_pyspell(self):
        """The claim is 'the same language', and only mpedb can check it."""
        from mpeqs import pyspell

        if not pyspell.available():
            pytest.skip("needs the optional mpedb backend: pip install mpeqs[pyspell]")
        source = (
            "def g(x):\n"
            "    y = 1\n"
            "    n = 0\n"
            "    while n < 3:\n"
            "        y = y * (2*x + 1)\n"
            "        n = n + 1\n"
            "    return y\n"
        )
        spell = pyspell.Spell()
        assert spell.call(source, 2) == 125          # (2*2+1)^3, in the database
        assert calculus.grad_pyspell(source, 2) == \
            calculus.grad("(2*x + 1)**3", 2)         # and differentiated out here


class TestRefusals:
    def test_an_unknown_name_is_named(self):
        with pytest.raises(Refusal, match="unknown name 'y'"):
            calculus.differentiate("x + y")

    def test_variable_exponents_are_refused(self):
        with pytest.raises(Refusal, match="integer literals"):
            calculus.differentiate("x**x")

    def test_calls_are_refused(self):
        with pytest.raises(Refusal, match="not differentiable"):
            calculus.differentiate("sin(x)")
