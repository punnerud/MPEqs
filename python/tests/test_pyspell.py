"""PySpell: running code a model wrote, with its claims checked rather than believed.

The optional backend is a compiled wheel needing CPython 3.12+, so every test here
skips cleanly without it -- mpeqs itself stays pure Python from 3.10, and a user
who never installs the extra must never see a failure from it.
"""

from __future__ import annotations

import pytest

from mpeqs import Refusal
from mpeqs import pyspell as spell

pytestmark = pytest.mark.skipif(
    not spell.available(), reason="needs the optional mpedb backend: pip install mpeqs[pyspell]")

# Every guard here was demanded by the verifier, one refusal at a time. Integers
# only (a fractional float breaks the round trip), positive (0 is excluded because
# +0.0 and -0.0 genuinely collide), and bounded above (1e308 is an integral float,
# and doubling it is inf).
GUARD = (
    "    if x % 1 != 0:\n        return 1 // 0\n"
    "    if x < 1:\n        return 1 // 0\n"
    "    if x > 1000000000:\n        return 1 // 0\n"
)
DOUBLE = f"def dbl(x):\n{GUARD}    return x * 2\n"
HALVE = f"def hlv(x):\n{GUARD}    return x // 2\n"

COLLATZ = (
    "def col(n):\n"
    "    if n < 1:\n        return 1 // 0\n"
    "    c = 0\n"
    "    while n != 1:\n"
    "        if n % 2 == 0:\n            n = n // 2\n"
    "        else:\n            n = 3 * n + 1\n"
    "        c = c + 1\n"
    "        if c > 100000:\n            return 1 // 0\n"
    "    return c\n"
)


def test_a_procedure_runs_and_returns_the_right_answer():
    """The capability an expression evaluator does not have: a loop.

    solve() handles "(17/100)*250" and always will. There is no expression for
    "how many steps does the Collatz sequence take from 27", and this is the
    difference between the two tools.
    """
    assert spell.call(COLLATZ, 27) == 111
    assert spell.call(COLLATZ, 97) == 118
    assert spell.call(COLLATZ, 871) == 178


def test_the_function_name_comes_from_the_definition():
    assert spell.function_name(COLLATZ) == "col"
    with pytest.raises(Refusal, match="exactly one"):
        spell.function_name("x = 1")
    with pytest.raises(Refusal, match="exactly one"):
        spell.function_name("def a(x):\n    return x\ndef b(x):\n    return x\n")


def test_a_true_claim_is_accepted_and_says_how_much_was_probed():
    probes = spell.check_bijective(DOUBLE, HALVE, name="doubling")
    # A count, not a bare "verified". A narrow domain guard exercises few probe
    # values, and a caller is entitled to know that before trusting the pair.
    assert probes > 0


def test_a_false_claim_is_refused_with_the_input_that_breaks_it():
    square = f"def sq(x):\n{GUARD}    return x * x\n"
    with pytest.raises(Refusal) as caught:
        spell.check_bijective(square, HALVE, name="squaring")
    assert "not bijective" in str(caught.value)
    # The refusal names a concrete counter-example rather than saying "no".
    assert "forward(" in str(caught.value)


def test_an_unguarded_domain_is_refused_even_when_the_maths_looks_right():
    """x*2 then x//2 is not bijective over floats, and the verifier knows it.

    This caught the author of this test twice: once on a denormal flooring to
    zero, and once on 1e308 doubling to inf. Both times the declaration was wrong
    and the verifier was right.
    """
    loose_double = "def ld(x):\n    return x * 2\n"
    loose_halve = "def lh(x):\n    return x // 2\n"
    with pytest.raises(Refusal) as caught:
        spell.check_bijective(loose_double, loose_halve, name="unguarded")
    assert "Float" in str(caught.value)


def test_the_language_refuses_imports_at_compile_time():
    """A model can write a loop. It still cannot open a socket.

    This is the whole reason for a subset rather than exec(): the refusal happens
    when the function is defined, with a line number, not when it runs.
    """
    with pytest.raises(Refusal) as caught:
        spell.call("def bad(x):\n    import os\n    return 1\n", 1)
    assert "import is not" in str(caught.value)
    assert "line 2" in str(caught.value)


@pytest.mark.parametrize(
    "source",
    [
        "def f(x):\n    return open('/etc/passwd').read()\n",
        "def f(x):\n    return __import__('os').system('echo pwned')\n",
        "def f(x):\n    return eval('1+1')\n",
        "def f(x):\n    return x.__class__.__bases__\n",
    ],
)
def test_nothing_outside_the_subset_compiles(source: str):
    with pytest.raises(Refusal):
        spell.call(source, 1)


def test_a_runaway_loop_fails_rather_than_hanging():
    """A fixed instruction budget, so this terminates identically everywhere."""
    with pytest.raises(Refusal):
        spell.call("def spin(x):\n    while x > 0:\n        x = x + 1\n    return x\n", 1)


def test_a_domain_refusal_refuses_the_value_rather_than_inventing_one():
    # The same contract as the rest of mpeqs: no number it could not derive.
    with pytest.raises(Refusal):
        spell.call(COLLATZ, 0)


def test_a_residual_pair_records_exactly_what_was_lost():
    """abs() is not injective, and registers fine when rex is the sign.

    x -> (|x|, sign) is injective even though |x| is not, which is the point of
    the residual class: a transform that throws something away and can say
    precisely what.
    """
    guard = "    if x % 1 != 0:\n        return 1 // 0\n    if x > 1000000 or x < -1000000:\n        return 1 // 0\n"
    probes = spell.check_residual(
        f"def mag(x):\n{guard}    if x < 0:\n        return 0 - x\n    return x\n",
        f"def sgn(x):\n{guard}    if x < 0:\n        return 1\n    return 0\n",
        "def unmag(m, s):\n    if s == 1:\n        return 0 - m\n    return m\n",
        residual_type="any",
        name="magnitude",
    )
    assert probes > 0


def test_an_instance_keeps_its_own_definitions():
    one = spell.Spell()
    two = spell.Spell()
    assert one.call(COLLATZ, 27) == 111
    assert two.call(COLLATZ, 27) == 111
    assert one.db is not two.db


def test_without_the_backend_the_error_says_what_to_install(monkeypatch):
    """A user who never installs the extra must get an instruction, not a traceback."""
    import builtins

    real_import = builtins.__import__

    def no_mpedb(name, *args, **kwargs):
        if name == "mpedb":
            raise ImportError("no mpedb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mpedb)
    assert spell.available() is False
    with pytest.raises(Refusal, match=r"mpeqs\[pyspell\]"):
        spell.Spell()
