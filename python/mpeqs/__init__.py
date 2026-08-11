"""Exact answers for the arithmetic a language model should not be doing itself.

The model names a solver and fills its slots; this evaluates the result exactly,
or refuses by name. It never guesses, and it never returns a number it could not
derive -- a refusal is the useful signal, not a consolation.

    >>> import mpeqs
    >>> mpeqs.solve({"solver": "arith", "answer": "(17/100)*250"})
    Fraction(85, 2)
    >>> float(_)
    42.5

Results come back as ``fractions.Fraction``, so nothing is lost to rounding on the
way out: ``(17/100)*250`` is exactly ``85/2``.

The exactness is in the arithmetic, not in the parsing. A ratio written as a ratio
stays exact; a decimal literal is read as a float first, so ``0.1+0.2`` carries the
binary expansion in. Write ``1/10+2/10`` when it matters.

A spec that cannot be evaluated raises ``Refusal`` with a reason:

    >>> mpeqs.solve({"solver": "arith", "answer": "1/0"})
    Traceback (most recent call last):
    mpeqs.Refusal: ...

Pure standard library. No dependencies, one wheel for every platform.

For a *procedure* rather than an expression -- a loop, a branch, an accumulator --
see ``mpeqs.pyspell``, which runs code the model wrote in a deterministic subset
with the property it claims verified rather than believed. That one needs the
optional ``mpeqs[pyspell]`` extra.
"""

from __future__ import annotations

import pathlib
import sys
from fractions import Fraction

# The solver modules bootstrap their own directory onto sys.path and import each
# other flatly. They are shipped verbatim rather than rewritten, so the library
# published here is the one the measurements in this repository were made against.
_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from solvers import Refusal
from solvers2 import SOLVERS2, run2

__version__ = "0.3.0"
__all__ = ["SPEC_HELP", "Refusal", "run2", "solve", "solvers"]

SPEC_HELP = (
    "A spec is a JSON object naming one solver and filling its slots, for example "
    '{"solver": "arith", "answer": "(17/100)*250"}. Ask a model for the spec, not '
    "for the arithmetic."
)


def solvers() -> list[str]:
    """Which solvers this build can dispatch to."""
    return sorted(SOLVERS2)


def solve(spec: dict, *, repair: bool = True):
    """Evaluate a spec exactly, or raise ``Refusal`` naming what stopped it.

    ``repair`` lets the SHAPE of the payload overrule the NAME when a model picks
    the wrong solver but fills the right slots -- a spec carrying ``residues`` is a
    remainder problem whatever it calls itself. Repairs are counted, not silent.
    """
    result, why = run2(spec, repair=repair)
    if result is None:
        raise Refusal(why)

    value = result.get("value", result) if isinstance(result, dict) else result
    # The solvers hand back an exact value as a string like "85/2". Returning it as
    # a Fraction is what makes the exactness usable: float(Fraction("85/2")) is
    # 42.5, while float("85/2") is a ValueError.
    if isinstance(value, str):
        try:
            return Fraction(value)
        except (ValueError, ZeroDivisionError):
            return value
    return value
