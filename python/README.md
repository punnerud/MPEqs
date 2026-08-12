# mpeqs

Exact answers for the arithmetic a language model should not be doing itself.

[![Linux](https://github.com/punnerud/MPEqs/actions/workflows/linux.yml/badge.svg?branch=main)](https://github.com/punnerud/MPEqs/actions/workflows/linux.yml)
[![macOS](https://github.com/punnerud/MPEqs/actions/workflows/macos.yml/badge.svg?branch=main)](https://github.com/punnerud/MPEqs/actions/workflows/macos.yml)
[![Windows](https://github.com/punnerud/MPEqs/actions/workflows/windows.yml/badge.svg?branch=main)](https://github.com/punnerud/MPEqs/actions/workflows/windows.yml)
[![PyPI](https://img.shields.io/pypi/v/mpeqs.svg)](https://pypi.org/project/mpeqs/)

```bash
pip install mpeqs
```

```python
import mpeqs

mpeqs.solve({"solver": "arith", "answer": "(17/100)*250"})   # Fraction(85, 2)
```

The model names a solver and fills its slots. This evaluates the spec exactly, or
refuses by name — it never guesses, and it never returns a number it could not derive.

**The refusal is the point.** A solver that cannot evaluate a spec raises `Refusal`
with a reason, and that reason is a routing signal: it tells the caller to ask the
model, rather than quietly handing back a wrong number.

```python
try:
    mpeqs.solve({"solver": "arith", "answer": "1/0"})
except mpeqs.Refusal as why:
    ...   # ask the model; the record declined this one
```

**Shape disposes over name.** A model that picks the wrong solver but fills the right
slots has still described the problem correctly, so a spec carrying an `answer` key is
arithmetic whatever it calls itself. Pass `repair=False` to turn that off.

## What is exact and what is not

Results come back as `fractions.Fraction`, so nothing is lost on the way out.
The exactness is in the arithmetic, not in the parsing: a ratio written as a ratio
stays exact, while a decimal literal is read as a float first.

```python
mpeqs.solve({"solver": "arith", "answer": "1/10+2/10"})   # Fraction(3, 10)  exact
mpeqs.solve({"solver": "arith", "answer": "0.1+0.2"})     # the binary expansion
```

## Solvers

`arith`, `geometry`, `iterate`, `modular`, `multisearch`, `polynomial`, and the named
families behind them — factorisation, gcd/lcm, linear systems, quadratics, remainders,
combinatorics, rates, mixtures, ratios, base conversion.

`mpeqs.solvers()` lists what a given build dispatches to.

## No dependencies

The whole library is standard library. One `py3-none-any` wheel serves Linux, macOS and
Windows on every supported interpreter, the install is instant, and a test asserts that
no third-party import creeps in.

## Calculus: three derivatives that must agree

`mpeqs.calculus` differentiates (chain rule included), integrates expanded polynomials
exactly, and solves linear and quadratic equations with exact rational roots — refusing
irrational discriminants **by value** rather than floating them.

The derivative ships three ways on purpose: symbolic rules, dual-number **autograd**
(the chain rule falls out of the arithmetic), and `grad_pyspell` — an interpreter for the
PySpell subset run over dual numbers, so a function with loops and branches differentiates
exactly at a point. The tests demand all three agree at scores of random rational points;
a bug in one must be a matching bug in another to survive.

```python
from mpeqs import calculus
calculus.differentiate("(3*x**2 + 5)**4")        # 4 * (3*x**2 + 5)**3 * (3 * (2*x))
calculus.integrate("x**2", lower=0, upper=1)      # Fraction(1, 3)
calculus.solve_quadratic(1, -5, 6)                # (Fraction(3), Fraction(2))
calculus.grad_pyspell("""
def f(x):
    y = 1
    n = 0
    while n < 4:
        y = y * (3*x*x + 5)
        n = n + 1
    return y
""", Fraction(1, 2))                              # Fraction(36501, 16) -- exact
```

## PySpell: when an expression is not enough

`solve()` evaluates *expressions*, and is safe by restriction — the source is parsed to
an AST, walked against a whitelist, and evaluated with `__builtins__` emptied. Nothing
runs that was not understood first.

What it cannot express is a *procedure*. There is no expression for "how many steps does
the Collatz sequence take from 27" — that needs a loop, a branch and an accumulator. The
usual answer is to let the model emit Python and `exec` it, which trades a whitelist for
a sandbox and hope.

`mpeqs.pyspell` is the third option. PySpell is [mpedb](https://github.com/punnerud/mpedb)'s
stored-function language: a small deterministic subset of Python with no imports, no
clock, no randomness, no file or network I/O, and a fixed instruction budget so a runaway
loop fails identically everywhere. **A model can write a loop. It still cannot open a
socket.**

```python
import mpeqs.pyspell as spell

spell.call("""
def col(n):
    if n < 1:
        return 1 // 0
    c = 0
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        c = c + 1
    return c
""", 27)
# 111
```

And when the model claims a *relationship* between two functions, the claim is checked
against a probe corpus rather than believed:

```python
spell.check_bijective(
    "def dbl(x):\n    return x * 2\n",
    "def hlv(x):\n    return x // 2\n")
# mpeqs.Refusal: dbl/hlv is not bijective: ... forward(Float(1e308)) = Float(inf) ...
```

That refusal is correct. `x*2` then `x//2` is genuinely not bijective over the values
PySpell admits — a denormal floors to zero, and `1e308` is an integral float whose double
is infinity. The verifier names the input that breaks it, every time. Guard the domain to
bounded positive integers and the same pair is accepted, reporting how many probe values
actually round-tripped rather than a bare "verified".

It caught the author of that example twice while it was being written. That is the
argument for "declare and check" over "generate and hope".

**Optional**, because it is not free: mpedb ships as a compiled wheel needing CPython
3.12 or newer, while mpeqs itself runs from 3.10 with one artefact for every platform.

```bash
pip install mpeqs[pyspell]
```

Without it, `mpeqs.pyspell.available()` is `False` and every entry point raises `Refusal`
naming the extra, rather than failing at import. Nothing else in mpeqs changes.

## Licence

[The mpedb License 1.0](../LICENSE) — free of charge for every person and every
organization, except that a group over five billion dollars in revenue or valuation
owes seven US cents per device, once. Not an OSI-approved licence.
