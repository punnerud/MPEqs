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

## Licence

[The mpedb License 1.0](../LICENSE) — free of charge for every person and every
organization, except that a group over five billion dollars in revenue or valuation
owes seven US cents per device, once. Not an OSI-approved licence.
