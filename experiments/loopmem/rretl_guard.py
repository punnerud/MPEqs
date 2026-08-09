#!/usr/bin/env python3
"""Reusable guard injection for rRETL pairs — LLMdb-side, against an unmodified mpedb.

Phase 49 measured the gap this closes: with the documented float traps verbatim in context, a
35B model writing lens triples registered 1 of 4 — it reads the guard discipline and does not
reliably apply it across all three functions. The same model writing only the arithmetic
bodies, with the guard injected by the harness, registered 3 of 4; the hand-written ceiling is
4 of 4. The guard is mechanical, so it is written here once, correctly, for every caller.

The guard refuses, in order: non-integral values (`x % 1 != 0` — fractions and subnormals),
zero (`x == 0` — PySpell comparisons cannot tell -0.0 from +0.0, so both leave the domain
together; the price is the integer 0), and magnitudes above a bound (integral floats above
2^52 pass the fraction test and then lose low bits in arithmetic — mpedb's verifier produced
Float(1.34e17) as the counter-example against an offset pair). The default bound is 2^42,
deliberately far inside exactness: a default nobody thinks about must not sit near the cliff.

This deliberately lives in LLMdb, not in mpedb: the engine stays the unbribable verifier, and
the wrap changes who WRITES the guard, never whether the probe corpus checks it. Wrap the
forward and rex of a residual pair; do NOT wrap the inverse — it only ever sees the pair's own
outputs, and its results may legitimately include values the input domain refuses (a `tens`
forward legitimately produces 0).

    from rretl_guard import create_guarded_residual_lens
    n = create_guarded_residual_lens(db, "off", FWD_SRC, REX_SRC, INV_SRC)
"""

DEFAULT_MAGNITUDE = 1 << 42


class GuardError(ValueError):
    """The source cannot be guarded; the reason says why."""


def _condition(param, refuse_zero, magnitude):
    terms = [f"{param} % 1 != 0"]
    if refuse_zero:
        terms.append(f"{param} == 0")
    if magnitude is not None:
        if magnitude <= 0:
            raise GuardError("guard: magnitude bound must be positive")
        terms.append(f"{param} > {magnitude}")
        terms.append(f"{param} < -{magnitude}")
    return " or ".join(terms)


def wrap_int_guard(source, refuse_zero=True, magnitude=DEFAULT_MAGNITUDE):
    """Prepend the integer-domain guard to a single-def PySpell source, on its FIRST parameter.

    Pure text transform; validation of the result belongs to mpedb's own compiler at
    define time, which refuses anything malformed with a line and column.
    """
    lines = source.split("\n")
    header_idx = next(
        (i for i, l in enumerate(lines)
         if l.lstrip().startswith("def ") and l.rstrip().endswith(":")), None)
    if header_idx is None:
        raise GuardError("guard: no `def name(...):` line in source")
    header = lines[header_idx]
    if "(" not in header or ")" not in header:
        raise GuardError("guard: malformed parameter list")
    params = header.split("(", 1)[1].rsplit(")", 1)[0]
    first = params.split(",")[0].strip()
    if not first:
        raise GuardError("guard: the function takes no parameters to guard")

    body = next((l for l in lines[header_idx + 1:] if l.strip()), None)
    if body is None:
        raise GuardError("guard: the function has no body")
    indent = body[: len(body) - len(body.lstrip())]
    if not indent:
        raise GuardError("guard: the body is not indented")

    cond = _condition(first, refuse_zero, magnitude)
    return "\n".join(
        lines[: header_idx + 1]
        + [f"{indent}if {cond}:", f"{indent}{indent}return 1 // 0"]
        + lines[header_idx + 1:]
    )


def define_guarded_function(db, source, refuse_zero=True, magnitude=DEFAULT_MAGNITUDE):
    """Wrap, then hand to mpedb — whose compiler and probe corpus stay the authority."""
    return db.define_function(wrap_int_guard(source, refuse_zero, magnitude))


def create_guarded_residual_lens(db, name, fwd_src, rex_src, inv_src,
                                 residual_type="any", refuse_zero=True,
                                 magnitude=DEFAULT_MAGNITUDE):
    """The whole recipe: guard forward and rex, define the inverse bare, register the lens.

    Returns the probe count from `create_residual_lens` — the engine's own verification,
    which this module never touches.
    """
    fwd_name, _ = define_guarded_function(db, fwd_src, refuse_zero, magnitude)
    rex_name, _ = define_guarded_function(db, rex_src, refuse_zero, magnitude)
    inv_name, _ = db.define_function(inv_src)
    return db.create_residual_lens(name, fwd_name, rex_name, inv_name, residual_type)
