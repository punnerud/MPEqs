#!/usr/bin/env python3
"""Islands, roads, and a route to the answer that never materialises the alternatives.

The pipeline has mapped every problem onto exactly ONE machine for forty phases. Phase 92
chained search into factor by hand and closed a real AIME problem exactly, and nothing has
been able to find such a chain since, because nothing looks for one.

The picture that fixes it is the user's: the quantities in a problem are ISLANDS, the
answer is a package that has to be DELIVERED, and the machines are the ROADS that
transform what you are carrying. Finding the solution is route-finding — and the routing
must be MPEE-shaped: the road matrix is generated cell by cell as the frontier asks for it
and the alternatives are never computed.

Two halves, both reusing what already exists rather than reinventing it:

  EXECUTION   relgraph's topological resolver already runs a system of named definitions
              exactly, with cycle, undefined-reference and number-completeness refusals
              that have perfect ground truth. What it could not do is let a MACHINE be an
              edge. Now a definition may be `@solver{slot=value, ...} report`, resolved in
              the same topological order, executed by solvers2.run2.
  DISCOVERY   when nobody supplies the chain, a bidirectional frontier over ISLAND TYPES
              finds one: forward from what the problem holds, backward from what it asks,
              meeting in the middle (phase 81's ping-pong, 6.9x at depth ten). Sources are
              masked exactly as formularoute does it — each island consumed once — and the
              frontier is deduplicated by type, so what is held in memory is a rim of
              types and never a matrix of pairs.

Self-tested before any model is allowed near it, including the phase 92 chain, which must
come out at 12.
"""
import json
import re
import sys
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from solvemap import answer_of  # noqa: E402
from solvers2 import compile_expr, eval_expr, run2  # noqa: E402

CALL = re.compile(r"^@(\w+)\{(.*)\}\s*(\w+)?\s*$", re.S)
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
BUDGET = 20000

# The road network: what each machine eats and what it leaves behind, in ISLAND types
# rather than unit dimensions. Written as data, so no solver is touched.
SOLVER_TYPES = {
    "search":        (("range",), "count"),
    "multisearch":   (("range",), "count"),
    "factor":        (("integer",), "count"),
    "primes":        (("range",), "count"),
    "partition":     (("integer",), "count"),
    "strcount":      (("word",), "count"),
    "logexp":        (("integer",), "count"),
    "datetime":      (("date", "date"), "count"),
    "statistics":    (("list",), "amount"),
    "finance":       (("amount",), "amount"),
    "convert":       (("amount",), "amount"),
    "arith":         (("amount",), "amount"),
    "iterate":       (("amount",), "fraction"),
    "probability":   (("range",), "fraction"),
    "approx":        (("fraction",), "fraction"),
    "sequence":      (("list",), "integer"),
    "equation":      (("equation",), "fraction"),
    "linear_system": (("equation",), "amount"),
    "quadratic":     (("equation",), "amount"),
    "crt":           (("list",), "integer"),
    "gcd_lcm":       (("list",), "integer"),
    "matrix":        (("matrix",), "amount"),
    "shape":         (("amount",), "amount"),
    "basearith":     (("digit_string",), "digit_string"),
    "roman":         (("integer",), "digit_string"),
    "checksum":      (("digit_string",), "count"),
    "modular":       (("integer",), "integer"),
    "polynomial":    (("list",), "amount"),
    "series":        (("amount",), "amount"),
    "combinatorics": (("integer",), "count"),
    "recurrence":    (("list",), "amount"),
    "kinematics":    (("amount",), "amount"),
    "inclusion_exclusion": (("list",), "count"),
    "formula":       (("list",), "amount"),
    "ratio_split":   (("amount",), "list"),
    "mixture":       (("list",), "amount"),
    "rate_work":     (("list",), "amount"),
    "digit_ops":     (("integer",), "count"),
    "interest":      (("amount",), "amount"),
    "base_convert":  (("integer",), "digit_string"),
    "geometry":      (("list",), "amount"),
}
# A count is an integer and an integer is an amount: the roads that widen without work.
WIDENS = {("count", "integer"), ("integer", "amount"), ("count", "amount"),
          ("fraction", "amount"), ("digit_string", "integer")}


# ------------------------------------------------------------------ execution

def parse_call(body):
    """`@solver{k=v, ...} report` -> (solver, {slots}, report) or None for arithmetic."""
    m = CALL.match(str(body).strip())
    if not m:
        return None
    solver, slots, report = m.group(1), m.group(2), m.group(3)
    spec = {"solver": solver}
    depth, cur, parts = 0, "", []
    for ch in slots:                        # split on commas outside brackets
        if ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    for part in filter(None, (p.strip() for p in parts)):
        if "=" not in part:
            spec.setdefault("conditions", []).append(part)
            continue
        k, _, v = part.partition("=")
        spec[k.strip()] = v.strip()
    if report:
        spec["report"] = report
    return spec


def coerce(spec):
    """Slot values arrive as strings; JSON-shaped ones become their objects."""
    out = {}
    for k, v in spec.items():
        if isinstance(v, str) and v[:1] in "[{":
            try:
                out[k] = json.loads(v)
                continue
            except json.JSONDecodeError:
                pass
        if isinstance(v, str) and re.fullmatch(r"-?\d+", v.strip()):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def solve_graph(defs, asked, problem=None):
    """relgraph's topological resolve, with machines allowed as edges.

    Arithmetic definitions behave exactly as before. A definition that names a machine is
    executed by the record, its slots first having every reference to another island
    replaced by that island's value — which is what makes a chain a chain.
    """
    if asked not in defs:
        return None, f"the asked node '{asked}' is never defined", {}
    values, visiting, trace = {}, set(), []

    def resolve(name):
        if name in values:
            return values[name]
        if name not in defs:
            raise KeyError(name)
        if name in visiting:
            raise RecursionError(name)
        visiting.add(name)
        body = str(defs[name])
        call = parse_call(body)
        if call:
            # Substitution must respect word boundaries and never touch a slot NAME:
            # a reference called "n" was rewriting the middle of "exponent_sum" and the
            # chain came out at 480 instead of 12.
            for ref in sorted(set(NAME.findall(body)), key=len, reverse=True):
                if ref in defs and ref != name:
                    val = resolve(ref)
                    call = {k: (re.sub(rf"\b{ref}\b", str(val), v)
                                if isinstance(v, str) and k != "solver" else v)
                            for k, v in call.items()}
            spec = coerce(call)
            res, why = run2(spec)
            if res is None:
                raise ValueError(f"{name}: {why}")
            got = answer_of(res, spec)
            trace.append({"node": name, "solver": spec.get("solver"), "value": got})
            values[name] = F(str(got)) if re.fullmatch(r"-?\d+(/\d+)?", str(got)) \
                else got
        else:
            expr = body
            for ref in sorted(set(NAME.findall(expr)), key=len, reverse=True):
                if ref in defs and ref != name:
                    expr = re.sub(rf"\b{ref}\b", f"({resolve(ref)})", expr)
            # Every name left in the expression must be a defined island; anything
            # else is an undefined reference, not a syntax error, and must say so.
            for ref in set(NAME.findall(expr)):
                raise KeyError(ref)
            if not re.fullmatch(r"[\d\s+*/().\-]+", expr):
                raise ValueError(f"{name} is not arithmetic and names no machine")
            # The sandbox from phase 102 rationalises literals whenever an expression
            # divides, so 20675 / 7 is a fraction and not a rounded double.
            code, _used = compile_expr(expr, set())
            values[name] = F(eval_expr(code, {}))
            trace.append({"node": name, "solver": None, "value": str(values[name])})
        visiting.discard(name)
        return values[name]

    try:
        answer = resolve(asked)
    except KeyError as e:
        return None, f"'{e.args[0]}' is referenced but never defined", {}
    except RecursionError as e:
        return None, f"'{e.args[0]}' defines itself", {}
    except ValueError as e:
        return None, str(e), {}
    return answer, "ok", {"trace": trace, "steps": len(trace)}


# ------------------------------------------------------------------ discovery

def widen(t):
    return {b for a, b in WIDENS if a == t} | {t}


def route(have, want, budget=BUDGET):
    """Bidirectional frontier over island TYPES; the road matrix is never built.

    Forward from the types the problem holds, backward from the type it asks for, the
    smaller rim expanding each round. A cell of the |types| x |types| matrix is computed
    only when a rim asks whether some road connects two types it is actually holding, so
    what lives in memory is two sets of types and the roads tried, never the product.
    """
    # Widening is FORWARD ONLY. A count is an integer, so holding a count means
    # holding an integer; wanting an integer does not mean wanting a count. Applying it
    # to both rims made them meet on the starting types and report an empty road.
    fwd = {t: [] for t in have}
    for t in list(fwd):
        for w in widen(t):
            fwd.setdefault(w, [])
    bwd = {want: []}
    cells = 0
    for _round in range(6):
        if set(fwd) & set(bwd):
            meet = sorted(set(fwd) & set(bwd))[0]
            return {"found": True, "meet": meet,
                    "road": fwd[meet] + list(reversed(bwd[meet])),
                    "cells_generated": cells,
                    "matrix_never_built": len(set(SOLVER_TYPES)) ** 2}
        forward = len(fwd) <= len(bwd)
        side, new = (fwd, {}) if forward else (bwd, {})
        for name, (consumes, produces) in SOLVER_TYPES.items():
            cells += 1
            if cells > budget:
                return {"found": False, "why": "budget", "cells_generated": cells}
            if forward:
                if all(c in side for c in consumes) and produces not in side:
                    src = max((side[c] for c in consumes), key=len, default=[])
                    for w in widen(produces):
                        new.setdefault(w, src + [name])
            else:
                if produces in side and any(c not in side for c in consumes):
                    for c in consumes:
                        new.setdefault(c, side[produces] + [name])
        if not new:
            break
        side.update(new)
    return {"found": False, "why": "no route", "cells_generated": cells,
            "matrix_never_built": len(set(SOLVER_TYPES)) ** 2}


# ------------------------------------------------------------------ self-test

CASES = [
    # Phase 92's chain, now expressed as a graph the record executes by itself.
    ({"n": "13",
      "s": "@search{domain={\"kind\": \"divisors_of_factorial\", \"k\": 13}, "
           "conditions=[{\"op\": \"quotient_is_square\", \"arg\": 6227020800}], "
           "aggregate=sum} value",
      "ans": "@factor{n=s, report=exponent_sum} exponent_sum"}, "ans", "12"),
    # Two machines and an arithmetic edge between them.
    ({"d": "@datetime{kind=days_between, from=1970-01-01, to=2026-08-10} value",
      "weeks": "d / 7"}, "weeks", "20675/7"),
    # A machine reading a value another machine produced.
    ({"total": "@factor{n=720720, report=divisor_sum} divisor_sum",
      "root": "@digit_ops{n=total} digit_sum"}, "root", "36"),
    # Pure arithmetic still behaves exactly as relgraph always did.
    ({"a": "48", "b": "a / 2", "t": "a + b"}, "t", "72"),
]
REFUSALS = [
    ({"a": "b + 1", "b": "a + 1"}, "a", "defines itself"),
    ({"a": "c + 1"}, "a", "never defined"),
    ({"a": "@nosuch{x=1} value"}, "a", "unknown solver"),
    # "the answer" now refuses as an undefined REFERENCE rather than as bad syntax,
    # which is the more useful of the two messages: it names what is missing.
    ({"a": "the answer"}, "a", "never defined"),
    ({"a": "1"}, "zz", "never defined"),
]


def main(out="data/custom/islands.json"):
    passed = 0
    rows = []
    for defs, asked, want in CASES:
        got, why, info = solve_graph(defs, asked)
        ok = str(got) == want
        passed += ok
        rows.append({"asked": asked, "got": str(got), "want": want, "why": why,
                     "steps": info.get("steps")})
        print(f"{'ok  ' if ok else 'FAIL'} {str(got):<14} want {want:<14} "
              f"{info.get('steps', '-')} steps  {why[:30]}")

    ref_ok = 0
    for defs, asked, needle in REFUSALS:
        got, why, _i = solve_graph(defs, asked)
        good = got is None and needle in why
        ref_ok += good
        print(f"{'refused' if good else 'MISSED '} [{why[:56]}]")

    print("\nrouting, forward from what is held to what is asked:")
    routes = []
    for have, want in ((["range"], "count"), (["integer"], "digit_string"),
                       (["date"], "amount"), (["word"], "count"),
                       (["matrix"], "fraction")):
        r = route(have, want)
        routes.append({"have": have, "want": want, **r})
        print(f"  {str(have):<12} -> {want:<14} "
              f"{'via ' + ' -> '.join(r['road'][:3]) if r.get('found') else r.get('why')}"
              f"   cells {r['cells_generated']} of {r.get('matrix_never_built')} "
              f"never built")

    found = sum(1 for r in routes if r.get("found"))
    cells = sum(r["cells_generated"] for r in routes)
    print(f"\n{passed}/{len(CASES)} graphs exact, {ref_ok}/{len(REFUSALS)} refusals named")
    print(f"{found}/{len(routes)} routes found; {cells} matrix cells generated in total "
          f"against {len(SOLVER_TYPES) ** 2} that were never built")
    print("\nA machine can now be an edge, so a chain is just a graph with two of them,")
    print("and the record runs it in topological order with every refusal it already")
    print("had. The route finder holds two rims of types and asks the road network for")
    print("one cell at a time — the alternatives are never computed.")
    summary = {"cases": len(CASES), "passed": passed, "refusals": len(REFUSALS),
               "refusals_named": ref_ok, "routes": routes, "routes_found": found,
               "cells_generated": cells,
               "matrix_never_built": len(SOLVER_TYPES) ** 2,
               "solvers_typed": len(SOLVER_TYPES), "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
