#!/usr/bin/env python3
"""Units as a routed graph: dimensional control, exact conversion, compositional coverage.

The proposal, implemented literally. Units are nodes; each stored conversion is an edge with
an exact rational factor; converting is ROUTING — find a path, multiply the factors along it.
A compound unit like miles per second is two routes at once, numerator and denominator, so
"3 miles per second in km/h" is the mile-to-km path over the second-to-hour path. This is the
MPEE shape on the smallest real graph in the project, in the product semiring — (min,+) in
log space — and reversibility is structural: the reverse route is exactly the reciprocal,
verified below on every pair rather than assumed.

Two measurable claims ride on it:

  COMPOSITION IS THE COMPRESSION. Each stored edge composes with every other, so coverage
  grows quadratically per dimension and multiplicatively across compounds: a handful of
  conversions someone once wrote down covers thousands a model would otherwise be asked to
  know. Counted exactly, not estimated.

  DIMENSIONS ARE A CONTROL. A plan step that adds miles to hours is wrong before any
  arithmetic happens. Quantities carry their dimension vector; addition requires identical
  vectors; multiplication adds them. This is the check the word-problem plans of phase 51
  were missing — it cannot say a plan is right, but it refuses a class of wrong ones for
  free, which is the same asymmetry as every verifier in this thread.

Everything is Fraction-exact and model-free. The model's only future role here is naming the
units in a text — the graph does not need it for anything else.
"""
import json
import sys
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path

# The stored knowledge: exact definitional factors, one edge each. Everything else is routed.
EDGES = [
    ("mile", "km", F(1609344, 1000000)), ("km", "m", F(1000)), ("m", "cm", F(100)),
    ("cm", "mm", F(10)), ("inch", "cm", F(254, 100)), ("foot", "inch", F(12)),
    ("yard", "foot", F(3)),
    ("hour", "minute", F(60)), ("minute", "second", F(60)), ("day", "hour", F(24)),
    ("week", "day", F(7)),
    ("kg", "g", F(1000)), ("pound", "kg", F(45359237, 100000000)),
    ("ounce", "pound", F(1, 16)),
    ("gallon", "litre", F(3785411784, 1000000000)), ("litre", "ml", F(1000)),
]

DIM = {u: d for d, us in {
    "length": ["mile", "km", "m", "cm", "mm", "inch", "foot", "yard"],
    "time": ["hour", "minute", "second", "day", "week"],
    "mass": ["kg", "g", "pound", "ounce"],
    "volume": ["gallon", "litre", "ml"],
}.items() for u in us}


def build():
    graph = {}
    for a, b, f in EDGES:
        graph.setdefault(a, {})[b] = f
        graph.setdefault(b, {})[a] = 1 / f
    return graph


def route(graph, src, dst):
    """BFS over the conversion graph; the factor is the product along the path."""
    if src == dst:
        return F(1), [src]
    seen, frontier = {src: (F(1), [src])}, [src]
    while frontier:
        nxt = []
        for u in frontier:
            f_u, path = seen[u]
            for v, f in graph[u].items():
                if v not in seen:
                    seen[v] = (f_u * f, path + [v])
                    if v == dst:
                        return seen[v]
                    nxt.append(v)
        frontier = nxt
    return None, None


def convert_rate(graph, value, num_src, den_src, num_dst, den_dst):
    """value num_src/den_src expressed in num_dst/den_dst: two routes, one ratio."""
    fn, pn = route(graph, num_src, num_dst)
    fd, pd = route(graph, den_src, den_dst)
    if fn is None or fd is None:
        return None, None
    return value * fn / fd, (pn, pd)


class DimError(ValueError):
    """Refused before any arithmetic: the dimensions do not agree."""


class Q:
    """A quantity that carries its dimensions and refuses ill-typed arithmetic."""

    def __init__(self, v, dims):
        self.v, self.dims = F(v), dict(dims)

    def __add__(self, o):
        if self.dims != o.dims:
            raise DimError(f"cannot add {self.dims} to {o.dims}")
        return Q(self.v + o.v, self.dims)

    __sub__ = __add__

    def __mul__(self, o):
        d = dict(self.dims)
        for k, e in o.dims.items():
            d[k] = d.get(k, 0) + e
        return Q(self.v * o.v, {k: e for k, e in d.items() if e})

    def __truediv__(self, o):
        d = dict(self.dims)
        for k, e in o.dims.items():
            d[k] = d.get(k, 0) - e
        return Q(self.v / o.v, {k: e for k, e in d.items() if e})


def main(out="data/custom/unitroute.json"):
    graph = build()
    units = sorted(graph)
    print(f"{len(EDGES)} stored conversions over {len(units)} units in "
          f"{len(set(DIM.values()))} dimensions\n")

    # Reversibility, verified on every ordered pair rather than trusted.
    pairs = rev_ok = 0
    per_dim = {}
    for a, b in combinations(units, 2):
        if DIM[a] != DIM[b]:
            continue
        f1, _ = route(graph, a, b)
        f2, _ = route(graph, b, a)
        pairs += 2
        rev_ok += 2 * (f1 is not None and f1 * f2 == 1)
        per_dim[DIM[a]] = per_dim.get(DIM[a], 0) + 2
    print(f"simple conversions routed: {pairs} ordered pairs from {len(EDGES)} edges "
          f"({pairs / len(EDGES):.1f}x), reverse exact on {rev_ok}/{pairs}")

    # The compositional coverage: rates, per dimension pair.
    length_u = [u for u in units if DIM[u] == "length"]
    time_u = [u for u in units if DIM[u] == "time"]
    rate_pairs = (len(length_u) * len(time_u)) ** 2 - len(length_u) * len(time_u)
    total_compound = 0
    dims = sorted(set(DIM.values()))
    for da, db in [(x, y) for x in dims for y in dims if x != y]:
        na = sum(1 for u in units if DIM[u] == da)
        nb = sum(1 for u in units if DIM[u] == db)
        total_compound += (na * nb) * (na * nb - 1)
    print(f"speed units alone: {len(length_u) * len(time_u)} distinct, "
          f"{rate_pairs:,} ordered conversions among them")
    print(f"all two-dimension rates: {total_compound:,} conversions from the same "
          f"{len(EDGES)} edges ({total_compound // len(EDGES):,}x)")

    # The example, exactly: 3 miles per second in km/h.
    v, (pn, pd) = convert_rate(graph, F(3), "mile", "second", "km", "hour")
    print(f"\n3 mile/second = {v} km/hour = {float(v):,.4f}")
    print(f"  numerator route  : {' -> '.join(pn)}")
    print(f"  denominator route: {' -> '.join(pd)}")
    checks = [
        (F(3), "mile", "second", "km", "hour", F(3) * F(1609344, 1000000) * 3600),
        (F(60), "mile", "hour", "m", "second", F(60) * F(1609344, 1000) / 3600),
        (F(5, 2), "pound", "gallon", "g", "litre",
         F(5, 2) * F(45359237, 100000) / F(3785411784, 1000000000)),
        (F(12), "yard", "minute", "cm", "second", F(12) * 3 * 12 * F(254, 100) / 60),
    ]
    conv_ok = 0
    for val, ns, ds, nd, dd, want in checks:
        got, _ = convert_rate(graph, val, ns, ds, nd, dd)
        conv_ok += got == want
        print(f"  {val} {ns}/{ds} -> {got} {nd}/{dd}  "
              f"{'exact' if got == want else f'WRONG (want {want})'}")

    # The control: dimensioned arithmetic refuses the ill-typed step before computing it.
    mile = Q(1, {"length": 1})
    hour = Q(1, {"time": 1})
    speed = Q(3, {"length": 1, "time": -1})
    dim_ok = dim_refused = 0
    dim_ok += ((speed * hour).dims == {"length": 1})            # rate x time = distance
    dim_ok += ((mile / hour).dims == speed.dims)                # distance / time = rate
    try:
        _ = mile + hour
    except DimError:
        dim_refused += 1
    try:
        _ = speed + mile
    except DimError:
        dim_refused += 1
    print(f"\ndimension control: {dim_ok}/2 well-typed steps accepted, "
          f"{dim_refused}/2 ill-typed steps refused before arithmetic")

    print("\nSixteen written-down facts route thousands of conversions, every one exact and")
    print("every one reversible by construction. The graph needs the model for one thing")
    print("only: reading which units a sentence is talking about.")
    summary = {"edges": len(EDGES), "units": len(units), "simple_pairs": pairs,
               "reverse_exact": rev_ok, "speed_conversions": rate_pairs,
               "compound_conversions": total_compound, "example_kmh": float(v),
               "rate_checks_exact": conv_ok, "dim_accepted": dim_ok,
               "dim_refused": dim_refused}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
