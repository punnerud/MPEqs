"""Translation between denominations, as a sparse graph rather than a table.

``solve`` is dimensionless: ``1 kg`` is a syntax error to it, and ``14*60`` is
840 whether the question asked for minutes or for seconds. That gap is not
theoretical. Measured on generated unit questions, a model with exact arithmetic
available got **none** of them right, and every failure was dimensional rather
than arithmetic:

    "how many seconds in 23 weeks"  ->  604800 / 161
                                        (seconds in a week, over DAYS)
    "how many seconds in 8 weeks"   ->  (8*7)*7, then *3600
                                        (days multiplied by 7 again, then
                                         seconds-per-hour used as per-day)

Each of those expressions was evaluated flawlessly. The expressions were
nonsense, and an exact evaluator cannot see that by construction -- it is handed
numbers, and numbers are all it gets.

**The model.** Units are nodes; a known conversion is an edge carrying an exact
factor. Translating between two units is a path, and its factor is the product
along that path. So the N x N table of every conversion is never built: it is
implied by a sparse graph with one edge per fact somebody actually knows, and any
cell is derived on demand by walking it. Adding a unit costs one edge, not a row
and a column.

**Exactness.** Factors are ``Fraction``s, so 1 kg is exactly 1000 g and a
kilometre is exactly 1000 metres -- and, since a path is a product of exact
ratios, the answer is exact however many hops it took. A foot is exactly
3048/10000 metres, not 0.3048.

**Consistency is checked, not assumed.** Where two paths connect the same pair,
they must give the same factor. ``audit()`` walks every such pair and reports the
disagreements; a table entered by hand always has some, and a graph that has been
audited is the only kind worth deriving from.

**Comparison, not only conversion.** "What weighs more, a kilogram of feathers or
a kilogram of lead" is not an exception to this module -- it is the ordinary case.
Normalise both sides to a base unit and the substance falls away, because a
substance is not a dimension. 1 kg = 1 kg, mechanically, on a question a language
model gets wrong by association.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from fractions import Fraction

from solvers import Refusal

# One edge per fact, as (from, to, factor): one `from` is `factor` of `to`. The
# reverse edge is implied, so nothing is written twice and nothing can disagree
# with itself.
#
# Deliberately not a conversion table. A table of n units is n^2 entries, all but
# n-1 of them redundant and every one of them a chance to typo a digit.
EDGES: list[tuple[str, str, Fraction]] = [
    # Time
    ("minute", "second", Fraction(60)),
    ("hour", "minute", Fraction(60)),
    ("day", "hour", Fraction(24)),
    ("week", "day", Fraction(7)),
    ("fortnight", "week", Fraction(2)),
    # A choice, not a fact: the mean tropical year is 365.2425 days, and a
    # calendar year is 365 or 366. 365 is what these questions mean, and saying so
    # here is better than a caller discovering it from a wrong total.
    ("year", "day", Fraction(365)),
    ("millisecond", "second", Fraction(1, 1000)),
    # Mass
    ("gram", "milligram", Fraction(1000)),
    ("kilogram", "gram", Fraction(1000)),
    ("tonne", "kilogram", Fraction(1000)),
    ("pound", "gram", Fraction(45359237, 100000)),
    ("ounce", "pound", Fraction(1, 16)),
    ("stone", "pound", Fraction(14)),
    # Length
    ("metre", "centimetre", Fraction(100)),
    ("centimetre", "millimetre", Fraction(10)),
    ("kilometre", "metre", Fraction(1000)),
    ("inch", "millimetre", Fraction(254, 10)),
    ("foot", "inch", Fraction(12)),
    ("yard", "foot", Fraction(3)),
    ("mile", "yard", Fraction(1760)),
    ("nautical_mile", "metre", Fraction(1852)),
    # Volume
    ("litre", "millilitre", Fraction(1000)),
    ("cubic_metre", "litre", Fraction(1000)),
    # The imperial gallon. The US gallon is 3.785411784 L and is a different unit
    # with the same name, which is why it is not silently one of these.
    ("gallon", "litre", Fraction(454609, 100000)),
    ("pint", "gallon", Fraction(1, 8)),
    # Digital
    ("byte", "bit", Fraction(8)),
    ("kibibyte", "byte", Fraction(1024)),
    ("mebibyte", "kibibyte", Fraction(1024)),
    ("gibibyte", "mebibyte", Fraction(1024)),
    ("kilobyte", "byte", Fraction(1000)),
    ("megabyte", "kilobyte", Fraction(1000)),
]

# What people write, mapped to the node name. Plurals are handled by the parser.
ALIASES = {
    "s": "second", "sec": "second", "secs": "second",
    "min": "minute", "mins": "minute",
    "h": "hour", "hr": "hour", "hrs": "hour",
    "d": "day", "ms": "millisecond",
    "yr": "year", "yrs": "year",
    "g": "gram", "kg": "kilogram", "mg": "milligram",
    "t": "tonne", "lb": "pound", "lbs": "pound", "oz": "ounce",
    "m": "metre", "meter": "metre", "meters": "metre",
    "cm": "centimetre", "centimeter": "centimetre",
    "mm": "millimetre", "millimeter": "millimetre",
    "km": "kilometre", "kilometer": "kilometre",
    "in": "inch", "ft": "foot", "feet": "foot", "yd": "yard", "mi": "mile",
    "l": "litre", "liter": "litre", "ml": "millilitre", "milliliter": "millilitre",
    "gal": "gallon",
    "b": "bit", "bits": "bit", "kb": "kilobyte", "mb": "megabyte",
    "kib": "kibibyte", "mib": "mebibyte", "gib": "gibibyte",
}

# Irregular plurals the "drop a trailing s" rule gets wrong.
IRREGULAR = {"feet": "foot", "inches": "inch", "ounces": "ounce", "stones": "stone"}


def _graph() -> dict[str, list[tuple[str, Fraction]]]:
    out: dict[str, list[tuple[str, Fraction]]] = {}
    for a, b, factor in EDGES:
        if factor <= 0:
            raise Refusal(f"conversion {a}->{b} has a non-positive factor")
        out.setdefault(a, []).append((b, factor))
        out.setdefault(b, []).append((a, 1 / factor))
    return out


GRAPH = _graph()


def canonical(name: str) -> str:
    """The node name for what somebody wrote, or a Refusal naming the unit."""
    text = str(name or "").strip().lower().replace(" ", "_")
    text = text.rstrip(".")
    for candidate in (text, IRREGULAR.get(text, ""), ALIASES.get(text, "")):
        if candidate in GRAPH:
            return candidate
    # Plurals, after the alias table so "mins" resolves before "min" + s.
    if text.endswith("s"):
        singular = text[:-1]
        for candidate in (singular, ALIASES.get(singular, "")):
            if candidate in GRAPH:
                return candidate
    raise Refusal(f"unknown unit {name!r}")


def factor(source: str, target: str) -> Fraction:
    """How many ``target`` in one ``source``, exactly.

    Breadth-first, so the answer comes from the fewest hops available -- which
    keeps the derivation short enough for a person to check by hand. The value is
    identical along any path in an audited graph; ``audit`` is what makes that
    claim rather than a hope.
    """
    start, end = canonical(source), canonical(target)
    if start == end:
        return Fraction(1)

    seen = {start}
    queue = deque([(start, Fraction(1))])
    while queue:
        node, acc = queue.popleft()
        for neighbour, edge in GRAPH[node]:
            if neighbour in seen:
                continue
            carried = acc * edge
            if neighbour == end:
                return carried
            seen.add(neighbour)
            queue.append((neighbour, carried))
    raise Refusal(f"no conversion from {start!r} to {end!r}: different dimensions")


def convert(value, source: str, target: str) -> Fraction:
    """``value`` in ``source`` units, expressed in ``target`` units, exactly."""
    return Fraction(value) * factor(source, target)


def path(source: str, target: str) -> list[str]:
    """The units walked through, so a derivation can be shown rather than asserted."""
    start, end = canonical(source), canonical(target)
    if start == end:
        return [start]
    previous = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbour, _ in GRAPH[node]:
            if neighbour in previous:
                continue
            previous[neighbour] = node
            if neighbour == end:
                chain, at = [], neighbour
                while at is not None:
                    chain.append(at)
                    at = previous[at]
                return list(reversed(chain))
            queue.append(neighbour)
    raise Refusal(f"no conversion from {start!r} to {end!r}: different dimensions")


# A quantity written out: "1 kg", "1 kg of feathers", "2.5 miles".
#
# The substance is captured and then ignored, which is the entire trick of the
# feathers-and-lead question. A substance is not a dimension, so it cannot affect
# a comparison of masses -- and a system that drops it is immune to a trap that
# catches a language model by association.
QUANTITY = re.compile(
    r"(?P<value>-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)\s*"
    r"(?P<unit>[A-Za-z_]+)"
    r"(?:\s+(?:of|av)\s+(?P<substance>[A-Za-z_ ]+))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Quantity:
    value: Fraction
    unit: str
    substance: str = ""

    def to(self, target: str) -> Fraction:
        return convert(self.value, self.unit, target)

    def __str__(self) -> str:
        of = f" of {self.substance}" if self.substance else ""
        return f"{self.value} {self.unit}{of}"


def parse(text: str) -> Quantity:
    """Read one written quantity, or refuse."""
    match = QUANTITY.search(str(text or ""))
    if not match:
        raise Refusal(f"no quantity found in {text!r}")
    raw = match.group("value").replace(" ", "")
    try:
        value = Fraction(raw)
    except (ValueError, ZeroDivisionError) as exc:
        raise Refusal(f"cannot read the number in {text!r}") from exc
    return Quantity(
        value=value,
        unit=canonical(match.group("unit")),
        substance=(match.group("substance") or "").strip(),
    )


def compare(left: str, right: str) -> int:
    """-1, 0 or 1 for whether the left quantity is less, equal or greater.

    "A kilogram of feathers or a kilogram of lead" resolves to 0 here, and not
    because the case was special-cased: both sides normalise to the same base
    unit and the substances are discarded, because a substance is not a
    dimension. The trap is in the association, and this never sees it.
    """
    a, b = parse(left), parse(right)
    try:
        converted = b.to(a.unit)
    except Refusal as exc:
        raise Refusal(f"cannot compare {a} with {b}: {exc}") from exc
    if a.value == converted:
        return 0
    return 1 if a.value > converted else -1


def audit() -> list[str]:
    """Every pair reachable two ways, checked for agreeing.

    A conversion table entered by hand always has a disagreement somewhere, and
    one that has never been looked for is the worst kind: silent, and load-bearing
    for everything derived through it. Empty output is the claim worth making.
    """
    complaints: list[str] = []
    for a, b, stated in EDGES:
        # Drop this edge and see whether the rest of the graph still connects the
        # pair. If it does, the two answers must agree.
        others = [e for e in EDGES if not (e[0] == a and e[1] == b)]
        saved = dict(GRAPH)
        try:
            globals()["GRAPH"] = _rebuild(others)
            try:
                derived = factor(a, b)
            except Refusal:
                continue  # Only one way there; nothing to disagree with.
            if derived != stated:
                complaints.append(
                    f"{a}->{b}: written as {stated}, but the rest of the graph "
                    f"derives {derived}")
        finally:
            globals()["GRAPH"] = saved
    return complaints


def _rebuild(edges) -> dict[str, list[tuple[str, Fraction]]]:
    out: dict[str, list[tuple[str, Fraction]]] = {}
    for a, b, f in edges:
        out.setdefault(a, []).append((b, f))
        out.setdefault(b, []).append((a, 1 / f))
    return out


def units() -> list[str]:
    """Every unit this build knows, by node name."""
    return sorted(GRAPH)
