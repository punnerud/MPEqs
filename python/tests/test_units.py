"""Translation between denominations, and the failures that motivated it.

The four conversions asserted below are not invented cases. They are the exact
questions a model with exact arithmetic available got wrong, taken from a measured
run, and every one of its expressions was evaluated flawlessly on the way to a
wrong answer. That is the point being tested: arithmetic being exact does not make
an expression mean anything.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise

import pytest

from mpeqs import Refusal
from mpeqs import units as u


def test_the_graph_agrees_with_itself():
    """Where two paths reach the same pair, they must give the same factor.

    A conversion table entered by hand always has a disagreement somewhere, and
    one nobody has looked for is the worst kind: silent, and load-bearing for
    everything derived through it.
    """
    assert u.audit() == []


@pytest.mark.parametrize(
    ("value", "source", "target", "expected"),
    [
        # Every one of these is a measured failure. The model's own expressions,
        # each evaluated exactly, were 604800/161, (8*7)*7*3600, and 86400*378.
        (23, "weeks", "seconds", 13_910_400),
        (54, "weeks", "seconds", 32_659_200),
        (8, "weeks", "seconds", 4_838_400),
        (37, "weeks", "seconds", 22_377_600),
        # The original bug this whole thread started from: "10080 minutes in a
        # fortnight" is a week, and 336, 840 and 1680 were all offered too.
        (1, "fortnight", "minutes", 20_160),
    ],
)
def test_the_conversions_the_model_got_wrong(value, source, target, expected):
    assert u.convert(value, source, target) == expected


def test_a_kilogram_of_feathers_and_a_kilogram_of_lead():
    """The substance is not a dimension, so it cannot affect a comparison of mass.

    This is not special-cased and it is not a trick. Both sides normalise to the
    same base unit and the substance is discarded, which is why a unit system is
    immune to a trap that catches a language model by association.
    """
    assert u.compare("1 kg of feathers", "1 kg of lead") == 0
    assert u.compare("1 kilogram of feathers", "1000 g of lead") == 0
    # And it still compares when the quantities genuinely differ.
    assert u.compare("1 kg of feathers", "1 lb of lead") == 1
    assert u.compare("1 ounce of gold", "1 stone of straw") == -1


def test_the_substance_is_read_and_then_ignored():
    quantity = u.parse("2.5 kg of feathers")
    assert quantity.substance == "feathers"
    assert quantity.unit == "kilogram"
    assert quantity.value == Fraction(5, 2)
    # Ignored where it matters: the conversion does not consult it.
    assert quantity.to("gram") == 2500


def test_conversions_are_exact_rather_than_rounded():
    """A path is a product of exact ratios, so the answer is exact however long it is."""
    assert u.convert(1, "mile", "metre") == Fraction(201168, 125)
    assert float(u.convert(1, "mile", "metre")) == 1609.344
    assert u.convert(1, "inch", "millimetre") == Fraction(254, 10)
    # A foot is exactly 3048/10000 metres, and six hops do not erode it.
    assert u.convert(1, "foot", "metre") == Fraction(3048, 10000)
    assert u.convert(1, "gibibyte", "byte") == 1_073_741_824


def test_a_round_trip_returns_exactly_what_went_in():
    for source, target in [("mile", "metre"), ("pound", "gram"), ("week", "second"),
                           ("gallon", "millilitre"), ("gibibyte", "bit")]:
        there = u.factor(source, target)
        back = u.factor(target, source)
        assert there * back == 1, f"{source}<->{target} does not round-trip"


def test_different_dimensions_are_refused_rather_than_forced():
    # The useful answer to "how many metres in a kilogram" is that the question
    # does not have one.
    with pytest.raises(Refusal, match="different dimensions"):
        u.convert(1, "kilogram", "metre")
    with pytest.raises(Refusal, match="different dimensions"):
        u.compare("1 kg of feathers", "1 metre of rope")


def test_an_unknown_unit_is_named_rather_than_guessed():
    with pytest.raises(Refusal, match="furlong"):
        u.convert(1, "furlong", "metre")


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("kg", "kilogram"), ("KG", "kilogram"), ("kilograms", "kilogram"),
        ("secs", "second"), ("mins", "minute"), ("hrs", "hour"),
        ("feet", "foot"), ("inches", "inch"), ("meters", "metre"),
        ("miles", "mile"), ("lbs", "pound"), ("MiB", "mebibyte"),
    ],
)
def test_the_names_people_actually_write(written, expected):
    assert u.canonical(written) == expected


def test_the_derivation_can_be_shown_rather_than_asserted():
    """A person has to be able to check it, which means seeing the hops."""
    walked = u.path("fortnight", "minute")
    assert walked[0] == "fortnight"
    assert walked[-1] == "minute"
    assert "day" in walked
    # Each consecutive pair is a real edge, so the path is not decorative.
    for a, b in pairwise(walked):
        assert any(n == b for n, _ in u.GRAPH[a])


def test_adding_a_unit_costs_one_edge_not_a_row_and_a_column():
    """The reason for a graph rather than a table.

    36 units as a table is 1296 entries, all but 35 of them redundant and every
    one a chance to mistype a digit. Here it is one fact per edge.
    """
    assert len(u.units()) > 30
    assert len(u.EDGES) < len(u.units()) * 2


def test_identity_is_free_and_exact():
    assert u.factor("kg", "kilograms") == 1
    assert u.convert(Fraction(7, 3), "metre", "metre") == Fraction(7, 3)
