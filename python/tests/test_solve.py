"""The contract: exact, or a refusal that names what stopped it."""

from fractions import Fraction

import pytest

import mpeqs


class TestExactness:
    def test_a_ratio_survives_the_round_trip(self):
        """float("85/2") is a ValueError; the point is to hand back a number."""
        assert mpeqs.solve({"solver": "arith", "answer": "(17/100)*250"}) == Fraction(85, 2)
        assert float(mpeqs.solve({"solver": "arith", "answer": "(17/100)*250"})) == 42.5

    def test_a_third_is_a_third(self):
        assert mpeqs.solve({"solver": "arith", "answer": "1/3"}) == Fraction(1, 3)
        assert mpeqs.solve({"solver": "arith", "answer": "1/3+1/3+1/3"}) == 1

    def test_ratios_written_as_ratios_do_not_drift(self):
        """0.1+0.2 != 0.3 in binary. Written as ratios it is exactly 3/10."""
        assert mpeqs.solve({"solver": "arith", "answer": "1/10+2/10"}) == Fraction(3, 10)

    def test_a_decimal_literal_is_read_as_a_float_first(self):
        """The documented limit, asserted so it cannot quietly change."""
        assert mpeqs.solve({"solver": "arith", "answer": "0.1+0.2"}) != Fraction(3, 10)


class TestRefusal:
    def test_division_by_zero_refuses_rather_than_returning_something(self):
        with pytest.raises(mpeqs.Refusal):
            mpeqs.solve({"solver": "arith", "answer": "1/0"})

    def test_a_wrong_name_with_right_slots_is_repaired_rather_than_refused(self):
        """Shape disposes over name, deliberately.

        A model that picks the wrong solver but fills the right slots has still
        described the problem correctly, and an ``answer`` key is arithmetic
        whatever the spec calls itself. Refusing that would waste a good spec.
        """
        assert mpeqs.solve({"solver": "not-a-solver", "answer": "1+1"}) == 2

    def test_repair_can_be_turned_off(self):
        with pytest.raises(mpeqs.Refusal):
            mpeqs.solve({"solver": "not-a-solver", "answer": "1+1"}, repair=False)

    def test_a_spec_with_no_usable_shape_refuses(self):
        with pytest.raises(mpeqs.Refusal):
            mpeqs.solve({"solver": "not-a-solver", "nonsense": True})

    def test_the_refusal_says_what_stopped_it(self):
        """A refusal is the routing signal, so it has to carry a reason."""
        with pytest.raises(mpeqs.Refusal) as excinfo:
            mpeqs.solve({"solver": "arith", "answer": "1/0"})
        assert str(excinfo.value).strip()


class TestSurface:
    def test_the_solvers_are_discoverable(self):
        names = mpeqs.solvers()
        assert "arith" in names
        assert names == sorted(names)

    def test_everything_exported_exists(self):
        for name in mpeqs.__all__:
            assert hasattr(mpeqs, name), name

    def test_it_imports_with_no_third_party_packages(self):
        """One wheel for every platform depends on this staying true."""
        import pathlib
        import sys

        here = pathlib.Path(mpeqs.__file__).parent
        stdlib = set(sys.stdlib_module_names)
        for source in here.glob("*.py"):
            for line in source.read_text().splitlines():
                if line.startswith(("import ", "from ")) and not line.startswith("from ."):
                    top = line.split()[1].split(".")[0]
                    assert top in stdlib or (here / f"{top}.py").exists(), f"{source.name}: {line}"
