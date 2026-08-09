#!/usr/bin/env python3
"""The guard module's properties, pinned. Text-level tests need no database; the last one
uses mpedb itself, because 'the wrapped source still compiles and registers' is mpedb's
verdict to give, not this module's to assume."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rretl_guard import (DEFAULT_MAGNITUDE, GuardError,  # noqa: E402
                         create_guarded_residual_lens, wrap_int_guard)

OFFSET = "def off_fwd(x):\n    return x - 1000\n"


class WrapText(unittest.TestCase):
    def test_wraps_and_preserves_body(self):
        w = wrap_int_guard(OFFSET)
        self.assertIn(f"if x % 1 != 0 or x == 0 or x > {DEFAULT_MAGNITUDE} "
                      f"or x < -{DEFAULT_MAGNITUDE}:", w)
        self.assertIn("        return 1 // 0", w)
        self.assertTrue(w.rstrip().endswith("return x - 1000"))

    def test_guards_first_parameter_only(self):
        w = wrap_int_guard("def inv(y, r):\n    return y + r\n")
        self.assertIn("if y % 1 != 0", w)
        self.assertNotIn("r % 1", w)

    def test_respects_author_indent(self):
        w = wrap_int_guard("def f(x):\n  return x\n")
        self.assertIn("\n  if x % 1 != 0", w)
        self.assertIn("\n    return 1 // 0", w)

    def test_magnitude_off(self):
        w = wrap_int_guard(OFFSET, magnitude=None)
        self.assertIn("if x % 1 != 0 or x == 0:", w)
        self.assertNotIn(">", w.split(":")[1])

    def test_refuses_the_unguardable(self):
        for bad in ("x = 1\n", "def f():\n    return 1\n", "def f(x):\n"):
            with self.assertRaises(GuardError):
                wrap_int_guard(bad)
        with self.assertRaises(GuardError):
            wrap_int_guard(OFFSET, magnitude=0)


class AgainstTheEngine(unittest.TestCase):
    """The motivating case, end to end: offset refuses unguarded, registers guarded."""

    def test_offset_pair(self):
        sys.path.insert(0, "/tmp/pymod")
        try:
            import mpedb
        except ImportError:
            # The engine half needs the locally built mpedb-py (see PYSPELL-RRETL.md §1).
            # Skipping is honest here: the text-level properties above still ran, and a
            # missing build must not read as the guard having been verified against the
            # engine when it was not.
            self.skipTest("mpedb-py not built at /tmp/pymod")
        for f in ("/tmp/guardtest.mpedb", "/tmp/guardtest.mpedb-lock"):
            Path(f).unlink(missing_ok=True)
        Path("/tmp/guardtest.toml").write_text(
            '[database]\npath = "/tmp/guardtest.mpedb"\nsize_mb = 32\nmax_readers = 8\n')
        db = mpedb.Database("/tmp/guardtest.toml")

        fwd = "def og_fwd(x):\n    return x - 1000\n"
        rex = "def og_rex(x):\n    return 0\n"
        inv = "def og_inv(y, r):\n    return y + 1000\n"

        # Unguarded: the engine's verifier must refuse it (the Float(1.3e17) trap).
        for s in (fwd.replace("og_", "raw_"), rex.replace("og_", "raw_"),
                  inv.replace("og_", "raw_")):
            db.define_function(s)
        with self.assertRaises(Exception):
            db.create_residual_lens("rawoff", "raw_fwd", "raw_rex", "raw_inv", "any")

        # Guarded through this module: registers, with the engine still the judge.
        probes = create_guarded_residual_lens(db, "guardoff", fwd, rex, inv)
        self.assertGreater(probes, 0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
