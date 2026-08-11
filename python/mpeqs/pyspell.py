"""Run code a model wrote, with the property it claims about it actually checked.

``mpeqs.solve`` evaluates *expressions*. That is the right tool most of the time
and it is safe by restriction: the source is parsed to an AST, walked against a
whitelist, and evaluated with ``__builtins__`` emptied, so nothing outside
arithmetic can run. Nothing is executed that was not understood first.

What it cannot express is a *procedure*. "How many steps does the Collatz
sequence take from 27" needs a loop, an accumulator and a branch, and there is no
expression for it. The usual answer is to let the model emit Python and exec it,
which trades a whitelist for a sandbox and hope.

This module is the third option. PySpell is mpedb's stored-function language: a
small deterministic subset of Python -- no imports, no clock, no randomness, no
file or network I/O, a fixed instruction budget so a runaway loop fails
identically everywhere -- compiled at define time and stored by content hash. A
model can write a loop. It still cannot open a socket.

And when the model claims a *relationship* between two functions, that claim is
checked against a probe corpus rather than believed:

    >>> import mpeqs.pyspell as spell
    >>> spell.check_bijective(          # doctest: +SKIP
    ...     "def dbl(x):\\n    return x * 2\\n",
    ...     "def hlv(x):\\n    return x // 2\\n")
    Traceback (most recent call last):
    mpeqs.Refusal: ... forward(Float(2.2250738585072014e-308)) ...

That refusal is not pedantry and it is not a bug. ``x*2`` followed by ``x//2`` is
genuinely not bijective over floats -- a denormal floors to zero -- and the
verifier says so with the input that breaks it. Guard the domain to integers and
the same pair is accepted. It caught the author of this docstring on the first
try, which is the most useful thing a verifier can do.

**Optional.** mpedb ships as a compiled wheel and needs CPython 3.12 or newer,
while mpeqs itself is pure Python from 3.10. So this is an extra:

    pip install mpeqs[pyspell]

Without it ``available()`` is False and every entry point raises ``Refusal``
rather than failing at import. Nothing else in mpeqs changes.
"""

from __future__ import annotations

import re
import tempfile
import threading
from pathlib import Path
from typing import Any

from solvers import Refusal

# One `def name(args):` per source, which is what PySpell accepts -- the name and
# arity come from the definition itself.
_DEF = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)

_CONFIG = """\
[database]
path = "{path}"
size_mb = 64
max_readers = 8
"""


def available() -> bool:
    """Is the optional mpedb backend installed and importable?"""
    try:
        import mpedb  # noqa: F401
    except Exception:  # noqa: BLE001
        # Deliberately broad. mpedb is a compiled extension, so a failed import is
        # not only ImportError: a wheel built for another ABI, or one whose shared
        # library will not load, raises something else entirely. Every one of them
        # means the same thing here -- the extra is not usable.
        return False
    return True


def _require():
    try:
        import mpedb
    except Exception as exc:  # broad for the reason given in available()
        raise Refusal(
            "PySpell needs the optional mpedb backend: pip install mpeqs[pyspell] "
            "(CPython 3.12+; Linux x86-64/aarch64/armv7, macOS arm64, Windows x86-64)"
        ) from exc
    return mpedb


def function_name(source: str) -> str:
    """The name PySpell will register, taken from the definition itself."""
    found = _DEF.findall(source)
    if len(found) != 1:
        raise Refusal(
            f"PySpell wants exactly one 'def' per source, found {len(found)}")
    return found[0]


class Spell:
    """A database of defined functions.

    Holding one open is cheaper than opening one per call, and functions defined
    on it stay available. Each instance gets its own file in a temporary
    directory unless given a path -- these are scratch definitions, not data.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        mpedb = _require()
        self._lock = threading.Lock()
        if path is None:
            self._dir = tempfile.TemporaryDirectory(prefix="mpeqs-pyspell-")
            root = Path(self._dir.name)
        else:
            self._dir = None
            root = Path(path)
            root.mkdir(parents=True, exist_ok=True)
        config = root / "spell.toml"
        # mpedb resolves the database path relative to the config, so both live
        # in the same directory and the config carries a bare filename.
        config.write_text(_CONFIG.format(path="spell.mpedb"))
        try:
            self._db = mpedb.Database(str(config))
        except Exception as exc:
            raise Refusal(f"could not open a PySpell database: {exc}") from exc
        self._defined: set[str] = set()

    @property
    def db(self):
        """The underlying mpedb Database, for anything this wrapper does not cover."""
        return self._db

    def define(self, source: str) -> str:
        """Compile and store one function. Returns its name.

        A compile error is a ``Refusal`` carrying the line number, because the
        source came from a model and "it did not compile" is the useful answer.
        """
        name = function_name(source)
        with self._lock:
            if name in self._defined:
                return name
            try:
                self._db.define_function(source)
            except Exception as exc:
                raise Refusal(f"PySpell rejected {name!r}: {exc}") from exc
            self._defined.add(name)
        return name

    def call(self, source: str, *args: Any) -> Any:
        """Define the function if needed, then call it with these arguments.

        Any runtime error refuses the value rather than returning a number that
        was never derived -- which is the same contract as the rest of mpeqs.
        """
        name = self.define(source)
        placeholders = ", ".join(f"${i}" for i in range(1, len(args) + 1))
        try:
            rows = self._db.query(f"SELECT {name}({placeholders})", list(args))
        except Exception as exc:
            raise Refusal(f"{name}{args!r} refused: {exc}") from exc
        if not rows or not rows[0]:
            raise Refusal(f"{name}{args!r} returned nothing")
        return rows[0][0]

    def check_bijective(self, forward: str, inverse: str, *, name: str = "") -> int:
        """Verify that ``inverse(forward(x)) == x``, or refuse with a counter-example.

        Returns how many probe values actually round-tripped -- statistical
        evidence, deliberately reported rather than a bare "verified". A pair with
        a narrow domain guard exercises few probes, and knowing that is the point.
        """
        forward_name = self.define(forward)
        inverse_name = self.define(inverse)
        label = name or f"{forward_name}_{inverse_name}"
        try:
            return int(self._db.create_lens(label, forward_name, inverse_name))
        except Exception as exc:
            raise Refusal(f"{forward_name}/{inverse_name} is not bijective: {exc}") from exc

    def check_residual(self, forward: str, rex: str, inverse: str,
                       *, residual_type: str = "any", name: str = "") -> int:
        """Verify a lossy-but-recoverable transform: ``inverse(forward(x), rex(x)) == x``.

        This is the class for a transform that throws something away and can say
        exactly what: ``abs`` loses the sign, and registers fine when ``rex`` is
        the sign, because ``x -> (|x|, sign)`` is injective even though ``|x|`` is
        not.
        """
        f, r, i = self.define(forward), self.define(rex), self.define(inverse)
        label = name or f"{f}_{r}_{i}"
        try:
            return int(self._db.create_residual_lens(label, f, r, i, residual_type))
        except Exception as exc:
            raise Refusal(f"{f}/{r}/{i} is not a residual pair: {exc}") from exc


_default: Spell | None = None
_default_lock = threading.Lock()


def _shared() -> Spell:
    global _default
    with _default_lock:
        if _default is None:
            _default = Spell()
    return _default


def call(source: str, *args: Any) -> Any:
    """Run a model-written function on a shared scratch database."""
    return _shared().call(source, *args)


def check_bijective(forward: str, inverse: str, *, name: str = "") -> int:
    return _shared().check_bijective(forward, inverse, name=name)


def check_residual(forward: str, rex: str, inverse: str,
                   *, residual_type: str = "any", name: str = "") -> int:
    return _shared().check_residual(forward, rex, inverse,
                                    residual_type=residual_type, name=name)
