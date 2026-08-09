#!/usr/bin/env python3
"""Three solvers over the brick world: binary, coupled, and the PySpell brick factory.

BINARY. Many two-sided edges hit at once make the pending space 2^k assignments — a binary
solver. One observed outcome then SOLVES the assignment: enumerate (k is small by
construction, only edges actually hit), keep the combinations reproducing the observation,
and if exactly one survives, every open question is answered at once, ledger cleared.

COUPLED. Sometimes the unknowns depend on each other — speed against weight — and no
assignment enumeration helps: that is an actual equation system. Momentum p = m*v and
energy E = m*v*v/2 known, m and v unknown: v = 2E/p and m = p*v exactly, and the DIMENSION
machinery validates the derivation (E/p types to m/s before any number moves). The solver
tier above the binary one, entered only when the pending questions share variables.

FACTORY. The bricks themselves can be authored by the model: a PySpell pair, verified by
mpedb's probe corpus (phases 48-50), is a black box the probe of phase 75 can read — two
calls infer its transform, a third verifies — and the inferred brick enters the routing
registry. Model writes, engine verifies, probe calibrates, N x N routes: every stage
measured earlier, joined here.
"""
import json
import sys
from fractions import Fraction as F
from itertools import product
from pathlib import Path

sys.path.insert(0, "/tmp/pymod")
sys.path.insert(0, str(Path(__file__).parent))
from bricks3 import declared_asymmetric, registry3  # noqa: E402
from brickassume import lazy_convert  # noqa: E402


def binary_solve(bricks, asym, by_name, value, path, observed):
    """Enumerate assignments over the two-sided edges the path actually hits; keep the
    ones reproducing the observation. Lazy: edges never hit never enter the space."""
    # DISTINCT questions, not hits: an edge crossed twice is one question with one
    # answer applied at every crossing. The first version gave each hit its own slot —
    # eight combos where four exist — and a both-flipped combo partially cancelled
    # through the eur loop into a second survivor.
    hit = {}
    for name in path:
        rev = name.split("->")[1] + "->" + name.split("->")[0]
        if name in asym and rev in by_name:
            hit.setdefault("|".join(sorted([name, rev])), (name, rev))
    keys = sorted(hit)
    combos = list(product(*[hit[k] for k in keys]))
    surviving = []
    for combo in combos:
        answers = dict(zip(keys, combo))
        v, _ = lazy_convert(bricks, asym, by_name, value, path, [], "solve",
                            answers=answers)
        if v == observed:
            surviving.append(answers)
    return len(combos), surviving


def coupled_solve(p, E):
    """m*v = p and m*v*v/2 = E: dimension-checked derivation, then exact values."""
    # Type first: E/p must be a velocity before any arithmetic is trusted.
    E_dim = {"kg": 1, "m": 2, "second": -2}
    p_dim = {"kg": 1, "m": 1, "second": -1}
    v_dim = {u: E_dim.get(u, 0) - p_dim.get(u, 0) for u in set(E_dim) | set(p_dim)}
    v_dim = {u: e for u, e in v_dim.items() if e}
    types_ok = v_dim == {"m": 1, "second": -1}
    v = 2 * E / p
    m = p / v
    checks = (m * v == p) and (m * v * v / 2 == E)
    return v, m, types_ok, checks


def pyspell_brick():
    """A PySpell pair through mpedb, probed as a black box, mounted as a brick."""
    try:
        import mpedb
    except ImportError:
        return {"available": False}
    import os
    for f in ("/tmp/factory.mpedb", "/tmp/factory.mpedb-lock"):
        try:
            os.unlink(f)
        except FileNotFoundError:
            pass
    Path("/tmp/factory.toml").write_text(
        '[database]\npath = "/tmp/factory.mpedb"\nsize_mb = 32\nmax_readers = 8\n')
    db = mpedb.Database("/tmp/factory.toml")
    # The offset pair in the PySpell subset — the phase 49 shape, guard included the
    # rretl_guard way. Registration puts it through the engine's probe corpus.
    from rretl_guard import create_guarded_residual_lens
    probes = create_guarded_residual_lens(
        db, "shift",
        "def sh_fwd(x):\n    return x - 1000\n",
        "def sh_rex(x):\n    return 0\n",
        "def sh_inv(y, r):\n    return y + 1000\n")
    # Now the probe of phase 75 reads it as a BLACK BOX through SQL calls only.
    def call(v):
        return F(db.query(f"SELECT sh_fwd({v})")[0][0])
    f1, f2, f3 = call(1), call(2), call(3)
    a = f2 - f1
    b = f1 - a
    verified = f3 == a * 3 + b
    return {"available": True, "engine_probes": int(probes),
            "inferred": (str(a), str(b)), "third_probe_ok": verified,
            "is_offset_brick": (a, b) == (F(1), F(-1000))}


def main(out="data/custom/bricksolve.json"):
    bricks = registry3()
    by_name = {b.name: b for b in bricks}
    asym = declared_asymmetric(bricks)

    # BINARY: a path crossing the quote, the toll, and the quote again via eur — three
    # two-sided hits, eight assignments, one observation.
    path = ["nok->usd", "usd->eur", "eur->nok", "nok->usd", "west->east"]
    truth_answers = {"nok->usd|usd->nok": "nok->usd",
                     "east->west|west->east": "west->east"}
    observed, _ = lazy_convert(bricks, asym, by_name, F(10000), path, [], "gt",
                               answers=truth_answers)
    n_combos, surviving = binary_solve(bricks, asym, by_name, F(10000), path, observed)
    unique = len(surviving) == 1
    recovered = unique and all(surviving[0][k] == v for k, v in truth_answers.items())
    print(f"BINARY: path hits {len(truth_answers)} distinct questions "
          f"({sum(1 for s in path if s in asym)} hits), {n_combos} assignments "
          f"enumerated, {len(surviving)} reproduce the observation")
    print(f"  unique and equal to the ground-truth directions: {recovered}")

    # COUPLED: fart mot tyngde. p = 3000 kg m/s, E = 225000 J -> v = 150 m/s, m = 20 kg.
    v, m, types_ok, checks = coupled_solve(F(3000), F(225000))
    print(f"\nCOUPLED: p=3000, E=225000 -> v = {v} m/s, m = {m} kg   "
          f"(types check: {types_ok}, equations check: {checks})")

    # FACTORY: PySpell -> mpedb verification -> black-box probe -> brick.
    fac = pyspell_brick()
    if fac["available"]:
        print(f"\nFACTORY: engine accepted the pair ({fac['engine_probes']} corpus probes),"
              f"\n  black-box probe inferred {fac['inferred']}, third probe "
              f"{'confirms' if fac['third_probe_ok'] else 'REFUTES'}, "
              f"offset brick recognised: {fac['is_offset_brick']}")
    else:
        print("\nFACTORY: mpedb build not present — skipped honestly")

    print("\nThree tiers, entered by need: assignments when the questions are independent,")
    print("equations when they couple, and the bricks themselves authored by the model,")
    print("verified by the engine, calibrated by the probe — the whole pipeline, joined.")
    summary = {"binary_assignments": n_combos, "binary_surviving": len(surviving),
               "binary_recovered": recovered,
               "coupled_v": str(v), "coupled_m": str(m),
               "coupled_types_ok": types_ok, "coupled_checks": checks,
               "factory": fac}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
