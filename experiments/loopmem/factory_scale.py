#!/usr/bin/env python3
"""Stage B: eight brick families through the factory, and what the registry grows by.

The scale question: does the pipeline that mounted three bricks hold when the model
authors EIGHT families in one run — and what does each mounted family buy the registry?
Truth factors are written here, before any model call. Every family goes model cores ->
guard injection -> engine corpus -> black-box probe -> judgement against the truth ->
mounted on exact match. Both judges are reported separately: the engine can accept a pair
whose transform is not the asked-for one (a correct reversible function for the wrong
task), and only the probe-versus-truth comparison catches that.

Growth is measured the phase 52 way on the merged registry: simple pairs routable before
and after, compound conversions before and after, and roads-per-currency-pair — the
audit capacity of phase 74, which grows with every alternative path a new family opens.
"""
import json
import re
import sys
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path

sys.path.insert(0, "/tmp/pymod")
sys.path.insert(0, str(Path(__file__).parent))
from bricks import lift  # noqa: E402
from bricks2 import ABrick, all_roads, registry2, tkey  # noqa: E402
from cutbig import ask  # noqa: E402
from factory2 import CORES, extract  # noqa: E402
from rretl_guard import create_guarded_residual_lens  # noqa: E402

# The truths, before the model writes a line. All scale bricks (offset families would be
# refused by the third probe when non-affine, correctly; réaumur is affine and declared).
FAMILIES = [
    ("nmi", "nautical miles to km: forward(2) = 3.704", "nmi", "km", (F(1852, 500), F(0))),
    ("fortnight", "fortnights to days: forward(2) = 28", "fortnight", "day",
     (F(14), F(0))),
    ("stone", "stone to pounds: forward(2) = 28", "stone", "pound", (F(14), F(0))),
    ("pint", "UK pints to litres: forward(1000) = 568.26125", "pint", "litre",
     (F(56826125, 100000000), F(0))),
    ("sek", "Swedish kronor to NOK at rate 0.97: forward(100) = 97", "sekr", "nok",
     (F(97, 100), F(0))),
    ("dkk", "Danish kroner to NOK at rate 1.52: forward(100) = 152", "dkk", "nok",
     (F(152, 100), F(0))),
    ("markup", "adds 40% markup: forward(10) = 14", "cost", "price", (F(7, 5), F(0))),
    ("dozen", "dozens to pieces: forward(3) = 36", "dozen", "piece", (F(12), F(0))),
]


def count_reach(bricks_list):
    """How many ordered simple-type pairs the registry can route, and compound reach for
    the currency block — the growth ledger."""
    units = sorted({u for b in bricks_list for u, _ in list(b.src) + list(b.dst)})
    reach = 0
    for a, c in combinations(units, 2):
        for s, d in ((a, c), (c, a)):
            frontier, seen = {tkey({s: 1})}, {tkey({s: 1})}
            found = False
            for _ in range(8):
                nxt = set()
                for t in frontier:
                    for b in bricks_list:
                        class _S:
                            pass
                        _S.src, _S.dst, _S.factor, _S.name = b.src, b.dst, b.t[0], b.name
                        lf = lift(_S, dict(t))
                        if lf and lf[0] not in seen:
                            nxt.add(lf[0])
                if tkey({d: 1}) in nxt:
                    found = True
                    break
                if not nxt:
                    break
                seen |= nxt
                frontier = nxt
            reach += found
    return len(units), reach


def main(out="data/custom/factory_scale.json"):
    import mpedb
    import os
    for f in ("/tmp/fscale.mpedb", "/tmp/fscale.mpedb-lock"):
        try:
            os.unlink(f)
        except FileNotFoundError:
            pass
    Path("/tmp/fscale.toml").write_text(
        '[database]\npath = "/tmp/fscale.mpedb"\nsize_mb = 32\nmax_readers = 8\n')
    db = mpedb.Database("/tmp/fscale.toml")

    base = registry2()
    units_before, reach_before = count_reach(base)
    roads_before = len(all_roads(base, {"nok": 1}, {"eur": 1}, max_len=3))

    authored = engine_ok = probe_exact = 0
    mounted = []
    rows = []
    for name, task, src_u, dst_u, truth in FAMILIES:
        reply = ask("qwen-35b", CORES.format(task=task, name=name), n=280)
        parts = extract(reply, {f"{name}_fwd", f"{name}_rex", f"{name}_inv"})
        row = {"family": name, "truth": (str(truth[0]), str(truth[1]))}
        if len(parts) != 3:
            row["stage"] = "no valid functions"
            rows.append(row)
            print(f"{name:<10} no valid functions")
            continue
        authored += 1
        try:
            create_guarded_residual_lens(
                db, name,
                f"def {name}_fwd(x):\n{parts[f'{name}_fwd'][1]}",
                f"def {name}_rex(x):\n{parts[f'{name}_rex'][1]}",
                f"def {name}_inv({parts[f'{name}_inv'][0]}):\n{parts[f'{name}_inv'][1]}")
        except Exception as e:  # noqa: BLE001
            row["stage"] = f"engine refused: {str(e)[:80]}"
            rows.append(row)
            print(f"{name:<10} engine refused: {str(e)[:60]}")
            continue
        engine_ok += 1

        def call(v):
            return F(str(db.query(f"SELECT {name}_fwd({v})")[0][0]))
        f1, f2, f3 = call(4), call(8), call(12)
        a = (f2 - f1) / 4
        b = f1 - a * 4
        third = f3 == a * 12 + b
        exact = third and (a, b) == truth
        probe_exact += exact
        row.update({"stage": "mounted" if exact else "probe vs truth mismatch",
                    "inferred": (str(a), str(b)), "third_ok": third})
        rows.append(row)
        print(f"{name:<10} engine ok, probe ({a}, {b}) "
              f"{'MOUNTED' if exact else '!= truth ' + str((str(truth[0]), str(truth[1])))}")
        if exact:
            mounted.append(ABrick(f"{src_u}->{dst_u}", {src_u: 1}, {dst_u: 1}, a))
            mounted.append(ABrick(f"{dst_u}->{src_u}", {dst_u: 1}, {src_u: 1}, 1 / a))

    merged = base + mounted
    units_after, reach_after = count_reach(merged)
    roads_after = len(all_roads(merged, {"nok": 1}, {"eur": 1}, max_len=3))

    print(f"\nauthored {authored}/{len(FAMILIES)}, engine accepted {engine_ok}, "
          f"probe-exact and mounted {probe_exact}")
    print(f"registry: {len(base)} -> {len(merged)} bricks; units {units_before} -> "
          f"{units_after}; routable ordered pairs {reach_before} -> {reach_after}")
    print(f"roads nok->eur (audit capacity): {roads_before} -> {roads_after}")
    print("\nEvery mounted family is three exact judges deep — corpus, third probe, truth")
    print("table — and what it buys is counted, not claimed: pairs the registry can now")
    print("route, and roads the auditor can now compare.")
    summary = {"families": len(FAMILIES), "authored": authored, "engine_ok": engine_ok,
               "probe_exact_mounted": probe_exact,
               "bricks_before": len(base), "bricks_after": len(merged),
               "units_before": units_before, "units_after": units_after,
               "reach_before": reach_before, "reach_after": reach_after,
               "roads_before": roads_before, "roads_after": roads_after, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
