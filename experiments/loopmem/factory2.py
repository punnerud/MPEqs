#!/usr/bin/env python3
"""Stage A: model-authored bricks mounted end to end, and a model-authored solver.

Phase 76's factory ran on a hand-written pair. This closes the stated loop with the model
holding the pen at both places it belongs:

  BRICKS   Qwen writes the arithmetic cores only (the 3/4 path of phase 49 — the record
           injects the guards); mpedb's corpus judges the pair; the black-box probe reads
           the registered function through SQL alone; an exact probe match against the
           PRE-WRITTEN truth mounts it in the routing registry; and one route crosses a
           model-written brick, exactly.

  SOLVER   Qwen writes v_from_pe(p, e) — the speed-against-weight closer — as a plain
           PySpell function. The record judges it with perfect ground truth: sample (m, v)
           pairs generate p = m*v and e = m*v*v/2, and the function must return v exactly
           for every sample, through SQL calls. One retry with the counter-example named.

Truth transforms are written down BEFORE any model call, so nothing can drift toward
whatever the model happens to produce.
"""
import json
import re
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, "/tmp/pymod")
sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402
from rretl_guard import create_guarded_residual_lens  # noqa: E402

# The truths, before the model writes anything.
BRICK_TASKS = [
    ("dbl", "forward doubles a value: forward(21) = 42", (F(2), F(0))),
    ("tax", "forward adds 25% tax: forward(400) = 500", (F(5, 4), F(0))),
    ("ore", "forward turns kroner into ore: forward(3) = 300", (F(100), F(0))),
]
SOLVER_SAMPLES = [(20, 150), (4, 10), (50, 2), (8, 25), (12, 100), (2, 6)]

CORES = """Write three tiny Python function bodies for a reversible transform. The input is
GUARANTEED to be a nonzero integer with |x| < 4 * 10**12 — invalid inputs are refused
before your code runs, so write no checks.

Rules: no imports, no annotations; only + - * / comparisons, if/else, return.
Each function at most 3 lines.

Task: {task}. rex(x) captures anything forward loses (0 if nothing). inverse(y, r)
restores x exactly.

Reply with exactly three functions: {name}_fwd(x), {name}_rex(x), {name}_inv(y, r).
Only the code.
"""

SOLVER = """Write ONE tiny Python function body. Inputs are exact positive numbers.

Physics: an object with mass m and speed v has momentum p = m * v and kinetic energy
e = m * v * v / 2. Given p and e, compute the speed v.

Rules: no imports, no annotations; only + - * / and return. At most 2 lines.

Reply with exactly one function: v_from_pe(p, e). Only the code.
{hint}"""


def extract(reply, names):
    out = {}
    for m in re.finditer(r"(def (\w+)\(([^)]*)\):\n((?:[ \t]+.*\n?)+))", reply):
        if m.group(2) in names:
            out[m.group(2)] = (m.group(3), m.group(4).rstrip() + "\n")
    return out


def main(out="data/custom/factory2.json"):
    import mpedb
    import os
    for f in ("/tmp/factory2.mpedb", "/tmp/factory2.mpedb-lock"):
        try:
            os.unlink(f)
        except FileNotFoundError:
            pass
    Path("/tmp/factory2.toml").write_text(
        '[database]\npath = "/tmp/factory2.mpedb"\nsize_mb = 32\nmax_readers = 8\n')
    db = mpedb.Database("/tmp/factory2.toml")

    mounted = {}
    rows = []
    for name, task, truth in BRICK_TASKS:
        reply = ask("qwen-35b", CORES.format(task=task, name=name), n=280)
        parts = extract(reply, {f"{name}_fwd", f"{name}_rex", f"{name}_inv"})
        row = {"brick": name, "truth": (str(truth[0]), str(truth[1]))}
        if len(parts) != 3:
            row["stage"] = "no valid functions"
            rows.append(row)
            print(f"{name:<5} no valid functions")
            continue
        try:
            probes = create_guarded_residual_lens(
                db, name,
                f"def {name}_fwd(x):\n{parts[f'{name}_fwd'][1]}",
                f"def {name}_rex(x):\n{parts[f'{name}_rex'][1]}",
                f"def {name}_inv({parts[f'{name}_inv'][0]}):\n{parts[f'{name}_inv'][1]}")
        except Exception as e:  # noqa: BLE001 - the engine's refusal is the datum
            row["stage"] = f"engine refused: {str(e)[:90]}"
            rows.append(row)
            print(f"{name:<5} engine refused: {str(e)[:70]}")
            continue
        # Black-box probe through SQL only: two calls infer, the third verifies.
        def call(v):
            return F(str(db.query(f"SELECT {name}_fwd({v})")[0][0]))
        f1, f2, f3 = call(1), call(2), call(3)
        a, b = f2 - f1, f1 - (f2 - f1)
        probe_ok = f3 == a * 3 + b
        exact = (a, b) == truth
        if probe_ok and exact:
            mounted[name] = (a, b)
        row.update({"stage": "mounted" if exact else "probe mismatch",
                    "engine_probes": int(probes), "inferred": (str(a), str(b)),
                    "third_ok": probe_ok})
        rows.append(row)
        print(f"{name:<5} engine ok ({probes} probes), probe inferred ({a}, {b}) "
              f"-> {'MOUNTED' if exact else 'mismatch vs truth ' + str(truth)}")

    # One route crossing a model-written brick: eur -> nok -> ore, expected exactly.
    route_exact = False
    if "ore" in mounted:
        eur_nok = F(109, 100) * F(105, 10)
        expected = F(1000) * eur_nok * F(100)
        got = F(1000) * eur_nok * mounted["ore"][0] + mounted["ore"][1]
        route_exact = got == expected
        print(f"\nroute eur->nok->ore: 1000 eur = {got} ore, exact: {route_exact}")

    # The solver brick, judged only by its defining equations.
    hint = ""
    solver_ok = 0
    solver_row = {}
    for attempt in range(2):
        reply = ask("qwen-35b", SOLVER.format(hint=hint), n=160)
        parts = extract(reply, {"v_from_pe"})
        if not parts:
            hint = "\nYour last reply held no valid function."
            continue
        sig, body = parts["v_from_pe"]
        try:
            db.define_function(f"def v_from_pe({sig}):\n{body}")
        except Exception as e:  # noqa: BLE001
            hint = f"\nRefused to compile: {str(e)[:120]}"
            continue
        fails = []
        for m, v in SOLVER_SAMPLES:
            p, e = m * v, m * v * v // 2 if (m * v * v) % 2 == 0 else None
            if e is None:
                continue
            got = F(str(db.query(f"SELECT v_from_pe({p}, {e})")[0][0]))
            if got != v:
                fails.append((p, e, str(got), v))
        solver_ok = len([1 for m, v in SOLVER_SAMPLES if (m * v * v) % 2 == 0]) - len(fails)
        solver_row = {"attempt": attempt, "body": body.strip(), "fails": fails[:2],
                      "passed": solver_ok}
        if not fails:
            break
        hint = (f"\nWrong on p={fails[0][0]}, e={fails[0][1]}: returned {fails[0][2]}, "
                f"the speed is {fails[0][3]}.")
    n_samples = len([1 for m, v in SOLVER_SAMPLES if (m * v * v) % 2 == 0])
    print(f"\nsolver v_from_pe: {solver_ok}/{n_samples} constraint samples exact "
          f"({'accepted' if solver_ok == n_samples else 'REFUSED'})")
    if solver_ok == n_samples:
        got = F(str(db.query("SELECT v_from_pe(3000, 225000)")[0][0]))
        print(f"  closes the phase 76 question through the MODEL'S brick: "
              f"v = {got} m/s (exact: {got == 150})")
        solver_row["closes_phase76"] = got == 150

    print("\nThe pen is the model's at both ends now — the bricks and the solver — and no")
    print("judge in the chain is anything but arithmetic: the engine's corpus, the probe's")
    print("third value, the truth table written before the first call.")
    summary = {"bricks_attempted": len(BRICK_TASKS), "bricks_mounted": len(mounted),
               "route_exact": route_exact, "solver_passed": solver_ok,
               "solver_samples": n_samples,
               "solver_closes": solver_row.get("closes_phase76", False),
               "rows": rows, "solver": solver_row}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
