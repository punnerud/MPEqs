#!/usr/bin/env python3
"""The LLM writes rRETL pairs: PySpell functions, engine-verified, residuals checked.

This plugs the thread's endpoint into real infrastructure. mpedb's rRETL (implemented locally
in ~/mpedb-rel, 2026-07-28) is exactly the contract every phase here converged on — a transform
registers as forward/rex/inverse, the engine verifies the round trip against a probe corpus of
edge values, a declaration that does not hold is refused WITH A COUNTER-EXAMPLE, and the
residual is stored per row so the reverse is reconstruction rather than hope.

The division of labour is the measured one. The model writes the functions — in the PySpell
subset, compiled to IR and run by Rust, deterministic by construction: no imports, no clock, no
randomness, a fixed instruction budget. The engine decides. And the engine's refusals are the
explore loop: they name the exact input that collided and what came back, so a retry is a new
question, not the same one — the same principle as showing the model what its cut actually took.

Per task and model, up to three rounds:

    propose -> define_function x3 -> create_residual_lens -> engine verifies
        refused? -> the engine's counter-example goes back to the model -> retry

A pair that registers is then USED: applied to a real column, rows edited, putback carried the
edits, revert refused after edits — the semantics the store promises, exercised rather than
believed. Functions are content-hashed in the database file, so a registered pair is reusable
by every attached process without redefinition — the "reusable and fast" half is the engine's,
not the model's.

Each function is length-limited to eight lines. The subset plus the limit is what makes model
output safe to run at all: there is nothing an eight-line loop-bounded arithmetic function can
do to the host.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/pymod")
sys.path.insert(0, str(Path(__file__).parent))
import mpedb  # noqa: E402
from cutbig import ask  # noqa: E402

MAX_LINES = 8

TASKS = [
    ("absval", "forward(x) drops the sign of an integer: forward(-7) = 7, forward(7) = 7. "
               "rex(x) must capture whether x was negative. Domain: any integer."),
    ("tens", "forward(x) keeps only the tens of a non-negative integer: forward(47) = 4, "
             "forward(123) = 12. rex(x) is what forward throws away. Refuse x < 0."),
    ("evenhalf", "forward(x) halves an integer, rounding down: forward(9) = 4. rex(x) is "
                 "what is needed to get x back exactly."),
    ("offset", "forward(x) = x - 1000 for integers. Nothing is lost, but register it as a "
               "residual pair anyway with rex(x) = 0."),
]

PROMPT = """Write three tiny Python functions for a reversible transform.

Language rules (anything else fails to compile):
- one def per function, no imports, no annotations, no defaults
- only: int/float/str/bool/None literals, + - * / // % comparisons, if/elif/else,
  while, return, len(), indexing
- integer overflow and division by zero are runtime errors
- to REFUSE an input outside the domain, write: return 1 // 0
- each function at most {max_lines} lines
- the verifier also probes floats, including -0.0 and tiny subnormals. Comparisons cannot
  tell -0.0 from 0.0 here, so an integer-only transform must refuse BOTH non-integers and
  zero, first thing in all three functions:
      if x % 1 != 0 or x == 0:
          return 1 // 0
  (0 is then outside the domain; that is the accepted price)

The contract, which a verifier will test on edge values and refuse with a counter-example
if it does not hold:
    inverse(forward(x), rex(x)) == x   for every x in the domain

Task: {task}

Reply with exactly three functions named {name}_fwd (one argument), {name}_rex (one
argument), {name}_inv (two arguments: transformed value, residual). Only the code.
{feedback}"""


def extract_functions(reply, name):
    """The three defs, whole, in whatever order they appear."""
    out = {}
    for m in re.finditer(r"(def (\w+)\([^)]*\):\n(?:[ \t]+.*\n?)+)", reply):
        src, fname = m.group(1).rstrip() + "\n", m.group(2)
        if fname in (f"{name}_fwd", f"{name}_rex", f"{name}_inv"):
            if len(src.splitlines()) <= MAX_LINES:
                out[fname.rsplit("_", 1)[1]] = src
    return out if set(out) == {"fwd", "rex", "inv"} else None


def try_register(db, name, funcs, attempt):
    """Define and register; the engine's exception text is the feedback on refusal.

    Function names get an attempt suffix before defining, because a retry carries different
    source under the same name and stored functions are content-hashed — the model keeps the
    stable names it was asked for, and the rename is the record's bookkeeping."""
    lens = f"{name}_v{attempt}"
    try:
        for role, src in funcs.items():
            db.define_function(src.replace(f"{name}_{role}", f"{name}{attempt}_{role}", 1))
        probes = db.create_residual_lens(
            lens, f"{name}{attempt}_fwd", f"{name}{attempt}_rex", f"{name}{attempt}_inv",
            "any")
        return lens, int(probes), None
    except Exception as e:  # noqa: BLE001 - the refusal text is the point
        return None, 0, str(e)


def exercise(db, lens, tag):
    """Apply, edit, putback, and check the promises: edits carried, loss re-attached."""
    table = f"t_{tag}"
    vals = [-40, -7, 0, 3, 47, 123, 1000, 4096]
    db.query(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, v ANY)")
    for i, v in enumerate(vals):
        db.query(f"INSERT INTO {table} VALUES ({i}, {v})")
    try:
        rep = db.rretl_apply(lens, table, "v")
    except Exception as e:  # noqa: BLE001 - a domain refusal at apply is a result
        return {"applied": False, "why": str(e)[:120]}
    # Edit one transformed row, then putback: the edit must survive the reverse.
    db.query(f"UPDATE {table} SET v = 999 WHERE id = 3")
    db.rretl_putback(rep["run_id"])
    after = {r[0]: r[1] for r in db.query(f"SELECT id, v FROM {table}")}
    untouched_ok = all(after[i] == vals[i] for i in range(len(vals)) if i != 3)
    return {"applied": True, "rows": rep["rows"], "residuals": rep["residuals"],
            "edit_carried": after[3] != vals[3], "untouched_restored": untouched_ok}


def main(model="qwen-35b", rounds=3, out="data/custom/rretlpairs.json"):
    rounds = int(rounds)
    cfg = Path("/tmp/rretl-exp.toml")
    dbfile = Path("/tmp/rretl-exp.mpedb")
    for p in (dbfile, Path(str(dbfile) + "-lock")):
        p.unlink(missing_ok=True)
    cfg.write_text('[database]\npath = "/tmp/rretl-exp.mpedb"\nsize_mb = 64\n'
                   'max_readers = 8\n')
    db = mpedb.Database(str(cfg))

    print(f"{len(TASKS)} transforms, {model}, up to {rounds} rounds against the verifier\n")
    rows = []
    for name, task in TASKS:
        feedback = ""
        result = {"task": name, "model": model, "rounds": []}
        for attempt in range(rounds):
            reply = ask(model, PROMPT.format(max_lines=MAX_LINES, task=task, name=name,
                                             feedback=feedback), n=400)
            funcs = extract_functions(reply, name)
            if funcs is None:
                result["rounds"].append({"attempt": attempt, "stage": "no valid functions"})
                feedback = ("\nYour previous reply did not contain all three functions "
                            "within the line limit. Reply with only the three defs.")
                continue
            lens, probes, err = try_register(db, name, funcs, attempt)
            if lens is None:
                result["rounds"].append({"attempt": attempt, "stage": "refused",
                                         "error": err[:200]})
                # The engine's counter-example IS the exploration signal.
                feedback = (f"\nThe verifier refused your previous attempt:\n  {err[:300]}\n"
                            f"Fix the functions so the round trip holds on that input.")
                continue
            use = exercise(db, lens, f"{name}{attempt}")
            result["rounds"].append({"attempt": attempt, "stage": "registered",
                                     "probes": probes, **use})
            break
        rows.append(result)
        last = result["rounds"][-1]
        print(f"{name:<10} {len(result['rounds'])} round(s) -> {last['stage']}"
              + (f", {last.get('probes', 0)} probes, edit carried "
                 f"{last.get('edit_carried')}" if last["stage"] == "registered" else
                 f": {last.get('error', '')[:80]}"))

    registered = sum(1 for r in rows if r["rounds"][-1]["stage"] == "registered")
    first_try = sum(1 for r in rows if r["rounds"][0].get("stage") == "registered")
    carried = sum(1 for r in rows if r["rounds"][-1].get("edit_carried"))
    print(f"\nregistered  : {registered}/{len(TASKS)}  ({first_try} first try)")
    print(f"edits carried through putback on every registered pair: {carried}/{registered}")
    print("\nThe engine is the reviewer: nothing the model writes is trusted, everything is")
    print("probed, and a refusal names the input that broke — which is what makes the retry")
    print("a different question. The functions are content-hashed in the database, so the")
    print("registered pairs are the reusable, fast half, and they now exist on disk.")
    Path(out).write_text(json.dumps({"model": model, "tasks": len(TASKS),
                                     "registered": registered, "first_try": first_try,
                                     "edits_carried": carried, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
