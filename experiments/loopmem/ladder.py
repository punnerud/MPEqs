#!/usr/bin/env python3
"""Graduated help: the model gets more context only when it fails, and only as much as needed.

Phase 48 ended with the diagnosis that the verifier's bar needs the engine's domain knowledge
IN the loop. This is that loop, built as a ladder so the help is graduated rather than
front-loaded — the model climbs only on failure, which measures how much help each task
actually needs instead of assuming all of it everywhere:

    L0   the bare task
    L1   + the engine's counter-example from the failed attempt
    L2   + the documented traps from PYSPELL-RRETL.md, verbatim knowledge of the engine
    L3   + every counter-example accumulated across ALL tasks so far

L3 is the thread's explored-set idea applied here: the record remembers the traps so the
model does not have to, and a lesson paid for on one task is available to the next. That is
the sense in which the small model "gets better without retraining" — nothing about the model
changes; what changes is what the record can put in front of it, and only when needed.

Same four transforms as phase 48, both models, fresh database per model.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/pymod")
sys.path.insert(0, str(Path(__file__).parent))
import mpedb  # noqa: E402
from cutbig import ask  # noqa: E402
from rretlpairs import MAX_LINES, TASKS, exercise, extract_functions, try_register  # noqa: E402

BASE = """Write three tiny Python functions for a reversible transform.

Language rules (anything else fails to compile):
- one def per function, no imports, no annotations, no defaults
- only: int/float/str/bool/None literals, + - * / // % comparisons, if/elif/else,
  while, return, len(), indexing
- integer overflow and division by zero are runtime errors
- to REFUSE an input outside the domain, write: return 1 // 0
- each function at most {max_lines} lines

The contract, which a verifier will test on edge values and refuse with a counter-example
if it does not hold:
    inverse(forward(x), rex(x)) == x   for every x in the domain

Task: {task}

Reply with exactly three functions named {name}_fwd (one argument), {name}_rex (one
argument), {name}_inv (two arguments: transformed value, residual). Only the code.
{help}"""

# The engine's documented traps, from PYSPELL-RRETL.md — the knowledge phase 48 showed the
# model cannot derive from counter-examples alone within three rounds.
TRAPS = """
Documented traps the verifier WILL probe:
- The probe corpus includes floats: +0.0, -0.0, tiny subnormals, ordinary floats.
- Int and Float are DIFFERENT values here. If forward(0.0) returns int 0, the inverse
  gives back int 0, not float 0.0, and the round trip fails. Never convert types.
- Comparisons cannot tell -0.0 from +0.0 (-0.0 < 0 is False). `0 - x` maps both zeros
  to +0.0; unary -x is safe.
- The standard fix for an integer-only transform: refuse floats AND zero, first thing
  in ALL THREE functions, identically:
      if x % 1 != 0 or x == 0:
          return 1 // 0
  Zero is then outside the domain; refused probe values do not count against you.
  In the inverse, guard the FIRST argument the same way (its second argument is your
  own rex output).
- After the guard, keep arithmetic in integers: use // and %, never /.
"""


def rung_help(rung, own_error, seen):
    if rung == 0:
        return ""
    parts = []
    if rung >= 1 and own_error:
        parts.append(f"\nThe verifier refused your previous attempt:\n  {own_error[:300]}\n"
                     f"Fix the functions so the round trip holds on that exact input.")
    if rung >= 2:
        parts.append(TRAPS)
    if rung >= 3 and seen:
        parts.append("Counter-examples the verifier has produced on earlier tasks — every "
                     "one is a trap your functions must also survive:\n" +
                     "\n".join(f"  - {e[:160]}" for e in seen[-6:]))
    return "\n".join(parts)


def main(rounds=4, out="data/custom/ladder.json"):
    rounds = int(rounds)
    results = {}
    for model in ("qwen-35b", "olmoe-1b"):
        cfg = Path(f"/tmp/ladder-{model}.toml")
        dbf = f"/tmp/ladder-{model}.mpedb"
        for p in (Path(dbf), Path(dbf + "-lock")):
            p.unlink(missing_ok=True)
        cfg.write_text(f'[database]\npath = "{dbf}"\nsize_mb = 64\nmax_readers = 8\n')
        db = mpedb.Database(str(cfg))

        seen = []                                # counter-examples across tasks: the record's
        rows = []                                # memory, not the model's
        print(f"\n{model}: ladder of {rounds} rungs, help only on failure")
        for name, task in TASKS:
            own_error = None
            row = {"task": name, "rungs": []}
            for rung in range(rounds):
                prompt = BASE.format(max_lines=MAX_LINES, task=task, name=name,
                                     help=rung_help(rung, own_error, seen))
                funcs = extract_functions(ask(model, prompt, n=420), name)
                if funcs is None:
                    row["rungs"].append({"rung": rung, "stage": "no valid functions"})
                    own_error = own_error or "reply did not contain three valid functions"
                    continue
                lens, probes, err = try_register(db, f"{name}L{rung}", funcs and {
                    k: v.replace(f"{name}_", f"{name}L{rung}_") for k, v in funcs.items()},
                    rung)
                if lens is None:
                    row["rungs"].append({"rung": rung, "stage": "refused",
                                         "error": (err or "")[:160]})
                    own_error = err
                    seen.append(err)
                    continue
                use = exercise(db, lens, f"{model[:2]}{name}{rung}")
                row["rungs"].append({"rung": rung, "stage": "registered",
                                     "probes": probes, **use})
                row["registered_at_rung"] = rung
                break
            rows.append(row)
            last = row["rungs"][-1]
            where = (f"rung {row['registered_at_rung']}" if "registered_at_rung" in row
                     else "never")
            print(f"  {name:<10} {where:<8} "
                  + (f"{last.get('probes', 0)} probes, applied {last.get('applied')}"
                     if last["stage"] == "registered" else last.get("error", "")[:70]))

        reg = [r for r in rows if "registered_at_rung" in r]
        results[model] = {
            "registered": len(reg), "tasks": len(TASKS),
            "rungs_used": {r["task"]: r.get("registered_at_rung") for r in rows},
            "mean_rung": (sum(r["registered_at_rung"] for r in reg) / len(reg)) if reg else None,
            "counter_examples_banked": len(seen), "rows": rows,
        }
        print(f"  -> {len(reg)}/{len(TASKS)} registered, "
              f"{len(seen)} counter-examples banked for later tasks")

    print("\nThe ladder is the point, not the top of it: help arrives only on failure, so the")
    print("rung a task registers at IS the measurement of how much engine knowledge it needs.")
    print("Nothing about either model changed — what changed is what the record could show it.")
    Path(out).write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
