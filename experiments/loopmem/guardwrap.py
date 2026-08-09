#!/usr/bin/env python3
"""The record injects the guards; the model writes only the arithmetic core.

The ladder measured what help-in-context is worth, and the answer was: not enough. With the
documented traps verbatim in the task text and twelve counter-examples banked, the 35B model
still registered 1 of 4 — it reads the guard discipline and does not reliably apply it across
all three functions. Meanwhile the hand-written ceiling is 4 of 4, using exactly the guards the
documentation prescribes (plus one it does not: a magnitude bound, because integral floats above
2^52 survive `x % 1 != 0` and then lose low bits in arithmetic — the engine caught that at
Float(1.34e17), the same class of failure as its own authors' celsius pair).

So the discipline moves into the record, which is where every phase of this thread has ended up
putting mechanics. The model is told the domain is already clean — nonzero integers within
+-2^42 — and writes only the three arithmetic bodies. The record prepends the full guard to
forward and rex (the inverse sees only forward/rex outputs, so it takes none), and the engine
verifies as before. Nothing is trusted: the wrap changes who WRITES the guard, not whether the
verifier checks it.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/pymod")
sys.path.insert(0, str(Path(__file__).parent))
import mpedb  # noqa: E402
from cutbig import ask  # noqa: E402
from rretlpairs import TASKS, exercise  # noqa: E402

# The wrap now lives in rretl_guard.py, the reusable module this experiment motivated —
# same guard, one writer, every caller. mpedb itself is untouched; the engine stays judge.
from rretl_guard import create_guarded_residual_lens  # noqa: E402

PROMPT = """Write three tiny Python function bodies for a reversible transform. The input is
GUARANTEED to be a nonzero integer with |x| < 4 * 10**12 — invalid inputs are refused before
your code runs, so do not write any checks.

Rules: no imports, no annotations; only + - * // % comparisons, if/else, return.
Use // and %, never /. Each function at most 4 lines.

The contract a verifier will test:
    inverse(forward(x), rex(x)) == x

Task: {task}

Reply with exactly three functions: {name}_fwd(x), {name}_rex(x), {name}_inv(y, r).
Only the code.
"""


def extract(reply, name):
    out = {}
    for m in re.finditer(r"(def (\w+)\(([^)]*)\):\n((?:[ \t]+.*\n?)+))", reply):
        fname = m.group(2)
        if fname in (f"{name}_fwd", f"{name}_rex", f"{name}_inv"):
            body = m.group(4).rstrip() + "\n"
            if len(body.splitlines()) <= 4:
                out[fname.rsplit("_", 1)[1]] = (m.group(3), body)
    return out if set(out) == {"fwd", "rex", "inv"} else None


def main(model="qwen-35b", out="data/custom/guardwrap.json"):
    cfg = Path("/tmp/guardwrap.toml")
    for p in (Path("/tmp/guardwrap.mpedb"), Path("/tmp/guardwrap.mpedb-lock")):
        p.unlink(missing_ok=True)
    cfg.write_text('[database]\npath = "/tmp/guardwrap.mpedb"\nsize_mb = 64\n'
                   'max_readers = 8\n')
    db = mpedb.Database(str(cfg))

    print(f"{len(TASKS)} transforms, {model}, the record owning the guards\n")
    rows = []
    for name, task in TASKS:
        reply = ask(model, PROMPT.format(task=task, name=name), n=320)
        parts = extract(reply, name)
        if parts is None:
            rows.append({"task": name, "stage": "no valid functions"})
            print(f"{name:<10} no valid functions")
            continue
        try:
            probes = create_guarded_residual_lens(
                db, name,
                f"def {name}_fwd(x):\n{parts['fwd'][1]}",
                f"def {name}_rex(x):\n{parts['rex'][1]}",
                f"def {name}_inv({parts['inv'][0]}):\n{parts['inv'][1]}")
        except Exception as e:  # noqa: BLE001 - the refusal is the datum
            rows.append({"task": name, "stage": "refused", "error": str(e)[:200]})
            print(f"{name:<10} refused: {str(e)[:80]}")
            continue
        use = exercise(db, name, name)
        rows.append({"task": name, "stage": "registered", "probes": int(probes), **use})
        print(f"{name:<10} registered, {probes} probes, applied {use.get('applied')}, "
              f"edit carried {use.get('edit_carried')}")

    reg = sum(1 for r in rows if r["stage"] == "registered")
    print(f"\nregistered: {reg}/{len(TASKS)}  (ladder with docs in context: 1/4; "
          f"hand-written ceiling: 4/4)")
    print("The guard was never the model's strength and was always the record's job. What")
    print("the model actually knows — the arithmetic that reverses — was there all along.")
    Path(out).write_text(json.dumps({"model": model, "registered": reg,
                                     "tasks": len(TASKS), "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
