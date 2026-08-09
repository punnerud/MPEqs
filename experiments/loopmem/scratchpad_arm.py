#!/usr/bin/env python3
"""Give the model a named value store — the graph memory in its simplest form.

Measuring the premise first said the dominant failure is not looping: with the harness's own
fence-parsing bug fixed, only 2 of 41 steps repeated, while 20 of 41 stated an operation
WITHOUT recording its result. The next step then has no value to carry and reaches back for a
stale number.

That changes what the external memory has to be. A loop detector would have caught 2 steps. A
store only helps if there is something storable, so the first thing to test is not the graph but
the format: force every step to name its result, keep the named values in an explicit store, and
feed the store back rather than the prose history.

If accuracy moves, the missing piece was the format and the graph memory can then hold something
real. If it does not, the format was not the problem and building the memory would be premature.
"""
import json, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from measure_loops import BIN, MODEL, TASKS

PROMPT = """Solve the problem by computing one value at a time.

Problem: {problem}

Known values:
{store}

Write ONE line, exactly in the form `name = expression = number`, using only numbers from the
problem or names already known. If a known value already equals the whole problem, instead write
`DONE: <number>`.
Line:"""


def step(problem, store):
    known = "\n".join(f"  {k} = {v}" for k, v in store.items()) or "  (none yet)"
    Path("/tmp/sp.txt").write_text(PROMPT.format(problem=problem, store=known))
    out = subprocess.run([BIN, "-m", MODEL, "-f", "/tmp/sp.txt", "-n", "40", "--temp", "0",
                          "-no-cnv", "-st", "-ngl", "99"],
                         capture_output=True, text=True).stdout
    i = out.rfind("Line:")
    tail = re.sub(r"<think>.*?</think>", " ", out[i + 5:] if i >= 0 else out, flags=re.S)
    tail = tail.split("[end of text]")[0]
    tail = re.sub(r"^\s*```[a-zA-Z]*\s*$", "", tail, flags=re.M)
    for line in tail.splitlines():
        line = line.strip().strip("`").strip()
        if line and not line.startswith("#"):
            return line[:120]
    return ""


def main(max_steps=6, out="data/custom/scratchpad.json"):
    max_steps = int(max_steps)
    print(f"{len(TASKS)} problems, named value store fed back, up to {max_steps} steps\n")
    print(f"{'problem':<28}{'steps':>6}{'outcome':>10}{'stored':>8}{'rejected':>10}  final")
    rows = []
    for problem, answer in TASKS:
        store, rejected, outcome, final = {}, 0, "ran out", None
        for _ in range(max_steps):
            s = step(problem, store)
            m = re.search(r"DONE:\s*(-?\d+)", s)
            if m:
                final = int(m.group(1))
                outcome = "correct" if final == answer else "wrong"
                break
            # Only a line that names a value AND gives a number enters the store. Rejecting the
            # rest is the whole mechanism: the store cannot fill with unfinished thoughts.
            g = re.match(r"\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*=\s*(-?\d+(?:\.\d+)?)\s*$", s)
            if not g:
                rejected += 1
                continue
            store[g.group(1)] = g.group(3)
            if float(g.group(3)) == answer:
                final, outcome = answer, "correct"
                break
        rows.append({"problem": problem, "answer": answer, "stored": len(store),
                     "rejected": rejected, "outcome": outcome, "final": final,
                     "store": store})
        print(f"{problem:<28}{len(store) + rejected:>6}{outcome:>10}{len(store):>8}"
              f"{rejected:>10}  {final}")
    n = len(rows)
    summary = {"tasks": n,
               "correct": sum(1 for r in rows if r["outcome"] == "correct"),
               "wrong": sum(1 for r in rows if r["outcome"] == "wrong"),
               "ran_out": sum(1 for r in rows if r["outcome"] == "ran out"),
               "total_stored": sum(r["stored"] for r in rows),
               "total_rejected": sum(r["rejected"] for r in rows)}
    print(f"\ncorrect {summary['correct']}/{n}   (free-form history baseline: 0/{n})")
    print(f"values stored {summary['total_stored']}, lines rejected "
          f"{summary['total_rejected']}")
    Path(out).write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
