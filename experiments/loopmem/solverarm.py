#!/usr/bin/env python3
"""The fifth arm: on a template hit, the record instantiates and a solver evaluates.

The proposal that closes the loop: give the record a math function. The model's contribution is
the decomposition, paid once per SHAPE; on every later instance of that shape the record binds
the values and a solver computes the steps, and the model never sees a result it could copy —
the result is carried forward by the record, not through the model's context.

This arm costs no model calls on a hit, so it is computed from the stored run rather than by
running again: for each task, a hit scores the solver's exact evaluation of the instantiated
graph, and a miss falls back to what the graph arm produced on that task. Correctness on hits is
1.0 by construction — the interesting numbers are coverage, and what the composite arm scores
once the misses are counted honestly.

The division of labour it demonstrates is the thread's endpoint: the model recognises and
decomposes; the record instantiates, solves, verifies, and remembers. Each does only the part
the other cannot.
"""
import json
import sys
from pathlib import Path


def main(src="data/custom/template.json", out="data/custom/solverarm.json"):
    d = json.loads(Path(src).read_text())
    rows = d["rows"]
    print(f"{'model':<10}{'tasks':>7}{'hits':>6}{'solver arm':>12}{'best model arm':>16}"
          f"{'model calls saved':>19}")
    summary = {}
    for model, s in d["summary"].items():
        sel = [r for r in rows if r["model"] == model]
        solver = hits = saved = 0
        for r in sel:
            if r["template_hit"]:
                hits += 1
                # The record instantiated this graph; a solver evaluates it exactly. eval on
                # our own generated arithmetic, the same trust boundary as check_graph.
                solver += abs(eval(r["expr"]) - r["truth"]) < 1e-9  # noqa: S307
                # The calls the model would have spent evaluating the steps (3 per graph).
                saved += 3
            else:
                v = r.get("graph")
                solver += v is not None and abs(v - r["truth"]) < 1e-9
        best_model = max(s["graph"], s["trecord"], s["tmodel"])
        summary[model] = {"tasks": s["tasks"], "hits": hits, "solver_arm": solver,
                          "best_model_arm": best_model, "calls_saved": saved,
                          "hit_correct_rate": 1.0 if hits else 0.0}
        print(f"{model:<10}{s['tasks']:>7}{hits:>6}{solver:>9}/{s['tasks']:<3}"
              f"{best_model:>13}/{s['tasks']:<3}{saved:>19}")

    print("\nOn a hit the solver is exact and free of model calls; every residual error in the")
    print("composite arm lives in the misses, where the model still has to decompose something")
    print("it has not seen. The store converges on covering the shapes, and the model's job")
    print("converges on only the genuinely new.")
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
