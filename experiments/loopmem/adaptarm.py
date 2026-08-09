#!/usr/bin/env python3
"""The ADAPT arm: retrieval aims, the model binds the roles. The missing mapstore arm.

Phases 51 and 54 measured every record-only path through the template store — positional
binding 0/60, voting 1/60, dimension-filtered 1/60 — and diagnosed the gap precisely: the
nearest template is found by topic, and binding is semantic, and only a language model can map
WHICH number plays WHICH role. Phase 44 measured that the model can (19/20 adapting a shown
plan to new numbers, one anti-copy line required). This is the two halves joined on real
problems: the store retrieves the two nearest solved templates, shows them as worked examples
with their own numbers, and the model writes the plan for the new problem.

Three arms per model on the SAME 60 test problems as phases 51/54, in one run so the
comparison is within-run rather than across Metal nondeterminism:

    SOLO     answer with brief working                 (the strongest baseline, phase 53)
    GRAPH    write the plan cold, record executes      (phase 53's arm)
    ADAPT    the same, with the two retrieved worked examples in front

If ADAPT beats GRAPH, retrieval-aimed examples pay on real problems and the store finally
earns its keep on words; if it only matches, the examples are noise to a model that already
plans; and the 1B model is the interesting case, because phase 45 showed memory helps exactly
the model that cannot plan alone.
"""
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402
from dimgraph import run_steps  # noqa: E402
from embednav import embed  # noqa: E402
from jsongraph import parse_graph  # noqa: E402
from mapstore import NUM, TEST, build_store, mask, norm  # noqa: E402
from olympiad import SOLO, last_number  # noqa: E402

GRAPH = """Solve the problem by writing ONLY the arithmetic plan as JSON. Each key is one
step using numbers from the problem or earlier keys; never write a computed result. Use the
numbers from the problem you are given, not from the examples. The last key is the answer.
{examples}
Problem: {problem}
"""

BASE_EXAMPLE = """
Example:
Problem: Tom has 3 boxes of 12 eggs and eats 5. How many eggs are left?
{"A": "3 * 12", "B": "A - 5"}
"""


def steps_to_graph(steps):
    """The store's S0/S1 step list as the A/B graph notation the models write."""
    keys = [chr(ord("A") + i) for i in range(len(steps))]
    out = {}
    for i, s in enumerate(steps):
        for j in range(i - 1, -1, -1):
            s = re.sub(rf"\bS{j}\b", keys[j], s)
        out[keys[i]] = s
    return out


def main(n_test=60, seed=5, out="data/custom/adaptarm.json"):
    import random
    n_test, seed = int(n_test), int(seed)
    store, kept, _, _ = build_store(2000)
    vecs = np.array(embed([t["masked"] for t in store]), dtype=np.float32)
    print(f"store: {kept} verified plans, embedded")

    tests = []
    for line in TEST.read_text().splitlines():
        d = json.loads(line)
        tests.append((d["question"],
                      Fraction(norm(d["answer"].rsplit("#### ", 1)[-1].strip()))))
    tests = random.Random(seed).sample(tests, n_test)
    qvecs = np.array(embed([mask(norm(q)) for q, _ in tests]), dtype=np.float32)

    results = {}
    for model in ("qwen-35b", "olmoe-1b"):
        tally = {"solo": 0, "graph": 0, "adapt": 0, "adapt_parsed": 0}
        rows = []
        for i, (q, truth) in enumerate(tests):
            a_solo = last_number(ask(model, SOLO.format(problem=q), n=512))

            g, _ = parse_graph(ask(model, GRAPH.format(
                examples=BASE_EXAMPLE, problem=q), n=512))
            a_graph = run_steps(g) if g else None

            # Retrieval aims: the two nearest solved problems, shown whole — their own story,
            # their own numbers, their own plan — so the binding is the model's only job.
            top = np.argsort(vecs @ qvecs[i])[::-1][:2]
            shown = "".join(
                f"\nExample:\nProblem: {' '.join(store[int(j)]['question'].split())}\n"
                f"{json.dumps(steps_to_graph([s for s in instantiate_steps(store[int(j)])]))}\n"
                for j in top)
            g2, _ = parse_graph(ask(model, GRAPH.format(examples=shown, problem=q), n=512))
            a_adapt = run_steps(g2) if g2 else None

            tally["solo"] += a_solo == truth
            tally["graph"] += a_graph == truth
            tally["adapt"] += a_adapt == truth
            tally["adapt_parsed"] += g2 is not None
            rows.append({"model": model, "truth": str(truth), "solo": str(a_solo),
                         "graph": str(a_graph), "adapt": str(a_adapt),
                         "retrieved": [store[int(j)]["masked"][:50] for j in top]})
        results[model] = {**tally, "n": n_test, "rows": rows}
        print(f"{model}: solo {tally['solo']}/{n_test}   graph {tally['graph']}/{n_test}   "
              f"ADAPT {tally['adapt']}/{n_test}   ({tally['adapt_parsed']} parsed)")

    print("\nRecord-only paths through the same store: positional 0/60, vote 1/60, dimension-")
    print("filtered 1/60. Whatever ADAPT scores above those is the value of role binding, and")
    print("whatever it scores against GRAPH is the value of the retrieval that aimed it.")
    Path(out).write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


def instantiate_steps(template):
    """The template's steps with its OWN source numbers back in — a concrete worked example."""
    values = re.findall(r"\d+(?:\.\d+)?", template["question"].replace(",", ""))
    out = []
    for s in template["steps"]:
        for k in sorted(range(len(values)), reverse=True):
            s = re.sub(rf"\bv{k + 1}\b", values[k], s)
        out.append(s)
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
