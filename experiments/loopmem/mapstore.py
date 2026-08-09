#!/usr/bin/env python3
"""Plan choice as retrieval: read the problem, map it against generalised solved types.

The reframing this implements: the plan is not something to think up, it is something to LOOK
UP. Read the problem, strip its numbers, map the residue against embeddings of generalised
solved problems — compressed into groups, phase 37's machinery — pick the nearest type, bind
this problem's numbers into that type's plan, and execute every step exactly. Thinking chose
plans in the olympiad run; here the store chooses them, and nothing chooses arithmetic because
arithmetic is the solver's.

The store is built without a single model call. GSM8K training solutions carry calculator
annotations — `<<48/2=24>>` — from which the record extracts the plan mechanically, chains
steps by value flow, EXECUTES the plan and keeps it only if it reproduces the training answer,
then generalises it under phase 45's round-trip rule: numbers become variables identically in
text and plan, and re-instantiating must reproduce the source exactly. Verified, generalised,
embedded, clustered. That is "komprimering i grupper" made literal: thousands of solved
problems become hundreds of typed groups with a plan each.

At test time, per problem, NO model at all in the first arm:

  RECORD-MAP   mask the numbers, embed, retrieve the nearest template with a compatible
               number count, bind positionally, execute. Pure lookup-and-solve.
  ADAPT        show the retrieved template as a worked example and let the model write the
               plan for the new numbers — phase 44's reuse, aimed by retrieval. (Run
               separately when the GPU is free.)

RECORD-MAP's failure mode is the measurement: retrieval can fetch a type whose plan shape
fits but whose semantics do not, and positional binding cannot notice. How often the nearest
generalised neighbour IS the right recipe is exactly the question the reframing raises.
"""
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from embednav import embed  # noqa: E402

ANNOT = re.compile(r"<<([^<>=]+)=([^<>]+)>>")
NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
TRAIN = Path("/tmp/gsm8k-train.jsonl")
TEST = Path("/tmp/gsm8k-test.jsonl")


def norm(s):
    return s.replace(",", "").replace("$", "")


def mask(text):
    """The problem with its numbers removed: what the TYPE of a problem looks like."""
    return NUM.sub("<N>", text)


def extract_plan(solution):
    """The chain of annotated steps, operands rewritten to references by value flow."""
    steps, results = [], []
    for expr, res in ANNOT.findall(norm(solution)):
        expr, res = expr.strip(), res.strip()
        out, pos = "", 0
        for m in NUM.finditer(expr):
            ref = next((f"S{j}" for j in range(len(results) - 1, -1, -1)
                        if results[j] == m.group(0)), None)
            out += expr[pos:m.start()] + (ref if ref else m.group(0))
            pos = m.end()
        steps.append(out + expr[pos:])
        results.append(res)
    return steps


def run_plan(steps, binding=None):
    values = {}
    for i, s in enumerate(steps):
        expr = s
        if binding:
            for var in sorted(binding, key=len, reverse=True):
                expr = re.sub(rf"\b{var}\b", binding[var], expr)
        for j in range(len(values) - 1, -1, -1):
            expr = re.sub(rf"\bS{j}\b", f"({values[f'S{j}']})", expr)
        if not re.fullmatch(r"[\d\s+*/().-]+", expr):
            return None
        try:
            values[f"S{i}"] = str(Fraction(eval(expr)))  # noqa: S307 - digits and ops only
        except Exception:  # noqa: BLE001
            return None
    return Fraction(values[f"S{len(steps) - 1}"]) if steps else None


def generalise(question, steps):
    """Numbers to variables, identically in text and plan; round trip must reproduce both."""
    qnums = [norm(m) for m in NUM.findall(norm(question))]
    tsteps, used = [], set()
    for s in steps:
        out, pos = "", 0
        for m in NUM.finditer(s):
            v = norm(m.group(0))
            idx = next((k for k, q in enumerate(qnums) if q == v), None)
            out += s[pos:m.start()] + (f"v{idx + 1}" if idx is not None else m.group(0))
            if idx is not None:
                used.add(idx)
            pos = m.end()
        tsteps.append(out + s[pos:])
    binding = {f"v{k + 1}": qnums[k] for k in range(len(qnums))}
    back = []
    for s in tsteps:
        e = s
        for var in sorted(binding, key=len, reverse=True):
            e = re.sub(rf"\b{var}\b", binding[var], e)
        back.append(e)
    if back != steps:
        return None
    return {"steps": tsteps, "nvars": len(qnums), "vars_used": len(used)}


def build_store(limit=2000):
    store = []
    kept = failed_exec = failed_round = 0
    for line in TRAIN.read_text().splitlines()[:limit]:
        d = json.loads(line)
        truth = Fraction(norm(d["answer"].rsplit("#### ", 1)[-1].strip()))
        steps = extract_plan(d["answer"])
        if not steps:
            continue
        got = run_plan(steps)
        if got != truth:
            failed_exec += 1
            continue
        tpl = generalise(d["question"], steps)
        if tpl is None:
            failed_round += 1
            continue
        tpl.update({"question": d["question"], "masked": mask(norm(d["question"])),
                    "answer": str(truth)})
        store.append(tpl)
        kept += 1
    return store, kept, failed_exec, failed_round


def main(n_store=2000, n_test=60, seed=5, out="data/custom/mapstore.json"):
    n_store, n_test, seed = int(n_store), int(n_test), int(seed)
    store, kept, fe, fr = build_store(n_store)
    print(f"store: {kept} verified generalised plans from {n_store} training solutions "
          f"({fe} failed execution, {fr} failed the round trip)")

    print("embedding the store...", flush=True)
    vecs = np.array(embed([t["masked"] for t in store]), dtype=np.float32)

    # The compression-in-groups half: how many distinct TYPES do these plans collapse into?
    shapes = {}
    for t in store:
        shapes.setdefault("|".join(re.sub(r"v\d+|S\d+|[\d.]+", "#", s)
                                   for s in t["steps"]), []).append(t)
    print(f"the {kept} plans collapse into {len(shapes)} distinct plan shapes; "
          f"largest {max(len(v) for v in shapes.values())} members")

    import random
    rng = random.Random(seed)
    tests = []
    for line in TEST.read_text().splitlines():
        d = json.loads(line)
        tests.append((d["question"],
                      Fraction(norm(d["answer"].rsplit("#### ", 1)[-1].strip()))))
    tests = rng.sample(tests, n_test)

    qvecs = np.array(embed([mask(norm(q)) for q, _ in tests]), dtype=np.float32)
    hits = compatible = 0
    rows = []
    for i, (q, truth) in enumerate(tests):
        sims = vecs @ qvecs[i]
        order = np.argsort(sims)[::-1]
        qnums = [norm(m) for m in NUM.findall(norm(q))]
        # The nearest template whose variable count this problem can bind.
        pick = next((int(j) for j in order[:20]
                     if store[int(j)]["nvars"] == len(qnums)), None)
        if pick is None:
            rows.append({"q": q[:70], "stage": "no compatible template"})
            continue
        compatible += 1
        binding = {f"v{k + 1}": qnums[k] for k in range(len(qnums))}
        got = run_plan(store[pick]["steps"], binding)
        ok = got == truth
        hits += ok
        rows.append({"q": q[:70], "template": store[pick]["masked"][:70],
                     "sim": float(sims[pick]), "got": str(got), "truth": str(truth),
                     "ok": ok})

    n = len(tests)
    print(f"\nRECORD-MAP on {n} test problems, zero model calls:")
    print(f"  compatible template found : {compatible}/{n}")
    print(f"  solved by pure lookup     : {hits}/{n}")
    print("\nEvery solve here is: mask, embed, retrieve, bind, execute. The plan came from a")
    print("different problem of the same type, generalised once and reused; the arithmetic")
    print("never touched a model. What lookup cannot do is bridge a type gap — and how often")
    print("the nearest type fits is now a number rather than a hope.")
    summary = {"store_kept": kept, "store_failed_exec": fe, "store_failed_round": fr,
               "plan_shapes": len(shapes), "tests": n, "compatible": compatible,
               "solved": hits}
    Path(out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
