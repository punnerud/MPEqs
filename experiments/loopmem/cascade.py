#!/usr/bin/env python3
"""The cascade: the 1B model only picks a formula and fills it; failure escalates to 35B.

The proposal, exactly: at 1B scale the model should not TRY to solve — it picks which known
plan shape fits and puts the values in, the record executes, and anything that does not
survive goes up a tier. Every component has a prior measurement: the small model fills shown
structures well (16/18 correct calls when it answers at all, phase 47) and plans terribly
(15/60, phase 60); agreement accepts with 100% observed precision (18/18, phase 53); and the
35B solo arm is the known ceiling here (51/60).

    TIER 1 (1B, cheap)   pick a shape from the store's eight commonest plan forms and bind
                         the values, one JSON; the record instantiates and executes exactly.
                         In parallel, a 1B solo answer. AGREEMENT between the two accepts.
    TIER 2 (35B)         solo with brief working, only for what tier 1 could not agree on.

The question is speed as much as accuracy, so wall time is measured per tier, and the
comparison is against running the 35B model on everything. The cascade wins if tier 1
accepts a decent share at high precision — phase 53 says unanimity is trustworthy, and this
is the cheapest unanimity there is: two different 1B paths through the same problem.
"""
import json
import re
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402
from mapstore import NUM, TEST, build_store, norm  # noqa: E402
from olympiad import SOLO, last_number  # noqa: E402

PICK = """Choose which formula shape fits this problem, and fill in the values in the order
the formula needs them. Do not solve anything. Use the numbers from the problem you are
given, not from the example.

Shapes:
{menu}

Example:
Problem: Tom has 3 boxes of 12 eggs and eats 5. How many eggs are left?
{{"shape": 2, "values": [3, 12, 5]}}

Problem: {problem}
"""


def shape_menu(store, top_n=8):
    """The store's commonest plan forms, as numbered readable shapes."""
    by_shape = Counter()
    example = {}
    for t in store:
        key = "|".join(re.sub(r"v\d+", "v", s) for s in t["steps"])
        by_shape[key] += 1
        example.setdefault(key, t)
    shapes = []
    for key, _ in by_shape.most_common(top_n):
        t = example[key]
        pretty = "; ".join(f"{chr(65 + i)} = {re.sub(r'S(\d+)', lambda m: chr(65 + int(m.group(1))), s)}"
                           for i, s in enumerate(t["steps"]))
        shapes.append({"steps": t["steps"], "nvars": t["nvars"], "pretty": pretty,
                       "members": by_shape[key]})
    menu = "\n".join(f"{i + 1}. {s['pretty']}   (needs {s['nvars']} values)"
                     for i, s in enumerate(shapes))
    return shapes, menu


def run_shape(steps, values):
    env = {}
    for i, s in enumerate(steps):
        expr = s
        for k in sorted(range(len(values)), reverse=True):
            expr = re.sub(rf"\bv{k + 1}\b", str(values[k]), expr)
        for j in range(i - 1, -1, -1):
            expr = re.sub(rf"\bS{j}\b", f"({env[j]})", expr)
        if not re.fullmatch(r"[\d\s+*/().-]+", expr):
            return None
        try:
            env[i] = Fraction(eval(expr))  # noqa: S307 - digits and operators only
        except Exception:  # noqa: BLE001
            return None
    return env[len(steps) - 1] if env else None


def main(n_test=60, seed=5, out="data/custom/cascade.json"):
    import random
    n_test, seed = int(n_test), int(seed)
    store, _, _, _ = build_store(2000)
    shapes, menu = shape_menu(store)
    covered = sum(s["members"] for s in shapes)
    print(f"menu: 8 shapes covering {covered}/{len(store)} store plans "
          f"({100 * covered / len(store):.0f}%)")

    tests = []
    for line in TEST.read_text().splitlines():
        d = json.loads(line)
        tests.append((d["question"],
                      Fraction(norm(d["answer"].rsplit("#### ", 1)[-1].strip()))))
    tests = random.Random(seed).sample(tests, n_test)

    t1_time = t2_time = 0.0
    stats = Counter()
    rows = []
    for q, truth in tests:
        t0 = time.monotonic()
        # Tier 1a: pick and fill. The 1B model never computes.
        reply = ask("olmoe-1b", PICK.format(menu=menu, problem=q), n=96)
        m = re.search(r'\{[^{}]*"shape"[^{}]*\}', reply)
        formula_ans = None
        if m:
            try:
                d = json.loads(m.group(0))
                sh = shapes[int(d["shape"]) - 1]
                vals = [Fraction(str(v)) for v in d["values"]]
                if len(vals) == sh["nvars"]:
                    formula_ans = run_shape(sh["steps"], vals)
            except Exception:  # noqa: BLE001 - a malformed pick escalates, that is the design
                pass
        # Tier 1b: the same model's direct answer — the second, independent 1B path.
        solo1 = last_number(ask("olmoe-1b", SOLO.format(problem=q), n=512))
        t1_time += time.monotonic() - t0

        agree = formula_ans is not None and solo1 is not None and formula_ans == solo1
        if agree:
            final, tier = formula_ans, 1
            stats["accepted_t1"] += 1
            stats["t1_right"] += final == truth
        else:
            t0 = time.monotonic()
            final = last_number(ask("qwen-35b", SOLO.format(problem=q), n=512))
            t2_time += time.monotonic() - t0
            tier = 2
            stats["escalated"] += 1
            stats["t2_right"] += final == truth
        stats["right"] += final == truth
        stats["formula_ran"] += formula_ans is not None
        rows.append({"truth": str(truth), "formula": str(formula_ans), "solo1": str(solo1),
                     "tier": tier, "final": str(final), "ok": final == truth})

    n = n_test
    total = t1_time + t2_time
    est_all35 = (t2_time / max(stats["escalated"], 1)) * n if stats["escalated"] else 0
    print(f"\ncascade on {n} problems:")
    print(f"  tier 1 accepted (1B paths agree) : {stats['accepted_t1']}/{n}, "
          f"right {stats['t1_right']}/{stats['accepted_t1']}")
    print(f"  escalated to 35B                 : {stats['escalated']}/{n}, "
          f"right {stats['t2_right']}/{stats['escalated']}")
    print(f"  TOTAL correct                    : {stats['right']}/{n}   "
          f"(35B everywhere: 51/60, 1B everywhere: 37/60)")
    print(f"  formula arm produced an answer   : {stats['formula_ran']}/{n}")
    print(f"\n  time: tier 1 {t1_time:.0f}s + tier 2 {t2_time:.0f}s = {total:.0f}s")
    if est_all35:
        print(f"  35B on everything would cost about {est_all35:.0f}s -> "
              f"{est_all35 / total:.2f}x the cascade")
    print("\nThe 1B model never solves: it names a shape and supplies values, agreement is")
    print("the gate, and the expensive model only sees what the cheap one could not settle.")
    summary = {"n": n, **{k: stats[k] for k in stats},
               "t1_seconds": round(t1_time, 1), "t2_seconds": round(t2_time, 1),
               "est_all35_seconds": round(est_all35, 1),
               "menu_coverage": covered / len(store)}
    Path(out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
