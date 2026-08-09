#!/usr/bin/env python3
"""Does more iteration help or hurt? The central claim, tested without a model.

The claim is that with this approach a model can think for a long time and iterate a lot
without it destroying the answer. That is a property of the RECORD, not of the driver, so it
can be tested with a scripted agent proposing at random — which also removes the confound that
sank phase 13, where every measurement was really measuring a 1B model's inability to choose.

A random proposer is the honest stand-in for "thinks a lot, often wrongly". If the record makes
even a random driver converge, and keeps it converged as the budget grows, the claim holds for
the architecture. If quality peaks and then falls, iteration destroys the answer and the claim
is false however good the driver.

Two arms, same proposals from the same seed:

  NAIVE    apply whatever is proposed, keep the latest value as the answer. No refusals, no
           memory, no invalidation — a scratchpad that is only a scratchpad.
  RECORD   the WorkPad, append-only: operands must be live, repeats are refused and remembered.
  BACKTRACK the same record, plus `undo()` when no live pair remains untried. Refusing a step
           is only half of protection; without a way back, one wrong move locks the state.

Reported two ways, because a solved-rate alone compares different things. The naive arm ends
every run by naming its most recent value, so it always CLAIMS an answer and is right about a
fifth of the time by coincidence. The record only claims when a complete derivation exists. The
comparison that means something is therefore:

  claim rate   how often the arm asserts an answer at all
  precision    of the answers it asserts, how many are right

"Iterating without destroying the answer" is a statement about precision holding up as the
budget grows, not about a lucky value appearing somewhere in a pool.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from measure_loops import TASKS  # noqa: E402
from workpad import OPS, Refused, WorkPad  # noqa: E402

BUDGETS = (10, 25, 50, 100, 200, 400)


def naive_run(problem, answer, budget, rng):
    """No record. Every proposal is applied to a flat pool, the newest value is the answer.

    This is what iterating without protection looks like: a wrong step overwrites a right one,
    and nothing remembers that the wrong step was already tried.
    """
    import re
    pool = [float(t) for t in re.findall(r"\d+(?:\.\d+)?", problem)]
    best = None
    for _ in range(budget):
        if len(pool) < 2:
            break
        i, j = rng.randrange(len(pool)), rng.randrange(len(pool))
        if i == j:
            continue
        op = rng.choice("+-*/")
        a, b = pool[i], pool[j]
        if op == "/" and b == 0:
            continue
        val = OPS[op](a, b)
        # No invalidation and no consumption: the pool just grows and the answer drifts.
        pool.append(val)
        best = val
    # It always names something — the latest value — which is why its claim rate is 1.0.
    return (best is not None), (best is not None and abs(best - answer) < 1e-6)


def record_run(problem, answer, budget, rng, backtrack=False):
    """With the record. Refusals, memory, and a claim only when everything is consumed."""
    import re
    givens = [float(t) for t in re.findall(r"\d+(?:\.\d+)?", problem)]
    pad = WorkPad(problem, givens)
    stuck = 0
    for step in range(budget):
        live = pad.live()
        if len(live) < 2:
            # A dead end, not an ending. Every given is consumed and the answer is wrong, so
            # the only move left is back. Without this the run stops after four steps whatever
            # the budget, which is why both record arms scored exactly 0.000 at every budget —
            # they were never given the chance to iterate at all.
            if backtrack and pad.undo() is not None:
                continue
            break
        a = rng.choice(live)["id"]
        b = rng.choice(live)["id"]
        if a == b:
            continue
        try:
            rid = pad.append(a, rng.choice("+-*/"), b)
        except Refused:
            stuck += 1
            # Everything reachable from here has been tried: step back one and try elsewhere.
            if backtrack and stuck > 8:
                pad.undo()
                stuck = 0
            continue
        stuck = 0
        rem = pad.live()
        if len(rem) == 1:
            # A complete derivation: every given consumed exactly once. This is the only point
            # at which the record is willing to assert anything.
            return True, abs(pad.read(rid)["value"] - answer) < 1e-6
    return False, False


def main(trials=20, out="data/custom/iterate.json"):
    trials = int(trials)
    print(f"{len(TASKS)} problems x {trials} random drivers, same proposals to both arms\n")
    print(f"{'budget':>8}{'naive':>15}{'record':>15}{'backtrack':>15}")
    print(f"{'':>8}" + "".join(f"{'claim/prec':>15}" for _ in range(3)))
    rows = []
    for budget in BUDGETS:
        stat = {k: [0, 0] for k in ("naive", "record", "backtrack")}
        for problem, answer in TASKS:
            for t in range(trials):
                seed = 1000 * t + budget
                for k, res in (("naive", naive_run(problem, answer, budget, random.Random(seed))),
                               ("record", record_run(problem, answer, budget,
                                                     random.Random(seed))),
                               ("backtrack", record_run(problem, answer, budget,
                                                        random.Random(seed), backtrack=True))):
                    stat[k][0] += res[0]
                    stat[k][1] += res[1]
        total = len(TASKS) * trials
        row = {"budget": budget, "trials": total}
        for k, (claims, right) in stat.items():
            row[k] = right / total
            row[f"{k}_claim_rate"] = claims / total
            row[f"{k}_precision"] = right / claims if claims else 0.0
        rows.append(row)
        print(f"{budget:>8}" + "".join(
            f"{row[k + '_claim_rate']:>7.2f}/{row[k + '_precision']:<7.2f}"
            for k in ("naive", "record", "backtrack")))

    def monotone(key):
        v = [r[key + "_precision"] for r in rows]
        return all(b >= a - 1e-9 for a, b in zip(v, v[1:])), max(v), v[-1]

    res = {k: monotone(k) for k in ("naive", "record", "backtrack")}
    for k, (m, mx, last) in res.items():
        print(f"{k:>10}: monotone {str(m):>5}   best {mx:.3f}   at the largest budget {last:.3f}")
    print("\nMonotone here is monotone PRECISION: of the answers an arm asserts, does the")
    print("share that are right hold up as it iterates longer. That is the claim.")
    Path(out).write_text(json.dumps(
        {"budgets": list(BUDGETS), "rows": rows,
         **{f"{k}_monotone": v[0] for k, v in res.items()},
         **{f"{k}_best": v[1] for k, v in res.items()},
         **{f"{k}_final": v[2] for k, v in res.items()}}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
