#!/usr/bin/env python3
"""One number across every task set: how much did the networks compress, while being right?

The experiments so far report accuracy and fetches side by side, and that hides the thing being
measured. Compression is the claim — a model has learned a task set when it can reproduce it
from less than the task set costs to write down — and accuracy is not a second axis, it is the
constraint the compression has to satisfy. A model that halves its footprint and answers wrong
has compressed nothing; it has discarded.

So both go into one quantity, summed over the networks:

    reproduced bits = (problems solved exactly) x log2(size of the answer space)
    stored bits     = (expert parameters actually used) x 32
    compression     = reproduced / stored

Only correctly solved problems count toward the numerator, so a wrong answer is worth zero
rather than partial credit — the same all-or-nothing rule the digit heads already use.

`stored` counts the experts the model actually routes to, not the ones it was given. An expert
that is never selected is not part of the description, and counting it would punish capacity
that costs nothing at inference. The router and the heads are counted too, since they are
resident for every problem.

Reads the JSON each experiment already writes; runs nothing.
"""
import json
import math
from pathlib import Path

BITS = 32


def load(p):
    q = Path(p)
    return json.loads(q.read_text()) if q.exists() else None


def rows():
    out = []

    d = load("data/custom/arith.json")
    if d:
        # 300 problems, answers span 91 classes.
        for r in d["rows"]:
            out.append(("1-digit a op b", d["n_problems"], math.log2(91),
                        r["exact_accuracy"], None, 64, 64,
                        f"{r['hops']} hops, lam={r['lambda']}",
                        r["block_fetches_per_problem"]))

    d = load("data/custom/bigmath.json")
    if d:
        for r in d["rows"]:
            # sign + 4 digits
            out.append((f"2-digit, {' '.join(r['ops'])}", r["problems"],
                        1 + 4 * math.log2(10), r["exact_accuracy"], r["experts_used"], 96, 96,
                        f"lam={r['lambda']} beta={r['beta']}", r["blocks_per_problem"]))

    d = load("data/custom/exprs.json")
    if d:
        for r in d["rows"]:
            out.append(("expressions + brackets", d["n_expr"], 1 + 3 * math.log2(10),
                        r["exact_accuracy"], r["experts_used"], 96, 96,
                        f"lam={r['lambda']} beta={r['beta']}", None))

    d = load("data/custom/equations.json")
    if d:
        for r in d["rows"]:
            out.append(("equations, solve for X", d["n_eq"], 1 + 2 * math.log2(10),
                        r["exact_accuracy"], r["experts_used"], 96, 96,
                        f"lam={r['lambda']} beta={r['beta']}", None))
    return out


def main(out="data/custom/compression.json"):
    print("Compression = bits of task data reproduced exactly, over bits of weight used.\n")
    print(f"{'task set':>24} {'problems':>9} {'exact':>7} {'experts':>8} "
          f"{'stored MiB':>11} {'ratio':>8} {'setting':>22}")
    recs = []
    for (name, n, abits, acc, used, d, _dd, setting, blocks) in rows():
        used = used if used else d          # arith.json predates the experts_used field
        stored = (used * d * d + d * 64 + d * 15) * BITS      # experts + router + heads
        repro = n * acc * abits
        ratio = repro / stored
        recs.append({"task_set": name, "problems": n, "answer_bits": round(abits, 2),
                     "exact_accuracy": acc, "experts_used": used,
                     "stored_bits": stored, "reproduced_bits": round(repro),
                     "compression": round(ratio, 4), "setting": setting,
                     "blocks_per_problem": blocks})
        print(f"{name:>24} {n:>9} {acc:>7.3f} {used:>8} {stored / 8 / 2**20:>11.2f} "
              f"{ratio:>8.3f} {setting:>22}")

    Path(out).write_text(json.dumps({"bits_per_param": BITS, "rows": recs}, indent=2))
    print(f"\nwrote {out}")

    # Summed over the networks, which is the question: on this whole body of tasks, how much
    # did the collection compress?
    tot_r = sum(r["reproduced_bits"] for r in recs)
    tot_s = sum(r["stored_bits"] for r in recs)
    print(f"\nover all {len(recs)} networks: {tot_r / 8 / 2**20:.2f} MiB reproduced from "
          f"{tot_s / 8 / 2**20:.2f} MiB stored — ratio {tot_r / tot_s:.3f}")
    best = max(recs, key=lambda r: r["compression"])
    print(f"best single: {best['task_set']} ({best['setting']}) at {best['compression']:.3f}")
    print("\nA ratio below 1 means the network is larger than the answers it reproduces — for\n"
          "these task sets that is expected, because the sets are small and the network is not.\n"
          "What matters is which settings move it, and in which direction.")


if __name__ == "__main__":
    main()
