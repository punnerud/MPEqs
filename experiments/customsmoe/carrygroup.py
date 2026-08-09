#!/usr/bin/env python3
"""Group the training data by carry count — the axis the embeddings could not see.

Clustering the problems with a pretrained embedder sorted them by surface form: 1.90x lift on
which operator is present, 1.57x on whether an operand is negative, and **1.15x on carry
count** — essentially blind to it. Carries are where held-out accuracy falls from 0.647 to
0.437, so the embedding groups by what the problems look like rather than by what is hard about
them.

The obvious next question is whether the grouping was the problem or the *source* of the
grouping was. Carry count is available from the solver, so we can group by it directly and stop
asking the embeddings for it. If that helps, the lesson is "the right partition helps, and
embeddings could not find it". If it does not help, then the partition was never the lever and
the embedding result was a red herring rather than a near miss.

Four arms, identical architecture, identical compute:

    uniform         sample as the problems naturally occur          the baseline
    balanced        equal mass to every carry bucket                hard cases oversampled
    easy-first      carries 0, then 1, then 2, then 3               the accelerator first
    hard-first      the reverse                                     direction is a control

Held-out accuracy is measured on the **natural** distribution in every case, so oversampling the
hard buckets has to pay for the easy ones it displaces.

Three seeds per arm, mean and spread, because single-seed comparisons in this project have
already produced one retracted result and the measured noise floor reaches sd 0.161.

Run with a torch venv, e.g. /Users/punnerud/Downloads/ainmt/venv/bin/python3.
"""
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dump_layers import (LayerMoE, TASKS, batch, block_compression, carries,  # noqa: E402
                         is_holdout)

SPEC = TASKS["add"]
A_LIM = 10 ** SPEC[0] - 1
MAX_CARRY = 4                      # buckets 0..3
POOL_PER_BUCKET = 40_000


def build_pools(seed=1234):
    """Pre-bucket a large pool once. Rejection sampling per step would stall on rare buckets."""
    rng = random.Random(seed)
    pools = [[] for _ in range(MAX_CARRY)]
    natural = [0] * MAX_CARRY
    tries = 0
    while min(len(p) for p in pools) < POOL_PER_BUCKET and tries < 40_000_000:
        tries += 1
        a, b = rng.randint(-A_LIM, A_LIM), rng.randint(-A_LIM, A_LIM)
        op = rng.randrange(2)
        if is_holdout(a, b, op, A_LIM, A_LIM):
            continue
        c = carries(a, b, op)
        if 0 <= c < MAX_CARRY:
            natural[c] += 1
            if len(pools[c]) < POOL_PER_BUCKET:
                pools[c].append((a, b, op))
    total = sum(natural)
    return pools, [v / total for v in natural]


def holdout(n, seed=999):
    """The natural distribution, never trained on. Every arm is scored on this same set."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        a, b = rng.randint(-A_LIM, A_LIM), rng.randint(-A_LIM, A_LIM)
        op = rng.randrange(2)
        if is_holdout(a, b, op, A_LIM, A_LIM):
            out.append((a, b, op))
    return out


def weights_for(arm, frac_done, natural):
    """Sampling mass per carry bucket at this point in training."""
    if arm == "uniform":
        return natural
    if arm == "balanced":
        return [1.0 / MAX_CARRY] * MAX_CARRY
    # Staged: a window that walks across the buckets, with a tail so nothing is dropped
    # outright — losing a bucket entirely would confound the schedule with forgetting.
    order = range(MAX_CARRY) if arm == "easy-first" else reversed(range(MAX_CARRY))
    order = list(order)
    pos = frac_done * MAX_CARRY
    w = [0.0] * MAX_CARRY
    for rank, bucket in enumerate(order):
        w[bucket] = max(0.15, 1.0 - abs(rank - pos))
    s = sum(w)
    return [v / s for v in w]


def run(arm, steps, pools, natural, hold, dev, seed, bs=384):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    m = LayerMoE(SPEC).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        w = weights_for(arm, s / steps, natural)
        items = []
        for _ in range(bs):
            u, acc_w, bucket = rng.random(), 0.0, MAX_CARRY - 1
            for c, wc in enumerate(w):
                acc_w += wc
                if u <= acc_w:
                    bucket = c
                    break
            items.append(pools[bucket][rng.randrange(len(pools[bucket]))])
        x, sg, dg = batch(items, SPEC, dev)
        sl, dl, _, probs, _ = m(x)
        loss = F.cross_entropy(sl, sg)
        for j, d in enumerate(dl):
            loss = loss + F.cross_entropy(d, dg[:, j])
        loss = loss + 0.3 * block_compression(probs)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    ok = torch.zeros(len(hold), dtype=torch.bool)
    for i in range(0, len(hold), 4096):
        part = hold[i:i + 4096]
        x, sg, dg = batch(part, SPEC, dev)
        with torch.no_grad():
            sl, dl, _, _, _ = m(x)
            good = sl.argmax(-1) == sg
            for j, d in enumerate(dl):
                good &= d.argmax(-1) == dg[:, j]
        ok[i:i + 4096] = good.cpu()
    by_carry = {}
    for (a, b, op), g in zip(hold, ok.tolist()):
        by_carry.setdefault(carries(a, b, op), []).append(g)
    return (float(ok.float().mean()),
            {str(k): round(sum(v) / len(v), 4) for k, v in sorted(by_carry.items())})


def main(steps=6000, seeds=3, out="data/custom/carrygroup.json", arms=None):
    steps, seeds = int(steps), int(seeds)
    # Three seeds put easy-first 8 pairwise wins out of 9 ahead of uniform at z = 2.9, which is
    # suggestive and nothing more at n = 3. The arm list is a parameter so the promising arms
    # can be re-run deeper without paying for the ones already settled.
    arm_list = arms.split(",") if arms else ["uniform", "balanced", "easy-first", "hard-first"]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print("building carry-bucketed pools…", flush=True)
    pools, natural = build_pools()
    hold = holdout(8000)
    nat_hold = {}
    for a, b, op in hold:
        nat_hold[carries(a, b, op)] = nat_hold.get(carries(a, b, op), 0) + 1
    print(f"natural carry mix: " +
          "  ".join(f"{c}:{100 * f:.0f}%" for c, f in enumerate(natural)))
    print(f"{steps} steps, {seeds} seeds per arm, held out on the natural mix, {dev}\n")
    print(f"{'arm':>12} {'accuracy':>16} {'0 carries':>10} {'1':>7} {'2':>7} {'3':>7}")

    rows = []
    for arm in arm_list:
        accs, per = [], []
        t0 = time.time()
        for seed in range(seeds):
            a, bc = run(arm, steps, pools, natural, hold, dev, seed)
            accs.append(a)
            per.append(bc)
        mean, sd = st.mean(accs), (st.pstdev(accs) if len(accs) > 1 else 0.0)
        avg = {k: round(st.mean([p[k] for p in per]), 3) for k in per[0]}
        rows.append({"arm": arm, "accs": [round(v, 4) for v in accs],
                     "mean": round(mean, 4), "sd": round(sd, 4),
                     "per_carry_mean": avg, "seconds": round(time.time() - t0, 1)})
        print(f"{arm:>12} {mean:>8.3f} +/- {sd:<5.3f} "
              f"{avg.get('0', 0):>10.3f} {avg.get('1', 0):>7.3f} "
              f"{avg.get('2', 0):>7.3f} {avg.get('3', 0):>7.3f}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(
        {"steps": steps, "seeds": seeds, "natural_carry_mix": [round(v, 4) for v in natural],
         "rows": rows}, indent=2))
    print(f"\nwrote {out}")

    base = next(r for r in rows if r["arm"] == "uniform")
    if seeds < 2:
        print("\nOne seed per arm: sd is 0 by construction, so no verdict is available. "
              "A comparison\nwithout a spread is exactly what produced this project's one "
              "retracted result.")
        return
    # Comparing a difference of MEANS against the spread of single runs is the wrong test, and
    # the first version of this did exactly that: it called a +0.226 gap with 34 of 36 pairwise
    # wins "inside the noise" because one arm had sd 0.114. The spread of individual runs is
    # not the uncertainty in their average. Two statistics are reported instead, one parametric
    # and one not, and they have to agree.
    import itertools

    print(f"\n{'arm':>12} {'diff':>7} {'sem':>7} {'z':>6} {'pairwise wins':>15} {'verdict':>18}")
    for r in rows:
        if r["arm"] == "uniform":
            continue
        d = r["mean"] - base["mean"]
        sem = (r["sd"] ** 2 / len(r["accs"]) + base["sd"] ** 2 / len(base["accs"])) ** 0.5
        z = d / sem if sem > 0 else 0.0
        n_pairs = len(r["accs"]) * len(base["accs"])
        wins = sum(1 for x, y in itertools.product(r["accs"], base["accs"]) if x > y)
        # Both must hold: z past 2, and a clear majority of head-to-head seed comparisons.
        ok = abs(z) > 2.0 and (wins >= 0.9 * n_pairs or wins <= 0.1 * n_pairs)
        verdict = "REAL" if ok else "not established"
        print(f"{r['arm']:>12} {d:>+7.3f} {sem:>7.3f} {z:>6.1f} "
              f"{f'{wins}/{n_pairs}':>15} {verdict:>18}")
    if seeds < 5:
        print("\nFewer than five seeds: treat any verdict above as provisional.")


if __name__ == "__main__":
    main(*sys.argv[1:])
