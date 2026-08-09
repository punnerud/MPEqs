#!/usr/bin/env python3
"""Learning something new without losing what the same experts already carried.

Adding knowledge to a routed network has a failure mode that a dense network does not: the new
task lands on particular experts, those experts are updated, and whatever *else* rode on them
degrades. Full replay avoids it and costs everything. Doing nothing is free and forgets.

The third option is to ask the network which of its old knowledge shares the experts the new
task is about to disturb, and rehearse only that. Concretely: measure which experts the new task
routes to, then run the routing graph **backwards** — for every old bucket, how much of its
expert usage overlaps — and rehearse the buckets that overlap most. It is the co-activation
graph read in reverse, and it is targeted rather than exhaustive, which is the whole point:
less rehearsal, a smaller network, and only the affected knowledge revisited.

Three arms, identical compute in the update phase:

    new-only     train on the new operation alone            forgetting, the lower bound
    full-replay  mix the new operation with old, uniformly   the upper bound, and expensive
    dream        mix with the old buckets that share experts targeted, and the claim

The measurement that matters is not "does dream beat new-only" — anything with rehearsal beats
that. It is whether dream reaches full-replay's retention while touching a fraction of the old
buckets. If it needs all of them anyway, the routing graph told us nothing.

Run with a torch venv, e.g. /Users/punnerud/Downloads/ainmt/venv/bin/python3.
"""
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from curriculum import (D, N_BUCKET, N_EXPERT, OPNAME, CurricMoE, assess, batch,
                        block_compression, holdout_set, sample_bucket)

OLD_OPS = (0, 1)          # + and -, learned first
NEW_OP = 2                # x, introduced afterwards


def bucket_usage(m, buckets, rng, dev, per_bucket=96):
    """Expert-usage histogram per bucket, normalised. This is the graph that gets reversed."""
    use = torch.zeros(len(buckets), N_EXPERT)
    for i, bk in enumerate(buckets):
        items = [s for s in (sample_bucket(bk, rng) for _ in range(per_bucket)) if s]
        if not items:
            continue
        x, _, _ = batch(items, dev)
        with torch.no_grad():
            picks = m(x)[3].reshape(-1).cpu()
        use[i] = torch.bincount(picks, minlength=N_EXPERT).float()
        if use[i].sum():
            use[i] /= use[i].sum()
    return use


def train(m, opt, steps, bucket_pool, rng, dev, batch_size=384):
    for _ in range(steps):
        items = []
        for _ in range(batch_size):
            s = sample_bucket(bucket_pool[rng.randrange(len(bucket_pool))], rng)
            if s:
                items.append(s)
        if len(items) < 32:
            continue
        x, sg, dg = batch(items, dev)
        sl, dl, probs, _ = m(x)
        loss = F.cross_entropy(sl, sg)
        for j, d in enumerate(dl):
            loss = loss + F.cross_entropy(d, dg[:, j])
        loss = loss + 0.3 * block_compression(probs)
        opt.zero_grad(); loss.backward(); opt.step()


def buckets_for(ops):
    return [bk for bk in range(N_BUCKET) if bk // 36 in ops]


def main(base_steps=2500, update_steps=1200, top_share=0.25,
         out="data/custom/dreaming.json"):
    base_steps, update_steps = int(base_steps), int(update_steps)
    top_share = float(top_share)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = random.Random(0)
    torch.manual_seed(0)

    old_bk, new_bk = buckets_for(OLD_OPS), buckets_for((NEW_OP,))
    print(f"base: {' '.join(OPNAME[o] for o in OLD_OPS)} over {len(old_bk)} buckets, "
          f"{base_steps} steps")
    print(f"new:  {OPNAME[NEW_OP]} over {len(new_bk)} buckets, {update_steps} steps, "
          f"device {dev}\n")

    base = CurricMoE().to(dev)
    opt = torch.optim.Adam(base.parameters(), lr=2e-3)
    t0 = time.time()
    train(base, opt, base_steps, old_bk, rng, dev)
    hold_old = holdout_set(OLD_OPS, rng, n=6000)
    ok, _ = assess(base, hold_old, dev)
    base_acc = float(ok.float().mean())
    print(f"base holdout accuracy on {'/'.join(OPNAME[o] for o in OLD_OPS)}: "
          f"{base_acc:.3f}  ({time.time() - t0:.0f}s)\n")
    state = {k: v.clone() for k, v in base.state_dict().items()}

    # Reverse the routing graph: which old buckets share experts with the new operation?
    usage_old = bucket_usage(base, old_bk, rng, dev)
    usage_new = bucket_usage(base, new_bk, rng, dev).mean(0)
    overlap = torch.minimum(usage_old, usage_new.unsqueeze(0)).sum(-1)   # shared mass
    n_pick = max(1, int(len(old_bk) * top_share))
    picked = [old_bk[i] for i in overlap.topk(n_pick).indices.tolist()]
    print(f"reverse lookup: {n_pick} of {len(old_bk)} old buckets share experts with "
          f"{OPNAME[NEW_OP]}")
    print(f"  shared expert mass: picked {overlap.topk(n_pick).values.mean():.3f}, "
          f"rest {overlap.topk(len(old_bk)).values[n_pick:].mean():.3f}\n")

    hold_new = holdout_set((NEW_OP,), rng, n=6000)
    arms = {
        "new-only": new_bk,
        "full-replay": new_bk + old_bk,
        "dream": new_bk + picked,
    }
    rows = []
    print(f"{'arm':>12} {'old buckets':>12} {'old acc':>9} {'retained':>9} {'new acc':>9} "
          f"{'experts':>8}")
    for name, pool in arms.items():
        m = CurricMoE().to(dev)
        m.load_state_dict(state)
        o = torch.optim.Adam(m.parameters(), lr=2e-3)
        train(m, o, update_steps, pool, random.Random(1), dev)
        ok_o, _ = assess(m, hold_old, dev)
        ok_n, _ = assess(m, hold_new, dev)
        a_o, a_n = float(ok_o.float().mean()), float(ok_n.float().mean())
        x, _, _ = batch(hold_old[:4096], dev)
        with torch.no_grad():
            used = len(set(m(x)[3].reshape(-1).tolist()))
        n_old = len(set(pool) & set(old_bk))
        rows.append({"arm": name, "old_buckets_rehearsed": n_old,
                     "old_accuracy": round(a_o, 4),
                     "retained_fraction": round(a_o / base_acc, 4) if base_acc else 0.0,
                     "new_accuracy": round(a_n, 4), "experts_used": used})
        print(f"{name:>12} {n_old:>12} {a_o:>9.3f} {a_o / max(base_acc, 1e-9):>9.3f} "
              f"{a_n:>9.3f} {used:>8}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(
        {"base_accuracy": round(base_acc, 4), "base_steps": base_steps,
         "update_steps": update_steps, "old_buckets": len(old_bk),
         "top_share": top_share, "rows": rows}, indent=2))
    print(f"\nwrote {out}")

    by = {r["arm"]: r for r in rows}
    gap = by["full-replay"]["old_accuracy"] - by["new-only"]["old_accuracy"]
    closed = ((by["dream"]["old_accuracy"] - by["new-only"]["old_accuracy"]) / gap
              if abs(gap) > 1e-6 else float("nan"))
    print(f"\nforgetting gap between full replay and no replay: {gap:+.3f}")
    print(f"dream closes {100 * closed:.0f}% of it while rehearsing "
          f"{by['dream']['old_buckets_rehearsed']}/{len(old_bk)} "
          f"({100 * by['dream']['old_buckets_rehearsed'] / len(old_bk):.0f}%) of the old "
          f"buckets.")
    print("Closing most of the gap on a quarter of the data means the routing graph knew\n"
          "which knowledge was at risk. Closing none of it means it did not.")


if __name__ == "__main__":
    main(*sys.argv[1:])
