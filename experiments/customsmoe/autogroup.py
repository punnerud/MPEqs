#!/usr/bin/env python3
"""Can the carry partition be found without being told what a carry is?

Grouping training data by carry count is worth +0.157 to +0.226 held-out accuracy — the one
effect measured in this project that clears its own noise floor. The count came from the solver.
The question is whether anything the model can compute for itself recovers the same partition.

"Without ground truth" means something specific here and it is worth being exact about it. We
have the *answers* — it is a solver task, they are free. What we do not have is the *concept* of
a carry: nothing tells the model that 47 + 8 and 12 + 3 differ in a way that matters. Every
signal below is computed from the model and the answers, never from the carry count.

Six candidates, each bucketed into four to match the four carry levels, and each scored by how
much of the true partition it recovers:

    minilm-cluster     k-means over a pretrained sentence embedder     already known: 1.15x
    layer-cluster      k-means over our own network's final layer      geometry, our side
    entropy            predictive entropy of the answer heads          the model's doubt
    loss               per-example cross-entropy                       the model's error
    gradnorm           per-example gradient norm                       how much it still moves
    residual           MPEE: mean landmark-base residual per point     the skeleton's blind spot

`residual` is the MPEE signal the whole landmark thread was built for, computed the streaming
way: landmarks chosen by greedy facility location over sampled pairs, then each point's residual
averaged over a sample of partners. O(n·L + n·K), never n².

Lift is purity over the majority baseline. A signal at 1.0x has found nothing; the carry
partition itself would score 4.0x-ish by construction. Anything that clears the 1.15x the
pretrained embedding already manages is worth training on, and that is the next experiment
rather than this one.

Run with a torch venv, e.g. /Users/punnerud/Downloads/ainmt/venv/bin/python3.
"""
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dump_layers import (HOPS, LayerMoE, TASKS, batch, block_compression,  # noqa: E402
                         carries, is_holdout)

SPEC = TASKS["add"]
A_LIM = 10 ** SPEC[0] - 1
NBUCKET = 4


def pool(n, seed=1234):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        a, b = rng.randint(-A_LIM, A_LIM), rng.randint(-A_LIM, A_LIM)
        op = rng.randrange(2)
        if not is_holdout(a, b, op, A_LIM, A_LIM):
            out.append((a, b, op))
    return out


def train(steps, dev, seed=0, bs=384):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    m = LayerMoE(SPEC).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x, sg, dg = batch(pool(bs, rng.randrange(1 << 30)), SPEC, dev)
        sl, dl, _, probs, _ = m(x)
        loss = F.cross_entropy(sl, sg)
        for j, d in enumerate(dl):
            loss = loss + F.cross_entropy(d, dg[:, j])
        loss = loss + 0.3 * block_compression(probs)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m


def per_example(m, items, dev, chunk=2048):
    """Entropy, loss and final-layer state for every example. No labels beyond the answer."""
    ent, loss, states = [], [], []
    for i in range(0, len(items), chunk):
        x, sg, dg = batch(items[i:i + chunk], SPEC, dev)
        with torch.no_grad():
            sl, dl, st, _, _ = m(x)
            e = -(sl.softmax(-1) * sl.log_softmax(-1)).sum(-1)
            ll = F.cross_entropy(sl, sg, reduction="none")
            for j, d in enumerate(dl):
                e = e - (d.softmax(-1) * d.log_softmax(-1)).sum(-1)
                ll = ll + F.cross_entropy(d, dg[:, j], reduction="none")
            ent.append(e.cpu())
            loss.append(ll.cpu())
            states.append(st[:, HOPS - 1].cpu().float())
    return torch.cat(ent), torch.cat(loss), torch.cat(states)


def grad_norms(m, items, dev, group=32):
    """Gradient norm per small group — per-example would need a backward pass each.

    Grouping trades resolution for time and is honest about it: a group of 32 sharing one value
    can still separate buckets if the signal is strong, and cannot invent one if it is not.
    """
    out = []
    for i in range(0, len(items), group):
        x, sg, dg = batch(items[i:i + group], SPEC, dev)
        sl, dl, _, _, _ = m(x)
        loss = F.cross_entropy(sl, sg)
        for j, d in enumerate(dl):
            loss = loss + F.cross_entropy(d, dg[:, j])
        m.zero_grad()
        loss.backward()
        g = torch.sqrt(sum((p.grad**2).sum() for p in m.parameters() if p.grad is not None))
        out.extend([float(g)] * len(items[i:i + group]))
    m.zero_grad()
    return torch.tensor(out)


def landmark_residual(states, n_lm=32, sample=256, pair_sample=512, seed=0):
    """MPEE's signal: how badly the landmark skeleton explains each point.

    Landmarks by greedy facility location over sampled pairs, exactly as `pick_landmarks` does,
    then each point's mean residual `min_a d(i,a)+d(a,j) - d(i,j)` over a sample of partners j.
    Streaming in the sense that matters: nothing of size n^2 is ever formed.
    """
    x = torch.nn.functional.normalize(states, dim=-1)
    n = x.shape[0]
    g = torch.Generator().manual_seed(seed)
    ang = lambda a, b: 2.0 * torch.atan2(
        (a - b).norm(dim=-1), (a + b).norm(dim=-1)
    )

    cand = torch.randperm(n, generator=g)[: min(n, 4096)]
    pi = torch.randint(0, n, (pair_sample,), generator=g)
    pj = torch.randint(0, n, (pair_sample,), generator=g)
    d_ic = ang(x[pi].unsqueeze(1), x[cand].unsqueeze(0))          # [P, C]
    d_jc = ang(x[pj].unsqueeze(1), x[cand].unsqueeze(0))
    cur = torch.full((pair_sample,), float("inf"))
    chosen, used = [], torch.zeros(len(cand), dtype=torch.bool)
    for _ in range(n_lm):
        via = d_ic + d_jc                                          # [P, C]
        cost = torch.minimum(cur.unsqueeze(1), via).sum(0)
        cost[used] = float("inf")
        c = int(cost.argmin())
        used[c] = True
        chosen.append(int(cand[c]))
        cur = torch.minimum(cur, via[:, c])

    lm = x[torch.tensor(chosen)]                                   # [L, dim]
    d_all_lm = ang(x.unsqueeze(1), lm.unsqueeze(0))                # [n, L]
    part = torch.randperm(n, generator=g)[:sample]
    d_true = ang(x.unsqueeze(1), x[part].unsqueeze(0))             # [n, S]
    base = (d_all_lm.unsqueeze(2) + d_all_lm[part].t().unsqueeze(0)).min(1).values
    return (base - d_true).mean(1)


def kmeans_labels(x, k, iters=30, seed=0):
    torch.manual_seed(seed)
    x = torch.nn.functional.normalize(x, dim=-1)
    c = x[torch.randperm(x.shape[0])[:k]].clone()
    for _ in range(iters):
        lab = (x @ c.t()).argmax(-1)
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = torch.nn.functional.normalize(x[m].mean(0), dim=-1)
    return (x @ c.t()).argmax(-1).tolist()


def quantile_labels(v, k):
    """Split a scalar signal into k equal-count buckets."""
    order = torch.argsort(v)
    lab = torch.zeros(len(v), dtype=torch.long)
    for j in range(k):
        lab[order[j * len(v) // k:(j + 1) * len(v) // k]] = j
    return lab.tolist()


def lift(labels, truth, k):
    hit = 0
    for c in range(max(labels) + 1):
        vals = [truth[i] for i in range(len(truth)) if labels[i] == c]
        if vals:
            hit += Counter(vals).most_common(1)[0][1]
    purity = hit / len(truth)
    base = max(Counter(truth).values()) / len(truth)
    return purity, base, purity / base


def separability(sig, truth, k):
    """Between-group over within-group variance: why a signal can be real and still useless.

    A signal can shift reliably with carries at the population level and still fail to sort
    individual examples, if the spread inside each carry level swamps the gap between levels.
    Purity alone cannot tell those apart, and they call for opposite conclusions — the first
    says "look harder for a better estimator", the second says "there is nothing to sort by".
    """
    means, sizes, within = [], [], 0.0
    for c in range(k):
        v = torch.tensor([float(sig[i]) for i in range(len(truth)) if truth[i] == c])
        if len(v) < 2:
            continue
        means.append(float(v.mean()))
        sizes.append(len(v))
        within += float(v.var()) * len(v)
    if len(means) < 2:
        return 0.0, []
    grand = sum(m_ * s for m_, s in zip(means, sizes)) / sum(sizes)
    between = sum(s * (m_ - grand) ** 2 for m_, s in zip(means, sizes))
    within /= sum(sizes)
    return (between / sum(sizes)) / max(within, 1e-12), [round(m_, 4) for m_ in means]


def analyse(m, items, truth, dev, label, n):
    ent, loss, states = per_example(m, items, dev)
    gn = grad_norms(m, items, dev)
    resid = landmark_residual(states)

    minilm = None
    p = Path("data/custom/arith-minilm.f32")
    if p.exists():
        e = torch.frombuffer(bytearray(p.read_bytes()), dtype=torch.float32).view(-1, 384)
        if e.shape[0] >= n:
            minilm = e[:n]

    cands = {
        "layer-cluster": (kmeans_labels(states, NBUCKET), None),
        "entropy": (quantile_labels(ent, NBUCKET), ent),
        "loss": (quantile_labels(loss, NBUCKET), loss),
        "gradnorm": (quantile_labels(gn, NBUCKET), gn),
        "residual (MPEE)": (quantile_labels(resid, NBUCKET), resid),
    }
    if minilm is not None:
        cands["minilm-cluster"] = (kmeans_labels(minilm, NBUCKET), None)

    rows = []
    for name, (lab, sig) in cands.items():
        pur, base, lf = lift(lab, truth, NBUCKET)
        sep, means = separability(sig, truth, NBUCKET) if sig is not None else (None, [])
        rows.append({"stage": label, "signal": name, "purity": round(pur, 4),
                     "majority": round(base, 4), "lift": round(lf, 4),
                     "between_over_within": (round(sep, 5) if sep is not None else None),
                     "mean_by_carry": means})
        sp = f"{sep:>10.4f}" if sep is not None else f"{'-':>10}"
        print(f"{label:>7} {name:>18} {lf:>7.2f}x {sp}   {means}")
    return rows


def main(n=8000, out="data/custom/autogroup.json", stages="1200,6000,19200"):
    n = int(n)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    items = pool(n)
    truth = [carries(a, b, op) for a, b, op in items]
    print(f"{n:,} problems, {dev}")
    print(f"true carry mix: {dict(sorted(Counter(truth).items()))}")
    print("\nlift 1.0 means the signal sorts nothing. between/within is the effect size a")
    print("population-level difference would show even when per-example sorting fails.\n")
    print(f"{'steps':>7} {'signal':>18} {'lift':>8} {'betw/with':>10}   mean by carry 0..3")

    rows = []
    for stage in stages.split(","):
        steps = int(stage)
        m = train(steps, dev)
        acc_items = pool(2000, seed=999)
        x, sg, dg = batch(acc_items, SPEC, dev)
        with torch.no_grad():
            sl, dl, _, _, _ = m(x)
            good = sl.argmax(-1) == sg
            for j, d in enumerate(dl):
                good &= d.argmax(-1) == dg[:, j]
        print(f"{steps:>7}  (training accuracy {float(good.float().mean()):.3f})")
        rows += analyse(m, items, truth, dev, stage, n)
        print()

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"n": n, "buckets": NBUCKET, "rows": rows}, indent=2))
    print(f"wrote {out}")
    best = max(rows, key=lambda r: r["lift"])
    print(f"\nbest anywhere: {best['signal']} at {best['stage']} steps, {best['lift']:.2f}x. "
          f"The pretrained\nembedding managed 1.15x and the true partition is worth +0.157 to "
          f"+0.226 accuracy.")


if __name__ == "__main__":
    main(*sys.argv[1:])
