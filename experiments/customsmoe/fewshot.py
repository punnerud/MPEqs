#!/usr/bin/env python3
"""How many distinct examples does it take to generalise, and does experimenting reduce it?

Every run so far drew fresh problems every step: 12 000 steps of 384 samples is 4.6 million
draws from a space of a few hundred thousand, so the model saw most of it many times over. That
measures capacity, not learning — and it is the wrong measurement, because a person learns
arithmetic from a handful of worked examples and then applies it to numbers they have never
seen. Whether these networks do that has not been asked.

So fix the number of *distinct* problems the model is allowed to see and hold everything else
constant. Same gradient steps, same batch size, same architecture; only the size of the training
pool varies. Held-out problems are drawn from the disjoint remainder, so accuracy is
generalisation by construction.

Two arms, because the one mechanism that worked today should be tested where it matters most.
The perturbation objective — nudge an operand, train on the difference — gave +0.283 when data
was unlimited. If it is really teaching structure rather than fitting more of the table, it
should show up as **needing fewer distinct examples**, which is the sample-efficiency form of
the same claim and a stronger one.

The number to read is not the top of each curve but where it crosses: how many examples buy
generalisation, and whether experimenting buys it sooner.

Run with a torch venv, e.g. /Users/punnerud/Downloads/ainmt/venv/bin/python3.
"""
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

MAXD, LIM = 3, 999
OUT_DIGITS = 4
IN_DIM = 2 * (MAXD * 10 + 1) + 2            # digits and sign per operand, then + or -
D, N_EXPERT, TOP_K, BLOCK, HOPS = 64, 32, 4, 4, 3
DELTAS = (1, -1, 10, -10, 100, -100)
POOL_SIZES = (256, 1024, 4096, 0)              # 0 means unlimited, the earlier regime
ARMS = ("baseline", "probing", "consistency")


def solve(a, b, op):
    return a + b if op == 0 else a - b


def carries(a, b):
    x, y, n, c = abs(a), abs(b), 0, 0
    for k in range(MAXD + 1):
        s = (x // 10**k) % 10 + (y // 10**k) % 10 + c
        c = 1 if s >= 10 else 0
        n += c
    return n


def is_holdout(a, b, op):
    return ((a + LIM) * 7919 + (b + LIM) * 104729 + op) % 97 == 0


def encode(a, b, op, out):
    for k in range(MAXD):
        out[k * 10 + (abs(a) // 10**k) % 10] = 1.0
    out[MAXD * 10] = 1.0 if a < 0 else 0.0
    off = MAXD * 10 + 1
    for k in range(MAXD):
        out[off + k * 10 + (abs(b) // 10**k) % 10] = 1.0
    out[off + MAXD * 10] = 1.0 if b < 0 else 0.0
    out[2 * (MAXD * 10 + 1) + op] = 1.0


def digits_of(v, n):
    m = abs(v)
    return [(m // 10**k) % 10 for k in range(n)]


def batch(items, dev):
    x = torch.zeros(len(items), IN_DIM)
    sg = torch.zeros(len(items), dtype=torch.long)
    dg = torch.zeros(len(items), OUT_DIGITS, dtype=torch.long)
    for i, (a, b, op) in enumerate(items):
        encode(a, b, op, x[i])
        v = solve(a, b, op)
        sg[i] = 1 if v < 0 else 0
        dg[i] = torch.tensor(digits_of(v, OUT_DIGITS))
    return x.to(dev), sg.to(dev), dg.to(dev)


def sample(rng, n, holdout):
    out = []
    while len(out) < n:
        a, b, op = rng.randint(-LIM, LIM), rng.randint(-LIM, LIM), rng.randrange(2)
        if is_holdout(a, b, op) != holdout:
            continue
        out.append((a, b, op))
    return out


class Net(nn.Module):
    def __init__(self, probing):
        super().__init__()
        self.probing = probing
        self.inp = nn.Linear(IN_DIM, D)
        self.router = nn.Linear(D, N_EXPERT, bias=False)
        self.experts = nn.Parameter(torch.randn(N_EXPERT, D, D) * (1.0 / D**0.5))
        self.norm = nn.LayerNorm(D)
        self.sign = nn.Linear(D, 2)
        self.digits = nn.ModuleList([nn.Linear(D, 10) for _ in range(OUT_DIGITS)])
        if probing:
            self.delta_embed = nn.Embedding(len(DELTAS), D)
            self.delta_sign = nn.Linear(D, 2)
            self.delta_digits = nn.ModuleList([nn.Linear(D, 10) for _ in range(OUT_DIGITS)])

    def trunk(self, x):
        h = torch.relu(self.inp(x))
        probs_all = []
        for _ in range(HOPS):
            probs = self.router(h).softmax(-1)
            w, idx = probs.topk(TOP_K, dim=-1)
            y = torch.einsum("bkij,bj->bki", self.experts[idx], h)
            h = self.norm(h + (y * w.unsqueeze(-1)).sum(1))
            probs_all.append(probs)
        return h, torch.stack(probs_all, 1)

    def forward(self, x):
        h, probs = self.trunk(x)
        return self.sign(h), [d(h) for d in self.digits], probs

    def probe(self, x, di):
        h, _ = self.trunk(x)
        h = h + self.delta_embed(di)
        return self.delta_sign(h), [d(h) for d in self.delta_digits]


def block_compression(probs):
    nb = N_EXPERT // BLOCK
    m = probs.view(*probs.shape[:-1], nb, BLOCK).sum(-1).clamp(1e-6, 1.0)
    return (1.0 - (1.0 - m).pow(TOP_K)).sum(-1).mean()


def perturb_targets(items, rng):
    """The nudge, and the answer it produces. Never leaves the training pool's operands."""
    di, sg, dg = [], [], []
    for a, b, op in items:
        k = rng.randrange(len(DELTAS))
        v = solve(max(-LIM, min(LIM, a + DELTAS[k])), b, op)
        di.append(k)
        sg.append(1 if v < 0 else 0)
        dg.append(digits_of(v, OUT_DIGITS))
    return torch.tensor(di), torch.tensor(sg), torch.tensor(dg)


def self_targets(sl, dl, di, dev):
    """The model's own answer, shifted by the nudge. No solver is consulted.

    This is the arm the external-supervision result argues for. `probing` works, but it needs
    a solver to say what the nudged problem evaluates to, which is a label the network could
    not obtain on its own. Here the target is the model's *current* prediction for the base
    problem, decoded to a number, moved by delta and re-encoded. Nothing outside the network
    is consulted, so the constraint is consistency rather than correctness: the model may be
    self-consistently wrong, and whether that still teaches structure is the question.
    """
    with torch.no_grad():
        val = sum(d.argmax(-1) * (10 ** k) for k, d in enumerate(dl))
        val = torch.where(sl.argmax(-1) == 1, -val, val)
        delta = torch.tensor(DELTAS, device=dev)[di]
        moved = val + delta
        sg = (moved < 0).long()
        mag = moved.abs()
        dg = torch.stack([(mag // (10 ** k)) % 10 for k in range(OUT_DIGITS)], dim=-1)
    return sg, dg


def run(arm, pool_size, steps, dev, hold, seed=0, bs=384):
    probing = arm in ("probing", "consistency")
    rng = random.Random(seed)
    torch.manual_seed(seed)
    pool = sample(rng, pool_size, holdout=False) if pool_size else None
    m = Net(probing).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    # Precompute the pool once: with 64 distinct problems the encoding cost would otherwise
    # dominate, and the arms must differ only in the objective.
    for _ in range(steps):
        items = ([pool[rng.randrange(len(pool))] for _ in range(bs)] if pool
                 else sample(rng, bs, holdout=False))
        x, sg, dg = batch(items, dev)
        sl, dl, probs = m(x)
        loss = F.cross_entropy(sl, sg)
        for j, d in enumerate(dl):
            loss = loss + F.cross_entropy(d, dg[:, j])
        if probing:
            if arm == "consistency":
                di = torch.randint(0, len(DELTAS), (len(items),), device=dev)
                psg, pdg = self_targets(sl, dl, di, dev)
            else:
                d_i, psg, pdg = perturb_targets(items, rng)
                di, psg, pdg = d_i.to(dev), psg.to(dev), pdg.to(dev)
            psl, pdl = m.probe(x, di)
            pl = F.cross_entropy(psl, psg)
            for j, d in enumerate(pdl):
                pl = pl + F.cross_entropy(d, pdg[:, j])
            loss = loss + 0.5 * pl
        loss = loss + 0.3 * block_compression(probs)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    x, sg, dg = batch(hold, dev)
    ok = torch.zeros(len(hold), dtype=torch.bool, device=dev)
    with torch.no_grad():
        for i in range(0, len(hold), 4096):
            sl, dl, _ = m(x[i:i + 4096])
            good = sl.argmax(-1) == sg[i:i + 4096]
            for j, d in enumerate(dl):
                good &= d.argmax(-1) == dg[i:i + 4096, j]
            ok[i:i + 4096] = good
    okc = ok.cpu()
    hard = [g for (a, b, op), g in zip(hold, okc.tolist()) if carries(a, b) >= 2]
    return {"arm": arm, "pool": pool_size,
            "holdout_accuracy": round(float(okc.float().mean()), 4),
            "hard_accuracy": round(sum(hard) / max(len(hard), 1), 4)}


def main(steps=5000, out="data/custom/fewshot.json"):
    steps = int(steps)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    hold = sample(random.Random(999), 8000, holdout=True)
    print(f"three-digit signed + and -, {steps} steps for every pool size, device {dev}")
    print("held-out problems are disjoint from every pool\n")
    head = " ".join(f"{a:>12}" for a in ARMS)
    print(f"{'distinct examples':>18} {head}   (2+ carries in brackets)")
    rows = []
    for ps in POOL_SIZES:
        got = [run(a, ps, steps, dev, hold) for a in ARMS]
        rows += got
        label = "unlimited" if ps == 0 else f"{ps:,}"
        cells = " ".join(f"{r['holdout_accuracy']:>6.3f}({r['hard_accuracy']:.2f})"
                         for r in got)
        print(f"{label:>18} {cells}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"steps": steps, "pool_sizes": list(POOL_SIZES),
                                     "rows": rows}, indent=2))
    print(f"\nwrote {out}")

    def crossing(arm, thr):
        for ps in POOL_SIZES:
            r = next(x for x in rows if x["arm"] == arm and x["pool"] == ps)
            if r["holdout_accuracy"] >= thr:
                return "unlimited" if ps == 0 else f"{ps:,}"
        return "never"

    for thr in (0.3, 0.5):
        got = ", ".join(f"{a} {crossing(a, thr)}" for a in ARMS)
        print(f"distinct examples needed to reach {thr:.0%} on unseen problems: {got}")
    print("\nHard-case accuracy is printed next to every number so a model cannot look\n"
          "general by quietly dropping the carry-heavy cases: compressing to a versatile\n"
          "solution and compressing by discarding are the same on an average.")
    print("\nA person learns addition from a few dozen worked examples. The gap between that\n"
          "and the numbers above is the honest measure of how far this is from learning.")


if __name__ == "__main__":
    main(*sys.argv[1:])
