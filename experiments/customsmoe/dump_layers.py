#!/usr/bin/env python3
"""Dump every hop's hidden state, with the ground truth needed to check what they mean.

The landmark work needs the network's own embeddings, one file per layer, and it needs them
alongside the facts that make an example hard — carry count, operand widths, operation, and
whether the trained model actually gets it right. Without those the residual analysis in phase 4
has nothing to be validated against, and "high residual means new knowledge" would be a
definition rather than a claim.

Two tasks, chosen for the contrast:

    add    three-digit signed + and -   the model reaches ~0.76 held out
    mul    three-digit by two-digit x   the model reaches ~0.05, and every attempt to fix it
                                        (slot memory, partial products, running sums) failed

If the road network looks the same on a task the model has learned and one it has not, it is not
measuring what the network knows.

Embeddings are written in the repository's existing convention — headerless `n x dim` f32,
row-major, dim passed separately — so `matstruct from-embeddings` reads them unchanged.

Run with a torch venv, e.g. /Users/punnerud/Downloads/ainmt/venv/bin/python3.
"""
import json
import random
import struct
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

D, N_EXPERT, TOP_K, BLOCK, HOPS = 64, 32, 4, 4, 3
HOLDOUT_MOD = 97

TASKS = {
    # name: (a digits, b digits, operations, output digits)
    "add": (3, 3, (0, 1), 4),
    "mul": (3, 2, (2,), 5),
}
OPNAME = ("+", "-", "x")


def limits(spec):
    a_dig, b_dig, _, _ = spec
    return 10**a_dig - 1, 10**b_dig - 1


def solve(a, b, op):
    return (a + b, a - b, a * b)[op]


def carries(a, b, op):
    """Carries in the digit-wise magnitude addition. -1 where the notion does not apply."""
    if op > 1:
        return -1
    x, y, n, c = abs(a), abs(b), 0, 0
    for k in range(4):
        s = (x // 10**k) % 10 + (y // 10**k) % 10 + c
        c = 1 if s >= 10 else 0
        n += c
    return n


def is_holdout(a, b, op, a_lim, b_lim):
    return ((a + a_lim) * 7919 + (b + b_lim) * 104729 + op) % HOLDOUT_MOD == 0


def in_dim(spec):
    a_dig, b_dig, ops, _ = spec
    return (a_dig * 10 + 1) + (b_dig * 10 + 1) + len(OPNAME)


def encode(a, b, op, spec, out):
    a_dig, b_dig, _, _ = spec
    for k in range(a_dig):
        out[k * 10 + (abs(a) // 10**k) % 10] = 1.0
    out[a_dig * 10] = 1.0 if a < 0 else 0.0
    off = a_dig * 10 + 1
    for k in range(b_dig):
        out[off + k * 10 + (abs(b) // 10**k) % 10] = 1.0
    out[off + b_dig * 10] = 1.0 if b < 0 else 0.0
    out[(a_dig + b_dig) * 10 + 2 + op] = 1.0


def batch(items, spec, dev):
    _, _, _, n_out = spec
    x = torch.zeros(len(items), in_dim(spec))
    sg = torch.zeros(len(items), dtype=torch.long)
    dg = torch.zeros(len(items), n_out, dtype=torch.long)
    for i, (a, b, op) in enumerate(items):
        encode(a, b, op, spec, x[i])
        v = solve(a, b, op)
        sg[i] = 1 if v < 0 else 0
        m = abs(v)
        for k in range(n_out):
            dg[i, k] = (m // 10**k) % 10
    return x.to(dev), sg.to(dev), dg.to(dev)


def sample(rng, n, spec, holdout):
    a_lim, b_lim = limits(spec)
    ops = spec[2]
    out = []
    while len(out) < n:
        a = rng.randint(-a_lim, a_lim)
        b = rng.randint(-b_lim, b_lim)
        op = ops[rng.randrange(len(ops))]
        if is_holdout(a, b, op, a_lim, b_lim) != holdout:
            continue
        out.append((a, b, op))
    return out


class LayerMoE(nn.Module):
    """The same block as everywhere else, but it hands back every hop's state."""

    def __init__(self, spec):
        super().__init__()
        _, _, _, n_out = spec
        self.inp = nn.Linear(in_dim(spec), D)
        self.router = nn.Linear(D, N_EXPERT, bias=False)
        self.experts = nn.Parameter(torch.randn(N_EXPERT, D, D) * (1.0 / D**0.5))
        self.norm = nn.LayerNorm(D)
        self.sign = nn.Linear(D, 2)
        self.digits = nn.ModuleList([nn.Linear(D, 10) for _ in range(n_out)])

    def forward(self, x):
        h = torch.relu(self.inp(x))
        states, probs_all, picks = [], [], []
        for _ in range(HOPS):
            probs = self.router(h).softmax(-1)
            w, idx = probs.topk(TOP_K, dim=-1)
            y = torch.einsum("bkij,bj->bki", self.experts[idx], h)
            h = self.norm(h + (y * w.unsqueeze(-1)).sum(1))
            states.append(h)
            probs_all.append(probs)
            picks.append(idx)
        return (self.sign(h), [d(h) for d in self.digits],
                torch.stack(states, 1), torch.stack(probs_all, 1), torch.stack(picks, 1))


def block_compression(probs):
    nb = N_EXPERT // BLOCK
    m = probs.view(*probs.shape[:-1], nb, BLOCK).sum(-1).clamp(1e-6, 1.0)
    return (1.0 - (1.0 - m).pow(TOP_K)).sum(-1).mean()


def train(spec, steps, dev, seed=0, bs=384):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    m = LayerMoE(spec).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        items = sample(rng, bs, spec, holdout=False)
        x, sg, dg = batch(items, spec, dev)
        sl, dl, _, probs, _ = m(x)
        loss = F.cross_entropy(sl, sg)
        for j, d in enumerate(dl):
            loss = loss + F.cross_entropy(d, dg[:, j])
        loss = loss + 0.3 * block_compression(probs)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m


def dump(m, items, spec, dev, out_dir, task, chunk=4096):
    """Write one f32 per hop, plus a 5-byte-per-example ground-truth record."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [open(out_dir / f"{task}-L{k}.f32", "wb") for k in range(HOPS)]
    meta = open(out_dir / f"{task}-meta.bin", "wb")
    n_ok = 0
    for i in range(0, len(items), chunk):
        part = items[i:i + chunk]
        x, sg, dg = batch(part, spec, dev)
        with torch.no_grad():
            sl, dl, states, _, _ = m(x)
            good = sl.argmax(-1) == sg
            for j, d in enumerate(dl):
                good &= d.argmax(-1) == dg[:, j]
        n_ok += int(good.sum())
        st = states.to("cpu").float()                     # [B, HOPS, D]
        for k in range(HOPS):
            files[k].write(st[:, k].contiguous().numpy().astype("<f4").tobytes())
        for (a, b, op), g in zip(part, good.tolist()):
            meta.write(struct.pack("<5B", carries(a, b, op) & 0xFF, op,
                                   len(str(abs(a))), len(str(abs(b))), 1 if g else 0))
    for f in files:
        f.close()
    meta.close()
    return n_ok / len(items)


def main(n=50000, steps=9600, out="data/custom/layers", tasks="add,mul"):
    n, steps = int(n), int(steps)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    out_dir = Path(out)
    print(f"{HOPS} hops, D={D}, {N_EXPERT} experts, device {dev}")
    print(f"dumping {n:,} held-out examples per task "
          f"({n * D * 4 / 2**20:.1f} MiB per layer)\n")
    summary = {}
    for task in tasks.split(","):
        spec = TASKS[task]
        t0 = time.time()
        m = train(spec, steps, dev)
        items = sample(random.Random(999), n, spec, holdout=True)
        acc = dump(m, items, spec, dev, out_dir, task)
        summary[task] = {
            "n": n, "dim": D, "hops": HOPS, "steps": steps,
            "a_digits": spec[0], "b_digits": spec[1],
            "ops": [OPNAME[o] for o in spec[2]],
            "holdout_accuracy": round(acc, 4),
            "meta_fields": ["carries", "op", "a_digits", "b_digits", "correct"],
            "layer_files": [f"{task}-L{k}.f32" for k in range(HOPS)],
            "seconds": round(time.time() - t0, 1),
        }
        print(f"{task:>5}  {' '.join(summary[task]['ops']):<5} "
              f"holdout accuracy {acc:.3f}  ({time.time() - t0:.0f}s)")
    (out_dir / "layers.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir}/layers.json and {HOPS * len(summary)} layer files")
    print("The two accuracies should be far apart — that contrast is what phase 3 reads.")


if __name__ == "__main__":
    main(*sys.argv[1:])
