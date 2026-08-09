#!/usr/bin/env python3
"""Does harder arithmetic recruit more experts, or does compression collapse it anyway?

`op_structure.py` and `floor.py` ran on 300 one-digit problems, and the objection to those
results is fair: a task that small can survive almost any collapse, so "8 of 32 experts and
still exactly right" says as much about the task as about the objective. The way to find out is
to make the mathematics harder and watch whether the model *recruits*.

Two-digit operands over five operations is 46 800 problems against 300, and the answers span
four digits and a sign rather than one small class. The rule behind them is still small — that
is the point of arithmetic — but the number of distinct behaviours required is not.

The prediction being tested, stated so it can fail:

    experts actually used should grow with the number of operations,
    and should keep growing when the floor term is present.

If it does not, the compression objective is finding one partition regardless of how much
structure the task contains, and the one-digit collapse was not an artefact of a small task.

Answers are emitted digit by digit — sign, thousands, hundreds, tens, units — and a problem
counts as solved only when all five heads are right. Nothing partial.

Run with a torch venv, e.g. /Users/punnerud/Downloads/ainmt/venv/bin/python3.
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE, WIDTH = 10, 2                # two-digit operands
N_OP_ALL = 5                       # +  -  x  //  %
OPNAME = ("+", "-", "x", "//", "%")
D, N_EXPERT, TOP_K, BLOCK = 96, 64, 4, 4
N_BLOCK = N_EXPERT // BLOCK
IN_DIM = 2 * WIDTH * BASE + N_OP_ALL


def apply_op(a, b, op):
    if op == 0:
        return a + b
    if op == 1:
        return a - b
    if op == 2:
        return a * b
    if op == 3:
        return a // b
    return a % b


def problems(n_op):
    """Every (a, op, b) for the first `n_op` operations. b > 0 where the op needs it."""
    xs, ops, signs, digs = [], [], [], []
    hi = BASE**WIDTH
    for op in range(n_op):
        lo_b = 1 if op >= 3 else 0
        for a in range(hi):
            for b in range(lo_b, hi):
                v = apply_op(a, b, op)
                x = torch.zeros(IN_DIM)
                for k in range(WIDTH):                       # digits of a, then of b
                    x[k * BASE + (a // BASE**k) % BASE] = 1.0
                    x[(WIDTH + k) * BASE + (b // BASE**k) % BASE] = 1.0
                x[2 * WIDTH * BASE + op] = 1.0
                xs.append(x)
                ops.append(op)
                signs.append(1 if v < 0 else 0)
                m = abs(v)
                digs.append([(m // 1000) % 10, (m // 100) % 10, (m // 10) % 10, m % 10])
    return (torch.stack(xs), torch.tensor(ops), torch.tensor(signs),
            torch.tensor(digs))


class BigMoE(nn.Module):
    """One MoE block applied `hops` times, then five digit heads."""

    def __init__(self, hops=2):
        super().__init__()
        self.hops = hops
        self.inp = nn.Linear(IN_DIM, D)
        self.router = nn.Linear(D, N_EXPERT, bias=False)
        self.experts = nn.Parameter(torch.randn(N_EXPERT, D, D) * (1.0 / D**0.5))
        self.norm = nn.LayerNorm(D)
        self.sign = nn.Linear(D, 2)
        self.digits = nn.ModuleList([nn.Linear(D, BASE) for _ in range(4)])

    def forward(self, x):
        h = torch.relu(self.inp(x))
        picks, probs_all = [], []
        for _ in range(self.hops):
            probs = self.router(h).softmax(-1)
            w, idx = probs.topk(TOP_K, dim=-1)
            y = torch.einsum("bkij,bj->bki", self.experts[idx], h)
            h = self.norm(h + (y * w.unsqueeze(-1)).sum(1))
            picks.append(idx)
            probs_all.append(probs)
        return (self.sign(h), [d(h) for d in self.digits],
                torch.stack(picks, 1), torch.stack(probs_all, 1))


def compression_loss(probs):
    m = probs.view(*probs.shape[:-1], N_BLOCK, BLOCK).sum(-1).clamp(1e-6, 1.0)
    return (1.0 - (1.0 - m).pow(TOP_K)).sum(-1).mean()


def separation(probs, ops, n_op):
    """Mean pairwise total-variation distance between per-operation routing profiles."""
    prof = [probs[ops == o].mean(0) for o in range(n_op) if (ops == o).any()]
    if len(prof) < 2:
        return probs.sum() * 0.0
    d = [0.5 * (prof[a] - prof[b]).abs().sum()
         for a in range(len(prof)) for b in range(a + 1, len(prof))]
    return torch.stack(d).mean()


def evaluate(m, x, sg, dg, chunk=4096):
    ok = torch.zeros(len(x), dtype=torch.bool)
    picks = []
    with torch.no_grad():
        for i in range(0, len(x), chunk):
            s, ds, pk, _ = m(x[i:i + chunk])
            good = s.argmax(-1) == sg[i:i + chunk]
            for j, dl in enumerate(ds):
                good &= dl.argmax(-1) == dg[i:i + chunk, j]
            ok[i:i + chunk] = good
            picks.append(pk)
    return float(ok.float().mean()), torch.cat(picks)


def run(n_op, lam, beta, steps, hops, dev):
    torch.manual_seed(0)
    x, ops, sg, dg = problems(n_op)
    x, ops, sg, dg = x.to(dev), ops.to(dev), sg.to(dev), dg.to(dev)
    m = BigMoE(hops).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    t0 = time.time()
    for s in range(steps):
        i = torch.randint(0, len(x), (512,), device=dev)
        sl, dl, _, probs = m(x[i])
        loss = F.cross_entropy(sl, sg[i])
        for j, d in enumerate(dl):
            loss = loss + F.cross_entropy(d, dg[i, j])
        if lam:
            loss = loss + lam * compression_loss(probs)
        if beta:
            loss = loss - beta * separation(probs[:, 0], ops[i], n_op)
        opt.zero_grad()
        loss.backward()
        opt.step()

    acc, picks = evaluate(m, x, sg, dg)
    flat = picks.reshape(len(x), -1)
    used = int(len(torch.unique(flat)))
    blocks = float(torch.tensor(
        [len({int(e) // BLOCK for e in row}) for row in flat[:4000].tolist()],
        dtype=torch.float).mean())
    # Per-operation expert profiles, for the overlap that showed the collapse before.
    use = torch.zeros(n_op, N_EXPERT)
    for o in range(n_op):
        sel = flat[ops == o]
        if len(sel):
            use[o] = torch.bincount(sel.reshape(-1).cpu(), minlength=N_EXPERT).float()
    worst = 0.0
    for a in range(n_op):
        for b in range(a + 1, n_op):
            if use[a].sum() and use[b].sum():
                ua, ub = use[a] / use[a].sum(), use[b] / use[b].sum()
                worst = max(worst, float(torch.minimum(ua, ub).sum()))
    return {"n_op": n_op, "ops": list(OPNAME[:n_op]), "problems": len(x),
            "lambda": lam, "beta": beta, "hops": hops,
            "exact_accuracy": round(acc, 4), "experts_used": used,
            "blocks_per_problem": round(blocks, 3),
            "worst_pair_overlap": round(worst, 3),
            "seconds": round(time.time() - t0, 1)}


def main(steps=4000, out="data/custom/bigmath.json", hops=2):
    steps, hops = int(steps), int(hops)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"two-digit arithmetic, {N_EXPERT} experts in {N_BLOCK} blocks, top-{TOP_K}, "
          f"{hops} hops, device {dev}\n")
    print(f"{'ops':>16} {'problems':>9} {'lam':>5} {'beta':>5} {'exact acc':>10} "
          f"{'experts':>8} {'blocks':>7} {'worst ovl':>10} {'s':>6}")
    rows = []
    for n_op in (1, 2, 3, 5):
        for lam, beta in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)):
            r = run(n_op, lam, beta, steps, hops, dev)
            rows.append(r)
            print(f"{' '.join(r['ops']):>16} {r['problems']:>9} {lam:>5.1f} {beta:>5.1f} "
                  f"{r['exact_accuracy']:>10.3f} {r['experts_used']:>8d} "
                  f"{r['blocks_per_problem']:>7.2f} {r['worst_pair_overlap']:>10.3f} "
                  f"{r['seconds']:>6.0f}")
        print()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(
        {"n_expert": N_EXPERT, "block": BLOCK, "top_k": TOP_K, "hops": hops,
         "steps": steps, "rows": rows}, indent=2))
    print(f"wrote {out}")

    for lam, beta, label in ((1.0, 0.0, "compression only"), (1.0, 1.0, "with the floor")):
        seq = [r["experts_used"] for r in rows if r["lambda"] == lam and r["beta"] == beta]
        grew = seq[-1] > seq[0]
        print(f"{label:>18}: experts used {seq} — "
              f"{'recruits with difficulty' if grew else 'DOES NOT recruit'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
