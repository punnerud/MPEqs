#!/usr/bin/env python3
"""Exact arithmetic, solved by looping one expert layer instead of stacking many.

Two things this tests that the synthetic-group task cannot.

**Exactness.** `a op b` has one right answer, so accuracy is not a soft score — the model either
computed it or did not. And the rule behind the data is tiny: three operations over ten digits.
A model that has learned it should need very little capacity, which is the compression claim in
its sharpest form. Fitting 300 input/output pairs with 300 experts is memorisation; fitting them
while the access pattern collapses is learning.

**Looping.** One forward pass through one layer is not how arithmetic is done, and it is not how
a hop-based retrieval architecture would work either. Here the same MoE block is applied T
times, so the model can revisit — and, importantly, *re-fetch the same experts* on later hops.
If the loop works, T hops over a small resident set beats one hop over a large one, which is
exactly the trade this whole repository is about: fetch less, more often, from a smaller set.

Measures fetches as blocks touched summed over hops, and counts how many are repeats, because a
repeat is free once the block is resident.

Run with a torch venv, e.g. /Users/punnerud/Downloads/ainmt/venv/bin/python3.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DIGITS, OPS = 10, 3                       # a, b in 0..9;  +  -  x
N_CLASS = 10 * 9 + 9 + 1                  # results span -9 .. 81, shifted to 0 .. 90
D, N_EXPERT, TOP_K, BLOCK = 64, 32, 4, 4
N_BLOCK = N_EXPERT // BLOCK


def problems():
    """Every (a, op, b) exactly once. 300 problems, one exact answer each."""
    xs, ys = [], []
    for a in range(DIGITS):
        for b in range(DIGITS):
            for op in range(OPS):
                v = (a + b, a - b, a * b)[op]
                x = torch.zeros(D)
                x[a] = 1.0
                x[DIGITS + b] = 1.0
                x[2 * DIGITS + op] = 1.0
                xs.append(x)
                ys.append(v + 9)          # shift so the smallest result is class 0
    return torch.stack(xs), torch.tensor(ys)


class LoopedMoE(nn.Module):
    """One MoE block, applied `hops` times with shared weights."""

    def __init__(self, hops):
        super().__init__()
        self.hops = hops
        self.inp = nn.Linear(D, D)
        self.router = nn.Linear(D, N_EXPERT, bias=False)
        self.experts = nn.Parameter(torch.randn(N_EXPERT, D, D) * (1.0 / D**0.5))
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, N_CLASS)

    def forward(self, x):
        h = self.inp(x)
        picks, all_probs = [], []
        for _ in range(self.hops):
            probs = self.router(h).softmax(-1)
            w, idx = probs.topk(TOP_K, dim=-1)
            y = torch.einsum("bkij,bj->bki", self.experts[idx], h)
            h = self.norm(h + (y * w.unsqueeze(-1)).sum(1))   # residual: hops accumulate
            picks.append(idx)
            all_probs.append(probs)
        return self.head(h), torch.stack(picks, 1), torch.stack(all_probs, 1)


def compression_loss(probs):
    """Expected distinct blocks touched per hop, relaxed. Same objective as compress.py."""
    m = probs.view(*probs.shape[:-1], N_BLOCK, BLOCK).sum(-1).clamp(1e-6, 1.0)
    return (1.0 - (1.0 - m).pow(TOP_K)).sum(-1).mean()


def fetch_stats(picks):
    """Blocks touched per hop, and how many are already resident from an earlier hop."""
    total, distinct = 0, 0
    for tok in picks.tolist():                     # [hops, top_k]
        seen = set()
        for hop in tok:
            blocks = {e // BLOCK for e in hop}
            total += len(blocks)
            distinct += len(blocks - seen)
            seen |= blocks
    n = picks.shape[0]
    return total / n, distinct / n


def run(hops, lam, steps, x, y, xv, yv):
    torch.manual_seed(0)
    m = LoopedMoE(hops)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(steps):
        i = torch.randint(0, len(x), (128,))
        logit, _, probs = m(x[i])
        loss = F.cross_entropy(logit, y[i]) + (lam * compression_loss(probs) if lam else 0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        logit, picks, _ = m(xv)
        acc = (logit.argmax(-1) == yv).float().mean().item()
    total, distinct = fetch_stats(picks)
    return {"hops": hops, "lambda": lam, "exact_accuracy": round(acc, 4),
            "block_fetches_per_problem": round(total, 3),
            "distinct_blocks_per_problem": round(distinct, 3),
            "repeat_fetches_saved_pct": round(100.0 * (1 - distinct / total), 1) if total else 0}


def main(steps=2500, lam=0.3, out="data/custom/arith.json"):
    steps, lam = int(steps), float(lam)
    x, y = problems()
    print(f"{len(x)} arithmetic problems, {N_CLASS} exact answers, "
          f"{N_EXPERT} experts in {N_BLOCK} blocks of {BLOCK}\n")
    # Train and test on all of them: the question is capacity, not generalisation. The rule is
    # tiny and the problem set is the whole domain, so memorising it *is* the failure mode we
    # want to see priced — a model that needs every expert has not compressed anything.
    print(f"{'hops':>5} {'compress':>9} {'exact acc':>10} {'fetches':>9} {'distinct':>9} "
          f"{'repeats':>8}")
    rows = []
    for hops in (1, 2, 4):
        for lm in (0.0, lam):
            r = run(hops, lm, steps, x, y, x, y)
            rows.append(r)
            print(f"{hops:>5} {('yes' if lm else 'no'):>9} {r['exact_accuracy']:>10.3f} "
                  f"{r['block_fetches_per_problem']:>9.2f} "
                  f"{r['distinct_blocks_per_problem']:>9.2f} "
                  f"{r['repeat_fetches_saved_pct']:>7.1f}%")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"n_problems": len(x), "n_class": N_CLASS,
                                     "n_expert": N_EXPERT, "block": BLOCK,
                                     "steps": steps, "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
