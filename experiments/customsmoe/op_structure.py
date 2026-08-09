#!/usr/bin/env python3
"""Do the experts split the way the *mathematics* splits?

The compression objective drove the arithmetic model down to 1.02 blocks per problem, and
accuracy stayed exact — but collapsing everything into one block is suspicious, not impressive.
Addition and subtraction are the same circuit with a sign flipped; multiplication is not. A
model that has learned arithmetic should share experts between + and - and separate x from
both. A model that has merely memorised 300 pairs has no reason to.

So this is a falsifiable prediction about structure, independent of accuracy:

    overlap(+, -)  >  overlap(+, x)  ~=  overlap(-, x)

If compression destroys that ordering while keeping accuracy at 1.000, the objective is pushing
too hard: it is buying fetches with structure the task actually needs. That is a real cost, and
the sweep below is how far the objective can be pushed before it appears.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from arith_loop import DIGITS, N_EXPERT, LoopedMoE, compression_loss, problems
import torch.nn.functional as F

OPNAME = ("+", "-", "x")


def train(lam, steps, x, y, hops=1):
    torch.manual_seed(0)
    m = LoopedMoE(hops)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(steps):
        i = torch.randint(0, len(x), (128,))
        logit, _, probs = m(x[i])
        loss = F.cross_entropy(logit, y[i]) + (lam * compression_loss(probs) if lam else 0.0)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        logit, picks, _ = m(x)
    return (logit.argmax(-1) == y).float().mean().item(), picks


def op_sets(x, picks):
    """Experts used by each operation, weighted by how often."""
    ops = x[:, 2 * DIGITS:2 * DIGITS + 3].argmax(-1)
    use = torch.zeros(3, N_EXPERT)
    for i, op in enumerate(ops.tolist()):
        for e in picks[i].flatten().tolist():
            use[op, e] += 1
    return use


def jaccard(use, a, b):
    """Weighted overlap: shared usage over total usage. 1.0 means identical expert profiles."""
    ua, ub = use[a] / use[a].sum(), use[b] / use[b].sum()
    return float(torch.minimum(ua, ub).sum())


def main(steps=2500, out="data/custom/op-structure.json"):
    steps = int(steps)
    x, y = problems()
    print(f"prediction: overlap(+,-) > overlap(+,x) ~= overlap(-,x)\n")
    print(f"{'lambda':>7} {'exact acc':>10} {'+ vs -':>8} {'+ vs x':>8} {'- vs x':>8} "
          f"{'experts':>8} {'holds?':>7}")
    rows = []
    for lam in (0.0, 0.1, 0.3, 1.0):
        acc, picks = train(lam, steps, x, y)
        use = op_sets(x, picks)
        pm, px, mx = jaccard(use, 0, 1), jaccard(use, 0, 2), jaccard(use, 1, 2)
        n_used = int((use.sum(0) > 0).sum())
        holds = pm > px and pm > mx
        rows.append({"lambda": lam, "exact_accuracy": round(acc, 4),
                     "overlap_plus_minus": round(pm, 3), "overlap_plus_times": round(px, 3),
                     "overlap_minus_times": round(mx, 3), "experts_used": n_used,
                     "prediction_holds": bool(holds)})
        print(f"{lam:>7.1f} {acc:>10.3f} {pm:>8.3f} {px:>8.3f} {mx:>8.3f} {n_used:>8d} "
              f"{('yes' if holds else 'NO'):>7}")
    Path(out).write_text(json.dumps({"steps": steps, "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
