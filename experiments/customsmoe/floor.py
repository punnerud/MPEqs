#!/usr/bin/env python3
"""Compression with a floor: pay for fetches, but not with distinctions.

`op_structure.py` showed the failure the pure compression objective hides. At lambda >= 0.3 the
arithmetic model uses 8 of 32 experts, subtraction and multiplication end up with identical
expert profiles (overlap 0.990), and accuracy stays at 1.000 the whole way — so nothing in the
accounting objects. On a task small enough to survive that collapse it looks like a triumph.
On one that is not, it would surface as capability loss long after the fetch numbers looked
excellent.

The missing term prices the *loss of distinctions*. Compression says "touch few blocks";
the floor says "and do not let two things that behave differently route identically":

    separation = mean pairwise total-variation distance between the mean routing
                 distributions of the operations

Total variation because it is bounded in [0, 1], so the two terms stay commensurable and the
trade can be read off directly rather than tuned into.

The question is whether the two can be had at once, or whether fetches genuinely cost
distinctions. Either answer is worth having; only one of them is good news.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from arith_loop import DIGITS, N_EXPERT, LoopedMoE, compression_loss, problems
from op_structure import jaccard, op_sets


def separation(probs, ops):
    """Mean pairwise total-variation distance between per-operation routing profiles."""
    prof = []
    for o in range(3):
        m = ops == o
        prof.append(probs[m].mean(0) if m.any() else probs.mean(0))
    d = [0.5 * (prof[a] - prof[b]).abs().sum()
         for a in range(3) for b in range(a + 1, 3)]
    return torch.stack(d).mean()


def run(lam, beta, steps, x, y, ops):
    torch.manual_seed(0)
    m = LoopedMoE(1)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(steps):
        i = torch.randint(0, len(x), (128,))
        logit, _, probs = m(x[i])
        p = probs[:, 0]                                   # one hop
        loss = F.cross_entropy(logit, y[i])
        if lam:
            loss = loss + lam * compression_loss(probs)
        if beta:
            loss = loss - beta * separation(p, ops[i])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        logit, picks, probs = m(x)
        acc = (logit.argmax(-1) == y).float().mean().item()
        sep = float(separation(probs[:, 0], ops))
    use = op_sets(x, picks)
    blocks = float(torch.tensor(
        [len({int(e) // 4 for e in row.flatten()}) for row in picks], dtype=torch.float).mean())
    pairs = [jaccard(use, 0, 1), jaccard(use, 0, 2), jaccard(use, 1, 2)]
    return {"lambda": lam, "beta": beta, "exact_accuracy": round(acc, 4),
            "blocks_per_problem": round(blocks, 3),
            "experts_used": int((use.sum(0) > 0).sum()),
            "separation": round(sep, 3),
            "max_pair_overlap": round(max(pairs), 3)}


def main(steps=2500, out="data/custom/floor.json"):
    steps = int(steps)
    x, y = problems()
    ops = x[:, 2 * DIGITS:2 * DIGITS + 3].argmax(-1)
    print(f"{'lambda':>7} {'beta':>6} {'acc':>6} {'blocks':>7} {'experts':>8} "
          f"{'separation':>11} {'worst overlap':>14}")
    rows = []
    for lam, beta in ((0.0, 0.0), (1.0, 0.0), (1.0, 0.1), (1.0, 0.3), (1.0, 1.0)):
        r = run(lam, beta, steps, x, y, ops)
        rows.append(r)
        print(f"{lam:>7.1f} {beta:>6.1f} {r['exact_accuracy']:>6.3f} "
              f"{r['blocks_per_problem']:>7.2f} {r['experts_used']:>8d} "
              f"{r['separation']:>11.3f} {r['max_pair_overlap']:>14.3f}")
    Path(out).write_text(json.dumps({"steps": steps, "rows": rows}, indent=2))
    print(f"\nwrote {out}")
    base = rows[1]
    best = max(rows[2:], key=lambda r: r["experts_used"])
    print(f"\ncompression alone: {base['blocks_per_problem']:.2f} blocks, "
          f"{base['experts_used']} experts, worst overlap {base['max_pair_overlap']:.3f}")
    print(f"with the floor:    {best['blocks_per_problem']:.2f} blocks, "
          f"{best['experts_used']} experts, worst overlap {best['max_pair_overlap']:.3f} "
          f"(beta={best['beta']})")


if __name__ == "__main__":
    main(*sys.argv[1:])
