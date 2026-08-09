#!/usr/bin/env python3
"""Update the router's weights by compression alone — no expert is ever told it was wrong.

`planted.py` writes the answer in: group g is *pushed* onto experts [4g, 4g+4), and the check
is whether the analysis chain recovers what was planted. Useful for validating the instrument,
but circular as a claim about learning — the structure was supplied.

This removes the supply. The auxiliary loss says nothing about which expert should serve which
input. It says only: **whatever you route to, route so that it compresses.** Experts are laid
out in fixed blocks the size of one disk fetch, and the penalty is the expected number of
distinct blocks a token touches. Fewer blocks is a shorter description of the access pattern
and fewer reads from disk, which here are the same quantity.

That is the thesis being tested: compression is the definition of having learned, so it should
be enough on its own to produce structure. If it is, the emergent groups are discovered rather
than planted, and the same objective can be applied to a model whose correct grouping nobody
knows — which is the case for every real MoE.

Measured against `plain` (no auxiliary loss) and `balanced` (the load-balancing loss real MoEs
train with), all three sharing a seed, a step count and an architecture.

Run with a torch venv, e.g. /Users/punnerud/Downloads/ainmt/venv/bin/python3.
"""
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from planted import N_EXPERT, N_GROUP, TOP_K, PlantedMoE, data, write_moet  # noqa: E402

BLOCK = 4                      # experts per disk fetch; the unit compression is measured in
N_BLOCK = N_EXPERT // BLOCK


def block_mass(probs):
    """Router mass per block, [B, n_block]."""
    return probs.view(probs.shape[0], N_BLOCK, BLOCK).sum(-1)


def compression_loss(probs):
    """Expected distinct blocks touched, relaxed so it has a gradient.

    A block is "touched" if any of its experts is selected. The hard count is a step function;
    `1 - (1 - m)^k` is the soft version, rising from 0 at no mass to 1 at full mass, and it is
    exactly the probability of touching the block if the top-k picks were independent draws.
    Summing over blocks gives expected fetches per token, which is the cost model's currency.
    """
    m = block_mass(probs).clamp(1e-6, 1.0)
    return (1.0 - (1.0 - m).pow(TOP_K)).sum(-1).mean()


def balance_loss(probs):
    """The standard load-balancing auxiliary: n_expert * sum_e f_e * P_e, minimised when flat."""
    return N_EXPERT * (probs.mean(0) * probs.mean(0)).sum()


def blocks_touched(idx):
    """The hard quantity the relaxation stands in for: distinct blocks per token."""
    return float(torch.tensor(
        [len({int(e) // BLOCK for e in row}) for row in idx], dtype=torch.float).mean())


def lift_stats(idx):
    """Max co-activation lift and the share of pairs above 2x, as `coact analyze` reports."""
    single, pair = Counter(), Counter()
    n = idx.shape[0]
    for row in idx.tolist():
        s = set(row)
        single.update(s)
        pair.update(itertools.combinations(sorted(s), 2))
    lifts = []
    for a, b in itertools.combinations(range(N_EXPERT), 2):
        if single[a] and single[b]:
            lifts.append((pair[(a, b)] / n) / ((single[a] / n) * (single[b] / n)))
    return (max(lifts) if lifts else 0.0,
            100.0 * sum(1 for v in lifts if v >= 2.0) / max(len(lifts), 1))


def run(name, aux, lam, steps, xt, yt, xv, yv):
    torch.manual_seed(0)                       # identical init across regimes
    m = PlantedMoE()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for s in range(steps):
        i = torch.randint(0, len(xt), (256,))
        logit, _, _, probs = m(xt[i])
        loss = F.cross_entropy(logit, yt[i]) + (lam * aux(probs) if aux else 0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        logit, idx, w, probs = m(xv)
        acc = (logit.argmax(-1) == yv).float().mean().item()
    bt = blocks_touched(idx)
    mx, ge2 = lift_stats(idx)
    used = len({int(e) for row in idx.tolist() for e in row})
    return {"regime": name, "accuracy": round(acc, 4), "blocks_per_token": round(bt, 3),
            "max_fetches": N_BLOCK, "experts_used": used,
            "max_lift": round(mx, 2), "pairs_ge_2x_pct": round(ge2, 2)}, idx, w


def main(steps=1500, lam=0.3, out="data/custom"):
    steps, lam = int(steps), float(lam)
    x, y = data(6000)
    xt, yt, xv, yv = x[:5000], y[:5000], x[5000:], y[5000:]
    Path(out).mkdir(parents=True, exist_ok=True)

    rows = []
    for name, aux in (("plain", None),
                      ("balanced", balance_loss),
                      ("compressed", compression_loss)):
        r, idx, w = run(name, aux, lam, steps, xt, yt, xv, yv)
        rows.append(r)
        write_moet(f"{out}/trace-{name}.bin", idx, w)
        print(f"{r['regime']:>11}  acc {r['accuracy']:.3f}  "
              f"blocks/token {r['blocks_per_token']:5.2f} of {N_BLOCK}  "
              f"experts used {r['experts_used']:3d}/{N_EXPERT}  "
              f"max lift {r['max_lift']:5.2f}  pairs>=2x {r['pairs_ge_2x_pct']:5.2f}%")

    base = next(r for r in rows if r["regime"] == "plain")["blocks_per_token"]
    for r in rows:
        r["fetch_reduction_vs_plain_pct"] = round(
            100.0 * (1.0 - r["blocks_per_token"] / base), 2)
    Path(f"{out}/compress.json").write_text(json.dumps(
        {"block": BLOCK, "n_block": N_BLOCK, "steps": steps, "lambda": lam, "rows": rows},
        indent=2))

    print(f"\nfetches per token against `plain`:")
    for r in rows:
        print(f"  {r['regime']:>11}  {r['fetch_reduction_vs_plain_pct']:+6.2f} %")
    print(f"\nwrote {out}/compress.json and one MOET trace per regime")
    print("No regime was told which expert should serve which input. `compressed` was told "
          "only\nto make the access pattern cheap to describe.")


if __name__ == "__main__":
    main(*sys.argv[1:])
