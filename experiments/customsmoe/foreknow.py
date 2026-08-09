#!/usr/bin/env python3
"""Two questions the disk cares about, asked of the trained arithmetic model.

**Is the same expert reused across hops, or does each hop recruit new ones?** A loop that keeps
returning to the same experts is cheap: the block is already resident, and a longer expression
costs hops rather than capacity. A loop that recruits fresh experts every hop is just a deeper
network wearing a loop's clothing, and it costs a fetch each time.

**Can the needed experts be known before the model runs?** If a linear probe on the input can
name the blocks a problem will touch, they can be fetched while the first hop is still
computing, and the disk latency disappears behind the arithmetic. This is prefetch, and it was
declared dead earlier in the project on the strength of a trace that turned out to be an
artefact — so it is being asked again, on a model whose routing is real by construction.

Reported against the only baseline that means anything: guessing the globally most popular
blocks, which needs no model at all.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from bigmath import (BLOCK, N_BLOCK, N_EXPERT, TOP_K, BigMoE, compression_loss,
                     problems, separation)


def train(n_op, lam, beta, steps, hops, dev):
    torch.manual_seed(0)
    x, ops, sg, dg = problems(n_op)
    x, ops, sg, dg = x.to(dev), ops.to(dev), sg.to(dev), dg.to(dev)
    m = BigMoE(hops).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for _ in range(steps):
        i = torch.randint(0, len(x), (512,), device=dev)
        sl, dl, _, probs = m(x[i])
        loss = F.cross_entropy(sl, sg[i])
        for j, d in enumerate(dl):
            loss = loss + F.cross_entropy(d, dg[i, j])
        if lam:
            loss = loss + lam * compression_loss(probs)
        if beta:
            loss = loss - beta * separation(probs[:, 0], ops[i], n_op)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        picks = torch.cat([m(x[i:i + 8192])[2] for i in range(0, len(x), 8192)])
    return m, x, picks


def reuse(picks):
    """Share of a hop's expert picks that the previous hop already had."""
    if picks.shape[1] < 2:
        return 0.0
    rep = 0.0
    for h in range(1, picks.shape[1]):
        a = picks[:, h - 1].unsqueeze(-1)
        b = picks[:, h].unsqueeze(-2)
        rep += float((a == b).any(-2).float().mean())
    return rep / (picks.shape[1] - 1)


def foreknowledge(x, picks, dev, steps=600):
    """Linear probe: input -> which blocks this problem will touch. Recall at the true count."""
    n = len(x)
    blocks = torch.zeros(n, N_BLOCK, device=dev)
    flat = (picks.reshape(n, -1) // BLOCK).to(dev)
    blocks.scatter_(1, flat, 1.0)
    need = int(blocks.sum(1).mean().round().clamp(min=1).item())

    cut = int(n * 0.8)
    probe = torch.nn.Linear(x.shape[1], N_BLOCK).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=3e-3)
    for _ in range(steps):
        i = torch.randint(0, cut, (512,), device=dev)
        loss = F.binary_cross_entropy_with_logits(probe(x[i]), blocks[i])
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        pred = probe(x[cut:]).topk(need, dim=-1).indices
        hit = blocks[cut:].gather(1, pred).sum()
        recall = float(hit / blocks[cut:].sum().clamp(min=1))
        # Baseline: always guess the globally most popular blocks.
        pop = blocks[:cut].sum(0).topk(need).indices
        base = float(blocks[cut:][:, pop].sum() / blocks[cut:].sum().clamp(min=1))
    return need, recall, base


def main(steps=2500, out="data/custom/foreknow.json", hops=3):
    steps, hops = int(steps), int(hops)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"{hops} hops, {N_EXPERT} experts in {N_BLOCK} blocks, top-{TOP_K}, device {dev}\n")
    print(f"{'ops':>12} {'lam':>5} {'beta':>5} {'hop reuse':>10} {'blocks':>7} "
          f"{'probe recall':>13} {'popularity':>11}")
    rows = []
    for n_op in (2, 5):
        for lam, beta in ((0.0, 0.0), (1.0, 1.0)):
            m, x, picks = train(n_op, lam, beta, steps, hops, dev)
            ru = reuse(picks)
            need, rec, base = foreknowledge(x, picks, dev)
            rows.append({"n_op": n_op, "lambda": lam, "beta": beta, "hops": hops,
                         "hop_reuse": round(ru, 4), "blocks_needed": need,
                         "probe_recall": round(rec, 4), "popularity_recall": round(base, 4)})
            print(f"{n_op:>12} {lam:>5.1f} {beta:>5.1f} {ru:>10.3f} {need:>7d} "
                  f"{rec:>13.3f} {base:>11.3f}")
    Path(out).write_text(json.dumps({"hops": hops, "steps": steps, "rows": rows}, indent=2))
    print(f"\nwrote {out}")
    print("hop reuse 1.0 means every hop returns to experts the previous hop already had —\n"
          "a loop over resident blocks. Probe recall above the popularity baseline means the\n"
          "needed blocks are knowable from the input, which is what makes prefetch possible.")


if __name__ == "__main__":
    main(*sys.argv[1:])
