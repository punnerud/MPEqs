#!/usr/bin/env python3
"""The network nudges a number and checks what happened.

`curriculum.py` measured the difficulty directly: held-out accuracy falls monotonically with
the number of carries — 0.647 at none, 0.437 at three. Carries are where addition stops being
digit-wise, and nothing in "predict the answer" teaches the model what a carry *is*. It only
ever sees finished answers.

So give it the other kind of evidence. Alongside each problem the model is also asked: if this
operand were nudged by ±1, ±10 or ±100, what would the answer be? The solver knows, so the
model can be trained on the difference between what it expected and what actually happened —
which is exactly the experiment a person runs when learning that 9 + 1 does something the other
digits do not. A perturbation of +1 crossing a 9 boundary *is* a carry, made visible as an
event rather than left implicit in a table of sums.

Two things are being tested, and they are separate:

1. does the perturbation objective improve accuracy on the base task, and specifically on the
   carry-heavy cases it is aimed at
2. does it cost anything — experts recruited, and therefore compression

And one control that decides whether the *content* of the perturbation matters or only the
extra gradient: a `shuffled` arm gets the same auxiliary head and the same number of updates,
but the perturbation label is drawn from a different problem. If shuffled does as well, the
model is only benefiting from more training signal, not from the experiment.

Staged like learning to drive: addition alone first, subtraction added afterwards.

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

sys.path.insert(0, str(Path(__file__).parent))
from curriculum import (D, LIM, MAXD, N_BUCKET, N_EXPERT, OUT_DIGITS, CurricMoE, assess,
                        batch, block_compression, bucket_of, carries, encode, holdout_set,
                        is_holdout, sample_bucket, solve)

DELTAS = (1, -1, 10, -10, 100, -100)


class ProbingMoE(CurricMoE):
    """The same network with one extra head: the answer after a nudge to the first operand."""

    def __init__(self, hops=3):
        super().__init__(hops)
        self.delta_embed = nn.Embedding(len(DELTAS), D)
        self.delta_sign = nn.Linear(D, 2)
        self.delta_digits = nn.ModuleList([nn.Linear(D, 10) for _ in range(OUT_DIGITS)])

    def probe_head(self, x, delta_idx):
        """Route the problem as usual, then read out the nudged answer from the same state."""
        h = torch.relu(self.inp(x))
        for _ in range(self.hops):
            probs = self.router(h).softmax(-1)
            w, idx = probs.topk(4, dim=-1)
            y = torch.einsum("bkij,bj->bki", self.experts[idx], h)
            h = self.norm(h + (y * w.unsqueeze(-1)).sum(1))
        h = h + self.delta_embed(delta_idx)
        return self.delta_sign(h), [d(h) for d in self.delta_digits]


def perturbed(items, rng, shuffle=False):
    """For each problem, a nudge and the answer it produces. `shuffle` breaks the pairing."""
    idx, sg, dg = [], [], []
    for a, b, op in items:
        di = rng.randrange(len(DELTAS))
        src = items[rng.randrange(len(items))] if shuffle else (a, b, op)
        v = solve(max(-LIM, min(LIM, src[0] + DELTAS[di])), src[1], src[2])
        idx.append(di)
        sg.append(1 if v < 0 else 0)
        m = abs(v)
        dg.append([(m // 10**k) % 10 for k in range(OUT_DIGITS)])
    return (torch.tensor(idx), torch.tensor(sg), torch.tensor(dg))


def buckets_for(ops):
    return [bk for bk in range(N_BUCKET) if bk // 36 in ops]


def run(arm, rounds, steps, dev, seed=0, batch_size=384, staged=True):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    m = ProbingMoE().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    t0 = time.time()

    for rd in range(rounds):
        # Staging is a variable, not a fixture. `fewshot.py` reproduced this experiment
        # without it and the perturbation gain reversed sign, so which of the two settings the
        # mechanism actually needs has to be measured rather than assumed.
        ops = ((0,) if rd < rounds // 2 else (0, 1)) if staged else (0, 1)
        pool = buckets_for(ops)
        for _ in range(steps):
            items = [s for s in (sample_bucket(pool[rng.randrange(len(pool))], rng)
                                 for _ in range(batch_size)) if s]
            if len(items) < 32:
                continue
            x, sg, dg = batch(items, dev)
            sl, dl, probs, _ = m(x)
            loss = F.cross_entropy(sl, sg)
            for j, d in enumerate(dl):
                loss = loss + F.cross_entropy(d, dg[:, j])
            loss = loss + 0.3 * block_compression(probs)
            if arm != "baseline":
                di, psg, pdg = perturbed(items, rng, shuffle=(arm == "shuffled"))
                di, psg, pdg = di.to(dev), psg.to(dev), pdg.to(dev)
                psl, pdl = m.probe_head(x, di)
                pl = F.cross_entropy(psl, psg)
                for j, d in enumerate(pdl):
                    pl = pl + F.cross_entropy(d, pdg[:, j])
                loss = loss + 0.5 * pl
            opt.zero_grad(); loss.backward(); opt.step()

    hold = holdout_set((0, 1), rng, n=12000)
    ok, _ = assess(m, hold, dev)
    by_carry = {}
    for (a, b, op), good in zip(hold, ok.tolist()):
        c = carries(a, b, op)
        if c >= 0:
            by_carry.setdefault(c, []).append(good)
    x, _, _ = batch(hold[:4096], dev)
    with torch.no_grad():
        used = len(set(m(x)[3].reshape(-1).tolist()))
    # The auxiliary head is not part of the base task's description, so it is excluded from
    # the parameter count the compression ratio charges for — otherwise the control arm would
    # be penalised for machinery it also carries.
    aux = (m.delta_embed.weight.numel() + sum(p.numel() for p in m.delta_sign.parameters())
           + sum(p.numel() for p in m.delta_digits.parameters()))
    resident = sum(p.numel() for p in m.parameters()) - aux - (N_EXPERT - used) * D * D
    return {"arm": arm, "holdout_accuracy": round(float(ok.float().mean()), 4),
            "per_carry": {str(k): round(sum(v) / len(v), 4) for k, v in sorted(by_carry.items())},
            "experts_used": used, "params_effective": resident,
            "seconds": round(time.time() - t0, 1)}


def main(rounds=8, steps=900, out="data/custom/experimenting.json", staged=1):
    rounds, steps = int(rounds), int(steps)
    global STAGED
    STAGED = bool(int(staged))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"three-digit signed + and -, "
          f"{'staged: addition first, then subtraction' if STAGED else 'unstaged: both from the start'}"
          f". {rounds} x {steps} steps, device {dev}\n")
    print(f"{'arm':>10} {'holdout':>8} {'0 carries':>10} {'1':>7} {'2':>7} {'3':>7} "
          f"{'experts':>8} {'s':>5}")
    rows = []
    for arm in ("baseline", "shuffled", "probing"):
        r = run(arm, rounds, steps, dev, staged=STAGED)
        r["staged"] = STAGED
        rows.append(r)
        pc = r["per_carry"]
        print(f"{arm:>10} {r['holdout_accuracy']:>8.3f} {pc.get('0', 0):>10.3f} "
              f"{pc.get('1', 0):>7.3f} {pc.get('2', 0):>7.3f} {pc.get('3', 0):>7.3f} "
              f"{r['experts_used']:>8} {r['seconds']:>5.0f}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"rounds": rounds, "steps_per_round": steps,
                                     "deltas": list(DELTAS), "rows": rows}, indent=2))
    print(f"\nwrote {out}")

    by = {r["arm"]: r for r in rows}
    d_all = by["probing"]["holdout_accuracy"] - by["baseline"]["holdout_accuracy"]
    d_ctl = by["probing"]["holdout_accuracy"] - by["shuffled"]["holdout_accuracy"]
    d_hard = (float(by["probing"]["per_carry"].get("3", 0))
              - float(by["baseline"]["per_carry"].get("3", 0)))
    print(f"probing vs baseline      {d_all:+.3f} overall, {d_hard:+.3f} at three carries")
    print(f"probing vs shuffled      {d_ctl:+.3f}  <- the part that is about the experiment")
    print("If those two are the same size, the gain is extra gradient, not the nudge.")


if __name__ == "__main__":
    main(*sys.argv[1:])
