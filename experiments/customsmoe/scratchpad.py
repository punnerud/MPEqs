#!/usr/bin/env python3
"""Multiplication, with somewhere to put the partial products.

Nothing built today learned multiplication. Every other operation reached usable accuracy and
`x` sat at 0.005, which is chance. The suspicion is structural rather than about capacity: state
passes between hops only through the residual, a fixed-width vector shared with everything the
network is also carrying. 432 x 7 is 400x7 + 30x7 + 2x7 — three partial products that have to be
held while the others are computed — and there is nowhere to hold them.

So give the experts a place to write. A small slot memory, read at the start of each hop and
written at the end, addressed by content, carried across hops. Not weights: per-example state
that exists for the duration of one problem and is thrown away.

And the second half, which the carry result argues for directly. Carries stopped being hard once
a carry was presented as an *event* — a nudge with a visible consequence — rather than left
implicit in finished answers. A partial product is the same kind of hidden intermediate, so it
gets the same treatment: an auxiliary head asks hop k for the k-th partial product, digit k of
`a` times `b`, which is a number the solver knows and the model otherwise never sees.

Four arms, so the two mechanisms can be told apart:

    baseline           neither                     the 0.005 starting point
    scratch            slot memory only            is a place to write enough?
    partials           partial-product head only   is making the intermediate visible enough?
    scratch+partials   both                        do they need each other?

Task: three-digit by one-digit signed multiplication, which is the worked example — 432 x 7 —
and small enough that failure means something other than "too hard".

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

# Operand widths are set from the command line, because the first run at 3x1 was solved
# exactly (1.000) by every arm including the baseline — which corrected an earlier claim of
# mine that multiplication is never learned. It is; it was 2x2 and 3x3 that failed. A task all
# arms solve cannot tell them apart, so the width has to be pushed to where it breaks.
A_DIG, B_DIG = 3, 1
A_LIM, B_LIM = 10**A_DIG - 1, 10**B_DIG - 1
OUT_DIGITS = A_DIG + B_DIG
IN_DIM = (A_DIG * 10 + 1) + (B_DIG * 10 + 1)
D, N_EXPERT, TOP_K, BLOCK, HOPS = 64, 32, 4, 4, 3


def set_widths(a_dig, b_dig, hops=None):
    """Rebuild the width-dependent globals. One partial-product row per digit of a."""
    global A_DIG, B_DIG, A_LIM, B_LIM, OUT_DIGITS, IN_DIM, HOPS
    A_DIG, B_DIG = int(a_dig), int(b_dig)
    A_LIM, B_LIM = 10**A_DIG - 1, 10**B_DIG - 1
    OUT_DIGITS = A_DIG + B_DIG
    IN_DIM = (A_DIG * 10 + 1) + (B_DIG * 10 + 1)
    HOPS = int(hops) if hops else max(3, A_DIG)
N_SLOT, SLOT_W = 8, 32
HOLDOUT_MOD = 97
PARTIAL_DIGITS = 6                               # wide enough for a running sum
RUNNING = False                                  # supervise the accumulation, not the rows


def is_holdout(a, b):
    return ((a + A_LIM) * 7919 + (b + B_LIM) * 104729) % HOLDOUT_MOD == 0


def encode(a, b, out):
    for k in range(A_DIG):
        out[k * 10 + (abs(a) // 10**k) % 10] = 1.0
    out[A_DIG * 10] = 1.0 if a < 0 else 0.0
    off = A_DIG * 10 + 1
    for k in range(B_DIG):
        out[off + k * 10 + (abs(b) // 10**k) % 10] = 1.0
    out[off + B_DIG * 10] = 1.0 if b < 0 else 0.0


def digits_of(v, n):
    m = abs(v)
    return [(m // 10**k) % 10 for k in range(n)]


def batch(items, dev):
    x = torch.zeros(len(items), IN_DIM)
    sg = torch.zeros(len(items), dtype=torch.long)
    dg = torch.zeros(len(items), OUT_DIGITS, dtype=torch.long)
    pp = torch.zeros(len(items), HOPS, PARTIAL_DIGITS, dtype=torch.long)
    for i, (a, b) in enumerate(items):
        encode(a, b, x[i])
        v = a * b
        sg[i] = 1 if v < 0 else 0
        dg[i] = torch.tensor(digits_of(v, OUT_DIGITS))
        for k in range(HOPS):
            if RUNNING:
                # The running sum after k rows, shifted as written: sum_{j<=k} d_j(a)*|b|*10^j.
                #
                # This is the change the 3x2 digit breakdown argued for. Supervising the k-th
                # partial product alone gave +0.007, because the partial products were never
                # the hard part — 10^0 was already at 1.000. Everything collapsed at 10^1 and
                # 10^2, which is where shifted rows are *summed* and carries propagate. So
                # supervise the accumulation instead of the ingredients, which is also what
                # made the perturbation objective work: a before-and-after pair rather than
                # one more target.
                acc = sum(((abs(a) // 10**j) % 10) * abs(b) * 10**j for j in range(k + 1))
                pp[i, k] = torch.tensor(digits_of(acc, PARTIAL_DIGITS))
            else:
                # The k-th partial product, as a human would write it: digit k of a, times |b|.
                pp[i, k] = torch.tensor(
                    digits_of(((abs(a) // 10**k) % 10) * abs(b), PARTIAL_DIGITS))
    return x.to(dev), sg.to(dev), dg.to(dev), pp.to(dev)


def sample(rng, n, holdout):
    out = []
    while len(out) < n:
        a = rng.randint(-A_LIM, A_LIM)
        b = rng.randint(-B_LIM, B_LIM)
        if is_holdout(a, b) != holdout:
            continue
        out.append((a, b))
    return out


class ScratchMoE(nn.Module):
    """MoE hops with an optional per-example slot memory read before and written after."""

    def __init__(self, scratch=True, partials=True):
        super().__init__()
        self.scratch, self.partials = scratch, partials
        self.inp = nn.Linear(IN_DIM, D)
        self.router = nn.Linear(D, N_EXPERT, bias=False)
        self.experts = nn.Parameter(torch.randn(N_EXPERT, D, D) * (1.0 / D**0.5))
        self.norm = nn.LayerNorm(D)
        if scratch:
            self.slot_key = nn.Parameter(torch.randn(N_SLOT, D) * (1.0 / D**0.5))
            self.read_in = nn.Linear(SLOT_W, D)
            self.write_val = nn.Linear(D, SLOT_W)
            self.write_gate = nn.Linear(D, 1)
        self.sign = nn.Linear(D, 2)
        self.digits = nn.ModuleList([nn.Linear(D, 10) for _ in range(OUT_DIGITS)])
        if partials:
            self.pp_heads = nn.ModuleList([
                nn.ModuleList([nn.Linear(D, 10) for _ in range(PARTIAL_DIGITS)])
                for _ in range(HOPS)])

    def forward(self, x):
        h = torch.relu(self.inp(x))
        mem = (x.new_zeros(x.shape[0], N_SLOT, SLOT_W) if self.scratch else None)
        probs_all, picks, per_hop = [], [], []
        for hop in range(HOPS):
            if self.scratch:
                addr = (h @ self.slot_key.t()).softmax(-1)          # [B, N_SLOT]
                h = h + self.read_in(torch.einsum("bs,bsw->bw", addr, mem))
            probs = self.router(h).softmax(-1)
            w, idx = probs.topk(TOP_K, dim=-1)
            y = torch.einsum("bkij,bj->bki", self.experts[idx], h)
            h = self.norm(h + (y * w.unsqueeze(-1)).sum(1))
            if self.scratch:
                addr = (h @ self.slot_key.t()).softmax(-1)
                g = torch.sigmoid(self.write_gate(h))               # [B, 1]
                val = self.write_val(h)                             # [B, SLOT_W]
                upd = torch.einsum("bs,bw->bsw", addr, val)
                keep = 1.0 - (g.unsqueeze(-1) * addr.unsqueeze(-1))
                mem = mem * keep + g.unsqueeze(-1) * upd
            probs_all.append(probs)
            picks.append(idx)
            per_hop.append(h)
        pp = None
        if self.partials:
            pp = [[head(per_hop[k]) for head in self.pp_heads[k]] for k in range(HOPS)]
        return (self.sign(h), [d(h) for d in self.digits], pp,
                torch.stack(probs_all, 1), torch.stack(picks, 1))


def block_compression(probs):
    nb = N_EXPERT // BLOCK
    m = probs.view(*probs.shape[:-1], nb, BLOCK).sum(-1).clamp(1e-6, 1.0)
    return (1.0 - (1.0 - m).pow(TOP_K)).sum(-1).mean()


def evaluate(m, items, dev, chunk=4096):
    x, sg, dg, _ = batch(items, dev)
    ok = torch.zeros(len(items), dtype=torch.bool, device=dev)
    digit_ok = torch.zeros(OUT_DIGITS, device=dev)
    used = set()
    with torch.no_grad():
        for i in range(0, len(items), chunk):
            sl, dl, _, _, picks = m(x[i:i + chunk])
            good = sl.argmax(-1) == sg[i:i + chunk]
            for j, d in enumerate(dl):
                hit = d.argmax(-1) == dg[i:i + chunk, j]
                digit_ok[j] += hit.sum()
                good &= hit
            ok[i:i + chunk] = good
            used |= set(picks.reshape(-1).tolist())
    return (float(ok.float().mean()),
            [round(float(v) / len(items), 4) for v in digit_ok], len(used))


def run(arm, steps, dev, seed=0, batch_size=384):
    global RUNNING
    RUNNING = "running" in arm
    rng = random.Random(seed)
    torch.manual_seed(seed)
    m = ScratchMoE(scratch="scratch" in arm,
                   partials=("partials" in arm or "running" in arm)).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for _ in range(steps):
        items = sample(rng, batch_size, holdout=False)
        x, sg, dg, pp = batch(items, dev)
        sl, dl, ppred, probs, _ = m(x)
        loss = F.cross_entropy(sl, sg)
        for j, d in enumerate(dl):
            loss = loss + F.cross_entropy(d, dg[:, j])
        if ppred is not None:
            aux = 0.0
            for k in range(HOPS):
                for j, head in enumerate(ppred[k]):
                    aux = aux + F.cross_entropy(head, pp[:, k, j])
            loss = loss + 0.5 * aux / HOPS
        loss = loss + 0.3 * block_compression(probs)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    hold = sample(rng, 8000, holdout=True)
    acc, per_digit, used = evaluate(m, hold, dev)
    # The partial-product heads are scaffolding for training, not part of answering, so they
    # are excluded from the parameter count the same way the perturbation head was.
    aux_params = (sum(p.numel() for mm in getattr(m, "pp_heads", []) for h in mm
                      for p in h.parameters()) if m.partials else 0)
    resident = sum(p.numel() for p in m.parameters()) - aux_params
    return {"arm": arm, "holdout_accuracy": round(acc, 4), "per_digit_accuracy": per_digit,
            "experts_used": used, "params_resident": resident,
            "seconds": round(time.time() - t0, 1)}


def main(steps=6000, out="data/custom/scratchpad.json", a_dig=3, b_dig=1, arms=None):
    steps = int(steps)
    set_widths(a_dig, b_dig)
    global ARMS
    ARMS = arms.split(",") if arms else ["baseline", "scratch", "partials",
                                         "scratch+partials"]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"{A_DIG}-digit by {B_DIG}-digit signed multiplication, {HOPS} hops, "
          f"{N_SLOT} slots x {SLOT_W}, {steps} steps, device {dev}")
    print("held-out problems are 1 in 97 and never trained on\n")
    heads = " ".join(f"{'10^' + str(k):>8}" for k in range(OUT_DIGITS))
    print(f"{'arm':>18} {'holdout':>8} {heads} {'experts':>8} {'s':>5}")
    rows = []
    for arm in ARMS:
        r = run(arm, steps, dev)
        rows.append(r)
        cells = " ".join(f"{v:>8.3f}" for v in r["per_digit_accuracy"])
        print(f"{arm:>18} {r['holdout_accuracy']:>8.3f} {cells} "
              f"{r['experts_used']:>8} {r['seconds']:>5.0f}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"a_digits": A_DIG, "b_digits": B_DIG, "hops": HOPS,
                                     "slots": N_SLOT, "slot_width": SLOT_W, "steps": steps,
                                     "rows": rows}, indent=2))
    print(f"\nwrote {out}")
    by = {r["arm"]: r["holdout_accuracy"] for r in rows}
    base = by.get("baseline")
    if base is not None:
        for arm in ARMS:
            if arm != "baseline":
                print(f"{arm:>22} {by[arm] - base:+.3f} against baseline")


if __name__ == "__main__":
    main(*sys.argv[1:])
