#!/usr/bin/env python3
"""A small MoE with a planted expert structure, as ground truth for the analysis chain.

Today's bug had one root cause that no amount of care in the analysis could have caught: the
router trace was an artefact, and there was nothing to check it against. Every tool downstream
agreed with every other tool because they all read the same wrong file. `make check-numbers`
guarded 40 numbers and passed, because a consistency gate cannot detect a consistent error.

The fix for that is not more care. It is a case where the right answer is known in advance.

So: train a real MoE, small, where the co-activation structure is *planted*. Inputs come from G
latent groups; the router is nudged during training so that group g uses its own slice of
experts. Then the ground-truth layout is known — experts of a group belong adjacent on disk —
and the whole chain (trace -> co-activation graph -> layout solver) can be scored against it
rather than against itself.

If the chain recovers the planted groups, it works. If it does not, the failure is in the chain,
not in the data, because here we know what the data contains.

Writes a trace in moetrace's MOET v2 format so the Rust tools read it unmodified.

Run with a torch venv, e.g. /Users/punnerud/Downloads/ainmt/venv/bin/python3.
"""
import json
import struct
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

N_EXPERT, TOP_K, N_GROUP, D = 32, 4, 8, 64
EXPERTS_PER_GROUP = N_EXPERT // N_GROUP          # 4 experts own each latent group


class PlantedMoE(nn.Module):
    """One MoE layer. Experts are tiny; only the routing matters here."""

    def __init__(self, n_expert=N_EXPERT, top_k=TOP_K, d=D):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(d, n_expert, bias=False)
        self.experts = nn.Parameter(torch.randn(n_expert, d, d) * (1.0 / d**0.5))
        self.head = nn.Linear(d, N_GROUP)

    def forward(self, x):
        logits = self.router(x)
        probs = logits.softmax(-1)
        w, idx = probs.topk(self.top_k, dim=-1)
        # Gather the selected experts and apply them. Small enough to do densely.
        sel = self.experts[idx]                              # [B, k, d, d]
        y = torch.einsum("bkij,bj->bki", sel, x)
        out = (y * w.unsqueeze(-1)).sum(1)
        return self.head(out), idx, w, probs


def data(n, seed=0):
    """Inputs from N_GROUP well-separated latent groups; the label is the group."""
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(N_GROUP, D, generator=g) * 3.0
    y = torch.randint(0, N_GROUP, (n,), generator=g)
    x = centres[y] + torch.randn(n, D, generator=g)
    return x, y


def plant_loss(probs, y):
    """Push group g's tokens onto experts [g*E, (g+1)*E).

    This is the ground truth being written into the model. It is a training-time nudge, not a
    constraint at inference: the router is free to disagree, and how much it does is itself
    informative.
    """
    mask = torch.zeros(N_GROUP, N_EXPERT, device=probs.device)
    for g in range(N_GROUP):
        mask[g, g * EXPERTS_PER_GROUP:(g + 1) * EXPERTS_PER_GROUP] = 1.0
    return -(probs * mask[y]).sum(-1).clamp(min=1e-6).log().mean()


def write_moet(path, idx, weights, n_layer=1):
    """MOET v2: 32-byte header, then layer u16 | pad u16 | token u32 | ids u16*k | w f32*k."""
    n_tok, k = idx.shape
    hdr = struct.pack("<6IQ", 0x5445_4F4D, 2, n_layer, N_EXPERT, k, 0, n_tok * n_layer)
    body = bytearray()
    for layer in range(n_layer):
        for t in range(n_tok):
            body += struct.pack("<HHI", layer, 0, t)
            body += struct.pack(f"<{k}H", *[int(v) for v in idx[t]])
            body += struct.pack(f"<{k}f", *[float(v) for v in weights[t]])
    Path(path).write_bytes(hdr + bytes(body))
    return n_tok


def main(steps=1500, out="data/custom", lam=1.0):
    steps, lam = int(steps), float(lam)
    torch.manual_seed(0)
    Path(out).mkdir(parents=True, exist_ok=True)
    x, y = data(6000)
    xt, yt = x[:5000], y[:5000]
    xv, yv = x[5000:], y[5000:]

    m = PlantedMoE()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for s in range(steps):
        i = torch.randint(0, len(xt), (256,))
        logit, _, _, probs = m(xt[i])
        loss = F.cross_entropy(logit, yt[i]) + lam * plant_loss(probs, yt[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % 500 == 0:
            with torch.no_grad():
                acc = (m(xv)[0].argmax(-1) == yv).float().mean()
            print(f"  step {s + 1:5d}  loss {loss.item():.4f}  holdout acc {acc:.3f}")

    with torch.no_grad():
        _, idx, w, _ = m(xv)

    # How well did the plant take? An expert's group is the one it fires for most.
    owner = torch.zeros(N_EXPERT, N_GROUP)
    for t in range(len(xv)):
        for e in idx[t]:
            owner[e, yv[t]] += 1
    intended = torch.arange(N_EXPERT) // EXPERTS_PER_GROUP
    actual = owner.argmax(-1)
    agree = (actual == intended).float().mean().item()
    print(f"\nexperts whose busiest group is the planted one: {100 * agree:.1f}%")

    trace = f"{out}/trace-planted.bin"
    n = write_moet(trace, idx, w)
    truth = {"n_expert": N_EXPERT, "top_k": TOP_K, "n_group": N_GROUP,
             "experts_per_group": EXPERTS_PER_GROUP,
             "planted_group_of_expert": intended.tolist(),
             "observed_group_of_expert": actual.tolist(),
             "plant_agreement": round(agree, 4), "n_tokens": n}
    Path(f"{out}/ground-truth.json").write_text(json.dumps(truth, indent=2))
    print(f"wrote {trace} ({n} tokens) and {out}/ground-truth.json")


if __name__ == "__main__":
    main(*sys.argv[1:])
