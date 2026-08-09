#!/usr/bin/env python3
"""Equations: something different on each side of =, a fraction line, and X to solve for.

`exprs.py` asks whether the hops recover the order in which an expression is *evaluated*.
An equation asks something harder, because the order is no longer given by the surface form —
it is given by the inverse of it. To solve `a*X + b = c` you subtract before you divide; to
solve `(X + a)/b = c` you multiply before you subtract. The written order and the solution
order run opposite ways, so a model that merely reads left to right cannot get this right.

Four forms, chosen so that the solution order splits two against two while the operators
involved stay the same:

    F0   a*X + b = c        subtract, then divide      additive step first
    F1   X/a + b = c        subtract, then multiply    additive step first
    F2   (X + a)/b = c      multiply, then subtract    multiplicative step first
    F3   (X - a)*b = c      divide, then add           multiplicative step first

Same four operations across the set, same three constants, only the order differs. So "hop 1
knows which step comes first" cannot be satisfied by memorising which operators appear — it has
to encode the procedure.

X is generated first and the right-hand side computed from it, so every equation has an exact
integer solution and the answer is checked digit by digit.

Run with a torch venv, e.g. /Users/punnerud/Downloads/ainmt/venv/bin/python3.
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = 10
N_FORM = 4
STEPNAME = ("sub", "add", "mul", "div")
N_STEP = 4
# Solution steps per form, in the order they must be applied.
FORM_STEPS = {0: (0, 3), 1: (0, 2), 2: (2, 0), 3: (3, 1)}
FORMNAME = ("a*X+b=c", "X/a+b=c", "(X+a)/b=c", "(X-a)*b=c")

D, N_EXPERT, TOP_K, BLOCK = 96, 64, 4, 4
N_BLOCK = N_EXPERT // BLOCK
# form, a, b, c: c spans a wide range so it gets three digits and a sign.
IN_DIM = N_FORM + BASE + BASE + (3 * BASE + 2)
N_DIGIT = 2                                    # |X| <= 99


def encode(form, a, b, c):
    x = torch.zeros(IN_DIM)
    x[form] = 1.0
    x[N_FORM + a] = 1.0
    x[N_FORM + BASE + b] = 1.0
    off = N_FORM + 2 * BASE
    m = abs(c)
    for k in range(3):
        x[off + k * BASE + (m // 10**k) % BASE] = 1.0
    x[off + 3 * BASE + (1 if c < 0 else 0)] = 1.0
    return x


def dataset():
    """Every (form, a, b, X) with an exact integer solution."""
    xs, ys, steps, forms = [], [], [], []
    for form in range(N_FORM):
        for a in range(1, BASE):
            for b in range(BASE):
                for xv in range(-9, 10):
                    if form == 0:
                        c = a * xv + b
                        X = xv
                    elif form == 1:
                        X = a * xv                      # so X/a is exact
                        c = xv + b
                    elif form == 2:
                        if b == 0:
                            continue
                        X = b * xv - a                  # (X + a)/b = xv
                        c = xv
                    else:
                        if b == 0:
                            continue
                        X = xv + a                      # (X - a)*b = c
                        c = xv * b
                    if abs(X) > 99 or abs(c) > 999:
                        continue
                    xs.append(encode(form, a, b, c))
                    m = abs(X)
                    ys.append([1 if X < 0 else 0, (m // 10) % 10, m % 10])
                    steps.append(list(FORM_STEPS[form]))
                    forms.append(form)
    return (torch.stack(xs), torch.tensor(ys), torch.tensor(steps), torch.tensor(forms))


class EqMoE(nn.Module):
    def __init__(self, hops=2):
        super().__init__()
        self.hops = hops
        self.inp = nn.Linear(IN_DIM, D)
        self.router = nn.Linear(D, N_EXPERT, bias=False)
        self.experts = nn.Parameter(torch.randn(N_EXPERT, D, D) * (1.0 / D**0.5))
        self.norm = nn.LayerNorm(D)
        self.sign = nn.Linear(D, 2)
        self.digits = nn.ModuleList([nn.Linear(D, BASE) for _ in range(N_DIGIT)])

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


def step_separation(probs, steps):
    """Separate hop h's routing by which solution step belongs at position h."""
    out = []
    for h in range(min(probs.shape[1], steps.shape[1])):
        prof = [probs[:, h][steps[:, h] == s].mean(0)
                for s in range(N_STEP) if (steps[:, h] == s).any()]
        if len(prof) > 1:
            out += [0.5 * (prof[i] - prof[j]).abs().sum()
                    for i in range(len(prof)) for j in range(i + 1, len(prof))]
    return torch.stack(out).mean() if out else probs.sum() * 0.0


def probe(feats, labels, n_class, dev, steps=500):
    """Linear probe accuracy on a held-out fifth, after shuffling.

    The shuffle is not cosmetic. Both datasets are generated form by form, so an unshuffled
    tail is a single class: the first run reported accuracy 0.000 against a majority baseline
    of 1.000, which is a split bug wearing the costume of a result. Anything with a majority
    baseline at 1.000 is measuring nothing.
    """
    g = torch.Generator(device="cpu").manual_seed(3)
    perm = torch.randperm(len(feats), generator=g).to(feats.device)
    feats, labels = feats[perm], labels[perm]
    cut = int(len(feats) * 0.8)
    lin = nn.Linear(feats.shape[1], n_class).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=3e-3)
    for _ in range(steps):
        i = torch.randint(0, cut, (512,), device=dev)
        loss = F.cross_entropy(lin(feats[i]), labels[i])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = float((lin(feats[cut:]).argmax(-1) == labels[cut:]).float().mean())
        maj = float(torch.bincount(labels[cut:], minlength=n_class).max() / len(labels[cut:]))
    return acc, maj


def run(lam, beta, steps_n, hops, dev, x, y, st, forms):
    torch.manual_seed(0)
    m = EqMoE(hops).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    t0 = time.time()
    for _ in range(steps_n):
        i = torch.randint(0, len(x), (512,), device=dev)
        sl, dl, _, probs = m(x[i])
        loss = F.cross_entropy(sl, y[i, 0])
        for j, d in enumerate(dl):
            loss = loss + F.cross_entropy(d, y[i, j + 1])
        if lam:
            loss = loss + lam * compression_loss(probs)
        if beta:
            loss = loss - beta * step_separation(probs, st[i])
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        sl, dl, picks, probs = m(x)
        ok = sl.argmax(-1) == y[:, 0]
        for j, d in enumerate(dl):
            ok &= d.argmax(-1) == y[:, j + 1]
        acc = float(ok.float().mean())
        per_form = {FORMNAME[f]: round(float(ok[forms == f].float().mean()), 3)
                    for f in range(N_FORM)}

    h0 = probs[:, 0]
    a_first, maj_first = probe(h0, st[:, 0], N_STEP, dev)
    a_second, maj_second = probe(h0, st[:, 1], N_STEP, dev)
    # Does hop 1 know only *which* form it is, or the order? Forms determine steps here, so
    # the sharper question is whether hop 1 separates additive-first from multiplicative-first
    # across forms that share operators.
    order_class = (st[:, 0] >= 2).long()          # 0 = additive step first, 1 = multiplicative
    a_order, maj_order = probe(h0, order_class, 2, dev)

    return {"lambda": lam, "beta": beta, "hops": hops, "exact_accuracy": round(acc, 4),
            "per_form_accuracy": per_form,
            "hop1_predicts_step1": round(a_first, 4), "step1_majority": round(maj_first, 4),
            "hop1_predicts_step2": round(a_second, 4), "step2_majority": round(maj_second, 4),
            "hop1_predicts_order_class": round(a_order, 4),
            "order_majority": round(maj_order, 4),
            "experts_used": int(len(torch.unique(picks))),
            "seconds": round(time.time() - t0, 1)}


def main(steps=4000, out="data/custom/equations.json", hops=2):
    steps, hops = int(steps), int(hops)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    x, y, st, forms = dataset()
    x, y, st = x.to(dev), y.to(dev), st.to(dev)
    print(f"{len(x)} equations over {N_FORM} forms: {', '.join(FORMNAME)}")
    print(f"{N_EXPERT} experts, top-{TOP_K}, {hops} hops, device {dev}\n")
    print(f"{'lam':>5} {'beta':>5} {'solve acc':>10} {'hop1->step1':>12} {'(maj)':>7} "
          f"{'hop1->step2':>12} {'order cls':>10} {'(maj)':>7} {'experts':>8}")
    rows = []
    for lam, beta in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)):
        r = run(lam, beta, steps, hops, dev, x, y, st, forms)
        rows.append(r)
        print(f"{lam:>5.1f} {beta:>5.1f} {r['exact_accuracy']:>10.3f} "
              f"{r['hop1_predicts_step1']:>12.3f} {r['step1_majority']:>7.3f} "
              f"{r['hop1_predicts_step2']:>12.3f} {r['hop1_predicts_order_class']:>10.3f} "
              f"{r['order_majority']:>7.3f} {r['experts_used']:>8d}")
    print()
    for r in rows:
        print(f"  lam={r['lambda']} beta={r['beta']}  per form: {r['per_form_accuracy']}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"n_eq": len(x), "n_expert": N_EXPERT, "hops": hops,
                                     "steps": steps, "forms": list(FORMNAME),
                                     "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
