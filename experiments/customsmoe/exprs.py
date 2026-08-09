#!/usr/bin/env python3
"""Multi-term expressions with parentheses, and whether the hops recover the evaluation order.

Every earlier arithmetic dataset here was one operation per problem, so there was no order to
discover and the loop had nothing to sequence. This one has two operations and a bracket, which
makes the *order* a property of the surface form rather than of the operands:

    a + b * c      multiplication first   (precedence)
    (a + b) * c    addition first         (bracket overrides it)

Same three operands, same two operators, different answer, and the only difference in the input
is one flag. A model that has learned arithmetic must route differently for those two, and if
the loop is doing what it looks like it is doing, hop 1 should engage whichever operation is
evaluated first.

Three measurements, in increasing order of how hard they are to fake:

1. exact accuracy, all digits and the sign
2. does hop h's routing predict the operation evaluated at step h, and does it predict it
   better than it predicts the *other* operation
3. for pairs of expressions with identical tokens but different evaluation order — the bracket
   flag flipped, and only where flipping it actually reorders — is hop 1 routing different

The third is the one that cannot be explained by anything except order, because the operands
and the operators are held fixed and the only thing that varies is which happens first.

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
N_OP = 3                                   # + - x
OPNAME = ("+", "-", "x")
PREC = (0, 0, 1)                           # x binds tighter than + and -
D, N_EXPERT, TOP_K, BLOCK = 96, 64, 4, 4
N_BLOCK = N_EXPERT // BLOCK
IN_DIM = 3 * BASE + 2 * N_OP + 1           # a, b, c, op1, op2, bracket flag
N_DIGIT = 3                                # |result| <= 162


def evaluate(a, b, c, op1, op2, bracket):
    """Value, and the order the two operations are applied in.

    Returns `(value, first, second)` where `first`/`second` are operation indices. With a
    bracket the left operation always goes first; without one, precedence decides.
    """
    def ap(x, y, op):
        return (x + y, x - y, x * y)[op]

    left_first = bracket or PREC[op1] >= PREC[op2]
    if left_first:
        return ap(ap(a, b, op1), c, op2), op1, op2
    return ap(a, ap(b, c, op2), op1), op2, op1


def dataset():
    xs, ys, order, flags = [], [], [], []
    for a in range(BASE):
        for b in range(BASE):
            for c in range(BASE):
                for op1 in range(N_OP):
                    for op2 in range(N_OP):
                        for br in (0, 1):
                            v, f, s = evaluate(a, b, c, op1, op2, br)
                            x = torch.zeros(IN_DIM)
                            x[a] = x[BASE + b] = x[2 * BASE + c] = 1.0
                            x[3 * BASE + op1] = 1.0
                            x[3 * BASE + N_OP + op2] = 1.0
                            x[3 * BASE + 2 * N_OP] = float(br)
                            xs.append(x)
                            m = abs(v)
                            ys.append([1 if v < 0 else 0] +
                                      [(m // 10**k) % 10 for k in (2, 1, 0)])
                            order.append([f, s])
                            # Identity of the token content, so bracket pairs can be found.
                            flags.append([a, b, c, op1, op2, br])
    return (torch.stack(xs), torch.tensor(ys), torch.tensor(order), torch.tensor(flags))


class ExprMoE(nn.Module):
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


def order_separation(probs, order):
    """Push hop h's routing apart by which operation is evaluated at step h.

    The floor term from `floor.py`, applied to the evaluation order instead of to a single
    operation label — the structure this dataset exists to expose.
    """
    total = []
    for h in range(probs.shape[1]):
        prof = [probs[:, h][order[:, h] == o].mean(0)
                for o in range(N_OP) if (order[:, h] == o).any()]
        if len(prof) > 1:
            total += [0.5 * (prof[i] - prof[j]).abs().sum()
                      for i in range(len(prof)) for j in range(i + 1, len(prof))]
    return torch.stack(total).mean() if total else probs.sum() * 0.0


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


def bracket_pairs(flags, probs, dev):
    """Routing divergence between the two bracketings of the same tokens.

    Only pairs where the bracket actually changes the evaluation order count — when precedence
    already puts the left operation first, the bracket is decoration and routing *should* be
    identical. Those pairs are the control: a model reacting to the flag rather than to the
    order would separate them too.
    """
    key = {}
    for i, f in enumerate(flags.tolist()):
        key[tuple(f)] = i
    changed, decorative = [], []
    for f, i in key.items():
        if f[5] != 0:
            continue
        j = key.get((f[0], f[1], f[2], f[3], f[4], 1))
        if j is None:
            continue
        reorders = PREC[f[3]] < PREC[f[4]]          # bracket only matters when x is on the right
        d = 0.5 * (probs[i, 0] - probs[j, 0]).abs().sum()
        (changed if reorders else decorative).append(float(d))
    return (sum(changed) / max(len(changed), 1), len(changed),
            sum(decorative) / max(len(decorative), 1), len(decorative))


def run(lam, beta, steps, hops, dev, x, y, order, flags):
    torch.manual_seed(0)
    m = ExprMoE(hops).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    t0 = time.time()
    for _ in range(steps):
        i = torch.randint(0, len(x), (512,), device=dev)
        sl, dl, _, probs = m(x[i])
        loss = F.cross_entropy(sl, y[i, 0])
        for j, d in enumerate(dl):
            loss = loss + F.cross_entropy(d, y[i, j + 1])
        if lam:
            loss = loss + lam * compression_loss(probs)
        if beta:
            loss = loss - beta * order_separation(probs, order[i])
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        sl, dl, picks, probs = m(x)
        ok = sl.argmax(-1) == y[:, 0]
        for j, d in enumerate(dl):
            ok &= d.argmax(-1) == y[:, j + 1]
        acc = float(ok.float().mean())

    # Does hop h's routing know which operation runs at step h — more than it knows the other?
    h0 = probs[:, 0]
    first_acc, first_maj = probe(h0, order[:, 0], N_OP, dev)
    second_acc, _ = probe(h0, order[:, 1], N_OP, dev)
    div_ch, n_ch, div_dec, n_dec = bracket_pairs(flags, probs, dev)

    return {"lambda": lam, "beta": beta, "hops": hops, "exact_accuracy": round(acc, 4),
            "hop1_predicts_first_op": round(first_acc, 4),
            "hop1_predicts_second_op": round(second_acc, 4),
            "majority_baseline": round(first_maj, 4),
            "routing_divergence_reordering": round(div_ch, 4), "n_reordering_pairs": n_ch,
            "routing_divergence_decorative": round(div_dec, 4), "n_decorative_pairs": n_dec,
            "experts_used": int(len(torch.unique(picks))),
            "seconds": round(time.time() - t0, 1)}


def main(steps=3000, out="data/custom/exprs.json", hops=2):
    steps, hops = int(steps), int(hops)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    x, y, order, flags = dataset()
    x, y, order = x.to(dev), y.to(dev), order.to(dev)
    print(f"{len(x)} expressions of the form  a op1 b op2 c  with and without a bracket")
    print(f"{N_EXPERT} experts, top-{TOP_K}, {hops} hops, device {dev}\n")
    print(f"{'lam':>5} {'beta':>5} {'exact acc':>10} {'hop1->1st':>10} {'hop1->2nd':>10} "
          f"{'majority':>9} {'div reorder':>12} {'div decor':>10} {'experts':>8}")
    rows = []
    for lam, beta in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)):
        r = run(lam, beta, steps, hops, dev, x, y, order, flags)
        rows.append(r)
        print(f"{lam:>5.1f} {beta:>5.1f} {r['exact_accuracy']:>10.3f} "
              f"{r['hop1_predicts_first_op']:>10.3f} {r['hop1_predicts_second_op']:>10.3f} "
              f"{r['majority_baseline']:>9.3f} {r['routing_divergence_reordering']:>12.3f} "
              f"{r['routing_divergence_decorative']:>10.3f} {r['experts_used']:>8d}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(
        {"n_expr": len(x), "n_expert": N_EXPERT, "hops": hops, "steps": steps,
         "reordering_pairs": rows[0]["n_reordering_pairs"],
         "decorative_pairs": rows[0]["n_decorative_pairs"], "rows": rows}, indent=2))
    print(f"\nwrote {out}")
    print(f"{rows[0]['n_reordering_pairs']} bracket pairs change the evaluation order, "
          f"{rows[0]['n_decorative_pairs']} do not.")
    print("`div reorder` above `div decor` means routing tracks the order, not the flag.")


if __name__ == "__main__":
    main(*sys.argv[1:])
