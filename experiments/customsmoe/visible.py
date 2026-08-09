#!/usr/bin/env python3
"""The control for the carry result: a difficulty axis that IS visible in the input.

No model-derived signal recovered the carry partition — lift 1.00 to 1.04 across six candidates
and three training levels — and the reason was the effect size, not the estimator: within-level
spread runs thirty to fifteen hundred times the between-level gap. The caveat attached to that
was that carries are nearly an adversarial case, being sharp, discrete and almost absent from
the surface form. A property the input *shows* should be found.

So test that, on the same architecture, with the same six signals and the same scoring.
Operation type is the cleanest visible axis available: the symbol is right there in the input,
the classes are balanced by construction rather than by resampling, and the difficulty really
does differ — three-digit multiplication is far harder than three-digit addition.

Both axes are scored side by side on the *same* problems and the same trained model, so the
comparison isolates the property and not the task, the model, or the metric:

    operation   visible in the input        expected: found
    carries     invisible, +/- only         known: not found, 1.00-1.04x

If the visible axis is found and the invisible one is not, the earlier negative is bounded
rather than general, and the rule is about what the signal can see. If *neither* is found, the
signals are weaker than the caveat allowed and the earlier result was too generous to them.

Run with a torch venv, e.g. /Users/punnerud/Downloads/ainmt/venv/bin/python3.
"""
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import autogroup as AG  # noqa: E402
from dump_layers import (HOPS, LayerMoE, OPNAME, TASKS, batch,  # noqa: E402
                         block_compression, carries, is_holdout)

# Three operations over three-digit signed operands. `solve` and the input encoding in
# dump_layers already cover +, - and x, so this only adds the width and the output size:
# |999 x 999| = 998001, six digits.
TASKS["ops"] = (3, 3, (0, 1, 2), 6)
SPEC = TASKS["ops"]
A_LIM = 999
NBUCKET = 3


def pool(n, seed=1234):
    """Equal mass per operation, so the visible axis is balanced without resampling."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        a, b = rng.randint(-A_LIM, A_LIM), rng.randint(-A_LIM, A_LIM)
        op = SPEC[2][len(out) % len(SPEC[2])]
        if not is_holdout(a, b, op, A_LIM, A_LIM):
            out.append((a, b, op))
    rng.shuffle(out)
    return out


def train(steps, dev, seed=0, bs=384):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    m = LayerMoE(SPEC).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x, sg, dg = batch(pool(bs, rng.randrange(1 << 30)), SPEC, dev)
        sl, dl, _, probs, _ = m(x)
        loss = F.cross_entropy(sl, sg)
        for j, d in enumerate(dl):
            loss = loss + F.cross_entropy(d, dg[:, j])
        loss = loss + 0.3 * block_compression(probs)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m


def per_example(m, items, dev, chunk=2048):
    ent, loss, states, ok = [], [], [], []
    for i in range(0, len(items), chunk):
        x, sg, dg = batch(items[i:i + chunk], SPEC, dev)
        with torch.no_grad():
            sl, dl, st, _, _ = m(x)
            e = -(sl.softmax(-1) * sl.log_softmax(-1)).sum(-1)
            ll = F.cross_entropy(sl, sg, reduction="none")
            good = sl.argmax(-1) == sg
            for j, d in enumerate(dl):
                e = e - (d.softmax(-1) * d.log_softmax(-1)).sum(-1)
                ll = ll + F.cross_entropy(d, dg[:, j], reduction="none")
                good &= d.argmax(-1) == dg[:, j]
            ent.append(e.cpu()); loss.append(ll.cpu())
            states.append(st[:, HOPS - 1].cpu().float()); ok.append(good.cpu())
    return torch.cat(ent), torch.cat(loss), torch.cat(states), torch.cat(ok)


def grad_norms(m, items, dev, group=32):
    out = []
    for i in range(0, len(items), group):
        x, sg, dg = batch(items[i:i + group], SPEC, dev)
        sl, dl, _, _, _ = m(x)
        loss = F.cross_entropy(sl, sg)
        for j, d in enumerate(dl):
            loss = loss + F.cross_entropy(d, dg[:, j])
        m.zero_grad(); loss.backward()
        g = torch.sqrt(sum((p.grad**2).sum() for p in m.parameters() if p.grad is not None))
        out.extend([float(g)] * len(items[i:i + group]))
    m.zero_grad()
    return torch.tensor(out)


def main(n=9000, out="data/custom/visible.json", stages="6000,19200"):
    n = int(n)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    items = pool(n)
    truth_op = [op for _, _, op in items]
    # Carries are only defined for + and -; multiplication rows carry -1 and are excluded from
    # that axis rather than folded into a bucket they do not belong in.
    truth_c = [carries(a, b, op) for a, b, op in items]
    keep_c = [i for i, c in enumerate(truth_c) if c >= 0]

    print(f"{n:,} problems, equal mass over {' '.join(OPNAME[o] for o in SPEC[2])}, {dev}")
    print(f"operation mix: {dict(sorted(Counter(truth_op).items()))}")
    print("\nsame model, same signals, two candidate partitions:")
    print("  operation — visible in the input")
    print("  carries   — invisible, and already measured at 1.00-1.04x\n")

    rows = []
    for stage in stages.split(","):
        steps = int(stage)
        t0 = time.time()
        m = train(steps, dev)
        ent, loss, states, ok = per_example(m, items, dev)
        gn = grad_norms(m, items, dev)
        resid = AG.landmark_residual(states)

        by_op = {OPNAME[o]: round(float(ok[[i for i, p in enumerate(truth_op) if p == o]]
                                        .float().mean()), 3) for o in SPEC[2]}
        print(f"{steps} steps ({time.time() - t0:.0f}s), accuracy by operation: {by_op}")

        sigs = {
            "layer-cluster": AG.kmeans_labels(states, NBUCKET),
            "entropy": AG.quantile_labels(ent, NBUCKET),
            "loss": AG.quantile_labels(loss, NBUCKET),
            "gradnorm": AG.quantile_labels(gn, NBUCKET),
            "residual (MPEE)": AG.quantile_labels(resid, NBUCKET),
        }
        scalars = {"entropy": ent, "loss": loss, "gradnorm": gn, "residual (MPEE)": resid}

        print(f"{'':>18} {'op lift':>9} {'op b/w':>9} {'carry lift':>11} {'carry b/w':>10}")
        for name, lab in sigs.items():
            _, _, lf_op = AG.lift(lab, truth_op, NBUCKET)
            sub_lab = [lab[i] for i in keep_c]
            sub_tr = [truth_c[i] for i in keep_c]
            _, _, lf_c = AG.lift(sub_lab, sub_tr, 4)
            sep_op = sep_c = None
            if name in scalars:
                sep_op, _ = AG.separability(
                    scalars[name], [SPEC[2].index(o) for o in truth_op], NBUCKET)
                sep_c, _ = AG.separability(
                    scalars[name][torch.tensor(keep_c)], sub_tr, 4)
            rows.append({"stage": stage, "signal": name,
                         "op_lift": round(lf_op, 4), "carry_lift": round(lf_c, 4),
                         "op_between_over_within": (round(sep_op, 5) if sep_op else None),
                         "carry_between_over_within": (round(sep_c, 5) if sep_c else None),
                         "accuracy_by_op": by_op})
            f = lambda v: f"{v:>9.4f}" if v is not None else f"{'-':>9}"
            print(f"{name:>18} {lf_op:>8.2f}x {f(sep_op)} {lf_c:>10.2f}x {f(sep_c)[1:]}")
        print()

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"n": n, "buckets": NBUCKET, "rows": rows}, indent=2))
    print(f"wrote {out}")
    best_op = max(r["op_lift"] for r in rows)
    best_c = max(r["carry_lift"] for r in rows)
    print(f"\nbest lift toward the VISIBLE axis:   {best_op:.2f}x")
    print(f"best lift toward the INVISIBLE axis: {best_c:.2f}x")
    print("Perfect recovery would be 3.00x on operation and 4.00x on carries.")


if __name__ == "__main__":
    main(*sys.argv[1:])
