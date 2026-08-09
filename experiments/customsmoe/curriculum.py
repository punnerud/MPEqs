#!/usr/bin/env python3
"""The network picks where to practise, from its own uncertainty.

Everything so far handed the model a fixed task set and measured what it did with it. This
inverts that: the task space is far too large to enumerate — three-digit signed operands over
four operations is about sixteen million problems — so the model never sees all of it, and what
it *chooses* to train on becomes part of the method.

The loop is deliberately simpler than AlphaZero. There is no opponent and no self-play; there
is a solver that always knows the right answer, and the only objective is to compress the space
while staying correct. Each round the model is scored on a probe set, the buckets it is least
certain about get a larger share of the next round's training, and the operations are widened
on a schedule. That is the whole search: not what move to play, but what to practise.

**Why the space has to be big.** `shrink.py` found that on 49 800 enumerated problems no
network compresses at a usable accuracy — the numerator is simply too small. Against sixteen
million problems the same network crosses 1.0 at almost any accuracy, which moves the question
where it belongs: reproduced bits may only be claimed for problems the model has *not* seen, so
the ratio is computed from **held-out** accuracy. Memorising the training sample buys nothing.

**Why carries and negatives.** Addition and subtraction stop being uniform once the numbers are
long enough to carry and borrow, and negative operands add a case split that has nothing to do
with the digits. Accuracy is reported per carry count so the difficulty is visible rather than
averaged away, and that breakdown is also what the uncertainty signal is supposed to find on
its own.

Two arms, identical compute: sampling driven by uncertainty against sampling uniform over the
same buckets. If the uncertainty arm does not win, the self-curriculum is decoration.

Run with a torch venv, e.g. /Users/punnerud/Downloads/ainmt/venv/bin/python3.
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

MAXD = 3                                     # up to three digits per operand
LIM = 10**MAXD - 1                           # 999
N_OP = 4                                     # +  -  x  //
OPNAME = ("+", "-", "x", "//")
OUT_DIGITS = 6                               # |a*b| <= 998001
ANSWER_BITS = 1 + OUT_DIGITS * math.log2(10)
IN_DIM = 2 * (MAXD * 10 + 1) + N_OP          # digits + sign per operand, then the operation
D, N_EXPERT, TOP_K, BLOCK = 64, 32, 4, 4
STAGES = ((0,), (0, 1), (0, 1, 2), (0, 1, 2, 3))
HOLDOUT_MOD = 97                             # problems in this residue class are never trained


def solve(a, b, op):
    if op == 0:
        return a + b
    if op == 1:
        return a - b
    if op == 2:
        return a * b
    return int(a / b) if b else 0            # truncating division, defined for b != 0


def carries(a, b, op):
    """Carries in an addition or borrows in a subtraction, on magnitudes. -1 if not applicable."""
    if op > 1:
        return -1
    x, y = abs(a), abs(b)
    if op == 1 and (a < 0) != (b < 0):
        pass                                 # a - (-b) is an addition of magnitudes
    n, c = 0, 0
    for k in range(MAXD + 1):
        da, db = (x // 10**k) % 10, (y // 10**k) % 10
        s = da + db + c
        c = 1 if s >= 10 else 0
        n += c
    return n


def is_holdout(a, b, op):
    return ((a + LIM) * 7919 + (b + LIM) * 104729 + op) % HOLDOUT_MOD == 0


def encode(a, b, op, out):
    """In-place one-hot encode into `out` (already zeroed)."""
    for k in range(MAXD):
        out[k * 10 + (abs(a) // 10**k) % 10] = 1.0
    out[MAXD * 10] = 1.0 if a < 0 else 0.0
    off = MAXD * 10 + 1
    for k in range(MAXD):
        out[off + k * 10 + (abs(b) // 10**k) % 10] = 1.0
    out[off + MAXD * 10] = 1.0 if b < 0 else 0.0
    out[2 * (MAXD * 10 + 1) + op] = 1.0


def bucket_of(a, b, op):
    """(op, digits of a, digits of b, sign a, sign b) — 4 x 3 x 3 x 2 x 2 = 144 buckets."""
    da = len(str(abs(a)))
    db = len(str(abs(b)))
    return ((op * 3 + da - 1) * 3 + db - 1) * 4 + (2 if a < 0 else 0) + (1 if b < 0 else 0)


N_BUCKET = N_OP * 3 * 3 * 4


def sample_bucket(bucket, rng):
    op, rest = divmod(bucket, 36)
    sgn = rest % 4
    rest //= 4
    da, db = rest // 3 + 1, rest % 3 + 1
    lo_a, hi_a = 10**(da - 1) if da > 1 else 0, 10**da - 1
    lo_b, hi_b = 10**(db - 1) if db > 1 else 0, 10**db - 1
    for _ in range(20):
        a = rng.randint(lo_a, hi_a) * (-1 if sgn & 2 else 1)
        b = rng.randint(lo_b, hi_b) * (-1 if sgn & 1 else 1)
        if op == 3 and b == 0:
            continue
        if is_holdout(a, b, op):
            continue
        return a, b, op
    return None


def batch(items, dev):
    x = torch.zeros(len(items), IN_DIM)
    sg = torch.zeros(len(items), dtype=torch.long)
    dg = torch.zeros(len(items), OUT_DIGITS, dtype=torch.long)
    for i, (a, b, op) in enumerate(items):
        encode(a, b, op, x[i])
        v = solve(a, b, op)
        sg[i] = 1 if v < 0 else 0
        m = abs(v)
        for k in range(OUT_DIGITS):
            dg[i, k] = (m // 10**k) % 10
    return x.to(dev), sg.to(dev), dg.to(dev)


class CurricMoE(nn.Module):
    def __init__(self, hops=3):
        super().__init__()
        self.hops = hops
        self.inp = nn.Linear(IN_DIM, D)
        self.router = nn.Linear(D, N_EXPERT, bias=False)
        self.experts = nn.Parameter(torch.randn(N_EXPERT, D, D) * (1.0 / D**0.5))
        self.norm = nn.LayerNorm(D)
        self.sign = nn.Linear(D, 2)
        self.digits = nn.ModuleList([nn.Linear(D, 10) for _ in range(OUT_DIGITS)])

    def forward(self, x):
        h = torch.relu(self.inp(x))
        probs_all, picks = [], []
        for _ in range(self.hops):
            probs = self.router(h).softmax(-1)
            w, idx = probs.topk(TOP_K, dim=-1)
            y = torch.einsum("bkij,bj->bki", self.experts[idx], h)
            h = self.norm(h + (y * w.unsqueeze(-1)).sum(1))
            probs_all.append(probs)
            picks.append(idx)
        return (self.sign(h), [d(h) for d in self.digits],
                torch.stack(probs_all, 1), torch.stack(picks, 1))


def block_compression(probs):
    nb = N_EXPERT // BLOCK
    m = probs.view(*probs.shape[:-1], nb, BLOCK).sum(-1).clamp(1e-6, 1.0)
    return (1.0 - (1.0 - m).pow(TOP_K)).sum(-1).mean()


def holdout_set(ops, rng, n=6000):
    """Problems the training sampler is forbidden to draw, so accuracy is generalisation."""
    out = []
    while len(out) < n:
        op = ops[rng.randrange(len(ops))]
        a = rng.randint(-LIM, LIM)
        b = rng.randint(-LIM, LIM)
        if op == 3 and b == 0:
            continue
        if is_holdout(a, b, op):
            out.append((a, b, op))
    return out


def assess(m, items, dev, chunk=4096):
    """Exact accuracy and mean predictive entropy, per item."""
    x, sg, dg = batch(items, dev)
    ok = torch.zeros(len(items), dtype=torch.bool, device=dev)
    ent = torch.zeros(len(items), device=dev)
    with torch.no_grad():
        for i in range(0, len(items), chunk):
            sl, dl, _, _ = m(x[i:i + chunk])
            good = sl.argmax(-1) == sg[i:i + chunk]
            e = -(sl.softmax(-1) * sl.log_softmax(-1)).sum(-1)
            for j, d in enumerate(dl):
                good &= d.argmax(-1) == dg[i:i + chunk, j]
                e = e - (d.softmax(-1) * d.log_softmax(-1)).sum(-1)
            ok[i:i + chunk] = good
            ent[i:i + chunk] = e
    return ok.cpu(), ent.cpu()


def run(arm, rounds, steps_per_round, dev, seed=0):
    import random
    rng = random.Random(seed)
    torch.manual_seed(seed)
    m = CurricMoE().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    weights = torch.ones(N_BUCKET)
    log = []

    for rd in range(rounds):
        ops = STAGES[min(rd * len(STAGES) // rounds, len(STAGES) - 1)]
        live = torch.zeros(N_BUCKET, dtype=torch.bool)
        for bk in range(N_BUCKET):
            if bk // 36 in ops:
                live[bk] = True
        w = torch.where(live, weights, torch.zeros_like(weights)).clamp(min=1e-6)
        w = w / w.sum()

        for _ in range(steps_per_round):
            items = []
            picks = torch.multinomial(w, 384, replacement=True).tolist()
            for bk in picks:
                s = sample_bucket(bk, rng)
                if s:
                    items.append(s)
            if len(items) < 32:
                continue
            x, sg, dg = batch(items, dev)
            sl, dl, probs, _ = m(x)
            loss = F.cross_entropy(sl, sg)
            for j, d in enumerate(dl):
                loss = loss + F.cross_entropy(d, dg[:, j])
            loss = loss + 0.3 * block_compression(probs)
            opt.zero_grad(); loss.backward(); opt.step()

        # Score on held-out problems, then decide where to practise next.
        hold = holdout_set(ops, rng)
        ok, ent = assess(m, hold, dev)
        acc = float(ok.float().mean())
        if arm == "uncertain":
            agg = torch.zeros(N_BUCKET)
            cnt = torch.zeros(N_BUCKET)
            for (a, b, op), e in zip(hold, ent.tolist()):
                bk = bucket_of(a, b, op)
                agg[bk] += e
                cnt[bk] += 1
            weights = torch.where(cnt > 0, agg / cnt.clamp(min=1), weights)
        log.append({"round": rd, "ops": [OPNAME[o] for o in ops],
                    "holdout_accuracy": round(acc, 4)})

    # Final assessment on the full operation set, with the carry breakdown.
    hold = holdout_set(STAGES[-1], rng, n=12000)
    ok, _ = assess(m, hold, dev)
    by_op, by_carry = {}, {}
    for (a, b, op), good in zip(hold, ok.tolist()):
        by_op.setdefault(OPNAME[op], []).append(good)
        c = carries(a, b, op)
        if c >= 0:
            by_carry.setdefault(c, []).append(good)
    used = set()
    x, _, _ = batch(hold[:4096], dev)
    with torch.no_grad():
        used |= set(m(x)[3].reshape(-1).tolist())
    resident = sum(p.numel() for p in m.parameters()) - (N_EXPERT - len(used)) * D * D

    acc = float(ok.float().mean())
    space = sum(len(range(-LIM, LIM + 1))**2 for _ in STAGES[-1])
    ratio = space * acc * ANSWER_BITS / (resident * 32)
    return {"arm": arm, "rounds": rounds, "final_holdout_accuracy": round(acc, 4),
            "per_op": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_op.items())},
            "per_carry": {str(k): round(sum(v) / len(v), 3)
                          for k, v in sorted(by_carry.items())},
            "experts_used": len(used), "params_effective": resident,
            "task_space": space, "compression_ratio": round(ratio, 2), "log": log}


def main(rounds=12, steps=400, out="data/custom/curriculum.json"):
    rounds, steps = int(rounds), int(steps)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    space = 4 * (2 * LIM + 1) ** 2
    print(f"three-digit signed operands, {N_OP} operations "
          f"=> {space:,} problems, {ANSWER_BITS:.1f} bits each")
    print(f"{rounds} rounds x {steps} steps, device {dev}. "
          f"Holdout is 1 problem in {HOLDOUT_MOD}, never trained on.\n")
    results = []
    for arm in ("uniform", "uncertain"):
        t0 = time.time()
        r = run(arm, rounds, steps, dev)
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)
        print(f"{arm:>10}  holdout acc {r['final_holdout_accuracy']:.3f}  "
              f"experts {r['experts_used']:>2}  ratio {r['compression_ratio']:>8.1f}  "
              f"{r['seconds']:.0f}s")
        print(f"            per op    {r['per_op']}")
        print(f"            per carry {r['per_carry']}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"task_space": space, "answer_bits": ANSWER_BITS,
                                     "rounds": rounds, "steps_per_round": steps,
                                     "results": results}, indent=2))
    print(f"\nwrote {out}")
    u, c = results
    d = c["final_holdout_accuracy"] - u["final_holdout_accuracy"]
    print(f"uncertainty-driven sampling changes held-out accuracy by {d:+.3f} "
          f"at equal compute — {'worth it' if d > 0.01 else 'not worth it'}.")


if __name__ == "__main__":
    main(*sys.argv[1:])
