#!/usr/bin/env python3
"""A cost function that is not geometric: how the network routes, not where it lands.

MPEE's road matrices cost travel time. Every measurement here has assumed the analogue is
angular distance between embeddings — where a point sits — and that assumption turned out to
matter enormously: the same landmarks give 0.91 % exact cells against the chord metric and
39.08 % against the geodesic. If the choice of cost swings the answer by forty times, it
deserves to be a variable rather than a default.

Weight strength is the obvious other candidate, and in a routed network it is directly
observable: the router's distribution over experts, per hop, is how strongly this input engages
each part of the model. Two inputs are close if they *use the network the same way*, which is a
statement about computation rather than position.

Three cost functions on the same trained model and the same examples:

    embedding    angular distance between final hidden states      what we have been using
    routing      angular distance between router distributions     how the weights are engaged
    selection    Jaccard distance between the sets of experts      the hard version of the same

Scored the same three ways as everything else: does it cluster (class or carry recovery), does
its kNN graph concentrate traffic, and does a betweenness-landmark index reproduce it exactly.

Run with a torch venv, e.g. /Users/punnerud/Downloads/ainmt/venv/bin/python3.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "hops"))
from dump_layers import HOPS, LayerMoE, TASKS, batch, block_compression, carries, is_holdout
import waypoints as WP

SPEC = TASKS["add"]
A_LIM = 999


def pool(n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        a, b = rng.randint(-A_LIM, A_LIM), rng.randint(-A_LIM, A_LIM)
        op = rng.randrange(2)
        if not is_holdout(a, b, op, A_LIM, A_LIM):
            out.append((a, b, op))
    return out


def train(steps, dev, seed=0, bs=384):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    m = LayerMoE(SPEC).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x, sg, dg = batch(pool(bs, rng.randrange(1 << 30)), SPEC, dev)
        sl, dl, _, probs, _ = m(x)
        loss = F.cross_entropy(sl, sg)
        for j, d in enumerate(dl):
            loss = loss + F.cross_entropy(d, dg[:, j])
        loss = loss + 0.3 * block_compression(probs)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m


def representations(m, items, dev, chunk=2048):
    emb, rout, sel = [], [], []
    for i in range(0, len(items), chunk):
        x, _, _ = batch(items[i:i + chunk], SPEC, dev)
        with torch.no_grad():
            _, _, st, probs, picks = m(x)
        emb.append(st[:, HOPS - 1].cpu().float())
        rout.append(probs.reshape(probs.shape[0], -1).cpu().float())
        oh = torch.zeros(picks.shape[0], probs.shape[-1])
        for h in range(picks.shape[1]):
            oh.scatter_(1, picks[:, h].cpu(), 1.0)
        sel.append(oh)
    return torch.cat(emb), torch.cat(rout), torch.cat(sel)


def score(name, x, truth, k_classes, out_rows, k=16):
    x = F.normalize(x, dim=-1)
    n = x.shape[0]
    torch.manual_seed(0)
    c = x[torch.randperm(n)[:k_classes]].clone()
    for _ in range(40):
        a = (x @ c.t()).argmax(-1)
        for j in range(k_classes):
            msk = a == j
            if msk.any():
                c[j] = F.normalize(x[msk].mean(0), dim=-1)
    pred = (x @ c.t()).argmax(-1).tolist()
    hit = sum(Counter(truth[i] for i in range(n) if pred[i] == cc).most_common(1)[0][1]
              for cc in set(pred))
    pur = hit / n
    base = max(Counter(truth).values()) / n

    adj = WP.knn_graph(x, k)
    g = torch.Generator().manual_seed(0)
    srcs = torch.randperm(n, generator=g)[:120].tolist()
    tgts = torch.randperm(n, generator=g)[:50].tolist()
    bet = Counter()
    for s in srcs:
        for p in WP.dijkstra_paths(adj, s, tgts):
            bet.update(p)
    cnt = torch.zeros(n)
    for node, v in bet.items():
        cnt[node] = v
    order = cnt.argsort(descending=True)
    top1 = max(1, n // 100)
    conc = float(cnt[order[:top1]].sum() / cnt.sum()) / (top1 / n) if cnt.sum() > 0 else 0.0

    out_rows.append({"cost": name, "purity": round(pur, 4), "majority": round(base, 4),
                     "carry_lift": round(pur / base, 3),
                     "gateway_concentration": round(conc, 2)})
    print(f"{name:>12} {pur:>8.3f} {base:>9.3f} {pur / base:>7.2f}x {conc:>13.1f}x")


def main(n=4000, steps=9600, out="data/custom/routingcost.json"):
    n, steps = int(n), int(steps)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    items = pool(n, 999)
    truth = [carries(a, b, op) for a, b, op in items]
    print(f"{n} problems, model trained {steps} steps, {dev}")
    print("truth axis: carry count (the partition worth +0.157 to +0.226)\n")
    m = train(steps, dev)
    emb, rout, sel = representations(m, items, dev)
    print(f"{'cost':>12} {'purity':>8} {'majority':>9} {'lift':>8} {'gateway conc':>14}")
    rows = []
    score("embedding", emb, truth, 4, rows)
    score("routing", rout, truth, 4, rows)
    score("selection", sel, truth, 4, rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"n": n, "steps": steps, "rows": rows}, indent=2))
    print(f"\nwrote {out}")
    print("A routing cost that beats the embedding cost would say the useful structure is in")
    print("how the weights are engaged, not in where the activations land.")


if __name__ == "__main__":
    main(*sys.argv[1:])
