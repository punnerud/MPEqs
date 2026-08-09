#!/usr/bin/env python3
"""Chains of strong links, not shortest paths — the algebra was wrong.

Everything in the landmark thread minimised a SUM of distances. Under (min, +) the triangle
inequality guarantees the direct edge is never worse than a detour, so extra hops cannot help
and repeated squaring is idempotent. That is exactly what was measured: 27.92 % at two hops,
27.90 % at five, unchanged. The conclusion drawn — "more hops does nothing" — is true of that
question and answers the wrong one.

The question worth asking is different. A pair may have no direct resemblance and still be
connected by a chain in which *every individual step* is strong: Python resembles Java, Java
resembles programming in general, and the far ends are linked through the middle rather than
directly. That is a discovered relationship, not a shortcut, and no triangle inequality caps it
because similarities multiply rather than add.

So the algebra is (max, min) over ABSOLUTE similarity in [0, 1], with a bounded per-hop cost:

    strength(chain) = min over links of  s(link)   -   penalty * (hops - 1)
    S_k(i, j)       = max over chains of at most k hops

**Not a product.** Multiplying similarities compounds: a chain of numbers below 1 decays
exponentially, and the fix of normalising them towards 1 removes the decay along with all the
information. Either way the product is ill-conditioned, in exactly the way exploding and
vanishing gradients are. The minimum has neither failure: a chain is exactly as strong as its
weakest link, which is also what "every step is strong" means.

**Absolute, not relative.** An earlier version scored the *relative* strength — each link
divided by its node's own strongest — and it produced a measurable disaster. A prose chunk's
nearest neighbour sits at 1.248 radians, nearly a right angle, and still scored 0.994 because it
was that node's best. The chain then wandered through prose at ~1.000 per hop while its true
similarity was 0.60. Relative values are kept, but only to decide which links are ADMITTED;
scoring uses the absolute value, because "strong for me" and "strong" are different claims.

The penalty is subtracted rather than multiplied so it cannot compound either, and it is clipped
at zero so nothing ever exceeds 1.

Three things are measured, and the third is the one that decides whether any of it is real:

1. how much of the graph is best explained by a chain rather than a direct link, per hop count
2. the correlation between consecutive hop levels — the specific ask: 1-to-5 may be uncorrelated
   while 1-to-2 and 2-to-3 are strong, which would say the chain degrades gradually rather than
   jumping to noise
3. whether the NEW connections are RIGHT. On the 39-class set a chain-discovered link can be
   checked against the labels: if chains link same-class pairs more often than direct links of
   equal strength, they have found real structure; if not, they have found a path through noise.

Run with a torch venv, e.g. /Users/punnerud/Downloads/ainmt/venv/bin/python3.
"""
import json
import struct
import sys
from pathlib import Path

import torch


def load(path, dim, limit=0):
    raw = Path(path).read_bytes()
    n = len(raw) // 4 // dim
    if limit:
        n = min(n, limit)
    x = torch.tensor(struct.unpack(f"<{n * dim}f", raw[: n * dim * 4])).view(n, dim)
    return torch.nn.functional.normalize(x, dim=-1)


def similarity(x, knn_k, rel_floor):
    """Relative link strength in [0,1], kept only where the link is strong *for that node*.

    Two decisions here, and both are the difference between a measure and a mirage.

    **Sparsify.** A dense similarity has an entry for every pair, so `max over chains` always
    finds the direct edge and there is nothing left to discover. The chain only means something
    when most pairs have no direct link.

    **Normalise per node, then floor.** Absolute similarity is not comparable across an
    embedding: a dense region has every neighbour at 0.9 and a sparse one has its nearest at
    0.4, so an absolute cutoff would keep every link in the dense region and none in the sparse
    one, and every chain found would be a chain through the dense part. Each node's links are
    therefore divided by its own strongest, and only links within `rel_floor` of that are kept.

    This is also what stops the search from wandering through weak links: k nearest keeps k
    whatever their quality, and the floor throws away the ones that are only nearest by default.
    """
    n = x.shape[0]
    sim = (x @ x.t()).clamp(-1.0, 1.0)
    sim = 1.0 - torch.arccos(sim) / torch.pi          # angle -> [0,1], 1 = identical
    sim.fill_diagonal_(0.0)
    keep = sim.topk(knn_k, dim=-1).indices
    sparse = torch.zeros_like(sim)
    sparse.scatter_(1, keep, sim.gather(1, keep))

    # Relative decides ADMISSION; the value kept is ABSOLUTE. Scoring on the relative value
    # rated a 1.248-radian link at 0.994 because it happened to be that node's best.
    best = sparse.max(dim=1, keepdim=True).values.clamp(min=1e-9)
    admitted = (sparse / best) >= rel_floor
    out = torch.where(admitted, sparse, torch.zeros(()))
    out = torch.maximum(out, out.t())                 # a link is a link in both directions
    out.fill_diagonal_(1.0)                           # a point reaches itself for free
    return out


def hop_closure(s1, hops, penalty):
    """S_k for k = 1..hops under (max, min): bottleneck strength, minus a per-hop cost.

    One step is `S_{k+1}[i,j] = max_m min(S_k[i,m], s1[m,j]) - penalty`, the widest-path
    recurrence. Bounded above by the weakest link on the route and below by zero, so no chain
    can compound its way to a large number and none can vanish to a denormal.
    """
    out = [s1]
    cur = s1
    for _ in range(hops - 1):
        nxt = torch.zeros_like(cur)
        block = max(1, 4096 * 4096 // max(cur.shape[0], 1))
        for a in range(0, cur.shape[0], block):
            b = min(a + block, cur.shape[0])
            # min over the two legs, maximised over the middle node
            legs = torch.minimum(cur[a:b].unsqueeze(2), s1.unsqueeze(0))   # [blk, m, j]
            nxt[a:b] = (legs.max(dim=1).values - penalty).clamp(min=0.0)
        cur = torch.maximum(cur, nxt)                          # "at most k hops"
        out.append(cur.clone())
    return out


def spearman(a, b):
    """Rank correlation on a sample, without scipy."""
    n = a.numel()
    idx = torch.randperm(n)[: min(n, 200_000)]
    ra = a.flatten()[idx].argsort().argsort().float()
    rb = b.flatten()[idx].argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm() + 1e-12))


def main(emb="data/labelled/emb.f32", dim=384, labels="data/labelled/labels.txt",
         knn_k=8, hops=6, penalty=0.02, rel_floor=0.9, limit=3000,
         out="data/custom/chains.json"):
    dim, knn_k, hops, limit = int(dim), int(knn_k), int(hops), int(limit)
    penalty, rel_floor = float(penalty), float(rel_floor)
    x = load(emb, dim, limit)
    n = x.shape[0]
    lab = None
    if labels and Path(labels).exists():
        lab = [l.strip() for l in open(labels)][:n]

    s1 = similarity(x, knn_k, rel_floor)
    edge_share = float((s1 > 0).float().mean())
    print(f"{n} points, kNN k={knn_k}, relative floor {rel_floor} "
          f"-> {100 * edge_share:.2f} % of pairs have a direct link")
    print(f"penalty {penalty} per hop (subtracted, clipped at 0), up to {hops} hops\n")

    levels = hop_closure(s1, hops, penalty)
    mask = ~torch.eye(n, dtype=torch.bool)

    print(f"{'hops':>5} {'reachable':>10} {'mean strength':>14} {'new vs prev':>12} "
          f"{'corr to prev':>13}")
    rows = []
    for k, sk in enumerate(levels, start=1):
        reach = float((sk[mask] > 0).float().mean())
        mean_s = float(sk[mask][sk[mask] > 0].mean()) if reach > 0 else 0.0
        if k == 1:
            new, corr = reach, 1.0
        else:
            prev = levels[k - 2]
            new = float(((sk[mask] > 0) & (prev[mask] == 0)).float().mean())
            corr = spearman(sk[mask], prev[mask])
        rows.append({"hops": k, "reachable_pct": round(100 * reach, 3),
                     "mean_strength": round(mean_s, 4),
                     "newly_connected_pct": round(100 * new, 3),
                     "spearman_to_prev": round(corr, 4)})
        print(f"{k:>5} {100 * reach:>9.2f}% {mean_s:>14.4f} {100 * new:>11.2f}% {corr:>13.3f}")

    # The correlation the question was actually about: 1-to-k, not k-to-(k-1).
    print(f"\n{'':>5} {'corr to 1 hop':>14}")
    far = []
    for k, sk in enumerate(levels, start=1):
        c = spearman(sk[mask], levels[0][mask])
        far.append({"hops": k, "spearman_to_1": round(c, 4)})
        print(f"{k:>5} {c:>14.3f}")

    # Are the new links right? Only answerable where labels exist.
    truth = None
    if lab is not None:
        same = torch.tensor([[1.0 if lab[i] == lab[j] else 0.0 for j in range(n)]
                             for i in range(n)])
        direct = (levels[0] > 0) & mask
        base_purity = float(same[direct].mean())
        print(f"\ndirect links that join the same class: {100 * base_purity:.1f} %")
        truth = {"direct_same_class_pct": round(100 * base_purity, 2), "by_hop": []}
        print(f"{'hops':>5} {'new links':>11} {'same class':>12} {'lift':>7}")
        for k in range(2, hops + 1):
            new_mask = (levels[k - 1] > 0) & (levels[k - 2] == 0) & mask
            cnt = int(new_mask.sum())
            if cnt == 0:
                print(f"{k:>5} {0:>11} {'-':>12} {'-':>7}")
                truth["by_hop"].append({"hops": k, "new_links": 0})
                continue
            pur = float(same[new_mask].mean())
            truth["by_hop"].append({"hops": k, "new_links": cnt,
                                    "same_class_pct": round(100 * pur, 2),
                                    "lift": round(pur / max(base_purity, 1e-9), 3)})
            print(f"{k:>5} {cnt:>11} {100 * pur:>11.1f}% {pur / base_purity:>6.2f}x")

    if lab is not None:
        nv = novelty_value(levels, lab, mask, hops)
        truth["novelty"] = nv
        print(f"\nDoes a chain link help SOLVE, or only restate what is already known?")
        print(f"one-step label propagation, {nv['classes']} classes")
        print(f"{'hops':>5} {'accuracy':>10} {'rescued':>9} {'broken':>8} {'net':>7}"
              f" {'cross%':>8} {'+cross only':>12}")
        print(f"{1:>5} {nv['direct_accuracy']:>10.3f} {'-':>9} {'-':>8} {'-':>7}"
              f" {'-':>8} {'-':>12}")
        _sweep = nv.get("threshold_sweep", [])
        for r in nv["by_hop"]:
            print(f"{r['hops']:>5} {r['accuracy']:>10.3f} {r['rescued']:>9} "
                  f"{r['broken']:>8} {r['net']:>+7} {r['cross_share_pct']:>7.1f}% "
                  f"{r['accuracy_with_cross_only']:>12.3f}")

    if lab is not None and nv.get("threshold_sweep"):
        print(f"\nAdmitting only the STRONGEST chain links — the relative adjustment applied")
        print(f"to the chains themselves, not just to the base graph:")
        print(f"{'hops':>5} {'keep top':>9} {'links':>8} {'cross%':>8} {'acc':>8} "
              f"{'rescued':>9} {'broken':>8} {'net':>7}")
        for r in nv["threshold_sweep"]:
            print(f"{r['hops']:>5} {1 - r['quantile']:>8.1%} {r['links_kept']:>8} "
                  f"{r['cross_share_pct']:>7.1f}% {r['accuracy']:>8.3f} {r['rescued']:>9} "
                  f"{r['broken']:>8} {r['net']:>+7}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(
        {"n": n, "knn_k": knn_k, "penalty": penalty, "rel_floor": rel_floor, "direct_edge_pct": round(100 * edge_share, 3),
         "levels": rows, "to_first_hop": far, "ground_truth": truth}, indent=2))
    print(f"\nwrote {out}")
    print("A chain is only interesting if the pairs it newly connects belong together. Lift")
    print("above 1 means the chains found real structure; at or below 1 they found a path")
    print("through noise, however strong every individual link on it was.")


def novelty_value(levels, lab, mask, hops):
    """Does a chain-found link help SOLVE something, rather than restate what is known?

    The same-class test above measures the wrong thing, and measuring it was a mistake worth
    keeping in the record. A link between two items already known to be alike carries no
    information: it is redundant by construction, and scoring chains on it rewards them for
    finding nothing. High same-class purity is what a *useless* discovery looks like.

    What matters is whether the connection contributes to an existing problem. That is
    measurable without an LLM in the loop, as compression: predict a held-out label from
    neighbours, once using only direct links and once with the chain-found links added. If the
    chains carry information the second prediction is better, and it is better precisely on the
    items the direct graph gets wrong.

    Reported both ways, because the split is the finding:
      - accuracy overall, direct against direct + chain
      - accuracy on the items direct links get WRONG, which is where new knowledge would show
    """
    n = len(lab)
    classes = sorted(set(lab))
    ci = {c: i for i, c in enumerate(classes)}
    y = torch.tensor([ci[c] for c in lab])
    onehot = torch.zeros(n, len(classes))
    onehot[torch.arange(n), y] = 1.0

    def predict(w):
        # Weighted vote from neighbours, self excluded — plain label propagation, one step.
        ww = w.clone()
        ww.fill_diagonal_(0.0)
        return (ww @ onehot).argmax(-1)

    # A cross-domain link is the interesting case, not the failure case. It looks like an error
    # to any same-class score, and it is exactly the kind of connection worth transferring:
    # the point of reaching outside a domain is that what is found there is *not* already in it.
    same = (y.unsqueeze(0) == y.unsqueeze(1))

    direct = predict(levels[0])
    base_ok = (direct == y)
    out = {"classes": len(classes),
           "direct_accuracy": round(float(base_ok.float().mean()), 4), "by_hop": []}
    for k in range(2, hops + 1):
        pk = predict(levels[k - 1])
        ok = (pk == y)
        rescued = int((ok & ~base_ok).sum())
        broken = int((~ok & base_ok).sum())
        new_links = (levels[k - 1] > 0) & (levels[k - 2] == 0)
        new_links.fill_diagonal_(False)
        cross = int((new_links & ~same).sum())
        within = int((new_links & same).sum())
        # Prediction using ONLY the cross-domain chain links: if transfer is real, a graph made
        # of nothing but links that leave the domain should still carry signal.
        cross_only = torch.where(new_links & ~same, levels[k - 1], torch.zeros(()))
        cross_pred = predict(cross_only + levels[0])
        out["by_hop"].append({
            "hops": k,
            "accuracy": round(float(ok.float().mean()), 4),
            "rescued": rescued,
            "broken": broken,
            "net": rescued - broken,
            "new_cross_domain": cross,
            "new_within_domain": within,
            "cross_share_pct": round(100.0 * cross / max(cross + within, 1), 1),
            "accuracy_with_cross_only": round(float((cross_pred == y).float().mean()), 4),
        })
    # The signal is real but swamped: ~35 items are rescued at every hop count, and 142 to
    # 1500 are broken. So do not admit chain links indiscriminately — admit the strong ones.
    # This is the relative adjustment applied to the chains themselves rather than only to the
    # base graph, and it is the difference between "transfer does not work" and "transfer was
    # never separated from noise".
    out["threshold_sweep"] = []
    for k in (2, 3):
        lv = levels[k - 1]
        new_links = (lv > 0) & (levels[k - 2] == 0)
        new_links.fill_diagonal_(False)
        if not bool(new_links.any()):
            continue
        vals = lv[new_links]
        for q in (0.5, 0.9, 0.99, 0.999):
            thr = float(torch.quantile(vals, q))
            kept = new_links & (lv >= thr)
            pred = predict(levels[0] + torch.where(kept, lv, torch.zeros(())))
            ok = pred == y
            out["threshold_sweep"].append({
                "hops": k,
                "quantile": q,
                "links_kept": int(kept.sum()),
                "cross_share_pct": round(100.0 * float((kept & ~same).sum())
                                         / max(int(kept.sum()), 1), 1),
                "accuracy": round(float(ok.float().mean()), 4),
                "rescued": int((ok & ~base_ok).sum()),
                "broken": int((~ok & base_ok).sum()),
                "net": int((ok & ~base_ok).sum()) - int((~ok & base_ok).sum()),
            })
    _ = mask
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
