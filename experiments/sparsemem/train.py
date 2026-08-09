"""Train the same memory net under three auxiliary-loss regimes and measure the fetch frontier.

The point is the comparison, so the three runs share a seed, a step count and an optimiser.
Only the auxiliary term differs:

    plain         nothing
    balanced      load balancing, as production MoE models carry
    concentrated  an entropy penalty rewarding peaked per-example routing

The measurement is accuracy against how many slots have to be retrieved at inference. That is
the same question this repository asked of OLMoE and Qwen3.6 — how much of the expert layer
must come off disk — but here the training signal is a variable rather than a given.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from data import embeddings, split, synthetic  # noqa: E402
from model import (MemoryNet, balance_loss, concentration_loss,  # noqa: E402
                   contiguous_runs, locality_loss)

def regimes(bal, conc, loc):
    """The three arms, with the auxiliary weights exposed.

    The first run used 1e-2 for both. Load balancing behaved; the concentration penalty
    collapsed the router onto a handful of slots and accuracy fell to barely above chance —
    which is precisely the failure mode load balancing exists to prevent, so the weight is a
    knob to sweep rather than a constant to guess.
    """
    return {"plain": (0.0, 0.0, 0.0), "balanced": (bal, 0.0, 0.0),
            "concentrated": (0.0, conc, 0.0), "grouped": (0.0, 0.0, loc)}


def evaluate(model, x, y, ks, device):
    """Accuracy at each retrieval depth, and which slots got used."""
    model.eval()
    out = {}
    with torch.no_grad():
        for k in ks:
            correct, used = 0, torch.zeros(model.n_slots, device=device)
            runs = []
            for i in range(0, len(x), 512):
                xb, yb = x[i:i + 512].to(device), y[i:i + 512].to(device)
                logits, top_idx, _ = model(xb, k=k)
                correct += (logits.argmax(-1) == yb).sum().item()
                used.scatter_add_(0, top_idx.reshape(-1),
                                  torch.ones(top_idx.numel(), device=device))
                runs.append(contiguous_runs(top_idx).cpu())
            out[k] = {"accuracy": correct / len(x), "slot_use": used.cpu(),
                      "runs": float(torch.cat(runs).mean())}
    return out


def cache_curve(slot_use, n_slots, val_dim, budgets_mib=(1, 2, 4, 8, 16, 32, 64)):
    """Hit rate and bytes off disk for a pinned cache of each size.

    Values are the payload: `val_dim` floats per slot. Keys stay resident and act as the
    index, so they are not charged here — that split is the point of the architecture.
    """
    slot_bytes = val_dim * 4
    full_mib = n_slots * slot_bytes / 2**20
    ranked = slot_use.sort(descending=True).values.double()
    total = float(ranked.sum())
    out = []
    for mib in budgets_mib:
        pinned = min(n_slots, int(mib * 2**20 // slot_bytes))
        hit = float(ranked[:pinned].sum()) / max(total, 1.0)
        out.append({
            "budget_mib": mib,
            "resident_pct": round(100.0 * pinned / n_slots, 2),
            "hit_rate_pct": round(100.0 * hit, 2),
            "bytes_from_disk_per_retrieval": round((1 - hit) * slot_bytes, 1),
        })
    return {"layer_mib": round(full_mib, 1), "slot_bytes": slot_bytes, "budgets": out}


def gini(counts):
    """Inequality of slot usage. 0 = every slot used equally, 1 = one slot used for everything.

    The direct analogue of the gate-mass tables measured on OLMoE and Qwen3.6, where the
    distribution turned out nearly flat and every locality lever failed as a result.
    """
    v = counts.sort().values.double()
    n = v.numel()
    if v.sum() <= 0:
        return 0.0
    idx = torch.arange(1, n + 1, dtype=torch.double)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["synthetic", "embeddings"], default="synthetic")
    ap.add_argument("--slots", type=int, default=16384)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--val-dim", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--balance-weight", type=float, default=1e-2)
    ap.add_argument("--concentration-weight", type=float, default=1e-3)
    ap.add_argument("--locality-weight", type=float, default=1e-1)
    # Hold the model fixed and vary the data instead of tuning the loss weight. Compression is
    # what learning *is*: the question is how much data a fixed memory can absorb before its
    # footprint has to grow, not how hard a penalty can be cranked to force the footprint down.
    ap.add_argument("--train-subset", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated regimes to run")
    ap.add_argument("--out", default="data/sparsemem/frontier.json")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    x, y, n_classes = (synthetic() if args.task == "synthetic" else embeddings())
    xtr, ytr, xte, yte = split(x, y)
    if args.train_subset and args.train_subset < len(xtr):
        g0 = torch.Generator().manual_seed(args.seed)
        keep = torch.randperm(len(xtr), generator=g0)[:args.train_subset]
        xtr, ytr = xtr[keep], ytr[keep]
    print(f"task={args.task} device={device} train={len(xtr)} test={len(xte)} "
          f"classes={n_classes} dim={x.shape[1]}")

    ks = [1, 2, 4, 8, 16, 32]
    ks = [k for k in ks if k <= args.top_k]
    results = {}

    wanted = [r.strip() for r in args.only.split(",") if r.strip()]
    for regime, (w_bal, w_conc, w_loc) in regimes(args.balance_weight,
                                                  args.concentration_weight,
                                                  args.locality_weight).items():
        if wanted and regime not in wanted:
            continue
        torch.manual_seed(args.seed)
        model = MemoryNet(x.shape[1], n_classes, args.slots, 256, args.val_dim,
                          args.top_k).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        g = torch.Generator().manual_seed(args.seed)
        t0 = time.time()

        for step in range(args.steps):
            idx = torch.randint(0, len(xtr), (args.batch,), generator=g)
            xb, yb = xtr[idx].to(device), ytr[idx].to(device)
            logits, _, route_logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if w_bal:
                loss = loss + w_bal * balance_loss(route_logits)
            if w_conc:
                loss = loss + w_conc * concentration_loss(route_logits)
            if w_loc:
                loss = loss + w_loc * locality_loss(route_logits)
            opt.zero_grad()
            loss.backward()
            opt.step()

        ev = evaluate(model, xte, yte, ks, device)
        full = ev[max(ks)]
        results[regime] = {
            "train_seconds": round(time.time() - t0, 1),
            "accuracy_by_k": {str(k): round(ev[k]["accuracy"], 4) for k in ks},
            "gini_at_full_k": round(gini(full["slot_use"]), 4),
            "slots_touched_pct": round(
                100.0 * (full["slot_use"] > 0).sum().item() / args.slots, 2),
            # Share of retrievals that go to the busiest 1 % of slots. Flat = 1 %.
            "top1pct_share": round(float(
                full["slot_use"].sort(descending=True).values[:max(1, args.slots // 100)].sum()
                / full["slot_use"].sum()), 4),
            # Static pinning, exactly as measured on the MoE models: rank slots by how often
            # they are retrieved, pin as many as the budget holds, and read the rest from
            # disk. The hit rate is then a property of how concentrated training made the
            # usage — which is the whole question.
            "cache": cache_curve(full["slot_use"], args.slots, args.val_dim),
            # Reads per inference at full k: the same quantity coact counts for MoE experts.
            "runs_at_full_k": round(full["runs"], 2),
        }
        acc = results[regime]["accuracy_by_k"]
        print(f"{regime:<14} " + "  ".join(f"k={k}:{acc[str(k)]:.3f}" for k in ks) +
              f"   gini {results[regime]['gini_at_full_k']:.3f}"
              f"   touched {results[regime]['slots_touched_pct']:.1f}%"
              f"   top1% {results[regime]['top1pct_share']:.3f}"
              f"   runs {results[regime]['runs_at_full_k']:.1f}/{max(ks)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"task": args.task, "slots": args.slots, "val_dim": args.val_dim,
         "top_k": args.top_k, "steps": args.steps,
         "balance_weight": args.balance_weight,
         "concentration_weight": args.concentration_weight,
         "regimes": results}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
