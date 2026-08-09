#!/usr/bin/env python3
"""Each pass decides the next one by walking the co-activation graph, not by re-scoring all.

`arith_loop.py` loops the same block T times, and every hop re-runs a dense router over all 32
experts. That is a loop, but not the one the architecture is for: the next pass is chosen by
recomputing everything rather than by following where the last pass landed.

Here hop 1 is dense — it has nothing to walk from — and every later hop scores **only the graph
neighbours of the experts the previous hop selected**. The graph is built online from
co-activation during training: pairs that fire together get an edge, which is the same
`fire together, wire together` rule the layout solver uses, applied as a routing structure
instead of a disk order.

Two things are then measurable that a dense router hides:

  experts scored per hop  — dense is always 32; the walk is the neighbourhood size, and this is
                            what makes the index cheap when the expert count grows
  reachability            — a walk can only reach what the graph connects, so accuracy is the
                            check on whether the graph learned enough edges

The interesting failure mode is a graph that connects everything, which costs as much as the
dense router and proves nothing. `avg_degree` is reported for exactly that reason.

Run with a torch venv, e.g. /Users/punnerud/Downloads/ainmt/venv/bin/python3.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from arith_loop import (BLOCK, D, N_BLOCK, N_CLASS, N_EXPERT, TOP_K,  # noqa: E402
                        fetch_stats, problems)


class GraphLoopedMoE(nn.Module):
    def __init__(self, hops, degree=8, graph_routing=True):
        super().__init__()
        self.hops, self.degree, self.graph_routing = hops, degree, graph_routing
        self.inp = nn.Linear(D, D)
        self.router = nn.Linear(D, N_EXPERT, bias=False)
        self.experts = nn.Parameter(torch.randn(N_EXPERT, D, D) * (1.0 / D**0.5))
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, N_CLASS)
        # Co-activation counts, accumulated during training. Not a parameter: it is measured
        # from behaviour, the same way the layout solver measures it from a trace.
        self.register_buffer("coact", torch.zeros(N_EXPERT, N_EXPERT))

    def neighbours(self, idx):
        """Candidate mask for the next hop: the graph neighbours of this hop's picks.

        The previously selected experts stay in the candidate set. Dropping them would forbid
        the walk from standing still, and standing still is a legitimate answer — it is what
        makes a repeat fetch free.
        """
        b = idx.shape[0]
        mask = torch.zeros(b, N_EXPERT, device=idx.device, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        deg = min(self.degree, N_EXPERT - 1)
        top = self.coact.topk(deg, dim=-1).indices           # [N_EXPERT, deg]
        nbr = top[idx].reshape(b, -1)                        # [b, top_k * deg]
        mask.scatter_(1, nbr, True)
        return mask

    def forward(self, x, learn_graph=False):
        h = self.inp(x)
        picks, scored = [], []
        mask = None
        for hop in range(self.hops):
            logits = self.router(h)
            if mask is not None:
                logits = logits.masked_fill(~mask, float("-inf"))
                scored.append(float(mask.sum(-1).float().mean()))
            else:
                scored.append(float(N_EXPERT))
            probs = logits.softmax(-1)
            w, idx = probs.topk(TOP_K, dim=-1)
            y = torch.einsum("bkij,bj->bki", self.experts[idx], h)
            h = self.norm(h + (y * w.unsqueeze(-1)).sum(1))
            picks.append(idx)
            if learn_graph:
                with torch.no_grad():
                    oh = torch.zeros(idx.shape[0], N_EXPERT, device=idx.device)
                    oh.scatter_(1, idx, 1.0)
                    self.coact += oh.t() @ oh
                    self.coact.fill_diagonal_(0)
            if self.graph_routing and hop + 1 < self.hops:
                mask = self.neighbours(idx)
        return self.head(h), torch.stack(picks, 1), scored


def compression_loss(probs):
    m = probs.view(*probs.shape[:-1], N_BLOCK, BLOCK).sum(-1).clamp(1e-6, 1.0)
    return (1.0 - (1.0 - m).pow(TOP_K)).sum(-1).mean()


def run(hops, graph, lam, steps, x, y, degree=8):
    torch.manual_seed(0)
    m = GraphLoopedMoE(hops, degree=degree, graph_routing=graph)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for s in range(steps):
        i = torch.randint(0, len(x), (128,))
        # The graph is meaningless for the first stretch, so let dense routing build it before
        # the walk relies on it. Routing through an empty graph would just pick expert 0.
        warm = s < steps // 5
        m.graph_routing = graph and not warm
        logit, _, _ = m(x[i], learn_graph=True)
        loss = F.cross_entropy(logit, y[i])
        if lam:
            loss = loss + lam * compression_loss(m.router(m.inp(x[i])).softmax(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.graph_routing = graph
    with torch.no_grad():
        logit, picks, scored = m(x)
        acc = (logit.argmax(-1) == y).float().mean().item()
    total, distinct = fetch_stats(picks)
    edges = (m.coact > 0).float()
    return {"hops": hops, "graph_routing": graph, "lambda": lam,
            "exact_accuracy": round(acc, 4),
            "experts_scored_per_hop": [round(v, 1) for v in scored],
            "mean_experts_scored": round(sum(scored) / len(scored), 1),
            "avg_graph_degree": round(float(edges.sum(-1).mean()), 1),
            "block_fetches_per_problem": round(total, 3),
            "distinct_blocks_per_problem": round(distinct, 3)}


def main(steps=2500, lam=0.3, out="data/custom/graphloop.json"):
    steps, lam = int(steps), float(lam)
    x, y = problems()
    print(f"{len(x)} arithmetic problems, {N_EXPERT} experts, top-{TOP_K}\n")
    print(f"{'hops':>5} {'routing':>9} {'compress':>9} {'exact acc':>10} "
          f"{'scored/hop':>11} {'degree':>7} {'distinct blk':>13}")
    rows = []
    for hops in (2, 4):
        for graph in (False, True):
            for lm in (0.0, lam):
                r = run(hops, graph, lm, steps, x, y)
                rows.append(r)
                print(f"{hops:>5} {('graph' if graph else 'dense'):>9} "
                      f"{('yes' if lm else 'no'):>9} {r['exact_accuracy']:>10.3f} "
                      f"{r['mean_experts_scored']:>11.1f} {r['avg_graph_degree']:>7.1f} "
                      f"{r['distinct_blocks_per_problem']:>13.2f}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"n_expert": N_EXPERT, "steps": steps, "rows": rows},
                                    indent=2))
    print(f"\nwrote {out}")
    print("`scored/hop` is the cost that matters as the expert count grows: a dense router is\n"
          "always N, a walk is the neighbourhood. Accuracy is the check that the walk can\n"
          "still reach what it needs.")


if __name__ == "__main__":
    main(*sys.argv[1:])
