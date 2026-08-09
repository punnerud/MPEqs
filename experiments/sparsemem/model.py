"""A product-key memory layer: keys are the index, values are the payload on disk.

The split is the whole point. Keys are small enough to stay resident and answer "which slots
does this query want"; values are large and live on disk, so only the retrieved ones are read.
That is the same shape as the MoE work in this repository — a router that selects, and expert
weights that must be fetched — except here the network is trained knowing that is the cost
model, rather than trained to spread across everything.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryNet(nn.Module):
    def __init__(self, in_dim, n_classes, n_slots=16384, key_dim=256, val_dim=1024, top_k=32):
        super().__init__()
        self.n_slots, self.key_dim, self.val_dim, self.top_k = n_slots, key_dim, val_dim, top_k
        self.encoder = nn.Sequential(nn.Linear(in_dim, key_dim), nn.GELU(),
                                     nn.Linear(key_dim, key_dim))
        # Keys and values initialised small: a large random memory dominates the signal early
        # and the encoder never learns to address it.
        self.keys = nn.Parameter(torch.randn(n_slots, key_dim) * (key_dim ** -0.5))
        self.values = nn.Parameter(torch.randn(n_slots, val_dim) * 0.02)
        self.head = nn.Linear(val_dim, n_classes)

    def route(self, x, k=None):
        """Returns (top-k slot indices, their softmax weights, the full logit distribution)."""
        k = k or self.top_k
        q = self.encoder(x)
        logits = q @ self.keys.t()                      # [B, n_slots]
        top_val, top_idx = logits.topk(k, dim=-1)
        w = F.softmax(top_val, dim=-1)
        return top_idx, w, logits

    def forward(self, x, k=None):
        top_idx, w, logits = self.route(x, k)
        vals = self.values[top_idx]                     # [B, k, val_dim]
        mixed = (vals * w.unsqueeze(-1)).sum(dim=1)
        return self.head(mixed), top_idx, logits


def balance_loss(logits):
    """Standard MoE load balancing: push the slot usage distribution towards uniform.

    This is the term production MoE models carry to stop the router collapsing. Here it is a
    switch, so its effect on how much of the layer must be fetched can be measured instead of
    inferred.
    """
    p = F.softmax(logits, dim=-1).mean(dim=0)
    n = p.numel()
    return (p * p).sum() * n                            # 1.0 when uniform, n when collapsed


def locality_loss(logits):
    """Force the slots an example uses to sit close together in the index.

    Phase 1 of this project permuted a *trained* model's experts so that co-selected ones
    became adjacent, and got 14.4 % fewer fetches — capped at 17 % of the clustering that was
    theoretically available, because the training had already decided which experts go
    together. This imposes the same objective at training time instead: minimise the variance
    of the routing distribution over slot *position*, so a query's mass lands in one contiguous
    stretch rather than scattered across the layer.

    Differentiable because it works on the full softmax over positions, not on the discrete
    top-k indices. Normalised by `n^2` so the weight means the same thing at any layer width.
    """
    p = F.softmax(logits, dim=-1)
    n = logits.shape[-1]
    pos = torch.arange(n, device=logits.device, dtype=p.dtype)
    mu = (p * pos).sum(dim=-1, keepdim=True)
    var = (p * (pos - mu) ** 2).sum(dim=-1)
    return (var / (n * n)).mean()


def contiguous_runs(top_idx):
    """How many separate reads the retrieved slots need. One run = one seek.

    The same quantity `coact`'s fetch planner counts for MoE experts, so the two halves of the
    project are measured on the same axis.
    """
    sorted_idx, _ = top_idx.sort(dim=-1)
    gaps = sorted_idx[:, 1:] - sorted_idx[:, :-1]
    return (1 + (gaps > 1).sum(dim=-1)).float()


def concentration_loss(logits):
    """The opposite: reward a peaked per-example distribution.

    Entropy of each example's own routing, not of the batch average — so it sharpens *which
    slots this input wants* without forcing every input to want the same slots.
    """
    logp = F.log_softmax(logits, dim=-1)
    return -(logp.exp() * logp).sum(dim=-1).mean()      # per-example entropy, minimised
