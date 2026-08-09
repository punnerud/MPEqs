"""Datasets for the sparse-memory experiment."""
import struct
from pathlib import Path

import torch


def synthetic(n=6000, in_dim=384, n_classes=8, n_factors=64, per_example=8,
              noise=0.25, seed=0):
    """A task whose answer requires *combining* several latent factors.

    Each example is a sum of `per_example` factors drawn from `n_factors`, and the label is
    decided by the summed contribution of exactly those factors. One factor is not enough: a
    memory layer solving this must retrieve several slots and mix them, so the
    accuracy-versus-k frontier has a known shape — it should saturate around `per_example` and
    fall off below it.

    The first attempt used one prototype per example, which every regime solved perfectly at
    k=1 and made the frontier flat and uninformative. A frontier only measures anything if
    small k genuinely destroys information.
    """
    g = torch.Generator().manual_seed(seed)
    directions = torch.randn(n_factors, in_dim, generator=g)
    contributions = torch.randn(n_factors, n_classes, generator=g)

    x = torch.zeros(n, in_dim)
    y = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        sel = torch.randperm(n_factors, generator=g)[:per_example]
        x[i] = directions[sel].sum(0)
        y[i] = contributions[sel].sum(0).argmax()
    x = x + noise * torch.randn(n, in_dim, generator=g)
    x = torch.nn.functional.normalize(x, dim=-1)
    return x, y, n_classes


def embeddings(path="data/labelled", min_class=130, seed=2):
    """The corpus registers, embedded one class per call, then balanced by capping.

    Classes smaller than `min_class` are dropped rather than up-weighted: Norwegian ends up at
    380 chunks against wikitext's 3200 because no.wikipedia's extracts API returns only a
    handful of the requested articles, and a class at 5 % of the data makes an accuracy curve
    meaningless — a model can ignore it entirely and still look good. The rest are capped to
    the smallest survivor so the task is genuinely balanced.
    """
    p = Path(path)
    raw = (p / "emb.f32").read_bytes()
    labels = [int(v) for v in (p / "labels.txt").read_text().split()]
    dim = len(raw) // 4 // len(labels)
    x = torch.tensor(struct.unpack(f"<{len(raw)//4}f", raw)).view(len(labels), dim)
    y = torch.tensor(labels)

    counts = torch.bincount(y)
    keep_classes = [c for c in range(len(counts)) if counts[c] >= min_class]
    cap = min(int(counts[c]) for c in keep_classes)
    g = torch.Generator().manual_seed(seed)
    idx = []
    for new_lab, c in enumerate(keep_classes):
        members = (y == c).nonzero(as_tuple=True)[0]
        pick = members[torch.randperm(len(members), generator=g)[:cap]]
        idx.append((pick, new_lab))
    xs = torch.cat([x[pick] for pick, _ in idx])
    ys = torch.cat([torch.full((len(pick),), lab, dtype=torch.long) for pick, lab in idx])
    dropped = [c for c in range(len(counts)) if c not in keep_classes]
    if dropped:
        print(f"  dropped classes {dropped} (under {min_class} chunks); "
              f"capped the rest to {cap}")
    return xs, ys, len(keep_classes)


def split(x, y, frac=0.2, seed=1):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x), generator=g)
    cut = int(len(x) * (1 - frac))
    tr, te = perm[:cut], perm[cut:]
    return x[tr], y[tr], x[te], y[te]
