#!/usr/bin/env python3
"""Compare two logits-dump trees produced by scripts/logits.sh.

Reports, per prompt: how many logits differ, the largest deviation relative to the peak
logit, and — the part that decides pass or fail — whether the argmax moved.

A permutation of the expert axis is exact in exact arithmetic. In float32 it is not, because
llama.cpp's router normalises with a softmax over all experts in storage order, so reordering
them perturbs every gate weight in the last bits. The right question is therefore not "are
the bytes identical" but "did the model's decisions change".
"""
import struct
import sys
from pathlib import Path


def load(d: Path):
    cands = [x for x in d.glob("*.bin") if "tokens" not in x.name]
    if not cands:
        return None
    raw = cands[0].read_bytes()
    return struct.unpack(f"<{len(raw) // 4}f", raw)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: logit-diff.py <baseline-dir> <current-dir>", file=sys.stderr)
        return 2
    base, cur = Path(sys.argv[1]), Path(sys.argv[2])

    print(f"{'prompt':>8} {'differing':>12} {'max rel dev':>13} {'argmax':>8}")
    worst = 0.0
    argmax_moved = 0
    n = 0
    for d in sorted(base.glob("p*")):
        a = load(d)
        b = load(cur / d.name)
        if a is None or b is None:
            print(f"{d.name:>8} {'MISSING':>12}")
            return 1
        n += 1
        scale = max(abs(x) for x in a) or 1.0
        rel = max(abs(x - y) for x, y in zip(a, b)) / scale
        worst = max(worst, rel)
        ndiff = sum(1 for x, y in zip(a, b) if x != y)
        same = a.index(max(a)) == b.index(max(b))
        argmax_moved += not same
        print(f"{d.name:>8} {ndiff:>7}/{len(a):<4} {rel:>13.3e} {'same' if same else 'MOVED':>8}")

    print(f"\n{n} prompts, worst relative deviation {worst:.3e}, argmax moved on {argmax_moved}")
    return 1 if argmax_moved else 0


if __name__ == "__main__":
    sys.exit(main())
