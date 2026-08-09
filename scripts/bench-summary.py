#!/usr/bin/env python3
"""Summarise a llama-bench JSON run as median tok/s per test."""
import json
import statistics
import sys
from collections import defaultdict


def main() -> int:
    rows = []
    for path in sys.argv[1:]:
        with open(path) as f:
            rows.extend(json.load(f))
    if not rows:
        print("no rows", file=sys.stderr)
        return 1

    by_test = defaultdict(list)
    for r in rows:
        # llama-bench emits one row per repetition with avg_ts/stddev_ts already folded in
        # when -r is used, so prefer the per-row samples when they exist.
        name = f"{r.get('n_prompt', 0)}p+{r.get('n_gen', 0)}g"
        by_test[name].append(r)

    print(f"{'test':<12} {'tok/s':>10} {'stddev':>10} {'reps':>5}")
    for name, rs in by_test.items():
        ts = [r["avg_ts"] for r in rs if "avg_ts" in r]
        sd = [r.get("stddev_ts", 0.0) for r in rs]
        if not ts:
            continue
        print(
            f"{name:<12} {statistics.median(ts):>10.2f} "
            f"{max(sd):>10.2f} {rs[0].get('reps', len(rs)):>5}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
