#!/usr/bin/env python3
"""The point was never the bytes. It was a search that is fast and right with the memory short.

This is the original question the whole thread came from: hold a small index resident, keep the
rest on disk, and fetch only what a query actually needs. Everything built along the way now
exists to answer it — clusters that route perfectly (phase 37), a 3-bit encoding that retrieves
perfectly (phase 37), and a cost model calibrated on this machine with real uncached reads
(C_fetch 227.8 us, C_byte 0.276 ns/B, `data/costmodel.json`, provenance measured).

The layout under test:

    RESIDENT   the C centroids, and nothing else. This is the memory budget.
    ON DISK    per cluster, its members' vectors at 3 bits per dimension, contiguous, so a probe
               is one seek and one contiguous read.

A query compares against the resident centroids, fetches the probed cluster's block, and scans
inside it. Predicted time is `fetches x C_fetch + bytes x C_byte`, which is the same model the
fetch work validated against measured uncached reads to within its stated tolerance.

Two baselines it has to be judged against, because a fast wrong answer is worthless and a
correct answer that needs everything in memory is not an index:

    everything resident at 3 bits   730 KB in memory, no fetches, 40/40
    everything resident as f32      13.7 MB in memory, no fetches, 40/40
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from clustercodec import kmeans  # noqa: E402
from embednav import CACHE, embed  # noqa: E402
from longdoc import DOC, sentences  # noqa: E402

COST = json.loads(Path("data/costmodel.json").read_text())
C_FETCH_NS, C_BYTE_NS = COST["c_fetch_ns"], COST["c_byte_ns"]
BITS = 3


def quantised_bytes(count, dim, bits=BITS):
    return (count * dim * bits + 7) // 8


def main(seed=3, n_queries=40, out="data/custom/diskindex.json"):
    seed, n_queries = int(seed), int(n_queries)
    sents = sentences(Path(DOC).read_text())
    X = np.array(embed(sents, CACHE), dtype=np.float32)
    n, dim = X.shape
    rng = np.random.default_rng(seed)
    idx = [i for i, s in enumerate(sents) if 60 < len(s) < 220]
    picks = [int(i) for i in rng.choice(idx, size=n_queries, replace=False)]
    qv = np.array(embed([sents[i] for i in picks]), dtype=np.float32)

    # The stored form: 3 bits per dimension, one global scale. Phase 37 measured this at 40/40.
    lim = float(np.abs(X).max())
    step = lim / (2 ** (BITS - 1) - 1)
    Xq = (np.clip(np.rint(X / step), -(2 ** (BITS - 1)), 2 ** (BITS - 1) - 1)
          * step).astype(np.float32)

    flat_resident = quantised_bytes(n, dim)
    flat_hits = sum(int(np.argmax(Xq @ qv[k])) == i for k, i in enumerate(picks))
    print(f"{n:,} sentences, {dim} dims, {BITS} bits stored\n")
    print(f"everything resident: {flat_resident / 1e6:.2f} MB, 0 fetches, "
          f"{flat_hits}/{len(picks)} correct, {n} comparisons\n")

    print(f"{'C':>6}{'probe':>7}{'resident':>11}{'fetches':>9}{'read KB':>10}"
          f"{'us/query':>10}{'correct':>9}")
    rows = []
    for C in (94, 256, 1024):
        assign, cent = kmeans(X, C, seed=seed)
        members = [np.where(assign == j)[0] for j in range(C)]
        resident = cent.nbytes                       # the whole memory budget
        for probe in (1, 2, 4):
            hits = fetch_n = read_b = 0
            for k, target in enumerate(picks):
                q = qv[k]
                top = np.argsort(cent @ q)[-probe:]
                pool = np.concatenate([members[j] for j in top])
                # One contiguous read per probed cluster.
                fetch_n += probe
                read_b += sum(quantised_bytes(len(members[j]), dim) for j in top)
                if len(pool):
                    hits += int(pool[np.argmax(Xq[pool] @ q)]) == target
            us = (fetch_n / len(picks) * C_FETCH_NS
                  + read_b / len(picks) * C_BYTE_NS) / 1000
            rows.append({"C": C, "probe": probe, "resident_bytes": int(resident),
                         "fetches": fetch_n / len(picks), "read_bytes": read_b / len(picks),
                         "us_per_query": us, "correct": hits})
            print(f"{C:>6}{probe:>7}{resident / 1e6:>9.2f}M{probe:>9}"
                  f"{read_b / len(picks) / 1e3:>10.1f}{us:>10.1f}{hits:>7}/{len(picks)}")

    # What a full scan would cost if it also had to come off disk.
    full_us = (C_FETCH_NS + flat_resident * C_BYTE_NS) / 1000
    print(f"\nreading the whole 3-bit block from disk instead: {flat_resident / 1e3:.0f} KB, "
          f"{full_us:.0f} us")

    ok = [r for r in rows if r["correct"] >= flat_hits]
    best = min(ok, key=lambda r: r["resident_bytes"]) if ok else None
    if best:
        print(f"smallest resident index that loses nothing: C={best['C']}, probe="
              f"{best['probe']}, {best['resident_bytes'] / 1e6:.2f} MB resident, "
              f"{best['read_bytes'] / 1e3:.1f} KB read, {best['us_per_query']:.1f} us")
        print(f"against {flat_resident / 1e6:.2f} MB to hold everything — "
              f"{flat_resident / best['resident_bytes']:.2f}x the memory, "
              f"{full_us / best['us_per_query']:.1f}x the time off a cold cache")
    print("\nThe index is the centroids and they are the only thing that has to stay resident.")
    print("Everything else is a contiguous read of the one cluster the query turned out to need.")

    summary = {"n": n, "dim": dim, "bits": BITS, "queries": len(picks),
               "flat_resident_bytes": flat_resident, "flat_correct": flat_hits,
               "full_scan_us": full_us, "c_fetch_ns": C_FETCH_NS, "c_byte_ns": C_BYTE_NS,
               "rows": rows, "best": best}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
