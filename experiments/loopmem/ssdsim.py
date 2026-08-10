#!/usr/bin/env python3
"""What the token tables are worth in bytes and milliseconds, at streaming-from-SSD ratios.

The point of the prefetch tables was never a coverage percentage. It is a model that does
not fit in RAM: keep the weights on SSD, hold a small resident set, and pay for everything
else in fetches. At 600 GB of experts against 32 GB of RAM the residency is five percent,
not the twenty-five the coverage work used, and the interesting question is what a
token-derived table buys down there — in MiB per token and milliseconds per token, using
this machine's measured cost model rather than a coverage ratio.

    c_fetch = 6,662 ns per fetch, c_byte = 0.0781 ns per byte   (data/cache.json)

Three policies over the same trace, at residencies from 50% down to 2%:

    STATIC PINNING   hold the globally most frequent experts. The README's winner, and
                     the thing to beat.
    TOKEN PREFETCH   hold the pinned set, and for each arriving token also fetch what its
                     table says it will want. Available at time zero for the whole prompt,
                     so these fetches overlap compute instead of stalling it.
    ORACLE           hold exactly what the next token needs. Not achievable, but it says
                     how much of the gap any predictor could ever close.

The simulator is validated first against fetchbench's own static-pinning rows: if it
cannot reproduce a measured hit rate it has no business estimating a new one.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tokenprefetch import read_trace, tokenise_corpus  # noqa: E402

TRACE = Path("data/trace-big.bin")
COST = json.loads(Path("data/cache.json").read_text())["cost_model"]
EXPERT_BYTES_TOTAL = json.loads(Path("data/cache.json").read_text())[
    "expert_bytes_total"]


def simulate(recs, toks, resident_frac, policy, tables, freq_rank, n_expert,
             expert_bytes, prefetch_k=8):
    """One pass over the trace: what is resident, what must be fetched, how much.

    The resident set is per layer and fixed for the pinned part; the token policy adds a
    per-token prefetch on top, which is charged as a fetch but marked overlapped because
    it is issued when the token arrives rather than when the layer needs it.
    """
    per_layer = max(1, int(round(resident_frac * n_expert)))
    pinned = {la: set(r[:per_layer]) for la, r in freq_rank.items()}
    fetch = overlapped = stalled = 0
    by_pos = defaultdict(dict)
    for layer, pos, ids in recs:
        by_pos[pos][layer] = ids

    for pos in sorted(by_pos):
        pre = {}
        if policy == "token":
            tok = toks[pos]
            for layer in by_pos[pos]:
                tab = tables.get((layer, tok))
                if tab:
                    want = {e for e, _c in tab.most_common(prefetch_k)}
                    pre[layer] = want - pinned[layer]
        elif policy == "oracle":
            for layer, ids in by_pos[pos].items():
                pre[layer] = set(ids) - pinned[layer]
        for layer, ids in by_pos[pos].items():
            need = set(ids)
            resident = pinned[layer]
            prefetched = pre.get(layer, set())
            fetch += len(prefetched)
            overlapped += len(need & prefetched)
            miss = need - resident - prefetched
            stalled += len(miss)
            fetch += len(miss)
    n_tok = len(by_pos)
    return {"fetches_per_token": fetch / n_tok,
            "stalled_per_token": stalled / n_tok,
            "overlapped_per_token": overlapped / n_tok,
            "mib_per_token": fetch * expert_bytes / n_tok / 1048576,
            "stall_ms_per_token": stalled * (COST["c_fetch_ns"]
                                             + expert_bytes * COST["c_byte_ns"])
            / n_tok / 1e6}


def main(out="data/custom/ssdsim.json"):
    meta, recs = read_trace(TRACE)
    toks = tokenise_corpus(max(t for _l, t, _i in recs) + 1)
    usable = min(len(toks), max(t for _l, t, _i in recs) + 1)
    recs = [(la, p, i) for la, p, i in recs if p < usable]
    n_expert = meta["experts"]
    n_layer = meta["layers"]
    expert_bytes = EXPERT_BYTES_TOTAL / (n_expert * n_layer)
    print(f"{n_layer} layers x {n_expert} experts, "
          f"{expert_bytes / 1048576:.1f} MiB each, "
          f"{EXPERT_BYTES_TOTAL / 1e9:.2f} GB of experts total")

    # A CONTIGUOUS split, train first half and test the distant second. The
    # interleaved split used earlier leaks locality — adjacent blocks are neighbouring
    # text — and inflated static pinning from 18% to 37% at the same budget, which is
    # most of the distance between this simulator and fetchbench's measured 13.4%.
    train_set = set(range(usable // 2))
    tables, freq = defaultdict(Counter), defaultdict(Counter)
    for layer, pos, ids in recs:
        if pos in train_set:
            tables[(layer, toks[pos])].update(ids)
            freq[layer].update(ids)
    freq_rank = {la: [e for e, _c in c.most_common()] for la, c in freq.items()}
    test = [(la, p, i) for la, p, i in recs if p not in train_set]

    # Validation must use a curve measured on the CORRECTED trace. The archived
    # cache-curve was replayed against data/trace.bin, which the README suspends as an
    # artefact — and its signature is visible in the numbers: hit rate tracking
    # residency almost exactly is what uniform-by-construction access looks like.
    # data/custom/fetch-corrected.json is fetchbench replayed on trace-big.bin.
    curve = json.loads(Path("data/custom/fetch-corrected256.json").read_text())["results"]
    known = [r for r in curve if r["cache_mib"] > 0]
    print("\nvalidation against fetchbench replayed on the CORRECTED trace:")
    val_rows = []
    for r in known[:4]:
        frac = (r["cache_mib"] * 1048576) / EXPERT_BYTES_TOTAL
        sim = simulate(test, toks, frac, "pinned", tables, freq_rank, n_expert,
                       expert_bytes)
        hit = 1 - sim["stalled_per_token"] / (n_layer * meta["top_k"])
        val_rows.append({"cache_mib": r["cache_mib"], "measured": r["hit_rate_pct"],
                         "simulated": round(hit * 100, 1)})
        print(f"  {r['cache_mib']:>5} MiB: measured {r['hit_rate_pct']:>5.1f}%   "
              f"simulated {hit * 100:>5.1f}%")

    print(f"\n{'residency':>10}{'policy':>10}{'stalls/tok':>12}{'MiB/tok':>10}"
          f"{'stall ms/tok':>14}")
    rows = []
    for frac in (0.50, 0.25, 0.10, 0.053, 0.02):
        for policy in ("pinned", "token", "oracle"):
            s = simulate(test, toks, frac, policy, tables, freq_rank, n_expert,
                         expert_bytes)
            rows.append({"residency": frac, "policy": policy, **s})
            tag = "  <- 600GB on 32GB" if abs(frac - 0.053) < 1e-9 and \
                policy == "token" else ""
            print(f"{frac:>10.1%}{policy:>10}{s['stalled_per_token']:>12.1f}"
                  f"{s['mib_per_token']:>10.1f}{s['stall_ms_per_token']:>14.1f}{tag}")

    at5 = {r["policy"]: r for r in rows if abs(r["residency"] - 0.053) < 1e-9}
    gain = 1 - at5["token"]["stall_ms_per_token"] / at5["pinned"]["stall_ms_per_token"]
    print(f"\nat the 600GB-on-32GB ratio the token table removes "
          f"{gain:.0%} of the stall time, and moves "
          f"{at5['token']['overlapped_per_token']:.1f} fetches per token off the "
          f"critical path")
    print("Every prefetched byte is still read from SSD — the saving is not in traffic")
    print("but in WHEN it is issued, which is the only thing a lookup available at time")
    print("zero can buy.")
    summary = {"expert_mib": round(expert_bytes / 1048576, 2),
               "validation": val_rows, "rows": rows,
               "stall_reduction_at_5pct": round(gain, 4)}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
