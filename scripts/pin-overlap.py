#!/usr/bin/env python3
"""How much do two registers agree on which experts are hot?

`fetchbench cache --pin-trace` showed that pinning on English prose and replaying Norwegian
gives the same hit rate as pinning on Norwegian itself. That has two possible explanations and
they mean opposite things:

  1. the hot set is the same in both registers  -> pinning transfers because routing is stable
  2. the hot set differs but does not matter    -> pinning transfers because nothing is hot

This decides between them by comparing the top-N sets directly. Reads the `MOET` trace format
from moetrace: 32-byte header, then `layer u16 | pad u16 | token u32 | expert u16 * top_k`,
with gate weights appended at version >= 2 and output norms at version >= 3.
"""
import struct
import sys
from collections import Counter


def counts(path):
    raw = open(path, "rb").read()
    magic, version, n_layer, n_expert, top_k, _, _ = struct.unpack_from("<6IQ", raw, 0)
    assert magic == 0x5445_4F4D, f"{path} is not a MOET trace"
    rec = 8 + 2 * top_k + (4 * top_k if version >= 2 else 0) + (4 * top_k if version >= 3 else 0)
    per_layer = {}
    for off in range(32, len(raw) - rec + 1, rec):
        layer = struct.unpack_from("<H", raw, off)[0]
        ids = struct.unpack_from(f"<{top_k}H", raw, off + 8)
        per_layer.setdefault(layer, Counter()).update(ids)
    return per_layer, n_layer, n_expert, top_k


def main(*traces):
    traces = traces or ("data/trace-wikitext.bin", "data/trace-norwegian.bin",
                        "data/trace-code-python.bin")
    loaded = {t.split("trace-")[-1].removesuffix(".bin"): counts(t) for t in traces}
    names = list(loaded)
    _, n_layer, n_expert, top_k = loaded[names[0]]
    print(f"{n_layer} layers, {n_expert} experts, top-{top_k}\n")

    # How concentrated is each register on its own? If the top half of the experts carries
    # barely more than half the traffic, there is no hot set to transfer in the first place.
    print(f"{'register':>14} {'share in hottest half':>22} {'gini':>7}")
    for name in names:
        pl = loaded[name][0]
        tot = half = 0
        ginis = []
        for c in pl.values():
            v = sorted(c.values(), reverse=True)
            tot += sum(v)
            half += sum(v[: len(v) // 2])
            n = len(v)
            asc = sorted(v)
            s = sum(asc)
            ginis.append(sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(asc)) / (n * s))
        print(f"{name:>14} {100.0 * half / tot:>21.1f}% {sum(ginis) / len(ginis):>7.3f}")

    print(f"\ntop-half overlap between registers, mean over layers")
    print(f"{'':>14} " + " ".join(f"{n:>14}" for n in names))
    for a in names:
        row = []
        for b in names:
            ov = []
            for layer, ca in loaded[a][0].items():
                cb = loaded[b][0][layer]
                k = max(1, n_expert // 2)
                sa = {e for e, _ in ca.most_common(k)}
                sb = {e for e, _ in cb.most_common(k)}
                ov.append(len(sa & sb) / k)
            row.append(f"{100.0 * sum(ov) / len(ov):>13.1f}%")
        print(f"{a:>14} " + " ".join(row))

    # Chance level: two independent random halves of n_expert overlap by half.
    print(f"\nchance overlap for two random halves: 50.0%")


if __name__ == "__main__":
    main(*sys.argv[1:])
