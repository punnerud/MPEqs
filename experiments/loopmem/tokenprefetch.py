#!/usr/bin/env python3
"""Does the TOKEN predict its experts? Prefetching at lookup time, before any compute.

If the word-to-vector step is a table rather than a forward pass, then at time zero — with
no layer having run — you already hold every token of the prompt, in parallel. The systems
consequence is the one this repository's other half exists for: a model that does not fit
in RAM spends its first tokens waiting for experts to arrive from disk, and anything
knowable before the compute starts can start the fetch earlier.

The headline table in the README tested prefetch ONE way: does layer L predict layer L+1
(0.66 of 5.96 bits — dead). It never asked whether the TOKEN predicts the experts, which
is the only predictor available at lookup time and the only one that parallelises across
the whole prompt.

Measured here on the CORRECTED contribution trace (not the suspended artefact): 16 layers,
64 experts, top-8, 8,192 token positions, aligned against the same corpus tokenised with
the same model. The table is built on the first half of the stream and tested on the
second, so nothing is scored on what it memorised.

    TOKEN     for a test position, prefetch the B experts this token most often used
              at this layer; unseen tokens fall back to the layer's global favourites
    FREQUENCY the layer's global top-B, which is the static pinning the README already
              measured as the best-performing cache policy
    ORACLE    the eight actually selected, an upper bound at B = 8

The number that matters is coverage: of the eight experts a position actually needs, how
many were already on their way?
"""
import json
import struct
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

TRACE = Path("data/trace-contrib.bin")
CORPUS = Path("data/corpus/corpus.txt")
MODEL = Path("models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf")


def read_trace(path):
    raw = path.read_bytes()
    _magic, ver, n_layer, n_expert, top_k, _pad = struct.unpack_from("<6I", raw, 0)
    (n_rec,) = struct.unpack_from("<Q", raw, 24)
    body = len(raw) - 32
    rec_bytes = body // n_rec
    out = []
    off = 32
    for _ in range(n_rec):
        layer, _p, tok = struct.unpack_from("<HHI", raw, off)
        ids = struct.unpack_from(f"<{top_k}H", raw, off + 8)
        out.append((layer, tok, ids))
        off += rec_bytes
    return {"version": ver, "layers": n_layer, "experts": n_expert, "top_k": top_k,
            "records": n_rec, "record_bytes": rec_bytes}, out


def tokenise_corpus(n_needed):
    """The same corpus through the same model's tokeniser, so position i is token i."""
    out = subprocess.run(
        ["llama-tokenize", "-m", str(MODEL), "-f", str(CORPUS), "--ids"],
        capture_output=True, text=True).stdout
    m = out.rfind("[")
    if m < 0:
        return []
    try:
        ids = json.loads(out[m:out.rfind("]") + 1])
    except json.JSONDecodeError:
        return []
    return ids[:n_needed]


def main(budget=16, out="data/custom/tokenprefetch.json"):
    budget = int(budget)
    if not TRACE.exists():
        print(f"no trace at {TRACE}")
        return
    meta, recs = read_trace(TRACE)
    positions = sorted({t for _l, t, _i in recs})
    print(f"trace: {meta['layers']} layers, {meta['experts']} experts, top-"
          f"{meta['top_k']}, {meta['records']} records over {len(positions)} positions "
          f"({meta['record_bytes']} bytes each)")

    toks = tokenise_corpus(max(positions) + 1)
    if len(toks) < max(positions) + 1:
        print(f"tokeniser returned {len(toks)} ids for {max(positions) + 1} positions; "
              f"aligning on the overlap")
    usable = min(len(toks), max(positions) + 1)
    split = usable // 2
    print(f"{usable} aligned positions, table built on the first {split}, "
          f"tested on the rest\n")

    by_layer_token = defaultdict(Counter)      # (layer, token) -> experts
    by_layer = defaultdict(Counter)            # layer -> experts
    test = []
    for layer, pos, ids in recs:
        if pos >= usable:
            continue
        if pos < split:
            v = toks[pos]
            for e in ids:
                by_layer_token[(layer, v)][e] += 1
                by_layer[layer][e] += 1
        else:
            test.append((layer, pos, ids))

    tok_hit = freq_hit = total = unseen = 0
    per_layer = defaultdict(lambda: [0, 0, 0])
    for layer, pos, ids in test:
        v = toks[pos]
        table = by_layer_token.get((layer, v))
        if table:
            pre = {e for e, _c in table.most_common(budget)}
            # A token seen only a few times names fewer than the budget allows; the
            # spare slots go to the layer's global favourites rather than idle.
            for e, _c in by_layer[layer].most_common():
                if len(pre) >= budget:
                    break
                pre.add(e)
        else:
            unseen += 1
            pre = {e for e, _c in by_layer[layer].most_common(budget)}
        base = {e for e, _c in by_layer[layer].most_common(budget)}
        t_hit = len(set(ids) & pre)
        f_hit = len(set(ids) & base)
        tok_hit += t_hit
        freq_hit += f_hit
        total += len(ids)
        per_layer[layer][0] += t_hit
        per_layer[layer][1] += f_hit
        per_layer[layer][2] += len(ids)

    print(f"prefetch budget {budget} experts of {meta['experts']} per layer "
          f"({budget / meta['experts']:.0%} residency)\n")
    print(f"{'layer':>6}{'token-conditioned':>20}{'frequency only':>17}")
    for layer in sorted(per_layer):
        t, f, n = per_layer[layer]
        print(f"{layer:>6}{t / n:>19.1%}{f / n:>17.1%}")
    print(f"\n{'ALL':>6}{tok_hit / total:>19.1%}{freq_hit / total:>17.1%}")
    print(f"positions whose token was never seen in the first half: {unseen} of "
          f"{len(test)} records")
    delta = (tok_hit - freq_hit) / total
    print(f"\ntoken knowledge is worth {delta:+.1%} coverage at this budget, and it is "
          f"available before any layer has run")
    print("The README measured prefetch across LAYERS and found 0.66 of 5.96 bits. This")
    print("is the other axis: what the identity of the word already tells you, in")
    print("parallel for the whole prompt, at the moment it is looked up.")
    summary = {"budget": budget, **meta, "aligned_positions": usable,
               "token_coverage": round(tok_hit / total, 4),
               "frequency_coverage": round(freq_hit / total, 4),
               "delta": round(delta, 4), "unseen_records": unseen,
               "test_records": len(test),
               "per_layer": {str(k): v for k, v in sorted(per_layer.items())}}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
