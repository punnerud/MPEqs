#!/usr/bin/env bash
# The many-small-experts case, which is where this technique should pay best.
#
# usage: many-experts-study.sh <model.gguf> [trace tokens]
#
# `coact project` ranks candidate models by the share of per-token cost that is per-request
# overhead. Qwen3.6-35B-A3B tops that list at 70 %, against OLMoE's 39 %, because 40 layers of
# 256 experts at ~110 KB each means request count dominates and bytes do not. This script runs
# the parts of the pipeline that answer the one open question: how much of the theoretically
# available clustering such a model actually has.
#
# Settings are tuned for a large expert count. The local search is quadratic in it, so it is
# disabled here — on OLMoE it overfits anyway, losing to the `chain` construction on holdout.
set -euo pipefail

MODEL="${1:?usage: many-experts-study.sh <model.gguf> [tokens]}"
TOKENS="${2:-20000}"
TAG=$(basename "$MODEL" .gguf)
BIN=./target/release

echo "== geometry =="
$BIN/coact stats --model "$MODEL"

echo
echo "== router trace, $TOKENS tokens =="
$BIN/moetrace --model "$MODEL" --text data/corpus/corpus.txt \
    --out "data/trace-$TAG.bin" --max-tokens "$TOKENS" --chunk 1024 2>&1 | tail -2

echo
echo "== layout search (construction only; local search is quadratic in expert count) =="
$BIN/coact build --model "$MODEL" --trace "data/trace-$TAG.bin" \
    --cost data/costmodel.json --outdir "data/layouts-$TAG" \
    --report "data/layout-report-$TAG.json" --sweeps 0 --search-tokens 2000 \
    2>&1 | grep -v searched

echo
echo "== how much of the available clustering was captured =="
python3 - "data/layout-report-$TAG.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
n, k = d["n_expert"], d["n_expert_used"]
layers, tensors = d["n_moe_layers"], 3
by = {m["method"]: m["fetches_per_token"] for m in d["methods"]}
# A random layout averages k(1 - (k-1)/(n-1)) runs per layer; a perfect one averages 1.
rand_runs = k * (1 - (k - 1) / (n - 1))
per_layer = lambda f: f / (layers * tensors)
best = min((v, m) for m, v in by.items() if m not in ("identity",) and not m.startswith("random"))
got = per_layer(best[0])
print(f"experts {n}, top-{k}, {layers} MoE layers")
print(f"random layout      {rand_runs:.2f} runs/layer")
print(f"{best[1]:<18} {got:.2f} runs/layer")
print(f"perfect            1.00 runs/layer")
print(f"captured           {100*(rand_runs-got)/(rand_runs-1):.1f}% of the available clustering")
print(f"fetch reduction    {100*(1-by[best[1]]/by['identity']):.1f}%")
PY

echo
echo "Compare that capture rate against OLMoE's 17 %. If it is similar, the layout gain is a"
echo "predictable fraction of the fetch overhead that 'coact project' already reports."
