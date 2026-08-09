#!/usr/bin/env bash
# Records the reference numbers for the shipped file, before anything is permuted.
#
# usage: baseline.sh <model.gguf>
set -euo pipefail

MODEL="${1:?usage: baseline.sh <model.gguf>}"
mkdir -p data

if ! ./target/release/ggufperm status --model "$MODEL" 2>/dev/null | grep -q 'identity *true'; then
    echo "refusing: $MODEL is not in its shipped order (run 'make revert' first)" >&2
    exit 1
fi

echo "== llama-bench (default: pp512 + tg128, Metal) =="
llama-bench -m "$MODEL" -r 5 -o json > data/bench-baseline.json 2>/dev/null
./scripts/bench-summary.py data/bench-baseline.json

echo
echo "== llama-bench (-ncmoe 16: all MoE weights kept host-side) =="
llama-bench -m "$MODEL" -r 5 -ncmoe 16 -o json > data/bench-baseline-ncmoe.json 2>/dev/null
./scripts/bench-summary.py data/bench-baseline-ncmoe.json

echo
echo "== reference logits =="
./scripts/logits.sh "$MODEL" data/logits-baseline

echo
echo "== reproducibility check: same file, second run must be bit-identical =="
./scripts/logits.sh "$MODEL" data/logits-baseline-repeat > /dev/null
# Compare only the payload: stdout.log carries wall-clock timings and always differs.
if diff -r -q -x 'stdout.log' data/logits-baseline data/logits-baseline-repeat > /dev/null; then
    echo "OK — llama.cpp is bit-reproducible on this backend, so the oracle is sound"
    rm -rf data/logits-baseline-repeat
else
    echo "WARNING: two runs of the *same* file already differ." >&2
    echo "The bit-exactness oracle is unusable on this backend; fall back to -ngl 0." >&2
    exit 1
fi
