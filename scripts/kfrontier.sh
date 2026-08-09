#!/usr/bin/env bash
# Quality-versus-k frontier: what does routing to fewer experts actually cost?
#
# usage: kfrontier.sh <model.gguf> [chunks]
#
# `--override-kv <arch>.expert_used_count` changes top-k at load time with no code change,
# and `--kl-divergence` measures the damage against the unmodified model's own logits. That
# is the right oracle for this regime: unlike a layout permutation, routing to fewer experts
# is not exact, so the claim has to be statistical — but it can still be rigorous.
set -euo pipefail

MODEL="${1:?usage: kfrontier.sh <model.gguf> [chunks]}"
CHUNKS="${2:-16}"
CTX="${3:-512}"
# CPU backend throughout. Metal specialises its MoE kernel per n_expert_used and has no
# variant for k=3, where it fails to compile the pipeline and segfaults. The base logits must
# come from the same backend as the variants or the KL would also measure the backend.
NGL="${4:-0}"
BASE=data/kl-base.dat
OUT=data/kfrontier.json
CORPUS=data/corpus/corpus.txt

ARCH=$(./target/release/coact stats --model "$MODEL" | awk '/^arch/ {print $2}')
KMAX=$(./target/release/coact stats --model "$MODEL" | awk '/^experts . token/ {print $4}')
echo "arch=$ARCH  top-k=$KMAX  chunks=$CHUNKS  ctx=$CTX"

if [ ! -s "$BASE" ]; then
    echo "== reference logits from the unmodified model =="
    llama-perplexity -m "$MODEL" -f "$CORPUS" -c "$CTX" --chunks "$CHUNKS" \
        --save-all-logits "$BASE" -ngl "$NGL" > data/kl-base.log 2>&1
    echo "wrote $BASE ($(du -h "$BASE" | cut -f1))"
fi

mkdir -p data/kl
for K in $(seq 1 "$KMAX"); do
    log="data/kl/k$K.log"
    if [ -s "$log" ]; then echo "have k=$K"; continue; fi
    echo "== k=$K =="
    if ! llama-perplexity -m "$MODEL" -f "$CORPUS" -c "$CTX" --chunks "$CHUNKS" \
        --kl-divergence --kl-divergence-base "$BASE" -ngl "$NGL" \
        --override-kv "$ARCH.expert_used_count=int:$K" > "$log" 2>&1
    then
        echo "  k=$K failed (see $log); continuing"
        mv "$log" "$log.failed"
        continue
    fi
    grep -E "^(Mean PPL\(Q\) |Mean PPL\(Q\)/|Mean    KLD|Median  KLD)" "$log" || true
done

./scripts/kfrontier-summary.py "$KMAX" > "$OUT"
echo
cat "$OUT"
echo "wrote $OUT"
