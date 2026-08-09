#!/usr/bin/env bash
# Dumps reference logits and greedy generations for every prompt in scripts/prompts.txt.
#
# usage: logits.sh <model.gguf> <outdir>
#
# The prompts span English, Norwegian, German, Japanese, C, Python, SQL, JSON and maths —
# a narrow prompt set would exercise only a handful of experts and could pass while a
# permutation had corrupted the rest.
set -euo pipefail

MODEL="${1:?usage: logits.sh <model.gguf> <outdir>}"
OUTDIR="${2:?usage: logits.sh <model.gguf> <outdir>}"
PROMPTS="$(dirname "$0")/prompts.txt"

rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

i=0
while IFS= read -r prompt; do
    [ -z "$prompt" ] && continue
    i=$((i + 1))
    d=$(printf '%s/p%02d' "$OUTDIR" "$i")
    mkdir -p "$d"
    llama-debug -m "$MODEL" -p "$prompt" --save-logits --logits-output-dir "$d" \
        --no-warmup > "$d/stdout.log" 2>&1
    # Keep only the binary logits; the .txt rendering is a lossy view of the same data.
    find "$d" -name '*.txt' -delete
    # llama-completion, not llama-cli: the instruct model makes llama-cli default to
    # conversation mode, where it blocks on stdin instead of exiting.
    llama-completion -m "$MODEL" -p "$prompt" -n 32 --temp 0 --seed 1 --no-warmup \
        -no-cnv -st 2>/dev/null < /dev/null > "$d/gen.txt"
done < "$PROMPTS"

echo "wrote $i prompt dumps to $OUTDIR"
