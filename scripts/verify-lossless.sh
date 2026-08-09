#!/usr/bin/env bash
# Checks that the currently applied expert layout has not changed what the model does.
#
# usage: verify-lossless.sh <model.gguf>
#
# Three checks, in increasing order of what they actually prove:
#
#   1. Greedy generations must be character-identical. Hard failure.
#   2. Argmax of the logits must not move on any prompt. Hard failure.
#   3. Raw logit bytes are compared and the deviation is reported, but a difference is
#      expected, not a bug. See below.
#
# Why (3) is not a failure: permuting the expert axis is exact in exact arithmetic, and
# `coact compare` proves the implementation is exact by showing the first MoE layer routes
# identically for every token. But llama.cpp computes the gate weights as a softmax over all
# experts, and that sum runs in storage order. Reordering experts therefore changes the last
# bits of every gate weight, and sixteen layers of that compounds into a visible logit delta.
# The transformation is exact; the float evaluation of it is not.
set -euo pipefail

MODEL="${1:?usage: verify-lossless.sh <model.gguf>}"
BASE=data/logits-baseline
CUR=data/logits-current

if [ ! -d "$BASE" ]; then
    echo "no baseline at $BASE — run 'make baseline' on the shipped file first" >&2
    exit 1
fi

"$(dirname "$0")/logits.sh" "$MODEL" "$CUR"

echo
echo "== greedy generations =="
fail=0
checked=0
for d in "$BASE"/p*; do
    p=$(basename "$d")
    checked=$((checked + 1))
    if ! cmp -s "$d/gen.txt" "$CUR/$p/gen.txt"; then
        echo "CHANGED  $p/gen.txt"
        fail=$((fail + 1))
    fi
done
if [ "$fail" -eq 0 ]; then
    echo "all $checked generations identical"
fi

echo
echo "== logits =="
if ! ./scripts/logit-diff.py "$BASE" "$CUR"; then
    echo "argmax moved — the layout changed the model's decisions" >&2
    fail=$((fail + 1))
fi

if [ "$fail" -ne 0 ]; then
    echo
    echo "BEHAVIOUR CHANGED. Revert: make revert"
    exit 1
fi

echo
echo "output-equivalent: generations and argmax unchanged on all $checked prompts"
echo "for proof that the permutation itself is exact, run: make verify-routing"
