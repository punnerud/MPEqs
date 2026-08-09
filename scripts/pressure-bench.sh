#!/usr/bin/env bash
# End-to-end token throughput when the page cache cannot hold the model.
#
# usage: pressure-bench.sh <model.gguf> <tag> [ballast GiB list]
#
# This is the measurement the whole project is for. Warm, a 4 GB model on a 36 GB machine is
# entirely cached and layout provably cannot matter. `ballast` occupies RAM so llama.cpp's
# mmap has to fault expert pages back from the SSD on every token — the regime of a laptop
# running a model that does not fit.
#
# -ngl 0 on purpose: Metal with shared buffers aliases the mapping and makes residency
# ambiguous, while the CPU path faults through the ordinary page cache.
set -uo pipefail

MODEL="${1:?usage: pressure-bench.sh <model.gguf> <tag> [gib list]}"
TAG="${2:?usage: pressure-bench.sh <model.gguf> <tag> [gib list]}"
GIBS="${3:-0 24 28 30}"
NGEN="${NGEN:-64}"
BIN=./target/release

mkdir -p data/pressure

for GIB in $GIBS; do
    OUT="data/pressure/${TAG}-${GIB}gib.json"
    if [ -s "$OUT" ]; then echo "have $OUT"; continue; fi

    BPID=""
    if [ "$GIB" != "0" ]; then
        "$BIN/ballast" --gib "$GIB" --refresh 3600 < /dev/null > /dev/null 2>&1 &
        BPID=$!
        # Wait for the ballast to actually be resident; benchmarking while it is still
        # faulting in would measure the ballast, not the model.
        for _ in $(seq 1 120); do
            rss=$(ps -o rss= -p "$BPID" 2>/dev/null | tr -d ' ')
            [ -z "$rss" ] && break
            if [ "$rss" -gt $((GIB * 1000000)) ]; then break; fi
            sleep 1
        done
        sleep 2
    fi

    ./scripts/drop-cache.sh > /dev/null 2>&1

    echo "== $TAG, ${GIB} GiB ballast =="
    llama-bench -m "$MODEL" -ngl 0 -p 0 -n "$NGEN" -r 3 -o json > "$OUT" 2>/dev/null
    ./scripts/bench-summary.py "$OUT" 2>/dev/null || echo "  (bench failed)"

    if [ -n "$BPID" ]; then
        kill "$BPID" 2>/dev/null
        wait "$BPID" 2>/dev/null
        sleep 3
    fi
done
