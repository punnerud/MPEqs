#!/usr/bin/env bash
# Chunks each corpus source separately and embeds them with a source label.
#
# The four registers (English prose, Python, C, Norwegian) give a real four-class task on data
# already on disk, which is what the sparse-routing experiment needs: genuine structure rather
# than synthetic noise.
set -euo pipefail
CHARS="${1:-400}"
MODEL="${EMBED_MODEL:-$HOME/.ollama/models/blobs/sha256-797b70c4edf85907fe0a49eb85811256f65fa0f7bf52166b147fd16be2be4662}"
OUT=data/labelled
mkdir -p "$OUT"

python3 scripts/chunk_labelled.py "$CHARS" "$OUT" "${2:-register}"
python3 scripts/embed_slices.py "$OUT" "$MODEL"
