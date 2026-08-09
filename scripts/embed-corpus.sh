#!/usr/bin/env bash
# Embeds the corpus one chunk per line and writes a raw [n, dim] f32 block.
#
# usage: embed-corpus.sh [chunk chars] [out]
#
# all-minilm is already present as an ollama blob (45 MB, 384 dimensions) and is plenty for
# the structural question: whether the resulting distance matrix has gateway structure depends
# on the geometry of the semantic space, not on the embedder being state of the art.
set -euo pipefail

CHARS="${1:-400}"
OUT="${2:-data/embeddings.f32}"
MODEL="${EMBED_MODEL:-$HOME/.ollama/models/blobs/sha256-797b70c4edf85907fe0a49eb85811256f65fa0f7bf52166b147fd16be2be4662}"
CORPUS=data/corpus/corpus.txt
CHUNKS=data/corpus/chunks.txt

if [ ! -s "$MODEL" ]; then
    echo "no embedding model at $MODEL — set EMBED_MODEL to a GGUF embedder" >&2
    exit 1
fi

# One chunk per line, newlines and tabs stripped: llama-embedding splits sequences on the
# newline separator, so a chunk containing one would silently become two.
python3 - "$CORPUS" "$CHUNKS" "$CHARS" <<'PY'
import sys, re, pathlib
src, dst, chars = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
text = re.sub(r"\s+", " ", src.read_text(encoding="utf-8", errors="replace")).strip()
lines = [text[i:i + chars] for i in range(0, len(text), chars)]
lines = [l for l in lines if len(l) > chars // 4]
dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{len(lines)} chunks of ~{chars} chars")
PY

N=$(wc -l < "$CHUNKS" | tr -d ' ')
echo "embedding $N chunks…"
llama-embedding -m "$MODEL" -f "$CHUNKS" --pooling mean --embd-normalize 2 \
    --embd-output-format array -c 512 -b 2048 --no-warmup 2>/dev/null > data/embeddings.json

python3 - data/embeddings.json "$OUT" <<'PY'
import json, struct, sys, pathlib
rows = json.load(open(sys.argv[1]))
if not rows:
    raise SystemExit("llama-embedding produced nothing")
dim = len(rows[0])
if any(len(r) != dim for r in rows):
    raise SystemExit("ragged embedding rows")
with open(sys.argv[2], "wb") as f:
    for r in rows:
        f.write(struct.pack(f"<{dim}f", *r))
print(f"wrote {sys.argv[2]}: {len(rows)} x {dim} f32")
PY
rm -f data/embeddings.json
