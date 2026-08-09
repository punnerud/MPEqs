#!/usr/bin/env bash
# Embeds the same arithmetic problems our own model trains on, but through a pretrained
# embedder, so the landmark measurement can tell two things apart:
#
#   "our 64-dim toy network has no narrow cuts"   vs   "embeddings have no narrow cuts"
#
# all-minilm is a real trained sentence embedder (384 dimensions). If a skeleton finds gateway
# structure in its view of these problems and not in ours, the negative result is about the toy
# network. If neither has it, the result is about embedding geometry.
set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-8000}"
OUT="${2:-data/custom/arith-minilm.f32}"
TXT="${OUT%.f32}.txt"
MODEL="${EMBED_MODEL:-$HOME/.ollama/models/blobs/sha256-797b70c4edf85907fe0a49eb85811256f65fa0f7bf52166b147fd16be2be4662}"
BIN="${BIN:-vendor/llama.cpp/build/bin}"

[ -s "$MODEL" ] || { echo "no embedder at $MODEL" >&2; exit 1; }

# The same held-out sample the network is scored on: same hash, same residue class, same seed.
python3 - "$TXT" "$N" <<'PY'
import random, sys
out, n = sys.argv[1], int(sys.argv[2])
LIM = 999
rng = random.Random(999)
lines, seen = [], 0
while len(lines) < n:
    a, b, op = rng.randint(-LIM, LIM), rng.randint(-LIM, LIM), rng.randrange(2)
    if ((a + LIM) * 7919 + (b + LIM) * 104729 + op) % 97 != 0:
        continue
    lines.append(f"{a} {'+' if op == 0 else '-'} {b}")
open(out, "w").write("\n".join(lines) + "\n")
print(f"{len(lines)} problems -> {out}")
PY

"$BIN/llama-embedding" -m "$MODEL" -f "$TXT" --pooling mean --embd-normalize 2 \
    --embd-output-format array -c 512 -b 2048 2>/dev/null > "$OUT.json"

# llama-embedding splits sequences on its own separator and has returned more rows than input
# lines before in this project (271 for 250 on code and Norwegian). A row/line mismatch would
# silently misalign every label, so it is checked rather than assumed.
python3 - "$OUT.json" "$OUT" "$TXT" <<'PY'
import json, struct, sys
rows = json.load(open(sys.argv[1]))
want = sum(1 for _ in open(sys.argv[3]))
if len(rows) != want:
    sys.exit(f"MISALIGNED: {len(rows)} embeddings for {want} input lines")
dim = len(rows[0])
with open(sys.argv[2], "wb") as f:
    for r in rows:
        f.write(struct.pack(f"<{dim}f", *r))
print(f"{len(rows)} x {dim} -> {sys.argv[2]}")
PY
rm -f "$OUT.json"
