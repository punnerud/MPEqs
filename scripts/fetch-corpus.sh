#!/usr/bin/env bash
# Builds the trace corpus. Genre diversity is deliberate: a layout tuned on English prose
# alone would overfit whichever experts that register happens to use, and the holdout split
# would not catch it because the holdout is drawn from the same text.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=data/corpus
mkdir -p "$OUT"

fetch() { # url dest
    if [ -s "$2" ]; then echo "have $(basename "$2")"; return; fi
    echo "fetch $(basename "$2")"
    curl -sL --fail "$1" -o "$2"
}

# 1. English prose — wikitext-2-raw, the standard llama.cpp perplexity corpus.
if [ ! -s "$OUT/wikitext.txt" ]; then
    fetch "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip" "$OUT/wikitext.zip"
    unzip -o -q "$OUT/wikitext.zip" -d "$OUT/wikitext-unzip"
    find "$OUT/wikitext-unzip" -name 'wiki.test.raw' -exec cp {} "$OUT/wikitext.txt" \;
    rm -rf "$OUT/wikitext.zip" "$OUT/wikitext-unzip"
fi

# 2. Code — two languages, several files each. The class-balanced task needs roughly as many
# code chunks as prose chunks, and one file per language gives 250 against wikitext's 3200.
if [ ! -s "$OUT/code-python.txt" ]; then
    for f in Lib/argparse.py Lib/asyncio/tasks.py Lib/typing.py Lib/dataclasses.py \
             Lib/pathlib.py Lib/inspect.py Lib/enum.py Lib/json/encoder.py \
             Lib/unittest/case.py Lib/http/client.py Lib/email/message.py Lib/logging/__init__.py; do
        curl -sL --fail "https://raw.githubusercontent.com/python/cpython/3.12/$f" \
            >> "$OUT/code-python.txt" || true
    done
    echo "fetch code-python.txt"
fi
if [ ! -s "$OUT/code-c.txt" ]; then
    for f in lib/url.c lib/http.c lib/multi.c lib/transfer.c lib/sendf.c lib/connect.c \
             lib/ftp.c lib/setopt.c lib/urlapi.c lib/cookie.c lib/vtls/openssl.c; do
        curl -sL --fail "https://raw.githubusercontent.com/curl/curl/master/$f" \
            >> "$OUT/code-c.txt" || true
    done
    echo "fetch code-c.txt"
fi

# 3. Norwegian — a non-English register, where routing should differ most.
#
# Fetched in batches of 20: the extracts API caps a single request at 20 pages however many
# titles are listed, so one call silently returned a fifth of what was asked for and left the
# class at 119 KB against the other three at ~1 MB.
if [ ! -s "$OUT/norwegian.txt" ]; then
    echo "fetch norwegian.txt"
    python3 - "$OUT/norwegian.txt" <<'PYEOF'
import json, sys, time, urllib.parse, urllib.request

TITLES = [
    "Norge", "Oslo", "Bergen", "Trondheim", "Stavanger", "Tromsø", "Kristiansand",
    "Ålesund", "Svalbard", "Lofoten", "Jotunheimen", "Hardangervidda", "Nordsjøen",
    "Golfstrømmen", "Nordlys", "Hurtigruten", "Andre verdenskrig", "Vikingtiden",
    "Norges historie", "Grunnloven", "Stortinget", "Samer", "Bokmål", "Nynorsk",
    "Norsk", "Norsk litteratur", "Henrik Ibsen", "Knut Hamsun", "Sigrid Undset",
    "Edvard Munch", "Norsk musikk", "Roald Amundsen", "Fridtjof Nansen",
    "Norges geografi", "Vannkraft", "Elbil", "Equinor", "NRK", "Norsk mat", "Ski",
    "Langrenn", "Håndball", "Fotball", "Klimaendring", "Kunstig intelligens",
    "Den norske kirke", "Utdanning i Norge", "Oljefondet", "Bunad", "Fjord",
    "Nordkapp", "Preikestolen", "Trolltunga", "Geirangerfjorden", "Norges økonomi",
    "Sápmi", "Nordmenn", "Norsk språkhistorie", "Olav Tryggvason", "Harald Hårfagre",
]
flat = TITLES

out, seen = [], set()
for i in range(0, len(flat), 20):
    batch = [t for t in flat[i:i + 20] if t not in seen]
    if not batch:
        continue
    seen.update(batch)
    q = urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": "1", "exlimit": "max",
        "format": "json", "redirects": "1", "titles": "|".join(batch)})
    # Wikipedia rejects urllib's default agent with a 403; curl only worked earlier because
    # it sends one of its own.
    req = urllib.request.Request(
        f"https://no.wikipedia.org/w/api.php?{q}",
        headers={"User-Agent": "LLMdb-corpus/0.1 (research; contact via repository)"})
    try:
        with urllib.request.urlopen(req) as r:
            pages = json.load(r)["query"]["pages"].values()
    except Exception as e:
        print(f"batch {i}: {e}", file=sys.stderr)
        continue
    out += [p.get("extract", "") for p in pages if p.get("extract")]
    time.sleep(0.2)

text = "\n\n".join(out)
open(sys.argv[1], "w", encoding="utf-8").write(text)
print(f"  {len(out)} articles, {len(text)//1024} KB")
PYEOF
fi

# Interleave in fixed-size blocks rather than concatenating: the tracer resets the KV cache
# on chunk boundaries, and concatenation would put each register in its own contiguous run
# of chunks, correlating genre with position in the trace.
python3 - "$OUT" <<'PY'
import sys, pathlib
out = pathlib.Path(sys.argv[1])
parts = []
for n in ("wikitext.txt", "code-python.txt", "code-c.txt", "norwegian.txt"):
    p = out / n
    if p.exists():
        parts.append(p.read_text(encoding="utf-8", errors="replace"))
    else:
        print(f"warning: {n} missing, continuing without it", file=sys.stderr)

BLOCK = 4000
blocks, i = [], 0
while any(i * BLOCK < len(t) for t in parts):
    for t in parts:
        s = t[i * BLOCK:(i + 1) * BLOCK]
        if s:
            blocks.append(s)
    i += 1
text = "\n\n".join(blocks)
(out / "corpus.txt").write_text(text, encoding="utf-8")
print(f"corpus.txt: {len(text)} chars from {len(parts)} sources")
PY
