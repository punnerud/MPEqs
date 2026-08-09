#!/usr/bin/env bash
# Downloads only the GGUF header of large models, so their expert geometry can be analysed
# without the weights.
#
# The header and tensor index sit at the front of the file, so a range request of a few
# megabytes is enough to learn every expert's size and offset. A 400 GB model costs 8 MB to
# inspect.
set -uo pipefail

OUT=data/headers
mkdir -p "$OUT"
BYTES="${BYTES:-8000000}"

# repo/file pairs. Sharded models: the first shard carries the whole tensor index.
MODELS=(
    "unsloth/Qwen3.6-35B-A3B-GGUF|Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"
    "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF|Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
    "ggml-org/gpt-oss-20b-GGUF|gpt-oss-20b-MXFP4.gguf"
    "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF|UD-Q2_K_XL/Qwen3-Coder-480B-A35B-Instruct-UD-Q2_K_XL-00001-of-00004.gguf"
    "unsloth/DeepSeek-V3-0324-GGUF|UD-IQ1_S/DeepSeek-V3-0324-UD-IQ1_S-00001-of-00004.gguf"
    "unsloth/Kimi-K2-Instruct-GGUF|UD-IQ1_S/Kimi-K2-Instruct-UD-IQ1_S-00001-of-00010.gguf"
)

: > "$OUT/sizes.txt"
for entry in "${MODELS[@]}"; do
    repo="${entry%%|*}"
    file="${entry##*|}"
    tag=$(basename "$file" .gguf)
    url="https://huggingface.co/$repo/resolve/main/$file"

    # Total size across all shards, so the projection reports the real footprint.
    total=$(curl -s "https://huggingface.co/api/models/$repo?blobs=true" | python3 -c "
import json,sys,os
d=json.load(sys.stdin)
want=os.environ['WANT']
pre=want.rsplit('-00',1)[0] if '-00' in want else want
tot=0
for s in d.get('siblings',[]):
    n=s['rfilename']
    if n.endswith('.gguf') and n.rsplit('-00',1)[0]==pre:
        tot+=s.get('size') or 0
print(tot)
" WANT="$file" 2>/dev/null)
    [ -z "$total" ] || [ "$total" = "0" ] && total=0

    if [ ! -s "$OUT/$tag.hdr" ]; then
        echo "fetching header of $tag"
        curl -sL --fail -r "0-$BYTES" "$url" -o "$OUT/$tag.hdr" || {
            echo "  failed"; continue; }
    fi
    echo "$OUT/$tag.hdr $total" >> "$OUT/sizes.txt"
    printf '  %-52s %8.1f GB\n' "$tag" "$(echo "$total/1000000000" | bc -l)"
done

echo
echo "wrote $OUT/sizes.txt"
