#!/usr/bin/env bash
# Best-effort eviction of the macOS unified buffer cache.
#
# fetchbench opens the model with F_NOCACHE, which stops it from *populating* the cache but
# not from being served pages that are already there. Right after ggufperm rewrites the file
# the whole model is resident, and a replay silently measures RAM bandwidth instead of the
# SSD. fetchbench flags that when throughput exceeds the calibrated ceiling; this script is
# how you clear it.
#
# `purge` does the job in one call but needs root. Without it, the fallback reads several
# gigabytes of unrelated read-only system files through the ordinary cached path, pushing the
# model's pages out by LRU pressure. It touches nothing and writes nothing.
set -uo pipefail

if purge 2>/dev/null; then
    echo "cache purged"
    exit 0
fi

echo "purge needs root; falling back to LRU pressure from unrelated system files"
TOTAL_MB=0
while IFS= read -r f; do
    sz=$(stat -f %z "$f" 2>/dev/null || echo 0)
    cat "$f" > /dev/null 2>&1 || true
    TOTAL_MB=$((TOTAL_MB + sz / 1048576))
    printf '\rread %s MB' "$TOTAL_MB"
done < <(find /System/Library /System/Volumes/Preboot/Cryptexes -type f -size +40M 2>/dev/null | head -60)
echo

cat <<EOF

Read ${TOTAL_MB} MB through the cache. This evicts by pressure, not by decree — if
fetchbench still warns about contamination, run the reliable version yourself:

    ! sudo purge
EOF
