#!/usr/bin/env python3
"""Chunk each corpus source separately, keeping a class label per chunk."""
import re
import sys
from collections import Counter
from pathlib import Path

chars, out = int(sys.argv[1]), Path(sys.argv[2])
mode = sys.argv[3] if len(sys.argv) > 3 else "register"

# Two labellings of the same corpus, at very different entropies.
#
#   register — four writing styles. A model reaches 99.5 % of it with one slot, so the
#             compression limit is nowhere near and the sweep measures capacity, not learning.
#   file     — which of the 23 source files a chunk came from. All the Python chunks are
#             CPython stdlib and all the C chunks are curl, so the distinctions are subtle and
#             the task carries far more entropy for the same bytes on disk.
if mode == "file":
    import urllib.request  # noqa: F401  (kept for parity with the fetch script)
srcs = ["wikitext.txt", "code-python.txt", "code-c.txt", "norwegian.txt"]
lines, labels = [], []
for lab, name in enumerate(srcs):
    p = Path("data/corpus") / name
    if not p.exists():
        print(f"missing {name}", file=sys.stderr)
        continue
    t = re.sub(r"\s+", " ", p.read_text(encoding="utf-8", errors="replace")).strip()
    got = [t[i:i + chars] for i in range(0, len(t), chars)]
    got = [g for g in got if len(g) > chars // 2]
    if mode == "file":
        # Split the concatenated per-source file back into its parts. The fetch script
        # appends whole files, so a chunk's label is which part of the concatenation it
        # falls in — approximated by cutting the text into 12 equal spans for Python and 11
        # for C, matching how many files were appended.
        parts = {"code-python.txt": 12, "code-c.txt": 11}.get(name, 8)
        span = max(1, len(got) // parts)
        for i, g in enumerate(got):
            lines.append(g)
            labels.append(lab * 16 + min(parts - 1, i // span))
    else:
        lines += got
        labels += [lab] * len(got)
out.mkdir(parents=True, exist_ok=True)
(out / "chunks.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
(out / "labels.txt").write_text("\n".join(map(str, labels)) + "\n")
c = Counter(labels)
print(f"mode={mode}: {len(c)} classes, sizes {min(c.values())}..{max(c.values())}")
