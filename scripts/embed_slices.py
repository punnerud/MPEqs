#!/usr/bin/env python3
"""Embed each class's chunks in its own call, so labels cannot drift.

llama-embedding returns more rows than input lines on some content — 271 for 250 on the C and
Norwegian slices, evidently splitting sequences somewhere — and a silent off-by-N would
misalign every label against its embedding, turning the task into noise. Embedding one class
per call removes the problem rather than working around it: every row that call returns is
that class, however many there are.
"""
import json
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

out, model = Path(sys.argv[1]), sys.argv[2]
lines = (out / "chunks.txt").read_text(encoding="utf-8").splitlines()
labels = [int(x) for x in (out / "labels.txt").read_text().split()]
assert len(lines) == len(labels), f"{len(lines)} chunks vs {len(labels)} labels"

by_class: dict[int, list[str]] = {}
for line, lab in zip(lines, labels):
    by_class.setdefault(lab, []).append(line)

rows, kept = [], []
for lab in sorted(by_class):
    part = by_class[lab]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(part) + "\n")
        tmp = f.name
    res = subprocess.run(
        ["llama-embedding", "-m", model, "-f", tmp, "--pooling", "mean",
         "--embd-normalize", "2", "--embd-output-format", "array", "-c", "512", "--no-warmup"],
        capture_output=True, text=True)
    Path(tmp).unlink(missing_ok=True)
    got = json.loads(res.stdout)
    if len(got) != len(part):
        print(f"class {lab}: {len(got)} embeddings for {len(part)} lines "
              f"(the splitter again; all rows are still class {lab})", file=sys.stderr)
    rows += got
    kept += [lab] * len(got)

dim = len(rows[0])
with open(out / "emb.f32", "wb") as f:
    for r in rows:
        f.write(struct.pack(f"<{dim}f", *r))
(out / "labels.txt").write_text("\n".join(map(str, kept)) + "\n")
print(f"wrote {len(rows)} x {dim}, classes {dict(Counter(kept))}")
