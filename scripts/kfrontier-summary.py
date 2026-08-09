#!/usr/bin/env python3
"""Parse the llama-perplexity KL-divergence logs written by kfrontier.sh into one table."""
import json
import re
import sys
from pathlib import Path

FIELDS = {
    "mean_kld": r"^Mean\s+KLD:\s+([0-9.eE+-]+)",
    "median_kld": r"^Median\s+KLD:\s+([0-9.eE+-]+)",
    "p99_kld": r"^99\.0%\s+KLD:\s+([0-9.eE+-]+)",
    "ppl_ratio": r"^Mean PPL\(Q\)/PPL\(base\)\s+:\s+([0-9.eE+-]+)",
    "ppl": r"^Mean PPL\(Q\)\s+:\s+([0-9.eE+-]+)",
}
SAME_TOP = re.compile(r"^Same top pair:\s+([0-9.]+)")


def parse(path: Path) -> dict:
    row = {}
    text = path.read_text(errors="replace")
    for name, pat in FIELDS.items():
        m = re.search(pat, text, re.M)
        if m:
            row[name] = float(m.group(1))
    m = SAME_TOP.search(text)
    if m:
        row["same_top_pct"] = float(m.group(1))
    else:
        # Fall back to the per-chunk table's last "Same top p" column.
        rows = re.findall(r"([0-9.]+)\s+±\s+[0-9.]+\s*%\s*$", text, re.M)
        if rows:
            row["same_top_pct"] = float(rows[-1])
    return row


def main() -> int:
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    out = []
    for k in range(1, kmax + 1):
        p = Path(f"data/kl/k{k}.log")
        if not p.exists():
            continue
        row = {"k": k, "fetch_reduction_pct": 100.0 * (kmax - k) / kmax}
        row.update(parse(p))
        out.append(row)

    hdr = f"{'k':>3} {'fetches':>9} {'PPL':>9} {'PPL ratio':>11} {'mean KLD':>10} {'median KLD':>11} {'p99 KLD':>9} {'same top':>9}"
    print("// " + hdr, file=sys.stderr)
    for r in out:
        print(
            f"// {r['k']:>3} {100 - r['fetch_reduction_pct']:>8.0f}% "
            f"{r.get('ppl', float('nan')):>9.3f} {r.get('ppl_ratio', float('nan')):>11.4f} "
            f"{r.get('mean_kld', float('nan')):>10.4f} {r.get('median_kld', float('nan')):>11.5f} "
            f"{r.get('p99_kld', float('nan')):>9.3f} {r.get('same_top_pct', float('nan')):>8.2f}%",
            file=sys.stderr,
        )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
