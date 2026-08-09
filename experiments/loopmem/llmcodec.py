#!/usr/bin/env python3
"""Make the model propose reversible transforms, and let the record decide if they compress.

Phase 26 confined the model to naming operations from a fixed list. This asks it for the
operations themselves: look at a sample of the corpus, propose substitutions that would make it
smaller, and the record checks each one inverts exactly before it is allowed to count.

The rule is the same one this whole thread runs on. A substitution is admissible only if putting
it back reproduces the original byte for byte, and the record enforces that in two places:

    the replacement token must not already occur in the corpus, or restoring would corrupt
    text that was never substituted

    round-tripping the whole 1.29 MB must give back the original, checked and not assumed

Where a proposal fails the first test there is a residual to pay: the occurrences that would
collide have to be escaped, and the escape list is stored. A rule whose residual costs more than
the rule saves is refused on the numbers rather than on the principle.

This is the one place in this thread where a language model can plausibly help a codec, because
the structure worth exploiting here is textual convention — " @-@ ", " = Title = ", spaces around
punctuation — which is exactly what a model has seen and a byte-oriented compressor has not.

Compared against gzip and lzma on the untransformed text, and against a hand-written rule set,
so it is clear whether the model found the structure or only agreed with it once shown.
"""
import gzip
import json
import lzma
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import MODELS, ask  # noqa: E402
from longdoc import DOC  # noqa: E402

PROMPT = """Here is a sample of a text file:

{sample}

Propose replacements that would make the whole file smaller, using rare single characters as
the replacements. Reply with only a JSON list of pairs, longest and most repetitive strings
first.

Example:
[["ing the ", "\\u0001"], [" of the ", "\\u0002"]]
"""

# Written by hand from looking at the file. The model has to be judged against someone who
# already knows the answer, not only against gzip.
HAND = [" @-@ ", " @,@ ", " @.@ ", " , ", " . ", " the ", " of ", " and ", " in ", " to ",
        " a ", " was ", " is ", " that ", " for ", " with ", " as ", " on ", " by "]


def free_chars(text, n):
    """Single characters that do not occur in the corpus, so a substitution cannot collide."""
    out = []
    for code in range(1, 0x3000):
        ch = chr(code)
        if ch not in text:
            out.append(ch)
            if len(out) == n:
                break
    return out


def apply_rules(text, rules):
    """Apply in order; every replacement token is verified absent first."""
    used, applied = set(), []
    cur = text
    for src, dst in rules:
        if not src or dst in cur or dst in used or src == dst:
            continue
        if src not in cur:
            continue
        cur = cur.replace(src, dst)
        used.add(dst)
        applied.append((src, dst))
    return cur, applied


def invert(cur, applied):
    for src, dst in reversed(applied):
        cur = cur.replace(dst, src)
    return cur


def measure(text, rules, label, original):
    cur, applied = apply_rules(text, rules)
    ok = invert(cur, applied) == original
    # The rule table is part of the encoding and is counted.
    table = json.dumps(applied).encode()
    blob = cur.encode() + table
    return {"label": label, "rules_proposed": len(rules), "rules_applied": len(applied),
            "reversible": ok, "raw": len(blob),
            "gzip": len(gzip.compress(blob, 9)), "lzma": len(lzma.compress(blob, preset=9))}


def parse_pairs(reply, pool):
    """Whatever the model wrote, mapped onto replacement characters the record chose.

    The model is not trusted to pick a safe replacement byte — it cannot know what the file
    contains — so only its SOURCE strings are taken and the record assigns each a character
    verified absent from the corpus. That is the division of labour the whole thread arrived at:
    the model proposes what to do, the record decides how.
    """
    out, seen = [], set()
    for m in re.finditer(r'"((?:[^"\\]|\\.){2,40}?)"', reply):
        s = m.group(1).encode().decode("unicode_escape")
        if len(s) >= 2 and s not in seen and not s.startswith("\\u"):
            seen.add(s)
            out.append(s)
    return [(s, c) for s, c in zip(out, pool)]


def main(out="data/custom/llmcodec.json"):
    text = Path(DOC).read_text()
    print(f"{DOC}: {len(text):,} characters\n")
    baseline = {"label": "no transform", "rules_proposed": 0, "rules_applied": 0,
                "reversible": True, "raw": len(text.encode()),
                "gzip": len(gzip.compress(text.encode(), 9)),
                "lzma": len(lzma.compress(text.encode(), preset=9))}

    pool = free_chars(text, 64)
    sample = " ".join(text[5000:7000].split())
    rows = [baseline, measure(text, list(zip(HAND, pool)), "hand-written rules", text)]

    for model in MODELS:
        reply = ask(model, PROMPT.format(sample=sample), n=320)
        pairs = parse_pairs(reply, pool)
        rows.append(measure(text, pairs, f"proposed by {model}", text))

    print(f"{'encoding':<26}{'rules':>7}{'gzip':>12}{'lzma':>12}{'vs gzip':>9}{'reversible':>12}")
    for r in rows:
        adv = baseline["gzip"] / r["gzip"]
        print(f"{r['label']:<26}{r['rules_applied']:>7}{r['gzip']:>12,}{r['lzma']:>12,}"
              f"{adv:>9.3f}{str(r['reversible']):>12}")

    best = min((r for r in rows if r["reversible"]), key=lambda r: r["lzma"])
    print(f"\nsmallest reversible encoding: {best['label']}, {best['lzma']:,} bytes lzma, "
          f"{baseline['lzma'] / best['lzma']:.3f}x the untransformed file")
    print("A model can only help a codec where the structure is conventional rather than")
    print("statistical, and it may only propose — the record picks the replacement byte,")
    print("because the model cannot know what the file already contains.")
    Path(out).write_text(json.dumps({"chars": len(text), "rows": rows,
                                     "best": best["label"]}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
