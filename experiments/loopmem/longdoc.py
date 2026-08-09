#!/usr/bin/env python3
"""Hierarchical splitting on a real document: 1.29 MB of Wikipedia text, not fifteen sentences.

The two-level demonstration in phase 28 narrowed a fifteen-sentence document in two questions and
reassembled it exactly. That is the right shape and the wrong scale — fifteen sentences fit in a
prompt, so nothing was actually being avoided.

This is `data/corpus/wikitext.txt`: 1,290,590 characters, 241,215 words, 64 articles, 8,919
sentences. Roughly five hundred pages. It does not fit in any of these models' context, so the
narrowing is load-bearing rather than decorative.

Two claims are separable here and both are measured.

  THE SPLIT IS LOSSLESS AT SCALE       a property of the record alone, no model involved. Every
                                       level must join back to its parent character for
                                       character, and the whole tree back to the original bytes.

  THE MODEL CAN NAVIGATE IT            choosing which branch to descend, from a preview, at every
                                       level. This is the part that can fail, and the honest
                                       baseline is that the record's own `find` needs zero model
                                       calls and is always right — so the model is only worth
                                       asking when the target is described rather than quoted.

Branching is fixed at eight, so a prompt never lists more than eight options however large the
document is. 8,919 sentences is then five levels: five questions to reach any sentence in five
hundred pages.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import MODELS, ask  # noqa: E402

DOC = "data/corpus/wikitext.txt"
BRANCH = 8

PICK = """A document has been split into {n} parts. Here is the start of each:

{options}

Which part contains this text?

    "{target}"

Reply with only the number.
"""


def sentences(text):
    """Split into sentences, keeping the separators so the join is exact.

    Every piece carries its own trailing delimiter, so `"".join(parts) == text` holds by
    construction rather than by convention. A split that needs a rule to rejoin is not lossless,
    it is a lossy split with a repair step.
    """
    parts = re.split(r"( \. )", text)
    out = []
    for i in range(0, len(parts), 2):
        chunk = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
        if chunk:
            out.append(chunk)
    return out


def group(items, b=BRANCH):
    """Chunks of b items each. Fixed SIZE, not a fixed count.

    Grouping into eight groups instead put 1,115 sentences in each and the level below offered
    1,115 options, which is the thing the split exists to avoid. Fixed size is what makes the
    prompt constant and the depth logarithmic.
    """
    return [items[i:i + b] for i in range(0, len(items), b)]


def build_tree(items):
    """Group repeatedly until the top is a single node. Returns the levels, bottom-up."""
    levels = [items]
    while len(levels[-1]) > BRANCH:
        levels.append(["".join(g) for g in group(levels[-1])])
    return levels


def verify_lossless(levels, original):
    """Every level must join to the same bytes. Checked, not asserted."""
    return all("".join(level) == original for level in levels)


def preview(text, width=90):
    t = " ".join(text.split())
    return (t[:width] + "...") if len(t) > width else t


def navigate(model, levels, target, tally):
    """Descend the tree, one choice per level. Returns (found, calls, per-level correctness)."""
    top = levels[-1]
    idx_path, calls, correct = [], 0, []
    # Which branch is right at each level is known, so a wrong turn is recorded where it happens
    # rather than only showing up as a miss at the bottom.
    node = list(range(len(top)))
    current = top
    for depth in range(len(levels) - 1, -1, -1):
        level = levels[depth]
        options = [level[i] for i in node]
        truth = next((k for k, i in enumerate(node) if target in level[i]), None)
        if truth is None:
            return False, calls, correct
        if len(options) == 1:
            node = expand(levels, depth, node[0])
            continue
        listing = "\n".join(f"{k + 1}. {preview(o)}" for k, o in enumerate(options))
        reply = ask(model, PICK.format(n=len(options), options=listing, target=preview(target, 70)),
                    n=16)
        calls += 1
        m = re.search(r"\d+", reply)
        pick = int(m.group(0)) - 1 if m else -1
        pick = pick if 0 <= pick < len(options) else 0
        correct.append(pick == truth)
        tally[f"level{depth}_right"] += pick == truth
        tally[f"level{depth}_seen"] += 1
        if depth == 0:
            return level[node[pick]] == target or target in level[node[pick]], calls, correct
        node = expand(levels, depth, node[pick])
        current = options[pick]
    return False, calls, correct


def expand(levels, depth, index):
    """The children of node `index` at `depth`: a contiguous run of BRANCH in the level below."""
    below = levels[depth - 1]
    start = index * BRANCH
    return list(range(start, min(start + BRANCH, len(below))))


def main(n_queries=8, seed=3, out="data/custom/longdoc.json"):
    import random
    rng = random.Random(int(seed))
    text = Path(DOC).read_text()
    sents = sentences(text)
    levels = build_tree(sents)
    lossless = verify_lossless(levels, text)

    print(f"{DOC}: {len(text):,} characters, {len(text.split()):,} words, "
          f"{len(sents):,} sentences")
    print(f"split tree: {len(levels)} levels, branching {BRANCH}, "
          f"top level has {len(levels[-1])} nodes")
    print(f"every level rejoins to the original bytes: {lossless}\n")

    # The record's own baseline. It is exact and needs no model, which is the honest yardstick:
    # a literal target is a job for `find`, not for a language model.
    targets = []
    while len(targets) < int(n_queries):
        s = rng.choice(sents)
        if 60 < len(s) < 220 and text.count(s) == 1:
            targets.append(s)
    record_hits = sum(1 for t in targets if text.find(t) >= 0)
    print(f"record find(): {record_hits}/{len(targets)} located, 0 model calls\n")

    print(f"{'model':<10}{'found':>7}{'calls':>7}  accuracy by level, top first")
    summary = {"chars": len(text), "words": len(text.split()), "sentences": len(sents),
               "levels": len(levels), "branch": BRANCH, "lossless": lossless,
               "queries": len(targets), "record_find": record_hits}
    rows = []
    for model in MODELS:
        tally = Counter()
        found = calls = 0
        for t in targets:
            ok, c, _ = navigate(model, levels, t, tally)
            found += ok
            calls += c
            rows.append({"model": model, "target": preview(t, 60), "found": ok, "calls": c})
        per_level = [
            f"{tally[f'level{d}_right']}/{tally[f'level{d}_seen']}"
            for d in range(len(levels) - 1, -1, -1) if tally[f"level{d}_seen"]
        ]
        summary[model] = {"found": found, "calls": calls,
                          "per_level": per_level,
                          "mean_calls": calls / len(targets)}
        print(f"{model:<10}{found:>4}/{len(targets):<2}{calls / len(targets):>7.1f}  "
              f"{'  '.join(per_level)}")

    print(f"\nFive hundred pages, {len(levels)} questions to reach any sentence, and the tree")
    print("rejoins to the original bytes. The narrowing is exact; whether the model can steer")
    print("it from a ninety-character preview is the separate question, answered above.")
    Path(out).write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
