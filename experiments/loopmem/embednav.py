#!/usr/bin/env python3
"""Put a vector on every node of the split tree and descend by similarity, not by prefix.

Phase 29 built a lossless split of 1.29 MB and then could not steer it: the top level offered
three nodes of 430,000 characters each, previewed by their first ninety characters, and both
models chose at 1/8. The diagnosis was that a node needs a REPRESENTATION rather than a prefix.
This is that representation.

Each leaf sentence is embedded once. A parent's vector is the mean of its children's, normalised
— so a node stands for everything beneath it rather than for its opening words, which is exactly
what the prefix could not do. Descending then costs eight comparisons per level instead of a
model call, and no language model is involved in the navigation at all.

Three ways to find the leaf, on the same targets:

  PREFIX     what phase 29 measured, for reference                       0/8, and 1/8 per level
  TREE       greedy descent by cosine, 8 comparisons per level
  FLAT       score every leaf and take the best — the brute-force answer, and the yardstick

FLAT is what TREE has to match, and the interesting number is what TREE costs to match it:
5 levels x 8 children is 40 comparisons against 8,920, which is the whole argument for the tree.
Mean pooling loses information, so whether routing survives it is a real question and not a
formality — a parent averaged over 1,115 sentences may point nowhere in particular.

Two query kinds, because they are not equally hard:
  EXACT   the target sentence itself, which tests routing alone
  HALF    its first half, which tests routing when the query is not what is stored
"""
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from longdoc import BRANCH, DOC, build_tree, group, sentences  # noqa: E402

EMBED_BIN = "llama-embedding"
EMBED_MODEL = str(Path.home() / ".ollama/models/blobs"
                  / "sha256-797b70c4edf85907fe0a49eb85811256f65fa0f7bf52166b147fd16be2be4662")
CACHE = Path("data/custom/wikitext-sentences.f32")


def embed(lines, cache=None):
    """Mean-pooled, L2-normalised MiniLM vectors, one per line.

    Newlines are the record separator for llama-embedding, so a line containing one would
    silently become two rows and every vector after it would belong to the wrong sentence.
    Whitespace is collapsed before writing for exactly that reason.
    """
    if cache and cache.exists():
        raw = cache.read_bytes()
        dim = 384
        n = len(raw) // (4 * dim)
        if n == len(lines):
            return [list(struct.unpack_from(f"<{dim}f", raw, i * 4 * dim)) for i in range(n)]
    # Truncated to fit the encoder's 512-token window. Some "sentences" here are tables and
    # lists that contain no " . " at all and run to thousands of characters; the embedder
    # segfaulted on the batch containing them. Only the VECTOR is computed from a prefix — the
    # leaf text in the tree is untouched, so the split stays lossless and only the description
    # of a long leaf is partial.
    clean = [re.sub(r"\s+", " ", ln).strip()[:1200] or "." for ln in lines]
    # In batches. All 8,920 lines in one call returned empty stdout with a zero exit status —
    # llama-embedding allocates for the whole file and quietly produced nothing — while 2,000
    # went through fine. A silent success that emits no rows is worse than a crash, so the row
    # count is checked per batch rather than only at the end.
    rows = []
    step = 1500
    for start in range(0, len(clean), step):
        batch = clean[start:start + step]
        src = Path("/tmp/embednav-in.txt")
        src.write_text("\n".join(batch) + "\n")
        out = subprocess.run(
            [EMBED_BIN, "-m", EMBED_MODEL, "-f", str(src), "--pooling", "mean",
             "--embd-normalize", "2", "--embd-output-format", "array",
             "-c", "512", "-b", "2048", "--no-warmup"],
            capture_output=True, text=True)
        try:
            part = json.loads(out.stdout)
        except json.JSONDecodeError:
            raise SystemExit(
                f"embedder produced no JSON for lines {start}-{start + len(batch)}: "
                f"{out.stderr[-200:]}") from None
        if len(part) != len(batch):
            raise SystemExit(f"batch at {start}: {len(part)} rows for {len(batch)} lines")
        rows.extend(part)
        if len(clean) > step:
            print(f"  {len(rows):,}/{len(clean):,}", flush=True)
    if len(rows) != len(lines):
        raise SystemExit(f"embedder returned {len(rows)} rows for {len(lines)} lines")
    if cache:
        with open(cache, "wb") as f:
            for r in rows:
                f.write(struct.pack(f"<{len(r)}f", *r))
    return rows


def norm(v):
    s = sum(x * x for x in v) ** 0.5
    return [x / s for x in v] if s else v


def mean_vec(vs):
    n = len(vs)
    dim = len(vs[0])
    return norm([sum(v[d] for v in vs) / n for d in range(dim)])


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def build_vectors(leaf_vecs):
    """A parent is the mean of its children, level by level, bottom up."""
    levels = [leaf_vecs]
    while len(levels[-1]) > BRANCH:
        levels.append([mean_vec(g) for g in group(levels[-1])])
    return levels


def build_samples(leaf_vecs, per_node=8):
    """A parent keeps a few actual descendants instead of their average.

    Averaging 1,115 sentences produces a vector pointing at the corpus centroid rather than at
    anything in particular, which is why mean routing found 18 of 40 where a flat scan found 40.
    Scoring a node by the BEST of a handful of real descendants is the bottleneck reasoning the
    hop phases arrived at: a node is worth as much as the best thing under it, not as much as its
    average thing. Evenly spaced rather than random, so the sample spans the node.
    """
    levels = [[[v] for v in leaf_vecs]]
    while len(levels[-1]) > BRANCH:
        parents = []
        for g in group(levels[-1]):
            pool = [v for child in g for v in child]
            step = max(1, len(pool) // per_node)
            parents.append(pool[::step][:per_node])
        levels.append(parents)
    return levels


def descend_samples(sample_levels, q):
    """Greedy descent scoring each node by its best kept descendant."""
    node = list(range(len(sample_levels[-1])))
    comparisons = 0
    for depth in range(len(sample_levels) - 1, -1, -1):
        level = sample_levels[depth]
        best, best_score = None, -2.0
        for i in node:
            for v in level[i]:
                comparisons += 1
                sc = dot(v, q)
                if sc > best_score:
                    best, best_score = i, sc
        if depth == 0:
            return best, comparisons
        node = list(range(best * BRANCH,
                          min((best + 1) * BRANCH, len(sample_levels[depth - 1]))))
    return None, comparisons


def descend_beam(vec_levels, q, beam=4):
    """Keep the best `beam` nodes at each level instead of committing to one.

    This project already measured the greedy-versus-beam gap on its own kNN graph: pure greedy
    reached the target 49% of the time and beam=4 reached it 99.5%. A greedy descent through a
    mean-pooled tree is the same failure mode — one bad average at the top discards everything
    beneath it, with no way back. Beam keeps the alternatives alive at a cost that is still
    logarithmic in the document.
    """
    frontier = list(range(len(vec_levels[-1])))
    comparisons = 0
    for depth in range(len(vec_levels) - 1, -1, -1):
        level = vec_levels[depth]
        scored = []
        for i in frontier:
            comparisons += 1
            scored.append((dot(level[i], q), i))
        scored.sort(reverse=True)
        keep = [i for _, i in scored[:beam]]
        if depth == 0:
            return keep[0], comparisons
        frontier = [c for i in keep
                    for c in range(i * BRANCH,
                                   min((i + 1) * BRANCH, len(vec_levels[depth - 1])))]
    return None, comparisons


def descend(vec_levels, q):
    """Greedy best child at each level. Returns (leaf index, comparisons)."""
    node = list(range(len(vec_levels[-1])))
    comparisons = 0
    for depth in range(len(vec_levels) - 1, -1, -1):
        level = vec_levels[depth]
        scores = [(dot(level[i], q), i) for i in node]
        comparisons += len(scores)
        best = max(scores)[1]
        if depth == 0:
            return best, comparisons
        node = list(range(best * BRANCH, min((best + 1) * BRANCH, len(vec_levels[depth - 1]))))
    return None, comparisons


def main(n_queries=40, seed=3, out="data/custom/embednav.json"):
    import random
    rng = random.Random(int(seed))
    text = Path(DOC).read_text()
    sents = sentences(text)
    levels = build_tree(sents)
    print(f"{DOC}: {len(sents):,} sentences, {len(levels)} levels, branching {BRANCH}")

    print("embedding leaves...", flush=True)
    leaf_vecs = embed(sents, CACHE)
    vec_levels = build_vectors(leaf_vecs)
    sample_levels = build_samples(leaf_vecs)
    print(f"vectors: {[len(v) for v in vec_levels]} nodes per level, "
          f"{len(leaf_vecs[0])} dimensions\n")

    idx = [i for i, s in enumerate(sents) if 60 < len(s) < 220]
    picks = rng.sample(idx, min(int(n_queries), len(idx)))

    results = {}
    for kind in ("exact", "half"):
        queries = [sents[i] if kind == "exact" else sents[i][:len(sents[i]) // 2] for i in picks]
        qvecs = embed(queries)
        tree_hits = flat_hits = agree = samp_hits = beam_hits = 0
        tree_cost = samp_cost = beam_cost = 0
        for k, i in enumerate(picks):
            q = qvecs[k]
            leaf, cost = descend(vec_levels, q)
            tree_cost += cost
            # The brute-force answer over every leaf, which is what the tree has to match.
            sleaf, scost = descend_samples(sample_levels, q)
            samp_cost += scost
            samp_hits += sleaf == i
            bleaf, bcost = descend_beam(vec_levels, q)
            beam_cost += bcost
            beam_hits += bleaf == i
            flat = max(range(len(leaf_vecs)), key=lambda j: dot(leaf_vecs[j], q))
            tree_hits += leaf == i
            flat_hits += flat == i
            agree += leaf == flat
        n = len(picks)
        results[kind] = {"queries": n, "tree_correct": tree_hits,
                         "sample_correct": samp_hits,
                         "beam_correct": beam_hits,
                         "mean_comparisons_beam": beam_cost / n,
                         "mean_comparisons_sample": samp_cost / n,
                         "flat_correct": flat_hits,
                         "tree_agrees_with_flat": agree,
                         "mean_comparisons_tree": tree_cost / n,
                         "comparisons_flat": len(leaf_vecs)}
        print(f"{kind:<7} greedy {tree_hits}/{n} ({tree_cost / n:.0f})   "
              f"best-of-8 {samp_hits}/{n} ({samp_cost / n:.0f})   "
              f"beam4 {beam_hits}/{n} ({beam_cost / n:.0f})   "
              f"flat {flat_hits}/{n} ({len(leaf_vecs):,})")

    speedup = len(leaf_vecs) / max(results["exact"]["mean_comparisons_tree"], 1)
    print(f"\nthe tree looks at {speedup:.0f}x fewer nodes than scanning every leaf")
    print("Phase 29 navigated the same tree from ninety-character prefixes and found 0 of 8.")
    print("What changed is what a node is described BY, not how it is searched.")
    summary = {"sentences": len(sents), "levels": len(levels), "branch": BRANCH,
               "dim": len(leaf_vecs[0]), "nodes_per_level": [len(v) for v in vec_levels],
               "speedup_vs_flat": speedup, **results}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
