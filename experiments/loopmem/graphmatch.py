#!/usr/bin/env python3
"""Graph-signature retrieval: nearest by STRUCTURE, measured against nearest by topic.

Five arms died on the same wound — retrieval by masked-text embedding fetches TOPIC, and
plans do not transfer along topic (phases 51, 54, 60). The proposal: represent each problem
as the record's own text graph — lexicon nouns as nodes, verbs as edges, numbers attached —
and retrieve by GRAPH similarity, the N x N discipline streamed in blocks so the 1,554 x
1,554 matrix never materialises.

The measurement the earlier phases lacked has a built-in ground truth: the store knows every
problem's PLAN SHAPE (599 of them). A retrieval is type-correct when its top-1 neighbour has
the same plan shape as the query. The anchor X — the masked-text embedding's shape-match
rate — is computed from the EXISTING vectors before the graph signature runs, so the
comparison cannot drift toward the new method.

Signatures are bags of structural features hashed into a sparse vector: noun-lemma unigrams
weighted low (they carry topic — deliberately damped), verb-noun and noun-verb-noun triples,
number-unit attachments, count of numbers, and the asked lemma. Cosine over L2-normalised
buckets. Leave-one-out over the store's own problems: every problem is a query against all
the others, both methods, same protocol.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dimcheck import asked_unit  # noqa: E402
from embednav import embed  # noqa: E402
from lexicon import Lexicon  # noqa: E402
from mapstore import NUM, build_store, mask, norm  # noqa: E402
from textgraph import units_of_v2  # noqa: E402

WORD = re.compile(r"[A-Za-z]+")
DIM = 4096


def shape_of(t):
    return "|".join(re.sub(r"v\d+|S\d+|[\d.]+", "#", s) for s in t["steps"])


def signature(problem, lex):
    """Structural features, hashed. Topic is damped on purpose: lemmas alone weigh 0.2."""
    feats = {}

    def add(f, w=1.0):
        feats[f] = feats.get(f, 0.0) + w

    sents = re.split(r"[.!?]", norm(problem))
    for sent in sents:
        words = WORD.findall(sent)
        tagged = []
        for i, w in enumerate(words):
            tag = lex.tag(words, i)
            lemma = lex.noun_lemma(w) if tag == "n" else w.lower()
            tagged.append((lemma, tag))
            if tag == "n":
                add(f"N:{lemma}", 0.2)
        # verb-noun bigrams and noun-verb-noun triples: the relations, weighted high
        for i, (lem, tag) in enumerate(tagged):
            if tag != "v":
                continue
            left = next((l for l, t2 in reversed(tagged[:i]) if t2 == "n"), None)
            right = next((l for l, t2 in tagged[i + 1:] if t2 == "n"), None)
            if left:
                add(f"NV:{left}:{lem}", 1.0)
            if right:
                add(f"VN:{lem}:{right}", 1.0)
            if left and right:
                add(f"NVN:{left}:{lem}:{right}", 2.0)

    values = [norm(m) for m in NUM.findall(norm(problem))]
    units = units_of_v2(problem, lex)
    for u in units:
        if u:
            add(f"NUMU:{u}", 1.5)
    add(f"NNUMS:{len(values)}", 1.0)
    ask = asked_unit(problem)
    if ask:
        add(f"ASK:{ask}", 2.0)

    v = np.zeros(DIM, np.float32)
    for f, w in feats.items():
        v[hash(f) % DIM] += w
    n = np.linalg.norm(v)
    return v / n if n else v


def topk_blocked(matrix, k=2, block=256):
    """Leave-one-out top-1 neighbour for every row, streamed — no n x n held at once."""
    n = len(matrix)
    best = np.full(n, -1, np.int64)
    best_s = np.full(n, -2.0, np.float32)
    for s in range(0, n, block):
        blk = matrix[s:s + block] @ matrix.T          # block x n, freed each iteration
        for r in range(blk.shape[0]):
            blk[r, s + r] = -2.0
        idx = np.argmax(blk, axis=1)
        val = blk[np.arange(blk.shape[0]), idx]
        best[s:s + block] = idx
        best_s[s:s + block] = val
    return best, best_s


def main(out="data/custom/graphmatch.json"):
    lex = Lexicon()
    store, kept, _, _ = build_store(2000)
    shapes = [shape_of(t) for t in store]

    # X FIRST: the existing masked-text embedding's shape-match rate, before Y exists.
    text_vecs = np.array(embed([t["masked"] for t in store]), dtype=np.float32)
    t_best, _ = topk_blocked(text_vecs)
    x_match = sum(shapes[i] == shapes[int(t_best[i])] for i in range(kept))
    print(f"X, topic embedding : top-1 same plan shape {x_match}/{kept} "
          f"({100 * x_match / kept:.1f}%)")

    g_vecs = np.stack([signature(t["question"], lex) for t in store])
    g_best, _ = topk_blocked(g_vecs)
    y_match = sum(shapes[i] == shapes[int(g_best[i])] for i in range(kept))
    print(f"Y, graph signature : top-1 same plan shape {y_match}/{kept} "
          f"({100 * y_match / kept:.1f}%)")

    # The baseline both must beat: picking a neighbour at random.
    from collections import Counter
    counts = Counter(shapes)
    chance = sum(c * (c - 1) for c in counts.values()) / (kept * (kept - 1))
    print(f"chance             : {100 * chance:.1f}%")

    # Graph-to-graph conversion, counted where everything binds: query's asked lemma exists,
    # neighbour shares the shape, and the neighbour's plan instantiates on the query's
    # numbers to the query's own training answer.
    from template import instantiate
    from mapstore import run_plan
    convertible = converted = 0
    for i, t in enumerate(store):
        j = int(g_best[i])
        if shapes[i] != shapes[j] or store[j]["nvars"] != t["nvars"]:
            continue
        convertible += 1
        binding = {f"v{k + 1}": v for k, v in
                   enumerate(NUM.findall(norm(t["question"])))}
        got = run_plan(store[j]["steps"], binding)
        converted += got is not None and str(got) == t["answer"]
    print(f"\ngraph-to-graph conversion where the shape binds: {converted}/{convertible} "
          f"exact against the training answers")
    print("\nY against X is the whole question: does structure retrieve TYPE where topic")
    print("retrieved themes. Whatever survives here is what the ADAPT arm was missing.")
    Path(out).write_text(json.dumps({
        "store": kept, "shapes": len(counts), "x_topic_match": x_match,
        "y_graph_match": y_match, "chance": chance,
        "convertible": convertible, "converted": converted}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
