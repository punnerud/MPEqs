#!/usr/bin/env python3
"""Can the routing tables transfer from a big model to a small one?

Not directly: expert 17 in Qwen-35B and expert 17 in OLMoE-1B are unrelated machines in
unrelated spaces, and the two tokenisers do not even agree on what a word is. What CAN
transfer, and what this measures, is the knowledge route:

    teacher tables      the big model's per-word routing tables, built offline from its
                        own trace of the same corpus
    A LEARNED CORRESPONDENCE  M[student_layer][teacher_expert] -> student experts,
                        trained on N positions where both models read the same text —
                        aligned by CHARACTER OFFSET, since the tokenisers differ
    transferred tables  teacher table composed with M: a per-word table in the
                        student's expert space that the student barely had to run for

The deployment story this serves: the small model lives on the device; the big one runs
once, offline, over reference text. If M is learnable from far fewer student positions
than the student's own tables need, the big model's knowledge arrives cheaply. So the
measurement is a sweep: at N = 256, 1024, 4096 student training positions, native tables
against transferred ones, on the same fixed student test set as phase 142.

Alongside it, the structural question: do the two models even agree on WHICH words are
routing-predictable? Per-token self-overlap in the student against the teacher, for
tokens aligned across the vocabularies.
"""
import json
import re
import subprocess
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tokenprefetch import read_trace  # noqa: E402

STUDENT_TRACE = Path("data/trace-big.bin")          # OLMoE, 65k positions
TEACHER_TRACE = Path("data/trace-qwen8k.bin")       # Qwen-35B, 8k positions
STUDENT_MODEL = Path("models/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf")
TEACHER_MODEL = Path("models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf")
CORPUS = Path("data/corpus/corpus.txt")
BUDGET = 16
LINE = re.compile(r"^\s*(\d+) -> '(.*)'$", re.S)


def pieces(model):
    """(token id, piece) per position; cumulative piece length gives BYTE offsets.

    Byte-level BPE means a piece can be invalid UTF-8 on its own and can contain a
    literal newline, so the output is parsed as BYTES with a lookahead for the next
    record, and pieces are decoded latin-1 — lossless byte-to-char, so len(piece) is
    exactly the byte width and offsets agree across both models' tokenisers."""
    out = subprocess.run(["llama-tokenize", "-m", str(model), "-f", str(CORPUS)],
                         capture_output=True).stdout
    rec = re.compile(rb"(?m)^\s*(\d+) -> '(.*?)'(?=\n(?:\s*\d+ -> '|\Z)|\Z)",
                     re.S)
    return [(int(m.group(1)), m.group(2).decode("latin-1"))
            for m in rec.finditer(out)]


def offsets(toks):
    """Character start per position; leading specials whose pieces are not corpus text
    contribute length anyway, which the alignment check below would expose."""
    starts, c = [], 0
    for _i, piece in toks:
        starts.append(c)
        c += len(piece)
    return starts


def main(out="data/custom/transfer.json"):
    s_meta, s_recs = read_trace(STUDENT_TRACE)
    t_meta, t_recs = read_trace(TEACHER_TRACE)
    print(f"student {s_meta['layers']}L x {s_meta['experts']}E top-{s_meta['top_k']}, "
          f"teacher {t_meta['layers']}L x {t_meta['experts']}E top-{t_meta['top_k']}")

    s_toks = pieces(STUDENT_MODEL)
    t_toks = pieces(TEACHER_MODEL)
    s_start = offsets(s_toks)
    t_start = offsets(t_toks)
    # align: student position -> teacher position covering the same character
    t_limit = max(t for _l, t, _i in t_recs) + 1
    align = {}
    for sp in range(min(len(s_toks), max(t for _l, t, _i in s_recs) + 1)):
        tp = bisect_right(t_start, s_start[sp]) - 1
        if 0 <= tp < t_limit:
            align[sp] = tp
    print(f"aligned {len(align)} student positions to teacher positions by character")

    Ls, Lt = s_meta["layers"], t_meta["layers"]
    lmap = {l: round(l * (Lt - 1) / (Ls - 1)) for l in range(Ls)}

    s_by = defaultdict(dict)
    for layer, pos, ids in s_recs:
        s_by[pos][layer] = ids
    t_by = defaultdict(dict)
    for layer, pos, ids in t_recs:
        t_by[pos][layer] = ids

    # teacher word tables from its ENTIRE trace: offline is where the teacher is cheap
    t_table = defaultdict(Counter)
    for layer, pos, ids in t_recs:
        t_table[(layer, t_toks[pos][0])].update(ids)

    def is_test(pos):
        return pos < 8192 and (pos // 1024) % 2 == 1

    test = [(l, p, ids) for l, p, ids in s_recs if is_test(p) and p in align]
    train_pool = [p for p in sorted(align) if p < 8192 and (p // 1024) % 2 == 0]

    results = {}
    for n_train in (256, 1024, 4096):
        train = train_pool[:n_train]
        freq = defaultdict(Counter)
        native = defaultdict(Counter)
        M = defaultdict(Counter)            # (student layer, teacher expert) -> student
        for p in train:
            for layer, ids in s_by[p].items():
                freq[layer].update(ids)
                native[(layer, s_toks[p][0])].update(ids)
                t_ids = t_by.get(align[p], {}).get(lmap[layer])
                if t_ids:
                    for te in t_ids:
                        M[(layer, te)].update(ids)
        frank = {la: [e for e, _c in c.most_common()] for la, c in freq.items()}

        def pad(pre, layer):
            pre = list(pre)
            for e in frank.get(layer, []):
                if len(pre) >= BUDGET:
                    break
                if e not in pre:
                    pre.append(e)
            return set(pre)

        hits = Counter()
        tot = 0
        for layer, pos, ids in test:
            idset = set(ids)
            tot += len(ids)
            hits["freq"] += len(idset & pad([], layer))
            nt = native.get((layer, s_toks[pos][0]))
            hits["native"] += len(idset & pad(
                [e for e, _c in nt.most_common(BUDGET)] if nt else [], layer))
            # transferred: teacher's table for the teacher word here, through M
            tp = align[pos]
            tl = lmap[layer]
            tt = t_table.get((tl, t_toks[tp][0]))
            score = Counter()
            if tt:
                total_t = sum(tt.values())
                for te, ct in tt.items():
                    row = M.get((layer, te))
                    if row:
                        rt = sum(row.values())
                        for se, cs in row.items():
                            score[se] += (ct / total_t) * (cs / rt)
            hits["transfer"] += len(idset & pad(
                [e for e, _s in score.most_common(BUDGET)], layer))
            both = [e for e, _s in score.most_common(BUDGET // 2)] + \
                ([e for e, _c in nt.most_common(BUDGET)] if nt else [])
            hits["both"] += len(idset & pad(both, layer))
        results[n_train] = {k: round(v / tot, 4) for k, v in hits.items()}
        r = results[n_train]
        print(f"N={n_train:<6} freq {r['freq']:.1%}  native {r['native']:.1%}  "
              f"transfer {r['transfer']:.1%}  both {r['both']:.1%}")

    # the structural question: is predictability the same words in both models?
    def overlaps(by, toks, layer, positions):
        groups = defaultdict(list)
        for p in positions:
            ids = by.get(p, {}).get(layer)
            if ids:
                groups[toks[p][0]].append(frozenset(ids))
        out = {}
        for tok, sets in groups.items():
            if len(sets) >= 3:
                pairs = [(a, b) for i, a in enumerate(sets[:6])
                         for b in sets[i + 1:6]]
                if pairs:
                    out[tok] = sum(len(a & b) / len(a | b) for a, b in pairs) / \
                        len(pairs)
        return out

    common = [p for p in align if p < t_limit]
    s_ov = overlaps(s_by, s_toks, Ls // 2, common)
    t_ov = overlaps(t_by, t_toks, Lt // 2, [align[p] for p in common])
    paired = [(s_ov[s_toks[p][0]], t_ov[t_toks[align[p]][0]])
              for p in common
              if s_toks[p][0] in s_ov and t_toks[align[p]][0] in t_ov]
    paired = list({(round(a, 4), round(b, 4)) for a, b in paired})
    if len(paired) > 4:
        xs, ys = zip(*paired)
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in paired)
        vx = sum((x - mx) ** 2 for x in xs) ** 0.5
        vy = sum((y - my) ** 2 for y in ys) ** 0.5
        corr = cov / (vx * vy) if vx and vy else 0.0
    else:
        corr = 0.0
    print(f"\npredictability correlation across models "
          f"({len(paired)} word pairs): r = {corr:.2f}")
    print("\nTables cannot cross expert spaces; knowledge can, if the correspondence is")
    print("cheaper to learn than the tables it replaces. The sweep above is that price,")
    print("and the correlation is whether the two models even agree on which words are")
    print("worth a table at all.")
    summary = {"student": {k: s_meta[k] for k in ("layers", "experts", "top_k")},
               "teacher": {k: t_meta[k] for k in ("layers", "experts", "top_k")},
               "aligned": len(align),
               "sweep": {str(k): v for k, v in results.items()},
               "predictability_r": round(corr, 3), "word_pairs": len(paired)}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
