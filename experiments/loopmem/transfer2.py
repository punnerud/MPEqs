#!/usr/bin/env python3
"""Three things phase 143 left open: which layer, which domain, how big.

The correspondence worked — 54.3% against the student's own 37.7% at a thousand positions
— on a layer mapping I assumed (student layer l reads teacher layer round(l * 39/15)), a
test set drawn from the same interleaved corpus, and a matrix whose size nobody counted.
Each of those is a way the result could be less than it looks.

    WHICH LAYER   for every student layer, which teacher layer actually predicts it?
                  Measured by building the correspondence against each candidate and
                  scoring it, rather than assuming proportionality. If the curve is not
                  the diagonal, the assumed mapping was leaving transfer on the table.
    WHICH DOMAIN  the corpus interleaves four registers in fixed blocks — English prose,
                  Python, C, Norwegian. Train the correspondence on ONE and test on the
                  others: that is the deployment question, since a device does not read
                  the text the correspondence was fitted on.
    HOW BIG       entries in the matrix, and what pruning to the top few student experts
                  per teacher expert costs. A transfer that needs a bigger table than the
                  tables it replaces has not saved anything.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tokenprefetch import read_trace  # noqa: E402
from transfer import (BUDGET, CORPUS, STUDENT_MODEL, STUDENT_TRACE, TEACHER_MODEL,
                      TEACHER_TRACE, offsets, pieces)  # noqa: E402

REGISTERS = ("wikitext", "python", "c", "norwegian")
BLOCK = 4000


def register_bounds():
    """Replay fetch-corpus.sh's interleave to get each block's byte span and register."""
    src = [(name, (Path("data/corpus") / f).read_text(encoding="utf-8",
                                                      errors="replace"))
           for name, f in zip(REGISTERS, ("wikitext.txt", "code-python.txt",
                                          "code-c.txt", "norwegian.txt"))]
    spans, cursor, i = [], 0, 0
    while any(i * BLOCK < len(t) for _n, t in src):
        for name, t in src:
            s = t[i * BLOCK:(i + 1) * BLOCK]
            if s:
                nbytes = len(s.encode("utf-8"))
                spans.append((cursor, cursor + nbytes, name))
                cursor += nbytes + 2          # the "\n\n" join
        i += 1
    return spans


def register_of(spans, byte_offset):
    lo, hi = 0, len(spans) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        a, b, name = spans[mid]
        if byte_offset < a:
            hi = mid - 1
        elif byte_offset >= b:
            lo = mid + 1
        else:
            return name
    return None


def main(out="data/custom/transfer2.json"):
    s_meta, s_recs = read_trace(STUDENT_TRACE)
    t_meta, t_recs = read_trace(TEACHER_TRACE)
    s_toks, t_toks = pieces(STUDENT_MODEL), pieces(TEACHER_MODEL)
    s_start, t_start = offsets(s_toks), offsets(t_toks)
    from bisect import bisect_right
    t_limit = max(t for _l, t, _i in t_recs) + 1
    align = {}
    for sp in range(min(len(s_toks), max(t for _l, t, _i in s_recs) + 1)):
        tp = bisect_right(t_start, s_start[sp]) - 1
        if 0 <= tp < t_limit:
            align[sp] = tp

    s_by, t_by = defaultdict(dict), defaultdict(dict)
    for layer, pos, ids in s_recs:
        s_by[pos][layer] = ids
    for layer, pos, ids in t_recs:
        t_by[pos][layer] = ids
    Ls, Lt = s_meta["layers"], t_meta["layers"]

    spans = register_bounds()
    reg = {sp: register_of(spans, s_start[sp]) for sp in align}
    counts = Counter(reg.values())
    print(f"aligned {len(align)} positions; registers "
          f"{ {k: v for k, v in counts.most_common()} }\n")

    # ---- WHICH LAYER ---------------------------------------------------------
    cand = list(range(0, Lt, 4))
    fit = [p for p in sorted(align) if (p // 1024) % 2 == 0][:1024]
    val = [p for p in sorted(align) if (p // 1024) % 2 == 1][:512]
    M = defaultdict(Counter)
    for p in fit:
        for sl in range(Ls):
            sids = s_by[p].get(sl)
            if not sids:
                continue
            for tl in cand:
                tids = t_by.get(align[p], {}).get(tl)
                if tids:
                    for te in tids:
                        M[(sl, tl, te)].update(sids)
    best = {}
    curve = {}
    for sl in range(Ls):
        scores = {}
        for tl in cand:
            hit = tot = 0
            for p in val:
                sids = s_by[p].get(sl)
                tids = t_by.get(align[p], {}).get(tl)
                if not sids or not tids:
                    continue
                sc = Counter()
                for te in tids:
                    row = M.get((sl, tl, te))
                    if row:
                        n = sum(row.values())
                        for se, c in row.items():
                            sc[se] += c / n
                hit += len(set(sids) & {e for e, _s in sc.most_common(BUDGET)})
                tot += len(sids)
            scores[tl] = hit / tot if tot else 0.0
        best[sl] = max(scores, key=scores.get)
        curve[sl] = {str(k): round(v, 3) for k, v in scores.items()}
        assumed = round(sl * (Lt - 1) / (Ls - 1))
        print(f"student layer {sl:>2}: best teacher layer {best[sl]:>2} "
              f"({scores[best[sl]]:.1%}), proportional guess would be {assumed:>2} "
              f"({scores.get(min(cand, key=lambda c: abs(c - assumed)), 0):.1%})")

    # ---- WHICH DOMAIN --------------------------------------------------------
    print()
    dom = {}
    for train_reg in REGISTERS:
        tr = [p for p in sorted(align) if reg.get(p) == train_reg][:1500]
        if len(tr) < 300:
            continue
        Md, native, freq = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
        t_table = defaultdict(Counter)
        for layer, pos, ids in t_recs:
            t_table[(layer, t_toks[pos][0])].update(ids)
        for p in tr:
            for sl, sids in s_by[p].items():
                freq[sl].update(sids)
                native[(sl, s_toks[p][0])].update(sids)
                tids = t_by.get(align[p], {}).get(best[sl])
                if tids:
                    for te in tids:
                        Md[(sl, te)].update(sids)
        frank = {la: [e for e, _c in c.most_common()] for la, c in freq.items()}

        def pad(pre, layer):
            pre = list(pre)
            for e in frank.get(layer, []):
                if len(pre) >= BUDGET:
                    break
                if e not in pre:
                    pre.append(e)
            return set(pre)

        row = {}
        for test_reg in REGISTERS:
            te_pos = [p for p in sorted(align)
                      if reg.get(p) == test_reg and p not in set(tr)][:1000]
            if len(te_pos) < 200:
                continue
            hits = Counter()
            tot = 0
            for p in te_pos:
                for sl, sids in s_by[p].items():
                    idset = set(sids)
                    tot += len(sids)
                    nt = native.get((sl, s_toks[p][0]))
                    hits["native"] += len(idset & pad(
                        [e for e, _c in nt.most_common(BUDGET)] if nt else [], sl))
                    tt = t_table.get((best[sl], t_toks[align[p]][0]))
                    sc = Counter()
                    if tt:
                        n_t = sum(tt.values())
                        for te, ct in tt.items():
                            r2 = Md.get((sl, te))
                            if r2:
                                n2 = sum(r2.values())
                                for se, cs in r2.items():
                                    sc[se] += (ct / n_t) * (cs / n2)
                    hits["transfer"] += len(idset & pad(
                        [e for e, _s in sc.most_common(BUDGET)], sl))
            row[test_reg] = {k: round(v / tot, 3) for k, v in hits.items()}
        dom[train_reg] = row
        cells = "  ".join(f"{r}: n {v['native']:.0%} t {v['transfer']:.0%}"
                          for r, v in row.items())
        print(f"M fitted on {train_reg:<10} -> {cells}")

    # ---- HOW BIG -------------------------------------------------------------
    full = defaultdict(Counter)
    for p in fit:
        for sl, sids in s_by[p].items():
            tids = t_by.get(align[p], {}).get(best[sl])
            if tids:
                for te in tids:
                    full[(sl, te)].update(sids)
    entries = sum(len(c) for c in full.values())
    native_entries = sum(1 for p in fit for sl in s_by[p]) * 1
    print(f"\ncorrespondence: {len(full)} rows, {entries} entries "
          f"({entries * 4 / 1024:.0f} KB at 4 bytes each)")
    for keep in (1, 2, 4, 8):
        kept = sum(min(keep, len(c)) for c in full.values())
        print(f"  pruned to top {keep} per row: {kept} entries "
              f"({kept / entries:.0%} of full)")
    summary = {"layer_best": best, "layer_curve": curve, "domains": dom,
               "matrix_rows": len(full), "matrix_entries": entries,
               "registers": dict(counts)}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
