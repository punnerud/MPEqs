#!/usr/bin/env python3
"""Repair via provenance: when a template fails on a new case, find the one covering both.

Phase 45's round-trip rule refuses a generalisation whose variable assignment it can see is
wrong. But duplicate numbers can HIDE a wrong assignment: `(6 + 2 * 6) / 3` round-trips
perfectly whichever six became which variable, because on the source the two choices are
indistinguishable. The defect is latent, and it surfaces on the first new instance whose
numbers differ — `(5 + 2 * 7) / 3` instantiated through the wrong assignment inlines to
`(7 + 2 * 5) / 3`, a different value, and the record's inline check catches it without any
model or any truth.

That is the repair scenario as proposed: the new case failing means the generalisation was
wrong, and the fix is a COMMON template covering the old examples and the new one. Concretely:

    enumerate every assignment consistent with each provenance example (exact round trip),
    intersect across examples, and keep the one that also verifies on the new expression.

The old examples are what make this possible — with only the source, the ambiguity is
unresolvable by construction; each new discriminating instance eliminates candidates. So
provenance is not bookkeeping, it is the repair material.

No model is involved: the defect, the detection and the repair are all properties of the
record's memory, and phase 45 already measured the model's part. Expressions are generated
with a forced duplicate so the latent-defect path is exercised often instead of rarely.
"""
import json
import random
import re
import sys
from itertools import permutations, product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from template import NUM, VARS, instantiate, skeleton  # noqa: E402


def gen_dup_expr(rng):
    """A four-term expression in which one number deliberately appears twice."""
    while True:
        a, b, c = rng.randint(2, 40), rng.randint(2, 20), rng.randint(2, 12)
        dup = rng.choice(["ab", "ac", "ad", "bd", "cd"])
        op1, op2 = rng.choice("+-"), rng.choice("*+")
        d = rng.choice([a, b, c]) if "d" in dup else rng.randint(2, 15)
        if dup == "ab":
            b = a
        elif dup == "ac":
            c = a
        inner = f"{a} {op1} {b} {op2} {c}"
        val = eval(inner)  # noqa: S307 - our own generated arithmetic
        if d and val % d == 0 and abs(val // d) > 1:
            return f"({inner}) / {d}", val // d


def canonical_graph(expr):
    """The decomposition the models write for this shape (Qwen: 20/20 in this exact form)."""
    m = re.match(r"\((\d+) ([+-]) (\d+) ([*+]) (\d+)\) / (\d+)", expr)
    a, op1, b, op2, c, d = m.groups()
    if op2 == "*":
        return {"A": f"{b} * {c}", "B": f"{a} {op1} A", "C": "B / " + d}
    return {"A": f"{a} {op1} {b}", "B": f"A + {c}", "C": "B / " + d}


def all_templates(expr, graph):
    """EVERY variable assignment whose round trip reproduces (expr, graph) exactly.

    Where phase 45's generaliser guessed greedily and checked once, this enumerates: for each
    duplicated value, each way its expression slots can map onto its graph occurrences. One
    template per surviving assignment, deduplicated. More than one survivor means the source
    alone cannot decide — which is the latent defect, now explicit instead of hidden.
    """
    values = NUM.findall(expr)
    texpr, pos = "", 0
    for k, m in enumerate(NUM.finditer(expr)):
        texpr += expr[pos:m.start()] + VARS[k]
        pos = m.end()
    texpr += expr[pos:]

    occ = []                                    # (key, span, value) per number in the graph
    for key, body in graph.items():
        occ += [(key, m.span(), m.group(0)) for m in NUM.finditer(body)]
    slots = {v: [k for k, x in enumerate(values) if x == v] for v in set(values)}
    by_val = {v: [i for i, o in enumerate(occ) if o[2] == v] for v in set(values)}

    choices = []
    for v, occs in by_val.items():
        if len(slots[v]) < len(occs):
            return texpr, []                    # a number the expression cannot supply
        choices.append([dict(zip(occs, p)) for p in permutations(slots[v], len(occs))])

    seen, out = set(), []
    for combo in product(*choices):
        mapping = {k: v for d_ in combo for k, v in d_.items()}
        tgraph = {}
        for key, body in graph.items():
            parts, pos2 = [], 0
            for i, (k2, span, _) in enumerate(occ):
                if k2 != key:
                    continue
                parts.append(body[pos2:span[0]] + VARS[mapping[i]])
                pos2 = span[1]
            tgraph[key] = "".join(parts) + body[pos2:]
        tpl = {"skeleton": skeleton(expr), "expr": texpr, "graph": tgraph,
               "nvars": len(values)}
        be, bg = instantiate(tpl, values)
        if be == expr and bg == graph:
            frozen = json.dumps(tgraph, sort_keys=True)
            if frozen not in seen:
                seen.add(frozen)
                out.append(tpl)
    return texpr, out


def inlines_ok(graph, expr):
    """The record's own check: the instantiated graph must reproduce the expression's value."""
    inlined = graph[list(graph)[-1]]
    for key in reversed(list(graph)[:-1]):
        inlined = re.sub(rf"\b{key}\b", f"({graph[key]})", inlined)
    try:
        return abs(eval(inlined) - eval(expr)) < 1e-9  # noqa: S307
    except Exception:  # noqa: BLE001
        return False


def repair(entry, new_expr):
    """The common template: consistent with every provenance example AND the new case."""
    survivors = None
    for src_expr, src_graph in entry["sources"]:
        _, cands = all_templates(src_expr, src_graph)
        keep = {json.dumps(t["graph"], sort_keys=True): t for t in cands}
        survivors = keep if survivors is None else \
            {k: v for k, v in survivors.items() if k in keep}
    fixed = [t for t in (survivors or {}).values()
             if inlines_ok(instantiate(t, NUM.findall(new_expr))[1], new_expr)]
    return fixed[0] if fixed else None, len(survivors or {}), len(fixed)


def main(n_tasks=80, seed=7, out="data/custom/repair.json"):
    n_tasks, seed = int(n_tasks), int(seed)
    rng = random.Random(seed)
    tasks = [gen_dup_expr(rng) for _ in range(n_tasks)]

    store, stats = {}, {"hits": 0, "covered": 0, "defects": 0, "repaired": 0,
                        "unrepairable": 0, "ambiguous_stored": 0}
    events = []
    for expr, truth in tasks:
        key = skeleton(expr)
        entry = store.get(key)
        if entry is None:
            g = canonical_graph(expr)
            texpr, cands = all_templates(expr, g)
            if not cands:
                continue
            # Greedy-first, exactly as phase 45 stored it, so the latent defect is preserved
            # rather than dodged — the ambiguity count is recorded, not resolved early.
            store[key] = {"tpl": cands[0], "sources": [(expr, g)],
                          "ambiguous": len(cands) > 1, "candidates_at_store": len(cands)}
            stats["ambiguous_stored"] += len(cands) > 1
            continue
        stats["hits"] += 1
        _, inst = instantiate(entry["tpl"], NUM.findall(expr))
        if inlines_ok(inst, expr):
            stats["covered"] += 1
            entry["sources"].append((expr, canonical_graph(expr)))
            continue
        # Detected: the stored generalisation is wrong in a way its sources could not show.
        stats["defects"] += 1
        fixed, n_surv, n_fit = repair(entry, expr)
        ev = {"expr": expr, "skeleton": key, "survivors": n_surv, "fit_new": n_fit,
              "sources_used": len(entry["sources"])}
        if fixed is None:
            stats["unrepairable"] += 1
            ev["repaired"] = False
        else:
            # The common template must still cover every old example — re-verified, not
            # trusted, because that is the entire promise being made.
            old_ok = all(instantiate(fixed, NUM.findall(e))[1] == g
                         for e, g in entry["sources"])
            new_ok = inlines_ok(instantiate(fixed, NUM.findall(expr))[1], expr)
            entry["tpl"] = fixed
            entry["sources"].append((expr, canonical_graph(expr)))
            stats["repaired"] += old_ok and new_ok
            ev.update({"repaired": True, "covers_all_old": old_ok, "covers_new": new_ok})
            stats["covered"] += old_ok and new_ok
        events.append(ev)

    print(f"{n_tasks} expressions, every one carrying a duplicated number\n")
    print(f"templates stored              : {len(store)}")
    print(f"  stored while ambiguous      : {stats['ambiguous_stored']} "
          f"(the source alone could not decide the assignment)")
    print(f"template hits                 : {stats['hits']}")
    print(f"  instantiated correctly      : {stats['covered']}")
    print(f"  latent defects detected     : {stats['defects']} "
          f"(caught by the inline check, no truth consulted)")
    print(f"  repaired via provenance     : {stats['repaired']}/{stats['defects']}")
    print(f"  unrepairable                : {stats['unrepairable']}")
    for ev in events[:5]:
        print(f"    {ev['expr']:<24} candidates {ev['survivors']}, fitting old+new "
              f"{ev['fit_new']}, repaired {ev.get('repaired')}")
    print("\nThe repair is exactly the proposal: the provenance examples plus the failing case")
    print("jointly pin down the assignment neither could alone. Every repaired template is")
    print("re-verified against ALL its old examples and the new one — covering both is the")
    print("claim, so it is checked, not assumed.")
    Path(out).write_text(json.dumps({"tasks": n_tasks, **stats,
                                     "templates": len(store), "events": events}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
