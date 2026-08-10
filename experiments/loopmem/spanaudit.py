#!/usr/bin/env python3
"""The graph audits the spans: two independent readings, cross-checked mechanically.

Phase 133 ended with both representations on the table and a tie between them — span
labelling reads 14 of 21 quantity roles, the per-token graph reads 13 — and one asymmetry:
the graph can expose itself. A cycle is a fact about the structure, checkable with no truth
to compare against. Spans have no equivalent; a mislabelled chunk looks exactly like a
correct one.

So the graph becomes the auditor. Four cross-checks, all mechanical, none needing an answer
key at run time:

    RELATION vs EDGE      a span labelled "divisibility" must contain a token the graph
                          marks "divisor"; remainder against modulus; bound against bound
    QUANTITY vs POSITION  the role the spans give a number must match the role its graph
                          position gives it
    ACTION vs ROOT        the action's span must contain the graph's root or its target
    STRUCTURE             exactly one root, no cycles, every edge on a real token

The thing being measured is NOT how often the two readings differ — it is whether the
disagreements are the WRONG ones. Phase 96's law applies at full force: a gate that flags
everything is not a gate, so the catch rate on real errors and the false-flag rate on
correct readings are reported side by side, both against the hand-written truths that
already exist in the phase 133 battery.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from grounded import audit as span_audit  # noqa: E402
from grounded import ground  # noqa: E402
from wordgraph import BATTERY, LABELS, POSITIONAL  # noqa: E402
from wordgraph import audit as graph_audit  # noqa: E402
from wordgraph import build as graph_build  # noqa: E402

# Which graph edge label a span's relation claim implies. A relation with no entry here
# makes no claim about edges and is never flagged on that ground.
RELATION_EDGE = {
    "divisibility": {"divisor"},
    "remainder": {"modulus", "divisor"},
    "inequality": {"bound"},
    "cardinality": {"quantity", "object"},
    "primality": {"predicate", "modifier"},
    "distinctness": {"predicate", "modifier"},
    "digit_property": {"predicate", "modifier", "quantity"},
}
# How a span-side quantity role maps onto the graph's positional vocabulary.
SPAN_ROLE = {"lower_bound": "bound", "upper_bound": "bound", "bound": "bound",
             "divisor": "divisor", "modulus": "modulus", "remainder": "quantity",
             "digit_sum": "quantity", "count": "quantity", "value": "quantity"}


def cross_check(toks, span_sig, span_res, edges, graph_res):
    """Every flag names its rule and its token, because a flag that cannot be located
    is an opinion."""
    flags = []
    if not span_sig or not span_res.get("parsed"):
        return [{"rule": "spans", "why": "no signature"}], {}
    if not edges or not graph_res.get("parsed"):
        return [{"rule": "graph", "why": "no graph"}], {}

    by_index = {}
    for e in edges:
        i, lab = e.get("i"), str(e.get("label", ""))
        if isinstance(i, int) and 0 <= i < len(toks) and lab in LABELS:
            by_index[i] = (e.get("head"), lab)

    # 1. relation against edge
    for rel in span_sig.get("relations") or []:
        if not isinstance(rel, dict):
            continue
        want = RELATION_EDGE.get(str(rel.get("value")))
        span = rel.get("span")
        if not want or not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        a, b = span
        if not (isinstance(a, int) and isinstance(b, int)):
            continue
        inside = {by_index.get(i, (None, ""))[1] for i in range(a, min(b, len(toks) - 1) + 1)}
        if not (inside & want):
            flags.append({"rule": "relation vs edge", "relation": rel.get("value"),
                          "span": [a, b], "graph_says": sorted(x for x in inside if x),
                          "expected": sorted(want)})

    # 2. quantity role against graph position
    graph_roles = graph_res.get("derived", {})
    for q in span_res.get("quantities", []):
        span_role = SPAN_ROLE.get(q["role"], "quantity")
        g = graph_roles.get(q["value"], "")
        g_role = g if g in POSITIONAL.values() else ("quantity" if g else "")
        if g_role and g_role != span_role:
            flags.append({"rule": "quantity vs position", "number": q["value"],
                          "span_role": span_role, "graph_role": g_role,
                          "index": q["index"]})

    # 3. action against root or target
    act = span_sig.get("action")
    if isinstance(act, dict) and isinstance(act.get("span"), (list, tuple)):
        a, b = act["span"][0], act["span"][1]
        if isinstance(a, int) and isinstance(b, int):
            labs = {by_index.get(i, (None, ""))[1] for i in range(a, min(b, len(toks) - 1) + 1)}
            heads = {by_index.get(i, (None, ""))[0] for i in range(a, min(b, len(toks) - 1) + 1)}
            if not ({"root", "target", "predicate", "subject"} & labs) and -1 not in heads:
                flags.append({"rule": "action vs root", "span": [a, b],
                              "graph_says": sorted(x for x in labs if x)})

    # 4. structure — the checks only the graph can make about itself
    if graph_res.get("roots") != 1:
        flags.append({"rule": "structure", "why": f"{graph_res.get('roots')} roots"})
    if graph_res.get("cyclic"):
        flags.append({"rule": "structure", "why": f"{graph_res['cyclic']} cyclic tokens"})
    if graph_res.get("malformed_edges"):
        flags.append({"rule": "structure",
                      "why": f"{graph_res['malformed_edges']} edges off the token list"})
    return flags, graph_roles


def main(out="data/custom/spanaudit.json"):
    tally = {"problems": 0, "flagged": 0, "quantities": 0,
             "span_wrong": 0, "span_wrong_flagged": 0,
             "span_right": 0, "span_right_flagged": 0, "flags": 0}
    rows = []
    for lang, story, truth in BATTERY:
        toks, span_sig = ground(story)
        sres = span_audit(toks, span_sig)
        _t2, edges = graph_build(story)
        gres = graph_audit(toks, edges)
        flags, graph_roles = cross_check(toks, span_sig, sres, edges, gres)

        # Scoring needs to know which span readings were actually wrong, which the
        # phase 133 hand truths already say, per number.
        flagged_numbers = {f.get("number") for f in flags if f.get("number")}
        for q in sres.get("quantities", []):
            want = truth.get(q["value"])
            if want is None:
                continue
            tally["quantities"] += 1
            got = SPAN_ROLE.get(q["role"], "quantity")
            was_wrong = got != want
            tally["span_wrong" if was_wrong else "span_right"] += 1
            if q["value"] in flagged_numbers:
                tally["span_wrong_flagged" if was_wrong else "span_right_flagged"] += 1

        tally["problems"] += 1
        tally["flagged"] += bool(flags)
        tally["flags"] += len(flags)
        rows.append({"lang": lang, "story": story[:56], "flags": flags,
                     "graph_roles": graph_roles,
                     "span_quantities": sres.get("quantities", [])})
        rules = sorted({f["rule"] for f in flags})
        print(f"{lang} {len(flags)} flag(s) {rules if rules else ''}  {story[:44]}")

    caught = tally["span_wrong_flagged"]
    wrong = tally["span_wrong"]
    false_flags = tally["span_right_flagged"]
    right = tally["span_right"]
    print(f"\nproblems audited            : {tally['problems']}, "
          f"{tally['flagged']} carried at least one flag ({tally['flags']} in total)")
    print(f"span quantity roles scored  : {tally['quantities']} "
          f"({wrong} genuinely wrong, {right} right)")
    print(f"CATCH  wrong roles flagged  : {caught}/{wrong}")
    print(f"COST   right roles flagged  : {false_flags}/{right}")
    print("\nA second reading is only worth having if its disagreements are the right")
    print("ones. Both columns are above, and the one that decides whether this is a")
    print("gate or a nuisance is the second.")
    summary = {**tally, "catch": caught, "catch_of": wrong,
               "false_flags": false_flags, "false_of": right, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
