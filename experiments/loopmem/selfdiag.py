#!/usr/bin/env python3
"""The model reads its own failure data and writes the improvement list.

The goal as stated: the model improves through new tasks. The loop that implements it needs a
step nothing has tested yet — WHO decides what the next task is. The record's counters already
name the gaps with numbers attached (role binding at 0-1/60, compound units behind 18 false
refusals, staged coherence at 9 against 16, the fill-failure template named by provenance).
This measures whether the model, shown the same raw failure data, arrives at the same list —
because if it does, the loop can close without a human prioritising, and if it invents
plausible-sounding gaps the data does not support, then diagnosis stays the record's job and
the model only executes.

The model gets machine summaries of six result files, verbatim numbers, no interpretation.
It writes a prioritised list. Scoring is against the record's own gap list, by hand-checkable
matching, and the full text is stored so the judgement is auditable.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402

# The record's own gap list — each entry a measured deficiency with its number, written down
# BEFORE the model is asked, so the comparison cannot drift toward whatever it answers.
RECORD_GAPS = {
    "role_binding": "retrieved plans fail because numbers are bound by position, not role "
                    "(0/60, 1/60, 1/60 across three lookup arms)",
    "compound_units": "the unit reader gives one noun per number; wages are dollars PER HOUR "
                      "(18 of 23 remaining refusals false)",
    "staged_coherence": "span-at-a-time absorption fragments cross-sentence relations "
                        "(9/20 against 16/20 one-shot)",
    "thinking_budget": "AIME thoughts truncate at 1,600 tokens (4 of 6 never closed)",
    "template_fill": "one template's model-fill failed 4 of 7 uses, named by provenance",
}

DATA_FILES = ["mapstore.json", "dimcheck.json", "modelunits.json", "stagedabs.json",
              "aimethink.json", "template.json"]

DIAGNOSE = """Below are result summaries from experiments where a language model and a
bookkeeping system solve problems together. Read the numbers and write the FIVE most
important problems to fix next, ordered by expected gain, each with one sentence of
evidence FROM THE DATA below. Do not propose anything the data does not show.

{data}

Reply as a numbered list, nothing else.
"""


def compact(name):
    d = json.loads(Path(f"data/custom/{name}").read_text())
    if "rows" in d:
        d = {k: v for k, v in d.items() if k != "rows"}
    for k in ("summary",):
        if k in d and isinstance(d[k], dict):
            d.update(d.pop(k))
    return f"--- {name} ---\n{json.dumps(d)[:900]}"


def main(model="qwen-35b", out="data/custom/selfdiag.json"):
    data = "\n".join(compact(f) for f in DATA_FILES)
    reply = ask(model, DIAGNOSE.format(data=data), n=700)
    items = [l.strip() for l in reply.splitlines()
             if l.strip() and l.strip()[0].isdigit()][:5]

    print(f"{model}'s list, against the record's:\n")
    hits, matches = 0, []
    keywords = {
        "role_binding": ["bind", "role", "position", "map", "retriev"],
        "compound_units": ["compound", "per hour", "rate", "unit"],
        "staged_coherence": ["staged", "span", "coheren", "fragment", "one-shot", "absorb"],
        "thinking_budget": ["token", "budget", "truncat", "1600", "1,600", "think"],
        "template_fill": ["fill", "template", "proven"],
    }
    for i, item in enumerate(items):
        low = item.lower()
        hit = next((g for g, kws in keywords.items() if any(k in low for k in kws)), None)
        hits += hit is not None
        matches.append({"rank": i + 1, "text": item, "matches_gap": hit})
        print(f"  {i + 1}. {'[' + hit + ']' if hit else '[NOT IN DATA]'} {item[:110]}")

    covered = {m["matches_gap"] for m in matches if m["matches_gap"]}
    print(f"\n{hits}/{len(items)} of the model's items correspond to measured gaps;")
    print(f"{len(covered)}/{len(RECORD_GAPS)} of the record's gaps were found unprompted.")
    print("Keyword matching is a first pass — the stored texts are the auditable record, and")
    print("a plausible item marked NOT IN DATA is exactly the failure mode being tested for.")
    Path(out).write_text(json.dumps({
        "model": model, "items": matches, "hits": hits,
        "gaps_covered": sorted(covered), "record_gaps": RECORD_GAPS,
        "raw_reply": reply}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
