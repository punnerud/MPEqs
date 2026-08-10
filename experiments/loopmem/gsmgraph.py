#!/usr/bin/env python3
"""The external control: does chaining help on problems I did not write?

Phase 136 measured the graph executor beating a one-spec pipeline five to nothing on the
practical half of a battery I wrote myself, which is exactly the self-confirmation the
plan flagged. GSM8K is the honest check: multi-step by construction, written by strangers,
and already measured twice — the model answering alone got 28 of 30 and the one-spec
pipeline 21 of 30 (phase 98).

Same thirty problems, same seed, same catalogue. Only the arm changes: the model writes a
system of definitions where an edge may be a machine, and the record runs it topologically.
If chaining is worth anything outside my own wording, this is where it shows.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gsmsolve import ask_spec_model, equal  # noqa: E402
from islands import solve_graph  # noqa: E402
from olympiad import load_problems  # noqa: E402
from solve import catalogue  # noqa: E402
from twostep import GRAPH_PROMPT  # noqa: E402


def main(n_problems=30, seed=3, out="data/custom/gsmgraph.json"):
    n_problems, seed = int(n_problems), int(seed)
    gsm, _ = load_problems()
    picks = random.Random(seed).sample(gsm, n_problems)
    cat = catalogue()
    t = {"parsed": 0, "ran": 0, "exact": 0, "wrong": 0, "steps": 0, "multi": 0}
    rows = []
    for story, truth in picks:
        reply = ask_spec_model("qwen-35b", GRAPH_PROMPT.format(story=story,
                                                               catalogue=cat), n=700)
        got, why, info = None, "no system", {}
        m = re.search(r"\{.*\}", reply, re.S)
        if m:
            try:
                sysd = json.loads(m.group(0))
                t["parsed"] += 1
                got, why, info = solve_graph(sysd.get("defs", {}),
                                             str(sysd.get("asked", "")))
            except json.JSONDecodeError:
                why = "malformed system"
        ok = got is not None and equal(got, truth)
        if got is not None:
            t["ran"] += 1
            t["exact" if ok else "wrong"] += 1
            t["steps"] += info.get("steps", 0) or 0
            t["multi"] += (info.get("steps", 0) or 0) > 1
        rows.append({"truth": str(truth), "got": str(got), "ok": bool(ok),
                     "steps": info.get("steps"), "why": why})
        print(f"truth {str(truth):>8}  graph {str(got)[:14]:<16}"
              f"{info.get('steps', '-')} steps  {'ok' if ok else why[:34]}")

    n = len(picks)
    print(f"\nGRAPH arm      : {t['exact']}/{n}  (parsed {t['parsed']}, ran {t['ran']}, "
          f"wrong {t['wrong']}, {t['multi']} used more than one step)")
    print(f"one-spec arm   : 21/30   (phase 98, same problems, same seed)")
    print(f"the model alone: 28/30   (phase 98)")
    print("\nThis is the arm's only test on wording nobody here chose. Whatever it says")
    print("about chaining is worth more than the battery I wrote to show chaining off.")
    Path(out).write_text(json.dumps({"n": n, **t, "oneshot_reference": 21,
                                     "solo_reference": 28, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
