#!/usr/bin/env python3
"""MPEqs' capability sheet, assembled from the measurements rather than from memory.

Every band result lives in its own JSON under data/custom, written by the run that
produced it. This reads them all and prints one table, so the claim "MPEqs is good at X"
always resolves to a number somebody measured, and a stale claim shows up as a missing
file rather than as prose that no longer matches.

No model calls, no recomputation: this is a report, and its only job is to be unable to
lie about what was run.
"""
import json
import sys
from pathlib import Path

ROWS = [
    ("Grade-school word problems", "gsmsolve.json", "solo", "m35_exact", 30,
     "the model already does these sums"),
    ("Hard exact arithmetic (tuned)", "hardarith.json", "solo", "mpeqs", 20,
     "folds, big integers, wide counts, factorial divisors"),
    ("Hard exact arithmetic (held out)", "hardarith_heldout.json", "solo", "mpeqs", 20,
     "same families, new numbers, no further tuning"),
    ("Probability and unit conversion", "newbands.json", "solo", "mpeqs", 20,
     "exact ratios the model cannot hold"),
    ("Statistics, calendars, percentages", "bands2.json", "solo", "mpeqs", 18,
     "calendar arithmetic is the sharpest split"),
    ("Shape, sets, formulas (easy)", "bands3.json", "solo", "mpeqs", 15,
     "a tie: no band without difficulty"),
    ("Shape, sets, formulas (hard)", "bands3_hard.json", "solo", "mpeqs", 15,
     "same classes, awkward numbers"),
    ("Nine classes mixed, generic examples", "mixedband.json", "solo", "mpeqs", 27,
     "two classes vanish without their exemplar"),
    ("Nine classes mixed, retrieved", "mixedretr.json", "solo", "mpeqs", 27,
     "retrieval closes the gap, nothing wrong"),
    ("The same 27, driven by a 1B", "mixedretr_1b.json", "solo", "mpeqs", 27,
     "27 of 27 specs valid, arithmetic never its job"),
    ("Fresh 24, the full policy live", "pipeline.json", "solo", "pipeline", 24,
     "23 right, zero wrong, one refusal"),
    ("AIME, model-mapped", "aimefewshot.json", None, "exact", 15,
     "the mapper is the bottleneck"),
    ("AIME, hand-mapped ceiling", "aimeceiling.json", None, "hand_exact", 15,
     "what the vocabulary reaches"),
]


def main(root="data/custom", out="data/custom/capability.json"):
    root = Path(root)
    print(f"{'band':<38}{'model':>7}{'MPEqs':>7}{'n':>5}   note")
    print("-" * 100)
    total_solo = total_mp = total_n = 0
    table = []
    for label, fname, solo_key, mp_key, n, note in ROWS:
        path = root / fname
        if not path.exists():
            print(f"{label:<38}{'--':>7}{'--':>7}{n:>5}   MISSING {fname}")
            continue
        d = json.loads(path.read_text())
        solo = d.get(solo_key) if solo_key else None
        mp = d.get(mp_key, 0)
        table.append({"band": label, "solo": solo, "mpeqs": mp, "n": n, "note": note})
        if solo is not None:
            total_solo += solo
            total_mp += mp
            total_n += n
        print(f"{label:<38}{('-' if solo is None else solo):>7}{mp:>7}{n:>5}   {note}")
    print("-" * 100)
    print(f"{'comparable totals (arms measured on both)':<38}{total_solo:>7}"
          f"{total_mp:>7}{total_n:>5}")

    lib = json.loads((root / "solvers2.json").read_text())
    print(f"\nlibrary: {lib['solvers_total']} solvers, {lib['passed']} self-tests "
          f"exact, {lib['refusals_named']} refusals named")
    print("recipe, each part measured before being combined:")
    print("  a machine that is exact and refuses by name")
    print("  a schema line that states its units and leads with its distinction")
    print("  an exemplar that is retrieved, because a catalogue cannot grow in a prompt")
    print("  and a router that asks whether the arithmetic fits in the model's head")
    summary = {"rows": table, "total_solo": total_solo, "total_mpeqs": total_mp,
               "total_n": total_n, "solvers": lib["solvers_total"]}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
