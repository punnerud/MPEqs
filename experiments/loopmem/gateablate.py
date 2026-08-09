#!/usr/bin/env python3
"""Which gate layer does the work, and what does each one cost?

The first spec-gate run delivered 3 of 30 on a battery where 29 mappings were right,
and 0 of 15 on AIME problems where every mapping was wrong. Both numbers are real, but
stacked together they cannot say WHICH layer bought the safety and which threw away the
score. This takes the recorded specs — no model call, the replies are cached — and runs
four gates over exactly the same data:

    none        deliver whatever road A produces        (phase 93's behaviour)
    echo        deliver unless a literal was invented
    agree       deliver only when two different machines produce the same number
    both        echo AND agree
    both+       both, with digit-scale constants counted as structure rather than data

Precision (of what was delivered, how much was right) against recall (of what was
right, how much was delivered) — for a delivery gate the asymmetry is the point: a
wrong delivery is a lie, a flag is a shrug, and the ablation prices both.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from solvemap import answer_of  # noqa: E402
from solvers2 import run2  # noqa: E402
from specgate import STRUCTURAL, STRUCTURAL_RELAXED, norm, value_echo  # noqa: E402

GATES = ("none", "echo", "agree", "both", "both+")


def judge(rows, truth_key="truth"):
    out = {g: {"delivered": 0, "right": 0, "wrong": 0, "flagged": 0} for g in GATES}
    for r in rows:
        sa, sb, story = r.get("spec_a"), r.get("spec_b"), r.get("story", "")
        truth = norm(str(r[truth_key]))
        ra = run2(sa)[0] if isinstance(sa, dict) else None
        rb = run2(sb)[0] if isinstance(sb, dict) else None
        na = norm(answer_of(ra)) if ra else None
        nb = norm(answer_of(rb)) if rb else None
        echo_bad = (bool(value_echo(sa, story, STRUCTURAL)) if isinstance(sa, dict)
                    else True)
        echo_bad_rel = (bool(value_echo(sa, story, STRUCTURAL_RELAXED))
                        if isinstance(sa, dict) else True)
        if isinstance(sb, dict):
            echo_bad = echo_bad or bool(value_echo(sb, story, STRUCTURAL))
            echo_bad_rel = echo_bad_rel or bool(
                value_echo(sb, story, STRUCTURAL_RELAXED))
        agree = na is not None and na == nb
        for g in GATES:
            ok_to_deliver = {
                "none": na is not None,
                "echo": na is not None and not echo_bad,
                "agree": agree,
                "both": agree and not echo_bad,
                "both+": agree and not echo_bad_rel,
            }[g]
            if not ok_to_deliver:
                out[g]["flagged"] += 1
                continue
            out[g]["delivered"] += 1
            out[g]["right" if na == truth else "wrong"] += 1
    return out


def show(title, table, n):
    print(f"\n{title} (n = {n})")
    print(f"{'gate':<8}{'delivered':>11}{'right':>7}{'WRONG':>7}{'flagged':>9}"
          f"{'precision':>11}{'recall':>9}")
    for g in GATES:
        r = table[g]
        prec = r["right"] / r["delivered"] if r["delivered"] else float("nan")
        rec = r["right"] / n
        print(f"{g:<8}{r['delivered']:>11}{r['right']:>7}{r['wrong']:>7}"
              f"{r['flagged']:>9}{prec:>11.2f}{rec:>9.2f}")


def main(src="data/custom/specgate.json", out="data/custom/gateablate.json"):
    d = json.loads(Path(src).read_text())
    easy = judge(d["rows"])
    hard = judge(d["aime"]["rows"])
    show("BATTERY — mappings mostly right", easy, len(d["rows"]))
    show("AIME — every mapping measured wrong in phase 94", hard,
         len(d["aime"]["rows"]))
    print("\nA wrong delivery is a lie and a flag is a shrug, so the columns are not")
    print("symmetric: the gate to keep is the one that holds WRONG at zero on the hard")
    print("set while giving back the most of the easy set. Whichever row that is, it")
    print("is now a measurement over identical data rather than a preference.")
    summary = {"battery": easy, "aime": hard, "n_battery": len(d["rows"]),
               "n_aime": len(d["aime"]["rows"])}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
