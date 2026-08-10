#!/usr/bin/env python3
"""The spec-level delivery gate: value echo, and two different machines agreeing.

Phase 93 found the hole by measurement: a mis-chosen solver still fills, still runs, and
still answers — zero specs were refused, so nothing between the model's choice and the
delivered number was mechanical. This builds that missing gate out of two ingredients
this study has already proved separately:

  VALUE ECHO (phase 91)  every literal in the spec must appear in the problem text, or
                         be a permitted structural constant. A hallucinated bound or an
                         invented modulus dies here with no model call at all.

  TWO MACHINES (phase 74) the model gives a SECOND spec that must use a DIFFERENT solver,
                         and the record runs both. Phase 89 showed agreement between two
                         PHRASINGS is correlated failure — but these are not two readings,
                         they are two computations: a closed form against an enumeration,
                         a system against a search. When they agree, two different
                         machines produced the same number from the same text.

Answers are compared after shape normalisation, because phase 93's one FREE "error" was
a correct answer in a list — [13] against 13 — and a gate that cannot see through
formatting is measuring itself, not the mapping.

The gate is also tested on TRUE NEGATIVES the record manufactures: for every problem,
a deliberately mis-mapped spec (the right numbers, the wrong machine) is pushed through
the same gate. A gate that never refuses is not a gate, and the catch rate on planted
mis-mappings is the number that says which this is.
"""
import json
import re
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aimecover import EXPR_FUNCS_HELP, SCHEMAS2, ask_spec  # noqa: E402
from olympiad import load_problems  # noqa: E402
from solvemap import BATTERY, PREDICATE_HELP, SCHEMAS, answer_of, parse_spec  # noqa: E402
from solvers2 import run2  # noqa: E402

STRUCTURAL = {0, 1, 2, 10, 100, 1000}      # constants a spec may use unquoted
# Digit bounds and small loop limits are structure, not data: a spec that searches
# digits 0..9 has invented nothing. Measured cost of NOT allowing these: 17 of 30
# battery problems flagged on literals like a repeated 9.
STRUCTURAL_RELAXED = STRUCTURAL | set(range(0, 13)) | {20, 50, 60, 360, 1000000}

CACHE = Path("data/custom/specgate-cache.json")


def cached_ask(prompt, n=600):
    """Replies are cached by prompt so the gate's ablation costs no GPU to recompute."""
    store = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    key = str(hash(prompt))
    if key in store:
        return store[key]
    reply = ask_spec(prompt, n=n)
    store[key] = reply
    CACHE.write_text(json.dumps(store))
    return reply

TWO_ROADS = """Map this problem onto the solver catalogue TWICE, using a DIFFERENT
solver each time, so two different machines compute the same answer. Do not compute
anything yourself.

Problem: {story}

Catalogue:
{catalogue}

For the search solver, conditions use these ops: {preds}
Expressions in multisearch may call: {funcs}

Reply with ONLY:
{{"spec_a": <spec>, "spec_b": <spec>}}
If a second, genuinely different mapping is impossible, reply with
{{"spec_a": <spec>, "spec_b": null}}"""


def literals(obj, out=None):
    """Every number appearing anywhere in a spec, as strings."""
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k != "solver":
                literals(v, out)
    elif isinstance(obj, list):
        for v in obj:
            literals(v, out)
    elif isinstance(obj, bool):
        pass          # a JSON true is not a number the problem failed to mention;
                      # isinstance(True, int) is True in Python, and the hard-battery
                      # gate flagged 'True' as an invented literal until this line
    elif isinstance(obj, (int, float)):
        out.append(str(obj))
    elif isinstance(obj, str):
        for m in re.findall(r"\d+(?:/\d+)?", obj):
            out.append(m)
    return out


def value_echo(spec, story, allowed=None):
    """Which literals are NOT in the text and not structural — the mechanical filter."""
    nums = re.findall(r"\d+(?:\.\d+)?(?:/\d+)?", story)
    text_vals = set(nums)
    for n in nums:                          # percentages and fractions written out
        try:
            text_vals.add(str(int(float(n))))
        except ValueError:
            pass
    bad = []
    for lit in literals(spec):
        if lit in text_vals:
            continue
        try:
            if float(F(lit)) in (allowed or STRUCTURAL):
                continue
        except (ValueError, ZeroDivisionError):
            pass
        bad.append(lit)
    return bad


def norm(ans):
    """Shape-normalise an answer so [13] and 13 compare equal, and 12/1 equals 12."""
    if ans is None:
        return None
    s = str(ans).strip()
    s = re.sub(r"^\[\s*'?([^,'\]]+)'?\s*\]$", r"\1", s)      # singleton list
    s = s.strip("'\" ")
    try:
        f = F(s)
        return str(f)
    except (ValueError, ZeroDivisionError):
        return s


MISMAP = {                                  # planted true negatives, right numbers only
    "search": lambda: {"solver": "digit_ops", "n": 200},
    "linear_system": lambda: {"solver": "gcd_lcm", "values": [12, 3]},
    "quadratic": lambda: {"solver": "gcd_lcm", "values": [5, 6]},
    "crt": lambda: {"solver": "gcd_lcm", "values": [3, 5, 7]},
    "gcd_lcm": lambda: {"solver": "digit_ops", "n": 12},
    "series": lambda: {"solver": "combinatorics", "kind": "choose", "n": 10, "k": 3},
    "combinatorics": lambda: {"solver": "gcd_lcm", "values": [10, 3]},
    "recurrence": lambda: {"solver": "series", "kind": "arithmetic", "first": 1,
                           "step": 1, "terms": 10},
    "rate_work": lambda: {"solver": "ratio_split", "total": 4, "ratio": [4, 6]},
    "ratio_split": lambda: {"solver": "gcd_lcm", "values": [2, 3, 5]},
    "mixture": lambda: {"solver": "rate_work", "times": [3, 7]},
    "interest": lambda: {"solver": "series", "kind": "geometric", "first": 1000,
                         "ratio": 10, "terms": 3},
    "base_convert": lambda: {"solver": "digit_ops", "n": 100},
    "digit_ops": lambda: {"solver": "factor", "n": 9174},
    "factor": lambda: {"solver": "digit_ops", "n": 360},
}


def main(n_inspect=0, out="data/custom/specgate.json"):
    n_inspect = int(n_inspect)
    catalogue = "\n".join(f"- {v}" for v in {**SCHEMAS, **SCHEMAS2}.values())

    if n_inspect:
        for solver, story, truth in (BATTERY[0], BATTERY[6]):
            r = ask_spec(TWO_ROADS.format(story=story, catalogue=catalogue,
                                          preds=PREDICATE_HELP,
                                          funcs=EXPR_FUNCS_HELP), n=600)
            print(f"[{solver}] {story[:56]}\n  RAW: {r[:400]!r}\n")
        return

    t = dict(both=0, agreed=0, delivered=0, right=0, wrong=0, flagged=0,
             single_road=0, echo_flagged=0, a_ran=0, b_ran=0,
             planted=0, planted_caught=0, planted_by_echo=0, planted_slipped=0)
    rows = []
    for solver, story, truth in BATTERY:
        reply = cached_ask(TWO_ROADS.format(story=story, catalogue=catalogue,
                                            preds=PREDICATE_HELP,
                                            funcs=EXPR_FUNCS_HELP))
        outer = parse_spec(reply)
        sa = sb = None
        if outer:
            sa, sb = outer.get("spec_a"), outer.get("spec_b")
            if sa is None and outer.get("solver"):     # a bare spec came back
                sa = outer
        row = {"solver": solver, "truth": truth}
        if not isinstance(sa, dict):
            t["flagged"] += 1
            row["verdict"] = "no spec_a"
            rows.append(row)
            print(f"{solver:<14} FLAG   no spec_a")
            continue

        bad_a = value_echo(sa, story)
        ra, wa = run2(sa)
        t["a_ran"] += ra is not None
        if isinstance(sb, dict):
            t["both"] += 1
            bad_b = value_echo(sb, story)
            rb, wb = run2(sb)
            t["b_ran"] += rb is not None
        else:
            t["single_road"] += 1
            bad_b, rb = [], None

        if bad_a or bad_b:
            t["echo_flagged"] += 1
            t["flagged"] += 1
            row["verdict"] = f"echo: {(bad_a + bad_b)[:3]}"
        elif ra is None:
            t["flagged"] += 1
            row["verdict"] = f"a refused: {wa[:40]}"
        elif rb is None:
            t["flagged"] += 1
            row["verdict"] = "no second road" if sb is None else f"b refused: {wb[:30]}"
        else:
            na, nb = norm(answer_of(ra, sa)), norm(answer_of(rb, sb))
            if na != nb:
                t["flagged"] += 1
                row["verdict"] = f"roads disagree: {na} vs {nb}"
            else:
                t["agreed"] += 1
                t["delivered"] += 1
                ok = na == norm(truth)
                t["right" if ok else "wrong"] += 1
                row["verdict"] = f"DELIVERED {na}" + ("" if ok else f" != {truth}")
        row["spec_a"], row["spec_b"], row["story"] = sa, sb, story
        rows.append(row)
        print(f"{solver:<14} {row['verdict'][:66]}")

        # The planted mis-mapping: same problem, right numbers, wrong machine.
        t["planted"] += 1
        pspec = MISMAP[solver]()
        pbad = value_echo(pspec, story)
        pres, _ = run2(pspec)
        caught_echo = bool(pbad)
        if caught_echo:
            t["planted_by_echo"] += 1
        if pres is None:
            t["planted_caught"] += 1
        elif caught_echo:
            t["planted_caught"] += 1
        elif ra is not None and norm(answer_of(pres)) != norm(answer_of(ra, sa)):
            t["planted_caught"] += 1        # disagrees with the real road
        else:
            t["planted_slipped"] += 1

    n = len(BATTERY)
    print(f"\nsecond road offered {t['both']}/{n} (single-road {t['single_road']}); "
          f"road A ran {t['a_ran']}, road B ran {t['b_ran']}")
    print(f"gate: delivered {t['delivered']} (right {t['right']}, WRONG {t['wrong']}), "
          f"flagged {t['flagged']} — of which value-echo caught {t['echo_flagged']}")
    print(f"planted mis-mappings: {t['planted_caught']}/{t['planted']} caught "
          f"({t['planted_by_echo']} by echo alone), {t['planted_slipped']} slipped")
    print(f"\nfor comparison, phase 93's ungated free arm on this battery: "
          f"29 delivered right, 1 wrong, 0 flagged")
    print("The gate's job is not to raise the score — it is to make the score mean")
    print("something. Two different machines agreeing is not two readings agreeing,")
    print("and a literal that never appeared in the problem is a mechanical lie.")
    # THE HARD ARM. Phase 94 measured 19 AIME mappings that ran and every one was
    # wrong — real mis-mappings, generated by the model, not planted by me. If the gate
    # is worth anything it flags them; if it flags everything it is worthless, which is
    # what the battery arm above is for.
    import random as _r
    _, aime = load_problems()
    apicks = _r.Random(5).sample(aime, 30)[:15]
    a = dict(seen=0, delivered=0, right=0, wrong=0, flagged=0, echo=0,
             disagree=0, no_second=0, refused=0)
    arows = []
    for problem, truth in apicks:
        arows.append({"story": problem[:400], "truth": str(truth)})
        a["seen"] += 1
        reply = cached_ask(TWO_ROADS.format(story=problem, catalogue=catalogue,
                                            preds=PREDICATE_HELP,
                                            funcs=EXPR_FUNCS_HELP))
        outer = parse_spec(reply)
        sa = outer.get("spec_a") if outer else None
        sb = outer.get("spec_b") if outer else None
        arows[-1].update({"spec_a": sa, "spec_b": sb})
        if not isinstance(sa, dict):
            a["flagged"] += 1
            continue
        if value_echo(sa, problem) or (isinstance(sb, dict)
                                       and value_echo(sb, problem)):
            a["echo"] += 1
            a["flagged"] += 1
            continue
        ra, _w = run2(sa)
        if ra is None:
            a["refused"] += 1
            a["flagged"] += 1
            continue
        if not isinstance(sb, dict):
            a["no_second"] += 1
            a["flagged"] += 1
            continue
        rb, _w = run2(sb)
        if rb is None:
            a["refused"] += 1
            a["flagged"] += 1
            continue
        if norm(answer_of(ra, sa)) != norm(answer_of(rb, sb)):
            a["disagree"] += 1
            a["flagged"] += 1
            continue
        a["delivered"] += 1
        ok = norm(answer_of(ra, sa)) == norm(str(truth))
        a["right" if ok else "wrong"] += 1
    a["rows"] = arows
    print(f"\nAIME arm ({a['seen']} problems where phase 94 measured every mapping "
          f"wrong):")
    print(f"  delivered {a['delivered']} (right {a['right']}, WRONG {a['wrong']}), "
          f"flagged {a['flagged']}")
    print(f"  flags by cause: echo {a['echo']}, no second road {a['no_second']}, "
          f"refused {a['refused']}, roads disagree {a['disagree']}")

    summary = {"n": n, **t, "aime": a, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
