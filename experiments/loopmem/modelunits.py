#!/usr/bin/env python3
"""The model reads the units — and the reading IS the abstraction of the problem.

Phase 55's verdict: the refusal gate inherited a 28% noise floor from a regex tagger, and unit
reading on informal text is a language problem sitting under a control that was supposed to
check the model. So the language problem goes to the language model: one call per problem
abstracts the story into quantities-with-units plus the asked unit, and every downstream
mechanism runs unchanged on that abstract form.

Three measurements hang off the same reading:

  GATE       phase 55's refusal loop re-run with model-read units. The heuristic judge shot
             down right plans 32 times in 36; the whole question is what the false-refusal
             count becomes when the judge's input is read properly.
  ROUTING    phase 56's dimensional triage re-run with model-read units. The heuristic left
             26 of 60 problems untaggable and solved 2; both numbers move if reading was the
             blocker.
  ABSTRACT   plans written FROM the abstract form instead of the story — quantities and the
             asked unit, no narrative. If detail was distracting the planner, removing it
             shows up here; if the story carries structure the abstraction loses, that shows
             up instead.

The reading itself is scored implicitly by all three: nothing else changed, so every delta is
the reader's.
"""
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402
from dimgraph import GRAPH, RETRY, check_plan, run_steps  # noqa: E402
from formularoute import route  # noqa: E402
from jsongraph import parse_graph  # noqa: E402
from mapstore import NUM, TEST, norm  # noqa: E402
from olympiad import load_problems, solve_graph  # noqa: E402

READ = """List every number in this problem and what it counts, then what the question asks
for. Use ONE lowercase singular word per unit ("dollar" for money, "none" for pure factors
like "twice" or percentages).

Reply with only JSON, exactly this shape:
{{"numbers": [[48, "clip"], [2, "none"]], "asked": "clip"}}

Problem: {problem}
"""

ABSTRACT_PLAN = """Solve the problem by writing ONLY the arithmetic plan as JSON. Each key is
one step using the given numbers or earlier keys; never write a computed result. The last key
is the final answer.

Example:
Given: 3 box, 12 egg-per-box, 5 egg.  Asked: egg
{{"A": "3 * 12", "B": "A - 5"}}

Given: {given}.  Asked: {asked}
"""


def read_units(model, problem):
    """One model call: the abstract form. Falls back to nothing on shape mismatch, counted."""
    reply = ask(model, READ.format(problem=problem), n=320)
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        nums = [(str(Fraction(str(v).replace(",", ""))), None if u in ("none", None, "")
                 else str(u).strip().lower()) for v, u in d["numbers"]]
        asked = d.get("asked")
        asked = None if asked in ("none", None, "") else str(asked).strip().lower()
        return {"numbers": nums, "asked": asked}
    except Exception:  # noqa: BLE001 - malformed reading counts as no reading
        return None


def main(n_gate=20, n_route=60, seed=5, out="data/custom/modelunits.json"):
    import random
    n_gate, n_route, seed = int(n_gate), int(n_route), int(seed)
    model = "qwen-35b"
    gsm, _ = load_problems()
    rng = random.Random(seed)
    gate_tests = rng.sample(gsm, n_gate)

    # ---- GATE with model-read units ------------------------------------------------
    base = dim = fired = true_ref = false_ref = fixed = unread = 0
    abstract_ok = 0
    for problem, truth in gate_tests:
        b_ans, _ = solve_graph(model, problem)
        base += b_ans == truth

        r = read_units(model, problem)
        if r is None:
            unread += 1
            dim += b_ans == truth              # no reading: the gate stands down, plan flows
            continue
        lu = {}
        for v, u in r["numbers"]:
            lu.setdefault(v, u)
        prompt = GRAPH.format(problem=problem)
        final, had_refusal = None, False
        for attempt in range(3):
            g, _ = parse_graph(ask(model, prompt, n=512))
            if g is None:
                break
            reason = check_plan(g, problem, lu=lu, want=r["asked"])
            if reason is None:
                final = run_steps(g)
                break
            refused_ans = run_steps(g)
            fired += 1
            had_refusal = True
            if refused_ans == truth:
                false_ref += 1
            else:
                true_ref += 1
            if attempt == 2:
                final = refused_ans
            units_txt = ", ".join(f"{v} = {u}" for v, u in lu.items() if u) or "unknown"
            prompt = RETRY.format(reason=reason, units=units_txt,
                                  want=r["asked"] or "the asked quantity",
                                  base=GRAPH.format(problem=problem))
        dim += final == truth
        if had_refusal and final == truth:
            fixed += 1

        # ---- ABSTRACT: plan from the reading, not the story ------------------------
        given = ", ".join(f"{v} {u or 'unitless'}" for v, u in r["numbers"])
        g2, _ = parse_graph(ask(model, ABSTRACT_PLAN.format(
            given=given, asked=r["asked"] or "a number"), n=512))
        a_ans = run_steps(g2) if g2 else None
        abstract_ok += a_ans == truth

    print(f"GATE, model-read units ({model}, {n_gate} problems):")
    print(f"  graph {base}/{n_gate} -> graph+dim {dim}/{n_gate}   "
          f"refusals {fired} ({true_ref} true, {false_ref} false), {fixed} fixed, "
          f"{unread} unreadable")
    print(f"  plans from the ABSTRACT form alone: {abstract_ok}/{n_gate}  "
          f"(story-based baseline above)")

    # ---- ROUTING with model-read units ---------------------------------------------
    tests = []
    for line in TEST.read_text().splitlines():
        d = json.loads(line)
        tests.append((d["question"],
                      Fraction(norm(d["answer"].rsplit("#### ", 1)[-1].strip()))))
    tests = random.Random(seed).sample(tests, n_route)
    tri = {"unique": 0, "unique_right": 0, "ambiguous": 0, "ambig_contains": 0,
           "no_route": 0, "untaggable": 0}
    for q, truth in tests:
        r = read_units(model, q)
        if r is None or r["asked"] is None or not r["numbers"]:
            tri["untaggable"] += 1
            continue
        quantities = [(Fraction(v), {u: 1} if u else {}) for v, u in r["numbers"]]
        cands = sorted(v for v in route(quantities, {r["asked"]: 1})
                       if v.denominator == 1 and v >= 0)
        if not cands:
            tri["no_route"] += 1
        elif len(cands) == 1:
            tri["unique"] += 1
            tri["unique_right"] += cands[0] == truth
        else:
            tri["ambiguous"] += 1
            tri["ambig_contains"] += truth in cands

    print(f"\nROUTING, model-read units ({n_route} problems):")
    print(f"  untaggable {tri['untaggable']}  no-route {tri['no_route']}  "
          f"unique {tri['unique']} (right {tri['unique_right']})  "
          f"ambiguous {tri['ambiguous']} (truth present {tri['ambig_contains']})")
    print("\nHeuristic judge, for the record: 32/36 false refusals; 26 untaggable, 2 solved.")
    print("Every delta above is the reader's, because nothing else moved.")
    summary = {"gate": {"base": base, "dim": dim, "fired": fired, "true": true_ref,
                        "false": false_ref, "fixed": fixed, "unread": unread,
                        "abstract": abstract_ok, "n": n_gate},
               "routing": {**tri, "n": n_route}}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
