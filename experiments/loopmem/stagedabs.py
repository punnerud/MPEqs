#!/usr/bin/env python3
"""Staged reversible absorption: the story abstracted away span by span, residues kept.

Phase A translates the whole problem in one call. This does it in ORDER, and the order is the
point: each step the model names one span of the remaining story and the definition it becomes;
the record cuts the span (exact match — the model quotes, the record edits, phase 27's rule),
stores (span, position, its numbers) as the step's RESIDUE, and adds the definition to the
graph. The residues are precisely the values central to the calculation, and the chain unwinds
LIFO — putting each span back at its position must reproduce the previous text byte for byte,
verified per step and for the whole chain, mpedb's rRETL shape applied to text-to-graph.

What the residue buys beyond safety is PROVENANCE. Every node knows the span it came from, so
when the finished system is refused structurally — an undefined reference, a missing number, a
cycle: checks with perfect ground truth — the refusal is localised to the sentence that
produced the offending node, and the model re-translates THAT step while the rest of the graph
stands. Phase 46's repair, on text.

Measured against phase A (one-shot) and the story-plan baseline:

    solved            staged vs one-shot vs story plan
    reversibility     share of steps and chains that restore byte-exact (claimed 100%,
                      verified anyway — a claim that cannot fail is worth checking)
    absorption        share of the problem's numbers absorbed when the model stops
    localisation      structural refusals traced to one source span, and repaired in place
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402
from mapstore import NUM, norm  # noqa: E402
from olympiad import load_problems  # noqa: E402
from relgraph import solve_system  # noqa: E402

STEP = """You are absorbing a word problem into named quantities, one piece at a time.

Remaining text:
{text}

Graph so far: {defs}

Quote ONE span from the remaining text (copy it exactly), and give the definition it becomes.
Use earlier names where the story refers to them. When the remaining text holds no more
quantities or relations, reply exactly DONE.

Reply with only JSON:
{{"span": "gives away 4 of his 9 marbles", "def": {{"kept": "9 - 4"}}}}
{hint}"""

ASKED = """The problem is fully absorbed. Graph: {defs}

Which name answers the question "{question}"? If none does, define one more:
{{"asked": "total"}} or {{"asked": "total", "def": {{"total": "april + may"}}}}
"""

REPAIR = """One definition in the graph is wrong. It came from this part of the problem:
  "{span}"
and became: {olddef}
The system was refused because: {reason}

Give a corrected definition for this span only, using the same style.
Reply with only JSON: {{"def": {{"name": "expression"}}}}
"""


def extract_json(reply):
    m = list(re.finditer(r"\{.*\}", reply, re.S))
    if not m:
        return None
    try:
        return json.loads(m[-1].group(0))
    except Exception:  # noqa: BLE001
        return None


def absorb(model, problem, max_steps=10):
    """The staged loop. Returns (defs, residues, provenance, log)."""
    text = problem
    defs, residues, prov = {}, [], {}
    log = {"steps": 0, "span_misses": 0, "step_rev_ok": 0}
    hint = ""
    for _ in range(max_steps):
        reply = ask(model, STEP.format(text=text, defs=json.dumps(defs) or "{}",
                                       hint=hint), n=400)
        if "DONE" in reply[:200] and "span" not in reply[:200]:
            break
        d = extract_json(reply)
        if not d or "span" not in d or "def" not in d or not isinstance(d["def"], dict):
            log["span_misses"] += 1
            hint = "\nYour last reply was not valid JSON with a span and a def."
            continue
        span = str(d["span"]).strip()
        pos = text.find(span)
        if pos < 0 or not span:
            log["span_misses"] += 1
            # The refusal changes the question — never the same prompt twice at temp 0.
            hint = (f"\nYour last span was not an exact quote of the remaining text: "
                    f"\"{span[:60]}\". Copy a span character for character.")
            continue
        new_text = text[:pos] + text[pos + len(span):]
        # The step's reversibility, verified rather than assumed.
        log["step_rev_ok"] += (new_text[:pos] + span + new_text[pos:]) == text
        residues.append({"span": span, "pos": pos,
                         "numbers": [norm(x) for x in NUM.findall(norm(span))]})
        for k, v in d["def"].items():
            defs[str(k)] = str(v)
            prov[str(k)] = len(residues) - 1
        text = new_text
        log["steps"] += 1
        hint = ""
    return text, defs, residues, prov, log


def unwind(final_text, residues):
    """LIFO, back to the original. The whole chain must restore byte-exact."""
    t = final_text
    for r in reversed(residues):
        t = t[:r["pos"]] + r["span"] + t[r["pos"]:]
    return t


def main(n_test=20, seed=5, model="qwen-35b", out="data/custom/stagedabs.json"):
    import random
    n_test, seed = int(n_test), int(seed)
    gsm, _ = load_problems()
    tests = random.Random(seed).sample(gsm, n_test)

    tally = {"solved": 0, "chains_exact": 0, "steps_total": 0, "steps_rev_ok": 0,
             "refusals": 0, "localised": 0, "repaired_in_place": 0, "span_misses": 0}
    absorb_frac = []
    rows = []
    for problem, truth in tests:
        rest, defs, residues, prov, log = absorb(model, problem)
        tally["steps_total"] += log["steps"]
        tally["steps_rev_ok"] += log["step_rev_ok"]
        tally["span_misses"] += log["span_misses"]
        tally["chains_exact"] += unwind(rest, residues) == problem

        nums_total = {norm(x) for x in NUM.findall(norm(problem))}
        nums_absorbed = {n for r in residues for n in r["numbers"]}
        absorb_frac.append(len(nums_absorbed & nums_total) / max(len(nums_total), 1))

        # The asked node is the graph's SINK — the node no other definition references.
        # The first run asked the model and 18 of 20 failures were that one call: ten
        # echoes of the example name "total", eight unparseable. The record can read the
        # answer off the graph's own shape, deterministically, with no call at all; the
        # model is only consulted when the sink is not unique.
        import re as _re
        refs = set()
        for body in defs.values():
            refs.update(_re.findall(r"[A-Za-z_][A-Za-z0-9_]*", body))
        sinks = [k for k in defs if k not in refs]
        if len(sinks) == 1:
            asked = sinks[0]
        else:
            a = extract_json(ask(model, ASKED.format(defs=json.dumps(defs),
                                                     question=problem.split(".")[-1]),
                                 n=200))
            if a and "def" in a and isinstance(a["def"], dict):
                defs.update({str(k): str(v) for k, v in a["def"].items()})
            asked = str(a.get("asked", "")) if a else ""
            if asked not in defs and sinks:
                # The model echoed the example name or invented one: the record falls back
                # to the newest sink rather than accepting a name the graph does not have.
                asked = sinks[-1]
        ans, why = solve_system({"defs": defs, "asked": asked}, problem)

        repaired = False
        if ans is not None and why is None:
            pass
        elif why and why.startswith("problem numbers missing"):
            # Localised to the RESIDUAL text: an unabsorbed number is by definition still in
            # what remains, so the repair is one more absorption step aimed at it.
            tally["refusals"] += 1
            tally["localised"] += 1
            miss = why.split(":", 1)[1].strip()
            reply = ask(model, STEP.format(
                text=rest, defs=json.dumps(defs),
                hint=f"\nThe numbers {miss} are still unabsorbed. Quote the span holding "
                     f"them and define it."), n=400)
            d2 = extract_json(reply)
            if d2 and "def" in d2 and isinstance(d2["def"], dict):
                defs.update({str(k): str(v) for k, v in d2["def"].items()})
                ans, why = solve_system({"defs": defs, "asked": asked}, problem)
                repaired = ans is not None
                tally["repaired_in_place"] += repaired
        if ans is None and why:
            if not repaired and not (why or "").startswith("problem numbers missing"):
                tally["refusals"] += 1
            # Localise: which node does the refusal name, and which span produced it?
            named = next((k for k in defs if k in why), None)
            src = prov.get(named)
            if src is not None:
                tally["localised"] += 1
                fix = extract_json(ask(model, REPAIR.format(
                    span=residues[src]["span"],
                    olddef=json.dumps({named: defs[named]}), reason=why), n=200))
                if fix and "def" in fix and isinstance(fix["def"], dict):
                    defs.update({str(k): str(v) for k, v in fix["def"].items()})
                    ans, why = solve_system({"defs": defs, "asked": asked}, problem)
                    repaired = ans is not None
                    tally["repaired_in_place"] += repaired
        ok = ans == truth
        tally["solved"] += ok
        rows.append({"truth": str(truth), "answer": str(ans), "ok": ok,
                     "steps": log["steps"], "refusal": why, "repaired": repaired,
                     "absorbed": absorb_frac[-1]})

    n = len(tests)
    print(f"{model}, {n} problems, staged reversible absorption:\n")
    print(f"  solved                    : {tally['solved']}/{n}")
    print(f"  chains restore byte-exact : {tally['chains_exact']}/{n}")
    print(f"  steps reversible          : {tally['steps_rev_ok']}/{tally['steps_total']} "
          f"({tally['span_misses']} span misses refused)")
    print(f"  numbers absorbed          : {100 * sum(absorb_frac) / n:.0f}% on average")
    print(f"  structural refusals       : {tally['refusals']}, localised to a span "
          f"{tally['localised']}, repaired in place {tally['repaired_in_place']}")
    print("\nThe residues are the story, kept; the graph is the story, absorbed. Either can")
    print("reproduce the other, which is what abstraction-without-loss means, and the")
    print("provenance is what turns a refusal into a repair instead of a restart.")
    Path(out).write_text(json.dumps({"model": model, "n": n, **tally,
                                     "mean_absorbed": sum(absorb_frac) / n,
                                     "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
