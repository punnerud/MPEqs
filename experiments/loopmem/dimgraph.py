#!/usr/bin/env python3
"""The dimension control on MODEL-WRITTEN plans: refuse unit errors, with the reason, retry.

Phase 53's graph arm lost to solo on word problems because nothing could refuse a wrong plan —
a text is not an expression to inline against. Units are the piece of an inline check a text
does carry: every number arrives with a noun, every + must agree, and the answer must be
denominated in what the question asked for. This wires phase 52's control onto the plan writer,
in mpedb's shape — a refusal NAMES what broke ("B adds clip to dollar"), and the model gets the
counter-example and another attempt.

Because the unit tagger has a measured 28% noise floor on GSM8K's ad-hoc nouns, false refusals
are a live danger, so every refused plan is ALSO executed post hoc and scored: a refusal is
TRUE if the refused plan's answer was in fact wrong, FALSE if the control shot down a plan that
would have been right. The control's worth is the balance of those, not its firing rate.

Arms, same 20 GSM8K problems as phase 53, both models:

    GRAPH          write plan, execute, no control       (the phase 53 baseline, re-run)
    GRAPH+DIM      the control in the loop, up to two refusal-and-retry rounds
"""
import ast
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402
from dimcheck import Ill, asked_unit, type_expr, units_of  # noqa: E402
from jsongraph import parse_graph  # noqa: E402
from mapstore import NUM, norm  # noqa: E402
from olympiad import GRAPH, load_problems, solve_graph  # noqa: E402

RETRY = """Your previous plan has a unit error:
  {reason}

Numbers in this problem carry these units: {units}.
Steps may only add or subtract matching units, and the final step must produce {want}.

{base}"""


def literal_units(problem):
    """value -> unit, from the text. First occurrence wins on duplicates; noise is measured."""
    out = {}
    values = [norm(m) for m in NUM.findall(norm(problem))]
    tags = units_of(problem)
    for v, u in zip(values, tags):
        out.setdefault(v, u)
    return out


def check_plan(g, problem, lu=None, want=None):
    """Type every step of a literal-number plan. Returns None if clean, else the reason.

    `lu` and `want` default to the heuristic tagger; phase 57 passes model-read units in
    instead, changing the judge's input and nothing else."""
    lu = literal_units(problem) if lu is None else lu
    want = asked_unit(problem) if want is None else want
    env = {}
    for key, body in g.items():
        expr, subs, n = body, {}, [0]

        def repl(m):
            name = f"n{n[0]}"
            n[0] += 1
            subs[name] = lu.get(norm(m.group(0)))
            return name

        expr = NUM.sub(repl, expr)
        try:
            t = type_expr(ast.parse(expr, mode="eval").body, {**env, **subs})
        except Ill as e:
            named = re.sub(r"n(\d+)", lambda m: "?", str(e))
            return f"step {key} mixes units: {named}"
        except SyntaxError:
            return None                       # unparseable is the structural checks' problem
        env[key] = t
    last = env.get(list(g)[-1])
    if want is not None and last is not None and "*" not in str(last) \
            and "/" not in str(last) and last != want:
        return f"the final step produces {last}, but the question asks for {want}"
    return None


def run_steps(g):
    values = {}
    for k, body in g.items():
        expr = body
        for k2, v2 in values.items():
            expr = re.sub(rf"\b{k2}\b", f"({v2})", expr)
        if not re.fullmatch(r"[\d\s+*/().-]+", expr):
            return None
        try:
            values[k] = Fraction(eval(expr))  # noqa: S307 - digits and operators only
        except Exception:  # noqa: BLE001
            return None
    return values[list(g)[-1]] if values else None


def main(n_test=20, seed=5, out="data/custom/dimgraph.json"):
    import random
    n_test, seed = int(n_test), int(seed)
    gsm, _ = load_problems()
    rng = random.Random(seed)
    tests = rng.sample(gsm, n_test)

    results = {}
    for model in ("olmoe-1b", "qwen-35b"):
        base = dim = 0
        fired = retr_fixed = true_ref = false_ref = 0
        rows = []
        for problem, truth in tests:
            b_ans, _ = solve_graph(model, problem)
            base += b_ans == truth

            lu = literal_units(problem)
            prompt = GRAPH.format(problem=problem)
            final, refusals = None, []
            for attempt in range(3):
                g, _ = parse_graph(ask(model, prompt, n=512))
                if g is None:
                    break
                reason = check_plan(g, problem)
                if reason is None:
                    final = run_steps(g)
                    break
                # The refusal, classified honestly: run the refused plan anyway.
                refused_ans = run_steps(g)
                fired += 1
                if refused_ans == truth:
                    false_ref += 1
                else:
                    true_ref += 1
                refusals.append(reason)
                if attempt == 2:
                    final = refused_ans       # out of retries: the last plan stands, noted
                units_txt = ", ".join(f"{v} = {u}" for v, u in lu.items() if u) or "unknown"
                prompt = RETRY.format(reason=reason, units=units_txt,
                                      want=asked_unit(problem) or "the asked quantity",
                                      base=GRAPH.format(problem=problem))
            dim += final == truth
            if refusals and final == truth:
                retr_fixed += 1
            rows.append({"model": model, "truth": str(truth), "base": str(b_ans),
                         "dim": str(final), "refusals": refusals})
        results[model] = {"n": n_test, "base": base, "dim": dim, "refusals_fired": fired,
                          "true_refusals": true_ref, "false_refusals": false_ref,
                          "fixed_after_refusal": retr_fixed, "rows": rows}
        print(f"{model}: graph {base}/{n_test} -> graph+dim {dim}/{n_test}   "
              f"refusals {fired} ({true_ref} true, {false_ref} false), "
              f"{retr_fixed} fixed after being shown the reason")

    print("\nA refusal that names the step and the units is the counter-example shape the")
    print("verifier work kept arriving at. Whether it pays depends entirely on the true-to-")
    print("false refusal balance, which the 28% tagger noise makes a real question — measured")
    print("here by running every refused plan anyway.")
    Path(out).write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
