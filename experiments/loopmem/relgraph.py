#!/usr/bin/env python3
"""Abstract the WHOLE problem into a relation graph — nothing stripped, everything named.

Phase 57 measured abstraction-as-deletion: quantities and units with the story thrown away,
and planning collapsed from 16/20 to 2/20 because the relations lived in the story. This is
the proposal as actually meant: abstraction-as-translation. The story is absorbed COMPLETELY —
every quantity becomes a named node, every "half as many" and "sold to" becomes a defining
relation between nodes, the question becomes a distinguished node — and when the residual of
the translation is empty, the record can solve the system without the story, exactly.

    {"defs": {"april": "48", "may": "april / 2", "total": "april + may"}, "asked": "total"}

What makes this arm different from every judged arm before it: the record's checks here have
PERFECT ground truth, like mpedb's and unlike the unit tagger's. A reference to an undefined
node, a cycle, an unparseable definition, a problem number that appears nowhere in the system
— each is a structural fact, refusable with the reason named and zero false positives by
construction. The noisy-judge lesson of phase 55 is respected by only gating on what can be
checked exactly.

Arms on the same 20 problems as phases 53-57:

    STORY GRAPH     plan straight from the story          (16/20 and 4/20, re-used context)
    QUANTITIES ONLY the phase 57 deletion                 (2/20, the negative control)
    RELATION GRAPH  the full translation, solved by the record, structural retry on refusal
"""
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402
from mapstore import NUM, norm  # noqa: E402
from olympiad import load_problems  # noqa: E402

TRANSLATE = """Translate the whole problem into named quantities and the relations between
them. Every number in the problem must appear; every relationship in the story must become a
definition. Reply with only JSON:

Example:
Problem: Natalia sold clips to 48 of her friends in April, and then she sold half as many
clips in May. How many clips did Natalia sell altogether in April and May?
{{"defs": {{"april": "48", "may": "april / 2", "total": "april + may"}}, "asked": "total"}}

Problem: {problem}
"""

RETRY = """Your previous translation was refused:
  {reason}

Every name you reference must be defined, definitions may not be circular, and every number
from the problem must appear in some definition.

{base}"""

NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def parse_system(reply):
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None, "no JSON object"
    try:
        d = json.loads(m.group(0))
        defs, asked = d["defs"], d["asked"]
    except Exception:  # noqa: BLE001
        return None, "malformed JSON"
    if not isinstance(defs, dict) or not all(isinstance(v, str) for v in defs.values()):
        return None, "defs must map names to expression strings"
    return {"defs": defs, "asked": str(asked)}, None


def solve_system(sys_, problem):
    """Topological, exact, and structurally verified with perfect ground truth."""
    defs, asked = sys_["defs"], sys_["asked"]
    if asked not in defs:
        return None, f"the asked node '{asked}' is never defined"
    # Every number in the problem must appear in the system — completeness of the translation.
    sys_nums = set()
    for body in defs.values():
        sys_nums.update(norm(x) for x in NUM.findall(body))
    missing = [v for v in {norm(x) for x in NUM.findall(norm(problem))}
               if v not in sys_nums]
    if missing:
        return None, f"problem numbers missing from the system: {', '.join(sorted(missing))}"

    values, visiting = {}, set()

    def resolve(name):
        if name in values:
            return values[name]
        if name not in defs:
            raise KeyError(name)
        if name in visiting:
            raise RecursionError(name)
        visiting.add(name)
        expr = defs[name]
        for ref in sorted(set(NAME.findall(expr)), key=len, reverse=True):
            if ref in defs:
                expr = re.sub(rf"\b{ref}\b", f"({resolve(ref)})", expr)
        if not re.fullmatch(r"[\d\s+*/().-]+", expr):
            raise ValueError(name)
        values[name] = Fraction(eval(expr))  # noqa: S307 - digits and operators only
        visiting.discard(name)
        return values[name]

    try:
        return resolve(asked), None
    except KeyError as e:
        return None, f"'{e.args[0]}' is referenced but never defined"
    except RecursionError as e:
        return None, f"'{e.args[0]}' is defined in terms of itself"
    except (ValueError, ZeroDivisionError, SyntaxError):
        return None, "a definition does not reduce to arithmetic"


def main(n_test=20, seed=5, out="data/custom/relgraph.json"):
    import random
    n_test, seed = int(n_test), int(seed)
    gsm, _ = load_problems()
    tests = random.Random(seed).sample(gsm, n_test)

    results = {}
    for model in ("qwen-35b", "olmoe-1b"):
        solved = translated = refused_total = fixed = 0
        rows = []
        for problem, truth in tests:
            prompt = TRANSLATE.format(problem=problem)
            ans, reasons = None, []
            for attempt in range(3):
                sys_, why = parse_system(ask(model, prompt, n=512))
                if sys_ is None:
                    reasons.append(why)
                    prompt = RETRY.format(reason=why, base=TRANSLATE.format(problem=problem))
                    continue
                ans, why = solve_system(sys_, problem)
                if ans is not None:
                    translated += 1
                    break
                reasons.append(why)
                refused_total += 1
                prompt = RETRY.format(reason=why, base=TRANSLATE.format(problem=problem))
            ok = ans == truth
            solved += ok
            if ok and len(reasons) > 0:
                fixed += 1
            rows.append({"truth": str(truth), "answer": str(ans), "ok": ok,
                         "refusals": reasons})
        results[model] = {"n": n_test, "solved": solved, "translated": translated,
                          "structural_refusals": refused_total, "fixed_after_refusal": fixed,
                          "rows": rows}
        print(f"{model}: relation graph {solved}/{n_test} solved "
              f"({translated} systems accepted, {refused_total} structural refusals, "
              f"{fixed} fixed after refusal)")

    print("\nContext: story-plan 16/20 (35B) and 4/20 (1B); quantities-only 2/20. If the full")
    print("translation lands near the story plan, the abstraction lost nothing that mattered")
    print("and gained a solvable, checkable object; the story was absorbed, not stripped.")
    Path(out).write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
