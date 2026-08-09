#!/usr/bin/env python3
"""Can the relations layer be carried cheaper than whole reading?

The residual chain ended with relations as the unpaid, load-bearing layer: values, names
and references were each paid by machinery, solving never moved, and the one-shot reader's
16/20 stands as what holding the whole story buys. The question left: does the RELATION —
how clip_may derives from clip_april — live in the story's STRUCTURE or in its VALUES?
Because the answer decides the economics:

    A  WHOLE READING       the 35B translates the full problem (phase 58's arm, re-run here
                           so the comparison is within-run)
    B  STRUCTURE ONLY      the 35B translates the MASKED problem — numbers replaced by
                           <N1>..<Nk> — into a relation skeleton over v1..vk; the record
                           binds the actual numbers mechanically. Same price today, but a
                           skeleton keyed by masked text is a TEMPLATE: payable once per
                           shape and reused, the phase 45 economy applied to relations.
    C  SMALL READER        the same masked translation by the 1B model, ten times cheaper
                           per instance. Reading structure might be easier than solving —
                           filling was (16/18), reading units was (phase 57).

If B holds near A, relations need the structure and not the values — cacheable, amortised
cheap. If C also holds, they are cheap outright. If B collapses, the values are part of the
relation itself and whole reading is irreducible — a real answer too.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402
from mapstore import NUM, norm  # noqa: E402
from olympiad import load_problems  # noqa: E402
from relgraph import TRANSLATE, parse_system, solve_system  # noqa: E402

MASKED = """Translate the whole problem into named quantities and the relations between
them. The numbers are hidden as <N1>, <N2>, ... — use v1, v2, ... for them in your
definitions; every <Nk> must appear as vk somewhere. Reply with only JSON:

Example:
Problem: Natalia sold clips to <N1> of her friends in April, and then she sold half as
many clips in May. How many clips did Natalia sell altogether in April and May?
{{"defs": {{"april": "v1", "may": "april / 2", "total": "april + may"}}, "asked": "total"}}

Problem: {problem}
"""


def mask_numbered(problem):
    values, out, pos, k = [], "", 0, 0
    for m in NUM.finditer(norm(problem)):
        k += 1
        values.append(norm(m.group(0)))
        out += norm(problem)[pos:m.start()] + f"<N{k}>"
        pos = m.end()
    return out + norm(problem)[pos:], values


def bind(defs, values):
    """v-tokens to the problem's own numbers, longest index first — purely mechanical."""
    out = {}
    for k, body in defs.items():
        e = str(body)
        for i in sorted(range(len(values)), reverse=True):
            e = re.sub(rf"\bv{i + 1}\b", values[i], e)
        out[str(k)] = e
    return out


def arm_masked(model, problem):
    masked, values = mask_numbered(problem)
    sys_, why = parse_system(ask(model, MASKED.format(problem=masked), n=512))
    if sys_ is None:
        return None, f"parse: {why}"
    bound = bind(sys_["defs"], values)
    return solve_system({"defs": bound, "asked": sys_["asked"]}, problem)


def main(n_test=20, seed=5, out="data/custom/relcarry.json"):
    import random
    n_test, seed = int(n_test), int(seed)
    gsm, _ = load_problems()
    tests = random.Random(seed).sample(gsm, n_test)

    tally = {"A_whole_35b": 0, "B_masked_35b": 0, "C_masked_1b": 0}
    rows = []
    for problem, truth in tests:
        sys_, _ = parse_system(ask("qwen-35b", TRANSLATE.format(problem=problem), n=512))
        a = solve_system(sys_, problem)[0] if sys_ else None
        b, _ = arm_masked("qwen-35b", problem)
        c, _ = arm_masked("olmoe-1b", problem)
        tally["A_whole_35b"] += a == truth
        tally["B_masked_35b"] += b == truth
        tally["C_masked_1b"] += c == truth
        rows.append({"truth": str(truth), "A": str(a), "B": str(b), "C": str(c)})

    n = n_test
    print(f"{n} problems — where do the relations live?\n")
    print(f"  A  whole reading, 35B        : {tally['A_whole_35b']}/{n}")
    print(f"  B  structure only, 35B       : {tally['B_masked_35b']}/{n}   "
          f"(skeleton cacheable by masked text)")
    print(f"  C  structure only, 1B        : {tally['C_masked_1b']}/{n}   "
          f"(ten times cheaper per instance)")
    print("\nB near A: relations are structural — pay once per shape, reuse forever.")
    print("C near B: they are cheap outright. B collapsed: the values are part of the")
    print("relation and whole reading is irreducible, which would close the chain for good.")
    Path(out).write_text(json.dumps({"n": n, **tally, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
