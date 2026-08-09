#!/usr/bin/env python3
"""Fan-out, compress, fan out again: iterating until the split's hidden residual is paid.

The observation this implements: a reversible sentence split is byte-lossless on the TEXT and
not on the DEPENDENCIES. A part that says "half as many" points outside itself; in one
left-to-right pass every forward reference is unresolvable, and that cross-part information is
the split's HIDDEN residual — invisible to the byte check, visible the moment a part cannot
translate alone.

The cycle pays it off in rounds:

    FAN OUT     every part is asked to translate, given the current graph
    COMPRESS    the parts' definitions merge into one graph — the compressed state
    REPEAT      parts that yielded nothing are asked again, now seeing what every OTHER part
                contributed; a reference that was forward in round 1 is resolved in round 2

Until a fixpoint: a round that adds no definition ends the loop. The hidden residual becomes a
measured number — the definitions that could only arrive in round two or later — and the claim
that iteration helps becomes the difference between round-1 accuracy and fixpoint accuracy on
the same problems.

Same 20 problems, anchors: one-shot 16/20, model-cut staged 9/20, single-pass record-cut
(rretlfan) measured alongside.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402
from olympiad import load_problems  # noqa: E402
from relgraph import solve_system  # noqa: E402
from rretlfan import split_parts  # noqa: E402
from stagedabs import extract_json  # noqa: E402

TRANSLATE_PART = """You are absorbing a word problem into named quantities. Definitions so
far, contributed by the OTHER parts of the problem: {defs}

This fragment (part {i} of {n}):
{part}

Give the definitions this fragment adds, using the names above where it refers to them. If
it adds none, reply {{}}.

Reply with only JSON, like: {{"may": "april / 2"}}
"""


def solve_with_sink(defs, problem):
    refs = set()
    for body in defs.values():
        refs.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(body)))
    sinks = [k for k in defs if k not in refs]
    for cand in (sinks[-1:] if sinks else []) + list(defs)[-1:]:
        ans, why = solve_system({"defs": defs, "asked": cand}, problem)
        if ans is not None:
            return ans
    return None


def main(n_test=20, seed=5, model="qwen-35b", max_rounds=3,
         out="data/custom/fanrounds.json"):
    import random
    n_test, seed, max_rounds = int(n_test), int(seed), int(max_rounds)
    gsm, _ = load_problems()
    tests = random.Random(seed).sample(gsm, n_test)

    tally = {"solved_r1": 0, "solved_fix": 0, "late_defs": 0, "defs_total": 0,
             "rounds_used": 0, "calls": 0}
    rows = []
    for problem, truth in tests:
        parts = split_parts(problem)
        defs, born = {}, {}
        r1_ans = None
        rounds = 0
        for rnd in range(1, max_rounds + 1):
            rounds = rnd
            added = 0
            for i, part in enumerate(parts):
                d = extract_json(ask(model, TRANSLATE_PART.format(
                    defs=json.dumps(defs) or "{}", i=i + 1, n=len(parts),
                    part=part.strip()), n=300))
                tally["calls"] += 1
                if isinstance(d, dict):
                    for k, v in d.items():
                        if isinstance(v, (str, int, float)) and str(k) not in defs:
                            defs[str(k)] = str(v)
                            born[str(k)] = rnd
                            added += 1
            if rnd == 1:
                r1_ans = solve_with_sink(dict(defs), problem)
            if added == 0:
                break                      # fixpoint: the residual is paid or unpayable
        fix_ans = solve_with_sink(defs, problem)

        late = sum(1 for r in born.values() if r >= 2)
        tally["late_defs"] += late
        tally["defs_total"] += len(defs)
        tally["rounds_used"] += rounds
        tally["solved_r1"] += r1_ans == truth
        tally["solved_fix"] += fix_ans == truth
        rows.append({"truth": str(truth), "round1": str(r1_ans), "fixpoint": str(fix_ans),
                     "parts": len(parts), "rounds": rounds, "late_defs": late,
                     "defs": len(defs)})

    n = n_test
    print(f"{model}, {n} problems, fan-out/compress cycles to fixpoint:\n")
    print(f"  solved after round 1       : {tally['solved_r1']}/{n}")
    print(f"  solved at fixpoint         : {tally['solved_fix']}/{n}")
    print(f"  definitions arriving late  : {tally['late_defs']} of {tally['defs_total']} "
          f"(the hidden residual, measured)")
    print(f"  mean rounds to fixpoint    : {tally['rounds_used'] / n:.1f}, "
          f"{tally['calls']} calls total")
    print("\nAnchors: one-shot 16/20, model-cut staged 9/20. The gap between round 1 and the")
    print("fixpoint is what iterating pays of the split's hidden residual; whatever gap to")
    print("one-shot remains after the fixpoint is the price of fragmentation itself.")
    Path(out).write_text(json.dumps({"model": model, "n": n, **tally, "rows": rows},
                                    indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
