#!/usr/bin/env python3
"""AIME once more, with everything the night's phases produced.

Phase 101 measured the mapper as the bottleneck: retrieval put the right shape in front
of the model five times out of five and one came back right. Since then the library grew
from 20 machines to 37, the schemas learned to state their units and lead with their
distinctions, slots that had to be composed became slots that can be filled, and the
exemplars learned to be templates. None of that was aimed at AIME.

So the oldest open thread gets one more measurement with the current machinery: the same
fifteen problems, the same five known to be reachable, the full catalogue, and a bank
whose entries for the reachable shapes are written the way phase 119 found a small model
needs them — copyable. If the number moves, the mapper was never the whole story; if it
does not, phase 99's diagnosis stands as the study's honest last word on competition
mathematics.

Phase 100 showed three worked mappings lift AIME from 0 to 1 of the 5 reachable
problems, and the three it showed covered bit-patterns, Diophantine minimisation and
aggregation-with-mod. Two of the five reachable problems need shapes NOT in that set: a
divisor search over a factorial with a quotient predicate, and a count over modular
inverses at growing powers of two. A fixed prompt cannot cover a growing library — which
is precisely the situation phase 88 solved for translations, so the same machinery moves
up a level.

Twelve exemplars, each a problem shape with its hand-written spec, all invented for the
bank so nothing leaks from the test set. Each is embedded (numbers masked, the phase 88
key), and for a new problem the two nearest are retrieved and shown. The bank is the
library's memory of HOW ITS MACHINES GET USED, which is a different thing from the
machines and a different thing from the catalogue.

Measured against phase 100's fixed three on the same fifteen problems, and separately:
does retrieval put the right SHAPE in front of the model for the five that are
reachable? That second number is the one that says whether a miss is retrieval's fault
or the mapper's.
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aimecover import EXPR_FUNCS_HELP, SCHEMAS2, ask_spec  # noqa: E402
from bands2 import SCHEMA_MORE  # noqa: E402
from bands3 import SCHEMA_3  # noqa: E402
from bands4 import SCHEMA_4  # noqa: E402
from bands5 import SCHEMA_5  # noqa: E402
from newbands import SCHEMA_NEW  # noqa: E402
from aimeceiling import HAND_SPECS  # noqa: E402
from embednav import embed  # noqa: E402
from olympiad import load_problems  # noqa: E402
from solvemap import PREDICATE_HELP, SCHEMAS, answer_of, parse_spec  # noqa: E402
from solvers2 import run2  # noqa: E402

# (shape tag, invented problem, spec) — the bank. No test problem appears here.
BANK = [
    ("bits", "Each vertex of a regular hexagon is painted black or white. How many "
     "paintings have no two opposite vertices both black?",
     '{"solver":"multisearch","variables":[{"name":"c","from":0,"to":63}],'
     '"conditions":["not ((c//2**0)%2 == 1 and (c//2**3)%2 == 1)",'
     '"not ((c//2**1)%2 == 1 and (c//2**4)%2 == 1)",'
     '"not ((c//2**2)%2 == 1 and (c//2**5)%2 == 1)"],"aggregate":"count"}'),
    ("diophantine", "Positive integers x and y satisfy 3x + 5y = 100. What is the "
     "least possible value of x + y?",
     '{"solver":"multisearch","variables":[{"name":"x","from":1,"to":100},'
     '{"name":"y","from":1,"to":100}],"conditions":["3*x + 5*y == 100"],'
     '"objective":"x + y","aggregate":"min"}'),
    ("aggregate_mod", "Find the remainder when the sum of k(k+1)/2 for k from 1 to 50 "
     "is divided by 1000.",
     '{"solver":"multisearch","variables":[{"name":"k","from":1,"to":50}],'
     '"objective":"(k*(k+1))//2","aggregate":"sum","post":{"op":"mod","arg":1000}}'),
    ("factorial_divisors", "The sum of all positive divisors d of 10! for which 10!/d "
     "is a perfect cube can be written as a product of prime powers. Find the sum of "
     "the exponents.",
     '{"solver":"search","domain":{"kind":"divisors_of_factorial","k":10},'
     '"conditions":[{"op":"quotient_is_cube","arg":3628800}],"aggregate":"sum",'
     '"post":{"op":"exponent_sum"}}'),
    ("exponent_sum", "The sum of all divisors of 720 can be written as a product of "
     "prime powers. Find the sum of the exponents.",
     '{"solver":"search","domain":{"kind":"divisors_of","n":720},"conditions":[],'
     '"aggregate":"sum","post":{"op":"exponent_sum"}}'),
    # Rewritten as a copyable template after phase 119: the condition is the whole
    # trick, so it is shown in the form the problem will need it.
    ("modular_inverse", "For each n let b(n) be the least positive multiple of 7 that "
     "is congruent to 1 modulo 2^n. How many n at most 300 satisfy b(n) = b(n+1)?",
     '{"solver":"multisearch","variables":[{"name":"n","from":1,"to":300}],'
     '"conditions":["inv_mod(7, pow2(n)) == inv_mod(7, pow2(n+1))"],'
     '"aggregate":"count"}'),
    ("power_mod", "What is the remainder when 3 raised to the 2024th power is divided "
     "by 1000?",
     '{"solver":"modular","kind":"power","base":3,"exponent":2024,"modulus":1000}'),
    ("digit_property", "How many four-digit numbers have digits that add to 9 and are "
     "divisible by 11?",
     '{"solver":"search","domain":{"kind":"range","from":1000,"to":9999},'
     '"conditions":[{"op":"digit_sum_eq","arg":9},{"op":"divisible_by","arg":11}],'
     '"aggregate":"count"}'),
    ("triples", "How many triples of positive integers a < b < c at most 30 satisfy "
     "a^2 + b^2 = c^2?",
     '{"solver":"multisearch","variables":[{"name":"a","from":1,"to":30},'
     '{"name":"b","from":1,"to":30},{"name":"c","from":1,"to":30}],'
     '"ordering":"strict_increasing","conditions":["a*a + b*b == c*c"],'
     '"aggregate":"count"}'),
    ("recurrence", "A sequence starts 3, 7 and each later term is three times the "
     "previous minus the one before. What is the twelfth term, counting the first as "
     "term 0?",
     '{"solver":"recurrence","coefficients":[3,-1],"initial":[3,7],"n":12}'),
    ("polynomial", "The polynomial x^3 - 7x^2 + 14x - 8 has three rational roots. What "
     "is their sum?",
     '{"solver":"polynomial","coefficients":[1,-7,14,-8]}'),
    ("steps", "A shop buys 14 crates of 24 bottles, breaks 17 bottles and sells the "
     "rest in packs of 6. How many full packs is that?",
     '{"solver":"arith","let":{"total":"14*24","left":"total - 17"},'
     '"answer":"left // 6"}'),
]

PROMPT = """Map this competition problem onto ONE solver, or say what is missing. An
exact executor runs your spec — never compute the answer yourself.

Worked mappings of the closest kind:

{examples}

Catalogue:
{catalogue}

For the search solver, conditions use these ops: {preds}
Expressions may call: {funcs}

Problem: {problem}

Reply with ONLY the JSON spec, or ONLY {{"none":"<the one capability that is missing>"}}"""

# Which bank shape each reachable problem actually needs — hand-labelled, so retrieval
# can be scored separately from mapping.
NEEDS = {0: "factorial_divisors", 1: "aggregate_mod", 7: "bits",
         12: "diophantine", 13: "modular_inverse"}


def mask(t):
    """Mask the numbers (phase 88's key) and STRIP LATEX before embedding.

    The encoder returned 17 vectors for 15 problems until the markup came out: a
    LaTeX-dense AIME statement tokenises past the 512-token window and llama-embedding
    silently emits a second row, which would have shifted every vector after it onto
    the wrong problem. Stripping \frac, \binom and the delimiters also makes the key
    what it should be — the WORDS of the problem shape, not its typesetting.
    """
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    t = re.sub(r"[$\\{}\[\]_^]", " ", t)
    t = re.sub(r"[^\x20-\x7e]", " ", t)
    t = re.sub(r"\d+", "N", t)
    return " ".join(t.split())[:280]


def main(k=2, out="data/custom/mapmemory2.json"):
    k = int(k)
    _, aime = load_problems()
    picks = random.Random(5).sample(aime, 30)[:15]
    catalogue = "\n".join(f"- {v}" for v in
                          {**SCHEMA_5, **SCHEMA_4, **SCHEMA_3, **SCHEMA_MORE,
                           **SCHEMA_NEW, **SCHEMAS, **SCHEMAS2}.values())
    bvecs = embed([mask(p) for _tag, p, _s in BANK])
    pvecs = embed([mask(p) for p, _t in picks])

    t = Counter()
    rows = []
    for i, (problem, truth) in enumerate(picks):
        sims = [sum(a * b for a, b in zip(pvecs[i], bv)) for bv in bvecs]
        order = sorted(range(len(BANK)), key=lambda j: -sims[j])[:k]
        shown = [BANK[j][0] for j in order]
        if i in NEEDS:
            t["retrieval_hit" if NEEDS[i] in shown else "retrieval_miss"] += 1
        examples = "\n\n".join(f"{n + 1}. \"{BANK[j][1]}\"\n{BANK[j][2]}"
                               for n, j in enumerate(order))
        spec = parse_spec(ask_spec(PROMPT.format(problem=problem, examples=examples,
                                                 catalogue=catalogue,
                                                 preds=PREDICATE_HELP,
                                                 funcs=EXPR_FUNCS_HELP), n=900))
        row = {"i": i, "truth": str(truth), "shown": shown,
               "reachable": i in HAND_SPECS}
        if spec is None:
            t["no_reply"] += 1
            row["outcome"] = "unparseable"
        elif ("none" in spec and "solver" not in spec) or spec.get("solver") == "none":
            t["declined"] += 1
            t["declined_reachable" if i in HAND_SPECS else "declined_ok"] += 1
            row["outcome"] = "declined"
        else:
            t["claimed"] += 1
            res, why = run2(spec)
            if res is None:
                t["refused"] += 1
                row["outcome"] = f"refused: {why[:44]}"
            else:
                t["ran"] += 1
                ok = str(answer_of(res, spec)) == str(truth)
                t["exact" if ok else "wrong"] += 1
                t["exact_reachable"] += ok and i in HAND_SPECS
                row["outcome"] = f"{answer_of(res, spec)}" + (" EXACT" if ok else
                                                        f" != {truth}")
            row["spec"] = spec
        rows.append(row)
        print(f"{'*' if i in HAND_SPECS else ' '}[{i:>2}] {'+'.join(shown):<28}"
              f"{row['outcome'][:52]}")

    n = len(picks)
    print(f"\nretrieved the needed shape for {t['retrieval_hit']}/{len(NEEDS)} "
          f"reachable problems (missed {t['retrieval_miss']})")
    print(f"claimed {t['claimed']}, ran {t['ran']}, EXACT {t['exact']} "
          f"(of the reachable five: {t['exact_reachable']}), wrong {t['wrong']}, "
          f"refused {t['refused']}, declined {t['declined']}, "
          f"unparseable {t['no_reply']}")
    print(f"baselines on the same fifteen: no examples 0, fixed three examples 1, "
          f"hand-mapped ceiling 5")
    print("\nA fixed prompt cannot cover a growing library. Retrieval makes the prompt")
    print("a function of the problem, and the two numbers above separate its two ways")
    print("of failing: showing the wrong shape, and showing the right one to a mapper")
    print("that still cannot use it.")
    summary = {"n": n, "k": k, **dict(t), "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
