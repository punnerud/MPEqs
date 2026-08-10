#!/usr/bin/env python3
"""Mapping problems onto the solver library: the model fills a spec, the record runs it.

Phase 92 built the machines; this measures whether a problem in language can be MAPPED
onto one. The model never computes and never writes code — it emits a JSON spec naming
a solver and filling its slots, and the record validates the spec, executes it exactly,
or refuses by name.

Two arms on the same thirty problems (two per solver, distractor quantities throughout,
truths written first):

  FREE     the model sees the whole catalogue and picks the solver itself
  GUIDED   the embedding picks the solver from its wordings (phase 91's lexical face),
           and the model is shown ONLY that solver's schema — its job shrinks from
           choosing-and-filling to filling, which is the division of labour that has
           worked at every scale in this study

Both arms deliver only what runs. The measured quantities are the mapping's real
anatomy: solver chosen right, spec valid, spec runnable, answer exact — and, separately,
how often a WRONG solver choice self-destructs (the slots cannot be filled, so the
record refuses) versus delivers a confident wrong number. A mis-mapping that refuses is
cheap; one that answers is the only expensive kind.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402
from embednav import embed  # noqa: E402
from solvers import WORDINGS, run  # noqa: E402

SCHEMAS = {
    "search": '{"solver":"search","domain":{"kind":"range","from":<int>,"to":<int>}'
              ' | {"kind":"divisors_of","n":<int>} | {"kind":"divisors_of_factorial",'
              '"k":<int>},"conditions":[{"op":<name>,"arg":<value>},...],'
              '"aggregate":"sum"|"count"|"min"|"max"|"product"|"list",'
              '"post":{"op":"mod","arg":<int>} (optional)}',
    "linear_system": '{"solver":"linear_system","rows":[[<coef>,...],...],'
                     '"rhs":[<number>,...]}',
    "quadratic": '{"solver":"quadratic","a":<number>,"b":<number>,"c":<number>}',
    "crt": '{"solver":"crt","residues":[<int>,...],"moduli":[<int>,...]}',
    "gcd_lcm": '{"solver":"gcd_lcm","values":[<int>,...],"report":"gcd"|"lcm"}',
    "series": '{"solver":"series","kind":"arithmetic","first":<n>,"step":<n>,'
              '"terms":<int>} | {"solver":"series","kind":"geometric","first":<n>,'
              '"ratio":<n>,"terms":<int>|"inf"}',
    "combinatorics": '{"solver":"combinatorics","kind":"choose"|"permute"|"factorial"'
                     '|"stars_and_bars"|"derangement","n":<int>,"k":<int>}',
    "recurrence": '{"solver":"recurrence","coefficients":[<n>,...],'
                  '"initial":[<n>,...],"n":<int>}  (a(n)=c1*a(n-1)+c2*a(n-2)+...)',
    "rate_work": '{"solver":"rate_work","times":[<n>,...]}',
    "ratio_split": '{"solver":"ratio_split","total":<n>,"ratio":[<n>,...]}',
    "mixture": '{"solver":"mixture","quantities":[<n>,...],'
               '"concentrations":[<n>,...]}',
    "interest": '{"solver":"interest","principal":<n>,"rate":<n>,"periods":<n>,'
                '"kind":"compound"|"simple"}',
    "base_convert": '{"solver":"base_convert","n":<int>,"base":<int>}',
    "digit_ops": '{"solver":"digit_ops","n":<int>}',
    # The k-versus-n distinction must lead, not trail: when "report" was appended to
    # this line the model started writing {"n":25} for "25 factorial" and the divisor
    # family regressed. A schema is read literally, and a buried distinction is lost.
    "factor": 'FOR A FACTORIAL USE k: {"solver":"factor","k":25} means 25 factorial; '
              'for a plain number use n: {"solver":"factor","n":720}. Add '
              '"report":"divisor_count"|"divisor_sum"|"exponent_sum" to choose which '
              'number you want back.',
}

PREDICATE_HELP = ("divisible_by, not_divisible_by, mod_eq [m,r], is_square, is_cube, "
                  "is_prime, square_free, coprime_to, divides, quotient_is_square, "
                  "quotient_is_cube, digit_sum_eq, digit_count_eq, digits_distinct, "
                  "palindrome, gt, ge, lt, le, eq, ne, in_set")

# (solver, story, expected answer as a string, key of the answer inside the result)
BATTERY = [
    ("search", "The class has 30 pupils. What is the sum of all multiples of 7 "
     "between 1 and 200 whose digits add up to 10?", "273"),
    ("search", "The book has 12 chapters. How many integers from 1 to 500 are "
     "perfect squares?", "22"),
    ("linear_system", "The shop opened in 1998. Two numbers satisfy 2x + 3y = 12 "
     "and x - y = 1. What are x and y?", "['3', '2']"),
    ("linear_system", "There are 4 seasons. Solve x + y = 10 and 3x - y = 14 for x "
     "and y.", "['6', '4']"),
    ("quadratic", "The room has 4 walls. Solve x^2 - 5x + 6 = 0.", "['3', '2']"),
    ("quadratic", "It rained 3 days. Find the roots of 2x^2 - 8x + 6 = 0.",
     "['3', '1']"),
    ("crt", "The team has 11 players. Find the smallest positive number that leaves "
     "remainder 2 modulo 3, remainder 3 modulo 5, and remainder 2 modulo 7.", "23"),
    ("crt", "The bus runs 9 times a day. What number leaves remainder 1 modulo 4 and "
     "remainder 3 modulo 5?", "13"),
    ("gcd_lcm", "The garden has 7 trees. What is the greatest common divisor and the "
     "least common multiple of 12, 18 and 30?", "{'gcd': 6, 'lcm': 180}"),
    ("gcd_lcm", "The car has 4 doors. Two lights blink every 14 and 21 seconds. Give "
     "the gcd and lcm of 14 and 21.", "{'gcd': 7, 'lcm': 42}"),
    ("series", "The hall seats 200. Add up the first 10 terms of the arithmetic "
     "sequence starting at 3 and increasing by 4 each time.", "210"),
    ("series", "The pen costs 20 kr. What is the total of the infinite geometric "
     "series with first term 1 and ratio 1/2?", "2"),
    ("combinatorics", "The shelf holds 50 books. In how many ways can 3 pupils be "
     "chosen from 10?", "120"),
    ("combinatorics", "The week has 7 days. How many ways can 7 identical sweets be "
     "handed out to 3 children, where a child may get none?", "36"),
    ("recurrence", "The tree is 20 years old. A sequence starts 1, 1 and every later "
     "term is the sum of the two before it. What is term number 10, counting the "
     "first term as term 0?", "89"),
    ("recurrence", "The lift holds 8 people. A sequence starts 2, 3 and each later "
     "term is twice the previous plus the one before that. What is term 5, counting "
     "the first as term 0?", "111"),
    ("rate_work", "The wall is 3 m high. One painter finishes in 4 hours and another "
     "in 6 hours. How long do they take together?", "12/5"),
    ("rate_work", "The pool is 2 m deep. One pipe fills it in 3 hours, another in 6 "
     "hours. How long together?", "2"),
    ("ratio_split", "The trip took 5 days. Split 120 kr between three people in the "
     "ratio 2 : 3 : 5.", "['24', '36', '60']"),
    ("ratio_split", "The bag weighs 2 kg. Divide 90 sweets between two children in "
     "the ratio 4 : 5.", "['40', '50']"),
    ("mixture", "The lab has 6 benches. Mix 3 litres at concentration 1/5 with 7 "
     "litres at concentration 1/2. What is the resulting concentration?", "41/100"),
    ("mixture", "The bottle is 1 m tall. Blend 2 kg of 30 percent syrup with 3 kg of "
     "80 percent syrup, concentrations 3/10 and 8/10. What concentration results?",
     "3/5"),
    ("interest", "The bank has 4 branches. 1000 kr grows at rate 1/10 compounded for "
     "3 periods. What is the final amount?", "1331"),
    ("interest", "The office has 12 desks. 500 kr at simple interest rate 1/20 for 4 "
     "periods gives what total?", "600"),
    ("base_convert", "The street has 30 houses. Write the number 100 in base 2.",
     "1100100"),
    ("base_convert", "There are 24 hours in a day. Express 255 in base 16.", "ff"),
    ("digit_ops", "The queue had 15 people. What is the sum of the digits of "
     "9174?", "21"),
    ("digit_ops", "The film lasts 120 minutes. What is the sum of the digits of "
     "58321?", "19"),
    ("factor", "The choir has 40 singers. Give the prime factorisation and the "
     "number of divisors of 360.", "24"),
    ("factor", "The ship has 3 masts. How many divisors does 13! have?", "1584"),
]

FREE = """Map this problem onto ONE solver from the catalogue and fill its slots.
Do not compute the answer — the executor computes it from your spec.

Problem: {story}

Catalogue:
{catalogue}

For the search solver, conditions use these ops: {preds}

Reply with ONLY the JSON spec."""

GUIDED = """Fill this solver's slots from the problem. Do not compute the answer —
the executor computes it from your spec.

Problem: {story}

Solver schema:
{schema}
{extra}
Reply with ONLY the JSON spec."""


def answer_of(res, spec=None):
    """The comparable payload of a result.

    Several machines return a small record rather than one number — gcd AND lcm, the
    divisor count AND the divisor sum — and nothing in the spec used to say which one
    was being asked for, so a right answer inside a dict scored as wrong. A spec may
    now carry "report" (or "op") naming the field; without one the historical default
    stands, so older specs mean exactly what they meant before.
    """
    v = res["value"]
    want = (spec or {}).get("report") or (spec or {}).get("op")
    if isinstance(v, dict) and want and want in v:
        return str(v[want])
    if isinstance(v, dict) and "divisor_count" in v:
        return str(v["divisor_count"])
    if isinstance(v, dict) and "digit_sum" in v:
        return str(v["digit_sum"])
    return str(v)


def parse_spec(reply):
    depth, start = 0, None
    for i, ch in enumerate(reply):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(reply[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    return None


def main(n_inspect=0, out="data/custom/solvemap.json"):
    n_inspect = int(n_inspect)
    catalogue = "\n".join(f"- {v}" for v in SCHEMAS.values())

    wtexts, wowner = [], []
    for name, ws in WORDINGS.items():
        for w in ws:
            wtexts.append(w)
            wowner.append(name)
    wvecs = embed(wtexts)
    svecs = embed([re.sub(r"\d+", "N", s) for _, s, _ in BATTERY])

    if n_inspect:
        for solver, story, truth in (BATTERY[0], BATTERY[15]):
            r = ask("qwen-35b", FREE.format(story=story, catalogue=catalogue,
                                            preds=PREDICATE_HELP), n=220)
            print(f"[{solver}] {story[:60]}\n  RAW free: {r!r}\n")
        return

    tally = {"embed_top1": 0}
    for arm in ("free", "guided"):
        tally.update({f"{arm}_{k}": 0 for k in
                      ("solver_right", "parsed", "ran", "exact", "wrong", "refused")})
    rows = []
    for i, (solver, story, truth) in enumerate(BATTERY):
        sims = [sum(a * b for a, b in zip(svecs[i], wv)) for wv in wvecs]
        prop = wowner[max(range(len(sims)), key=lambda j: sims[j])]
        tally["embed_top1"] += prop == solver

        row = {"solver": solver, "story": story[:52], "embed": prop}
        for arm in ("free", "guided"):
            if arm == "free":
                prompt = FREE.format(story=story, catalogue=catalogue,
                                     preds=PREDICATE_HELP)
            else:
                extra = (f"\nFor conditions use these ops: {PREDICATE_HELP}\n"
                         if prop == "search" else "")
                prompt = GUIDED.format(story=story, schema=SCHEMAS[prop], extra=extra)
            spec = parse_spec(ask("qwen-35b", prompt, n=220))
            if spec is None:
                row[arm] = "no spec"
                continue
            tally[f"{arm}_parsed"] += 1
            chosen = spec.get("solver", prop if arm == "guided" else None)
            spec["solver"] = chosen
            tally[f"{arm}_solver_right"] += chosen == solver
            res, why = run(spec)
            if res is None:
                tally[f"{arm}_refused"] += 1
                row[arm] = f"refused ({why[:38]})"
                continue
            tally[f"{arm}_ran"] += 1
            got = answer_of(res)
            ok = got == truth
            tally[f"{arm}_exact"] += ok
            tally[f"{arm}_wrong"] += not ok
            row[arm] = f"{got}" + ("" if ok else f" != {truth}")
        rows.append(row)
        print(f"{solver:<14} embed {prop:<14} free {str(row.get('free'))[:26]:<28}"
              f"guided {str(row.get('guided'))[:26]}")

    n = len(BATTERY)
    print(f"\nembedding picks the right solver : {tally['embed_top1']}/{n}")
    for arm in ("free", "guided"):
        print(f"{arm.upper():<7} solver right {tally[f'{arm}_solver_right']}/{n}, "
              f"spec parsed {tally[f'{arm}_parsed']}, ran {tally[f'{arm}_ran']}, "
              f"refused {tally[f'{arm}_refused']} -> EXACT {tally[f'{arm}_exact']}, "
              f"wrong {tally[f'{arm}_wrong']}")
    print("\nA mis-mapping that refuses costs nothing; one that answers is the only")
    print("expensive kind. The two arms separate choosing from filling, and the")
    print("record's validation is what turns a bad choice into a refusal instead of")
    print("a confident number.")
    summary = {"n": n, **tally, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
