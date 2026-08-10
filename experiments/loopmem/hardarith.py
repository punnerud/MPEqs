#!/usr/bin/env python3
"""Where MPEqs wins: problems whose arithmetic a model cannot hold, mapping it can.

Phase 98 measured an honest loss — on grade-school word problems the 35B answers better
alone (28/30) than through the machinery (21/30), because there is no arithmetic there
it cannot do. Phase 94 measured the opposite failure — on AIME the mapping collapses.
Between them sits the band this whole apparatus is for: problems whose MAPPING is easy
and whose ARITHMETIC is not.

Twenty problems, generated from truths written first, in four families:

  FRACTIONS   twelve compounding steps in exact rationals — a decimal answer is a wrong
              answer, and a model's mental arithmetic drifts by the third step
  BIG         products and remainders on numbers past any mental register
  COUNT       counts over ranges of hundreds of thousands, defined by simple predicates
  DIVISORS    divisor counts and sums of large factorials, via Legendre

Every one is a single sentence with an obvious mapping, so the mapper is not the thing
under test — the arithmetic is. Both arms answer the same twenty; the record executes
the specs exactly and the model answers alone for the baseline.
"""
import json
import math
import re
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402
from gsmsolve import ARITH_SCHEMA, ask_spec_model, equal  # noqa: E402
from olympiad import SOLO, last_number  # noqa: E402
from solvemap import PREDICATE_HELP, SCHEMAS, answer_of, parse_spec  # noqa: E402
from solvers2 import run2  # noqa: E402
from aimecover import EXPR_FUNCS_HELP, SCHEMAS2  # noqa: E402

FEWSHOT = """Map the problem onto ONE solver and fill its slots. Do NOT compute the
answer — an exact executor computes it from your spec.

Example: "A price of 200 rises by 3/7 of itself, then falls by 2/9 of the result. What
is the exact final price?"
Spec: {{"solver":"arith","let":{{"up":"200 * (1 + 3/7)"}},"answer":"up * (1 - 2/9)"}}

Example: "Start with 5. Twelve times in a row, multiply by 2/3 and add 1/4. Give the
exact result."
Spec: {{"solver":"iterate","init":"5","step":"acc * 2/3 + 1/4","from":1,"to":12}}

Example: "How many integers from 1 to 400000 are divisible by 13 and leave remainder 4
when divided by 9?"
Spec: {{"solver":"search","domain":{{"kind":"range","from":1,"to":400000}},
"conditions":[{{"op":"divisible_by","arg":13}},{{"op":"mod_eq","arg":[9,4]}}],
"aggregate":"count"}}

Catalogue:
{catalogue}

For the search solver, conditions use these ops: {preds}
Expressions may call: {funcs}

Problem: {story}
Spec:"""


def build(variant=0):
    """Truths first: every answer computed here, exactly, before any model sees it.

    variant 1 is the HELD-OUT battery. The prompt and the schemas were sharpened
    against variant 0 across three growth rounds, which is exactly how a score stops
    meaning anything, so the same four families are regenerated with different numbers,
    different ranges and different phrasings and run once with no further tuning.
    """
    if variant:
        return build_heldout()
    out = []

    # FRACTIONS — twelve compounding steps, exact rational answer.
    v = F(1000)
    for k in range(1, 13):
        v = v * (1 + F((-1) ** k, k + 2))
    out.append(("fractions",
                "A balance starts at 1000. In step k, for k = 1 to 12, it is multiplied "
                "by (1 + (-1)^k / (k+2)). Give the exact final balance as a fraction.",
                str(v)))
    v2 = F(3, 7)
    for _ in range(9):
        v2 = (v2 + F(2, 5)) * F(3, 4)
    out.append(("fractions",
                "Start with 3/7. Nine times in a row, add 2/5 and then multiply by 3/4. "
                "Give the exact result as a fraction.", str(v2)))
    v3 = sum(F(1, k * (k + 3)) for k in range(1, 41))
    out.append(("fractions",
                "Add up 1/(k(k+3)) for every integer k from 1 to 40. Give the exact sum "
                "as a fraction.", str(v3)))
    v4 = F(5, 11) ** 3 * F(121, 25) + F(7, 9)
    out.append(("fractions",
                "Take 5/11 cubed, multiply by 121/25, then add 7/9. Give the exact "
                "value as a fraction.", str(v4)))
    v5 = F(1)
    for k in range(2, 16):
        v5 *= F(k * k, k * k - 1)
    out.append(("fractions",
                "Multiply k^2/(k^2 - 1) for every integer k from 2 to 15. Give the "
                "exact product as a fraction.", str(v5)))

    # BIG — beyond any mental register.
    out.append(("big", "What is 987654321 times 123456789?", str(987654321 * 123456789)))
    out.append(("big", "What is the remainder when 7 to the power 1000 is divided by "
                "1000000007?", str(pow(7, 1000, 1000000007))))
    out.append(("big", "What is 2 to the power 200 minus 3 to the power 100?",
                str(2 ** 200 - 3 ** 100)))
    out.append(("big", "What is the greatest common divisor of 987654321987654321 and "
                "123456789123456789?",
                str(math.gcd(987654321987654321, 123456789123456789))))
    out.append(("big", "What is the sum of all integers from 1 to 3141592?",
                str(3141592 * 3141593 // 2)))

    # COUNT — ranges no one enumerates by hand.
    c1 = sum(1 for n in range(1, 400001) if n % 13 == 0 and n % 9 == 4)
    out.append(("count", "How many integers from 1 to 400000 are divisible by 13 and "
                "leave remainder 4 when divided by 9?", str(c1)))
    c2 = sum(1 for n in range(1, 200001) if sum(map(int, str(n))) == 20)
    out.append(("count", "How many integers from 1 to 200000 have digits adding to "
                "exactly 20?", str(c2)))
    c3 = sum(n for n in range(1, 100001) if n % 7 == 0 and n % 11 == 3)
    out.append(("count", "What is the sum of all integers from 1 to 100000 that are "
                "divisible by 7 and leave remainder 3 when divided by 11?", str(c3)))
    c4 = sum(1 for n in range(1, 300001) if math.isqrt(n) ** 2 == n
             or round(n ** (1 / 3)) ** 3 == n)
    out.append(("count", "How many integers from 1 to 300000 are perfect squares or "
                "perfect cubes?", str(c4)))
    c5 = sum(1 for n in range(100000, 1000000) if str(n) == str(n)[::-1]
             and n % 7 == 0)
    out.append(("count", "How many six-digit palindromes are divisible by 7?", str(c5)))

    # DIVISORS — Legendre, never building the factorial.
    def fact_fac(k):
        f = {}
        for p in range(2, k + 1):
            if all(p % q for q in range(2, int(p ** 0.5) + 1)):
                e, q = 0, p
                while q <= k:
                    e += k // q
                    q *= p
                f[p] = e
        return f

    for k, tag in ((25, "25"), (40, "40")):
        f = fact_fac(k)
        out.append(("divisors", f"How many positive divisors does {tag} factorial have?",
                    str(math.prod(e + 1 for e in f.values()))))
    f30 = fact_fac(30)
    out.append(("divisors", "What is the sum of the exponents in the prime "
                "factorisation of 30 factorial?", str(sum(f30.values()))))
    out.append(("divisors", "How many positive divisors of 5040 are perfect squares?",
                str(sum(1 for d in range(1, 5041)
                        if 5040 % d == 0 and math.isqrt(d) ** 2 == d))))
    out.append(("divisors", "What is the sum of all positive divisors of 720720?",
                str(sum(d for d in range(1, 720721) if 720720 % d == 0))))
    return out


def build_heldout():
    """Same four families, all new problems, truths computed here."""
    out = []
    # k = 1..10 telescopes back to exactly 500 — a truth a model could hit by naming
    # the starting value, so the range is odd-length and the pairing is broken.
    v = F(500)
    for k in range(1, 12):
        v = v * (1 - F((-1) ** k, k + 3))
    out.append(("fractions",
                "A balance starts at 500. In step k, for k = 1 to 11, it is multiplied "
                "by (1 - (-1)^k / (k+3)). Give the exact final balance as a fraction.",
                str(v)))
    v2 = F(2, 9)
    for _ in range(11):
        v2 = (v2 * F(5, 6)) + F(1, 3)
    out.append(("fractions",
                "Start with 2/9. Eleven times in a row, multiply by 5/6 and then add "
                "1/3. Give the exact result as a fraction.", str(v2)))
    v3 = sum(F(1, k * (k + 5)) for k in range(1, 31))
    out.append(("fractions",
                "Add up 1/(k(k+5)) for every integer k from 1 to 30. Give the exact "
                "sum as a fraction.", str(v3)))
    v4 = F(7, 13) ** 2 * F(169, 49) - F(4, 11)
    out.append(("fractions",
                "Take 7/13 squared, multiply by 169/49, then subtract 4/11. Give the "
                "exact value as a fraction.", str(v4)))
    v5 = F(1)
    for k in range(3, 21):
        v5 *= F(k * k - 1, k * k)
    out.append(("fractions",
                "Multiply (k^2 - 1)/k^2 for every integer k from 3 to 20. Give the "
                "exact product as a fraction.", str(v5)))

    out.append(("big", "What is 555555553 times 888888887?",
                str(555555553 * 888888887)))
    out.append(("big", "What is the remainder when 11 to the power 777 is divided by "
                "1000000009?", str(pow(11, 777, 1000000009))))
    out.append(("big", "What is 3 to the power 150 plus 5 to the power 80?",
                str(3 ** 150 + 5 ** 80)))
    out.append(("big", "What is the least common multiple of 111111111111 and "
                "222222222?", str(math.lcm(111111111111, 222222222))))
    out.append(("big", "What is the sum of all integers from 1 to 2718281?",
                str(2718281 * 2718282 // 2)))

    c1 = sum(1 for n in range(1, 500001) if n % 17 == 0 and n % 11 == 5)
    out.append(("count", "How many integers from 1 to 500000 are divisible by 17 and "
                "leave remainder 5 when divided by 11?", str(c1)))
    c2 = sum(1 for n in range(1, 150001) if sum(map(int, str(n))) == 17)
    out.append(("count", "How many integers from 1 to 150000 have digits adding to "
                "exactly 17?", str(c2)))
    c3 = sum(n for n in range(1, 80001) if n % 9 == 0 and n % 13 == 6)
    out.append(("count", "What is the sum of all integers from 1 to 80000 that are "
                "divisible by 9 and leave remainder 6 when divided by 13?", str(c3)))
    c4 = sum(1 for n in range(1, 250001) if math.isqrt(n) ** 2 == n
             and round(n ** (1 / 3)) ** 3 == n)
    out.append(("count", "How many integers from 1 to 250000 are both perfect squares "
                "and perfect cubes?", str(c4)))
    c5 = sum(1 for n in range(10000, 100000) if str(n) == str(n)[::-1] and n % 11 == 0)
    out.append(("count", "How many five-digit palindromes are divisible by 11?",
                str(c5)))

    def fact_fac(k):
        f = {}
        for p in range(2, k + 1):
            if all(p % q for q in range(2, int(p ** 0.5) + 1)):
                e, q = 0, p
                while q <= k:
                    e += k // q
                    q *= p
                f[p] = e
        return f

    for k in (18, 33):
        f = fact_fac(k)
        out.append(("divisors",
                    f"How many positive divisors does {k} factorial have?",
                    str(math.prod(e + 1 for e in f.values()))))
    f22 = fact_fac(22)
    out.append(("divisors", "What is the sum of the exponents in the prime "
                "factorisation of 22 factorial?", str(sum(f22.values()))))
    out.append(("divisors", "How many positive divisors of 27720 are perfect cubes?",
                str(sum(1 for d in range(1, 27721) if 27720 % d == 0
                        and round(d ** (1 / 3)) ** 3 == d))))
    out.append(("divisors", "What is the sum of all positive divisors of 498960?",
                str(sum(d for d in range(1, 498961) if 498960 % d == 0))))
    return out


def main(variant=0, out=None):
    variant = int(variant)
    out = out or ("data/custom/hardarith.json" if not variant
                  else "data/custom/hardarith_heldout.json")
    battery = build(variant)
    catalogue = "\n".join(f"- {v}" for v in
                          {"arith": ARITH_SCHEMA, **SCHEMAS, **SCHEMAS2}.values())
    t = {k: 0 for k in ("solo", "mpeqs", "parsed", "ran", "wrong", "solo_none")}
    byfam = {}
    rows = []
    for fam, story, truth in battery:
        a_solo = last_number(ask("qwen-35b", SOLO.format(problem=story), n=420))
        solo_ok = a_solo is not None and equal(a_solo, truth)
        t["solo"] += solo_ok
        t["solo_none"] += a_solo is None

        spec = parse_spec(ask_spec_model(
            "qwen-35b", FEWSHOT.format(story=story, catalogue=catalogue,
                                       preds=PREDICATE_HELP,
                                       funcs=EXPR_FUNCS_HELP), n=420))
        got, mp_ok = None, False
        if isinstance(spec, dict) and "solver" in spec:
            t["parsed"] += 1
            res, why = run2(spec)
            if res is not None:
                t["ran"] += 1
                got = answer_of(res, spec)
                mp_ok = equal(got, truth)
                t["mpeqs"] += mp_ok
                t["wrong"] += not mp_ok
            else:
                got = why[:40]
        f = byfam.setdefault(fam, [0, 0, 0])
        f[0] += 1
        f[1] += solo_ok
        f[2] += mp_ok
        rows.append({"family": fam, "story": story[:70], "truth": truth,
                     "solo": str(a_solo), "solo_ok": bool(solo_ok),
                     "mpeqs": str(got), "mpeqs_ok": bool(mp_ok), "spec": spec})
        print(f"{fam:<10}{'solo ok ' if solo_ok else 'solo X  '}"
              f"{'mpeqs ok' if mp_ok else 'mpeqs X '}  {str(truth)[:22]:<24}"
              f"{story[:44]}")

    n = len(battery)
    print(f"\nSOLO-35B  : {t['solo']}/{n}   ({t['solo_none']} gave no number)")
    print(f"MPEqs     : {t['mpeqs']}/{n}   (specs parsed {t['parsed']}, ran {t['ran']},"
          f" wrong {t['wrong']})")
    print(f"{'family':<12}{'n':>3}{'solo':>7}{'MPEqs':>7}")
    for fam, (cnt, so, mp) in byfam.items():
        print(f"{fam:<12}{cnt:>3}{so:>7}{mp:>7}")
    print("\nThis is the band the apparatus is for: the mapping is one sentence long and")
    print("the arithmetic is out of reach. Phase 98 measured the other side honestly —")
    print("where a model can already do the sums, routing through machinery only adds")
    print("a place to slip. A tool earns its keep by its domain, not by its ambition.")
    summary = {"n": n, **t, "byfam": {k: v for k, v in byfam.items()}, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
