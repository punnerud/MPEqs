#!/usr/bin/env python3
"""Four more: base arithmetic, rational approximation, word counting, primes.

Same rule, same recipe, four more places a model fails for a mechanical reason:

  BASEARITH  adding in base seven is a carry chain a model does by translating to
             decimal in its head and back, which is where it slips
  APPROX     the closest fraction under a denominator cap, and continued fractions
  STRCOUNT   distinct arrangements of a word with repeated letters
  PRIMES     counting primes below a bound, the nth prime, the next prime

Sixteen problems, truths computed here before any model call, two arms.
"""


import json
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aimecover import EXPR_FUNCS_HELP, SCHEMAS2  # noqa: E402
from bands2 import SCHEMA_MORE  # noqa: E402
from cutbig import ask  # noqa: E402
from embednav import embed  # noqa: E402
from gsmsolve import ARITH_SCHEMA, ask_spec_model, equal  # noqa: E402
from mapmemory import mask  # noqa: E402
from newbands import SCHEMA_NEW  # noqa: E402
from olympiad import SOLO, last_number  # noqa: E402
from solvemap import PREDICATE_HELP, SCHEMAS, answer_of, parse_spec  # noqa: E402
from solvers2 import run2  # noqa: E402

SCHEMA_5 = {
    "basearith": '{"solver":"basearith","op":"add"|"subtract"|"multiply"|"convert",'
                 '"values":["<numerals written in from_base>",...],"from_base":<int>,'
                 '"to_base":<int>}  (convert takes one value)',
    "approx": '{"solver":"approx","kind":"best_rational","value":"<a number or '
              'fraction>","max_denominator":<int>} | {"kind":"continued_fraction",'
              '"value":"<number>","terms":<int>} | {"kind":"convergents",...}',
    "strcount": '{"solver":"strcount","word":"<the word>","kind":"arrangements"|'
                '"letter_count"|"distinct_letters"|"is_palindrome","letter":"<a>"}',
    "primes": '{"solver":"primes","kind":"count"|"sum","from":<int>,"to":<int>} | '
              '{"kind":"nth","n":<int>} | {"kind":"next","value":<int>}',
}

SCHEMA_4 = {
    "sequence": '{"solver":"sequence","terms":[<the given terms in order>],'
                '"n":<index to report, counting the FIRST given term as index 0>}',
    "matrix": '{"solver":"matrix","kind":"determinant"|"inverse","matrix":[[..],[..]]} '
              '| {"kind":"multiply","a":[[..]],"b":[[..]]} | {"kind":"power",'
              '"matrix":[[..]],"exponent":<int>}',
    "partition": '{"solver":"partition","kind":"unordered"|"ordered"|"distinct",'
                 '"total":<int>,"parts":[<allowed part sizes>]}  unordered = coins in '
                 'any number, ordered = order matters, distinct = each part at most '
                 'once',
    "logexp": '{"solver":"logexp","kind":"digits_of_power","base":<int>,'
              '"exponent":<int>} | {"kind":"trailing_zeros_factorial","n":<int>} | '
              '{"kind":"integer_log","base":<int>,"value":<int>}',
}

SCHEMA_3 = {
    "shape": '{"solver":"shape","shape":"circle"|"rectangle"|"triangle"|"trapezium"|'
             '"cylinder"|"sphere"|"cone"|"cube"|"box", plus its measurements by name '
             '(radius, length, width, height, base, side, a, b), '
             '"report":"area"|"perimeter"|"volume"|"surface"}  answers keep pi exact',
    "inclusion_exclusion": '{"solver":"inclusion_exclusion","sizes":{"a":<n>,"b":<n>,'
                           '"a&b":<n>,...},"total":<n, optional>,'
                           '"report":"union"|"neither"}',
    "formula": '{"solver":"formula","name":"speed"|"distance"|"momentum"|"kinetic"|'
               '"density"|"mass_dv"|"area"|"perimeter"|"gap"|"mean_side"|"price"|'
               '"unitprice","args":[<the formula\'s inputs in order>]}',
}

BANK = [
    ("basearith", "What is 3412 plus 2033 in base 5, written in base 5?",
     '{"solver":"basearith","op":"add","values":["3412","2033"],"from_base":5,'
     '"to_base":5}'),
    ("approx", "What is the closest fraction to 141/100 with denominator at most 30?",
     '{"solver":"approx","kind":"best_rational","value":"141/100",'
     '"max_denominator":30}'),
    ("strcount", "How many distinct arrangements are there of the letters of BANANA?",
     '{"solver":"strcount","word":"BANANA","kind":"arrangements"}'),
    ("primes", "How many prime numbers are there below 500?",
     '{"solver":"primes","kind":"count","from":1,"to":499}'),
    ("sequence", "A sequence begins 5, 11, 21, 35, 53. What is the term at index 12, "
     "counting the first as index 0?",
     '{"solver":"sequence","terms":[5,11,21,35,53],"n":12}'),
    ("matrix", "What is the determinant of [[2,5],[7,3]]?",
     '{"solver":"matrix","kind":"determinant","matrix":[[2,5],[7,3]]}'),
    ("partition", "In how many ways can 60 be made from coins of 1, 2, 5 and 10?",
     '{"solver":"partition","kind":"unordered","total":60,"parts":[1,2,5,10]}'),
    ("logexp", "How many digits does 3 to the power 500 have?",
     '{"solver":"logexp","kind":"digits_of_power","base":3,"exponent":500}'),
    ("shape", "A cylinder has radius 4 and height 9. What is its exact volume?",
     '{"solver":"shape","shape":"cylinder","radius":4,"height":9,'
     '"report":"volume"}'),
    ("inclusion", "Of 80 pupils, 44 play football, 31 play chess and 17 play both. How "
     "many play neither?",
     '{"solver":"inclusion_exclusion","sizes":{"a":44,"b":31,"a&b":17},"total":80,'
     '"report":"neither"}'),
    ("formula", "An object of mass 9 kg moves at 4 m/s. What is its kinetic energy?",
     '{"solver":"formula","name":"kinetic","args":[9,4]}'),
]

PROMPT = """Map the problem onto ONE solver and fill its slots. Do NOT compute the
answer — an exact executor computes it from your spec.

{examples}

Catalogue:
{catalogue}

For the search solver, conditions use these ops: {preds}
Expressions may call: {funcs}   (^ is a power; / is exact rational division)

Problem: {story}
Spec:"""


def build():
    """Truths computed here, before any model call."""
    import math
    out = []

    def base_add(a, b, base, text):
        va = int(str(a), base)
        vb = int(str(b), base)
        v, digs = va + vb, []
        while v:
            digs.append("0123456789abcdefghijklmnopqrstuvwxyz"[v % base])
            v //= base
        return ("basearith", text, "".join(reversed(digs)) or "0")

    out.append(base_add("46135", "25264", 7,
                        "What is 46135 plus 25264 in base 7? Give the answer in "
                        "base 7."))
    out.append(base_add("110110101", "101101011", 2,
                        "What is 110110101 plus 101101011 in binary? Give the answer "
                        "in binary."))
    out.append(("basearith", "What is 3fa9c in base 16, written in base 8?",
                oct(int("3fa9c", 16))[2:]))
    def render(v, base):
        digs = []
        while v:
            digs.append("0123456789abcdefghijklmnopqrstuvwxyz"[v % base])
            v //= base
        return "".join(reversed(digs)) or "0"

    out.append(("basearith", "What is 8172 times 341 in base 9, written in base 9? "
                "Both numbers are written in base 9.",
                render(int("8172", 9) * int("341", 9), 9)))
    out.append(("approx", "What is the closest fraction to 3126535/995207 with "
                "denominator at most 200? Give it as a fraction.",
                str(F(3126535, 995207).limit_denominator(200))))
    out.append(("approx", "What is the closest fraction to 271828/100000 with "
                "denominator at most 60? Give it as a fraction.",
                str(F(271828, 100000).limit_denominator(60))))
    out.append(("approx", "What is the closest fraction to 1414213/1000000 with "
                "denominator at most 99? Give it as a fraction.",
                str(F(1414213, 1000000).limit_denominator(99))))
    out.append(("approx", "What is the closest fraction to 16180339/10000000 with "
                "denominator at most 150? Give it as a fraction.",
                str(F(16180339, 10000000).limit_denominator(150))))

    def arrangements(word):
        letters = [c for c in word.lower() if c.isalnum()]
        t = math.factorial(len(letters))
        counts = {}
        for c in letters:
            counts[c] = counts.get(c, 0) + 1
        for c in counts.values():
            t //= math.factorial(c)
        return t

    for w in ("ABRACADABRA", "COMBINATORICS", "PARALLELOGRAM", "MITOCHONDRIA"):
        out.append(("strcount", f"How many distinct arrangements are there of the "
                    f"letters of {w}?", str(arrangements(w))))

    def sieve_count(hi):
        sv = bytearray([1]) * (hi + 1)
        sv[0:2] = b"\x00\x00"
        for i in range(2, int(hi ** 0.5) + 1):
            if sv[i]:
                sv[i * i::i] = bytearray(len(sv[i * i::i]))
        return sv

    sv = sieve_count(200000)
    out.append(("primes", "How many prime numbers are there below 200000?",
                str(sum(sv[:200000]))))
    out.append(("primes", "What is the sum of all prime numbers below 20000?",
                str(sum(i for i in range(2, 20000) if sv[i]))))
    out.append(("primes", "What is the 5000th prime number?",
                str([i for i in range(2, 60000) if sv[i]][4999])))
    out.append(("primes", "What is the smallest prime number greater than 999983?",
                str(next(v for v in range(999984, 1000100)
                         if all(v % d for d in range(2, int(v ** 0.5) + 1))))))
    return out


def main(k=2, out="data/custom/bands5.json"):
    k = int(k)
    battery = build()
    catalogue = "\n".join(f"- {v}" for v in
                          {"arith": ARITH_SCHEMA, **SCHEMA_5, **SCHEMA_4,
                           **SCHEMA_3,
                           **SCHEMA_MORE,
                           **SCHEMA_NEW, **SCHEMAS, **SCHEMAS2}.values())
    bvecs = embed([mask(p) for _t, p, _s in BANK])
    pvecs = embed([mask(s) for _f, s, _t in battery])

    t = {key: 0 for key in ("solo", "mpeqs", "parsed", "ran", "wrong")}
    byfam = {}
    rows = []
    for i, (fam, story, truth) in enumerate(battery):
        sims = [sum(a * b for a, b in zip(pvecs[i], bv)) for bv in bvecs]
        order = sorted(range(len(BANK)), key=lambda j: -sims[j])[:k]
        examples = "\n\n".join('Example: "' + BANK[j][1] + '"\nSpec: ' + BANK[j][2]
                               for j in order)
        raw = ask("qwen-35b", SOLO.format(problem=story), n=420)
        num = last_number(raw)
        solo_ok = (equal(num, truth) if num is not None else False) or \
            (not str(truth).replace("-", "").isdigit() and str(truth) in raw)
        t["solo"] += solo_ok

        spec = parse_spec(ask_spec_model(
            "qwen-35b", PROMPT.format(story=story, catalogue=catalogue,
                                      examples=examples, preds=PREDICATE_HELP,
                                      funcs=EXPR_FUNCS_HELP), n=420))
        got, ok = None, False
        if isinstance(spec, dict) and "solver" in spec:
            t["parsed"] += 1
            res, why = run2(spec)
            if res is not None:
                t["ran"] += 1
                got = answer_of(res, spec)
                ok = str(got) == str(truth) or equal(got, truth)
                t["mpeqs"] += ok
                t["wrong"] += not ok
            else:
                got = why[:36]
        f = byfam.setdefault(fam, [0, 0, 0])
        f[0] += 1
        f[1] += solo_ok
        f[2] += ok
        rows.append({"family": fam, "truth": str(truth), "solo_ok": bool(solo_ok),
                     "mpeqs": str(got), "mpeqs_ok": bool(ok), "spec": spec})
        print(f"{fam:<11}{'solo ok' if solo_ok else 'solo X '} "
              f"{'mpeqs ok' if ok else 'mpeqs X '} {str(truth)[:14]:<16}{story[:40]}")

    n = len(battery)
    print(f"\nSOLO-35B : {t['solo']}/{n}")
    print(f"MPEqs    : {t['mpeqs']}/{n}  (parsed {t['parsed']}, ran {t['ran']}, "
          f"wrong {t['wrong']})")
    for fam, (cnt, so, mp) in sorted(byfam.items()):
        print(f"  {fam:<12}{cnt:>3}{so:>6}{mp:>7}")
    print("\nThree classes added the way the recipe says: a machine, a schema line that")
    print("names its units, and an exemplar in the bank from the first minute. The")
    print("formula library was written fifteen phases ago and answered nothing until")
    print("today, because nothing could address it.")
    summary = {"n": n, **t, "byfam": byfam, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
