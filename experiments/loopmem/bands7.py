#!/usr/bin/env python3
"""Three last classes: constant acceleration, roman numerals, checksums.

Same rule, same recipe, three more mechanical places a model slips — SUVAT under
fractions, roman numerals past a few hundred, and check digits that require a weighted
sum nobody carries in their head. Twelve problems, truths computed here, two arms.
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

SCHEMA_7 = {
    "kinematics": '{"solver":"kinematics","u":<initial speed>,"v":<final speed>,'
                  '"a":<acceleration>,"s":<distance>,"t":<time>,"find":"u"|"v"|"a"|'
                  '"s"|"t"}  give any three, ask for one',
    "roman": '{"solver":"roman","number":<int 1-3999>} to write a numeral, or '
             '{"solver":"roman","roman":"MCMLXXXVII"} to read one',
    "checksum": '{"solver":"checksum","kind":"luhn"|"isbn10"|"isbn13",'
                '"code":"<the digits>"}',
}

SCHEMA_6 = {
    "equation": '{"solver":"equation","equation":"<the equation exactly as written, '
                'e.g. 3/2x + 7 = 5x - 2>","variable":"x"}',
}

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
    ("kinematics", "A body starts at 4 m/s and accelerates at 3/2 m/s^2 for 6 s. How "
     "far does it travel?",
     '{"solver":"kinematics","u":4,"a":"3/2","t":6,"find":"s"}'),
    ("roman", "Write 2024 as a roman numeral.",
     '{"solver":"roman","number":2024}'),
    ("checksum", "What is the ISBN-13 check digit for 978030640615?",
     '{"solver":"checksum","kind":"isbn13","code":"978030640615"}'),
    ("equation", "Solve 5/3x - 4 = 2x + 1 for x.",
     '{"solver":"equation","equation":"5/3x - 4 = 2x + 1","variable":"x"}'),
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
    """Twelve problems, truths computed here."""
    from fractions import Fraction as _F
    out = []

    def suvat_s(u, a, t, text):
        return ("kinematics", text, str(_F(u) * _F(t) + _F(a) * _F(t) ** 2 / 2))

    out.append(suvat_s("7/3", "5/4", 9, "A body starts at 7/3 m/s and accelerates at "
               "5/4 m/s^2 for 9 seconds. Exactly how far does it travel? Give a "
               "fraction."))
    out.append(suvat_s("-11/6", "8/5", 13, "A body starts at -11/6 m/s and accelerates "
               "at 8/5 m/s^2 for 13 seconds. Exactly how far does it travel? Give a "
               "fraction."))
    out.append(("kinematics", "A body speeds up from 9/2 m/s to 27/4 m/s over 15/8 "
                "seconds. Exactly what is its acceleration? Give a fraction.",
                str((_F(27, 4) - _F(9, 2)) / _F(15, 8))))
    out.append(("kinematics", "A body starts at rest and reaches 45 m/s over 200 "
                "metres. Exactly what is its acceleration? Give a fraction.",
                str(_F(45 * 45, 2 * 200))))

    def roman_of(n):
        table = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
                 (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"),
                 (4, "IV"), (1, "I")]
        o = []
        for v, sym in table:
            while n >= v:
                o.append(sym)
                n -= v
        return "".join(o)

    for n in (3888, 1476, 2947):
        out.append(("roman", f"Write {n} as a roman numeral.", roman_of(n)))
    out.append(("roman", "What number is the roman numeral MCMXCIV?", "1994"))

    def isbn13(body):
        total = sum(int(d) * (3 if i % 2 else 1) for i, d in enumerate(body))
        return str((10 - total % 10) % 10)

    def isbn10(body):
        total = sum((10 - i) * int(d) for i, d in enumerate(body))
        c = (11 - total % 11) % 11
        return "X" if c == 10 else str(c)

    out.append(("checksum", "What is the ISBN-13 check digit for 978147674137?",
                isbn13("978147674137")))
    out.append(("checksum", "What is the ISBN-13 check digit for 978052156781?",
                isbn13("978052156781")))
    out.append(("checksum", "What is the ISBN-10 check digit for 030640615?",
                isbn10("030640615")))
    out.append(("checksum", "Is the card number 4539578763621486 valid under the Luhn "
                "check? Answer True or False.", "True"))
    return out


def main(k=2, out="data/custom/bands7.json"):
    k = int(k)
    battery = build()
    catalogue = "\n".join(f"- {v}" for v in
                          {"arith": ARITH_SCHEMA, **SCHEMA_7, **SCHEMA_6,
                           **SCHEMA_5,
                           **SCHEMA_4,
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
