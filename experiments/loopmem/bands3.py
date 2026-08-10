#!/usr/bin/env python3
"""Three more classes, and the retrieval recipe applied from the start.

Phases 105 and 106 added five classes one battery at a time; phase 108 then showed that
what makes a machine reachable is a RETRIEVED exemplar rather than a catalogue line. So
these three go in the way the recipe says, with a bank entry each from the beginning:

  SHAPE      standard figures with pi kept SYMBOLIC — the area of a circle of radius 5
             is 25*pi, and 78.54 is a different answer to a different question
  INCLUSION  the survey word problem, exactly: unions, neither-counts, up to four sets
  FORMULA    phase 91's formula library, addressable at last. Twelve named relations
             that existed for fifteen phases and could not be REACHED by a mapped
             problem, which is the same lesson the unit router taught in phase 105

Fifteen problems, truths computed before any model call, two arms.
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


def build(hard=False):
    """The easy set first; hard=True keeps the CLASSES and changes only the sizes.

    The easy set came back 15/15 for BOTH arms, which is the phase 105 rule measured
    from its negative side: a machine does not widen the band unless the class is
    actually beyond the model. Same three classes with awkward fractions and large
    numbers then asks the sharper question — is the band about the CLASS or about the
    arithmetic?
    """
    if hard:
        return build_hard_variant()
    out = []
    out.append(("shape", "A circular pond has radius 7 metres. What is its exact area? "
                "Give the answer as a multiple of pi, written like 25*pi.", "49*pi"))
    out.append(("shape", "A sphere has radius 6. What is its exact volume? Give the "
                "answer as a multiple of pi, written like 25*pi.", "288*pi"))
    out.append(("shape", "A box measures 7 by 11 by 13. What is its total surface "
                "area?", str(2 * (7 * 11 + 11 * 13 + 7 * 13))))
    out.append(("shape", "A cone has radius 9 and height 14. What is its exact volume? "
                "Give the answer as a multiple of pi, written like 25*pi.",
                str(F(1, 3) * 81 * 14) + "*pi"))
    out.append(("shape", "A trapezium has parallel sides 13 and 21 and height 9. What "
                "is its exact area? Give a fraction if it is not whole.",
                str(F(13 + 21) * 9 / 2)))

    out.append(("inclusion", "Of 200 people, 118 drink coffee, 94 drink tea and 47 "
                "drink both. How many drink neither?", str(200 - (118 + 94 - 47))))
    out.append(("inclusion", "In a class of 60, 35 study French, 28 study German and "
                "14 study both. How many study at least one of them?",
                str(35 + 28 - 14)))
    out.append(("inclusion", "Of 300 households, 180 have a car, 145 have a bicycle, "
                "97 have a boat, 88 have a car and a bicycle, 55 have a car and a "
                "boat, 42 have a bicycle and a boat, and 27 have all three. How many "
                "have at least one?", str(180 + 145 + 97 - 88 - 55 - 42 + 27)))
    out.append(("inclusion", "Of 500 readers, 260 read fiction, 190 read history and "
                "85 read both. How many read neither?",
                str(500 - (260 + 190 - 85))))
    out.append(("inclusion", "Of 120 pupils, 66 play football, 51 play chess, 40 swim, "
                "29 do football and chess, 21 do football and swimming, 18 do chess "
                "and swimming, and 11 do all three. How many do at least one?",
                str(66 + 51 + 40 - 29 - 21 - 18 + 11)))

    out.append(("formula", "An object of mass 12 kg moves at 7 m/s. What is its "
                "momentum in kg*m/s?", str(12 * 7)))
    out.append(("formula", "An object of mass 12 kg moves at 7 m/s. What is its exact "
                "kinetic energy in joules? Give a fraction if it is not whole.",
                str(F(12 * 7 * 7, 2))))
    out.append(("formula", "A syrup of mass 17 kg fills 4 litres. What is its exact "
                "density in kg per litre? Give a fraction.", str(F(17, 4))))
    out.append(("formula", "A train covers 430 metres in 12 seconds. What is its exact "
                "speed in metres per second? Give a fraction.", str(F(430, 12))))
    out.append(("formula", "A box of 23 pieces costs 391 kroner. What is the exact "
                "price per piece?", str(F(391, 23))))
    return out


def build_hard_variant():
    out = []
    out.append(("shape", "A circular pond has radius 47/6 metres. What is its exact "
                "area? Give it as a fraction times pi, written like 25/4*pi.",
                str(F(47, 6) ** 2) + "*pi"))
    out.append(("shape", "A sphere has radius 23/7. What is its exact volume? Give it "
                "as a fraction times pi, written like 25/4*pi.",
                str(F(4, 3) * F(23, 7) ** 3) + "*pi"))
    out.append(("shape", "A box measures 137 by 249 by 313. What is its total surface "
                "area?", str(2 * (137 * 249 + 249 * 313 + 137 * 313))))
    out.append(("shape", "A cone has radius 31/4 and height 58/9. What is its exact "
                "volume? Give it as a fraction times pi.",
                str(F(1, 3) * F(31, 4) ** 2 * F(58, 9)) + "*pi"))
    out.append(("shape", "A trapezium has parallel sides 227/8 and 419/12 and height "
                "53/6. What is its exact area? Give a fraction.",
                str((F(227, 8) + F(419, 12)) * F(53, 6) / 2)))

    out.append(("inclusion", "Of 41873 people, 24518 drink coffee, 19764 drink tea and "
                "11209 drink both. How many drink neither?",
                str(41873 - (24518 + 19764 - 11209))))
    out.append(("inclusion", "Of 98765 subscribers, 45312 read news, 38907 read sport, "
                "29184 read culture, 21445 read news and sport, 17332 read news and "
                "culture, 14027 read sport and culture, and 9118 read all three. How "
                "many read at least one?",
                str(45312 + 38907 + 29184 - 21445 - 17332 - 14027 + 9118)))
    out.append(("inclusion", "Of 250000 records, 133891 match A, 118447 match B, "
                "77213 match both. How many match neither?",
                str(250000 - (133891 + 118447 - 77213))))
    out.append(("inclusion", "Of 60000 users, 31118 use W, 27409 use X, 19822 use Y, "
                "15003 use W and X, 11276 use W and Y, 9884 use X and Y, and 6117 use "
                "all three. How many use at least one?",
                str(31118 + 27409 + 19822 - 15003 - 11276 - 9884 + 6117)))
    out.append(("inclusion", "Of 7777 tickets, 4321 are A, 3456 are B and 1234 are "
                "both. How many are neither?", str(7777 - (4321 + 3456 - 1234))))

    out.append(("formula", "An object of mass 3719 kg moves at 2843 m/s. What is its "
                "momentum in kg*m/s?", str(3719 * 2843)))
    out.append(("formula", "An object of mass 47/6 kg moves at 23/5 m/s. What is its "
                "exact kinetic energy in joules? Give a fraction.",
                str(F(47, 6) * F(23, 5) ** 2 / 2)))
    out.append(("formula", "A syrup of mass 8641 kg fills 2397 litres. What is its "
                "exact density in kg per litre? Give a fraction.", str(F(8641, 2397))))
    out.append(("formula", "A train covers 41983 metres in 1277 seconds. What is its "
                "exact speed in metres per second? Give a fraction.",
                str(F(41983, 1277))))
    out.append(("formula", "A box of 3187 pieces costs 918456 kroner. What is the "
                "exact price per piece? Give a fraction.", str(F(918456, 3187))))
    return out


def main(k=2, hard=0, out=None):
    k, hard = int(k), int(hard)
    out = out or (f"data/custom/bands3{'_hard' if hard else ''}.json")
    battery = build(bool(hard))
    catalogue = "\n".join(f"- {v}" for v in
                          {"arith": ARITH_SCHEMA, **SCHEMA_3, **SCHEMA_MORE,
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
