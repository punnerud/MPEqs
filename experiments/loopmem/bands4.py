#!/usr/bin/env python3
"""Four more classes: sequences, matrices, partitions, logarithms.

The growth loop has a rule now (mechanical where the model is weak, one sentence where
the mapping is strong) and a recipe (a machine, a schema line that names its units, a
retrieved exemplar). These four go in under both, chosen where a model reliably fails:

  SEQUENCE   fit the rule from the terms and extrapolate — and REFUSE when no rule
             explains every term, because a rule that fits four of five is not the rule
  MATRIX     determinants, inverses, products and powers in exact Fractions
  PARTITION  the counting DP a bounded search cannot reach: ways to make a total
  LOGEXP     digits of huge powers, trailing zeros of factorials, exact integer logs

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

    def seq_quad(a, b, c, n, text):
        terms = [a * k * k + b * k + c for k in range(5)]
        return ("sequence", text.format(t=", ".join(str(x) for x in terms)),
                str(a * n * n + b * n + c))

    out.append(seq_quad(3, 2, 5, 25,
                        "A sequence begins {t}. What is the term at index 25, "
                        "counting the first given term as index 0?"))
    out.append(seq_quad(1, -4, 9, 40,
                        "A sequence begins {t}. What is the term at index 40, "
                        "counting the first given term as index 0?"))
    fib = [4, 7]
    while len(fib) < 5:
        fib.append(3 * fib[-1] - fib[-2])
    ext = list(fib)
    while len(ext) <= 22:
        ext.append(3 * ext[-1] - ext[-2])
    out.append(("sequence", "A sequence begins " + ", ".join(str(x) for x in fib)
                + ". What is the term at index 22, counting the first as index 0?",
                str(ext[22])))
    geo = [7 * 3 ** k for k in range(4)]
    out.append(("sequence", "A sequence begins " + ", ".join(str(x) for x in geo)
                + ". What is the term at index 15, counting the first as index 0?",
                str(7 * 3 ** 15)))

    m = [[4, 7, 2], [9, 1, 8], [3, 6, 5]]
    det = (4 * (1 * 5 - 8 * 6) - 7 * (9 * 5 - 8 * 3) + 2 * (9 * 6 - 1 * 3))
    out.append(("matrix", f"What is the determinant of {m}?", str(det)))
    m2 = [[13, 29, 7, 4], [5, 18, 22, 11], [31, 2, 16, 27], [8, 24, 3, 19]]

    def det4(mm):
        from fractions import Fraction as _F
        mm = [[_F(x) for x in row] for row in mm]
        n, out_ = len(mm), _F(1)
        for c in range(n):
            piv = next((r for r in range(c, n) if mm[r][c] != 0), None)
            if piv is None:
                return 0
            if piv != c:
                mm[c], mm[piv] = mm[piv], mm[c]
                out_ = -out_
            out_ *= mm[c][c]
            inv = 1 / mm[c][c]
            mm[c] = [x * inv for x in mm[c]]
            for r in range(c + 1, n):
                if mm[r][c]:
                    f_ = mm[r][c]
                    mm[r] = [x - f_ * y for x, y in zip(mm[r], mm[c])]
        return out_

    out.append(("matrix", f"What is the determinant of {m2}?", str(det4(m2))))
    fibm = [[1, 1], [1, 0]]
    a, b = 1, 0
    for _ in range(30):
        a, b = a + b, a
    out.append(("matrix", f"What is the top-left entry of {fibm} raised to the power "
                f"30?", str(a)))
    out.append(("matrix", "What is the determinant of [[12, 35], [47, 19]]?",
                str(12 * 19 - 35 * 47)))

    def count_partitions(total, parts, kind="unordered"):
        dp = [0] * (total + 1)
        dp[0] = 1
        if kind == "unordered":
            for p_ in parts:
                for v in range(p_, total + 1):
                    dp[v] += dp[v - p_]
        elif kind == "ordered":
            for v in range(1, total + 1):
                dp[v] = sum(dp[v - p_] for p_ in parts if p_ <= v)
        else:
            for p_ in parts:
                for v in range(total, p_ - 1, -1):
                    dp[v] += dp[v - p_]
        return dp[total]

    out.append(("partition", "In how many ways can 200 kroner be paid with coins of 1, "
                "5, 10, 20 and 50 kroner, if the order does not matter?",
                str(count_partitions(200, [1, 5, 10, 20, 50]))))
    out.append(("partition", "In how many ordered ways can 30 be written as a sum of "
                "1s, 2s and 3s, where the order of the parts matters?",
                str(count_partitions(30, [1, 2, 3], "ordered"))))
    out.append(("partition", "In how many ways can 75 be written as a sum of distinct "
                "numbers chosen from 1 to 20?",
                str(count_partitions(75, list(range(1, 21)), "distinct"))))
    out.append(("partition", "In how many ways can 143 be made from parts of 7, 11 and "
                "13, if the order does not matter?",
                str(count_partitions(143, [7, 11, 13]))))

    out.append(("logexp", "How many digits does 7 to the power 1234 have?",
                str(len(str(7 ** 1234)))))
    out.append(("logexp", "How many trailing zeros does 2025 factorial have?",
                str(sum(2025 // 5 ** k for k in range(1, 6)))))
    out.append(("logexp", "How many digits does 12 to the power 890 have?",
                str(len(str(12 ** 890)))))
    out.append(("logexp", "How many trailing zeros does 10000 factorial have?",
                str(sum(10000 // 5 ** k for k in range(1, 7)))))
    return out


def main(k=2, out="data/custom/bands4.json"):
    k = int(k)
    battery = build()
    catalogue = "\n".join(f"- {v}" for v in
                          {"arith": ARITH_SCHEMA, **SCHEMA_4, **SCHEMA_3,
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
