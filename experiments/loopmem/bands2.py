#!/usr/bin/env python3
"""Three more classes: statistics, calendar arithmetic, and percentage chains.

Phase 105 widened the band by two classes and the ratio held (17/20 against 1/20). The
same test is applied to three more, chosen by the same rule — a model fails them for a
mechanical reason while the mapping stays one sentence:

  STATISTICS  exact mean, median and variance. A model produces a plausible decimal;
              the answer is a fraction, and the two are not the same answer.
  DATETIME    days between dates, weekdays, leap years. The classic thing a language
              model cannot do and a record does in a line.
  FINANCE     successive percentage changes and annuity payments. Percentages do not
              add, and every model that says "up 20 then down 20 is back to par" is
              making the record's case.

Eighteen problems, truths computed here before any model call, both arms as before.
"""
import datetime as dt
import json
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aimecover import EXPR_FUNCS_HELP, SCHEMAS2  # noqa: E402
from cutbig import ask  # noqa: E402
from gsmsolve import ARITH_SCHEMA, ask_spec_model, equal  # noqa: E402
from newbands import SCHEMA_NEW  # noqa: E402
from olympiad import SOLO, last_number  # noqa: E402
from solvemap import PREDICATE_HELP, SCHEMAS, answer_of, parse_spec  # noqa: E402
from solvers2 import run2  # noqa: E402

SCHEMA_MORE = {
    "statistics": '{"solver":"statistics","values":[<numbers>],"report":"mean"|'
                  '"median"|"population_variance"|"sample_variance"|'
                  '"population_sd"|"sample_sd"|"range"|"sum"|"mode"|"count"}',
    "datetime": '{"solver":"datetime","kind":"days_between","from":"YYYY-MM-DD",'
                '"to":"YYYY-MM-DD"} | {"kind":"weekday","date":"YYYY-MM-DD"} | '
                '{"kind":"add_days","date":"YYYY-MM-DD","days":<int>,"weeks":<int>} | '
                '{"kind":"leap_years","from_year":<int>,"to_year":<int>}',
    "finance": '{"solver":"finance","kind":"percent_chain","start":<number>,'
               '"changes":[<percent as a number, negative for a fall>,...]} | '
               '{"kind":"annuity","principal":<n>,"rate":<n per period>,'
               '"periods":<int>} | {"kind":"vat","net":<n>,"percent":<whole percent, e.g. 25>}',
}

FEWSHOT = """Map the problem onto ONE solver and fill its slots. Do NOT compute the
answer — an exact executor computes it from your spec.

Example: "What is the exact population variance of 2, 4, 4, 4, 5, 5, 7, 9?"
Spec: {{"solver":"statistics","values":[2,4,4,4,5,5,7,9],
"report":"population_variance"}}

Example: "How many days are there from 1999-12-31 to 2000-03-01?"
Spec: {{"solver":"datetime","kind":"days_between","from":"1999-12-31",
"to":"2000-03-01"}}

Example: "A price of 100 rises 20 percent and then falls 20 percent. What is it now?"
Spec: {{"solver":"finance","kind":"percent_chain","start":100,"changes":[20,-20]}}

Catalogue:
{catalogue}

For the search solver, conditions use these ops: {preds}
Expressions may call: {funcs}   (^ is a power; / is exact rational division)

Problem: {story}
Spec:"""


def build():
    out = []

    def stats(vals, report, text):
        n = len(vals)
        xs = [F(v) for v in vals]
        mean = sum(xs) / n
        srt = sorted(xs)
        table = {
            "mean": mean,
            "median": srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2,
            "population_variance": sum((x - mean) ** 2 for x in xs) / n,
            "sample_variance": sum((x - mean) ** 2 for x in xs) / (n - 1),
            "range": srt[-1] - srt[0], "sum": sum(xs),
        }
        return ("statistics", text, str(table[report]))

    out.append(stats([2, 4, 4, 4, 5, 5, 7, 9], "population_variance",
                     "What is the exact population variance of 2, 4, 4, 4, 5, 5, 7 "
                     "and 9? Give the answer as a fraction if it is not a whole "
                     "number."))
    out.append(stats([13, 7, 21, 4, 19, 8, 11], "median",
                     "What is the median of 13, 7, 21, 4, 19, 8 and 11?"))
    out.append(stats([17, 23, 31, 5, 12, 44, 9, 28], "mean",
                     "What is the exact mean of 17, 23, 31, 5, 12, 44, 9 and 28? Give "
                     "the answer as a fraction."))
    out.append(stats([3, 8, 15, 2, 11, 6], "sample_variance",
                     "What is the exact sample variance of 3, 8, 15, 2, 11 and 6? Give "
                     "the answer as a fraction."))
    out.append(stats([101, 87, 93, 78, 115, 99], "mean",
                     "What is the exact mean of 101, 87, 93, 78, 115 and 99? Give the "
                     "answer as a fraction."))
    out.append(stats([5, 9, 14, 2, 27, 3, 18, 6, 21], "population_variance",
                     "What is the exact population variance of 5, 9, 14, 2, 27, 3, 18, "
                     "6 and 21? Give the answer as a fraction."))

    def days(a, b, text):
        return ("datetime", text,
                str(abs((dt.date(*b) - dt.date(*a)).days)))

    out.append(days((1999, 12, 31), (2000, 3, 1),
                    "How many days are there from 31 December 1999 to 1 March 2000?"))
    out.append(days((1970, 1, 1), (2026, 8, 10),
                    "How many days are there from 1 January 1970 to 10 August 2026?"))
    out.append(days((2024, 2, 28), (2025, 3, 1),
                    "How many days are there from 28 February 2024 to 1 March 2025?"))
    out.append(("datetime", "What day of the week was 15 March 1848?",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
                 "Sunday"][dt.date(1848, 3, 15).weekday()]))
    out.append(("datetime", "What is the date 1000 days after 2 February 2020? Give it "
                "as YYYY-MM-DD.",
                (dt.date(2020, 2, 2) + dt.timedelta(days=1000)).isoformat()))
    out.append(("datetime", "How many leap years are there from 1896 to 2104 "
                "inclusive?",
                str(sum(1 for y in range(1896, 2105)
                        if y % 4 == 0 and (y % 100 or y % 400 == 0)))))

    def chain(start, changes, text):
        v = F(start)
        for c in changes:
            v *= 1 + F(c) / 100
        return ("finance", text, str(v))

    out.append(chain(100, [20, -20], "A price of 100 rises 20 percent and then falls "
                     "20 percent. What is it exactly now?"))
    out.append(chain(2500, [15, -10, 8, -25],
                     "A holding of 2500 changes by +15 percent, then -10 percent, then "
                     "+8 percent, then -25 percent. What is it exactly now? Give a "
                     "fraction if it is not whole."))
    out.append(chain(880, [-12, -12, -12],
                     "An 880 kr item is discounted 12 percent three times in a row. "
                     "What is the exact final price? Give a fraction if needed."))
    out.append(chain(1, [7] * 10,
                     "A value of 1 grows 7 percent per year for 10 years. What is the "
                     "exact value at the end? Give it as a fraction."))
    pay = F(200000) * F(1, 100) / (1 - (1 + F(1, 100)) ** -120)
    out.append(("finance", "A loan of 200000 at 1 percent per period over 120 periods "
                "has what exact level payment per period? Give a fraction.", str(pay)))
    out.append(("finance", "A net price of 4800 carries 25 percent tax. What is the "
                "exact gross price?", str(F(4800) * F(5, 4))))
    return out


def main(out="data/custom/bands2.json"):
    battery = build()
    catalogue = "\n".join(f"- {v}" for v in
                          {"arith": ARITH_SCHEMA, **SCHEMA_MORE, **SCHEMA_NEW,
                           **SCHEMAS, **SCHEMAS2}.values())
    t = {k: 0 for k in ("solo", "mpeqs", "parsed", "ran", "wrong")}
    byfam = {}
    rows = []
    for fam, story, truth in battery:
        a_solo = ask("qwen-35b", SOLO.format(problem=story), n=420)
        num = last_number(a_solo)
        solo_ok = (equal(num, truth) if num is not None else False) or \
            (str(truth) in a_solo if not str(truth).replace("-", "").isdigit() else
             False)
        t["solo"] += solo_ok

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
                mp_ok = equal(got, truth) or str(got) == str(truth)
                t["mpeqs"] += mp_ok
                t["wrong"] += not mp_ok
            else:
                got = why[:40]
        f = byfam.setdefault(fam, [0, 0, 0])
        f[0] += 1
        f[1] += solo_ok
        f[2] += mp_ok
        rows.append({"family": fam, "story": story[:70], "truth": str(truth),
                     "solo": str(num), "solo_ok": bool(solo_ok), "mpeqs": str(got),
                     "mpeqs_ok": bool(mp_ok), "spec": spec})
        print(f"{fam:<11}{'solo ok' if solo_ok else 'solo X '} "
              f"{'mpeqs ok' if mp_ok else 'mpeqs X '}  {str(truth)[:20]:<22}"
              f"{story[:38]}")

    n = len(battery)
    print(f"\nSOLO-35B : {t['solo']}/{n}")
    print(f"MPEqs    : {t['mpeqs']}/{n}  (parsed {t['parsed']}, ran {t['ran']}, "
          f"wrong {t['wrong']})")
    for fam, (cnt, so, mp) in byfam.items():
        print(f"  {fam:<12}{cnt:>3}{so:>6}{mp:>7}")
    print("\nThree classes, the same rule: mechanical where the model is weak, one")
    print("sentence where the pipeline is strong. The band is not a discovery any")
    print("more, it is a construction — every addition is another slice of it.")
    summary = {"n": n, **t, "byfam": byfam, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
