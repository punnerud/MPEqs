#!/usr/bin/env python3
"""The retry rung: the record names its refusal, the model rewrites only the check.

Pass one flipped the bottleneck — demanding a verifier lifted the solving itself to
6/10, but five right answers were flagged because their CHECKS broke the subset
(While, a whitelist violation, a syntax slip, a missing block). The session's ladder
discipline (phase 49: one retry with the refusal NAMED) applies verbatim: for every
flagged problem that HAS an answer, the model gets the problem, its own answer, its
failed check, and the record's exact refusal — and rewrites only the check. The
no-answer cases are not retried: temp 0, unchanged prompt, unchanged reply is this
session's measured law, and a retry there would be theatre.

Same gate as pass one: subset, self-accept, six decoys rejected. The merged numbers
are the phase's final word: deliveries, rights, WRONGS across both passes, against
the agreement gate's 1 right / 2 wrong on the same ten problems.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aimebudget import ask_work  # noqa: E402
from aimecheck import extract, run_check, subset_ok  # noqa: E402
from olympiad import load_problems  # noqa: E402

RETRY = """{problem}

The answer is known to be a = {ans}. Write ONLY the verification function.

Your previous check was refused by the executor: {why}
Previous check:
```python
{old}
```

Rules: only arithmetic, for-loops over range(...), if, and comparisons. No imports,
no while, no strings, no attribute access (nothing with a dot). Allowed calls: range,
len, sum, abs, min, max, int, gcd, all, any, sorted, set, factorial, comb, divmod,
pow, round, enumerate.

Reply with ONLY:
```python
def check(a):
    ...
```"""


def main(out="data/custom/aimecheck2.json"):
    prev = json.loads(Path("data/custom/aimecheck.json").read_text())
    rng = random.Random(5)
    _, aime = load_problems()
    picks = rng.sample(aime, 15)[:10]
    decoy_rng = random.Random(43)

    retried = delivered = right = wrong = still_flagged = 0
    rows = []
    for i, row in enumerate(prev["rows"]):
        if not row["verdict"].startswith("flagged") or row["answer"] is None:
            continue
        problem, truth = picks[i]
        retried += 1
        reply = ask_work(RETRY.format(problem=problem, ans=row["answer"],
                                      why=row["why"], old=row["check"]))
        _, code = extract("Answer: 0\n" + reply)  # reuse the block extractor
        verdict, why = "flagged", "no check block"
        if code:
            tree, why = subset_ok(code)
            if tree:
                ok_self, why = run_check(code, row["answer"])
                if ok_self:
                    decoys = [row["answer"] - 1, row["answer"] + 1,
                              row["answer"] - 7, row["answer"] + 7]
                    while len(decoys) < 6:
                        d = decoy_rng.randint(0, 999)
                        if d != row["answer"] and d not in decoys:
                            decoys.append(d)
                    if all(run_check(code, d)[0] is False for d in decoys):
                        delivered += 1
                        ok = row["answer"] == int(truth)
                        right += ok
                        wrong += not ok
                        verdict = f"DELIVERED {row['answer']}" + \
                            ("" if ok else " (WRONG)")
                    else:
                        why = "accepted decoys"
                elif ok_self is False:
                    why = "check rejects its own answer"
        if verdict == "flagged":
            still_flagged += 1
        rows.append({"i": i, "truth": str(truth), "answer": row["answer"],
                     "verdict": verdict, "why": why if verdict == "flagged" else ""})
        print(f"{i:>3} truth {str(truth):>6}  ans {str(row['answer']):>6}  {verdict}"
              f"{('  [' + why + ']') if verdict == 'flagged' else ''}")

    p1 = prev
    total_del = p1["delivered"] + delivered
    total_right = p1["delivered_right"] + right
    total_wrong = p1["delivered_wrong"] + wrong
    print(f"\nretried {retried} flagged-with-answer cases: delivered {delivered} "
          f"(right {right}, WRONG {wrong}), still flagged {still_flagged}")
    print(f"MERGED both passes: delivered {total_del}/10, right {total_right}, "
          f"WRONG {total_wrong}, flagged {10 - total_del}")
    print(f"the agreement gate on the same ten: delivered 3, right 1, WRONG 2")
    print("\nOne named refusal is the whole ladder here: the solving was already done,")
    print("and the record's error message is the only new information the model needs.")
    summary = {"retried": retried, "delivered": delivered, "right": right,
               "wrong": wrong, "still_flagged": still_flagged,
               "merged_delivered": total_del, "merged_right": total_right,
               "merged_wrong": total_wrong, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
