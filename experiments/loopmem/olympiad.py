#!/usr/bin/env python3
"""The machinery on real problems: GSM8K word problems and AIME olympiad questions.

Everything so far ran on synthetic four-term expressions, which is where mechanisms are
isolated, not where they are earned. This runs the thread's three survivors on two real sets —
GSM8K (grade-school word problems, the floor) and AIME (the Kaggle AIMO validation set,
olympiad problems with integer answers, the ceiling):

  SOLO          answer the problem, brief working allowed, last number taken
  GRAPH+SOLVER  the model writes only the arithmetic plan as a JSON graph — numbers and
                references, never results — and the record executes every step exactly with
                Fraction arithmetic. Phase 45's division of labour on problems nobody generated.
  AGREEMENT     three differently-phrased attempts; do they agree, and does agreement predict
                correctness? Phase 34's grader-free confidence signal, on real mathematics.

What the record CANNOT do here is the phase 21 inline check: a word problem is not an
expression, so there is nothing to verify the decomposition against without the answer. The
structural checks still hold (parse, defined-before-use, one result), and the solver still
removes model arithmetic entirely — what is lost is the refusal of wrong plans, and that loss
is part of what this measures.

Two models where meaningful: the 1B on GSM8K only (AIME is beyond it by construction), the 35B
on both. Thinking is prefilled off for BOTH models on every arm, so the comparison is between
architectures of work, not token budgets; olympiad scores here are therefore floors, not the
model's ceiling.
"""
import json
import re
import sys
import urllib.request
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402
from jsongraph import KEY, parse_graph  # noqa: E402

GSM_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/master/"
           "grade_school_math/data/test.jsonl")
AIME_URL = ("https://datasets-server.huggingface.co/rows?dataset=AI-MO%2F"
            "aimo-validation-aime&config=default&split=train&offset=0&length=60")

SOLO = """{problem}

Work it out briefly, then write the final line exactly as:
Answer: <number>
"""

RECHECK = """{problem}

Read the problem again carefully, watch for what it is actually asking, then write the
final line exactly as:
Answer: <number>
"""

GRAPH = """Solve the problem by writing ONLY the arithmetic plan as JSON. Each key is one
step using numbers from the problem or earlier keys; never write a computed result. The last
key is the final answer.

Example:
Problem: Tom has 3 boxes of 12 eggs and eats 5. How many eggs are left?
{{"A": "3 * 12", "B": "A - 5"}}

Problem: {problem}
"""

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def last_number(text):
    m = re.findall(r"Answer:\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    pool = m if m else NUM.findall(text.replace(",", ""))
    if not pool:
        return None
    try:
        return Fraction(pool[-1].replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def solve_graph(model, problem):
    """The model plans; the record computes. Exact arithmetic, no model results anywhere."""
    g, why = parse_graph(ask(model, GRAPH.format(problem=problem), n=512))
    if g is None:
        return None, f"parse: {why}"
    values = {}
    for k, body in g.items():
        expr = body
        for k2, v2 in values.items():
            expr = re.sub(rf"\b{k2}\b", f"({v2})", expr)
        if KEY.search(expr):
            return None, f"{k} uses an undefined step"
        if not re.fullmatch(r"[\d\s+*/().-]+", expr):
            return None, f"{k} is not arithmetic"
        try:
            values[k] = Fraction(eval(expr))  # noqa: S307 - digits and operators only, checked
        except Exception:  # noqa: BLE001
            return None, f"{k} does not evaluate"
    return values[list(g)[-1]], "ok"


def load_problems():
    gsm_path = Path("/tmp/gsm8k-test.jsonl")
    if not gsm_path.exists():
        urllib.request.urlretrieve(GSM_URL, gsm_path)
    gsm = []
    for line in gsm_path.read_text().splitlines()[:400]:
        d = json.loads(line)
        ans = d["answer"].rsplit("#### ", 1)[-1].replace(",", "").strip()
        gsm.append((d["question"], Fraction(ans)))
    aime_path = Path("/tmp/aime.json")
    if not aime_path.exists():
        urllib.request.urlretrieve(AIME_URL, aime_path)
    aime = [(r["row"]["problem"], Fraction(int(r["row"]["answer"])))
            for r in json.loads(aime_path.read_text())["rows"]]
    return gsm, aime


def run(model, problems, label, out_rows):
    stats = {"solo": 0, "graph": 0, "graph_ran": 0, "vote": 0}
    agree_cells = {}
    for problem, truth in problems:
        a1 = last_number(ask(model, SOLO.format(problem=problem), n=512))
        a2, stage = solve_graph(model, problem)
        a3 = last_number(ask(model, RECHECK.format(problem=problem), n=512))
        answers = [a for a in (a1, a2, a3) if a is not None]
        vote = max(set(answers), key=answers.count) if answers else None
        agree = len(set(answers)) == 1 and len(answers) == 3
        stats["solo"] += a1 == truth
        stats["graph"] += a2 == truth
        stats["graph_ran"] += stage == "ok"
        stats["vote"] += vote == truth
        cell = agree_cells.setdefault(agree, [0, 0])
        cell[0] += 1
        cell[1] += vote == truth
        out_rows.append({"set": label, "model": model, "truth": str(truth),
                         "solo": str(a1), "graph": str(a2), "graph_stage": stage,
                         "recheck": str(a3), "vote": str(vote), "all_agree": agree})
    n = len(problems)
    print(f"\n{model} on {label} ({n} problems)")
    print(f"  solo {stats['solo']}/{n}   graph+solver {stats['graph']}/{n} "
          f"({stats['graph_ran']} plans ran)   majority-of-3 {stats['vote']}/{n}")
    for agree in sorted(agree_cells):
        c, r = agree_cells[agree]
        print(f"  {'all three agree' if agree else 'they disagree':<17}: "
              f"{c} cases, vote right {r}/{c}")
    return {**stats, "n": n,
            "agree": {str(k): v for k, v in agree_cells.items()}}


def main(n_gsm=20, n_aime=15, seed=5, out="data/custom/olympiad.json"):
    import random
    n_gsm, n_aime = int(n_gsm), int(n_aime)
    rng = random.Random(int(seed))
    gsm, aime = load_problems()
    gsm = rng.sample(gsm, n_gsm)
    aime = rng.sample(aime, min(n_aime, len(aime)))

    rows, summary = [], {}
    summary["olmoe-1b/gsm8k"] = run("olmoe-1b", gsm, "gsm8k", rows)
    summary["qwen-35b/gsm8k"] = run("qwen-35b", gsm, "gsm8k", rows)
    summary["qwen-35b/aime"] = run("qwen-35b", aime, "aime", rows)

    print("\nGSM8K is the floor and AIME the ceiling; thinking is off for every arm, so the")
    print("olympiad numbers are floors. What is being asked is not 'how smart is the model'")
    print("but whether the record's division of labour and the agreement signal survive")
    print("contact with problems nobody generated to fit them.")
    Path(out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
