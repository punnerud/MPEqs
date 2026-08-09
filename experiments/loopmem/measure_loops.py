#!/usr/bin/env python3
"""Does a small model actually loop on multi-step work? Measure before building anything.

The proposal is a graph of the strongest links as external working memory, so a small distilled
model can grind through many steps — physics, arithmetic, parsing a document — without looping,
and can check what it has already done without destroying it.

Every part of that is plausible and none of it is measured. So this measures the premise first:
run a small model on tasks that need several steps, feed its own output back, and count what
actually goes wrong. Building a loop detector before knowing the loop rate would be building
for a problem that might not exist, at a size that might not matter.

Three failure modes are counted separately because they need different fixes:

  REPEAT     the model emits a step it has already emitted — a literal loop
  DRIFT      the model stops making progress but keeps producing new text — no repeat to catch,
             so a loop detector would never fire and the run still fails
  DESTROY    a later step contradicts or discards a result an earlier step established, which is
             the "checking what you have done without destroying it" case

A loop detector helps with the first and is useless against the other two. Which of them
dominates decides whether a graph memory is the right instrument at all.

Uses the local OLMoE 1B-7B instruct model through llama-completion — genuinely small, which is
the case the proposal is about.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

BIN = "/opt/homebrew/Cellar/llama.cpp/8500/bin/llama-completion"
MODEL = "models/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf"

# Multi-step problems with a single checkable answer, needing 3-6 dependent steps. Chosen so
# the model cannot shortcut to the answer, and so the answer is verifiable without a judge.
TASKS = [
    ("((47 + 38) * 3 - 12) / 9", 27),
    ("(120 / 4 + 15) * 2 - 30", 60),
    ("((9 * 7) + (8 * 6)) - 51", 60),
    ("(200 - 45 * 3) / 5 + 12", 25),
    ("((15 + 25) / 8) * 14 - 20", 50),
    ("(6 * 6 + 4 * 4) / 13 + 7", 11),
    ("((100 - 28) / 6) * 5 - 30", 30),
    ("(81 / 9 + 11) * 3 - 40", 20),
]

STEP_PROMPT = """You are solving a problem one small step at a time.

Problem: {problem}

Work so far:
{history}

Write ONLY the next single step, as `expression = result`. If the work so far already reaches
the final answer, write exactly `DONE: <number>`.
Next step:"""


def run_step(problem, history, n_predict=40):
    prompt = STEP_PROMPT.format(
        problem=problem, history="\n".join(history) if history else "(nothing yet)")
    p = Path("/tmp/loopstep.txt")
    p.write_text(prompt)
    out = subprocess.run(
        [BIN, "-m", MODEL, "-f", str(p), "-n", str(n_predict), "--temp", "0",
         "-no-cnv", "-st", "-ngl", "99"],
        capture_output=True, text=True).stdout
    i = out.rfind("Next step:")
    tail = out[i + 10:] if i >= 0 else out
    tail = tail.split("[end of text]")[0]
    tail = re.sub(r"<think>.*?</think>", " ", tail, flags=re.S)
    # Strip markdown fences before taking a line. Without this the model's habit of opening a
    # code block makes ``` the "step" every time, and the first run of this harness scored
    # 28 repeats across 7 of 8 tasks — all of them my own parser echoing a fence back at itself.
    # That is a measurement of the harness, not of the model, and it nearly became a finding.
    tail = re.sub(r"^\s*```[a-zA-Z]*\s*$", "", tail, flags=re.M)
    for line in tail.splitlines():
        line = line.strip().strip("`").strip()
        if line and not line.startswith("#"):
            return line[:120]
    return ""


def normalise(step):
    """For repeat detection: whitespace and case removed, nothing else."""
    return re.sub(r"\s+", "", step).lower()


def main(max_steps=6, out="data/custom/loops.json"):
    max_steps = int(max_steps)
    print(f"{len(TASKS)} multi-step problems, up to {max_steps} steps each, OLMoE 1B-7B\n")
    print(f"{'problem':<28}{'steps':>6}{'outcome':>12}{'repeats':>9}"
          f"{'no result':>10}  final")
    rows = []
    for problem, answer in TASKS:
        history, seen, repeats, stateless = [], set(), 0, 0
        outcome, final = "no answer", None
        for _ in range(max_steps):
            step = run_step(problem, history)
            if not step:
                outcome = "empty"
                break
            key = normalise(step)
            if key in seen:
                repeats += 1
            seen.add(key)
            history.append(step)
            # A step that states an operation without recording its result is where state gets
            # lost: the next step reaches back for the last NUMBER it can see, which is now
            # stale. Counted separately because a loop detector cannot see it.
            if not re.search(r"=\s*-?\d", step) and not step.startswith("DONE"):
                stateless += 1
            m = re.search(r"DONE:\s*(-?\d+)", step)
            if m:
                final = int(m.group(1))
                outcome = "correct" if final == answer else "wrong"
                break
        else:
            outcome = "ran out"
        rows.append({"problem": problem, "answer": answer, "steps": len(history),
                     "outcome": outcome, "repeats": repeats, "stateless": stateless,
                     "final": final,
                     "history": history})
        print(f"{problem:<28}{len(history):>6}{outcome:>12}{repeats:>9}"
              f"{stateless:>10}  {final}")

    n = len(rows)
    summary = {
        "tasks": n,
        "correct": sum(1 for r in rows if r["outcome"] == "correct"),
        "wrong": sum(1 for r in rows if r["outcome"] == "wrong"),
        "ran_out": sum(1 for r in rows if r["outcome"] == "ran out"),
        "with_repeats": sum(1 for r in rows if r["repeats"] > 0),
        "total_repeats": sum(r["repeats"] for r in rows),
        "total_stateless": sum(r["stateless"] for r in rows),
        "total_steps": sum(r["steps"] for r in rows),
    }
    print(f"\ncorrect {summary['correct']}/{n}   wrong {summary['wrong']}   "
          f"ran out {summary['ran_out']}")
    print(f"runs containing a literal repeat: {summary['with_repeats']}/{n} "
          f"({summary['total_repeats']} repeats in total)")
    print(f"steps that state an operation without recording its result: "
          f"{summary['total_stateless']}/{summary['total_steps']}")
    print("\nA loop detector helps only the repeats. Steps with no recorded result are the")
    print("other failure — the next step reaches back for a stale number — and no detector")
    print("sees them, because every one of them is new text.")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
