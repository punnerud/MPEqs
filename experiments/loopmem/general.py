#!/usr/bin/env python3
"""The same record, generalised past arithmetic: preconditions in, effects out.

Splitting to single digits took a 1B model from 6/12 to 11/12 on carry-heavy addition, so the
atom was too big and making it smaller was the whole fix. The question is whether that
generalises, and the examples that prompt it are not arithmetic at all: how do you build a
table, how do you open a bottle in the fridge and pour a glass.

Those have the same shape as a digit column. A column consumes two digits and a carry and
produces a digit and a carry; an action consumes conditions that must hold and produces
conditions that then hold. Legs before top. Open before pour. Glass on the table before pouring
into it. The record does not need to know arithmetic — it needs to know that a step may only be
taken when what it depends on is already true, which is exactly what `WorkPad.append` enforces
for operands.

So the generalisation is one substitution: operands become preconditions, values become facts.
Everything else — refusal, memory of what was tried, transitive invalidation, backtracking —
carries over unchanged, and this measures whether the small model can order steps when the
record checks them.

Two arms, same tasks, same model:
  FREE     ask for the whole plan in one go
  STEPWISE ask for one action at a time; the record refuses any whose preconditions are unmet
           and tells the model which are missing
"""
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from measure_loops import BIN, MODEL  # noqa: E402

# name: (preconditions, effects). Ordinary physical tasks, ordered by dependency rather than
# by arithmetic. The goal is the last effect in each list.
TASKS = {
    "pour a glass of soda": {
        "actions": {
            "open fridge": ([], ["fridge open"]),
            "take out bottle": (["fridge open"], ["bottle on counter"]),
            "unscrew cap": (["bottle on counter"], ["bottle open"]),
            "fetch glass": ([], ["glass on counter"]),
            "pour": (["bottle open", "glass on counter"], ["glass full"]),
        },
        "goal": "glass full",
    },
    "build a table": {
        "actions": {
            "cut the top": ([], ["top cut"]),
            "cut four legs": ([], ["legs cut"]),
            "sand the parts": (["top cut", "legs cut"], ["parts sanded"]),
            "attach legs to top": (["parts sanded"], ["frame assembled"]),
            "stand it up": (["frame assembled"], ["table standing"]),
        },
        "goal": "table standing",
    },
    "make tea": {
        "actions": {
            "fill kettle": ([], ["kettle full"]),
            "boil water": (["kettle full"], ["water boiled"]),
            "put teabag in cup": ([], ["teabag in cup"]),
            "pour water into cup": (["water boiled", "teabag in cup"], ["tea brewing"]),
            "remove teabag": (["tea brewing"], ["tea ready"]),
        },
        "goal": "tea ready",
    },
}

FREE = """<|endoftext|><|user|>
Task: {task}

List the steps in the correct order, one per line, choosing only from these actions:
{menu}
<|assistant|>
"""

STEP = """<|endoftext|><|user|>
Task: {task}

Already done:
{done}

Remaining actions to choose from:
{menu}
{hint}
Name exactly ONE action to do next, copied exactly from the list. Nothing else.
<|assistant|>
"""


def ask(prompt, n=64):
    Path("/tmp/gen.txt").write_text(prompt)
    out = subprocess.run(
        [BIN, "-m", MODEL, "-f", "/tmp/gen.txt", "-n", str(n), "--temp", "0",
         "-no-cnv", "-st", "-ngl", "99"],
        capture_output=True, text=True).stdout
    i = out.rfind("<|assistant|>")
    tail = out[i + 13:] if i >= 0 else out
    return re.sub(r"<think>.*?</think>", " ", tail, flags=re.S).split("[end of text]")[0]


STOP = {"the", "a", "an", "it", "to", "in", "on", "of", "into", "up", "and", "your", "then"}


def match_action(text, actions):
    """Which action the reply names. Exact substring first, then content-word overlap.

    Exact substring alone was too strict and it cost a whole task. Asked what makes the table
    stand, the model replied "Attach legs to the top." — the menu says "attach legs to top", one
    article apart — and the matcher returned None, so the step was counted as a refusal without
    the record ever checking whether the action pointed the right way. Every "build a table" run
    failed on that article. Content words decide; articles and prepositions do not.
    """
    low = text.lower()
    best, pos = None, len(low) + 1
    for name in actions:
        p = low.find(name.lower())
        if 0 <= p < pos:
            best, pos = name, p
    if best is not None:
        return best
    words = set(re.findall(r"[a-z]+", low)) - STOP
    scored = []
    for name in actions:
        need = set(re.findall(r"[a-z]+", name.lower())) - STOP
        if need and len(need & words) / len(need) >= 0.75:
            scored.append((len(need & words) / len(need), len(need), name))
    return max(scored)[2] if scored else None


def run_free(task, spec):
    menu = "\n".join(f"  - {a}" for a in spec["actions"])
    reply = ask(FREE.format(task=task, menu=menu))
    facts, taken = set(), []
    for line in reply.splitlines():
        a = match_action(line, spec["actions"])
        if not a or a in taken:
            continue
        pre, eff = spec["actions"][a]
        taken.append(a)
        if all(p in facts for p in pre):        # only a valid step actually changes the world
            facts.update(eff)
    return spec["goal"] in facts, taken


def run_stepwise(task, spec, max_steps=10):
    facts, taken, refused = set(), [], 0
    for _ in range(max_steps):
        remaining = [a for a in spec["actions"] if a not in taken]
        if not remaining:
            break
        done = "\n".join(f"  - {t}" for t in taken) or "  (nothing yet)"
        menu = "\n".join(f"  - {a}" for a in remaining)
        hint = ""
        if refused:
            missing = {p for a in remaining for p in spec["actions"][a][0]} - facts
            hint = ("\nNot yet true, so anything needing these will be refused: "
                    + ", ".join(sorted(missing)) + "\n") if missing else ""
        a = match_action(ask(STEP.format(task=task, done=done, menu=menu, hint=hint)),
                         {k: v for k, v in spec["actions"].items() if k in remaining})
        if a is None:
            refused += 1
            continue
        pre, eff = spec["actions"][a]
        if not all(p in facts for p in pre):
            # The record refuses it: its preconditions are not yet true. Nothing is consumed.
            refused += 1
            continue
        taken.append(a)
        facts.update(eff)
        if spec["goal"] in facts:
            return True, taken, refused
    return spec["goal"] in facts, taken, refused


def main(out="data/custom/general.json"):
    print("The same record, on ordinary tasks: preconditions in, effects out\n")
    print(f"{'task':<24}{'free':>7}{'stepwise':>10}{'refused':>9}  order taken")
    rows, f_ok, s_ok = [], 0, 0
    for task, spec in TASKS.items():
        ok_f, order_f = run_free(task, spec)
        ok_s, order_s, refused = run_stepwise(task, spec)
        f_ok += ok_f
        s_ok += ok_s
        rows.append({"task": task, "free_ok": ok_f, "free_order": order_f,
                     "stepwise_ok": ok_s, "stepwise_order": order_s, "refused": refused})
        print(f"{task:<24}{'ok' if ok_f else '.':>7}{'ok' if ok_s else '.':>10}"
              f"{refused:>9}  {' -> '.join(order_s) if order_s else '(none)'}")

    n = len(TASKS)
    summary = {"tasks": n, "free_correct": f_ok, "stepwise_correct": s_ok,
               "total_refused": sum(r["refused"] for r in rows)}
    print(f"\nwhole plan in one go   : {f_ok}/{n}")
    print(f"one action at a time   : {s_ok}/{n}")
    print(f"steps the record refused as having unmet preconditions: "
          f"{summary['total_refused']}")
    print("\nThe substitution is operands -> preconditions. If it holds here the record is not")
    print("an arithmetic device, and the digit result was an instance of something general.")
    Path(out).write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
