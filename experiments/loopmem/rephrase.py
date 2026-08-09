#!/usr/bin/env python3
"""Let the model turn the residual into a question it already knows how to answer.

Three things are now established about going backwards. Blank-filling — `110 + ? = 228` — is
5/24. Rewriting that by hand into arithmetic — `228 - 110` — is 13/24. And repetition cannot
close the gap: five samples vote no better than one (13/24 both), thinking longer is worse
(9/24), and unanimity beats the base rate by five points, so the failures are systematic rather
than stochastic.

The hand rewrite was mine, though, and a fixed table of inverse operators is not something that
generalises past arithmetic. The interesting version is to hand the residual to the model and ask
it for a form it finds easier — which is the same move that worked twice already. Asking for JSON
took the graph arm from 3/20 to 11/20 not because JSON is expressive but because it is what the
model has seen; a word problem may be the same kind of win for an inversion.

Four arms over the identical 24 facts:

  BLANK       110 + ? = 228, answer the blank                        (the 5/24 floor)
  INVOP       our rewrite, 228 - 110                                 (the 13/24 hand version)
  ASK_MODEL   the model rewrites the blank as an ordinary question, then answers its own
  WORD        the model turns it into a word problem, then answers that

The last two are the same information in a different dress, so any difference between them and
BLANK is presentation and not arithmetic.
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from invert import INVOP, facts  # noqa: E402
from repeat import ask  # noqa: E402

BLANK = """<|endoftext|><|user|>
{left} = {c}

What number goes in the blank? Reply with only the number.
<|assistant|>
"""

PLAIN = """<|endoftext|><|user|>
What is {expr}? Reply with only the number.
<|assistant|>
"""

# Ending the prompt where the answer starts. Without the trailing "Question:" the model
# echoed the equation back unchanged — "110 + ? = 228" in, "110 + ? = 228" out — because the
# example read as a pattern to copy rather than a transformation to apply.
REWRITE = """<|endoftext|><|user|>
Rewrite the equation as an ordinary arithmetic question with no blank in it.

Example:
Equation: 6 * ? = 30
Question: What is 30 / 6?

Equation: {left} = {c}
<|assistant|>
Question:"""

WORDIFY = """<|endoftext|><|user|>
Turn the equation into a short word problem using the same numbers. Do not solve it.

Example:
Equation: 6 * ? = 30
Problem: Each box holds 6 apples and there are 30 apples in total. How many boxes are there?

Equation: {left} = {c}
<|assistant|>
Problem:"""

ANSWER = """<|endoftext|><|user|>
{question}

Reply with only the number.
<|assistant|>
"""


def first_question(text):
    """The question the model wrote, which is the only part we are going to ask back."""
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    for line in text.strip().splitlines():
        line = line.strip().lstrip("Question:").lstrip("Problem:").strip()
        if line.endswith("?") and any(ch.isdigit() for ch in line):
            return line
    # A word problem often runs over several lines and ends with the question on the last one.
    joined = " ".join(text.split())
    m = re.search(r"[^.?!]*\?", joined)
    return m.group(0).strip() if m and any(ch.isdigit() for ch in joined) else None


def ask_text(prompt, n=96):
    """Same call as `ask`, but the words are wanted rather than the number."""
    from general import ask as gen_ask
    return gen_ask(prompt, n=n)


def main(n_facts=24, seed=11, out="data/custom/rephrase.json"):
    rng = random.Random(int(seed))
    fs = facts(int(n_facts), rng)
    print(f"{len(fs)} inversions. Who writes the question matters more than who answers it.\n")
    print(f"{'blank':>20}{'want':>8}{'blank':>8}{'invop':>8}{'asked':>8}{'word':>7}"
          f"  the model's own question")
    tally, rows = Counter(), []
    for a, op, b, c in fs:
        left = f"{a} {op} ?"
        want = b if op in "+*" else (a - c if op == "-" else a / c)

        v_blank = ask(BLANK.format(left=left, c=c))
        v_invop = ask(PLAIN.format(expr=f"{c} {INVOP[op]} {a}"))

        # The model writes the question. Whatever it produces is asked back verbatim, so a bad
        # rewrite costs it the point — the arm is the whole pipeline, not just the answering.
        q1 = first_question(ask_text(REWRITE.format(left=left, c=c)))
        v_asked = ask(ANSWER.format(question=q1)) if q1 else None
        q2 = first_question(ask_text(WORDIFY.format(left=left, c=c), n=128))
        v_word = ask(ANSWER.format(question=q2)) if q2 else None

        got = {"blank": v_blank, "invop": v_invop, "asked": v_asked, "word": v_word}
        for k, v in got.items():
            tally[k] += v is not None and abs(v - want) < 1e-9
        tally["wrote_a_question"] += q1 is not None
        tally["wrote_a_word_problem"] += q2 is not None
        rows.append({"left": left, "c": c, "want": want, "q_rewrite": q1, "q_word": q2, **got})
        mark = {k: ("ok" if got[k] is not None and abs(got[k] - want) < 1e-9 else ".")
                for k in got}
        print(f"{f'{left} = {c}':>20}{want:>8g}{mark['blank']:>8}{mark['invop']:>8}"
              f"{mark['asked']:>8}{mark['word']:>7}  {(q1 or '-')[:44]}")

    n = len(fs)
    print(f"\nanswer the blank directly        : {tally['blank']}/{n}")
    print(f"our rewrite into arithmetic      : {tally['invop']}/{n}")
    print(f"the model rewrites, then answers : {tally['asked']}/{n}"
          f"   ({tally['wrote_a_question']}/{n} produced a question)")
    print(f"as a word problem                : {tally['word']}/{n}"
          f"   ({tally['wrote_a_word_problem']}/{n} produced one)")
    print("\nIf the model's own rewrite matches the hand one, the residual only needs to be")
    print("handed over in a form it recognises — and nothing has to know what an inverse is.")
    summary = {"facts": n, **{f"{k}_correct": tally[k]
                              for k in ("blank", "invop", "asked", "word")},
               "wrote_a_question": tally["wrote_a_question"],
               "wrote_a_word_problem": tally["wrote_a_word_problem"]}
    Path(out).write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
