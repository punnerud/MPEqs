#!/usr/bin/env python3
"""Ask the same thing several times, and let it think longer. Does either recover the gap?

Backward is 13/24 at best against forward's 22/24 on identical facts, so something is being lost
that is not the arithmetic. Two cheap things are worth trying before concluding the model simply
cannot invert: ask repeatedly in case it is sometimes wrong rather than always wrong, and give it
room to work rather than demanding the number immediately.

At temperature zero a repeat is not a repeat — the same prompt returns the same tokens, which is
the trap that made an earlier planning loop count one refusal ten times. So repetition here means
sampling, with a different seed each time.

The second measurement is the more useful one and it is free once the samples exist:

    does UNANIMITY predict correctness?

If five samples agreeing means the answer is right far more often than the base rate, there is a
confidence signal available with no grader at all — which is what the round trip was reaching for
and could not deliver, because it needed an inversion the model cannot do.

  SINGLE   one sample, temperature zero — the number every other phase used
  VOTE     five samples at temperature 0.8, majority answer
  LONGER   temperature zero, room to work through it before answering
"""
import json
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from invert import INVOP, facts  # noqa: E402
from measure_loops import BIN, MODEL  # noqa: E402

ASK = """<|endoftext|><|user|>
What is {expr}? Reply with only the number.
<|assistant|>
"""

THINK = """<|endoftext|><|user|>
What is {expr}?

Work it out, then write the answer on the last line as a plain number.
<|assistant|>
"""


def ask(prompt, n=48, temp=0.0, seed=0):
    Path(f"/tmp/rep{seed}.txt").write_text(prompt)
    out = subprocess.run(
        [BIN, "-m", MODEL, "-f", f"/tmp/rep{seed}.txt", "-n", str(n), "--temp", str(temp),
         "-s", str(seed), "-no-cnv", "-st", "-ngl", "99"],
        capture_output=True, text=True).stdout
    i = out.rfind("<|assistant|>")
    tail = out[i + 13:] if i >= 0 else out
    tail = re.sub(r"<think>.*?</think>", " ", tail, flags=re.S).split("[end of text]")[0]
    nums = re.findall(r"-?\d+(?:\.\d+)?", tail.replace(",", ""))
    return float(nums[-1]) if nums else None


def main(n_facts=24, seed=11, k=5, out="data/custom/repeat.json"):
    k = int(k)
    rng = random.Random(int(seed))
    fs = facts(int(n_facts), rng)
    print(f"{len(fs)} inversions, asked once, {k} times, and with room to think\n")
    print(f"{'question':>20}{'want':>8}{'single':>8}{'vote':>8}{'longer':>8}{'agree':>7}")
    tally = Counter()
    rows = []
    for a, op, b, c in fs:
        # The inversion posed as arithmetic, which is the best backward form measured.
        iop = INVOP[op]
        expr = f"{c} {iop} {a}"
        want = b if op in "+*" else (a - c if op == "-" else a / c)

        single = ask(ASK.format(expr=expr))
        samples = [ask(ASK.format(expr=expr), temp=0.8, seed=1000 + j) for j in range(k)]
        got = [s for s in samples if s is not None]
        counts = Counter(got)
        vote = counts.most_common(1)[0][0] if counts else None
        agree = counts.most_common(1)[0][1] if counts else 0
        longer = ask(THINK.format(expr=expr), n=192)

        ok = {"single": single is not None and abs(single - want) < 1e-9,
              "vote": vote is not None and abs(vote - want) < 1e-9,
              "longer": longer is not None and abs(longer - want) < 1e-9}
        for key, v in ok.items():
            tally[key] += v
        tally["unanimous"] += agree == k
        tally["unanimous_right"] += (agree == k) and ok["vote"]
        rows.append({"expr": expr, "want": want, "single": single, "vote": vote,
                     "longer": longer, "samples": samples, "agreement": agree, **ok})
        print(f"{expr:>20}{want:>8g}{str(single):>8}{str(vote):>8}{str(longer):>8}"
              f"{agree}/{k:>5}")

    n = len(fs)
    print(f"\none sample, temperature zero : {tally['single']}/{n}")
    print(f"majority of {k} at temp 0.8     : {tally['vote']}/{n}")
    print(f"room to work it out          : {tally['longer']}/{n}")
    una = tally["unanimous"]
    prec = tally["unanimous_right"] / una if una else 0.0
    print(f"\nall {k} samples agreed on {una}/{n}, and {tally['unanimous_right']} of those "
          f"are right ({prec:.0%})")
    print(f"base rate is {tally['vote'] / n:.0%}, so unanimity is worth "
          f"{prec - tally['vote'] / n:+.0%}")
    print("\nUnanimity is only a usable signal if it beats the base rate by enough to act on.")
    summary = {"facts": n, "k": k, "single_correct": tally["single"],
               "vote_correct": tally["vote"], "longer_correct": tally["longer"],
               "unanimous": una, "unanimous_right": tally["unanimous_right"],
               "precision_when_unanimous": prec, "base_rate": tally["vote"] / n}
    Path(out).write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
