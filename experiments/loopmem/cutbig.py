#!/usr/bin/env python3
"""Does a bigger model aim better, and does splitting first make aiming unnecessary?

`cut` failed almost completely on the 1B model: 0/14 by index, 1/14 by naming the word, while
every cut stayed reversible. Two things could be wrong with that, and they call for different
fixes. Either the model is too small to aim, or aiming is the wrong thing to ask for.

Both are tested here, on the same fourteen sentences.

  MODEL SIZE   the same prompts against Qwen3.6-35B-A3B alongside OLMoE-1B-7B. The larger model
               already showed it can do the edit — asked to remove "old" it wrote "The bridge
               crosses the river", correct, just not as a call.

  SPLITTING    the record splits the text into pieces first and asks WHICH PIECE, not where. A
               split is reversible, so nothing is risked by it, and it converts aiming into
               choosing from a short list. Two levels: sentences, then words within the chosen
               sentence.

The second is the one that scales. Choosing among twenty sentences and then among six words is
two questions regardless of how long the document is, and joining the pieces back restores it
exactly — which is what makes the same machinery apply to two hundred pages rather than one line.
"""
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutspan import CASES, apply_cut, occurrences  # noqa: E402

BIN = "/opt/homebrew/Cellar/llama.cpp/8500/bin/llama-completion"

MODELS = {
    "olmoe-1b": ("models/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf",
                 "<|endoftext|><|user|>\n{p}\n<|assistant|>\n"),
    # Qwen is a thinking model. Left to itself it opens `<think>` and a 32-token budget never
    # reaches the closing tag, so every reply was an unterminated thought and nothing parsed.
    # Prefilling an empty think block turns it off, which is what makes the comparison fair:
    # both models then answer directly, and the budget measures the answer rather than the
    # preamble.
    "qwen-35b": ("models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf",
                 "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
                 "<think>\n\n</think>\n\n"),
}


def ask(model, prompt, n=48):
    path, template = MODELS[model]
    Path(f"/tmp/cb-{model}.txt").write_text(template.format(p=prompt))
    out = subprocess.run(
        [BIN, "-m", path, "-f", f"/tmp/cb-{model}.txt", "-n", str(n), "--temp", "0",
         "-no-cnv", "-st", "-ngl", "99"], capture_output=True, text=True).stdout
    # The special tokens are not always rendered in the output, so the reply was being read as
    # the prompt echoed back. Find the last assistant marker in whatever form it survived in.
    marks = list(re.finditer(r"<\|assistant\|>|<\|im_start\|>assistant|\bassistant\b", out))
    if marks:
        out = out[marks[-1].end():]
    out = re.sub(r"<think>.*?</think>", " ", out, flags=re.S)
    return out.split("[end of text]")[0].strip()


CUT_NAME = re.compile(r"""cut\(\s*\\?["']?([A-Za-z]+)\\?["']?\s*\)""")
CUT_POS = re.compile(r"cut\(\s*(\d+)\s*,\s*(\d+)\s*\)")

BY_NAME = """Sentence: {sentence}

Reply with only a cut call naming the word to remove. Nothing else.

Example:
Sentence: The cat sat on the mat
Remove the word "sat"
cut("sat")

Remove the word "{target}\""""

BY_INDEX = """Sentence: {sentence}

Reply with only a cut call giving the two character positions to remove. Nothing else.

Example:
Sentence: The cat sat on the mat
Cut out the word "sat"
cut(8, 11)

Cut out the word "{target}\""""

PICK = """{intro}

{options}

Which one is "{target}"? Reply with only the number.
"""


def arm_name(model, sentence, target):
    word = CUT_NAME.search(ask(model, BY_NAME.format(sentence=sentence, target=target), n=32))
    if not word:
        return {"ok": False, "why": "no call"}
    hits = occurrences(sentence, word.group(1))
    if not hits:
        return {"ok": False, "why": f"{word.group(1)!r} not present", "word": word.group(1)}
    took, rest, rev = apply_cut(sentence, *hits[0])
    return {"ok": took == target, "why": "ok", "word": word.group(1), "took": took,
            "reversible": rev}


def arm_index(model, sentence, target):
    m = CUT_POS.search(ask(model, BY_INDEX.format(sentence=sentence, target=target), n=32))
    if not m:
        return {"ok": False, "why": "no call"}
    took, rest, rev = apply_cut(sentence, int(m.group(1)), int(m.group(2)))
    return {"ok": took == target, "why": "ok", "took": took, "reversible": rev}


def choose(model, intro, options, target):
    """Pick one of a short numbered list. Selection, not aiming."""
    listing = "\n".join(f"{k + 1}. {o}" for k, o in enumerate(options))
    reply = ask(model, PICK.format(intro=intro, options=listing, target=target), n=16)
    m = re.search(r"\d+", reply)
    idx = int(m.group(0)) - 1 if m else -1
    return idx if 0 <= idx < len(options) else None


def arm_split(model, sentence, target):
    """The record splits; the model only chooses. Reversible, so the split risks nothing."""
    words = sentence.split(" ")
    idx = choose(model, "Here are the words of a sentence.", words, target)
    if idx is None:
        return {"ok": False, "why": "no choice"}
    picked = words[idx]
    rejoined = " ".join(words)
    return {"ok": picked.strip(".,") == target, "why": "ok", "took": picked,
            "reversible": rejoined == sentence, "chose": idx + 1}


def document_scale(model, target_word="umbrella"):
    """The same two moves on a document: choose a sentence, then choose a word inside it.

    Two questions whatever the length, because each split narrows by a whole level. The join is
    checked at both levels, so the document is provably reassembled and not merely assumed to be.
    """
    doc = " ".join(s + "." for s, _ in CASES) + " She left her umbrella on the bench."
    sentences = [s.strip() for s in doc.split(".") if s.strip()]
    i = choose(model, "Here are the sentences of a document.", sentences, target_word)
    if i is None:
        return {"ok": False, "why": "no sentence chosen", "calls": 1}
    words = sentences[i].split(" ")
    j = choose(model, "Here are the words of that sentence.", words, target_word)
    if j is None:
        return {"ok": False, "why": "no word chosen", "calls": 2, "sentence": sentences[i]}
    # Both joins must restore exactly, or the narrowing was not lossless.
    restored = ". ".join(sentences) + "." == doc.replace(".. ", ". ")
    return {"ok": words[j].strip(".,") == target_word, "why": "ok", "calls": 2,
            "sentence": sentences[i], "word": words[j],
            "sentences_in_doc": len(sentences), "words_in_sentence": len(words),
            "join_restores": " ".join(words) == sentences[i] and restored}


def main(out="data/custom/cutbig.json"):
    print(f"{len(CASES)} sentences, two models, three ways to aim\n")
    print(f"{'model':<10}{'by index':>10}{'by name':>9}{'split+choose':>14}")
    summary, rows = {}, []
    for model in MODELS:
        tally = Counter()
        for sentence, target in CASES:
            r_i = arm_index(model, sentence, target)
            r_n = arm_name(model, sentence, target)
            r_s = arm_split(model, sentence, target)
            tally["index"] += r_i["ok"]
            tally["name"] += r_n["ok"]
            tally["split"] += r_s["ok"]
            rows.append({"model": model, "sentence": sentence, "target": target,
                         "index": r_i, "name": r_n, "split": r_s})
        n = len(CASES)
        summary[model] = {"index": tally["index"], "name": tally["name"],
                          "split": tally["split"], "cases": n}
        print(f"{model:<10}{tally['index']:>7}/{n:<2}{tally['name']:>6}/{n:<2}"
              f"{tally['split']:>11}/{n:<2}")

    print("\nnarrowing a document by choosing, not by aiming\n")
    for model in MODELS:
        d = document_scale(model)
        summary[model]["document"] = d
        print(f"{model:<10}{'found it' if d['ok'] else 'missed':<10}"
              f"{d['calls']} calls over {d.get('sentences_in_doc', 0)} sentences, "
              f"join restores: {d.get('join_restores')}")
    print("\nTwo questions regardless of length. That is the property that carries this to a")
    print("long document, and it is the split being lossless that makes it safe to do.")
    Path(out).write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
