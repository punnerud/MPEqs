#!/usr/bin/env python3
"""An English lexicon the record can consult: POS lookup, lemmatisation, measured quality.

The namespace residual of phases 64-65 exists because the model INVENTS node names. A
dictionary lets the record hand out the names instead — the text's own nouns — and for that
the record needs to know which words are nouns. WordNet 3.1's index files supply the
lemma-to-POS map (117,982 nouns, 11,569 verbs) and its .exc files the irregular inflections;
suffix rules cover the regular ones.

The lexicon's quality is MEASURED BEFORE ANYTHING IS BUILT ON IT, on Universal Dependencies
English-EWT dev, because phase 55 established what an undeclared noise floor does to every
judge downstream: the 72%-reliable unit tagger turned a sound refusal gate into a net
negative. Whatever precision and coverage this file reports is the ceiling everything in
phases 67-69 inherits, in the open.

Disambiguation is heuristic and declared: a token found only in one POS is that POS; a token
in several is resolved by local context (after a determiner, number or adjective it is a
noun; after 'to', a pronoun or an auxiliary it is a verb; otherwise the noun reading wins,
because nouns are what the graph is made of and a false noun is a spurious node while a
missed noun is a hole).
"""
import json
import re
import sys
from pathlib import Path

DICT = Path("/tmp/dict")
UD = Path("/tmp/ud-dev.conllu")

DETS = {"the", "a", "an", "this", "that", "these", "those", "his", "her", "their", "its",
        "my", "your", "our", "each", "every", "some", "any", "no", "another", "both"}
PRONOUNS = {"i", "you", "he", "she", "we", "they", "it", "who"}
AUX = {"will", "would", "can", "could", "shall", "should", "may", "might", "must", "do",
       "does", "did", "don", "didn", "doesn"}
NUMWORD = re.compile(r"^\d")


class Lexicon:
    def __init__(self, root=DICT):
        self.pos = {}                    # lemma -> set of 'n','v','a','r'
        for tag, fname in (("n", "index.noun"), ("v", "index.verb"),
                           ("a", "index.adj"), ("r", "index.adv")):
            for line in (root / fname).read_text().splitlines():
                if line.startswith(" "):
                    continue
                lemma = line.split(" ", 1)[0].replace("_", " ")
                self.pos.setdefault(lemma, set()).add(tag)
        self.exc = {}                    # inflected -> (lemma, pos)
        for tag, fname in (("n", "noun.exc"), ("v", "verb.exc")):
            for line in (root / fname).read_text().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    self.exc.setdefault(parts[0], (parts[1], tag))

    def lemmas(self, token):
        """Candidate (lemma, pos) readings for a surface token, lexicon-confirmed only."""
        w = token.lower()
        out = set()
        if w in self.exc:
            lem, tag = self.exc[w]
            out.add((lem, tag))
        for tag in self.pos.get(w, ()):
            out.add((w, tag))
        # Regular inflections, each candidate kept only if the dictionary knows the lemma.
        cands = []
        if w.endswith("ies") and len(w) > 4:
            cands.append((w[:-3] + "y", "nv"))
        if w.endswith("es") and len(w) > 3:
            cands.append((w[:-2], "nv"))
        if w.endswith("s") and not w.endswith("ss") and len(w) > 2:
            cands.append((w[:-1], "nv"))
        if w.endswith("ed") and len(w) > 3:
            cands.extend([(w[:-2], "v"), (w[:-1], "v")])
        if w.endswith("ing") and len(w) > 4:
            cands.extend([(w[:-3], "v"), (w[:-3] + "e", "v")])
        for lem, tags in cands:
            for tag in tags:
                if tag in self.pos.get(lem, ()):
                    out.add((lem, tag))
        return out

    def tag(self, tokens, i):
        """One token's POS in context, or None when the lexicon has no reading."""
        readings = self.lemmas(tokens[i])
        tags = {t for _, t in readings}
        if not tags:
            return None
        if len(tags) == 1:
            return next(iter(tags))
        prev = tokens[i - 1].lower() if i > 0 else ""
        nxt = tokens[i + 1].lower() if i + 1 < len(tokens) else ""
        nouny = (prev in DETS or NUMWORD.match(prev)
                 or ("a" in {t for _, t in self.lemmas(prev)} and prev not in PRONOUNS))
        if "n" in tags and nouny:
            return "n"
        if "v" in tags and (prev in PRONOUNS or prev == "to" or prev in AUX):
            return "v"
        # An ambiguous token with no nouny context is NOT called a noun. The first policy
        # defaulted ties to noun and paid 0.474 precision on UD — half the graph's nodes
        # would have been fabrications. Preferring the verb reading on bare ambiguity trades
        # noun recall 0.988 for precision, and a graph wants clean nodes more than all
        # nodes: a false noun is a spurious entity, a missed one is a hole a later pass can
        # fill. Measured on UD both ways; this is the one calibration iteration, on the
        # calibration set only.
        if "v" in tags:
            return "v"
        return next(iter(tags - {"n"})) if tags - {"n"} else "n"

    def noun_lemma(self, token):
        """The noun lemma for a token, or None — the graph's node namer."""
        for lem, tag in self.lemmas(token):
            if tag == "n":
                return lem
        return None


def measure_on_ud(lex, path=UD):
    """Precision and coverage on NOUN and VERB against UD gold, tokens the lexicon knows."""
    sents, cur = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            if cur:
                sents.append(cur)
            cur = []
            continue
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]:
            continue
        cur.append((f[1], f[3]))                       # (form, UPOS)
    if cur:
        sents.append(cur)

    stats = {"n": [0, 0, 0], "v": [0, 0, 0]}           # [predicted, correct, gold]
    covered = total = 0
    gold_map = {"NOUN": "n", "PROPN": "n", "VERB": "v"}
    for sent in sents:
        tokens = [w for w, _ in sent]
        for i, (w, gold) in enumerate(sent):
            if not w.isalpha():
                continue
            total += 1
            pred = lex.tag(tokens, i)
            if pred is None:
                continue
            covered += 1
            g = gold_map.get(gold)
            for tag in ("n", "v"):
                if pred == tag:
                    stats[tag][0] += 1
                    stats[tag][1] += g == tag
                if g == tag:
                    stats[tag][2] += 1
    out = {"tokens": total, "coverage": covered / total}
    for tag, label in (("n", "noun"), ("v", "verb")):
        p, c, g = stats[tag]
        out[f"{label}_precision"] = c / p if p else 0.0
        out[f"{label}_recall"] = c / g if g else 0.0
    return out


def main(out="data/custom/lexicon.json"):
    lex = Lexicon()
    print(f"lexicon: {len(lex.pos):,} lemmas, {len(lex.exc):,} irregular forms")
    m = measure_on_ud(lex)
    print(f"\nUD English-EWT dev, {m['tokens']:,} alphabetic tokens:")
    print(f"  coverage (lexicon has a reading) : {m['coverage']:.3f}")
    print(f"  noun precision / recall          : {m['noun_precision']:.3f} / "
          f"{m['noun_recall']:.3f}")
    print(f"  verb precision / recall          : {m['verb_precision']:.3f} / "
          f"{m['verb_recall']:.3f}")
    print("\nThese numbers are the declared noise floor for every phase built on this file —")
    print("measured before anything downstream exists, which is the phase 55 rule.")
    Path(out).write_text(json.dumps({"lemmas": len(lex.pos), "irregulars": len(lex.exc),
                                     **m}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
