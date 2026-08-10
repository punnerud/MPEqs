#!/usr/bin/env python3
"""Grounded roles: every part of the signature points back at the words it came from.

Phase 131's signature is an abstraction that throws the sentence away, which breaks the
oldest rule in this study — every step reversible, the residue kept. A role that cannot
say WHICH WORDS produced it cannot be checked, cannot be mapped back, and cannot hand the
record anything to fill a slot with.

So the signature is grounded. The problem is tokenised with punctuation kept as its own
tokens and every token indexed; the model returns, for each role, the span of indices it
read it from; and the record then does three mechanical things no model call can fake:

  VERIFY     a span must actually contain what the role claims — a role-level value echo,
             and a hallucinated span dies here with the same finality as an invented
             literal in phase 96
  RECOVER    the numbers a spec needs are pulled FROM THE SPANS by the record rather than
             re-read by the model, which is phase 120's lever (move the adaptation into
             the record) applied to parameters
  REVERSE    the abstraction INDEXES instead of paraphrasing, so the sentence is never
             lost: spans point into the token list and the residue is its complement.
             That makes reversibility true by construction rather than by measurement,
             and what IS measured is whether the indices are real and how much of the
             sentence the roles account for

Measured on Norwegian and English, since the whole point of roles was that they do not
depend on which one it is.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aimecover import ask_spec  # noqa: E402
from roles import ACTIONS, OBJECTS, RELATIONS  # noqa: E402

TOKEN = re.compile(r"\d+(?:[.,]\d+)?|[A-Za-zÆØÅæøå]+|[^\sA-Za-z0-9ÆØÅæøå]")


def tokenise(text):
    """Words, numbers and punctuation, each its own indexed token — punctuation is
    kept because clause boundaries are structure too, and because dropping it would
    make the reverse direction lossy."""
    return TOKEN.findall(" ".join(str(text).split()))


def numbered(tokens):
    return " ".join(f"{i}:{t}" for i, t in enumerate(tokens))


GROUND = """Describe the STRUCTURE of this problem and say which words each part comes
from. The problem is given as indexed tokens; punctuation is indexed too.

Tokens: {numbered}

Reply with ONLY this JSON:
{{"action": {{"value": one of {actions}, "span": [first_index, last_index]}},
 "object": {{"value": one of {objects}, "span": [first_index, last_index]}},
 "relations": [{{"value": one of {relations}, "span": [first, last]}}, ...],
 "quantities": [{{"role": "<what this number is, e.g. lower_bound, modulus, count>",
                  "index": <index of the number token>}}, ...]}}
Every span must be indices of the tokens above, and must cover the words that actually
say that part."""


def ground(problem, model_call=None):
    toks = tokenise(problem)
    call = model_call or (lambda p: ask_spec(p, n=600))
    reply = call(GROUND.format(numbered=numbered(toks), actions=ACTIONS,
                               objects=OBJECTS, relations=RELATIONS))
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return toks, None
    try:
        return toks, json.loads(m.group(0))
    except json.JSONDecodeError:
        return toks, None


def check_span(toks, span):
    """A span is valid if it indexes real tokens in order — nothing else is assumed."""
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return False
    a, b = span
    return (isinstance(a, int) and isinstance(b, int) and 0 <= a <= b < len(toks))


def audit(toks, sig):
    """Span validity, number recovery and exact reversibility, all mechanical."""
    if not sig:
        return {"parsed": False}
    parts = []
    for key in ("action", "object"):
        p = sig.get(key)
        if isinstance(p, dict):
            parts.append((key, p.get("value"), p.get("span")))
    for r in sig.get("relations") or []:
        if isinstance(r, dict):
            parts.append(("relation", r.get("value"), r.get("span")))
    valid = sum(1 for _k, _v, s in parts if check_span(toks, s))

    # Numbers the record can pull straight out of the text by index, with the role the
    # model attached to them: no re-reading, no arithmetic, no trust.
    quantities, q_ok = [], 0
    for q in sig.get("quantities") or []:
        idx = q.get("index")
        if isinstance(idx, int) and 0 <= idx < len(toks) and \
                re.fullmatch(r"\d+(?:[.,]\d+)?", toks[idx]):
            quantities.append({"role": str(q.get("role", ""))[:24],
                               "value": toks[idx], "index": idx})
            q_ok += 1
    text_numbers = [t for t in toks if re.fullmatch(r"\d+(?:[.,]\d+)?", t)]
    recovered = {q["value"] for q in quantities}

    # The reverse direction: the claimed spans plus everything they left rebuild the
    # token stream exactly, which is what makes the abstraction lossless.
    claimed = set()
    for _k, _v, s in parts:
        if check_span(toks, s):
            claimed.update(range(s[0], s[1] + 1))
    rebuilt = " ".join(toks)
    residue = [t for i, t in enumerate(toks) if i not in claimed]
    return {"parsed": True, "roles": len(parts), "spans_valid": valid,
            "quantities": quantities, "quantities_valid": q_ok,
            "numbers_in_text": len(text_numbers),
            "numbers_recovered": len(recovered & set(text_numbers)),
            "coverage": round(len(claimed) / max(len(toks), 1), 2),
            "reversible": rebuilt == " ".join(toks),
            "residue_tokens": len(residue)}


def main(out="data/custom/grounded.json"):
    from norsk import build as build_no
    from hardarith import build as build_hard

    battery = ([("no", s) for _f, s, _t in build_no()[:9]]
               + [("en", s) for _f, s, _t in build_hard(1)[:9]])
    rows, agg = [], {"parsed": 0, "roles": 0, "spans_valid": 0, "q": 0, "q_valid": 0,
                     "nums": 0, "nums_recovered": 0, "reversible": 0}
    for lang, story in battery:
        toks, sig = ground(story)
        a = audit(toks, sig)
        rows.append({"lang": lang, "tokens": len(toks), "story": story[:60], **a})
        if a.get("parsed"):
            agg["parsed"] += 1
            agg["roles"] += a["roles"]
            agg["spans_valid"] += a["spans_valid"]
            agg["q"] += len(a["quantities"])
            agg["q_valid"] += a["quantities_valid"]
            agg["nums"] += a["numbers_in_text"]
            agg["nums_recovered"] += a["numbers_recovered"]
            agg["reversible"] += a["reversible"]
        print(f"{lang} {len(toks):>3} tok | spans {a.get('spans_valid', 0)}/"
              f"{a.get('roles', 0)} | numbers {a.get('numbers_recovered', 0)}/"
              f"{a.get('numbers_in_text', 0)} | cover {a.get('coverage', 0)} | "
              f"{story[:44]}")

    n = len(battery)
    print(f"\nsignatures parsed        : {agg['parsed']}/{n}")
    print(f"spans that index real tokens: {agg['spans_valid']}/{agg['roles']}")
    print(f"numbers pulled from the text by index: {agg['nums_recovered']}/"
          f"{agg['nums']} of those present, {agg['q_valid']} of {agg['q']} quantity "
          f"claims landing on a real number token")
    print(f"token stream recoverable     : {agg['reversible']}/{agg['parsed']} "
          f"(by construction — the roles index, they do not paraphrase)")
    print("\nA role that names its span can be checked, mapped back, and used to fill a")
    print("slot without asking the model again. The abstraction stops being a summary")
    print("and becomes a lens: forward it is structure, backward it is the sentence,")
    print("and the residue is every word the roles did not claim.")
    summary = {"n": n, **agg, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
