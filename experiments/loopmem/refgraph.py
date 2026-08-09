#!/usr/bin/env python3
"""The reference layer: the model reads what points at what, the record verifies it.

Phase 68 killed the namespace residual — record-issued names, zero cross-part mismatches —
and solving did not move because the residual cascaded into REFERENCE: missing-number
failures doubled. The autopsy of that doubling is specific: two quantities that both count
clips (April's and May's) cannot share the one issued name `clip`, so a part dropped one of
them. Ellipsis ("half as many"), pronouns, and same-noun quantities all live one layer below
nouns — hva som peker på hva — and that layer is invariant across word orders, which is why
it is a READING job and not a lexicon job.

The mechanism: before any part translates, the model reads the WHOLE problem once and maps
every number to a reference — a compound name built ONLY from record-issued atoms:

    {"48": "clip_april", "2": "none"}

The record verifies the map with perfect ground truth (every key must be a problem number,
every component of every compound an issued lemma, one retry with the violation named), then
the parts translate exactly as in phase 68 but with the map in context and the compounds
added to the allowed vocabulary. Names stay consistent by construction; reference becomes
expressible; and the chain of anchors decides what this layer was worth:

    one-shot 16/20  |  parts, record names (68): 2/20, missing numbers 8/20, name errors 0
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402
from lexicon import Lexicon  # noqa: E402
from mapstore import NUM, norm  # noqa: E402
from namefix import NAME, allowed_names  # noqa: E402
from olympiad import load_problems  # noqa: E402
from relgraph import solve_system  # noqa: E402
from rretlfan import split_parts  # noqa: E402
from stagedabs import extract_json  # noqa: E402

READ_REFS = """Read the problem and say what every number points at. Build each reference
name ONLY from these words, joined with _ when one word is not enough to tell two
quantities apart:
  {names}
Use "none" for pure factors like "half" or "times".

Reply with only JSON mapping each number to its reference, like:
{{"48": "clip_april", "2": "none"}}

Problem: {problem}
{hint}"""

TRANSLATE = """You are absorbing a word problem into named quantities. Every number's
reference is already known:
  {refmap}
You may ONLY use those reference names, these words: {names}, and "answer" for the asked
quantity. Definitions so far: {defs}

This fragment (part {i} of {n}):
{part}

Give the definitions this fragment adds, as JSON like {{"clip_may": "clip_april / 2"}}.
If the fragment adds nothing, reply {{}}.
{hint}"""


def read_refs(model, problem, atoms):
    """The reference map, verified componentwise: perfect-ground-truth checks only."""
    values = {norm(m) for m in NUM.findall(norm(problem))}
    hint = ""
    for _ in range(2):
        d = extract_json(ask(model, READ_REFS.format(
            names=", ".join(atoms), problem=problem, hint=hint), n=300))
        if not isinstance(d, dict):
            hint = "\nYour last reply was not a JSON object."
            continue
        bad = []
        refs = {}
        for k, v in d.items():
            kv = norm(str(k))
            if kv not in values:
                # A key that is not a problem number is commentary, not a violation — the
                # model maps "half" to "none" unprompted, and refusing that cost the whole
                # map on the first sanity check. Required coverage is of the real numbers;
                # extras are ignored, never punished.
                continue
            name = str(v).strip().lower()
            if name in ("none", ""):
                refs[kv] = None
                continue
            parts = name.split("_")
            foreign = [p for p in parts if p not in atoms and not p.isdigit()]
            if foreign:
                bad.append(f"{name} uses {foreign[0]}, which is not an allowed word")
                continue
            refs[kv] = name
        if not bad and refs:
            missing = values - set(refs)
            if missing:
                hint = (f"\nYou did not map these numbers: "
                        f"{', '.join(sorted(missing))}. Map every number.")
                continue
            return refs
        hint = "\nRefused: " + "; ".join(bad[:3]) if bad else "\nMap every number."
    return None


def main(n_test=20, seed=5, model="qwen-35b", out="data/custom/refgraph.json"):
    import random
    n_test, seed = int(n_test), int(seed)
    lex = Lexicon()
    gsm, _ = load_problems()
    tests = random.Random(seed).sample(gsm, n_test)

    tally = {"solved": 0, "ref_maps": 0, "distinct_compounds": 0, "not_reduce": 0,
             "missing_numbers": 0, "foreign_refusals": 0, "calls": 0}
    rows = []
    for problem, truth in tests:
        parts = split_parts(problem)
        atoms, attached = allowed_names(problem, lex)
        atoms = [a for a in atoms if a != "answer"]

        refs = read_refs(model, problem, atoms)
        tally["calls"] += 2 if refs is None else 1
        refnames = sorted({v for v in (refs or {}).values() if v})
        compounds = [r for r in refnames if "_" in r]
        tally["ref_maps"] += refs is not None
        tally["distinct_compounds"] += len(set(compounds))
        vocab = set(atoms) | set(refnames) | {"answer"}
        refmap = ", ".join(f"{v} -> {r or 'factor'}"
                           for v, r in sorted((refs or {}).items()))

        defs = {}
        for i, part in enumerate(parts):
            hint = ""
            for attempt in range(2):
                d = extract_json(ask(model, TRANSLATE.format(
                    refmap=refmap or "unknown", names=", ".join(atoms),
                    defs=json.dumps(defs) or "{}", i=i + 1, n=len(parts),
                    part=part.strip(), hint=hint), n=300))
                tally["calls"] += 1
                if not isinstance(d, dict):
                    break
                foreign = [w for k, v in d.items()
                           for w in [str(k)] + NAME.findall(str(v))
                           if w not in vocab and w not in defs
                           and not re.fullmatch(r"\d+", w)]
                if foreign:
                    tally["foreign_refusals"] += 1
                    hint = (f"\nRefused: {', '.join(sorted(set(foreign))[:3])} is not an "
                            f"allowed name.")
                    continue
                for k, v in d.items():
                    if isinstance(v, (str, int, float)) and str(k) not in defs:
                        defs[str(k)] = str(v)
                break

        asked = "answer" if "answer" in defs else None
        if asked is None:
            used = set()
            for body in defs.values():
                used.update(NAME.findall(str(body)))
            sinks = [k for k in defs if k not in used]
            asked = sinks[-1] if sinks else (list(defs)[-1] if defs else "")
        ans, why = solve_system({"defs": defs, "asked": asked}, problem)
        ok = ans == truth
        tally["solved"] += ok
        if why:
            tally["not_reduce"] += "does not reduce" in why
            tally["missing_numbers"] += why.startswith("problem numbers missing")
        rows.append({"truth": str(truth), "answer": str(ans), "ok": ok, "why": why,
                     "refs": refs, "defs": len(defs)})

    n = n_test
    print(f"{model}, {n} problems, the model reading what points at what:\n")
    print(f"  solved                      : {tally['solved']}/{n}   "
          f"(68-anchor 2/20, one-shot 16/20)")
    print(f"  missing-number failures     : {tally['missing_numbers']}/{n}   (was 8/20)")
    print(f"  'does not reduce' failures  : {tally['not_reduce']}/{n}   (was 0/20)")
    print(f"  reference maps accepted     : {tally['ref_maps']}/{n}, "
          f"{tally['distinct_compounds']} compound names coined")
    print(f"  foreign-name refusals       : {tally['foreign_refusals']}, "
          f"calls {tally['calls']} ({tally['calls'] / n:.1f}/problem)")
    print("\nThe compounds are the point: clip_april and clip_may can now be two quantities,")
    print("which one issued noun never could. If missing numbers collapse and solving rises,")
    print("reference was the layer; whatever still stands is the arithmetic of relations —")
    print("the last thing left that only a whole-problem reader holds.")
    Path(out).write_text(json.dumps({"model": model, "n": n, **tally, "rows": rows},
                                    indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
