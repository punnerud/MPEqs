#!/usr/bin/env python3
"""Digging into HOW the model fails: does `2 - 1 = 2` route to the copy pathway?

The sharpest behavioural failure this project measured is also its smallest: asked to
subtract two single digits, the 1B model answers `2 - 1` with 2 — an echo — while `9 - 3`
comes back right. Phase 24 called it the floor no decomposition can reach. This digs under
it with the project's own instrument: moetrace reads the router's expert choices
(ffn_moe_topk) per token per layer, so the failure can be looked at as ROUTING rather than
as output.

The hypothesis is specific and falsifiable: the wrong cases are routed like COPYING, not
like arithmetic. Three trace families make that testable:

    SUBTRACTIONS   all 55 single-digit pairs a >= b, prompt exactly as phase 17 verified it
    ECHO CONTROLS  "Repeat this number: a" — the copy pathway's signature, by construction
    the behavioural map of which subtractions actually fail, measured in the same run

Per trace, the signature is the set of experts chosen at the last prompt tokens (where the
answer forms), per layer. Then three numbers decide:

    within-right similarity        do correct cases share a routing signature?
    right-versus-wrong similarity  do the failures deviate from it?
    echo proximity                 are the failures CLOSER to the copy signature than the
                                   successes are — the hypothesis, directly

If wrong cases sit nearer the echo signature, the failure is a routing event: the router
hands the tokens to the pathway that repeats instead of the one that computes. That would
be an internals-level target — the first in this project addressable below the prompt.
"""
import json
import re
import struct
import subprocess
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402
from measure_loops import MODEL  # noqa: E402

MOETRACE = "./target/release/moetrace"
PROMPT = ("<|endoftext|><|user|>\nSubtract two single digits: {a} - {b}\n\n"
          "Reply with only the number.\n<|assistant|>\n")
ECHO = ("<|endoftext|><|user|>\nRepeat this number: {a}\n\n"
        "Reply with only the number.\n<|assistant|>\n")


def load_trace(path):
    """header: MOET | ver | n_layer | n_expert | top_k | pad | n_records(u64); then
    records of u16 layer | u16 pad | u32 token | top_k x u16 expert ids."""
    raw = Path(path).read_bytes()
    magic, _ver, n_layer, _ne, top_k, _pad = struct.unpack_from("<6I", raw, 0)
    assert magic == 0x5445_4F4D, "not a MOET trace"
    (n_rec,) = struct.unpack_from("<Q", raw, 24)
    off = 32
    # Version 2 records carry the gate weights as f32 after the ids:
    # u16 layer | u16 pad | u32 token | top_k x u16 ids | top_k x f32 weights.
    # The first parse assumed the v1 shape and read weight bytes as layer numbers,
    # yielding 373 "layers" from a 16-layer model.
    rec = (len(raw) - 32) // n_rec
    per_layer = {}
    for _ in range(n_rec):
        layer, _p, token = struct.unpack_from("<HHI", raw, off)
        ids = struct.unpack_from(f"<{top_k}H", raw, off + 8)
        per_layer.setdefault(layer, {})[token] = ids
        off += rec
    return per_layer, n_layer


def signature(per_layer, last_n=3):
    """Experts at the final `last_n` prompt tokens, per layer — where the answer forms."""
    sig = {}
    for layer, toks in per_layer.items():
        top = sorted(toks)[-last_n:]
        sig[layer] = frozenset(e for t in top for e in toks[t])
    return sig


def jaccard(a, b):
    per = []
    for layer in a:
        if layer in b:
            u = len(a[layer] | b[layer])
            per.append(len(a[layer] & b[layer]) / u if u else 0.0)
    return sum(per) / len(per) if per else 0.0


def trace_prompt(text, out):
    Path("/tmp/ft-in.txt").write_text(text)
    subprocess.run([MOETRACE, "--model", MODEL, "--text", "/tmp/ft-in.txt",
                    "--out", out, "--ngl", "0"], capture_output=True, text=True)
    return load_trace(out)


def main(out="data/custom/failtrace.json"):
    cases = [(a, b) for a in range(10) for b in range(a + 1)]
    print(f"behavioural map: {len(cases)} single-digit subtractions...")
    right, wrong = [], []
    for a, b in cases:
        r = ask("olmoe-1b", f"Subtract two single digits: {a} - {b}\n\n"
                            f"Reply with only the number.", n=24)
        m = re.findall(r"-?\d+", r or "")
        got = int(m[-1]) if m else None
        (right if got == a - b else wrong).append((a, b, got))
    print(f"  right {len(right)}, wrong {len(wrong)}: "
          f"{[(f'{a}-{b}', g) for a, b, g in wrong][:8]}")

    print("tracing routing (CPU, hook active)...")
    sigs = {}
    for a, b in cases:
        pl, _ = trace_prompt(PROMPT.format(a=a, b=b), f"/tmp/ft-s{a}{b}.bin")
        sigs[(a, b)] = signature(pl)
    echo_sigs = {}
    for a in range(10):
        pl, _ = trace_prompt(ECHO.format(a=a), f"/tmp/ft-e{a}.bin")
        echo_sigs[a] = signature(pl)

    right_k = [(a, b) for a, b, _ in right]
    wrong_k = [(a, b) for a, b, _ in wrong]
    within_right = [jaccard(sigs[x], sigs[y]) for x, y in combinations(right_k, 2)]
    cross = [jaccard(sigs[x], sigs[y]) for x in right_k for y in wrong_k]
    # Echo proximity: each subtraction against ITS OWN left digit's echo — the copy target
    # a repeater would land on.
    echo_right = [jaccard(sigs[(a, b)], echo_sigs[a]) for a, b in right_k]
    echo_wrong = [jaccard(sigs[(a, b)], echo_sigs[a]) for a, b in wrong_k]

    def mean(v):
        return sum(v) / len(v) if v else 0.0

    print(f"\nrouting signatures (last 3 prompt tokens, per layer, Jaccard):")
    print(f"  within right cases            : {mean(within_right):.3f}")
    print(f"  right against wrong           : {mean(cross):.3f}")
    print(f"  ECHO proximity, right cases   : {mean(echo_right):.3f}")
    print(f"  ECHO proximity, wrong cases   : {mean(echo_wrong):.3f}")
    verdict = mean(echo_wrong) - mean(echo_right)
    print(f"\n  wrong-case echo excess        : {verdict:+.3f}")
    print("\nPositive excess: the failures route toward the copy signature — an internals-")
    print("level finding with an internals-level lever (steer or mask the copy experts).")
    print("Near zero: routing does not distinguish them, and the failure lives in the")
    print("weights the router feeds, below what routing can see or fix.")
    Path(out).write_text(json.dumps({
        "cases": len(cases), "right": len(right), "wrong": len(wrong),
        "wrong_list": [[a, b, g] for a, b, g in wrong],
        "within_right": mean(within_right), "right_vs_wrong": mean(cross),
        "echo_right": mean(echo_right), "echo_wrong": mean(echo_wrong),
        "echo_excess": verdict}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
