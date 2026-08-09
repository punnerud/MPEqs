#!/usr/bin/env python3
"""Verify the work record with a scripted agent that makes deliberate mistakes.

Scripted, not a model, and deliberately. Phase 13 showed the store and the driver fail for
different reasons, and mixing them meant every store bug looked like a model limitation and
every model limitation looked like a store bug. A scripted agent makes exactly the mistakes
being tested for, so a failure here is the store's.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from workpad import Refused, WorkPad  # noqa: E402

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  ok    {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}  {detail}")


print("1. sequential reading and derivation")
p = WorkPad("(120/4 + 15) * 2 - 30", [120, 4, 15, 2, 30])
check("givens become rows", len(p) == 5)
r0 = p.append(0, "/", 1)                      # 120 / 4 = 30
check("derived row appended", p.read(r0)["value"] == 30)
check("operands are consumed", p.read(0)["used"] and p.read(1)["used"])
check("consumed rows leave the live set", 0 not in [r["id"] for r in p.live()])
try:
    p.append(0, "+", 2)
    check("a consumed operand is refused", False)
except Refused as e:
    check("a consumed operand is refused", "already been used" in str(e), str(e))

print("\n2. the explored set survives, so nothing is retried")
before = p.seen(0, "/", 1)
check("a successful attempt is remembered", before and before.startswith("ok"))
try:
    p.append(0, "/", 1)
    check("an exact repeat is refused", False)
except Refused as e:
    check("an exact repeat is refused", "already tried" in str(e))
try:
    p.append(1, "+", 0)
    check("a commuted repeat is refused too", False)
except Refused:
    check("a commuted repeat is refused too", True)

print("\n3. an early error reaches everything downstream")
q = WorkPad("chain", [10, 2, 5, 3])
a = q.append(0, "+", 1)      # 10 + 2 = 12
b = q.append(a, "*", 2)      # 12 * 5 = 60
c = q.append(b, "-", 3)      # 60 - 3 = 57
check("chain built", q.read(c)["value"] == 57)
stale = q.patch(a, "-")      # the first step was wrong: 10 - 2 = 8
check("the patched row is corrected", q.read(a)["value"] == 8)
check("rows before it are untouched", q.read(0)["value"] == 10 and q.read(1)["value"] == 2)
check("both descendants go stale", set(stale) == {b, c}, f"got {stale}")
check("a stale row is not live", b not in [r["id"] for r in q.live()])
try:
    q.append(b, "+", 3)
    check("a stale operand is refused", False)
except Refused as e:
    check("a stale operand is refused", "stale" in str(e))
check("the patched row is usable again", a in [r["id"] for r in q.live()])

print("\n4. the fix is remembered WITHOUT blocking the retry")
# These two assertions previously required the opposite — that a patched step could never be
# retried — and that was wrong. Correcting an operand changes what the step computes, so
# refusing it makes the correction useless. The memory must record the old attempt and still
# let the new one through. Kept as a note because the test was reversed deliberately, not bent
# to make the code pass.
check("seen() reports nothing on the NEW premise", q.seen(a, "*", 2) is None)
check("history() still holds the attempt made on the old premise",
      len(q.history(a, "*", 2)) >= 1, f"{q.history(a, '*', 2)}")
try:
    redone = q.append(a, "*", 2)
    check("the step can be retried on corrected input", q.read(redone)["value"] == 8 * 5,
          f"got {q.read(redone)['value']}")
except Refused as e:
    check("the step can be retried on corrected input", False, str(e))

print("\n5. a small window is enough to work from")
w = WorkPad("window", list(range(1, 12)))
for i in range(0, 10, 2):
    w.append(i, "+", i + 1)
check("window shows only the tail", len(w.render(window=3).splitlines()) == 3)
check("full record still readable in sequence", len(w) == 16, f"{len(w)} rows")
check("every row reachable by index", all(w.read(i)["id"] == i for i in range(len(w))))

print("\n6. a correction reopens what it invalidated")
r = WorkPad("reopen", [120, 4, 15])
x = r.append(0, "-", 1)                 # wrong: 120 - 4 = 116
y = r.append(x, "+", 2)                 # 116 + 15 = 131, built on the error
r.patch(x, "/")                         # corrected: 120 / 4 = 30
check("the descendant went stale", r.read(y)["stale"])
try:
    y2 = r.append(x, "+", 2)            # same pair, different premise — must be allowed
    check("the same step is allowed again on corrected input", r.read(y2)["value"] == 45,
          f"got {r.read(y2)['value']}")
except Refused as e:
    check("the same step is allowed again on corrected input", False, str(e))
check("the old attempt is still remembered", len(r.history(x, "+", 2)) >= 2,
      f"{r.history(x, '+', 2)}")
try:
    r.append(x, "+", 2)
    check("but an exact repeat on the NEW premise is still refused", False)
except Refused:
    check("but an exact repeat on the NEW premise is still refused", True)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
