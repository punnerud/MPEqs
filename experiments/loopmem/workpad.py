#!/usr/bin/env python3
"""A work record a small model can drive: read a row at a time, append, patch, never rewrite.

Phase 13 established the split that matters. The scratchpad and its derivation graph did their
jobs — 58 underivable steps refused, every repeat caught — and the 1B model still solved 0 of 8,
because it could not choose a second step. So the store and the driver have to be tested apart,
and this is the store, verified against a scripted agent that makes deliberate mistakes. What a
model can do with it is a separate question with a separate answer.

The interface is the one the design asks for:

  read(i) / rows()   sequential access, so only the current row and the next need to be held
  append(...)        add a derived row; operands must already exist, or it is refused
  patch(i, ...)      correct a row IN PLACE without touching anything before it — and every row
                     derived from it, transitively, is marked STALE rather than silently kept
  attempt(...)       record that something was tried and how it went
  seen(...)          has this been tried before, ON THE SAME PREMISE, and what happened
  history(...)       everything ever tried on this pair, across versions

Two properties are the point, and both are tested rather than asserted:

**An early error reaches everything downstream.** Rows carry their operands, so patching row 2
marks rows 5 and 9 stale if they descend from it. Without that a correction leaves a record that
looks consistent and is not.

**A fixed error is remembered.** The explored set survives patches and successes; it is the
memory of what has been tried, kept outside the model so a fresh context cannot lose it. In
phase 13 the refusal list was cleared on progress and the model immediately re-proposed a step
it had already had refused four times.
"""
import json
import operator
from pathlib import Path

OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


class Refused(Exception):
    """A step the record will not accept, with the reason the caller should be shown."""


class WorkPad:
    def __init__(self, name="", givens=()):
        self.name = name
        self.rows = []                 # each: dict(id, expr, value, deps, stale, note)
        self.given = {}                # literal -> row id, so givens are rows like any other
        self.explored = {}             # canonical attempt -> outcome, never cleared
        for g in givens:
            self._add(f"{g:g}", float(g), [], note="given")

    # ---- reading -------------------------------------------------------------------
    def __len__(self):
        return len(self.rows)

    def read(self, i):
        """One row. The unit the model is meant to hold in mind."""
        if not 0 <= i < len(self.rows):
            raise IndexError(f"row {i} does not exist; the record has {len(self.rows)}")
        return dict(self.rows[i])

    def live(self):
        """Values usable as operands: rows that are not stale and not yet consumed."""
        return [r for r in self.rows if not r["stale"] and not r["used"]]

    def render(self, window=None):
        """The record as text. `window` shows only the last few rows, for a small context."""
        rows = self.rows if window is None else self.rows[-window:]
        out = []
        for r in rows:
            flag = " STALE" if r["stale"] else ("" if not r["used"] else " used")
            out.append(f"  [{r['id']}] {r['expr']} = {r['value']:g}{flag}")
        return "\n".join(out) or "  (empty)"

    # ---- writing -------------------------------------------------------------------
    def _add(self, expr, value, deps, note=""):
        rid = len(self.rows)
        self.rows.append({"id": rid, "expr": expr, "value": value, "deps": list(deps),
                          "stale": False, "used": False, "note": note, "version": 0})
        return rid

    def key(self, a_id, op, b_id):
        """Canonical form of an attempt, INCLUDING the version of each operand.

        Versioning is what lets a correction reopen what was previously wrong. Keyed on row ids
        alone, the record refused `[5] + [2]` after row 5 had been patched from `-` to `/` —
        but that is a different computation, 30 + 15 rather than 116 + 15, and refusing it makes
        a fix useless. New knowledge has to reopen the attempts it invalidates.

        The old attempt is not deleted. It stays under its old key, so "this failed before, on a
        different premise" remains answerable — which is the difference between a memory and a
        blocklist.
        """
        va = self.rows[a_id]["version"] if 0 <= a_id < len(self.rows) else 0
        vb = self.rows[b_id]["version"] if 0 <= b_id < len(self.rows) else 0
        pair = ((a_id, va), (b_id, vb))
        lo, hi = sorted(pair) if op in "+*" else pair
        return f"{lo[0]}v{lo[1]}{op}{hi[0]}v{hi[1]}"

    def seen(self, a_id, op, b_id):
        """What happened last time this was tried on the CURRENT premise, or None."""
        return self.explored.get(self.key(a_id, op, b_id))

    def history(self, a_id, op, b_id):
        """Every attempt on this pair, whatever the version — the memory a fix does not erase."""
        stem = f"{min(a_id, b_id) if op in '+*' else a_id}v"
        return {k: v for k, v in self.explored.items()
                if k.startswith(stem) and op in k}

    def attempt(self, a_id, op, b_id, outcome):
        self.explored[self.key(a_id, op, b_id)] = outcome

    def append(self, a_id, op, b_id):
        """Derive a new row from two existing ones. Refuses anything it cannot justify."""
        k = self.key(a_id, op, b_id)
        if k in self.explored:
            raise Refused(f"already tried ({self.explored[k]})")
        if op not in OPS:
            self.attempt(a_id, op, b_id, "bad operator")
            raise Refused(f"unknown operator {op!r}")
        for rid in (a_id, b_id):
            if not 0 <= rid < len(self.rows):
                self.attempt(a_id, op, b_id, "no such row")
                raise Refused(f"row {rid} does not exist")
            if self.rows[rid]["stale"]:
                self.attempt(a_id, op, b_id, "operand stale")
                raise Refused(f"row {rid} is stale; fix or rederive it first")
            if self.rows[rid]["used"]:
                self.attempt(a_id, op, b_id, "operand already consumed")
                raise Refused(f"row {rid} has already been used")
        a, b = self.rows[a_id]["value"], self.rows[b_id]["value"]
        if op == "/" and b == 0:
            self.attempt(a_id, op, b_id, "division by zero")
            raise Refused("division by zero")
        val = OPS[op](a, b)
        self.rows[a_id]["used"] = True
        self.rows[b_id]["used"] = True
        rid = self._add(f"[{a_id}] {op} [{b_id}]", val, [a_id, b_id])
        self.attempt(a_id, op, b_id, f"ok -> row {rid} = {val:g}")
        return rid

    def undo(self):
        """Drop the newest row and release its operands. Backtracking, not rewriting.

        The record refuses steps and remembers them, but refusing is only half of protection:
        a driver that can only append is locked by its first wrong move, because the operands
        are consumed and there is no way back. Measured — a random driver scored 0.000 at every
        budget until this existed, while a naive pool with no protection at all scored 0.219.
        Protection without a way back is not protection, it is paralysis.

        The attempt stays in `explored`, so the same wrong step is not re-taken; only the state
        is rolled back.
        """
        if not self.rows or not self.rows[-1]["deps"]:
            return None
        r = self.rows.pop()
        for d in r["deps"]:
            self.rows[d]["used"] = False
        return r["id"]

    def patch(self, rid, op):
        """Correct a row's operator in place. Everything downstream of it becomes stale.

        Nothing before the row is touched — that is the point of patching rather than
        rewriting — and nothing derived from it is quietly left standing.
        """
        r = self.rows[rid]
        if not r["deps"]:
            raise Refused("a given cannot be patched")
        a, b = (self.rows[d]["value"] for d in r["deps"])
        r["value"] = OPS[op](a, b)
        r["expr"] = f"[{r['deps'][0]}] {op} [{r['deps'][1]}]"
        r["note"] = "patched"
        r["stale"] = False
        r["version"] += 1
        return self._mark_stale_below(rid)

    def _mark_stale_below(self, rid):
        """Transitive descendants become stale, and their operands become usable again."""
        stale = []
        changed = True
        while changed:
            changed = False
            for r in self.rows:
                if r["stale"] or not r["deps"]:
                    continue
                if any(d == rid or d in stale for d in r["deps"]):
                    r["stale"] = True
                    r["version"] += 1     # its premise moved, so attempts using it reopen
                    stale.append(r["id"])
                    for d in r["deps"]:
                        if d != rid and self.rows[d]["stale"] is False:
                            self.rows[d]["used"] = False
                    changed = True
        # the patched row itself is live again and unconsumed
        self.rows[rid]["used"] = False
        return stale

    def save(self, path):
        Path(path).write_text(json.dumps(
            {"name": self.name, "rows": self.rows, "explored": self.explored}, indent=2))
