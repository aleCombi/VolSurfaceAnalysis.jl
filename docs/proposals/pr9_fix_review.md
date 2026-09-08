# Review of the PR #9 fix sequence

Status: **implemented**. Three findings, all landed on `data-kinds` in
`f41a2b5` (findings 1 and 2) and `f8a49e7` (finding 3). Gate after them:
1198 passed, 0 failed.

This is the round after the round. The review of `data-kinds` produced
[pr9_correctness_fixes.md](pr9_correctness_fixes.md) (six confirmed defects) and
[pr9_remaining_findings.md](pr9_remaining_findings.md) (four lower-confidence
findings plus cleanup); those were implemented over ten commits, `df593ef` to
`b5dae8d`, and two follow-ups were recorded in
[pr9_followups.md](pr9_followups.md) and sequenced in
[pr9_followups_plan.md](pr9_followups_plan.md). This document reviews **the
fixes themselves** and records what that turned up.

Method: the ten commits read as diffs and the touched modules re-read as
current source (a fix reads differently in place than in a diff); the full gate
re-run to confirm the recorded number; and two behaviours probed in the live
REPL rather than argued from the code, because both are about what a reader
returns, not about what it looks like it returns.

---

## Verdict on the six correctness findings

All six are fixed, and fixed at the level the decisions asked for rather than
patched at the call site that reported them. Worth recording per finding,
because "it went green" is not the same claim:

- **6**, the `Constant.asof` stamp guard. The three other shapes already
  honoured the stamp and identity already hashed it, so this closes a gap rather
  than choosing a side.
- **5**, collapse-or-throw in the spot `between` plus `unique!` on
  `timestamps`. The right rule, incompletely applied — see finding 1 below.
- **2 and 3**, three-valued `serves`, checked in the four map-level shapes. The
  check sits at the map level rather than inside each provider, so
  provider-level delegation stays unchecked and an unserved *input* names its
  own kind: a `SurfaceFrom` asked for SPX reports `OptionBar`/SPX, not "no
  surface". That placement is what makes the error worth having.
- **4**, the bounded walk-back on `SurfaceReader.asof` with
  `DerivationExhausted` past the bound. The bound is a spec field, so it reaches
  identity and the config surface, and the three outcomes are separated cleanly:
  no chain is empty, a chain that builds is the surface, chains that never build
  within the bound throw.
- **1**, `settle(trd::Trade)`, `min(expiry, window_end)` and
  `declared_underlyings`. Replacing the expiry rather than adding the trade
  beside it is what makes a mismatched pair unrepresentable, and collapsing the
  two settlement branches into one lookup is a simplification rather than a
  repair.

The four test gaps closed in the step-0 commit — the three the correctness
document listed, plus the `BySelector` spec that asserted the rejected
alternative — were closed *before* the fixes landed, which is what makes the
sequence's red-to-green transitions mean anything.

---

## 1. Spot `asof` never reached the de-duplication rule

**Recorded as open before this review** ([pr9_followups.md](pr9_followups.md)
item 2), so the finding here is not the gap itself but its second symptom, which
changes what the right fix is.

### What is wrong

Decision 5 applied collapse-or-throw in `between`, with the provider-level `at`
inheriting it. `asof(::ParquetSpotsReader, ...)` inherits neither: it walked the
partition list backward and sliced the winning instant straight out of the block
in hand.

The recorded symptom was mild — a row delivered twice at the winning instant
comes back as two records and `only_or_missing` throws the bare `ArgumentError`
decision 5 exists to replace. Probed:

```
IDENTICAL DUPLICATE: at   -> 1
IDENTICAL DUPLICATE: asof -> 2
only_or_missing(asof) -> THREW: ArgumentError: Collection has multiple
                                elements, must contain exactly 1 element
```

The second symptom is worse, and follows from reading one *block* meaning
reading one *partition*. Where both candidate partitions carry the same instant
at different prices, `asof` and `at` disagreed, and `asof` took the silent side:

```
XPART CONFLICT asof -> 499.9
XPART CONFLICT at   -> THREW: ConflictingRecords: two SpotPrice records for SPY
                       at 2024-01-16T00:30:00 disagree (480.7 vs 499.9)
```

A number nobody verified, returned through the shape that reports the newest
known price, is precisely what decision 5 exists to prevent.

### Why it matters for the fix, not just the report

The follow-up recorded two routes. The cheap one — apply
`_collapse_duplicates!` to the single-block slice — closes the first symptom and
**not** the second, because the conflicting copy lives in the block that read
never touches. Without the probe, the cheap route looks adequate; with it, it is
disqualified on correctness rather than on taste.

### Fix

`asof` takes its winning instant from the backward walk and reads it through
`between`. `_candidate_partitions` over `[win, win]` spans exactly the overlap
window `at` already merges, so the two shapes can no longer disagree, and "every
spot read obeys the rule" is true by construction rather than by three call
sites remembering. Cost is two searchsorted pairs on cached vectors, on a shape
called roughly once per run in practice.

### Ripple

- `docs/modules/market_data.md`: the de-duplication paragraph gains the
  invariant and how each shape reaches it.
- `docs/proposals/pr9_followups.md` item 2 and
  [pr9_followups_plan.md](pr9_followups_plan.md) commit 1: closed, with the
  measured symptom recorded as what decided between the two routes.

---

## 2. The convention tightening left decision 5's justification stale

### What is wrong

Two paragraphs of `docs/modules/market_data.md`, written one commit apart,
disagreed.

Decision 5 (`3688040`) justified spot de-duplication partly by the partition
convention: "the convention permits an after-midnight row in both the earlier
partition's spill and the later partition's body". Finding A (`a48a123`, the
next commit) then tightened that convention to **time-ordered** — every row in
partition `D-1` precedes every row in partition `D` — which forbids exactly that
layout. The earlier paragraph kept citing the permission the later one had
withdrawn. The same sentence sat in the `_collapse_duplicates!` comment and in
the test tree's comment, and the test itself writes a layout the convention now
forbids.

### Root cause

Sequencing. The remaining-findings document flagged it ("decision 5 changes
`between` for spots in exactly this code. Sequence the two together"), and the
two commits *were* sequenced together — but only the code was reconciled, not
the prose that motivated it. A justification is not automatically re-checked
when the thing it cites changes one commit later.

### Decision

**Keep the rule, correct the motivation.** The rule is not weakened by the
tightened convention, because the convention is a statement about what a
collector writes, not a constraint the store enforces: the store is a directory
tree, and two partitions carrying one instant is unrepresentable in the
convention and perfectly representable on disk. A vendor re-delivering a minute
into one partition is the case that survives untouched.

The alternative — drop the cross-partition motivation as obsolete — was rejected
because finding 1 above demonstrates the cross-partition case is reachable and
was, until `f41a2b5`, silently wrong.

### Fix

Both texts now say the convention forbids the layout and nothing enforces it.
The convention paragraph also now says `asof` *takes its winning instant from*
the newest candidate partition rather than "returns at" it, which stays true
after finding 1's fix and keeps the interleaving argument intact.

---

## 3. `DerivationExhausted` named an instant it never tried

### What is wrong

The walk in `SurfaceReader.asof` threw with `cursor` — `win - Millisecond(1)`,
the cursor *after* the last attempt — in the `oldest` field, while
`showerror` says "the oldest tried was ..." and the docstring says the walk went
"as far as `oldest`". The reported instant was one millisecond before any
instant the walk actually examined.

### Root cause

The loop carries one variable for two meanings: where to look next, and how far
it got. They differ by the millisecond that makes `asof` strict.

### Decision

**Fix the field, not the message.** The message and the docstring both describe
the useful quantity — the oldest input timestamp examined is what a reader
compares against their data — so the field should carry it. The regression test
pinned the old value with a comment naming it as the cursor, which is how a
diagnostic off-by-one becomes permanent; it asserts the tried instant now.

Message only: nothing branches on the field, and the walk is unchanged.

---

## What this review did not change

Recorded so the next round does not re-derive them:

- **`serves` on the hot path.** Every map-level read now costs a `serves` call
  before the read. For `InMemory` that is a `Set{Any}` lookup, for the parquet
  readers a `Dict` hit on an already-listed partition vector. Looked at,
  judged negligible against a DuckDB query or a chain synthesis, and left alone.
  If finding B's measurement ever profiles the tick loop, this is in frame.
- **`InMemory` accepting conflicting rows** stays open, as decided in
  [pr9_followups.md](pr9_followups.md) item 1: it needs a per-kind "one record
  per instant" trait that does not exist.
- **Finding B** (quotes re-synthesized per call) stays deferred pending the
  measurement [pr9_followups_plan.md](pr9_followups_plan.md) commits 2 and 3
  specify, and the chain-per-tick half of it is unblocked and unstarted.
- **The duplication cleanups** (SQL helpers, reader scaffolding) stay in the
  triage table of [pr9_implementation_plan.md](pr9_implementation_plan.md).
  Note the reader-scaffolding item would have closed finding 1 as a side effect,
  which is a point in its favour it did not have before.

## Gate

`Pkg.test()` on the DevBox (2 cores, 3.7 GB, Julia 1.12.7): **1198 passed, 0
failed**, 1m09 wall. The sequence under review recorded 1181; all seventeen new
assertions belong to finding 1 — the `asof` cases, including the
cross-partition conflict tree the original write-up did not foresee. Finding 3
changed an existing assertion rather than adding one, and finding 2 is prose.
