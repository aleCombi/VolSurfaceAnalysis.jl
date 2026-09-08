# PR #9 follow-ups

Status: two items, one decided and one open. Work sequenced in
[pr9_followups_plan.md](pr9_followups_plan.md), which also plans the
measurement finding B was deferred pending. Both come out of implementing
[pr9_correctness_fixes.md](pr9_correctness_fixes.md); neither was in scope for
that sequence. The larger deferrals (finding B, the SQL and reader duplication)
stay in the triage table of
[pr9_implementation_plan.md](pr9_implementation_plan.md) and are not repeated
here.

---

## 1. `InMemory` should reject conflicting rows, like the parquet reader

**Decided.** `InMemory` behaves like the rest.

The remaining-findings document originally *refuted* "InMemory can hold
duplicate rows for one selector and instant", on the grounds that it is
deliberately permissive and `only_or_missing` enforces the singleton rule
downstream. Decision 5 reopened that: the parquet spot reader now collapses
identical rows at one instant and throws `ConflictingRecords` when two rows at
one instant disagree, so the fixture provider is the only place left where a
store may quietly contradict itself.

Two providers disagreeing about what a duplicate means is exactly the ambiguity
findings 2, 3 and 5 exist to remove. A fixture that cannot represent a state the
real reader rejects is a better fixture, not a worse one.

What this means concretely, to be settled when it is implemented:

- **Where the check goes.** The parquet rule is applied on read, in `between`,
  because a reader assembles its result from partitions it did not write.
  `InMemory` is handed its whole world at construction, so the natural place is
  the inner constructor -- one pass over the already-sorted rows, paid once per
  fixture rather than once per read.
- **What the key is.** `SpotPrice` conflicts on `(selector, timestamp)` with
  different `price`. That generalises to any *snapshot* kind, but not to a grid
  kind: an `OptionQuote` or `OptionBar` chain has many rows per instant by
  design, so its de-duplication key is the contract, not the instant. `InMemory`
  is generic over `R`, so the check needs a per-kind trait saying "one record per
  selector per instant" rather than a rule hard-coded to `SpotPrice`. That trait
  does not exist yet and is the substance of this item; the `market_data.md`
  shapes table already draws the same distinction informally ("grid, one per
  instant" vs "grid, many per instant").
- **Blast radius.** Every fixture in the suite that carries two rows for one
  selector and instant would have to be corrected. None is known to, but this is
  the part to measure before writing code.

Consequence worth stating: the "Refuted during verification" entry in
[pr9_remaining_findings.md](pr9_remaining_findings.md) is now wrong and should be
struck when this lands, not merely annotated.

---

## 2. Spot `asof` does not go through the collapse-or-throw rule

**Open.** A gap in decision 5 as specified, not a deviation from it.

Decision 5 says to apply collapse-or-throw "in `between`, after the sort, so it
covers both the repeated row and the cross-partition overlap. The default `at`
inherits it", plus a uniqueness pass on `timestamps`. The implementation does
exactly that.

`asof(::ParquetSpotsReader, ...)` reaches neither. It walks the partition list
backward, and on the first block holding a timestamp `<= ts` it returns
`_append_spots!(SpotPrice[], u, b, win, win)` -- a direct read of one block, not
a call to `between`. So a vendor row delivered twice at the winning instant still
comes back as two records, and `only_or_missing` still throws the bare
`ArgumentError` that decision 5 exists to replace with a named, informative one.

Why it was left alone: the correctness document describes the defect as the
reader "gathers candidate partitions, concatenates them, sorts by timestamp, and
never checks for duplicates", which is a description of `between`. `asof` on
spots takes a different path that the analysis did not cover, and widening a
decided fix past what it specifies is worse than recording the gap.

Two ways to close it, to be chosen when it is:

- **Route `asof` through the same helper.** One line: apply
  `_collapse_duplicates!` to the single-instant result before returning. Cheapest,
  and keeps the rule in one place.
- **Make `asof` call `between(r, ctx, SpotPrice, u, win, win)`.** Removes the
  duplicated block-slicing entirely and makes "every read obeys the rule" true by
  construction rather than by three call sites remembering to. Slightly more work
  per call (it re-derives candidate partitions), which is why the direct read
  exists.

The bars reader has no equivalent gap: bars are deliberately outside the rule
(see decision 5's "Deferred"), because a chain has many rows per timestamp by
design.

Note this interacts with the deferred "duplicated reader scaffolding" item: if
`ParquetBarsReader` and `ParquetSpotsReader` are ever unified, the backward walk
becomes one implementation and this gap closes with it.
