# PR #9 follow-ups

Status: two items. Item 2 (spot `asof`) is **implemented**; item 1
(`InMemory`) is decided and open. Work sequenced in
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

**Implemented**, by the route this section preferred: `asof` walks the
partition list back to the winning instant and then reads it through
`between`. A gap in decision 5 as specified, not a deviation from it.

Decision 5 says to apply collapse-or-throw "in `between`, after the sort, so it
covers both the repeated row and the cross-partition overlap. The default `at`
inherits it", plus a uniqueness pass on `timestamps`. The implementation does
exactly that.

`asof(::ParquetSpotsReader, ...)` reached neither. It walked the partition list
backward, and on the first block holding a timestamp `<= ts` returned
`_append_spots!(SpotPrice[], u, b, win, win)` -- a direct read of one block, not
a call to `between`. So a vendor row delivered twice at the winning instant came
back as two records, and `only_or_missing` threw the bare `ArgumentError` that
decision 5 exists to replace with a named, informative one.

**The second symptom, found while reviewing the sequence and worse than the
first.** Reading one block also means reading one *partition*, so where the two
candidate partitions carry the same instant at different prices, `asof` and `at`
disagreed -- and `asof` took the silent side. Measured on a two-partition tree
with the spill row at 480.7 in the earlier partition and 499.9 in the later:

```
asof -> 499.9
at   -> THREW: ConflictingRecords: two SpotPrice records for SPY
                at 2024-01-16T00:30:00 disagree (480.7 vs 499.9)
```

`asof` returning a number nobody verified is exactly what decision 5 exists to
prevent, and it decides between the two routes below: applying
`_collapse_duplicates!` to the single-block slice fixes the first symptom and
not this one, because the conflicting copy is in the block it never reads.

Why it was left alone: the correctness document describes the defect as the
reader "gathers candidate partitions, concatenates them, sorts by timestamp, and
never checks for duplicates", which is a description of `between`. `asof` on
spots takes a different path that the analysis did not cover, and widening a
decided fix past what it specifies is worse than recording the gap.

Two ways to close it; the second is what landed:

- **Route `asof` through the same helper.** One line: apply
  `_collapse_duplicates!` to the single-instant result before returning. Cheapest,
  and keeps the rule in one place. Rejected: it closes the first symptom only,
  and it leaves three call sites each remembering to apply the rule, which is the
  shape of the defect being fixed.
- **Make `asof` call `between(r, ctx, SpotPrice, u, win, win)`.** Taken. Removes
  the duplicated block-slicing entirely and makes "every read obeys the rule" true
  by construction rather than by three call sites remembering to. Slightly more
  work per call -- it re-derives the candidate partitions and slices both blocks
  instead of the one already in hand, which is why the direct read existed; both
  are cached, so it is two searchsorted pairs on vectors already in memory, on a
  shape called roughly once per run in practice.

The bars reader has no equivalent gap: bars are deliberately outside the rule
(see decision 5's "Deferred"), because a chain has many rows per timestamp by
design.

Note this interacts with the deferred "duplicated reader scaffolding" item: if
`ParquetBarsReader` and `ParquetSpotsReader` are ever unified, the backward walk
becomes one implementation and this gap closes with it.
