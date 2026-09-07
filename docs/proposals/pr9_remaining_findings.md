# PR #9 remaining review findings

Status: **not decided.** Companion to
[pr9_correctness_fixes.md](pr9_correctness_fixes.md), which covers the six
confirmed correctness defects and is decided.

This file records everything else the review of `data-kinds` (PR #9) produced:
four lower-confidence findings and a set of cleanup items. Nothing here has an
agreed fix. Each entry states what was observed and, where the reviewer offered
one, the suggested direction. Treat the suggestions as input, not as decisions.

---

## Lower-confidence findings

These were posted as inline comments on the PR. Each was verified as plausible
rather than confirmed, generally because the failing input is permitted by the
documented conventions but does not occur in the data actually collected.

### A. Partition overlap is handled inconsistently across shapes

`asof` for bars and for spots both return at the first partition, walking
backward, that holds any row at or before the requested instant. The `at`,
`between` and `timestamps` shapes instead merge both candidate partitions.

The documented convention allows a partition for date D to hold rows spilling
into the early hours of D+1. Under that convention the two behaviours disagree:
`asof` can return an earlier record than the one `at`/`timestamps` report as the
latest, which breaks the identity `asof == at(last(timestamps(...)))` asserted in
`test/market_data/test_parquet.jl`.

The lazy `between` for bars concatenates the two candidate partitions without a
cross-partition sort, so under the same layout it can yield out-of-order records
and make `by_timestamp` throw.

Unreachable with real vendor output, because a local-date collector does not
write pre-dawn rows into the following partition. Neither the code nor the stated
convention excludes it.

Suggested direction: either take the maximum over both candidate partitions in
`asof`, as `at` already does, or tighten the documented convention so partitions
are time-disjoint.

**Note:** decision 5 in the correctness document changes `between` for spots in
exactly this code. Sequence the two together.

### B. Quotes are re-synthesized from bars on every call

`at(::QuotesFromBars, ...)` rebuilds the entire `OptionQuote` chain from the
cached `OptionBar` chain on every call. The engine calls it at least twice per
tick, plus once per order inside the per-order loop: once from the surface
reader, once from `decide`, and once per order from `resolve_quote`.

The previous data layer cached synthesized chains per timestamp, so a tick that
now performs several full-chain synthesis passes previously performed one. On a
dense chain this is thousands of freshly allocated records per tick.

Suggested direction: give the provider a reader holding a bounded cache keyed on
underlying and timestamp, mirroring `SurfaceReader`, and hoist the chain fetch
out of the per-order loop in `run_backtest` so `resolve_quote` receives the
chain.

This is the only performance finding in the review. It is worth judging against
a measurement rather than on the reasoning alone.

### C. A sub-second range bound is truncated

The SQL timestamp helper formats to whole seconds, so a `between` whose lower
bound carries sub-second precision admits the bar sitting at the floor of that
bound, while `at` and `timestamps` compare full-precision values. The three
shapes disagree on the same bound.

Nothing excludes sub-second input: `between` is public, TOML datetimes with
fractional seconds parse through, and `TimeCut` passes bounds through untouched.
Stored bars are minute-aligned, so no current caller reaches it.

### D. Documentation drift

Three separate places, all covered by design rule 1:

- The "add a new type" checklist in `docs/modules/experiment.md` lists the curve,
  synthesizer, policy and agent builder tables but omits the provider builder
  table, which is the one registry a new provider spec must appear in. The diff
  renamed that table in code and dropped the old name from the doc without adding
  the new one.
- `docs/modules/surfaces.md` still attributes raw chain access to the old data
  module, although `docs/modules/data.md` now assigns readers to `market_data`.
- The `core_hash` docstring still describes the hashed components as source,
  agent and window, while the implementation hashes data, clock, agent and
  window.

---

## Cleanup items

Confirmed during review, below the reporting cap. None affect behaviour.

- **Dead duplicate check.** The lifecycle check runs twice in
  `build_market_data`; the copy inside the input-kinds loop is unreachable in any
  case the following loop does not already cover.
- **Duplicated SQL helpers.** The parquet timestamp formatter duplicates one in
  the store module, and the parquet path quoter duplicates one in the polygon
  module.
- **Duplicated reader scaffolding.** `ParquetBarsReader` and
  `ParquetSpotsReader` repeat the same open, close, partition-listing, backward
  walk and grid code, differing only in how a timestamp is read from a partition.
- **Unused import.** `using SHA` in the store module.
- **Unreachable cache knobs.** The `[data.<kind>]` builders silently drop unknown
  keys, including `max_days_cached` which was meaningful in the old layer. And
  `open_data(::MarketData)` never forwards its `max_days`, `max_chains` or
  `max_surfaces` keyword arguments, so no cache bound can be set from a run at
  all. Decision 4 in the correctness document adds a lookback bound as a spec
  field, which is a different and working route; the keyword route beside it is
  dead either way.

---

## Refuted during verification

Recorded so they are not re-raised:

- The partition list is snapshotted per reader rather than re-read. Documented
  design.
- `InMemory` can hold duplicate rows for one selector and instant. By design,
  enforced downstream at `only_or_missing`.
- A NULL close aborts a spot read. Unchanged from master.

**Caveat on the second.** Decision 5 in the correctness document makes
conflicting spot rows an error in the parquet reader. If that lands, the fixture
provider should probably agree rather than staying permissive, so this refutation
is partially reopened.
