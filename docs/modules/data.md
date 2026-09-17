# `data` module

What a market datum **is** (kind), what can be **asked** of it
(protocol), and **where the answer comes from** (provider). The source
tree has one folder per concept.

## Kind, protocol, provider

A **kind** is a plain immutable record type. Kinds are keyed by type;
the config loader owns the only string-to-type table.

A **provider** answers the protocol for one kind. Nothing else
distinguishes providers to a consumer. A provider has a declared form,
the **spec** (immutable, resource-free, what config builds and
persistence writes), and an opened form, the **reader** (connection,
caches, partition list). A spec needing nothing is its own reader. A
spec is per storage, not per selector: one tree, every series in it.

A **map** holds one provider per kind; a **cut** (`TimeCut`) wraps the
map. Every map-level call passes the map itself down as context, and a
derived provider reads its inputs through that context. So the cut is
applied once, at the top, and every read underneath inherits it. That
is what makes no-lookahead structural.

## What every kind answers

- `timestamp` is **visibility time**, the only thing the cut filters
  on. A source that knows only effective dates must declare its
  visibility convention.
- A **selector** names one series of a kind (`Underlying`, `Currency`).
  A value type, so dispatch enforces the contract.
- A **shape** trait: *snapshot* (one record per selector per instant)
  or *grid* (many). Declared on the type, so a reader can settle
  duplicates as rows enter.

A payload that is meaningful for any selector at any instant gets a
record wrapped around it: a `Curve` is stamped by `RateCurve` /
`DivCurve`, and a stamp at the start of time means "always known". A
surface is already one underlying at one instant, so it is its own
kind. Giving it a curve's split is parked, not rejected.

## The protocol

Four reads and one structural question, no sixth: `at` (this instant),
`between` (a range), `asof` (latest visible), `timestamps` (the grid),
`serves` (is this selector served).

Each has a map level, which a consumer calls, and a provider level,
which receives the context. The structural check runs at the map level
only, so provider-level calls and internal delegation are unchecked.

Answers are sorted by `timestamp`. Ranges are bounded. `between` is an
iterable that expires when its reader closes. No read answers
`missing`; that is for absent fields inside a record.

**Empty means temporal absence, nothing else** (design rule 7). Every
other unanswerable question is named:

| state | answer |
|---|---|
| nothing serves this selector | `UnservedSelector`, thrown even past a cutoff |
| served, nothing at this instant | empty |
| two records, two answers | `ConflictingRecords` |
| a derivation failing past its bound | `DerivationExhausted` |

**`serves` is three-valued.** `missing` is "cannot say": a spec whose
tree is not open yet, or a derived provider, which delegates so the
refusal names the input that is actually unserved.

**A derived `asof` walks back**, bounded, past input instants where the
build fails. `timestamps` over a derived kind reports the input grid,
an over-estimate; `between` yields only what built.

**A derived cache is keyed on (selector, instant) and nothing else.**
Safe because every input a derivation reads is at or before its
instant, so an entry is valid under any cut at or after it. Absence is
cached on the same key.

## What a provider owes

- **Opt in to the lifecycle.** `open_data` has no fallback; a spec with
  no method fails at load time.
- **Refuse late.** Construction touches no storage; a missing root is
  refused at opening.
- **Throw on use after close.** The storage would segfault.

## What the protocol assumes about a store

- **Partitions are globally time-ordered.** `asof` reads the newest
  candidate partition; `at` and `timestamps` merge candidates. An
  interleaved layout makes them disagree, and nothing reports it.
- **A snapshot kind's duplicates are resolved where rows enter**: equal
  collapses, different throws `ConflictingRecords`. Taking the first
  would be a silent choice. Grid kinds are exempt; many per instant is
  their shape.

The Massive collector writes a US session into its local date, so a
partition spills past its date and every read consults the date asked
about and the one before.

## Decisions

| Decision | Why |
|---|---|
| **Bar end is visibility time, a constant** | A bar's values are knowable only once the minute has ended. A bar-open stamp puts up to a minute of the future below the cut. No config key, because the wrong clock must not be reachable. |
| **`OptionBar` is a kind, not a reader internal** | The synthesis policy is explicit and testable. A feed carrying real quotes configures no synthesizer. |
| **The synthesizer is declared, never defaulted** | Bid/ask construction is provenance. A bar missing an input yields a missing bid/ask, never a zero spread, which would invent a market that did not trade. |
| **Selector types hash by content** | The `objectid` default changes per build, so a `Dict` keyed on a selector would iterate in build-dependent order. |
| **A ticker disagreeing with its partition throws** | Under `symbol=` partitioning that is a corrupt store, not a row to skip. |
| **Only a derived provider may call into `pricing`** | Deriving is computing, and the computation lives in `pricing` (rule 9). The crossing is one-way. |
| **The protocol never names a concrete provider** | Generics are declared inward and implemented outward. |
| **One provider per kind** | Comparing two synthesizers is two runs. |

## Config and identity

Config builds specs, one `[data.<kind>]` table per kind plus a clock,
through the [`experiment`](experiment.md) loader. Identity projects each
spec's record-determining fields; cache bounds are opening arguments no
config reaches, so a machine knob cannot fork a run id. The bar-stamp
convention projects as a constant. A spec's root sits in a `dataset`
slot; versioning is dropped ([status.md](../status.md)): the trees are
trusted stable and a divergence attributes to code by elimination.

## Conventions consulted

One entry per naming decision (design rule 5).

- **`between`.** TimeSeries.jl uses `from`/`to`; pandas has no range
  verb; `Base.between` is unexported and `Integer`-only, so the project
  owns its generic and a test pins the Base method count.
- **`asof`.** No settled Julia name (TimeSeries.jl `findwhen`,
  DataInterpolations.jl left-constant, Impute.jl `locf`); pandas'.
- **`open_data` / `close_data!`.** DBInterface.jl and DuckDB.jl own
  their `connect`/`close!` pair rather than extending `Base`, and the
  scoped form matches `with_run_store`.
- **Kind as a type marker after the source; providers duck-typed.**
  `read(io, T)` / `parse(T, s)` order; StructTypes-style traits; no
  abstract provider supertype, as Tables.jl has none.
- **`between` yields records, not tables.** `Tables.partitions` is an
  iterator of tables and DuckDB's is forward-only.
