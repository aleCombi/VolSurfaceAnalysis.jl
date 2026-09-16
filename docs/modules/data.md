# `data` module

Everything about a market datum: what one **is**, what can be **asked**
of it, and **where the answer comes from**. Three concepts; the source
tree mirrors them.

## Kind, protocol, provider

A **kind** is what a datum is: a plain immutable record type, keyed by
type rather than by name -- the config loader owns the only
string-to-type table.

The **protocol** is what can be asked of any kind.

A **provider** is where an answer comes from: a parquet tree, a single
constant record, a thing that derives one kind out of another. Nothing
distinguishes them to a consumer -- answering is the whole interface.

They are wired by a map holding one provider per kind, with a cut
(`TimeCut`) wrapped around it:

```
      consumer  ──asks──►  cut  ──►  map  ──►  provider for that kind
                                      ▲              │
                                      └──────────────┘
```

Every map-level call passes the map *itself* down as context. A raw
provider ignores it; a derived provider reads its own inputs back
through it; composition forwards it untouched. That re-entrancy is what
makes no-lookahead structural: the cut is applied once, at the top, and
every read underneath inherits it -- including inputs a derived provider
fetches on its own.

## What every kind must answer

- **`timestamp` is visibility time** -- when the record became knowable,
  and the only thing the cut filters on. A source that knows only
  effective dates must declare a visibility convention, because an
  assumption about lookahead belongs in identity.
- **A selector** distinguishes parallel series of one kind -- an
  `Underlying`, a `Currency`. A value type, not a string, so the
  contract is enforceable by dispatch; no non-kind type implements it.
- **A shape**: `snapshot(R)` is `true` for a kind holding one record per
  selector per instant, `false` for a grid kind holding many.

A curve kind carries a payload: the `Curve` *as of* a visibility time,
evaluated by calling it, with the curve types themselves left as pure
math. Stamping one at the start of time is how "always known" is said.

## The protocol

Four questions, each in two arities -- map-level for consumers,
provider-level for the things that answer, where `ctx` is the map or cut
the call arrived through:

| question | map-level | provider-level |
|---|---|---|
| what is there at this instant | `at(m, R, sel, ts)` | `at(p, ctx, R, sel, ts)` |
| what is there across a range | `between(m, R, sel, from, to)` | `between(p, ctx, R, sel, from, to)` |
| what is the latest visible | `asof(m, R, sel, ts)` | `asof(p, ctx, R, sel, ts)` |
| when is there anything | `timestamps(m, R, sel, from, to)` | `timestamps(p, ctx, R, sel, from, to)` |
| is this served at all | `serves(m, R, sel)` | `serves(p, ctx, R, sel)` |

Answers are sorted by `timestamp`; `asof` returns every record at the
largest visible one `<= ts`. Ranges are always bounded. `between`
promises an iterable, not a container, so a large provider yields
lazily, and the result expires when that provider closes. No answer is
ever
`missing` -- that is reserved for absent scalar fields *inside* a
record. There is no query language: anything beyond these four is plain
Julia over the result.

Which question a kind is read with follows its shape. `only_or_missing`
takes the single record of a singleton answer, or `missing` when there
is none, and throws when handed two.

| kind | natural call |
|---|---|
| `OptionQuote`, `OptionBar` | `at` |
| `SpotPrice`, `VolatilitySurface` | `only_or_missing(at(...))` |
| `RateCurve`, `DivCurve` | `only_or_missing(asof(...))` |
| event kinds (future) | `between`, then filter on the effective field |

**An empty answer means temporal absence and nothing else** (design rule
7). Every other unanswerable question has a name:

| state | answer |
|---|---|
| nothing serves this selector | throws `UnservedSelector` (structural) |
| served, nothing at this instant | empty -- the only legitimate case |
| two records, two answers | throws `ConflictingRecords` |
| a derivation failing past its bound | throws `DerivationExhausted` |

Structural beats temporal: an unserved selector throws even past a cut's
cutoff. The check lives at the map level, so provider-level calls stay
unchecked and internal delegation does not re-trigger it.

**`serves` is three-valued**, and `missing` -- "cannot say" -- is
required. A provider that has not yet opened its storage cannot answer
without walking a tree, and a derived provider does not answer at all --
it delegates, so the
refusal names the real cause (a surface asked for SPX reports
`OptionBar`/SPX unserved, not "no surface"). A provider answering
`false` also says what it *does* serve, which is what makes the message
enough to fix a config.

**A derived kind's `asof` walks back.** `asof` promises the largest
timestamp at which a record of that kind exists, which for a derived
kind is not where its input exists: a provider whose build fails at the
newest input instant keeps walking. The walk is bounded, so repeated
failure surfaces as a truncated dataset rather than as absence. For the
same reason `timestamps` and `between` over a derived kind are
**over-estimates** -- they report the input grid. Making them exact
would mean building every record in the range.

## What a provider owes

A provider has two forms. A **spec** is the declared one: immutable,
resource-free, what config builds and persistence writes --
`ParquetOptionBars(root)` is a spec. Opening it yields a **reader**,
which owns what cannot be written down: the connection, the caches, the
partition list. Specs needing nothing are their own reader.

A spec is per *storage*, not per kind -- the selector is a query argument,
so one spec serves every series in its tree. Constructing one touches no
storage: a missing root is refused at opening, never at construction.
Opening and closing are the project's own generics with explicit opt-ins,
so a spec that forgot to opt in fails a check rather than silently
becoming a no-op. **Use after close throws** rather than being left to the
storage, because the storage segfaults.

Two invariants belong to the store rather than to any reader.

**Partitions are time-ordered, with a one-day spill.** A partition may
hold rows past its own date, because the collector writes a US session
into its local date, so every shape consults the date asked about and
the one before it. Every row in one partition precedes every row in the
next, and **the four questions agree only under that ordering**: `asof`
takes its instant from the newest candidate partition while `at` and
`timestamps` merge both, so an interleaved layout would break
`asof == at(last(timestamps(...)))`, and a lazy `between` concatenates
candidates without a cross-partition sort, so it would yield records out
of order.

**A snapshot kind de-duplicates where rows enter.** A vendor can
re-deliver a minute, and consulting two partitions can read one row
twice; either aborts a read through `only_or_missing`. So the reader
applies one rule after its sort: *equal timestamp and equal value
collapse silently; equal timestamp and different value throws
`ConflictingRecords`*, naming both. Taking the first would be a silent
choice between two answers on a number nobody verified. Every read
inherits it rather than remembering it, because `at` and `asof` both
reach their instant through `between`. Grid kinds are excluded
deliberately: many records per instant is their shape, so their
duplicate key is the contract, not the instant.

## Decisions

| Decision | Why |
|---|---|
| **Bar end is the visibility time -- a constant, not a setting** | Every value read off a minute bar, and any spread synthesized from one, is knowable only once the minute has finished. A bar-open stamp hands each fill and settlement price up to a minute of future information *below* the cut, where nothing can see it. Offering both conventions would leave the incorrect clock reachable, so there is no config key. A feed whose bars are not one minute needs its own reader stating its own interval. |
| **`OptionBar` is a first-class vendor kind, not a reader internal** | Keeps the synthesis policy explicit and testable instead of buried in a reader. Policies depend on `OptionQuote`, so a live feed carrying real quotes simply does not configure a synthesizer. |
| **The synthesizer is declared, never defaulted** | Bid/ask construction is part of provenance, so its parameter is required at the type level and appears in the experiment record. The store carries OHLCV and no bid/ask, so today's quotes are *synthesized, not observed*. Missing inputs yield a missing bid/ask rather than an invented market. |
| **Selector types hash by content** | The default falls back to `objectid`, which changes with every build for a type in a precompiled package, so a `Dict` keyed on a selector would iterate in build-dependent order. |
| **A ticker that disagrees with its partition throws** | Under `symbol=` partitioning a foreign ticker is a corrupt store, not a row to skip. |
| **The protocol never names a concrete provider** | Generics are declared by the protocol and implemented outward, so the dependency runs one way and the concepts do not fold back on each other. |
| **One provider per kind** | Comparing two synthesizers, or two surface conventions, is two runs -- which is what the run store is for. |

## Config and identity

Config builds specs -- one `[data.<kind>]` table per kind plus a clock --
through the builders in the [`experiment`](experiment.md) loader.
Identity projects one entry per kind, the clock, and the per-spec fields
that decide which records are served. Cache bounds are arguments to
opening that no config table reaches, so a machine knob cannot become a
false difference between runs. The bar-stamp convention projects as a
constant: no key can vary it, but it decides which minute every decision
reads. A spec's root sits in a reserved `dataset` slot, where a logical
dataset id and version would go.

## Conventions consulted

Checked before these names and shapes were fixed (design rule 5); the
sources were the depot copies of Tables.jl, DBInterface.jl and DuckDB.jl,
Julia 1.12 Base and manual, and the documented APIs of TimeSeries.jl,
DataInterpolations.jl, Impute.jl, StructTypes.jl and JSON3.jl.

- **`between` as the range verb.** TimeSeries.jl uses a `from`/`to` pair
  and pandas has no range verb. `Base.between` exists, unexported and
  `Integer`-only; the project defines its own generic, never imports it,
  and a test pins the Base method count.
- **`asof` as "latest at or before".** Julia has no settled name
  (TimeSeries.jl `findwhen`, DataInterpolations.jl left-constant
  interpolation, Impute.jl `locf`); the name is pandas'. The other verbs
  collide with nothing in Base or Dates.
- **Opening and closing as a project-owned pair.** DBInterface.jl
  declares its own `connect`/`close!` generics with a scoped form and
  DuckDB.jl follows it, so nothing is added to `Base.open`/`Base.close`
  and the scoped form matches the repo's `with_run_store`.
- **Kind as a type marker after the source; providers duck-typed.**
  `at(src, R, sel, ts)` follows `read(io, T)` and `parse(T, s)`; the kind
  traits are StructTypes-style; there is no abstract provider supertype,
  as Tables.jl has none for tables.
- **`between` yields records, not tables.** `Tables.partitions` is an
  iterator of tables and DuckDB's is forward-only, so the lazy iterator
  mirrors that contract instead of implementing the Tables hook.
- **Style.** No `get_` prefix on accessors, bang only on mutation, files
  `include`d into the one module with no submodules.
