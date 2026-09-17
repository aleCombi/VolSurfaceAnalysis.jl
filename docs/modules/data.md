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

A provider has two forms. A **spec** is the declared one: immutable,
resource-free, what config builds and persistence writes --
`ParquetOptionBars(root)` is a spec. Opening it yields a **reader**,
which owns what cannot be written down: the connection, the caches, the
partition list. Specs needing nothing are their own reader. A spec is
per *storage*, not per kind -- the selector is a query argument, so one
spec serves every series in its tree.

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
- **A shape**: whether the kind is a *snapshot*, holding one record per
  selector per instant, or a *grid*, holding many at one instant. It is
  declared as a trait on the type, so the protocol can branch on it
  without having a record in hand.

A kind's payload may be a math object owned by [`pricing`](pricing.md),
and whether it needs a record wrapped around it follows from the object
rather than from taste. A `Curve` is meaningful for any currency at any
instant, so `RateCurve` / `DivCurve` are what stamp it and say whose it
is -- and one stamped at the start of time is how "always known" is
said. A surface is already one underlying at one instant, so it *is* the
kind and carries no wrapper. Giving the surface a curve's split is
deferred, not rejected.

## The protocol

Four shapes of read -- what is there at this instant (`at`), what is
there across a range (`between`), what the latest visible record is
(`asof`), when there is anything at all (`timestamps`) -- and one
structural question, whether a selector is served (`serves`). Five, and
no sixth.

Each is asked at two levels: the one a consumer asks of a map or a cut,
and the one a provider answers. What separates them is not the call but
the context -- only a provider is handed the map or cut the call arrived
through, which is what the re-entrancy above rests on. Keeping the
levels apart is also what lets the structural check sit at the map
level, where it is asked once, instead of in every provider.

Answers are sorted by `timestamp`; `asof` returns every record at the
largest visible timestamp at or before the instant asked about. Ranges
are always bounded. `between` promises an iterable, not a container, so
a large provider yields lazily, and the result expires when that
provider closes. No answer is ever `missing` -- that is reserved for
absent scalar fields *inside* a record. There is no query language:
anything beyond these five is plain Julia over the result.

Which question a kind is read with follows from its shape, not from a
convention a consumer has to remember: a grid kind is read per instant, a
snapshot kind per instant or as-of depending on whether it is stamped at
every instant or holds until superseded. Reading a snapshot through
`only_or_missing` is what turns two records at one instant into a refusal
instead of a silent pick between two answers.

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
it delegates, so the refusal names the real cause (a surface asked for SPX
reports `OptionBar`/SPX unserved, not "no surface"). A provider answering
`false` also says what it *does* serve, which is what makes the message
enough to fix a config.

**A derived kind's `asof` walks back.** `asof` promises the largest
timestamp at which a record of that kind exists, which for a derived
kind is not where its input exists: a provider whose build fails at the
newest input instant keeps walking. The walk is bounded, so repeated
failure surfaces as a truncated dataset rather than as absence. For the
same reason `timestamps` over a derived kind is an **over-estimate** --
it reports the input grid, so it can name instants where no derived
record exists. Making it exact would mean building every record in the
range. `between` traverses that same grid but yields only records that
were built, so its output is exact and its cost is not.

**A derived provider's cache must not be reachable by the cut.** Caching
is what makes a derived kind affordable, and it is also where lookahead
could re-enter after the cut has already been applied. The key is the
selector and the instant, and nothing else. That holds only because every
input a derivation reads is at or before the instant asked about, which
is what makes an entry valid under any cut at or after it -- the same
object comes back through the bare map and through any such cut. A
derivation that read an input past its own instant would break the
invariant, and the breakage would surface as one cut serving another's
record rather than as an error. Absence is cached on the same key, so a
derivation that found nothing is not retried.

## What a provider owes

Answering is the whole interface, but a provider holding a resource owes
three things beyond it.

- **Opt in to the lifecycle.** Opening and closing are the project's own
  generics with no fallback, so a provider type with no `open_data`
  method fails a load-time check naming its kind, rather than being
  silently treated as needing nothing.
- **Refuse late.** Constructing a spec touches no storage; a missing
  root is refused at opening, never at construction.
- **Throw on use after close**, rather than leaving it to the storage,
  because the storage segfaults.

## What the protocol assumes about a store

Two of the protocol's promises are not self-enforcing. They hold only if
the rows on disk are arranged a particular way, so a store that breaks
them is wrong in a way nothing reports.

**Partitions must be globally time-ordered**: every row in one precedes
every row in the next. `asof` takes its instant from the newest
candidate partition while `at` and `timestamps` merge candidates, so an
interleaved layout breaks the agreement between them: the latest visible
record would stop being the one found at the last instant the kind
reports having anything at.

**A snapshot kind's duplicates must be resolved where rows enter.** Read
through `only_or_missing`, one instant carrying two records aborts, so
the reader settles it at the boundary: *equal value collapses, different
value throws `ConflictingRecords`*, naming both. Resolving it by taking
the first would be a silent choice between two answers on a number
nobody verified. Grid kinds are excluded: many records per instant is
their shape, so their duplicate key is the contract, not the instant.

The Massive tree meets the first by construction rather than by
enforcement, because the collector writes a US session into its local
date -- so a partition holds rows past its own date and every shape
consults the date asked about and the one before it. That is this
store's reason; the requirement is any store's.

## Decisions

| Decision | Why |
|---|---|
| **Bar end is the visibility time -- a constant, not a setting** | Every value read off a minute bar, and any spread synthesized from one, is knowable only once the minute has finished. A bar-open stamp hands each fill and settlement price up to a minute of future information *below* the cut, where nothing can see it. Offering both conventions would leave the incorrect clock reachable, so there is no config key. A feed whose bars are not one minute needs its own reader stating its own interval. |
| **`OptionBar` is a first-class vendor kind, not a reader internal** | Keeps the synthesis policy explicit and testable instead of buried in a reader. Policies depend on `OptionQuote`, so a live feed carrying real quotes simply does not configure a synthesizer. |
| **The synthesizer is declared, never defaulted** | Bid/ask construction is part of provenance, so its parameter is required at the type level and appears in the experiment record. The store carries OHLCV and no bid/ask, so today's quotes are *synthesized, not observed*. A bar missing an input yields a missing bid/ask, never a zero spread: a fallback there would invent a market that did not trade. |
| **Selector types hash by content** | The default falls back to `objectid`, which changes with every build for a type in a precompiled package, so a `Dict` keyed on a selector would iterate in build-dependent order. |
| **A ticker that disagrees with its partition throws** | Under `symbol=` partitioning a foreign ticker is a corrupt store, not a row to skip. |
| **A derived provider may call into `pricing`; nothing else in `data` may** | Deriving a kind means computing it, and the computation is valuation math, which design rule 9 puts in `pricing`. The crossing is one-way -- `pricing` never reads through a map -- so the dependency stays a line rather than a cycle. Today `SurfaceFrom` is the only provider that crosses it. |
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
reads. A spec's root sits in a `dataset` slot. Dataset versioning is dropped
rather than deferred ([status.md](../status.md)): the slot holds a path,
the trees are trusted as stable, and a divergence attributes to code or
dependencies by elimination.

## Conventions consulted

One entry per naming decision, with the source checked (design rule 5).

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
- **Kind as a type marker after the source; providers duck-typed.** A
  read names its source first and the kind as a type marker after it,
  following `read(io, T)` and `parse(T, s)`; the kind traits are
  StructTypes-style; there is no abstract provider supertype, as
  Tables.jl has none for tables.
- **`between` yields records, not tables.** `Tables.partitions` is an
  iterator of tables and DuckDB's is forward-only, so the lazy iterator
  mirrors that contract instead of implementing the Tables hook.
