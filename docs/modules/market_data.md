# `market_data` module

How market data is *obtained*: the protocol every consumer reads
through, the specs that describe where records come from, and (as the
layer lands) the readers, map, time cut and lifecycle behind it. What a
datum *is* stays in [`data`](data.md): record types and vendor row
mapping. Design and rationale are in
[proposals/data_kinds.md](../proposals/data_kinds.md); this doc states
the rules the code keeps.

Status: landing beside the old `DataSource` / `ModelDataSource` layer,
which is deleted at step 3 of the plan. Until then both exist and
nothing downstream reads through this module yet.

## Kinds and the visibility rule

A kind is a plain immutable record type (`OptionBar`, `OptionQuote`,
`SpotPrice`; curve and surface kinds follow). Kinds are keyed **by
type**: the config loader owns the only string-to-type table, nothing
on the runtime path knows a name. Every kind carries:

- **`timestamp` is visibility time** — the moment the record became
  knowable. The time cut filters on it and on nothing else; any other
  date a record carries (expiry, ex-date, effective date) is an
  ordinary field. Consequence: a source that only knows effective dates
  must declare a visibility convention on its spec, where an assumption
  about lookahead belongs in identity.
- **A selector** — the value that distinguishes parallel series of one
  kind: `selector(r)` (an `Underlying` for market data, a `Currency`
  for rate curves) and the trait `selector_type(R)`. Every kind defines
  both; `Currency` is a distinct value type, not a string, so the
  contract is enforceable by dispatch.

*Bar-time allowance.* Polygon minute bars are stamped at the bar open
while their close, high and low are knowable only at bar end. The open
stamp is kept as the visibility time, so a decision at `t` sees the
`[t, t+1min)` bar. This is a stated one-minute simplification, not a
shift; a bar-end stamp option on the parquet spec is backlog.

## The protocol

Four shapes, each in two arities:

| shape | map-level (consumers) | provider-level (specs, readers) |
|---|---|---|
| exact instant | `at(m, R, sel, ts) -> Vector{R}` | `at(p, ctx, R, sel, ts)` |
| range | `between(m, R, sel, from, to) -> iterable of R` | `between(p, ctx, R, sel, from, to)` |
| latest visible | `asof(m, R, sel, ts) -> Vector{R}` | `asof(p, ctx, R, sel, ts)` |
| grid | `timestamps(m, R, sel, from, to) -> Vector{DateTime}` | `timestamps(p, ctx, R, sel, from, to)` |

`ctx` is the map (or cut) the call came through. Raw providers ignore
it, derived providers read their inputs through it, composition
forwards it untouched. Every provider also answers `kind(p)`.

Rules:

- Results are sorted by `timestamp`. `at` and `between` return only
  records in range; `asof` returns every record at the largest visible
  timestamp `<= ts` (a whole chain for grid kinds, one record for a
  snapshot kind). **Empty means absent** for all four shapes; no shape
  returns `missing`, which is reserved for absent scalar fields inside
  a record.
- `between` promises an iterable, not a container. Large providers
  yield lazily, one partition in memory at a time; the iterator is
  valid only while its reader is open.
- `asof` has no default. Every provider implements it with what its
  storage does well.
- Ranges are always bounded. There is no discovery verb and `asof` is
  not a scan.
- The provider-level default `at` is `collect(R, between(p, ctx, R,
  sel, ts, ts))`; providers override it when they have a faster path.
- The data layer defines no query language. Anything beyond these
  shapes is plain Julia over the result.

Which shape a consumer reaches for follows the kind: grid kinds with
many records per instant (`OptionQuote`, `OptionBar`) use `at`; grid
kinds with one record per instant (`SpotPrice`) use
`only_or_missing(at(...))`; snapshot kinds that hold until superseded
(curves) use `only_or_missing(asof(...))`; event kinds use `between`
over a bounded lookback and filter on the effective field.

## Library

Ordinary functions over protocol results, not part of the protocol:

- `only_or_missing(v)` — the one record of a singleton result, or
  `missing` when empty; more than one record throws, because a
  singleton kind with two records at one instant is a data error.
- `by_timestamp(it)` — lazy run-length grouping of a sorted iterable
  into `(timestamp, Vector{R})` pairs; reads one group plus one record
  ahead, so it composes with a lazy `between` without materializing the
  range. Unsorted input throws.

## Provider specs

A spec is an immutable value describing *where* records of one kind
come from: what config builds, identity hashes, persistence writes. It
holds no resources and answers `kind(spec)`. Specs are per storage, not
per kind, and the selector is a query argument, so one spec serves
every series in its storage.

- `InMemory{R}(rows)` — fixtures; rows kept stably sorted by timestamp.
- `Constant{R}(record)` — one record visible from the start of time,
  the flat-curve case. `asof` returns it only for its own selector (a
  constant for SPY says nothing about SPX); `between` and `timestamps`
  never contain it over a real window.

Specs that need nothing at run time are their own readers (see
Lifecycle, once it lands).

## Derived providers

A derived provider serves a kind by reading other kinds **through the
map it is called from**. It holds only its own parameters, never its
inputs, and declares them with `inputs(spec)` so the loader can check
they are present. `QuotesFromBars(synthesizer)` is the first: it serves
`OptionQuote` from the map's `OptionBar` entry, which moves OHLCV-to-
quote synthesis out of the storage reader. The parquet reader becomes
vendor-only code; a future feed that has quotes serves `OptionQuote`
directly and `QuotesFromBars` is simply not configured. Policies depend
on `OptionQuote`; `OptionBar` is addressable but vendor-level.

## The map

`MarketData` is an immutable tuple of providers, one per kind, looked
up by type (`entry(m, R)`); lookup over a concrete tuple folds at
compile time. Every map-level shape passes the map itself as the
context to the provider. Consequences:

- **One reader per entry.** Every read of `OptionBar`, whether from the
  quote provider, the surface provider or the engine, reaches the one
  `OptionBar` entry, so one connection and one cache serve them all.
- **No ordering at open.** Derived providers resolve on each call.
- **What an experiment gets** is the set of kinds in its map. A kind not
  provided fails at `entry`, with the kind's name.
- One entry per kind is deliberate. Comparing two synthesizers or two
  surface conventions is two runs, which is what the run store is for.

Each distinct provider tuple type compiles once; with one config family
that is a few seconds. Accepted.

## Time cut

`TimeCut(m, cutoff)` masks every shape at the cutoff and passes
**itself** down as the context, so a derived provider's input reads go
through the cut. No-lookahead through derived data is therefore
structural: the provider was handed the cut and can see nothing else.
Because `timestamp` is visibility time, the cut is the complete
no-lookahead rule.

Derived caches stay cut-independent by one invariant: a derived
provider reads its inputs at or before the requested `ts`, so a cache
entry keyed on `(sel, ts)` is valid under any cutoff `>= ts`.

## Clock

`Clock{R}(sel)` names the grid the engine ticks on: the timestamps of
one kind for one selector, enumerated with `timestamps(m, clock, from,
to)`. The selector is checked against `selector_type(R)` at
construction and stored concretely typed. The clock is declared per
experiment and is part of core identity.

## Naming

`between` is the project's own generic function. `Base.between`
exists, unexported and `Integer`-only; it is never imported or
extended (a test pins its method count). `at`, `asof`, `timestamps`,
`kind`, `entry`, `selector` collide with nothing in Base or Dates.
Conventions consulted before fixing these names are recorded in
proposal section 10.
