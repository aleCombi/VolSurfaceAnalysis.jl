# `market_data` module

How market data is *obtained*: the protocol every consumer reads
through, the specs that describe where records come from, and (as the
layer lands) the readers, map, time cut and lifecycle behind it. What a
datum *is* stays in [`data`](data.md): record types and vendor row
mapping. Design and rationale are in
[proposals/data_kinds.md](../proposals/data_kinds.md); this doc states
the rules the code keeps.

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

## Curve kinds

`RateCurve(currency, curve[, timestamp])` and `DivCurve(underlying,
curve[, timestamp])` are `(t, T)`-dependent records: the `Curve` as of
the visibility time `timestamp`, evaluated at a maturity by calling it.
The `Curve` types themselves (`FlatCurve`, `PCCurve`) live in
`market_data/curves.jl` and stay pure math. The two-argument
constructors stamp the start of time, which under the visibility rule
reads "always known": the `Constant` case, today's flat rate and
dividend yield. A stamped curve is visible from its stamp and not
before — in every shape, `asof` included — so an experiment can say
"this curve became known in June" and a January query sees nothing. A
curve history is the same kind with real timestamps.
Snapshot kinds are read with `only_or_missing(asof(...))`, and the
earlier rate/div time-cut passthrough is gone: a curve snapshot is
visible or it is not.

| kind | shape | natural call |
|---|---|---|
| `OptionQuote`, `OptionBar` | grid, many per instant | `at` |
| `SpotPrice`, `VolatilitySurface` | grid, one per instant | `only_or_missing(at(...))` |
| `RateCurve`, `DivCurve` | snapshot, holds until superseded | `only_or_missing(asof(...))` |
| event kinds (future) | bounded lookback | `between`, then filter on the effective field |

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
- `Constant{R}(record)` — one record visible **from its own
  timestamp**, the flat-curve case. `asof` returns it only for its own
  selector (a constant for SPY says nothing about SPX) and only at or
  after its stamp, so the visibility rule holds for this provider as it
  does for every other; `between` and `timestamps` never contain it
  over a real window.

Specs that need nothing at run time are their own readers (see
Lifecycle).

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

The second derived provider, `SurfaceFrom`, lives with the
[`surfaces`](surfaces.md) module it builds for.

## Composition: `BySelector`

`BySelector{R}(sel => provider, ...)` is "SPY spots from parquet, SPX
spots from csv" inside the one `SpotPrice` entry. Invariants, checked
at construction: at least one part, every selector a `selector_type(R)`,
every part of kind `R`, no duplicate selector. Every shape routes on the
selector and forwards the context untouched, so a cut or a derived
provider above it sees no difference; an unknown selector throws
`KeyError`. Routing on a runtime selector yields a small union of part
types whose shapes all return the same record type, which keeps call
sites inferable (union-split routing; measured in proposal section 10).

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

## Lifecycle

A reader is the opened form of a spec: it owns what the storage needs
at run time (a connection, bounded caches, a partition list). Specs that
need nothing are their own reader. `Experiment` holds the spec map; the
run opens and closes it.

- **Project-owned pair, no `Any` fallback.** `open_data(spec)` and
  `close_data!(reader)` are the project's own generics with explicit
  one-line opt-ins for resource-free specs. A spec without `open_data`
  fails `has_lifecycle`, which the config loader checks; nothing is
  silently a no-op. No method is added to `Base.open` / `Base.close`.
- **Unwind on open failure.** A composite (`MarketData`, `BySelector`)
  opens its parts in order through a recursive tuple open; a failure
  closes what was opened, quietly (a close error during the unwind is a
  warning), so the original error is the one that propagates. The
  recursive form is also type-stable, so `open_data(m)` infers to the
  reader map's concrete type.
- **Best-effort close.** Every part is closed in reverse order even if
  one throws; the first error is rethrown after the loop.
- **Scoped form.** `with_data(f, m)` opens, calls `f`, closes. If `f`
  throws, the close is quiet and `f`'s error propagates; on success a
  close error propagates normally.
- **Use after close** throws `ArgumentError` from the reader. The
  proposal hoped to leave this to the storage, but DuckDB segfaults on
  a query against a closed handle, so the parquet readers carry a
  closed flag checked at every shape, including a lazy `between`
  iterator that outlives its reader. `close_data!` is idempotent.

## Parquet readers

`ParquetOptionBars(root)` serves `OptionBar`, `ParquetSpots(root)`
serves `SpotPrice`; `root` is the kind-specific tree (`.../options_1min`,
`.../spots_1min`) in the collector's Hive layout
`date=<D>/symbol=<T>/data.parquet`. One spec per storage tree, every
symbol under it. Construction is pure, so a saved run rehydrates
silently off-machine; `open_data` throws when the root is missing.

The opened reader owns one DuckDB connection, the per-selector
**partition list** (listed once, the bound for every walk), bounded
caches of per-partition metadata (distinct timestamps, column
presence), and for bars a small exact-instant chain cache plus the
shared contract-identity dict. Vendor rows become `OptionBar` only;
synthesis is `QuotesFromBars` above the reader.

- **`at`** reads one instant through the chain cache.
- **`between`** is a lazy walk over the candidate partitions, one
  partition's rows in memory at a time, never cached; valid while the
  reader is open.
- **`asof`** walks the partition list backward from `Date(ts)`,
  consulting cached timestamp lists until one has a timestamp `<= ts`,
  then reads that instant. Bounded by the partitions that exist and
  called once per run in practice.
- **`timestamps`** is the cached per-partition lists intersected with
  the range; a partition absent from the list costs no file probe.

*Partition convention.* A partition `D` may hold any timestamp in
`[D 00:00, D+1 02:00)` UTC: the collector writes a US session into its
local date, so after-midnight UTC rows spill past `Date(ts)`. Every
shape consults partitions `Date(ts) - 1` and `Date(ts)`, which is what
makes `at == collect(between(ts, ts))` an identity rather than a
coincidence. A ticker whose underlying is not the partition's throws:
under `symbol=` partitioning that is a corrupt store.

The convention is **time-ordered**: every row in partition `D - 1`
precedes every row in partition `D`. One contiguous session per
partition, the after-midnight spill belonging to the earlier session, is
what a local-date collector produces — and the four shapes agree with
each other only under it. `asof` returns at the newest candidate
partition holding a row at or before `ts` while `at` and `timestamps`
merge both candidates, so an interleaved layout would break
`asof == at(last(timestamps(...)))`; and the lazy bar `between`
concatenates `D - 1` then `D` without a cross-partition sort, so it would
yield out-of-order records. The ordering makes both correct by
construction, which is why it is a convention on the store rather than a
merge in four shapes.

*Spot de-duplication.* Consulting two partitions means the same row can
be read twice — the convention permits an after-midnight row in both
the earlier partition's spill and the later partition's body — and a
vendor can re-deliver a minute into one partition. Spots are a snapshot
kind read through `only_or_missing`, so either one would abort the read.
The spot reader therefore applies one rule after its sort: **equal
timestamp and equal price collapse silently; equal timestamp and
different price throws `ConflictingRecords`**, naming the instant and
both values. Taking the first would be a silent choice between two
answers, on a number nobody verified. `timestamps` on the same reader is
made distinct for the same reason.

Bars are left alone deliberately. A chain has many rows per timestamp by
design, so its de-duplication key is the contract, not the instant, and
what "conflicting" means over six fields is a separate question.

*Bar-time allowance.* Rows carry Polygon's bar-open stamp, kept as the
visibility time (see Kinds).

## Config and identity

Config builds specs, one `[data.<kind>]` table per kind plus a
`clock`; the string-to-kind table and the provider builders live in
the [`experiment`](experiment.md) loader, nothing on the runtime path
knows a name. Identity (`to_dict`) projects one entry per kind, the
clock, and per-spec fields that determine the records served; readers,
cache sizes and part order never enter the hash. The parquet specs'
root sits in a reserved `dataset` slot, the place a logical dataset id
and version would go.

## Naming

`between` is the project's own generic function. `Base.between`
exists, unexported and `Integer`-only; it is never imported or
extended (a test pins its method count). `at`, `asof`, `timestamps`,
`kind`, `entry`, `selector` collide with nothing in Base or Dates.
Conventions consulted before fixing these names are recorded in
proposal section 10.
