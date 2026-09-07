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

## Naming

`between` is the project's own generic function. `Base.between`
exists, unexported and `Integer`-only; it is never imported or
extended (a test pins its method count). `at`, `asof`, `timestamps`,
`kind`, `entry`, `selector` collide with nothing in Base or Dates.
Conventions consulted before fixing these names are recorded in
proposal section 10.
