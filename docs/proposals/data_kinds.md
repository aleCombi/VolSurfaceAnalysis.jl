# Proposal: kinds, providers, readers -- a scalable data layer (v2)

Status: proposal, revised after two reviews (Appendix A). Supersedes the
`DataSource` / `ModelDataSource` split in `docs/modules/data.md` and
`docs/modules/model_data.md` once accepted. Section 9 maps every review
finding to what changed.

## 1. Why

The current data layer hardcodes *which* data exists. `DataSource`
knows two things, chains and spots, through two verbs (`get_chain`,
`get_spot`). `ModelDataSource` hardcodes four slots on top (chain
source, spot source, rate curve, dividend curve) and one derived object
(`get_surface`). Every new data need -- splits, dividends, a rate-curve
history, inflation prints, a second underlying -- costs a struct field,
a protocol verb, a `TimeCutModelDataSource` forwarder, a config builder,
and a `to_dict` branch.

Three smaller problems ride along:

- `ParquetDataSource` fuses the *description* of a source (underlying,
  roots, synthesizer) with its *running machinery* (DuckDB connection,
  three LRU caches, a `closed` flag). The description is what config
  writes and identity hashes; the machinery is what a run needs. Fusing
  them makes `Experiment` hold a live database handle, forces `to_dict`
  to hand-exclude cache knobs, leaks `close` semantics to users, blocks
  sharing across threads, and is the only reason the struct is `mutable`.
- The protocol is point-query shaped and chains have no range read. Spots
  got a day-block cache; chains did not. The flagship policy hides this by
  ticking once a day; any dense policy pays a DuckDB query per minute.
- Quote synthesis lives inside the parquet reader, so "what the vendor
  has" (OHLCV bars) and "what we make of it" (bid/ask quotes) are
  entangled.

## 2. Concepts

### 2.1 Kind

A kind is a plain immutable record type with a `timestamp::DateTime`
field. It says *what* a datum is. Each kind has a **selector**: the
field that distinguishes parallel series of the same kind
(`Underlying` for market data, a currency for rate curves). Kinds
without parallel series have no selector.

```julia
struct OptionBar   ...; underlying::Underlying; timestamp::DateTime end   # what Polygon stores
struct OptionQuote ...; underlying::Underlying; timestamp::DateTime end   # bid/ask/mark per contract
struct SpotPrice   underlying::Underlying; price::Float64; timestamp::DateTime end
struct Split       underlying::Underlying; ratio::Float64; timestamp::DateTime end
struct Dividend    underlying::Underlying; amount::Float64; timestamp::DateTime end
struct RateCurve   currency::Currency; curve::Curve; timestamp::DateTime end   # snapshot as of t, evaluated at T
struct DivCurve    underlying::Underlying; curve::Curve; timestamp::DateTime end
struct VolSurface  underlying::Underlying; ...; timestamp::DateTime end        # derived, see 2.5
```

`RateCurve` and `DivCurve` are `(t, T)`-dependent: the record is the
curve *as of* `t`; the curve is a function of maturity `T`. This
replaces today's `get_rate(ts)::Float64`, which conflated the two axes.
Zero rate versus discount factor is a `Curve` concern.

Kinds are keyed **by type**. There is no name registry in the runtime
path; the config loader owns the one string-to-type table (section 4).

### 2.2 Protocol

Two shapes and a timestamp enumerator. `sel` is the kind's selector.

```julia
at(src, ::Type{R}, sel, ts::DateTime)                 -> Vector{R}
between(src, ::Type{R}, sel, from::DateTime, to::DateTime) -> iterable of R
timestamps(src, ::Type{R}, sel, from::DateTime, to::DateTime) -> Vector{DateTime}
```

Rules:

- Results are sorted by `timestamp` and contain only records with
  `timestamp == ts` (`at`) or `from <= timestamp <= to` (`between`).
- **Empty means absent.** `at` returns an empty vector; `between` an
  empty iterable. `missing` is only for absent scalar fields *inside*
  a record.
- **`between` promises an iterable, not a container.** Providers with
  small data return a vector; providers with large data return a lazy
  iterator (one day file in memory at a time). Consumers use Julia's
  iteration protocol: `collect`, `Iterators.filter`, `Iterators.map`.
  An iterator is valid only while its reader is open (section 3).
- Ranges are always bounded. There is no unbounded discovery verb.
- The default `at` is `collect(between(src, R, sel, ts, ts))`;
  providers override it when they have a faster path.
- The data layer defines no query language. Anything beyond these
  shapes is plain Julia over the result.

### 2.3 Library

Ordinary functions over the protocol. They are not part of it and
providers do not implement them.

```julia
asof(src, R, sel, ts)  = last_or_missing(between(src, R, sel, typemin(DateTime), ts))
only_or_missing(v)     = isempty(v) ? missing : only(v)          # errors on duplicates
last_or_missing(it)    = (x = missing; for r in it; x = r; end; x)
by_timestamp(it)       = ...   # lazy run-length grouping of a sorted iterable into (ts, Vector{R})
```

Which helper a consumer reaches for follows the kind's time semantics.
This is documentation, not a construct:

| Kind | Shape | Natural call |
|---|---|---|
| `OptionQuote`, `OptionBar` | grid, many per `ts` | `at` |
| `SpotPrice` | grid, one per `ts` | `only_or_missing(at(...))` |
| `RateCurve`, `DivCurve` | snapshot, holds until superseded | `asof` |
| `Split`, `Dividend` | event | `asof` for latest, `between` for a window |
| `VolSurface` | grid, one per `ts` | `only_or_missing(at(...))` |

`asof` with an unbounded lower bound is acceptable because it is only
ever used on snapshot and event kinds, which are thousands of rows, never
on chains. A DuckDB-backed provider may override the pattern with
`ORDER BY timestamp DESC LIMIT 1`; the protocol does not know.

Knowledge time versus effective time (revised macro data) is not
modelled. A kind that needs both carries both fields and its provider
decides which one `between` bounds on. Nothing on the current list
needs it.

### 2.4 Provider specs

A spec is an immutable value describing *where* records of one kind come
from. It is what config builds, identity hashes, and persistence writes.
It holds no resources. Every spec answers `kind(spec)`.

```julia
struct ParquetOptionBars;  root::String end           # OptionBar, every symbol= partition under root
struct ParquetSpots;       root::String end           # SpotPrice, same
struct CsvEvents{R};       path::String end           # R for every selector in the file
struct InMemory{R};        rows::Vector{R} end        # fixtures
struct Constant{R};        record::R end              # one record at typemin(DateTime)
struct ByUnderlying{P};    parts::Dict{Underlying,P} end   # composition: route selector -> sub-provider
```

Specs are **per storage, not per kind**: `CsvEvents{Split}` and
`CsvEvents{Dividend}` are the same code. The selector is a query
argument, so one parquet spec serves every underlying in its tree.
"SPY spots from parquet, SPX spots from csv" is `ByUnderlying`, a
dozen lines of forwarding, inside the one `SpotPrice` entry.

`Constant{RateCurve}` is one record timestamped at the start of time:
`asof` always finds it, `between` over any real window never contains
it, `timestamps` is empty. A constant is a model input, not an
observation, and the shapes say so without special cases.

### 2.5 Derived providers

A derived provider serves a kind by reading other kinds **through the
map it is called from** (2.6). It holds only its own parameters, never
its inputs.

```julia
struct QuotesFromBars{Q<:QuoteSynthesizer}; synthesizer::Q end                  # OptionQuote from OptionBar
struct SurfaceFrom; spot_for::Dict{Underlying,Underlying} end                    # VolSurface; optional spot remap
```

This is where OHLCV-to-quote synthesis moves. The parquet reader becomes
vendor-only code; a future live feed serves `OptionQuote` directly and
`QuotesFromBars` is simply not configured. Policies only ever ask for
`OptionQuote`; `OptionBar` is addressable (an experiment studying the
synthesizer wants it) but documented as vendor-level.

### 2.6 The map

`MarketData` is an immutable tuple of providers, one per kind, looked
up by type. Type lookup over a concrete tuple folds at compile time.

```julia
struct MarketData{P<:Tuple}; entries::P end
entry(m::MarketData, ::Type{R}) where R = _entry(R, m.entries...)
_entry(::Type{R}, p, rest...) where R = kind(p) === R ? p : _entry(R, rest...)
_entry(::Type{R}) where R = error("no provider for $R")

at(m::MarketData, ::Type{R}, sel, ts) where R = at(entry(m, R), m, R, sel, ts)
```

The verb on the map passes **the map itself** as a context argument to
the provider. Raw providers ignore it. Derived providers read their
inputs through it. `ByUnderlying` forwards it.

```julia
at(r::ParquetBarsReader, ::Any, ::Type{OptionBar}, u, ts) = _load_chain_at(r, u, ts)
at(p::QuotesFromBars, m, ::Type{OptionQuote}, u, ts) =
    map(b -> synthesize(p.synthesizer, b), at(m, OptionBar, u, ts))
function at(r::SurfaceReader, m, ::Type{VolSurface}, u, ts)
    get!(r.cache, (u, ts)) do
        su    = get(r.spec.spot_for, u, u)
        chain = at(m, OptionQuote, u, ts)
        spot  = only_or_missing(at(m, SpotPrice, su, ts))
        rate  = asof(m, RateCurve, USD, ts)
        div   = asof(m, DivCurve,  u, ts)
        any_absent(chain, spot, rate, div) && return VolSurface[]
        [build_surface(chain, spot.price, rate.curve, div.curve)]
    end
end
```

Consequences:

- **One reader per storage.** Every read of `OptionBar` reaches the one
  `OptionBar` entry, so the surface provider, the quote provider, and
  the engine's `resolve_quote` share one connection and one chain cache.
- **No ordering at `open`.** Derived providers resolve on each call, so
  `open` is `map(open, entries)`. The loader walks the kind graph once
  for missing inputs.
- "What an experiment gets" is the set of kinds in its map, declared per
  experiment in config. Asking for a kind not provided fails at `entry`.

### 2.7 Time cut

```julia
struct TimeCut{M}; inner::M; cutoff::DateTime end
at(c::TimeCut, ::Type{R}, sel, ts) where R =
    ts <= c.cutoff ? at(entry(c.inner, R), c, R, sel, ts) : R[]
between(c::TimeCut, ::Type{R}, sel, from, to) where R =
    from <= c.cutoff ? between(entry(c.inner, R), c, R, sel, from, min(to, c.cutoff)) : R[]
timestamps(c::TimeCut, ::Type{R}, sel, from, to) where R =
    from <= c.cutoff ? timestamps(entry(c.inner, R), c, R, sel, from, min(to, c.cutoff)) : DateTime[]
```

The cut passes **itself** down as the context, so a derived provider's
input reads go through the cut. No-lookahead through derived data is
structural, not a convention. `asof` past the cutoff clamps to the
snapshot as of the cutoff for free, because it is `between` underneath.
The `from > cutoff` case returns empty immediately.

## 3. Lifecycle

A reader is the opened form of a spec: it owns what the storage needs at
run time (a DuckDB connection, bounded LRU caches, a socket for a future
feed). `open(spec)` returns it; `close(reader)` releases it. Specs that
need nothing are their own reader. Neither is `mutable`.

```julia
struct ParquetBarsReader; spec::ParquetOptionBars; con::DuckDB.DB; days::LRU; chains::LRU end
open(s::ParquetOptionBars) = ParquetBarsReader(s, DuckDB.DB(":memory:"), LRU(200), LRU(10))
close(r::ParquetBarsReader) = DBInterface.close!(r.con)
open(s::SurfaceFrom) = SurfaceReader(s, LRU(64))
open(s) = s;  close(x) = nothing
```

The run opens and closes; `Experiment` holds the spec map only.

```julia
function run_experiment(exp)
    data = open(exp.data)
    try
        positions = run_backtest(exp.agent, data, exp.from, exp.to)   # TimeCut(data, t) per tick
        ...
    finally
        close(data)
    end
end
```

Parallel sweeps get one reader set per task from one shared spec set,
which resolves the concurrent-caching item in the data doc's future work
without locks. `with_data(f, market_data)` is the REPL convenience.

The range read on the parquet bars reader is the sequential day pass:

```julia
between(r::ParquetBarsReader, ::Any, ::Type{OptionBar}, u, from, to) =
    Iterators.flatten(_day_bars(r, u, d, from, to) for d in Date(from):Day(1):Date(to))
```

`run_experiment`'s window-end lookup walks `timestamps` backward by day
from `to` and stops at the first non-empty day, replacing the full-window
scan.

## 4. Config

One table per kind. The loader owns the only string-to-kind table and
the provider builder registry, next to the existing synthesizer and
curve builders. Nested providers (`ByUnderlying`) are nested tables.

```toml
[data.option_bar]    type = "parquet_option_bars"  root = "C:/repos/options-collector/data/massive"
[data.option_quote]  type = "from_bars"            synthesizer = { type = "ohlcv_spread", lambda = 0.7 }
[data.rate_curve]    type = "constant"             currency = "USD"    value = 0.045
[data.div_curve]     type = "constant"             underlying = "SPY"  value = 0.013
[data.vol_surface]   type = "surface_from"         spot_for = { SPY = "SPX" }

[data.spot_price]
type = "by_underlying"
SPY  = { type = "parquet_spots", root = "C:/repos/options-collector/data/massive" }
SPX  = { type = "csv_spots",     path = "C:/data/spx.csv" }
```

Load-time checks: each table's `type` builds a spec whose `kind` matches
the table name; no two tables share a kind; every derived provider's
input kinds are present. Cache sizes are `open` kwargs, never config,
never identity.

Policies name kinds and selectors, never entries:

```julia
chain = at(cut, OptionQuote, p.underlying, t)
spot  = only_or_missing(at(cut, SpotPrice, p.underlying, t))
surf  = only_or_missing(at(cut, VolSurface, p.underlying, t))
divs  = between(cut, Dividend, p.underlying, t, t + Day(30))
```

## 5. Identity

`to_dict(::MarketData)` emits one entry per kind, keyed by the loader's
kind name, sorted, each the `to_dict` of its spec. With one entry per
kind and derived providers holding no inputs, there is no duplicated or
shared sub-spec to canonicalize. Connections, caches and readers never
appear because they are not on specs.

**Every existing run id changes.** The projection's `[source]` shape
becomes `[data.*]`. Decision: break once, now, while the store holds at
most one run. No migration script. The manifest gains a `schema_version`
field, outside the hash, so `load_run` refuses an old-format run with a
clear message instead of an obscure loader error, and so a later
identity change (dataset version rather than local path, reviewer B10)
has a hook.

## 6. Rule changes surfaced (design rule 3)

1. **Absence convention.** `docs/modules/data.md`: `nothing` for a
   missing aggregate, `missing` for a missing scalar. New rule: empty
   result for no records; `missing` only inside records.
2. **`OptionBar` status.** Adapter-layer today. New rule: a first-class
   kind, documented as vendor-level; policies depend on `OptionQuote`.
3. **`model_data` module.** Dissolves. `Curve` types stay as values
   inside `RateCurve` / `DivCurve` records; surface construction is a
   derived provider that lives with `surfaces`.
4. **Rate/div time-cut passthrough.** Removed with the `(t, T)` records.
5. **Unbounded discovery.** Unchanged in substance, enforced by the verb
   shape rather than a throwing method.
6. **Run identity.** One-time break; `schema_version` in the manifest.

## 7. Migration map

| Today | Proposal |
|---|---|
| `DataSource`, `get_chain`, `get_spot`, `get_spots`, `available_timestamps` | `at` / `between` / `timestamps` on providers |
| `InMemoryDataSource` | `InMemory{R}` |
| `ParquetDataSource{S}` | `ParquetOptionBars` + `ParquetSpots` specs and readers; `QuotesFromBars{S}` |
| `OptionBar` as adapter type | `OptionBar` as kind |
| `ModelDataSource` | `MarketData` |
| `get_rate(ts)`, `get_div(ts)` | `asof(m, RateCurve, ccy, t)`, `asof(m, DivCurve, u, t)` |
| `get_surface` + unbounded `surface_cache` | `SurfaceFrom` with a bounded reader cache |
| `TimeCutModelDataSource` | `TimeCut{M}` |
| `clear_cache!`, `with_parquet_source` | removed; `with_data` |
| `run_experiment` full-window timestamp scan | backward day walk |
| `resolve_quote` linear scan | unchanged; keyed chain is a later change |

## 8. Execution plan

Each step is one or more commits that leave the suite green and update
the affected module docs **in the same commit** (design rule 1). Steps
1 and 2 are exactly the increments both reviews recommended first.

0. **Convention check (design rule 5).** Look at how `Tables.jl`,
   `TimeSeries.jl`, `DataInterpolations.jl` and `DBInterface.jl` shape
   type-marker dispatch, `open`/`close` on non-IO handles, and lazy
   partitioned iteration. Record findings in section 10 before code.
1. **Kinds, protocol, library, map, cut.** `at` / `between` /
   `timestamps`, `asof` and friends, `MarketData`, `TimeCut`,
   `InMemory`, `Constant`, `ByUnderlying`. Fixture tests. New
   `docs/modules/market_data.md`. Nothing existing changes.
2. **Parquet split.** `ParquetOptionBars` / `ParquetSpots` specs and
   readers, both arities, ported from `parquet_source.jl`; fixture tests
   moved over; `at` versus `collect(between)` equality test.
   Wall-clock: point versus range over one month of minute data.
3. **Derived providers.** `QuotesFromBars`, `SurfaceFrom` with bounded
   cache, `RateCurve` / `DivCurve` kinds. Test that a derived read
   through a `TimeCut` cannot see past the cutoff.
4. **Engine and policies.** `run_backtest`, `DailyShortStrangle`,
   `resolve_quote`, `_build_settle`, window-end walk.
5. **Config, identity, persistence.** `[data.*]` loader and kind table,
   `to_dict`, `schema_version`, `load_run` refusal of old runs.
   Existing configs rewritten.
6. **Delete the old layer.** `DataSource`, `ModelDataSource`,
   `TimeCutModelDataSource`, `clear_cache!`, `with_parquet_source`,
   `docs/modules/model_data.md`. Final `data.md` and `status.md`.

Gate for steps 2 through 5: rerun `configs/strangle_spy_16d_1dte.toml`
against the real parquet store and diff `metrics` and `pnl_series`
against the run saved before step 1.

## 9. Response to the v1 reviews

| Finding | v2 |
|---|---|
| A1, B4: duplicate opens, ownership graph | Derived providers hold no inputs; they read through the map. One entry per kind, `open` is a map, no graph. |
| A2, B2, B9: cardinality pushed to consumers | `at` for grid kinds, `asof` for snapshots, `only_or_missing` for grid singletons; all library, per-kind table in 2.3. |
| A3, B8: `Flat` semantics | `Constant{R}`: one record at the start of time; `asof` finds it, `between` never does, `timestamps` empty. |
| A4: no underlying dimension | Selector is a verb argument; one provider per kind serves every selector; `ByUnderlying` composes sources. |
| A5, B1: range materialization | `between` returns an iterable; parquet reader yields one day at a time via `Iterators.flatten`. |
| A6, B10: run-id break, canonical identity | Break once, `schema_version` in manifest; no shared sub-specs left to canonicalize. Dataset-version identity is separate work. |
| A7: type stability | Type-keyed tuple lookup folds at compile time; no symbols in the runtime path. |
| A8, B12: rule 5 deferred | Step 0, recorded before code. |
| B3: as-of semantics | `asof` in the library over `between`; clamps under the cut for free. Knowledge vs effective time left out, stated. |
| B5: derived no-lookahead only a convention | The cut passes itself as the context; derived reads cannot escape it. `from > cutoff` returns empty. |
| B6, B7: raw/model boundary, bars leaking | Boundary is derived-over-raw, not a separate struct; `OptionBar` documented vendor-level, policies use `OptionQuote`. |
| B11: docs per commit | Plan rewritten; every step updates module docs in the same commit. |
| "Park the generic layer" | Not taken. The holes are closed above; steps 1 and 2 are the reviewers' own increments. |

## 10. Convention check findings

To be filled in at step 0.

## Appendix A. Reviews of v1

Two independent reviews of this proposal against `master` at `d08b76e`,
reaching the same verdict: adopt the spec/reader split and the synthesis
move now as independent commits; park the generic kinds/records layer
until a second data kind actually lands. Kept here so the decision on
section 7 is traceable.

### Review A (Claude, 2026-09-06)

Reviewed against `master` at `d08b76e`. Verdict: **adopt with substantial
changes, not as-is.** Two of the three motivating debts are real and worth
fixing now, independently. The universal kinds/records layer built on top
of them is premature and has unresolved semantic holes.

#### Motivating claims checked against the code

- **Spec/reader fusion: real.** `ParquetDataSource`
  (`src/data/parquet_source.jl`) is `mutable`, holds a `DuckDB.DB`, a
  finalizer, a `closed` flag and three caches. `Experiment` therefore
  carries a live database handle, and `to_dict(::ParquetDataSource)` in
  `src/experiment/identity.jl` hand-excludes the cache knobs. This is the
  strongest part of the proposal.
- **Five files per kind: accurate.** Adding e.g. dividends today touches
  `src/data/source.jl` (verb), `src/model_data/source.jl` (field +
  accessor), `src/backtest/time_cut.jl` (forwarder),
  `src/experiment/config.jl` (builder) and `src/experiment/identity.jl`
  (projection). But this cost has been paid zero times so far; nothing in
  `docs/status.md` in-flight or backlog asks for a new kind.
- **Point-query cost: weak for the current workload.** `get_chain` is
  indeed one DuckDB query per timestamp, but `DailyShortStrangle` supplies
  one tick per day through `tick_times`, so per-minute chain queries never
  happen. The one present hot spot is `run_experiment`
  (`src/experiment/experiment.jl`) calling `available_timestamps` over the
  full ten-year window only to find the last timestamp: one `DISTINCT`
  query per day, ~2500 queries. That is a ten-line fix and needs no
  redesign.

#### Flaws

1. **Provider graph opens readers twice.** `SurfaceFrom` cannot hold the
   `MarketData` it lives in (an immutable `NamedTuple` cannot contain
   itself), so it must hold copies of its input specs. `open` then creates
   two DuckDB connections and two chain caches for the same bars: one under
   `option_quote`, one nested under `vol_surface`. The strangle policy calls
   `get_surface`, `get_chain` and the engine calls `resolve_quote` at the
   same tick; today all three share one chain cache. Section 4 calls the
   duplication "harmless" for identity; it is a resource regression at run
   time. Fix: memoize `open` by spec identity, or have derived providers
   reference keys and resolve against the map at open time.
2. **Cardinality is pushed onto every consumer.** `SpotPrice`, `RateCurve`,
   `DivCurve` and `VolSurface` are singletons at a timestamp but come back
   as `Vector{R}`, so every call site does `isempty` then `only`. Today's
   `get_chain` vs `get_spot` makes the cardinality difference explicit.
3. **`Flat{RateCurve}` has no honest `timestamps` semantics.** A constant
   curve fabricates a record at any queried instant. A bounded range of it
   is either one record or infinitely many; the proposal does not say. The
   existing `FlatCurve` is a timeless model input and is simpler and more
   honest.
4. **No underlying dimension.** The map is keyed by kind only, yet "a
   second underlying" is listed in section 1 as a motivation. The design
   does not support its own use case; it needs either `(kind, underlying)`
   keys or an underlying argument on the verb.
5. **Range results materialize whole intervals.** `records(src, OptionBar,
   from, to)::Vector` over a month of minute-level SPY chains conflicts with
   the bounded-memory rule in `docs/modules/data.md` and with the 3.7 GB
   dev box. The sequential day pass should be an iterator of per-day (or
   per-timestamp) blocks, not one vector.
6. **Every stored run id changes.** The identity projection moves from a
   `[source]` table to `[data.*]` keys, so every `run_id` in the `RunStore`
   knowledge base stops matching its rehydrated experiment. Section 7 says
   "existing configs rewritten" and stops. `docs/vision.md` makes the
   accumulating KB a first-class goal; this needs a migration story (or a
   documented id break) before step 5.
7. **Type-stability claim is asserted, not shown.** `getproperty(nt,
   key(R))` is only type-stable if `key(R)` constant-folds. Doable with
   `Val` or a generated function, but the nested parametric types
   (`TimeCut{MarketData{NamedTuple{..., QuotesFromBars{...}}}}`) also mean
   every distinct config is a new type and a recompile, which matters on a
   2-core box with Revise.
8. **Design rule 5 deferred rather than met.** The public API shape
   (`open`/`close` on non-IO types, `key(::Type)` trait, single verb with a
   type marker) is chosen before the convention check the rule requires.

#### Recommended path

Take the valuable pieces now as independent small commits; park the
generic layer until a second kind actually arrives.

1. Split `ParquetDataSource` into an immutable spec and a run-scoped
   reader; `run_experiment` opens and closes in `try/finally`. Keep the
   downstream `get_*` API unchanged. This alone removes the `mutable`
   struct, the finalizer, the `closed` flag and the identity hack, and
   unlocks per-task readers for parallel sweeps.
2. Fix the window-end scan in `run_experiment`: walk days backward from
   `to` and stop at the first non-empty day.
3. Bound the surface cache in `ModelDataSource`.
4. Move OHLCV-to-quote synthesis into an adapter between reader and
   consumer, keeping `OptionQuote` as the only canonical downstream output.
5. Revisit the kinds/records verb when the first real second kind lands.
   Before that design is accepted it must answer the shared-reader,
   underlying-key, cardinality, as-of and run-id questions above, and cite
   the rule-5 convention check.

### Review B (Codex, 2026-09-06)

#### Verdict

Recommend **adopting with substantial changes**, not as-is.

The proposal correctly identifies architectural debt in `src/data/parquet_source.jl` and a likely future extensibility problem in `src/data/source.jl`. Its best ideas are worth implementing independently:

- Separate immutable, serializable source specifications from opened runtime readers.
- Add efficient bounded-range reads.
- Move quote synthesis out of the parquet reader.
- Represent historical rate/dividend curves with separate observation time and maturity time.

The proposed universal kinds/providers/readers system, however, introduces unresolved cardinality, ownership, time-semantics, and memory problems. It should not replace `DataSource` and `ModelDataSource` wholesale in its current form.

#### Does it solve a real problem?

Partly.

The spec/reader problem is real. `ParquetDataSource` currently contains identity-bearing configuration, cache policy, mutable caches, a DuckDB connection, and lifecycle state in one object. That forces `src/experiment/identity.jl` to manually omit operational fields and makes a persisted `Experiment` carry something conceptually live. Splitting:

```julia
ParquetSpec → open → ParquetReader
```

would materially improve serialization, lifecycle management, parallel-run isolation, and testing.

The hardcoded-kind problem is plausible but not yet severe. The current protocol has only chains and spots, while `src/model_data/source.jl` adds rates, dividends, and surfaces. Adding splits, historical curves, or multiple underlyings would indeed spread changes through the time cut, config, identity, and model composition. A typed generic access mechanism could reduce that repetition.

The point-query performance claim is weaker for the current flagship workflow. `ParquetDataSource` really does execute a DuckDB chain query per requested timestamp, but `DailyShortStrangle` already supplies one candidate tick per day in `src/policies/daily_short_strangle.jl`. It does not walk every market minute. Meanwhile spots already have efficient day-block reads. Range-loading chains becomes important for dense policies and historical feature construction, but the proposal correctly calls for a benchmark because the current repo has not yet demonstrated that bottleneck.

#### Strengths

1. **Spec/reader separation is the clearest improvement.**

   Immutable specs fit naturally into `Experiment`, config, hashing, and persistence. Per-run readers fit the lifecycle of `src/experiment/experiment.jl`. This eliminates the brittle special projection currently needed in `to_dict(::ParquetDataSource)`.

2. **The `(observation time, maturity)` distinction for curves is correct.**

   Current `get_rate(ts)` conflates “the curve known at time `t`” with “the curve value for maturity `T`.” A record containing a curve observed at `t`, later evaluated at `T`, is a materially better foundation for historical rates and dividends. It also makes the no-lookahead interpretation clearer than the current passthrough exception in `src/backtest/time_cut.jl`.

3. **Separating vendor bars from quote synthesis is sound.**

   `src/data/parquet_source.jl` currently reads Polygon bars and synthesizes `OptionQuote`s in the same row loop. A `QuotesFromBars` transformation would let a live quote feed serve canonical quotes directly while keeping the OHLCV assumption explicit in experiment identity.

4. **Bounded point and range operations are useful.**

   Retaining bounded discovery preserves the important anti-accidental-scan rule documented in `docs/modules/data.md`. Requiring point and bounded-range access also creates a proper place for storage-specific batching.

5. **The proposal surfaces rule changes explicitly.**

   Its section 6 handles design rule 3 from `docs/design.md` well. Changes to absence semantics, `OptionBar`, curve handling, and module boundaries are stated rather than silently introduced.

6. **The migration includes meaningful validation gates.**

   Comparing saved metrics and `PnLSeries`, plus benchmarking range versus point reads, is the right general shape.

#### Flaws and risks

##### 1. The interval return type does not scale to option chains

The proposal promises:

```julia
records(source, OptionBar, from, to)::Vector{OptionBar}
```

A month of minute-level SPY option rows can be enormous. Materializing the entire interval into one vector conflicts with the bounded-memory motivation and with the current bounded-cache policy in `docs/modules/data.md`.

It also loses the natural chain grouping by timestamp. Most consumers want a sequence of timestamped chain blocks, not one flat vector that they must regroup.

The range API should expose a lazy or chunked shape, such as an iterator of `RecordBatch{R}`, daily partitions, or `(timestamp, records)` groups. Point lookup can still return a materialized vector.

##### 2. Cardinality is underspecified

`OptionQuote` is many-valued at a timestamp; `SpotPrice`, `RateCurve`, and `DivCurve` are expected to be exactly one. Returning `Vector{R}` for everything merely moves cardinality enforcement into every consumer:

```julia
only(records(data, SpotPrice, ts))
```

The proposal uses `only` inside `SurfaceFrom`, but does not define what duplicates mean, how they are validated, or what error users receive.

The design needs declared cardinality or explicit helpers such as:

```julia
records_at(...)
record_at(...)       # exactly zero or one
required_record_at(...)
record_asof(...)
```

Without that, one generic verb creates uniform syntax but not a reliable protocol.

##### 3. Exact timestamp lookup is wrong for several proposed kinds

Splits, dividends, inflation releases, and curve snapshots have more complex time semantics:

- announcement time;
- publication or ingestion time;
- effective date;
- ex-dividend date;
- curve observation time;
- maturity time.

An exact `records(..., ts)` operation will often return nothing even though the consumer needs the latest record known as of `ts`. This is especially important for avoiding lookahead around revised macroeconomic data.

The proposal needs an explicit temporal model before claiming these kinds fit the same protocol. At minimum it needs `asof` semantics and a distinction between knowledge time and effective time.

##### 4. Provider ownership forms an ambiguous graph

The config includes `option_bar` as a top-level provider and embeds/references it again beneath `option_quote`. `SurfaceFrom` then contains another `MarketData`. As written, recursively calling `open` can:

- open the same parquet specification more than once;
- create duplicate DuckDB connections and caches;
- synthesize duplicate reader subtrees;
- close a shared dependency more than once if sharing is later introduced.

The proposal calls this an immutable tree, but the conceptual structure is a dependency graph. It needs one owner responsible for opening each distinct provider once, dependency resolution, topological close order, cycle detection, and reuse.

This also undermines the claim that specs are “per storage, not per kind”: `ParquetOptionBars` and `ParquetSpots` are separate specs for two kinds even when they come from the same dataset root. A single dataset/storage spec with multiple typed capabilities may be the better boundary.

##### 5. Derived-provider no-lookahead is only a convention

The invariant that derived providers “may not widen” a query is useful, but unenforced. A derived provider directly holding unrestricted inputs can simply request a later timestamp. Wrapping only the outer `MarketData` in `TimeCut` does not constrain those internal calls.

The current system’s protection is also limited to its supported interface, but `src/backtest/time_cut.jl` makes every supported accessor enforce the cut. In the proposed design, derived providers should receive cut-aware dependency handles, or queries should carry an explicit bounded context that cannot be widened.

There is also a concrete edge case:

```julia
records(c.inner, R, from, min(to, c.cutoff))
```

When `from > cutoff`, this forwards an invalid range instead of immediately returning `R[]`.

##### 6. It weakens the raw-data/model-object boundary

`docs/modules/model_data.md` deliberately says raw sources know no math, model objects know no I/O, and builders are where they meet. Putting `OptionBar`, `OptionQuote`, `RateCurve`, `DivCurve`, and `VolSurface` into one undifferentiated provider map erases that distinction.

A volatility surface is not merely another stored record. It is a derived model object whose result depends on quote convention, spot alignment, rate/dividend curves, pricing assumptions, and failure policy.

The generic machinery could be reused on both sides, but the architectural distinction should remain—perhaps as raw `MarketData` plus a model-facing derived view. Dissolving `model_data` is unnecessary to obtain the proposal’s useful extensibility.

##### 7. Making `OptionBar` first-class leaks vendor shape downstream

The current boundary in `docs/modules/data.md` promises canonical records to downstream users and treats `OptionBar` as an adapter representation. Letting experiments request bars directly makes vendor storage details part of the public experiment vocabulary.

Bars may deserve a typed internal provider, but ordinary policies and model builders should generally depend on canonical `OptionQuote`s. Otherwise consumers can accidentally couple themselves to Polygon-specific OHLCV availability.

##### 8. `Flat{R}` has unclear record and timestamp semantics

A constant curve is naturally a timeless model input. Turning it into a provider of timestamped `RateCurve` records raises unanswered questions:

- Does a point request fabricate a record whose timestamp equals the query?
- What does a range request return—one record, one per tick, or infinitely many conceptual observations?
- Which timestamp enters experiment identity?
- Is a flat rate known for the whole run or observed separately at every instant?

The current `FlatCurve` in `src/model_data/curves.jl` is simpler and more honest. Historical curve providers and timeless configured curves should not be forced into identical observational semantics.

##### 9. Absence unification is cosmetically neat but loses useful meaning

Today:

- `nothing` means no aggregate chain/surface;
- `missing` means an absent scalar or field.

That convention is documented in `docs/modules/data.md` and used consistently by `src/model_data/source.jl` and the engine.

An empty vector is natural for zero-to-many records, but less expressive for required singleton observations. Combined with the missing cardinality rules, it can turn corrupted duplicate data and genuinely absent data into ad hoc consumer checks. Absence should follow operation cardinality rather than forcing every operation into a vector.

##### 10. Identity is not sufficiently canonical

Inlining nested specs may be deterministic, but it does not establish whether two logically identical provider graphs have identical identity. For example, one config could share a bar provider by reference while another spells the same spec twice. The runtime result may be identical while the structural projection differs—or two entries may drift despite supposedly referring to the same source.

Identity needs canonical provider IDs/references and clear rules about:

- repeated specifications;
- local paths versus logical dataset identity;
- transformations and their parameters;
- data version/snapshot identity;
- ordering;
- historical-data revisions.

The current implementation already hashes resolved local paths but not the actual parquet dataset version. A redesign is a good opportunity to address that larger reproducibility gap.

##### 11. The execution plan conflicts with design rule 1

The plan says steps 1–5 introduce the new public API, migrate readers, change engine/policy signatures, and rewrite config/identity, while step 6 updates the module docs. That violates rule 1 in `docs/design.md`, which requires module documentation to remain coherent with code on every commit.

Each migration step must update affected module docs in the same commit, even while both systems coexist.

##### 12. Design rule 5 has not yet been satisfied

The proposal acknowledges that Julia ecosystem conventions still need research, but it has already selected a public API shape involving `open`, `close`, trait-like `key(::Type)`, heterogeneous `NamedTuple` dispatch, and record-table semantics.

That research should be completed before accepting the API, not merely before implementation. In particular, the proposal should determine whether interoperability with `Tables.jl` tables/partitions or iterator conventions would solve the range-materialization problem more idiomatically.

#### What is over-engineered?

For the repo’s current one-provider, one-policy state, these pieces are premature:

- A universal provider for arbitrary CSV event kinds.
- Separate `RateCurve` and `DivCurve` record wrappers before there is a historical curve source.
- A configurable provider entry for every intermediate representation.
- Exposing both raw bars and synthesized quotes as independently addressable experiment inputs.
- Dissolving the entire `model_data` layer.
- A heterogeneous generic map before multi-asset lookup and key parameterization are designed.

The existing system’s five-file-change argument is somewhat overstated. Some repetition is real, but explicit methods can be valuable when kinds have genuinely different cardinality and time semantics. Eliminating method names is not automatically scalability.

#### What is missing?

Before implementation, the design needs:

- Cardinality contracts for zero/one/many records.
- Exact versus as-of lookup semantics.
- Knowledge time versus effective time for event data.
- Underlying/instrument keys for multi-asset queries; a kind alone is insufficient.
- Lazy/chunked range results for large option datasets.
- Provider dependency-graph construction, sharing, cycle detection, and close ordering.
- Capability validation when constructing `MarketData`, rather than a late `getproperty` failure.
- Explicit error behavior for missing, duplicate, malformed, and misaligned inputs.
- Data-version identity, not just provider configuration identity.
- Tests proving time-cut safety through derived providers.
- A clear boundary between raw observations and derived mathematical objects.
- A lifecycle policy for use-after-close and partial failure during `open`.
- Benchmarks demonstrating that the proposed range reader improves an actual workload.

#### Recommended path

Adopt the valuable changes incrementally:

1. Extract an immutable `ParquetDataSpec` from a run-scoped `ParquetDataReader`, without changing downstream APIs.
2. Make `run_experiment` open and close readers in `try/finally`.
3. Move OHLCV-to-quote synthesis into a separate adapter while keeping `OptionQuote` as the canonical public output.
4. Add a chunked/grouped bounded-range chain API and benchmark it against point queries on a dense workload.
5. Introduce historical `(t, T)` curve observations only when the first real historical curve source is added.
6. Prototype a typed `MarketData` capability map once a third genuinely distinct data kind or multi-asset policy requires it.
7. Preserve a model-facing layer for surface construction even if it internally uses the same provider machinery.
8. Update module docs alongside every migration commit, as required by `docs/design.md`.

That path captures the real benefits without committing the repository to an overly uniform abstraction before its semantics are established.

