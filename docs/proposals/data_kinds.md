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

## Appendix B. End-to-end sketch

The whole path in one place, config to policy. Reference for the
execution plan; sections 2 and 3 quote pieces of it.

```julia
# ================= Kinds: immutable records with a timestamp and a selector field.
struct OptionBar   instrument_id::String; underlying::Underlying; ...; timestamp::DateTime end
struct OptionQuote instrument_id::String; underlying::Underlying; ...; timestamp::DateTime end
struct SpotPrice   underlying::Underlying; price::Float64; timestamp::DateTime end
struct Split       underlying::Underlying; ratio::Float64; timestamp::DateTime end
struct Dividend    underlying::Underlying; amount::Float64; timestamp::DateTime end
struct RateCurve   currency::Currency; curve::Curve; timestamp::DateTime end   # snapshot as of t, evaluated at T
struct DivCurve    underlying::Underlying; curve::Curve; timestamp::DateTime end
struct VolSurface  underlying::Underlying; ...; timestamp::DateTime end        # derived

# ================= Protocol: two shapes plus timestamps. `sel` is the kind's selector.
#   at(src, ::Type{R}, sel, ts)               -> Vector{R}        timestamp == ts. Sorted. Empty = absent.
#   between(src, ::Type{R}, sel, from, to)    -> iterable of R    from <= timestamp <= to. Sorted. Lazy allowed.
#   timestamps(src, ::Type{R}, sel, from, to) -> Vector{DateTime}
at(src, ::Type{R}, sel, ts::DateTime) where R = collect(between(src, R, sel, ts, ts))   # default

# ================= Library over the protocol. Plain Julia, not protocol.
asof(src, ::Type{R}, sel, ts) where R = last_or_missing(between(src, R, sel, typemin(DateTime), ts))
only_or_missing(v)  = isempty(v) ? missing : only(v)
last_or_missing(it) = (x = missing; for r in it; x = r; end; x)
by_timestamp(it)    = ...            # lazy run-length grouping of a sorted iterable into (ts, Vector{R})

# ================= Provider specs: immutable, per storage, no resources. Config builds; identity hashes.
struct ParquetOptionBars;  root::String end                    # OptionBar for every symbol= partition
struct ParquetSpots;       root::String end                    # SpotPrice, same
struct CsvEvents{R};       path::String end                    # R for every selector in the file
struct InMemory{R};        rows::Vector{R} end
struct Constant{R};        record::R end                       # one record at typemin(DateTime)
struct ByUnderlying{P};    parts::Dict{Underlying,P} end       # composition: route sel -> sub-provider
struct QuotesFromBars{Q};  synthesizer::Q end                  # derived: reads OptionBar through the map
struct SurfaceFrom;        spot_for::Dict{Underlying,Underlying} end   # derived; optional spot remap

kind(::ParquetOptionBars) = OptionBar;   kind(::ParquetSpots) = SpotPrice
kind(::CsvEvents{R}) where R = R;        kind(::InMemory{R}) where R = R;   kind(::Constant{R}) where R = R
kind(p::ByUnderlying) = kind(first(values(p.parts)))
kind(::QuotesFromBars) = OptionQuote;    kind(::SurfaceFrom) = VolSurface

# ================= The map: one provider per kind, looked up by type. Immutable. What Experiment stores.
struct MarketData{P<:Tuple}; entries::P end
entry(m::MarketData, ::Type{R}) where R = _entry(R, m.entries...)
_entry(::Type{R}, p, rest...) where R = kind(p) === R ? p : _entry(R, rest...)    # folds at compile time
_entry(::Type{R}) where R = error("MarketData has no provider for $R")
# Loader checks once: no two entries share a kind; derived entries' input kinds are present.

# ================= open / close: each entry once. No ordering: derived entries hold parameters, not inputs.
open(m::MarketData)  = MarketData(map(open, m.entries))
close(m::MarketData) = foreach(close, m.entries)

struct ParquetBarsReader;  spec::ParquetOptionBars; con::DuckDB.DB; days::LRU; chains::LRU end
struct ParquetSpotsReader; spec::ParquetSpots;      con::DuckDB.DB; days::LRU end
struct SurfaceReader;      spec::SurfaceFrom;       cache::LRU{Tuple{Underlying,DateTime},Vector{VolSurface}} end

open(s::ParquetOptionBars) = ParquetBarsReader(s, DuckDB.DB(":memory:"), LRU(200), LRU(10))
open(s::ParquetSpots)      = ParquetSpotsReader(s, DuckDB.DB(":memory:"), LRU(200))
open(s::ByUnderlying)      = ByUnderlying(Dict(u => open(p) for (u, p) in s.parts))
open(s::SurfaceFrom)       = SurfaceReader(s, LRU(64))
open(s)                    = s                                 # Constant, InMemory, CsvEvents, QuotesFromBars
close(r::ParquetBarsReader)  = DBInterface.close!(r.con)
close(r::ParquetSpotsReader) = DBInterface.close!(r.con)
close(r::ByUnderlying)       = foreach(close, values(r.parts))
close(x)                     = nothing
kind(r::ParquetBarsReader)   = OptionBar;  kind(r::ParquetSpotsReader) = SpotPrice;  kind(r::SurfaceReader) = VolSurface

# ================= Verb on the map: entry by type, pass the map down as context.
at(m::MarketData, ::Type{R}, sel, ts) where R              = at(entry(m, R), m, R, sel, ts)
between(m::MarketData, ::Type{R}, sel, from, to) where R   = between(entry(m, R), m, R, sel, from, to)
timestamps(m::MarketData, ::Type{R}, sel, from, to) where R = timestamps(entry(m, R), m, R, sel, from, to)

# Raw providers ignore the context.
at(r::ParquetBarsReader, ::Any, ::Type{OptionBar}, u::Underlying, ts) = _load_chain_at(r, u, ts)
between(r::ParquetBarsReader, ::Any, ::Type{OptionBar}, u::Underlying, from, to) =
    Iterators.flatten(_day_bars(r, u, d, from, to) for d in Date(from):Day(1):Date(to))
at(r::ParquetSpotsReader, ::Any, ::Type{SpotPrice}, u::Underlying, ts) = _spot_at(r, u, ts)
between(c::Constant{R}, ::Any, ::Type{R}, sel, from, to) where R =
    from <= c.record.timestamp <= to ? [c.record] : R[]
between(p::InMemory{R}, ::Any, ::Type{R}, sel, from, to) where R =
    filter(r -> selector(r) == sel && from <= r.timestamp <= to, p.rows)

# Composition forwards the context untouched.
at(p::ByUnderlying, m, ::Type{R}, u::Underlying, ts) where R          = at(p.parts[u], m, R, u, ts)
between(p::ByUnderlying, m, ::Type{R}, u::Underlying, from, to) where R = between(p.parts[u], m, R, u, from, to)

# Derived providers read through the context. Whatever `m` is, cut or not, is all they can see.
at(p::QuotesFromBars, m, ::Type{OptionQuote}, u, ts) =
    map(b -> synthesize(p.synthesizer, b), at(m, OptionBar, u, ts))
between(p::QuotesFromBars, m, ::Type{OptionQuote}, u, from, to) =
    Iterators.map(b -> synthesize(p.synthesizer, b), between(m, OptionBar, u, from, to))

function at(r::SurfaceReader, m, ::Type{VolSurface}, u::Underlying, ts)
    get!(r.cache, (u, ts)) do
        su    = get(r.spec.spot_for, u, u)                    # e.g. SPY options against SPX spot, if configured
        chain = at(m, OptionQuote, u, ts)
        spot  = only_or_missing(at(m, SpotPrice, su, ts))
        rate  = asof(m, RateCurve, USD, ts)                   # currency source: decide at step 3
        div   = asof(m, DivCurve,  u, ts)
        (isempty(chain) || ismissing(spot) || ismissing(rate) || ismissing(div)) && return VolSurface[]
        [build_surface(chain, spot.price, rate.curve, div.curve)]
    end
end

# ================= Time cut: wraps the map, passes ITSELF down. Derived reads cannot escape it.
struct TimeCut{M}; inner::M; cutoff::DateTime end
at(c::TimeCut, ::Type{R}, sel, ts) where R =
    ts <= c.cutoff ? at(entry(c.inner, R), c, R, sel, ts) : R[]
between(c::TimeCut, ::Type{R}, sel, from, to) where R =
    from <= c.cutoff ? between(entry(c.inner, R), c, R, sel, from, min(to, c.cutoff)) : R[]
timestamps(c::TimeCut, ::Type{R}, sel, from, to) where R =
    from <= c.cutoff ? timestamps(entry(c.inner, R), c, R, sel, from, min(to, c.cutoff)) : DateTime[]

# ================= Lifecycle: the run opens and closes. Experiment holds the spec map.
function run_experiment(exp)
    data = open(exp.data)
    try
        positions = run_backtest(exp.agent, data, exp.from, exp.to)   # engine builds TimeCut(data, t) per tick
        ...
    finally
        close(data)
    end
end

# ================= Policy view. Only kinds and selectors; never entry names.
function decide(p::DailyShortStrangle, t, cut, positions)
    surf  = only_or_missing(at(cut, VolSurface,  p.underlying, t))
    chain = at(cut, OptionQuote, p.underlying, t)
    spot  = only_or_missing(at(cut, SpotPrice,   p.underlying, t))
    ...
end
```

Config, one table per kind:

```toml
[data.option_bar]    type = "parquet_option_bars"  root = "C:/.../massive"
[data.option_quote]  type = "from_bars"            synthesizer = { type = "ohlcv_spread", lambda = 0.7 }
[data.rate_curve]    type = "constant"             currency = "USD"    value = 0.045
[data.div_curve]     type = "constant"             underlying = "SPY"  value = 0.013
[data.vol_surface]   type = "surface_from"         spot_for = { SPY = "SPX" }

[data.spot_price]
type = "by_underlying"
SPY  = { type = "parquet_spots", root = "C:/.../massive" }
SPX  = { type = "csv_spots",     path = "C:/.../spx.csv" }
```

Follow `at(cut, VolSurface, SPY, t)` through: the cut checks `t`, finds
the `VolSurface` entry, and calls the surface reader with `m = cut`. The
reader asks `at(cut, OptionQuote, SPY, t)`, which reaches
`QuotesFromBars`, which asks `at(cut, OptionBar, SPY, t)`, which reaches
the one parquet bars reader. The engine's `resolve_quote` asks
`at(cut, OptionQuote, SPY, t)` at the same tick and lands on the same
bars reader and the same chain cache. Every read a derived provider
makes goes through `cut`, because `cut` is the only map it was handed.

## Appendix C. Reviews of v2

The same two reviewers, on v2 at `7fe3382`. Both moved from "park the
generic layer" to "adopt with changes": the structural holes are
closed; what remains are protocol contracts to settle before config,
identity and persistence migrate.

### Review A2 (Claude, 2026-09-06)

Reviewed v2 at `7fe3382` (including Appendix B) against `master` at
`d08b76e`. Verdict: **adopt with changes.** v1's structural holes are
closed; what remains are protocol contracts that are cheap to fix now
and expensive after config, identity and persistence migrate.

#### Resolved from Review A

- **A1 duplicate readers.** Derived providers hold parameters only and
  read through the map they are handed. Surface, quotes and
  `resolve_quote` share one bars reader and one chain cache.
- **A2 cardinality.** `only_or_missing` / `asof` in the library make
  singleton call sites one-liners. Acceptable as documentation plus
  helpers; a per-kind cardinality trait would let the loader check it.
- **A3 `Flat` semantics.** `Constant{R}` at `typemin(DateTime)` is
  coherent: `asof` finds it, `between` over a real window never does.
- **A4 underlying.** Selector as a verb argument; `ByUnderlying`
  composes sources per symbol.
- **A5 materialization.** `between` is an iterable; one day in memory.
- **A6 run-id break.** Decided explicitly; `schema_version` in the
  manifest.
- **A7 type stability.** Tuple lookup by `kind(p) === R` folds when
  every entry is concrete. See item 6 below for the one entry that is
  not.
- **A8 rule 5.** A gated step 0 rather than an afterthought. Section 10
  is still empty, so the proposal is not yet acceptable under rule 5.

#### Still needed, ranked by cost of reversing later

1. **The engine has no clock.** Under v2, `run_backtest`'s default tick
   enumeration and `run_experiment`'s window-end walk both call
   `timestamps(data, R, sel, from, to)`, and neither knows `R` or
   `sel`. Today `available_timestamps(mds)` is implicitly the chain
   source. The experiment (or the agent) must declare the clock kind and
   selector, or `tick_times` must become mandatory. Not in the doc.
2. **`asof` scans from the start of time.** The library default
   `between(src, R, sel, typemin(DateTime), ts)` contradicts "ranges
   are always bounded" and, on any day-partitioned provider modelled on
   the parquet iterator, walks dates from year zero. Make `asof` a
   provider operation with a required efficient implementation for
   snapshot kinds.
3. **Visibility time vs effective time.** Section 2.3 says nothing on
   the list needs it. `Split` and `Dividend` are on the list and both
   have a declaration date and an effective date. Appendix B's own
   `between(cut, Dividend, u, t, t + Day(30))` is clamped to `t` by the
   cut and returns nothing, so "events effective in the next 30 days
   that were known at `t`" cannot be expressed. The cut must filter on
   when a record became knowable; the effective date is a separate
   field.
4. **`open(s) = s` and `close(x) = nothing` on `Any`.** If these extend
   `Base.open` / `Base.close` they are catch-all methods on exported
   Base generics that silently absorb missing lifecycle
   implementations. Use narrow methods or a project-owned pair.
5. **Partial `open` failure leaks connections.** `map(open, entries)`
   with the third entry throwing leaves the first two DuckDB connections
   open and never returns a `MarketData` for the `finally` to close.
   Today's finalizer and `closed` flag are removed without a
   replacement. Open must close already-opened entries on failure;
   use-after-close needs a defined error.
6. **`ByUnderlying` breaks the type-stability claim.** `kind(p) =
   kind(first(values(p.parts)))` is a runtime Dict read, so
   `entry(m, SpotPrice)` does not fold when that entry is a
   `ByUnderlying`. The motivating example (SPY parquet, SPX csv) also
   needs two provider types in one `Dict{Underlying,P}`, which the
   field cannot hold without widening `P`. Make the kind a type
   parameter and define heterogeneous routing explicitly.
7. **Steps 1 and 2 are not the reviewers' increments.** Section 8 says
   they are; both v1 reviews asked for the spec/reader split behind the
   *current* `get_*` API first, then the range benchmark, then the
   generic map. v2's step 1 lands the whole generic layer before any
   parquet code moves. Reordering costs nothing and de-risks the gate.
8. **Section 10 is empty.** Fill it before calling the API accepted.

Deferrable (real, but no harder to add after the migration): a dataset
version fingerprint in the identity hash; more than one provider per
kind for variant comparison; a structural raw-versus-model boundary
beyond documentation.

#### Practical notes for execution

- `docs/status.md` says execution waits for "the machine with the
  data". The DevBox is that machine: `~/data/massive` holds the parquet
  tree (about 2974 option days) and
  `configs/strangle_spy_16d_1dte.local.toml` already points at it. The
  section 8 gate and the step 2 benchmark can run here.
- No saved run exists in any store on the DevBox. The gate diffs
  against "the run saved before step 1", so a run must be saved before
  any code lands or the gate has nothing to compare to.

### Review B2 (Codex, 2026-09-06)

#### Scope

The requested branch currently resolves to `7fe3382`, not `3379c5a`. Commit `3379c5a` contains the v2 revision and is its parent; `7fe3382` adds the end-to-end sketch in Appendix B. This review follows the requested `git show origin/claude/data-ingestion-scalability-av3tzj:docs/proposals/data_kinds.md`, so it evaluates v2 including that sketch. No branch was checked out and no repository file was modified.

#### Updated verdict

**Adopt with changes.**

V2 is substantially stronger than v1. It resolves the most serious mechanical defects in the original proposal: range reads can now be lazy, queries have selectors, derived providers no longer embed and reopen their dependencies, the time cut is propagated through derived reads, invalid post-cut ranges return empty, and the migration plan now respects the documentation rule. The end-to-end sketch also makes the intended dispatch path much easier to audit.

It is still not ready to adopt as-is. The largest remaining issues are:

- `asof` is expressed as a scan from `typemin(DateTime)`, contradicting the bounded-scan principle and potentially producing pathological day iteration.
- Cardinality is documented but still not part of the protocol, and snapshot duplicates can be silently accepted.
- The one-provider-per-kind rule prevents two quote or surface variants for the same underlying.
- The selector contract and heterogeneous routing story are underspecified and may undermine the claimed type stability.
- The lifecycle sketch leaks resources if opening a later entry fails and uses dangerously broad `open`/`close` fallback methods.
- Knowledge time versus effective time remains missing even though `Split` and `Dividend` are already proposed kinds that need richer temporal semantics.
- Dataset versioning remains outside run identity.
- The proposal still dissolves the raw/model boundary rather than merely making it extensible.

The generic architecture is now credible enough to prototype, but those contracts should be settled before it replaces the current `DataSource` / `ModelDataSource` path in `src/data/source.jl` and `src/model_data/source.jl`.

#### Findings v2 resolved

##### B1: range materialization

**Resolved in design.**

`between` now promises an iterable rather than `Vector{R}`, and the parquet example uses a lazy day-by-day `Iterators.flatten`. The proposal also supplies `by_timestamp` for lazy chain grouping. This directly addresses the risk of collecting a month of minute-level option rows in memory and is consistent with the bounded-cache intent in `docs/modules/data.md`.

The proposed benchmark and `at == collect(between)` equivalence test are appropriate validation gates. Implementation still needs to prove that `_day_bars` releases or evicts each day predictably, but the protocol-level flaw is fixed.

##### B4: duplicated provider trees and reader ownership

**Resolved for derived dependencies.**

Derived providers now contain only transformation parameters and resolve inputs through the `MarketData` context. Consequently, `QuotesFromBars`, `SurfaceFrom`, and `resolve_quote` all reach the same `OptionBar` entry and reader. This removes v1's duplicated nested specs, duplicated caches, and ambiguous close ownership.

The load-time check for required input kinds and duplicate kind entries is also a useful improvement.

##### B5: no-lookahead through derived providers

**Resolved within the supported interface.**

`TimeCut` now passes itself as the context to the selected provider. A derived provider reading another kind therefore calls back through the cut rather than through an unrestricted embedded map. The explicit `from > cutoff` case also fixes the invalid-range bug called out in Review B.

This matches the practical guarantee of the current `src/backtest/time_cut.jl`: code can still deliberately reach into public fields if it tries, but every supported data operation enforces the cut. The new step-3 derived-read cutoff test should be retained.

##### B11: documentation updates lagging implementation

**Resolved in the execution plan.**

V2 explicitly requires affected module docs to change in the same commit. That brings the migration into compliance with rule 1 of `docs/design.md`.

##### Missing multi-asset selector dimension

**Resolved at the query level.**

The `sel` argument gives queries an underlying or currency dimension, and `ByUnderlying` demonstrates per-symbol routing across storage implementations. This is materially better than v1's kind-only key and addresses the multi-asset limitation noted in `docs/modules/backtest.md`.

##### Full-window timestamp scan

**Resolved conceptually.**

Walking bounded timestamp partitions backward from `to` avoids the current second full-window enumeration in `src/experiment/experiment.jl`. The exact API should be specified, but the proposed behavior is sound.

#### Findings only partially addressed

##### B2 and B9: cardinality and absence

**Partially addressed.**

V2 documents natural shapes and introduces `only_or_missing`, which distinguishes absent singleton data from duplicates: empty becomes `missing`, while `only` errors when there is more than one record. That is an improvement over forcing each consumer to invent its own check.

However, cardinality remains documentation rather than a provider or kind contract. All `at` calls still return `Vector{R}` regardless of whether a kind is zero-or-one or zero-to-many. Every consumer must remember whether to call `only_or_missing`, and nothing lets the loader validate that a provider satisfies the declared shape.

There is also an inconsistency in `asof`: `last_or_missing` silently selects the last record and does not reject two snapshot records at the same latest timestamp. Thus exact singleton lookup rejects duplicate spot records, while snapshot lookup can conceal duplicate curves.

Concrete change: define cardinality and temporal mode as traits or explicit operations, and make singleton/as-of helpers detect duplicate records at the selected timestamp. Empty iterables are fine for many-valued queries; singleton lookup should have a distinct typed result contract.

##### B3: exact versus as-of semantics

**Partially addressed.**

Adding `asof` and identifying `RateCurve` / `DivCurve` as snapshot kinds solves the basic exact-time problem. Time-cut clamping also composes correctly.

But this implementation is unsafe:

```julia
asof(src, R, sel, ts) = last_or_missing(between(src, R, sel, typemin(DateTime), ts))
```

It violates the proposal's claim that ranges are always meaningfully bounded and the existing rule against accidental whole-dataset scans in `docs/modules/data.md`. On a day-partitioned provider modeled after the proposed parquet iterator, it can attempt to enumerate dates from year 0000 to the query date. Saying providers “may override the pattern” turns `asof` into a de facto protocol method despite calling it an ordinary library helper.

Concrete change: make `asof` an explicit provider capability with an efficient required implementation for snapshot kinds, or add bounded reverse timestamp discovery such as `latest_at_or_before`. Do not define it using `typemin(DateTime)`.

##### B8: constant-source semantics

**Partially addressed.**

`Constant{R}` no longer fabricates one record per queried timestamp, and its `asof` behavior is deterministic. That fixes the worst ambiguity in `Flat{R}`.

Timestamping a timeless configured value at `typemin(DateTime)` is still a sentinel workaround. It creates deliberately inconsistent public behavior: `asof` finds the value, `at` at every real timestamp does not, `between` over an experiment window excludes it, and `timestamps` is empty. The value is admitted to be “a model input, not an observation,” yet it is forced into an observation record.

Concrete change: model configured constants separately from historical snapshot providers, or give snapshot sources a direct `value_asof` operation whose constant implementation does not invent a timestamp. If a provenance timestamp matters, require an explicit `known_from` rather than using `typemin`.

##### B6 and B7: raw/model boundary and `OptionBar` exposure

**Partially addressed, despite section 9 calling them answered.**

It is good that derived providers now read raw providers through a constrained map and that policies are documented to use `OptionQuote`, not `OptionBar`. The transformation direction is clearer than in v1.

Nevertheless, the proposal still places vendor bars, canonical quotes, curve observations, and derived volatility surfaces in one undifferentiated `MarketData` tuple and explicitly dissolves `model_data`. Documentation alone does not preserve the current invariant in `docs/modules/model_data.md`: raw data knows no math, model objects know no I/O, and builders are their meeting point. Any policy can call `at(cut, OptionBar, ...)`, and any provider can reach any other kind through the map.

Concrete change: retain a structural boundary. One option is a raw observation catalog plus a model-facing view whose allowed derived kinds are built from that catalog. Another is capability-restricted contexts: `QuotesFromBars` receives access to `OptionBar`, while policies receive only canonical/model capabilities. `OptionBar` can remain inspectable in diagnostic workflows without becoming part of every policy's supported interface.

##### B10: identity canonicalization and reproducibility

**Partially addressed.**

Removing nested provider specs resolves v1's structural duplication problem, and adding `schema_version` gives persistence a clean compatibility failure. Explicitly accepting a one-time run-ID break is reasonable while the store is tiny.

But the proposal explicitly defers dataset-version identity. A root path still identifies location, not data contents or snapshot. The same experiment hash can therefore produce different results after parquet files are corrected or extended. This is the larger reproducibility issue identified in Review B and remains unresolved in `src/experiment/identity.jl`'s current path-based projection as well.

Concrete change: the initial schema should reserve and hash a logical dataset ID plus immutable version/snapshot/fingerprint. `schema_version` outside the hash only versions the manifest shape; it does not identify the data used.

##### B12: community conventions

**Partially addressed procedurally.**

Moving the convention check to step 0 is the right sequence, but section 10 is still empty. Therefore rule 5 in `docs/design.md` has not yet been satisfied, and the proposal cannot be adopted as-is today.

The check must specifically resolve lazy partition interfaces, safe resource APIs, whether extending `Base.open` / `Base.close` is appropriate, and the performance characteristics of tuple/type-marker lookup. Its conclusions may require changing the public API, not merely documenting the already-selected one.

##### Over-engineering and timing

**Still only partially answered.**

The proposal intentionally rejects the recommendation to park the generic map. V2 makes that map much more coherent, but it does not create present demand for `CsvEvents`, inflation data, historical curve sources, or multiple provider families. The current flagship policy still uses sparse daily ticks in `src/policies/daily_short_strangle.jl`.

Steps 1 and 2 are not quite the incremental path Review B recommended: step 1 introduces the entire generic kind/map/cut framework before extracting the concrete spec/reader split. The lower-risk sequence remains to extract lifecycle first behind the current API, benchmark range access, and then introduce the generic public layer when at least one additional real kind exercises it.

#### Findings unaddressed

##### Knowledge time versus effective time

**Unaddressed.**

V2 states that a future kind can carry both fields and that the provider will decide which one `between` bounds on, then claims nothing currently listed needs this distinction. That is not convincing: `Split` and `Dividend` are already listed kinds, and both naturally have announcement/declaration dates and effective/ex-dividend dates. Revised macro data makes the problem more obvious, but it is not the first use case.

Leaving the choice to each provider also breaks the supposedly uniform meaning of `between`: two providers for the same kind could bound on different timestamps.

Concrete change: define the protocol's visibility time explicitly and represent effective time separately. No-lookahead must always filter on when the record became knowable; domain logic can separately query its effective date.

##### Lifecycle failure behavior

**Unaddressed.**

The proposal still does not handle partial failure during `open(MarketData)`. If opening entry three fails after entries one and two acquired DuckDB connections, no `MarketData` value is returned and the `finally` in `run_experiment` cannot close those earlier readers. Similarly, a failure while closing one entry can prevent later entries from being closed.

The current `src/data/parquet_source.jl` has explicit closed-state checks and a finalizer. V2 removes those defenses without replacing failure cleanup or defining use-after-close behavior.

Concrete change: implement exception-safe acquisition that closes already-opened entries in reverse order, best-effort close aggregation, and a specified use-after-close error. Add tests that inject failures during both open and close.

##### Multiple providers or derived variants of one kind

**Unaddressed and made more restrictive.**

One entry per type means an experiment cannot simultaneously carry:

- two `OptionQuote` series synthesized from the same bars with different spread assumptions;
- vendor quotes and synthesized quotes for comparison;
- two volatility surfaces built with different conventions;
- two feeds for the same underlying and kind.

The selector distinguishes instruments, not provenance or model variants. `ByUnderlying` cannot represent two sources for the same underlying. This matters directly for research and contradicts the proposal's goal of scalable composition.

Concrete change: key entries by a typed capability plus a semantic instance ID, or introduce typed handles such as `DataRef{OptionQuote}` selected in policy/surface configuration. Preserve type dispatch while allowing more than one instance of a kind.

##### Capability and selector contracts

**Unaddressed.**

The proposal says each kind has a selector, but does not define a `selector(::R)` or `selector_type(::Type{R})` contract in the main design. Appendix B calls `selector(r)` without defining it. Nothing statically or at load time prevents calling `RateCurve` with an `Underlying` or `SpotPrice` with a currency.

`ByUnderlying{P}` is also shown with `Dict{Underlying,P}`, yet the motivating example routes SPY to parquet and SPX to CSV—different provider types. That requires an abstract/union-valued dictionary, a different tuple representation, or a wrapper, each with consequences for type stability. `kind(p::ByUnderlying) = kind(first(values(p.parts)))` fails for an empty map and assumes all parts share a kind.

Concrete change: specify selector traits and validation, generalize routing to `BySelector`, define heterogeneous storage explicitly, reject empty/mixed-kind routes at construction, and benchmark the actual representation before claiming compile-time folding end to end.

#### New problems introduced by v2

##### 1. Blanket `open` and `close` fallbacks are unsafe

Appendix B shows:

```julia
open(s) = s
close(x) = nothing
```

If these extend Julia's exported `Base.open` and `Base.close`, they are extremely broad methods that can intercept unrelated values throughout the module and hide missing lifecycle implementations. If they create new module-local generics, their names are confusingly indistinguishable from Base operations.

Use narrowly typed methods on a project-owned lifecycle API—such as `open_data`, `close_data!`, or `with_data`—and never define catch-all methods on `Any`. A no-resource provider should implement an explicit marker or narrow method.

##### 2. `asof` reintroduces unbounded discovery under another name

V1 preserved bounded ranges. V2 now legitimizes a `typemin(DateTime)` lower bound because expected event tables are small. That is a semantic regression: size assumptions belong to providers, and a generic helper should not silently turn into a whole-history scan. It is especially dangerous with the shown daily-partition iterator.

##### 3. Future-looking event queries conflict with `TimeCut`

The policy example includes:

```julia
between(cut, Dividend, underlying, t, t + Day(30))
```

The cut clamps this to `t`, so it cannot return future effective dividends even if those dividends were announced before `t`. This exposes why visibility time and effective time cannot be collapsed into one `timestamp`. The design currently cannot express “events effective in the next 30 days that were already known at the decision time.”

##### 4. The “one reader per storage” claim is imprecise

`ParquetOptionBars` and `ParquetSpots` have the same root but open separate DuckDB connections. V2 successfully guarantees one reader per *kind entry*, not necessarily one per storage. Separate readers may be acceptable, but the proposal should state that accurately or introduce a shared dataset/session layer if one connection per physical store is intended.

##### 5. The type-stability claim is not established

Tuple lookup over fully concrete entries may optimize well, but `_entry` uses a runtime `kind(p) === R` test, `ByUnderlying` may contain heterogeneous providers, derived calls pass a general context, and configuration construction may erase concrete types. The proposal should treat type stability as a benchmarkable hypothesis until step 0 and a prototype confirm inference with `@code_warntype` and realistic loaded configs.

#### Concrete changes required before adoption

1. Replace the `typemin(DateTime)` implementation of `asof` with an explicit efficient `latest_at_or_before`/`asof` provider operation or bounded reverse-index API.
2. Define kind metadata for selector type, cardinality, and temporal mode; validate it at construction and make snapshot duplicate handling explicit.
3. Model visibility/knowledge time separately from effective time before adding `Split`, `Dividend`, or revised macro kinds.
4. Allow multiple named or typed instances of the same kind so research can compare quote and surface variants.
5. Specify heterogeneous selector routing with a type-correct `BySelector` representation and construction invariants.
6. Replace broad `open`/`close` fallbacks with a project-owned, narrowly dispatched lifecycle API.
7. Make multi-resource opening and closing exception-safe; define use-after-close behavior and test partial failures.
8. Preserve a structural raw-observation versus model-derived boundary, or use capability-restricted contexts instead of handing every provider and policy the full map.
9. Include a dataset version/snapshot identifier in hashed experiment identity from the first new schema version.
10. Complete and record the rule-5 convention study before accepting public names or dispatch shapes.
11. Prototype and benchmark tuple lookup, heterogeneous routing, lazy day iteration, point-versus-range performance, and memory retention before deleting the old layer.
12. Reorder migration so the spec/reader extraction and range benchmark can land behind existing APIs before the generic map becomes a required public surface.

#### Final assessment

V2 should not be rejected: it has turned the central idea into a plausible architecture and directly fixed several of Review B's strongest objections. It also should not be adopted as-is, because several unresolved points are protocol-level decisions that would be expensive to reverse after config, identity, policies, and persistence migrate.

The right decision is **adopt with changes**: approve spec/reader separation, lazy bounded ranges, selector-aware access, quote synthesis as a derived transformation, and cut-aware derived reads; require the twelve changes above before approving the complete replacement of `DataSource` and `ModelDataSource`.

