# Proposal: kinds, providers, readers -- a scalable data layer

Status: proposal, not implemented. Supersedes the `DataSource` /
`ModelDataSource` split described in `docs/modules/data.md` and
`docs/modules/model_data.md` once accepted.

## 1. Why

The current data layer hardcodes *which* data exists. `DataSource`
knows two things, chains and spots, through two verbs (`get_chain`,
`get_spot`). `ModelDataSource` then hardcodes four slots on top
(chain source, spot source, rate curve, dividend curve) and one
derived object (`get_surface`). Every new data need -- splits,
dividends, a rate-curve history, inflation prints, a second
underlying -- costs a new struct field, a new protocol verb, a new
`TimeCutModelDataSource` forwarder, a new config builder, and a new
`to_dict` branch. Five files per data kind.

Three smaller problems ride along:

- `ParquetDataSource` fuses the *description* of a source (underlying,
  roots, synthesizer) with its *running machinery* (DuckDB connection,
  three LRU caches, a `closed` flag). The description is what config
  writes and identity hashes; the machinery is what a run needs. Fusing
  them makes `Experiment` hold a live database handle, forces
  `to_dict` to hand-exclude cache knobs, leaks `close` semantics to
  users, blocks sharing across threads, and is the only reason the
  struct is `mutable`.
- The protocol is point-query shaped. `get_chain(ds, ts)` is one
  DuckDB query per timestamp. Spots got a day-block cache; chains did
  not. Any policy that ticks every minute pays a query per minute.
- The quote synthesizer lives inside the parquet reader as a type
  parameter and a branch in the row loop, so "what the vendor has"
  (OHLCV bars) and "what we make of it" (bid/ask quotes) are entangled.

## 2. Concepts

Five concepts. Each scales on its own axis.

### 2.1 Kind

A kind is a plain immutable record type with a `timestamp::DateTime`
field. It says *what* a datum is. Adding a data need starts with
adding a kind, and often ends there.

```julia
struct OptionBar    ...; timestamp::DateTime end   # what Polygon stores
struct OptionQuote  ...; timestamp::DateTime end   # bid/ask/mark at a contract
struct SpotPrice    ...; timestamp::DateTime end
struct Split        ...; timestamp::DateTime end
struct Dividend     ...; timestamp::DateTime end
struct RateCurve    timestamp::DateTime; curve::Curve end   # curve as of t, evaluated at T
struct DivCurve     timestamp::DateTime; curve::Curve end
struct VolSurface   ...; timestamp::DateTime end   # derived, see 2.5
```

`RateCurve` and `DivCurve` are `(t, T)`-dependent: the record is the
curve *as of* `t`, and the curve is a function of maturity `T`. This
replaces today's `get_rate(ts)::Float64`, which conflated the two
axes. Whether the curve returns zero rates, discount factors, or
forwards is a `Curve` concern, not a data-layer one.

Each kind declares its config/map key explicitly:

```julia
key(::Type{OptionQuote}) = :option_quote
key(::Type{SpotPrice})   = :spot_price
```

### 2.2 Verb

One verb, two arities, one return shape.

```julia
records(source, ::Type{R}, ts::DateTime)                 -> Vector{R}
records(source, ::Type{R}, from::DateTime, to::DateTime) -> Vector{R}
timestamps(source, ::Type{R}, from::DateTime, to::DateTime) -> Vector{DateTime}
```

Rules:

- The result is sorted by `timestamp` and contains only records with
  `from <= timestamp <= to` (or `== ts`).
- **Empty vector means absent.** There is no `nothing` and no
  `missing` at the record level. `missing` remains the convention for
  absent scalar *fields inside* a record (`bid`, `ask`, `iv`, ...).
- Point-in-time and interval are two ways of asking, not two families
  of kinds. Every kind supports both. The default point arity is the
  degenerate range, `records(p, R, ts) = records(p, R, ts, ts)`;
  providers override it when they have a faster path.
- Ranges are always bounded. There is no unbounded discovery verb.
  This keeps the existing "no whole-dataset scans by accident" decision.
- Anything beyond these two shapes is plain Julia over the result:
  `filter`, `map`, `DataFrame(...)`. The data layer defines no query
  language. Providers backed by DuckDB may expose a raw-SQL escape hatch
  for push-down, outside the protocol.

Naming note: `Base.fetch` already exists for tasks and futures. This
proposal uses `records` to avoid shadowing; the name is the one open
bikeshed and is cheap to change before code lands.

### 2.3 Provider spec

A provider spec is an immutable value describing *where* records of one
or more kinds come from. It is what config builds, what identity
hashes, and what persistence writes. It holds no resources.

```julia
struct ParquetOptionBars            # serves OptionBar
    underlying::Underlying
    root::String
end
struct ParquetSpots                 # serves SpotPrice
    underlying::Underlying
    root::String
end
struct CsvEvents{R}                 # serves any event kind from a csv
    path::String
end
struct Flat{R}                      # serves any constant kind (RateCurve with FlatCurve, ...)
    value::Float64
end
struct InMemory{R}                  # serves any kind from a fixture vector
    rows::Vector{R}
end
```

Specs are **per storage, not per kind**. `CsvEvents{Split}` and
`CsvEvents{Dividend}` are the same code. A new kind on an existing
storage costs one struct (the kind) and nothing else.

### 2.4 Reader

A reader is the opened form of a spec: it owns whatever the storage
needs at run time (a DuckDB connection, bounded LRU caches, a socket
for a future live feed). `open(spec)` returns it; `close(reader)`
releases it. Specs whose storage needs nothing are their own reader.

```julia
struct ParquetOptionBarsReader
    spec::ParquetOptionBars
    con::DuckDB.DB
    days::LRU{Date, DayBlock}        # per-day timestamps + column layout
    chains::LRU{DateTime, Vector{OptionBar}}
end
open(s::ParquetOptionBars) = ParquetOptionBarsReader(s, DuckDB.DB(":memory:"), LRU(200), LRU(10))
close(r::ParquetOptionBarsReader) = DBInterface.close!(r.con)

open(s::Flat) = s
open(s::InMemory) = s
```

Neither the spec nor the reader is `mutable`. Caches are mutable
containers held by an immutable reader; there is no `closed` flag
because a reader's lifetime is scoped by whoever opened it (see 3).

`records` is implemented on readers. The parquet bars reader
implements both arities: the point arity is today's `_load_chain_at`;
the range arity is a sequential day pass that loads each day's file
once and slices in memory, the way `SpotDay` already does for spots.

### 2.5 Derived providers

A provider may serve a kind by reading other kinds. It holds its
inputs directly.

```julia
struct QuotesFromBars{Q<:QuoteSynthesizer, P}   # serves OptionQuote
    bars::P                                     # any provider serving OptionBar
    synthesizer::Q
end
records(p::QuotesFromBars, ::Type{OptionQuote}, args...) =
    map(b -> synthesize(p.synthesizer, b), records(p.bars, OptionBar, args...))

struct SurfaceFrom{M}                           # serves VolSurface
    data::M                                     # a MarketData (or reader map)
end
function records(p::SurfaceFrom, ::Type{VolSurface}, ts::DateTime)
    chain = records(p.data, OptionQuote, ts)
    spot  = records(p.data, SpotPrice,   ts)
    rate  = records(p.data, RateCurve,   ts)
    div   = records(p.data, DivCurve,    ts)
    any(isempty, (chain, spot, rate, div)) && return VolSurface[]
    [build_surface(chain, only(spot).price, only(rate).curve, only(div).curve)]
end
open(p::SurfaceFrom) = SurfaceFromReader(p, LRU(...))   # bounded surface cache
```

This is where the OHLCV-to-quote synthesis moves. The parquet reader
becomes vendor-only code; a future live feed serves `OptionQuote`
directly and `QuotesFromBars` is simply not configured. Policies only
ever ask for `OptionQuote`.

Invariant: **a derived provider may not widen the query it forwards.**
It reads its inputs at exactly the timestamp or range it was asked
for. This is what makes derived providers safe under the time cut
(2.7) even though they hold their inputs directly.

### 2.6 The composite

`MarketData` is an immutable map from kind key to provider. It is what
an experiment declares and what the engine walks.

```julia
struct MarketData{P<:NamedTuple}
    providers::P
end
records(m::MarketData, ::Type{R}, args...) where R =
    records(getproperty(m.providers, key(R)), R, args...)

open(m::MarketData)  = MarketData(map(open, m.providers))
close(m::MarketData) = foreach(close, m.providers)
```

A `NamedTuple` rather than a `Dict{DataType,Any}` keeps dispatch
type-stable in the tick loop. Asking for a kind the experiment did not
provide fails at `getproperty`, which is the right place.

"What an experiment gets" is exactly the set of keys in its
`MarketData`. It is declared per experiment in config (section 4), not
fixed in code.

### 2.7 Time cut

```julia
struct TimeCut{M}
    inner::M
    cutoff::DateTime
end
records(c::TimeCut, ::Type{R}, ts) where R =
    ts <= c.cutoff ? records(c.inner, R, ts) : R[]
records(c::TimeCut, ::Type{R}, from, to) where R =
    records(c.inner, R, from, min(to, c.cutoff))
timestamps(c::TimeCut, ::Type{R}, from, to) where R =
    timestamps(c.inner, R, from, min(to, c.cutoff))
```

Two one-liners replace the per-kind forwarders. The rate/div
passthrough exception in today's `TimeCutModelDataSource` is no longer
needed: a `RateCurve` fetched at `t` and evaluated at a future `T` is
a legitimate forward query on data known at `t`.

## 3. Lifecycle

`run_experiment` opens, the engine uses, `run_experiment` closes.

```julia
function run_experiment(exp::Experiment)
    data = open(exp.data)                        # MarketData of readers
    try
        positions = run_backtest(exp.agent, data, exp.from, exp.to)
        ...
    finally
        close(data)
    end
end
```

`Experiment` holds a `MarketData` of *specs*, so it hashes, persists,
and rehydrates without touching disk. Readers exist only inside a run.
Parallel sweeps get one reader set per task from one shared spec set,
which resolves the "concurrent-safe caching" item in the data doc's
future work without locks.

REPL convenience: `with_data(f, market_data)` does the open/close.

## 4. Config

One table per kind key. `type` selects a provider builder from a
registry, extending the existing `_SYNTHESIZER_BUILDERS` /
`_CURVE_BUILDERS` pattern. Derived providers reference other keys by
name; the loader resolves references into nested specs, so the Julia
value is an ordinary immutable tree.

```toml
[data.option_bar]
type = "parquet_option_bars"
underlying = "SPY"
root = "C:/repos/options-collector/data/massive"

[data.option_quote]
type = "from_bars"
bars = "option_bar"
synthesizer = { type = "ohlcv_spread", lambda = 0.7 }

[data.spot_price]
type = "parquet_spots"
underlying = "SPY"
root = "C:/repos/options-collector/data/massive"

[data.rate_curve]
type = "flat"
value = 0.045

[data.div_curve]
type = "flat"
value = 0.013

[data.vol_surface]
type = "surface_from"        # inputs implied: option_quote, spot_price, rate_curve, div_curve

[data.split]
type = "csv"
path = "..."
```

Cache sizes are not config: they are `open` kwargs with defaults, and
never part of identity.

### Identity

`to_dict(::MarketData)` emits one entry per key, each the `to_dict` of
its spec. Derived specs emit their nested inputs inline. This
duplicates the bars spec under both `option_bar` and `option_quote`,
which is deterministic and harmless. Cache knobs, connections, and
readers never appear because they are not on specs.

## 5. Migration map

| Today | Proposal |
|---|---|
| `DataSource`, `get_chain`, `get_spot`, `get_spots`, `available_timestamps` | `records` / `timestamps` on providers |
| `InMemoryDataSource` | `InMemory{R}` |
| `ParquetDataSource{S}` | `ParquetOptionBars` + `ParquetSpots` specs, their readers, and `QuotesFromBars{S}` |
| `OptionBar` as adapter type | `OptionBar` as first-class kind |
| `QuoteSynthesizer` on the parquet source | `QuoteSynthesizer` on `QuotesFromBars` |
| `ModelDataSource` | `MarketData` |
| `get_rate(ts)::Float64`, `get_div(ts)::Float64` | `RateCurve` / `DivCurve` records at `t` |
| `get_surface` + unbounded `surface_cache` | `SurfaceFrom` provider with a bounded reader cache |
| `TimeCutModelDataSource` | `TimeCut{M}` |
| `clear_cache!` | removed; readers are scoped to a run |
| `with_parquet_source` | `with_data` |
| `run_experiment` scanning the whole window for the last timestamp | `timestamps` walked backward from `to`, or the collector index when it lands |
| `resolve_quote` linear scan | unchanged for now; a keyed chain is a later, independent change |

## 6. Rule changes surfaced (design rule 3)

Accepting this proposal changes documented conventions. Each is listed
so it is decided, not absorbed.

1. **Absence convention.** `docs/modules/data.md` says `nothing` for a
   missing aggregate and `missing` for a missing scalar. New rule:
   empty vector for no records; `missing` only inside records.
2. **`OptionBar` status.** `synth.jl` and the data doc call it an
   adapter-layer type. New rule: it is a first-class kind; an experiment
   may ask for bars.
3. **`model_data` module.** Dissolves. `Curve` types stay as values;
   `ModelDataSource` and the time-cut wrapper go.
4. **Rate/div time-cut passthrough.** Removed with the `(t, T)`
   curve records.
5. **Unbounded discovery.** Unchanged in substance, now enforced by
   the verb shape rather than by a throwing method.

## 7. Execution plan

Small, deliberate steps, each leaving the test suite green and the
strangle config reproducing its saved metrics.

1. **Kinds and verb.** Add `records` / `timestamps`, `key`, the
   `MarketData` map, `TimeCut`, `InMemory{R}`, `Flat{R}`. Tests with
   fixtures only. Nothing existing changes yet.
2. **Parquet split.** `ParquetOptionBars` / `ParquetSpots` specs and
   readers, both arities, ported from `parquet_source.jl` with the
   existing parquet fixture tests moved over. Add the range arity's
   sequential day pass and test it against the point arity for equality.
3. **Derived providers.** `QuotesFromBars`, `SurfaceFrom` with bounded
   cache. `RateCurve` / `DivCurve` kinds.
4. **Engine and policies on the new verb.** `run_backtest`,
   `DailyShortStrangle`, `resolve_quote`, `_build_settle`.
5. **Config, identity, persistence.** `[data.*]` loader, `to_dict`,
   `load_run` rehydration. Existing configs rewritten.
6. **Delete the old layer.** `DataSource`, `ModelDataSource`,
   `TimeCutModelDataSource`, `clear_cache!`, `with_parquet_source`.
   Update `docs/modules/data.md`, remove `docs/modules/model_data.md`,
   update `status.md`.

Validation gate for steps 2 through 5: rerun
`configs/strangle_spy_16d_1dte.toml` against the real parquet store
and diff `metrics` and `pnl_series` against the saved run. Step 2 also
wants a wall-clock comparison of point versus range arity over one
month of minute data, so the sequential pass is shown to be the win it
is claimed to be.

## 8. Open questions

- **Verb name.** `records` here; `observe`, `series`, `pull` are all
  defensible. Decide before step 1.
- **`SurfaceFrom` inputs.** Implicit (reads fixed keys from the map) or
  explicit (config names each input key)? Implicit is shorter; explicit
  allows two surfaces from two quote sources in one experiment. Start
  implicit, switch when a second surface is needed.
- **Curve content.** Whether `RateCurve.curve(T)` returns a zero rate
  or a discount factor is a `surfaces` / pricing decision. The data
  layer only promises "the curve as of `t`".
- **Community conventions (design rule 5).** Before step 1, check
  current practice for trait-style dispatch on marker types in the
  Julia data ecosystem (`Tables.jl`, `DataInterpolations.jl`,
  `TimeSeries.jl`) and record findings in this section.

## 9. Reviews

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

