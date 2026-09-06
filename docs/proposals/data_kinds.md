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
