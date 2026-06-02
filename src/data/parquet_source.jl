using DuckDB
using Tables
using OrderedCollections

const ContractMeta = NamedTuple{(:expiry, :strike, :option_type),Tuple{DateTime,Float64,OptionType}}

struct SpotDay
    timestamps::Vector{DateTime}
    prices::Vector{Float64}
end

# Per-day metadata: the distinct chain timestamps in the day's parquet plus
# the column-presence flags for the file. Filled lazily: `timestamps` is
# always populated when a DayMeta lands in the cache, `cols_loaded` /
# `has_*` are filled the first time a chain at any timestamp in this day is
# actually requested (`_load_chain_at` reads them). One DuckDB
# `parquet_schema` query per day instead of per-timestamp.
mutable struct DayMeta
    timestamps::Vector{DateTime}
    cols_loaded::Bool
    has_volume::Bool
    has_open::Bool
    has_high::Bool
    has_low::Bool
    has_parsed::Bool
end

mutable struct ParquetDataSource{S<:QuoteSynthesizer} <: DataSource
    underlying::Underlying
    synthesizer::S
    options_root::String
    spot_root::String
    max_chains_cached::Int
    max_days_cached::Int
    chain_cache::OrderedDict{DateTime,Vector{OptionQuote}}
    day_meta_cache::OrderedDict{Date,DayMeta}
    spot_cache::OrderedDict{Date,SpotDay}
    contract_cache::Dict{String,ContractMeta}
    con::DuckDB.DB
    closed::Bool
end

const DEFAULT_OPTIONS_SUBDIR = "options_1min"
const DEFAULT_SPOTS_SUBDIR = "spots_1min"

function ParquetDataSource(
    underlying::Union{Underlying,AbstractString};
    options_root::AbstractString,
    spot_root::AbstractString,
    synthesizer::QuoteSynthesizer,
    max_chains_cached::Int=10,
    max_days_cached::Int=200,
)
    max_chains_cached >= 1 || throw(ArgumentError("max_chains_cached must be >= 1"))
    max_days_cached   >= 1 || throw(ArgumentError("max_days_cached must be >= 1"))
    # Roots are not hard-validated here -- a source can be constructed
    # against paths that do not yet exist (e.g. when rehydrating an
    # `Experiment` from a saved config in a workspace whose data store
    # is elsewhere). A missing root warns now and throws on the first
    # actual read, so a typo or wrong workspace surfaces immediately
    # without breaking the rehydration path.
    isdir(options_root) || @warn "ParquetDataSource: options_root does not exist (reads will throw)" options_root
    isdir(spot_root)    || @warn "ParquetDataSource: spot_root does not exist (reads will throw)" spot_root
    u = underlying isa Underlying ? underlying : Underlying(underlying)
    ds = ParquetDataSource(
        u, synthesizer, String(options_root), String(spot_root),
        max_chains_cached, max_days_cached,
        OrderedDict{DateTime,Vector{OptionQuote}}(),
        OrderedDict{Date,DayMeta}(),
        OrderedDict{Date,SpotDay}(),
        Dict{String,ContractMeta}(),
        DuckDB.DB(":memory:"),
        false,
    )
    finalizer(_close_con, ds)
    ds
end

function ParquetDataSource(
    underlying::Union{Underlying,AbstractString},
    root::AbstractString;
    synthesizer::QuoteSynthesizer,
    options_subdir::AbstractString=DEFAULT_OPTIONS_SUBDIR,
    spot_subdir::AbstractString=DEFAULT_SPOTS_SUBDIR,
    max_chains_cached::Int=10,
    max_days_cached::Int=200,
)
    # Lazy validation: see the kwarg constructor's note.
    ParquetDataSource(
        underlying;
        options_root=joinpath(root, options_subdir),
        spot_root=joinpath(root, spot_subdir),
        synthesizer=synthesizer,
        max_chains_cached=max_chains_cached,
        max_days_cached=max_days_cached,
    )
end

function _close_con(ds::ParquetDataSource)
    if !ds.closed
        try
            DBInterface.close!(ds.con)
        catch
        end
        ds.closed = true
    end
end

function Base.close(ds::ParquetDataSource)
    _close_con(ds)
    empty!(ds.chain_cache)
    empty!(ds.day_meta_cache)
    empty!(ds.spot_cache)
    return ds
end

Base.isopen(ds::ParquetDataSource) = !ds.closed

function _assert_open(ds::ParquetDataSource)
    isopen(ds) || throw(ArgumentError("ParquetDataSource is closed"))
end

"""
    with_parquet_source(f, args...; kwargs...)

Construct a `ParquetDataSource`, call `f(ds)`, and close the source in
a `finally` block. Supports Julia's `do`-block style:

```julia
with_parquet_source("SPY", root) do ds
    get_spots(ds, from, to)
end
```
"""
function with_parquet_source(f::Function, args...; kwargs...)
    ds = ParquetDataSource(args...; kwargs...)
    try
        return f(ds)
    finally
        close(ds)
    end
end

option_path(ds::ParquetDataSource, d::Date) = joinpath(
    ds.options_root,
    "date=" * Dates.format(d, "yyyy-mm-dd"),
    "symbol=" * ticker(ds.underlying),
    "data.parquet",
)

spot_path(ds::ParquetDataSource, d::Date) = joinpath(
    ds.spot_root,
    "date=" * Dates.format(d, "yyyy-mm-dd"),
    "symbol=" * ticker(ds.underlying),
    "data.parquet",
)

function _lru_touch!(cache::OrderedDict, key)
    if haskey(cache, key)
        v = cache[key]
        delete!(cache, key)
        cache[key] = v
    end
    cache
end

function _lru_evict!(cache::OrderedDict, max_n::Int)
    while length(cache) > max_n
        popfirst!(cache)
    end
    cache
end

function _parquet_columns(ds::ParquetDataSource, path::AbstractString)::Set{Symbol}
    sql = "SELECT name FROM parquet_schema('$(_sql_path(path))')"
    cols = Set{Symbol}()
    for row in Tables.rows(DBInterface.execute(ds.con, sql))
        push!(cols, Symbol(row.name))
    end
    cols
end

function _contract_meta_from_parsed(parsed_expiry, parsed_strike::Float64,
                                    parsed_option_type::AbstractString)::ContractMeta
    expiry_date = Date(_coerce_dt(parsed_expiry))
    expiry = et_to_utc(expiry_date, Time(16, 0))
    otype = parsed_option_type == "C" ? Call : Put
    return (expiry=expiry, strike=parsed_strike, option_type=otype)
end

# DISTINCT-timestamps query per day. Column-pruned, no row materialization
# of the chain itself -- typically <100ms for a SPY 1-min day.
function _query_distinct_timestamps(ds::ParquetDataSource, path::AbstractString)::Vector{DateTime}
    sql = "SELECT DISTINCT timestamp FROM '$(_sql_path(path))' ORDER BY timestamp"
    result = DBInterface.execute(ds.con, sql)
    out = DateTime[]
    for row in Tables.rows(result)
        push!(out, _coerce_dt(row.timestamp))
    end
    out
end

# Populate the column-presence flags on a DayMeta (first chain-load for the
# day; subsequent loads on the same day reuse these without re-querying
# parquet_schema). No-op if already loaded.
function _load_day_cols!(ds::ParquetDataSource, dm::DayMeta, path::AbstractString)
    dm.cols_loaded && return dm
    cols = _parquet_columns(ds, path)
    dm.has_volume = :volume in cols
    dm.has_open   = :open in cols
    dm.has_high   = :high in cols
    dm.has_low    = :low in cols
    dm.has_parsed = :parsed_expiry in cols && :parsed_strike in cols &&
                    :parsed_option_type in cols && :parsed_underlying in cols
    dm.cols_loaded = true
    dm
end

# Load the chain at a single timestamp. WHERE-filtered DuckDB query (parquet
# row-group pruning kicks in if statistics are present), then columnar
# materialization via `Tables.columntable` so the per-row work is type-stable
# index access rather than `getproperty` dispatch.
function _load_chain_at(ds::ParquetDataSource, ts::DateTime)::Vector{OptionQuote}
    isdir(ds.options_root) ||
        throw(ArgumentError("options_root not a directory: $(ds.options_root)"))
    path = option_path(ds, Date(ts))
    isfile(path) || return OptionQuote[]

    dm = _ensure_day_meta!(ds, Date(ts))
    _load_day_cols!(ds, dm, path)

    base = "ticker, close, timestamp"
    base = dm.has_volume ? base * ", volume" : base
    base = dm.has_open   ? base * ", open"   : base
    base = dm.has_high   ? base * ", high"   : base
    base = dm.has_low    ? base * ", low"    : base
    select_list = dm.has_parsed ?
        base * ", parsed_underlying, parsed_expiry, parsed_strike, parsed_option_type" :
        base

    ts_str = Dates.format(ts, "yyyy-mm-dd HH:MM:SS")
    sql = "SELECT $select_list FROM '$(_sql_path(path))' WHERE timestamp = TIMESTAMP '$ts_str'"
    ct = Tables.columntable(DBInterface.execute(ds.con, sql))

    tickers = ct.ticker
    closes  = ct.close
    tstamps = ct.timestamp
    volumes = dm.has_volume ? ct.volume : nothing
    opens   = dm.has_open   ? ct.open   : nothing
    highs   = dm.has_high   ? ct.high   : nothing
    lows    = dm.has_low    ? ct.low    : nothing
    p_und   = dm.has_parsed ? ct.parsed_underlying : nothing
    p_exp   = dm.has_parsed ? ct.parsed_expiry     : nothing
    p_strk  = dm.has_parsed ? ct.parsed_strike     : nothing
    p_otype = dm.has_parsed ? ct.parsed_option_type : nothing

    n = length(tickers)
    # Vector{OptionQuote}(undef, n) + index assignment avoids the per-element
    # `push!` resize check and one bounds check per row. With ds.synthesizer
    # concretely typed (`ParquetDataSource{S}`), the `synthesize` call below
    # is statically dispatched and Julia can inline it into this loop.
    out = Vector{OptionQuote}(undef, n)
    expected = ticker(ds.underlying)

    @inbounds for i in 1:n
        tk = String(tickers[i])
        meta = get(ds.contract_cache, tk, nothing)
        if meta === nothing
            u_str, m = if dm.has_parsed
                String(p_und[i]),
                _contract_meta_from_parsed(p_exp[i],
                                           Float64(p_strk[i]),
                                           String(p_otype[i]))
            else
                u, expiry, otype, strike = parse_polygon_ticker(tk)
                u, (expiry=expiry, strike=strike, option_type=otype)
            end
            u_str == expected || throw(ArgumentError(
                "ticker $tk (underlying $u_str) does not match data source underlying $expected"))
            meta = m
            ds.contract_cache[tk] = meta
        end
        cv = closes[i]
        close_val = cv === missing ? missing : Float64(cv)
        open_val  = opens   === nothing ? missing : (opens[i]   === missing ? missing : Float64(opens[i]))
        high_val  = highs   === nothing ? missing : (highs[i]   === missing ? missing : Float64(highs[i]))
        low_val   = lows    === nothing ? missing : (lows[i]    === missing ? missing : Float64(lows[i]))
        vol       = volumes === nothing ? missing : (volumes[i] === missing ? missing : Float64(volumes[i]))
        row_ts = _coerce_dt(tstamps[i])
        bar = OptionBar(
            tk, ds.underlying, meta.expiry, meta.strike, meta.option_type,
            open_val, high_val, low_val, close_val, vol, row_ts,
        )
        out[i] = synthesize(ds.synthesizer, bar)
    end
    out
end

function _load_spot_day(ds::ParquetDataSource, d::Date)::SpotDay
    isdir(ds.spot_root) ||
        throw(ArgumentError("spot_root not a directory: $(ds.spot_root)"))
    path = spot_path(ds, d)
    isfile(path) || return SpotDay(DateTime[], Float64[])

    sql = "SELECT timestamp, close FROM '$(_sql_path(path))' ORDER BY timestamp"
    result = DBInterface.execute(ds.con, sql)

    ts_buf = DateTime[]
    px_buf = Float64[]
    for row in Tables.rows(result)
        row.close === missing && continue
        push!(ts_buf, _coerce_dt(row.timestamp))
        push!(px_buf, Float64(row.close))
    end
    SpotDay(ts_buf, px_buf)
end

function _ensure_day_meta!(ds::ParquetDataSource, d::Date)::DayMeta
    _assert_open(ds)
    if haskey(ds.day_meta_cache, d)
        _lru_touch!(ds.day_meta_cache, d)
        return ds.day_meta_cache[d]
    end
    isdir(ds.options_root) ||
        throw(ArgumentError("options_root not a directory: $(ds.options_root)"))
    path = option_path(ds, d)
    tstamps = isfile(path) ? _query_distinct_timestamps(ds, path) : DateTime[]
    dm = DayMeta(tstamps, false, false, false, false, false, false)
    ds.day_meta_cache[d] = dm
    _lru_evict!(ds.day_meta_cache, ds.max_days_cached)
    dm
end

function _ensure_chain!(ds::ParquetDataSource, ts::DateTime)::Vector{OptionQuote}
    _assert_open(ds)
    if haskey(ds.chain_cache, ts)
        _lru_touch!(ds.chain_cache, ts)
    else
        ds.chain_cache[ts] = _load_chain_at(ds, ts)
        _lru_evict!(ds.chain_cache, ds.max_chains_cached)
    end
    ds.chain_cache[ts]
end

function _ensure_spot_day!(ds::ParquetDataSource, d::Date)
    _assert_open(ds)
    if haskey(ds.spot_cache, d)
        _lru_touch!(ds.spot_cache, d)
    else
        ds.spot_cache[d] = _load_spot_day(ds, d)
        _lru_evict!(ds.spot_cache, ds.max_days_cached)
    end
    ds.spot_cache[d]
end

function get_chain(ds::ParquetDataSource, ts::DateTime)::Union{Vector{OptionQuote},Nothing}
    chain = _ensure_chain!(ds, ts)
    isempty(chain) ? nothing : chain
end

function get_spot(ds::ParquetDataSource, ts::DateTime)::Union{Float64,Missing}
    sd = _ensure_spot_day!(ds, Date(ts))
    isempty(sd.timestamps) && return missing
    i = searchsortedfirst(sd.timestamps, ts)
    (i <= length(sd.timestamps) && sd.timestamps[i] == ts) || return missing
    return sd.prices[i]
end

function get_spots(ds::ParquetDataSource, from::DateTime, to::DateTime)::Vector{SpotPrice}
    out = SpotPrice[]
    from <= to || return out
    d = Date(from)
    d_end = Date(to)
    while d <= d_end
        sd = _ensure_spot_day!(ds, d)
        if !isempty(sd.timestamps)
            lo = searchsortedfirst(sd.timestamps, from)
            hi = searchsortedlast(sd.timestamps, to)
            for i in lo:hi
                push!(out, SpotPrice(ds.underlying, sd.prices[i], sd.timestamps[i]))
            end
        end
        d += Day(1)
    end
    out
end

function available_timestamps(ds::ParquetDataSource, from::DateTime, to::DateTime)::Vector{DateTime}
    out = DateTime[]
    from <= to || return out
    d = Date(from)
    d_end = Date(to)
    while d <= d_end
        dm = _ensure_day_meta!(ds, d)
        for ts in dm.timestamps          # per-day vector is already sorted
            from <= ts <= to && push!(out, ts)
        end
        d += Day(1)
    end
    out                                   # concatenation of sorted-per-day blocks in date order is sorted
end

available_timestamps(::ParquetDataSource) = throw(ArgumentError(
    "available_timestamps(::ParquetDataSource) without bounds would scan the entire dataset; " *
    "call available_timestamps(ds, from, to) with a bounded range."))

function clear_cache!(ds::ParquetDataSource)
    empty!(ds.chain_cache)
    empty!(ds.day_meta_cache)
    empty!(ds.spot_cache)
    return ds
end
