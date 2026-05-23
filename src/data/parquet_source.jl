using DuckDB
using Tables
using OrderedCollections

const ContractMeta = NamedTuple{(:expiry, :strike, :option_type),Tuple{DateTime,Float64,OptionType}}

struct SpotDay
    timestamps::Vector{DateTime}
    prices::Vector{Float64}
end

mutable struct ParquetDataSource <: DataSource
    underlying::Underlying
    synthesizer::QuoteSynthesizer
    options_root::String
    spot_root::String
    max_days_cached::Int
    chain_cache::OrderedDict{Date,Dict{DateTime,Vector{OptionQuote}}}
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
    max_days_cached::Int=3,
)
    max_days_cached >= 1 || throw(ArgumentError("max_days_cached must be >= 1"))
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
        u, synthesizer, String(options_root), String(spot_root), max_days_cached,
        OrderedDict{Date,Dict{DateTime,Vector{OptionQuote}}}(),
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
    max_days_cached::Int=3,
)
    # Lazy validation: see the kwarg constructor's note.
    ParquetDataSource(
        underlying;
        options_root=joinpath(root, options_subdir),
        spot_root=joinpath(root, spot_subdir),
        synthesizer=synthesizer,
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

function _load_chain_day(ds::ParquetDataSource, d::Date)::Dict{DateTime,Vector{OptionQuote}}
    isdir(ds.options_root) ||
        throw(ArgumentError("options_root not a directory: $(ds.options_root)"))
    path = option_path(ds, d)
    isfile(path) || return Dict{DateTime,Vector{OptionQuote}}()

    cols = _parquet_columns(ds, path)
    has_volume = :volume in cols
    has_open   = :open in cols
    has_high   = :high in cols
    has_low    = :low in cols
    has_parsed = :parsed_expiry in cols && :parsed_strike in cols &&
                 :parsed_option_type in cols && :parsed_underlying in cols

    base = "ticker, close, timestamp"
    base = has_volume ? base * ", volume" : base
    base = has_open   ? base * ", open"   : base
    base = has_high   ? base * ", high"   : base
    base = has_low    ? base * ", low"    : base
    select_list = has_parsed ?
        base * ", parsed_underlying, parsed_expiry, parsed_strike, parsed_option_type" :
        base

    sql = "SELECT $select_list FROM '$(_sql_path(path))'"
    result = DBInterface.execute(ds.con, sql)

    by_ts = Dict{DateTime,Vector{OptionQuote}}()
    expected = ticker(ds.underlying)

    for row in Tables.rows(result)
        tk = String(row.ticker)
        meta = get(ds.contract_cache, tk, nothing)
        if meta === nothing
            u_str, m = if has_parsed
                String(row.parsed_underlying),
                _contract_meta_from_parsed(row.parsed_expiry,
                                           Float64(row.parsed_strike),
                                           String(row.parsed_option_type))
            else
                u, expiry, otype, strike = parse_polygon_ticker(tk)
                u, (expiry=expiry, strike=strike, option_type=otype)
            end
            u_str == expected || throw(ArgumentError(
                "ticker $tk (underlying $u_str) does not match data source underlying $expected"))
            meta = m
            ds.contract_cache[tk] = meta
        end
        close_val = row.close === missing ? missing : Float64(row.close)
        open_val  = has_open ? (row.open === missing ? missing : Float64(row.open)) : missing
        high_val  = has_high ? (row.high === missing ? missing : Float64(row.high)) : missing
        low_val   = has_low  ? (row.low  === missing ? missing : Float64(row.low))  : missing
        vol = if has_volume
            v = row.volume
            v === missing ? missing : Float64(v)
        else
            missing
        end
        ts = _coerce_dt(row.timestamp)
        bar = OptionBar(
            tk, ds.underlying, meta.expiry, meta.strike, meta.option_type,
            open_val, high_val, low_val, close_val, vol, ts,
        )
        q = synthesize(ds.synthesizer, bar)
        push!(get!(() -> OptionQuote[], by_ts, ts), q)
    end
    by_ts
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

function _ensure_chain_day!(ds::ParquetDataSource, d::Date)
    _assert_open(ds)
    if haskey(ds.chain_cache, d)
        _lru_touch!(ds.chain_cache, d)
    else
        ds.chain_cache[d] = _load_chain_day(ds, d)
        _lru_evict!(ds.chain_cache, ds.max_days_cached)
    end
    ds.chain_cache[d]
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
    day = _ensure_chain_day!(ds, Date(ts))
    isempty(day) && return nothing
    return get(day, ts, nothing)
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
        day = _ensure_chain_day!(ds, d)
        for ts in keys(day)
            from <= ts <= to && push!(out, ts)
        end
        d += Day(1)
    end
    sort!(out)
    out
end

available_timestamps(::ParquetDataSource) = throw(ArgumentError(
    "available_timestamps(::ParquetDataSource) without bounds would scan the entire dataset; " *
    "call available_timestamps(ds, from, to) with a bounded range."))

function clear_cache!(ds::ParquetDataSource)
    empty!(ds.chain_cache)
    empty!(ds.spot_cache)
    return ds
end
