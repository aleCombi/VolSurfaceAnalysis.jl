# BacktestDataSource protocol: abstracts how the backtest engine accesses
# volatility surfaces (for entry) and spot prices (for settlement).

using Dates

# ============================================================================
# Abstract Protocol
# ============================================================================

"""
    BacktestDataSource

Abstract type for backtest data providers. Implementations must define:

- `available_timestamps(source)::Vector{DateTime}` — sorted timestamps where entry surfaces exist
- `get_surface(source, ts)::Union{VolatilitySurface, Nothing}` — surface at `ts`, or `nothing`
- `get_settlement_spot(source, ts)::Union{Float64, Missing}` — settlement spot at `ts`, or `missing`
"""
abstract type BacktestDataSource end

"""
    available_timestamps(source::BacktestDataSource) -> Vector{DateTime}

Return sorted vector of timestamps at which entry surfaces are available.
"""
function available_timestamps(::BacktestDataSource)::Vector{DateTime}
    error("available_timestamps not implemented for this BacktestDataSource")
end

"""
    get_surface(source::BacktestDataSource, ts::DateTime) -> Union{VolatilitySurface, Nothing}

Return the volatility surface at timestamp `ts`, or `nothing` if unavailable.
"""
function get_surface(::BacktestDataSource, ::DateTime)::Union{VolatilitySurface, Nothing}
    error("get_surface not implemented for this BacktestDataSource")
end

"""
    get_settlement_spot(source::BacktestDataSource, ts::DateTime) -> Union{Float64, Missing}

Return the settlement spot price at timestamp `ts`, or `missing` if unavailable.
"""
function get_settlement_spot(::BacktestDataSource, ::DateTime)::Union{Float64, Missing}
    error("get_settlement_spot not implemented for this BacktestDataSource")
end

# ============================================================================
# DictDataSource — wraps pre-loaded Dict{DateTime, VolatilitySurface} + spots
# ============================================================================

"""
    DictDataSource <: BacktestDataSource

Wraps pre-loaded dictionaries of surfaces and spot prices. This is the simplest
implementation and matches the original `backtest_strategy(strategy, surfaces, spots)` API.
"""
struct DictDataSource <: BacktestDataSource
    surfaces::AbstractDict{DateTime, VolatilitySurface}
    spots::AbstractDict{DateTime, Float64}
end

available_timestamps(d::DictDataSource) = sort(collect(keys(d.surfaces)))
get_surface(d::DictDataSource, ts::DateTime) = get(d.surfaces, ts, nothing)
get_settlement_spot(d::DictDataSource, ts::DateTime) = get(d.spots, ts, missing)

# ============================================================================
# ParquetDataSource — lazy loading from parquet files
# ============================================================================

"""
    ParquetDataSource <: BacktestDataSource

Lazily loads volatility surfaces and spot prices from parquet files on demand.
Surfaces are loaded via `path_for_timestamp` + `read_records` (same closures used
by `build_surfaces_for_timestamps`). Spots are loaded per-date from the Polygon
spot directory structure.

# Constructor
    ParquetDataSource(timestamps; path_for_timestamp, read_records,
                      spot_root, spot_symbol, spot_multiplier=1.0, ts_col=:timestamp)
"""
struct ParquetDataSource <: BacktestDataSource
    timestamps::Vector{DateTime}
    path_for_timestamp::Function
    read_records::Function
    spot_root::String
    spot_symbol::String
    spot_multiplier::Float64
    ts_col::Symbol
    surface_cache::Dict{DateTime, Union{VolatilitySurface, Nothing}}
    spot_date_cache::Dict{Date, Dict{DateTime, Float64}}
end

function ParquetDataSource(
    timestamps::Vector{DateTime};
    path_for_timestamp::Function,
    read_records::Function,
    spot_root::String,
    spot_symbol::Union{String, Underlying},
    spot_multiplier::Float64=1.0,
    ts_col::Symbol=:timestamp
)
    sym = spot_symbol isa Underlying ? ticker(spot_symbol) : uppercase(String(spot_symbol))
    ParquetDataSource(
        sort(timestamps),
        path_for_timestamp,
        read_records,
        spot_root,
        sym,
        spot_multiplier,
        ts_col,
        Dict{DateTime, Union{VolatilitySurface, Nothing}}(),
        Dict{Date, Dict{DateTime, Float64}}()
    )
end

available_timestamps(s::ParquetDataSource) = s.timestamps

function get_surface(s::ParquetDataSource, ts::DateTime)::Union{VolatilitySurface, Nothing}
    if haskey(s.surface_cache, ts)
        return s.surface_cache[ts]
    end

    path = s.path_for_timestamp(ts)
    if !isfile(path)
        s.surface_cache[ts] = nothing
        return nothing
    end

    ts_str = Dates.format(ts, "yyyy-mm-dd HH:MM:SS")
    where = "$(String(s.ts_col)) = '$ts_str'"
    records = s.read_records(path; where=where)

    if isempty(records)
        s.surface_cache[ts] = nothing
        return nothing
    end

    surf = try
        build_surface(records)
    catch
        nothing
    end

    s.surface_cache[ts] = surf
    return surf
end

function get_settlement_spot(s::ParquetDataSource, ts::DateTime)::Union{Float64, Missing}
    d = Date(ts)

    # Load all spots for this date if not cached
    if !haskey(s.spot_date_cache, d)
        date_str = Dates.format(d, "yyyy-mm-dd")
        path = joinpath(s.spot_root, "date=$date_str", "symbol=$(s.spot_symbol)", "data.parquet")

        if isfile(path)
            dict = read_polygon_spot_prices(path; underlying=s.spot_symbol)
            if s.spot_multiplier != 1.0
                for (k, v) in dict
                    dict[k] = v * s.spot_multiplier
                end
            end
            s.spot_date_cache[d] = dict
        else
            s.spot_date_cache[d] = Dict{DateTime, Float64}()
        end
    end

    return get(s.spot_date_cache[d], ts, missing)
end
