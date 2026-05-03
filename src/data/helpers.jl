# Shared data-loading helpers for scripts
# Consolidates functions previously duplicated across ml_strike_selector.jl,
# evaluate_condor_prediction_vs_baseline.jl, and backtest scripts.

"""
    build_entry_timestamps(dates, entry_times) -> Vector{DateTime}

Build UTC entry timestamps for each (date, time) combination.
"""
function build_entry_timestamps(dates::Vector{Date}, entry_times::Vector{Time})::Vector{DateTime}
    ts = DateTime[]
    for date in dates
        for t in entry_times
            push!(ts, et_to_utc(date, t))
        end
    end
    return sort(ts)
end

function build_entry_timestamps(dates::Vector{Date}, entry_time::Time)::Vector{DateTime}
    return build_entry_timestamps(dates, [entry_time])
end

"""
    load_minute_spots(start_date, end_date; kwargs...) -> Dict{DateTime,Float64}

Load minute-level spot prices from parquet, optionally with a lookback window and price multiplier.
"""
function load_minute_spots(
    start_date::Date,
    end_date::Date;
    lookback_days::Union{Nothing,Int}=nothing,
    symbol::String="SPY",
    multiplier::Float64=1.0,
    store::LocalDataStore=DEFAULT_STORE
)::Dict{DateTime,Float64}
    all_dates = available_polygon_dates(store, symbol)
    isempty(all_dates) && error("No spot dates found for $symbol")

    min_date = lookback_days === nothing ? minimum(all_dates) : start_date - Day(lookback_days)
    filtered_dates = filter(d -> d >= min_date && d <= end_date, all_dates)

    spots = Dict{DateTime,Float64}()
    for d in filtered_dates
        path = polygon_spot_path(store, d, symbol)
        isfile(path) || continue
        dict = read_polygon_spot_prices(path; underlying=symbol)
        merge!(spots, dict)
    end

    if multiplier != 1.0
        for (k, v) in spots
            spots[k] = v * multiplier
        end
    end

    return spots
end

"""
    polygon_parquet_source(symbol; start_date, end_date, entry_time,
                           rate=0.0, div_yield=0.0, spread_lambda=0.0,
                           min_volume=0, store=DEFAULT_STORE)
        -> (source::ParquetDataSource, sched::Vector{DateTime})

Assemble a lazy Polygon `ParquetDataSource` over `[start_date, end_date]` with
one entry per trading day at `entry_time`. Accepts either a single `Time` or a
`Vector{Time}` for multiple entries per day. Closes over entry spots loaded
up front so IV inversion uses the proper spot at each entry. Returns the
source plus its sorted entry schedule.
"""
function polygon_parquet_source(
    symbol::AbstractString;
    start_date::Date,
    end_date::Date,
    entry_time::Union{Time,Vector{Time}},
    rate::Float64=0.0,
    div_yield::Float64=0.0,
    spread_lambda::Float64=0.0,
    min_volume::Real=0,
    store::LocalDataStore=DEFAULT_STORE,
)
    sym = uppercase(String(symbol))
    all_dates = available_polygon_dates(store, sym)
    filtered = filter(d -> start_date <= d <= end_date, all_dates)

    entry_ts = build_entry_timestamps(filtered, entry_time)
    entry_spots = read_polygon_spot_prices_for_timestamps(
        polygon_spot_root(store), entry_ts; symbol=sym,
    )
    source = ParquetDataSource(entry_ts;
        path_for_timestamp = ts -> polygon_options_path(store, Date(ts), sym),
        read_records = (path; where="") -> read_polygon_option_records(
            path, entry_spots;
            where=where, min_volume=min_volume, warn=false,
            spread_lambda=spread_lambda, rate=rate, div_yield=div_yield,
        ),
        spot_root = polygon_spot_root(store),
        spot_symbol = sym,
    )
    return (source=source, sched=available_timestamps(source))
end

"""
    load_surfaces_and_spots(start_date, end_date; kwargs...) -> (surfaces, entry_spots, settlement_spots)

Load volatility surfaces and spot prices for a date range. Returns a tuple of:
- `surfaces::Dict{DateTime,VolatilitySurface}`
- `entry_spots::Dict{DateTime,Float64}`
- `settlement_spots::Dict{DateTime,Float64}`
"""
function load_surfaces_and_spots(
    start_date::Date,
    end_date::Date;
    symbol::String,
    spot_symbol::String="SPY",
    spot_multiplier::Float64=1.0,
    entry_times::Union{Time,Vector{Time}},
    min_volume::Int=0,
    spread_lambda::Float64=0.0,
    expiry_interval::Period=Day(1),
    warn::Bool=false,
    rate::Float64=0.0,
    div_yield::Float64=0.0,
    store::LocalDataStore=DEFAULT_STORE,
)
    println("  Loading dates from $start_date to $end_date...")
    all_dates = available_polygon_dates(store, symbol)
    filtered_dates = filter(d -> d >= start_date && d <= end_date, all_dates)

    if isempty(filtered_dates)
        error("No dates found in range $start_date to $end_date")
    end
    n_times = entry_times isa Vector ? length(entry_times) : 1
    println("  Found $(length(filtered_dates)) trading days x $(n_times) entry times")

    entry_ts = build_entry_timestamps(filtered_dates, entry_times isa Vector ? entry_times : [entry_times])

    # Load entry spots (from spot proxy symbol, scaled)
    entry_spots = read_polygon_spot_prices_for_timestamps(
        polygon_spot_root(store),
        entry_ts;
        symbol=spot_symbol
    )
    if spot_multiplier != 1.0
        for (k, v) in entry_spots
            entry_spots[k] = v * spot_multiplier
        end
    end
    println("  Loaded $(length(entry_spots)) entry spots (via $spot_symbol × $spot_multiplier)")

    # Build surfaces (options from symbol, spots already scaled)
    path_for_ts = ts -> polygon_options_path(store, Date(ts), symbol)
    read_records = (path; where="") -> read_polygon_option_records(
        path,
        entry_spots;
        where=where,
        min_volume=min_volume,
        warn=warn,
        spread_lambda=spread_lambda,
        rate=rate,
        div_yield=div_yield,
    )
    surfaces = build_surfaces_for_timestamps(
        entry_ts;
        path_for_timestamp=path_for_ts,
        read_records=read_records
    )
    println("  Built $(length(surfaces)) surfaces")

    # Load settlement spots (need to compute expiry times first)
    expiry_ts = DateTime[]
    for (ts, surface) in surfaces
        expiries = unique(rec.expiry for rec in surface.records)
        for exp in expiries
            tau = time_to_expiry(exp, ts)
            tau_target = time_to_expiry(ts + expiry_interval, ts)
            if abs(tau - tau_target) < 0.1  # Within ~36 days
                push!(expiry_ts, exp)
            end
        end
    end
    expiry_ts = unique(expiry_ts)

    settlement_spots = read_polygon_spot_prices_for_timestamps(
        polygon_spot_root(store),
        expiry_ts;
        symbol=spot_symbol
    )
    if spot_multiplier != 1.0
        for (k, v) in settlement_spots
            settlement_spots[k] = v * spot_multiplier
        end
    end
    println("  Loaded $(length(settlement_spots)) settlement spots (via $spot_symbol × $spot_multiplier)")

    return surfaces, entry_spots, settlement_spots
end

