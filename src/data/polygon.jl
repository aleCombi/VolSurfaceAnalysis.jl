# Polygon Data Ingestion
# Structs + DataFrame Parsing + Internal Conversions

using Dates
using DataFrames
using TimeZones

# Global flag to suppress repeated warnings
const _POLYGON_WARNINGS_SHOWN = Ref(false)

# US Eastern timezone (handles DST automatically)
const TZ_ET = tz"America/New_York"

# ============================================================================
# Timezone Conversion Helpers
# ============================================================================

"""
    et_to_utc(date::Date, et_time::Time) -> DateTime

Convert an Eastern Time (ET) date and time to a UTC DateTime.
Automatically handles EDT/EST transitions using TimeZones.jl.

# Example
```julia
et_to_utc(Date(2024, 7, 15), Time(10, 0))  # Summer (EDT): 14:00 UTC
et_to_utc(Date(2024, 12, 15), Time(10, 0)) # Winter (EST): 15:00 UTC
```
"""
function et_to_utc(date::Date, et_time::Time)::DateTime
    local_dt = DateTime(date) + Hour(Dates.hour(et_time)) + Minute(Dates.minute(et_time))
    zdt = ZonedDateTime(local_dt, TZ_ET)
    return DateTime(zdt, UTC)
end

"""
    et_to_utc(dt::DateTime) -> DateTime

Convert an Eastern Time DateTime to UTC.
"""
function et_to_utc(dt::DateTime)::DateTime
    zdt = ZonedDateTime(dt, TZ_ET)
    return DateTime(zdt, UTC)
end

# ============================================================================
# Ticker Parsing
# ============================================================================

"""
    parse_polygon_ticker(ticker::String) -> (underlying, expiry, option_type, strike)

Parse Polygon options ticker format: O:UNDERLYING[YY]MMDD[C/P][STRIKE_PADDED]

Expiry date is encoded in the ticker; the time-of-day is assumed to be
US equity market close (4 PM ET), mapped to UTC using US DST rules.

# Example
```julia
parse_polygon_ticker("O:SPY240129C00406000")
# → ("SPY", DateTime(2024, 1, 29, 21, 0, 0), Call, 406.0)
```

# Returns
- Tuple of (underlying::String, expiry::DateTime, option_type::OptionType, strike::Float64)
"""
function parse_polygon_ticker(ticker::String)::Tuple{String, DateTime, OptionType, Float64}
    # Pattern: O:UNDERLYING[YY]MMDD[C/P][STRIKE]
    pattern = r"^O:([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$"
    m = match(pattern, ticker)

    if m === nothing
        error("Invalid Polygon ticker format: $ticker")
    end

    underlying = m[1]
    year = 2000 + parse(Int, m[2])  # YY → YYYY
    month = parse(Int, m[3])
    day = parse(Int, m[4])
    option_type_str = m[5]
    strike_padded = parse(Int, m[6])

    # US equity options expire at market close: 4 PM ET
    # Use TimeZones.jl to handle EDT/EST automatically
    expiry_date = Date(year, month, day)
    expiry = et_to_utc(expiry_date, Time(16, 0))  # 4 PM ET → UTC

    # Option type
    option_type = option_type_str == "C" ? Call : Put

    # Strike: padded integer in thousandths
    # E.g., 00406000 → 406.000
    strike = strike_padded / 1000.0

    return (underlying, expiry, option_type, strike)
end

# ============================================================================
# Source Records (match parquet/file schemas exactly)
# ============================================================================

"""
    PolygonBar

A single OHLC minute bar from Polygon options trade data.
Matches the Polygon parquet schema exactly.

# Fields
- `ticker::String`: Polygon ticker (e.g., "O:SPY240129C00406000")
- `underlying::Underlying`: Underlying asset
- `expiry::DateTime`: Option expiration (normalized to 20:00/21:00 UTC using US DST)
- `strike::Float64`: Strike price
- `option_type::OptionType`: Call or Put
- `open::Float64`: First trade price in minute
- `high::Float64`: Highest trade price in minute
- `low::Float64`: Lowest trade price in minute
- `close::Float64`: Last trade price in minute
- `volume::Int64`: Number of contracts traded
- `transactions::Int64`: Number of separate trades
- `timestamp::DateTime`: Minute bar timestamp
"""
struct PolygonBar
    ticker::String
    underlying::Underlying
    expiry::DateTime
    strike::Float64
    option_type::OptionType
    open::Float64
    high::Float64
    low::Float64
    close::Float64
    volume::Int64
    transactions::Int64
    timestamp::DateTime
end

# ============================================================================
# DataFrame Conversions (for DuckDB query results)
# ============================================================================

"""
    PolygonBar(row) -> PolygonBar

Construct a PolygonBar from a DataFrame row (e.g., DuckDB query result).
Parses the ticker to extract underlying, expiry, option_type, strike.

Expected columns: ticker, open, high, low, close, volume, transactions, timestamp
"""
function PolygonBar(row)::PolygonBar
    ticker_str = string(row.ticker)
    underlying, expiry, option_type, strike = parse_polygon_ticker(ticker_str)

    return PolygonBar(
        ticker_str,
        Underlying(underlying),
        expiry,
        strike,
        option_type,
        Float64(row.open),
        Float64(row.high),
        Float64(row.low),
        Float64(row.close),
        Int64(row.volume),
        Int64(row.transactions),
        DateTime(row.timestamp)
    )
end

# ============================================================================
# Conversion to Internal Format
# ============================================================================

"""
    to_option_record(bar::PolygonBar, spot::Float64; warn::Bool=true) -> OptionRecord

Convert a PolygonBar to the unified OptionRecord format.

# IMPORTANT: Synthetic Bid/Ask
Polygon data contains OHLC trade prices, NOT bid/ask quotes.
This function uses a CONSERVATIVE approximation:
- bid_price = low / spot (worst observed buy price)
- ask_price = high / spot (worst observed sell price)
- mark_price = close / spot

This is pessimistic and may underestimate strategy performance.

# IMPORTANT: Implied Volatility
Polygon trade data does not include IV. We infer mark_iv from mark_price
using Black-76 with spot as forward and r=0.0. Stored as a percentage.

# Arguments
- `bar`: The PolygonBar to convert
- `spot`: Spot price of underlying (required for price normalization)
- `warn`: Show warning about synthetic spreads (default: true, shown once)
"""
function to_option_record(bar::PolygonBar, spot::Float64; warn::Bool=true)::OptionRecord
    # Show warning once
    if warn && !_POLYGON_WARNINGS_SHOWN[]
        _POLYGON_WARNINGS_SHOWN[] = true
        @warn """
        USING SYNTHETIC BID/ASK SPREADS (CONSERVATIVE)

        Polygon data contains trade prices (OHLC), NOT market quotes.
        Converting with PESSIMISTIC approximation:
          - bid_price = low  (worst buy price observed in minute)
          - ask_price = high (worst sell price observed in minute)

        This may UNDERESTIMATE strategy performance.
        (This warning will only be shown once per session)
        """
    end

    # CONSERVATIVE bid/ask from OHLC (as fraction of spot)
    bid_frac = bar.low / spot
    ask_frac = bar.high / spot
    mark_frac = bar.close / spot

    # Infer IV from mark price (if possible)
    T = time_to_expiry(bar.expiry, bar.timestamp)
    mark_iv = if T <= 0.0
        missing
    else
        iv = price_to_iv(mark_frac, spot, bar.strike, T, bar.option_type)
        isnan(iv) ? missing : iv * 100.0
    end

    return OptionRecord(
        bar.ticker,
        bar.underlying,
        bar.expiry,
        bar.strike,
        bar.option_type,
        bid_frac,
        ask_frac,
        mark_frac,
        mark_iv,
        missing,
        Float64(bar.volume),
        spot,
        bar.timestamp
    )
end

# ============================================================================
# Spot Parquet Readers
# ============================================================================

function _find_parquet_files(path::AbstractString)::Vector{String}
    files = String[]
    if isfile(path) && endswith(lowercase(String(path)), ".parquet")
        push!(files, String(path))
    elseif isdir(path)
        for (root, _, filenames) in walkdir(path)
            for f in filenames
                endswith(f, ".parquet") && push!(files, joinpath(root, f))
            end
        end
    end
    sort!(files)
    return files
end

"""
    read_polygon_spot_parquet(path; where="", price_col=:close, ts_col=:timestamp, underlying=nothing)
        -> Vector{SpotPrice}

Load Polygon spot data from a parquet file using DuckDB.
By default, uses the `close` column as the spot price.
"""
function read_polygon_spot_parquet(
    path::AbstractString;
    where::AbstractString="",
    price_col::Symbol=:close,
    ts_col::Symbol=:timestamp,
    underlying::Union{Nothing,Underlying,AbstractString}=nothing
)::Vector{SpotPrice}
    cols = "$(String(ts_col)), $(String(price_col))"
    df = _duckdb_parquet_df(path; columns=cols, where=where)

    u = underlying === nothing ? missing :
        (underlying isa Underlying ? underlying : Underlying(underlying))

    spots = SpotPrice[]
    for row in eachrow(df)
        price = to_float_or_missing(getproperty(row, price_col))
        ismissing(price) && continue
        ts = to_datetime(getproperty(row, ts_col))
        push!(spots, SpotPrice(u, price, ts))
    end

    return spots
end

"""
    read_polygon_spot_prices(path; where="", price_col=:close, ts_col=:timestamp, underlying=nothing)
        -> Dict{DateTime,Float64}

Load Polygon spot data and return a timestamp -> spot dictionary.
"""
function read_polygon_spot_prices(
    path::AbstractString;
    where::AbstractString="",
    price_col::Symbol=:close,
    ts_col::Symbol=:timestamp,
    underlying::Union{Nothing,Underlying,AbstractString}=nothing
)::Dict{DateTime,Float64}
    spots = read_polygon_spot_parquet(
        path;
        where=where,
        price_col=price_col,
        ts_col=ts_col,
        underlying=underlying
    )
    return spot_dict(spots)
end

"""
    read_polygon_spot_prices_for_timestamps(root, timestamps; symbol, price_col=:close, ts_col=:timestamp)
        -> Dict{DateTime,Float64}

Load Polygon spot data only for a specific set of timestamps, grouped by date.
This avoids scanning entire daily files when only a few minutes are needed.
"""
function read_polygon_spot_prices_for_timestamps(
    root::AbstractString,
    timestamps::Vector{DateTime};
    symbol::Union{Underlying,AbstractString},
    price_col::Symbol=:close,
    ts_col::Symbol=:timestamp
)::Dict{DateTime,Float64}
    by_date = Dict{Date, Vector{DateTime}}()
    for ts in timestamps
        d = Date(ts)
        push!(get!(by_date, d, DateTime[]), ts)
    end

    sym = symbol isa Underlying ? ticker(symbol) : uppercase(String(symbol))
    spots = Dict{DateTime,Float64}()

    for (d, ts_list) in sort(collect(by_date); by=first)
        date_str = Dates.format(d, "yyyy-mm-dd")
        path = joinpath(root, "date=$date_str", "symbol=$sym", "data.parquet")
        isfile(path) || continue

        ts_unique = unique(ts_list)
        ts_strs = [Dates.format(ts, "yyyy-mm-dd HH:MM:SS") for ts in ts_unique]
        in_clause = join(["'$s'" for s in ts_strs], ", ")
        where = "timestamp IN ($in_clause)"

        dict = read_polygon_spot_prices(
            path;
            where=where,
            price_col=price_col,
            ts_col=ts_col,
            underlying=sym
        )
        merge!(spots, dict)
    end

    return spots
end

"""
    read_polygon_spot_prices_dir(root; symbol=nothing, where="", price_col=:close, ts_col=:timestamp)
        -> Dict{DateTime,Float64}

Load Polygon spot data from all parquet files under a root directory.
Optionally filters to `symbol=...` subfolders.
"""
function read_polygon_spot_prices_dir(
    root::AbstractString;
    symbol::Union{Nothing,Underlying,AbstractString}=nothing,
    where::AbstractString="",
    price_col::Symbol=:close,
    ts_col::Symbol=:timestamp
)::Dict{DateTime,Float64}
    files = _find_parquet_files(root)

    if symbol !== nothing
        sym = symbol isa Underlying ? ticker(symbol) : uppercase(String(symbol))
        needle = "symbol=$(sym)"
        files = filter(f -> occursin(lowercase(needle), lowercase(f)), files)
    end

    spots = Dict{DateTime,Float64}()
    for f in files
        dict = read_polygon_spot_prices(
            f;
            where=where,
            price_col=price_col,
            ts_col=ts_col,
            underlying=symbol
        )
        merge!(spots, dict)
    end

    return spots
end

# ============================================================================
# DuckDB Parquet Readers
# ============================================================================

"""
    read_polygon_parquet(path; where="", min_volume=0) -> Vector{PolygonBar}

Load Polygon minute bars from a parquet file using DuckDB.
"""
function read_polygon_parquet(
    path::AbstractString;
    where::AbstractString="",
    min_volume::Int=0
)::Vector{PolygonBar}
    cols = "ticker, open, high, low, close, volume, transactions, timestamp"

    conds = String[]
    !isempty(strip(where)) && push!(conds, "($where)")
    min_volume > 0 && push!(conds, "volume >= $min_volume")
    where_clause = isempty(conds) ? "" : join(conds, " AND ")

    df = _duckdb_parquet_df(path; columns=cols, where=where_clause)
    return [PolygonBar(row) for row in eachrow(df)]
end

"""
    read_polygon_option_records(path, spot; where="", min_volume=0, warn=true) -> Vector{OptionRecord}

Load Polygon minute bars from parquet and convert to OptionRecord using a constant spot.
"""
function read_polygon_option_records(
    path::AbstractString,
    spot::Float64;
    where::AbstractString="",
    min_volume::Int=0,
    warn::Bool=true
)::Vector{OptionRecord}
    bars = read_polygon_parquet(path; where=where, min_volume=min_volume)
    return [to_option_record(bar, spot; warn=warn) for bar in bars]
end

"""
    read_polygon_option_records(path, spot_by_ts; where="", min_volume=0, warn=true) -> Vector{OptionRecord}

Load Polygon minute bars from parquet and convert to OptionRecord using a spot map.
Records with missing spot are skipped.
"""
function read_polygon_option_records(
    path::AbstractString,
    spot_by_ts::AbstractDict{DateTime,Float64};
    where::AbstractString="",
    min_volume::Int=0,
    warn::Bool=true
)::Vector{OptionRecord}
    bars = read_polygon_parquet(path; where=where, min_volume=min_volume)
    records = OptionRecord[]
    for bar in bars
        spot = get(spot_by_ts, bar.timestamp, missing)
        ismissing(spot) && continue
        push!(records, to_option_record(bar, spot; warn=warn))
    end
    return records
end
