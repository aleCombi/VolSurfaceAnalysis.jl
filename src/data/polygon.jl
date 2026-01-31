# Polygon Data Ingestion
# Structs + DataFrame Parsing + Internal Conversions

using Dates

# Global flag to suppress repeated warnings
const _POLYGON_WARNINGS_SHOWN = Ref(false)

# ============================================================================
# US DST Helpers (Eastern time, post-2007 rules)
# ============================================================================

function _first_sunday(year::Int, month::Int)::Date
    d = Date(year, month, 1)
    offset = (7 - Dates.dayofweek(d)) % 7
    return d + Day(offset)
end

function _second_sunday_in_march(year::Int)::Date
    return _first_sunday(year, 3) + Day(7)
end

function _first_sunday_in_november(year::Int)::Date
    return _first_sunday(year, 11)
end

function _is_us_dst(date::Date)::Bool
    year = Dates.year(date)
    dst_start = _second_sunday_in_march(year)
    dst_end = _first_sunday_in_november(year)
    return date >= dst_start && date < dst_end
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
    # EDT (Mar-Nov): 20:00 UTC | EST (Nov-Mar): 21:00 UTC
    # Assumption: apply US DST rules (post-2007) for the expiry date.
    expiry_date = Date(year, month, day)
    expiry_hour = _is_us_dst(expiry_date) ? 20 : 21
    expiry = DateTime(expiry_date) + Hour(expiry_hour)

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

    return OptionRecord(
        bar.ticker,
        bar.underlying,
        bar.expiry,
        bar.strike,
        bar.option_type,
        bid_frac,
        ask_frac,
        mark_frac,
        missing,
        missing,
        Float64(bar.volume),
        spot,
        bar.timestamp
    )
end
