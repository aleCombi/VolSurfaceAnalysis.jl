# Deribit Data Ingestion
# Structs + DataFrame Parsing + Internal Conversions

using Dates
using DataFrames

# ============================================================================
# Source Records (match parquet/file schemas exactly)
# ============================================================================

"""
    DeribitQuote

A single option quote from Deribit parquet data.
Matches the Deribit data schema exactly.

# Fields
- `instrument_name::String`: Full instrument identifier (e.g., "BTC-27DEC24-50000-C")
- `underlying::Underlying`: The underlying asset
- `expiry::DateTime`: Option expiration date (normalized to 08:00 UTC)
- `strike::Float64`: Strike price
- `option_type::OptionType`: Call or Put
- `bid_price::Union{Float64,Missing}`: Best bid price (fraction of underlying)
- `ask_price::Union{Float64,Missing}`: Best ask price (fraction of underlying)
- `last_price::Union{Float64,Missing}`: Last traded price
- `mark_price::Union{Float64,Missing}`: Mark price
- `mark_iv::Union{Float64,Missing}`: Mark implied volatility (percentage, e.g., 65.0)
- `open_interest::Union{Float64,Missing}`: Open interest
- `volume::Union{Float64,Missing}`: Trading volume
- `underlying_price::Float64`: Spot price of underlying
- `timestamp::DateTime`: Observation timestamp
"""
struct DeribitQuote
    instrument_name::String
    underlying::Underlying
    expiry::DateTime
    strike::Float64
    option_type::OptionType
    bid_price::Union{Float64,Missing}
    ask_price::Union{Float64,Missing}
    last_price::Union{Float64,Missing}
    mark_price::Union{Float64,Missing}
    mark_iv::Union{Float64,Missing}
    open_interest::Union{Float64,Missing}
    volume::Union{Float64,Missing}
    underlying_price::Float64
    timestamp::DateTime
end

"""
    DeribitDelivery

A delivery/settlement price record from Deribit.

# Fields
- `underlying::Underlying`: The underlying asset
- `delivery_price::Float64`: The 30-minute TWAP settlement price
- `timestamp::DateTime`: Delivery timestamp (08:00 UTC on expiry day)
"""
struct DeribitDelivery
    underlying::Underlying
    delivery_price::Float64
    timestamp::DateTime
end

# ============================================================================
# DataFrame Conversions (for DuckDB query results)
# ============================================================================

"""
    DeribitQuote(row) -> DeribitQuote

Construct a DeribitQuote from a DataFrame row (e.g., DuckDB query result).
Normalizes expiry to 08:00 UTC.

Expected columns: instrument_name, underlying, expiry, strike, option_type,
bid_price, ask_price, last (optional), mark_price, mark_iv, open_interest,
volume, underlying_price, ts
"""
function DeribitQuote(row)::DeribitQuote
    # Normalize expiry to 08:00 UTC (Deribit settlement time)
    raw_expiry = to_datetime(row.expiry)
    expiry_8am = DateTime(Dates.Date(raw_expiry)) + Dates.Hour(8)

    # last_price is optional
    last_price = hasproperty(row, :last) ? to_float_or_missing(row.last) : missing

    return DeribitQuote(
        string(row.instrument_name),
        Underlying(string(row.underlying)),
        expiry_8am,
        Float64(row.strike),
        parse_option_type(string(row.option_type)),
        to_float_or_missing(row.bid_price),
        to_float_or_missing(row.ask_price),
        last_price,
        to_float_or_missing(row.mark_price),
        to_float_or_missing(row.mark_iv),
        to_float_or_missing(row.open_interest),
        to_float_or_missing(row.volume),
        Float64(row.underlying_price),
        to_datetime(row.ts)
    )
end

# ============================================================================
# Conversion to Internal Format
# ============================================================================

"""
    to_option_record(dq::DeribitQuote) -> OptionRecord

Convert a DeribitQuote to the unified OptionRecord format.

# Example
```julia
record = to_option_record(quote)
records = to_option_record.(quotes)  # broadcast for vectors
```
"""
function to_option_record(dq::DeribitQuote)::OptionRecord
    return OptionRecord(
        dq.instrument_name,
        dq.underlying,
        dq.expiry,
        dq.strike,
        dq.option_type,
        dq.bid_price,
        dq.ask_price,
        dq.mark_price,
        dq.mark_iv,
        dq.open_interest,
        dq.volume,
        dq.underlying_price,
        dq.timestamp
    )
end

# ============================================================================
# DuckDB Parquet Readers
# ============================================================================

"""
    read_deribit_parquet(path; where="") -> Vector{DeribitQuote}

Load Deribit quotes from a parquet file using DuckDB.
"""
function read_deribit_parquet(
    path::AbstractString;
    where::AbstractString=""
)::Vector{DeribitQuote}
    df = _duckdb_parquet_df(path; where=where)
    return [DeribitQuote(row) for row in eachrow(df)]
end

"""
    read_deribit_option_records(path; where="") -> Vector{OptionRecord}

Load Deribit quotes from parquet and convert to OptionRecord.
"""
function read_deribit_option_records(
    path::AbstractString;
    where::AbstractString=""
)::Vector{OptionRecord}
    quotes = read_deribit_parquet(path; where=where)
    return to_option_record.(quotes)
end
