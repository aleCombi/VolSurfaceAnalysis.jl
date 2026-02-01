# Option Record Definitions
# Internal record (unified domain model) + Shared utilities

using Dates

# ============================================================================
# Enums
# ============================================================================

"""Option type: Call or Put"""
@enum OptionType Call Put

# ============================================================================
# Underlying Asset
# ============================================================================

"""
    Underlying

Represents an underlying asset. Currently wraps a ticker symbol string,
but can be extended in the future with additional metadata (asset class,
exchange, etc.).

# Examples
```julia
spy = Underlying("SPY")
btc = Underlying("BTC")
ticker(spy)  # "SPY"
```
"""
struct Underlying
    ticker::String
    Underlying(s::AbstractString) = new(uppercase(String(s)))
end

# Convenience: allow string comparison
Base.:(==)(u::Underlying, s::AbstractString) = u.ticker == uppercase(s)
Base.:(==)(s::AbstractString, u::Underlying) = u == s

# Get the ticker string
ticker(u::Underlying) = u.ticker

# For display
Base.show(io::IO, u::Underlying) = print(io, u.ticker)


# ============================================================================
# Internal Record (unified domain model)
# ============================================================================

"""
    OptionRecord

Unified option record used throughout the library.
All data sources convert to this format.

This is the internal representation that pricing, surfaces, and backtesting use.
Source-specific details are normalized away during conversion.

# Fields
- `instrument_name::String`: Identifier string
- `underlying::Underlying`: Underlying asset
- `expiry::DateTime`: Expiration timestamp
- `strike::Float64`: Strike price
- `option_type::OptionType`: Call or Put
- `bid_price::Union{Float64,Missing}`: Bid price (fraction of underlying)
- `ask_price::Union{Float64,Missing}`: Ask price (fraction of underlying)
- `mark_price::Union{Float64,Missing}`: Mark/mid price (fraction of underlying)
- `mark_iv::Union{Float64,Missing}`: Implied volatility (percentage, e.g., 65.0 for 65%)
- `open_interest::Union{Float64,Missing}`: Open interest
- `volume::Union{Float64,Missing}`: Trading volume
- `spot::Float64`: Underlying spot price
- `timestamp::DateTime`: Observation time
"""
struct OptionRecord
    instrument_name::String
    underlying::Underlying
    expiry::DateTime
    strike::Float64
    option_type::OptionType
    bid_price::Union{Float64,Missing}
    ask_price::Union{Float64,Missing}
    mark_price::Union{Float64,Missing}
    mark_iv::Union{Float64,Missing}
    open_interest::Union{Float64,Missing}
    volume::Union{Float64,Missing}
    spot::Float64
    timestamp::DateTime
end

# ============================================================================
# Spot Records (independent spot time series)
# ============================================================================

"""
    SpotPrice

Spot price record used when you only need the underlying price time series
(e.g., settlement at expiry without a full volatility surface).

# Fields
- `underlying::Union{Underlying,Missing}`: Underlying asset, if known
- `price::Float64`: Spot price
- `timestamp::DateTime`: Observation time
"""
struct SpotPrice
    underlying::Union{Underlying,Missing}
    price::Float64
    timestamp::DateTime
end

# ============================================================================
# Parsing Helpers
# ============================================================================

"""
    parse_option_type(s::AbstractString) -> OptionType

Parse a string to an OptionType enum value.
"""
function parse_option_type(s::AbstractString)::OptionType
    s_upper = uppercase(s)
    s_upper == "C" && return Call
    s_upper == "CALL" && return Call
    s_upper == "P" && return Put
    s_upper == "PUT" && return Put
    error("Unknown option type: $s")
end

"""
    to_datetime(val) -> DateTime

Convert various types to DateTime.
Handles DateTime, Date, String, and Unix milliseconds (Integer).
"""
function to_datetime(val)::DateTime
    if val isa DateTime
        return val
    elseif val isa Dates.Date
        return DateTime(val)
    elseif val isa AbstractString
        return DateTime(val)
    elseif val isa Integer
        return unix2datetime(val / 1000)
    else
        error("Cannot convert to DateTime: $(typeof(val))")
    end
end

"""
    to_float_or_missing(val) -> Union{Float64, Missing}

Convert a value to Float64, or missing if null/nothing.
"""
function to_float_or_missing(val)::Union{Float64,Missing}
    ismissing(val) && return missing
    val === nothing && return missing
    return Float64(val)
end

# ============================================================================
# Generic Interface
# ============================================================================

"""
    to_option_record(x) -> OptionRecord

Convert a source record (DeribitQuote, PolygonBar, etc.) to an OptionRecord.
"""
function to_option_record end

"""
    spot_dict(spots; underlying=nothing) -> Dict{DateTime,Float64}

Build a timestamp -> spot map from a vector of SpotPrice records.
If `underlying` is provided, filters to that underlying.
"""
function spot_dict(
    spots::Vector{SpotPrice};
    underlying::Union{Nothing,Underlying,AbstractString}=nothing
)::Dict{DateTime,Float64}
    if underlying === nothing
        return Dict(s.timestamp => s.price for s in spots)
    end

    u = underlying isa Underlying ? underlying : Underlying(underlying)
    return Dict(
        s.timestamp => s.price
        for s in spots
        if s.underlying isa Underlying && s.underlying == u
    )
end
