# Data Types and Import Functions
# Reading and parsing Deribit option chain data

"""
Option type: Call or Put
"""
@enum OptionType Call Put

"""
Underlying asset type
"""
@enum Underlying BTC ETH

"""
    VolRecord

A single volatility surface record from Deribit option chain data.

# Fields
- `instrument_name::String`: Full instrument identifier (e.g., "BTC-27DEC24-50000-C")
- `underlying::Underlying`: The underlying asset (BTC or ETH)
- `expiry::DateTime`: Option expiration date
- `strike::Float64`: Strike price
- `option_type::OptionType`: Call or Put
- `bid_price::Union{Float64,Missing}`: Best bid price
- `ask_price::Union{Float64,Missing}`: Best ask price
- `last_price::Union{Float64,Missing}`: Last traded price
- `mark_price::Union{Float64,Missing}`: Mark price
- `mark_iv::Union{Float64,Missing}`: Mark implied volatility (%)
- `open_interest::Union{Float64,Missing}`: Open interest
- `volume::Union{Float64,Missing}`: Trading volume
- `underlying_price::Float64`: Spot price of underlying
- `timestamp::DateTime`: Observation timestamp
"""
struct VolRecord
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

function parse_underlying(s::AbstractString)::Underlying
    s_upper = uppercase(s)
    s_upper == "BTC" && return BTC
    s_upper == "ETH" && return ETH
    error("Unknown underlying: $s")
end

function parse_option_type(s::AbstractString)::OptionType
    s_upper = uppercase(s)
    s_upper == "C" && return Call
    s_upper == "P" && return Put
    error("Unknown option type: $s")
end

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

function to_float_or_missing(val)::Union{Float64,Missing}
    ismissing(val) && return missing
    val === nothing && return missing
    return Float64(val)
end

"""
    read_vol_records(path::AbstractString) -> Vector{VolRecord}

Read a parquet file containing Deribit volatility data and return a vector of `VolRecord` objects.

# Arguments
- `path`: Path to the parquet file

# Returns
- Vector of `VolRecord` objects
"""
function read_vol_records(path::AbstractString)::Vector{VolRecord}
    df = DataFrame(Parquet2.Dataset(path))
    return dataframe_to_records(df)
end

"""
    dataframe_to_records(df::DataFrame) -> Vector{VolRecord}

Convert a DataFrame to a vector of VolRecord objects.
Normalizes expiry times to 08:00 UTC (Deribit's standard settlement time).
"""
function dataframe_to_records(df::DataFrame)::Vector{VolRecord}
    records = Vector{VolRecord}(undef, nrow(df))

    # Check if 'last' column exists (optional in some data sources)
    has_last = hasproperty(df, :last)

    for (i, row) in enumerate(eachrow(df))
        # Normalize expiry to 08:00 UTC (Deribit settlement time)
        raw_expiry = to_datetime(row.expiry)
        expiry_8am = DateTime(Dates.Date(raw_expiry)) + Dates.Hour(8)

        # last_price is optional
        last_price = has_last ? to_float_or_missing(row.last) : missing

        records[i] = VolRecord(
            string(row.instrument_name),
            parse_underlying(string(row.underlying)),
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

    return records
end

"""
    split_by_timestamp(records::Vector{VolRecord}) -> Dict{DateTime, Vector{VolRecord}}

Split a vector of VolRecords into groups by their timestamp.

# Arguments
- `records`: Vector of VolRecord objects

# Returns
- Dictionary mapping each unique timestamp to a vector of records at that timestamp
"""
function split_by_timestamp(records::Vector{VolRecord})::Dict{DateTime,Vector{VolRecord}}
    result = Dict{DateTime,Vector{VolRecord}}()

    for record in records
        ts = record.timestamp
        if !haskey(result, ts)
            result[ts] = VolRecord[]
        end
        push!(result[ts], record)
    end

    return result
end

"""
    split_by_timestamp(records::Vector{VolRecord}, resolution::Period) -> Dict{DateTime, Vector{VolRecord}}

Split records by timestamp, rounding timestamps to the given resolution (e.g., Minute(1)).

# Arguments
- `records`: Vector of VolRecord objects
- `resolution`: Time period to round timestamps to (e.g., `Minute(1)`, `Hour(1)`)

# Returns
- Dictionary mapping rounded timestamps to vectors of records
"""
function split_by_timestamp(records::Vector{VolRecord}, resolution::Period)::Dict{DateTime,Vector{VolRecord}}
    result = Dict{DateTime,Vector{VolRecord}}()

    for record in records
        ts = floor(record.timestamp, resolution)
        if !haskey(result, ts)
            result[ts] = VolRecord[]
        end
        push!(result[ts], record)
    end

    return result
end
