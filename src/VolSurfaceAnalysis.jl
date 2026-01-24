module VolSurfaceAnalysis

using Dates
using DataFrames
using Parquet2
using Distributions: Normal, cdf, pdf

export VolRecord, OptionType, Underlying
export read_vol_records, split_by_timestamp
export VolPoint, VolatilitySurface, build_surface
export TermStructure, atm_term_structure
export black76_price, vol_to_price, black76_vega, price_to_iv, bid_iv, ask_iv, time_to_expiry

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
function split_by_timestamp(records::Vector{VolRecord})::Dict{DateTime, Vector{VolRecord}}
    result = Dict{DateTime, Vector{VolRecord}}()

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
function split_by_timestamp(records::Vector{VolRecord}, resolution::Period)::Dict{DateTime, Vector{VolRecord}}
    result = Dict{DateTime, Vector{VolRecord}}()

    for record in records
        ts = floor(record.timestamp, resolution)
        if !haskey(result, ts)
            result[ts] = VolRecord[]
        end
        push!(result[ts], record)
    end

    return result
end

# ============================================================================
# Volatility Surface
# ============================================================================

const DAYS_PER_YEAR = 365.25

"""
    VolPoint

A single point on the volatility surface in (log-moneyness, time-to-expiry) coordinates.

# Fields
- `log_moneyness::Float64`: log(K/S), negative for ITM calls, positive for ITM puts
- `τ::Float64`: time to expiry in years
- `vol::Float64`: mark implied volatility (decimal)
- `bid_vol::Union{Float64, Missing}`: bid implied volatility (decimal), missing if no bid
- `ask_vol::Union{Float64, Missing}`: ask implied volatility (decimal), missing if no ask
"""
struct VolPoint
    log_moneyness::Float64
    τ::Float64
    vol::Float64
    bid_vol::Union{Float64, Missing}
    ask_vol::Union{Float64, Missing}
end

"""
    VolatilitySurface

A volatility surface at a single point in time.

# Fields
- `spot::Float64`: spot price of underlying
- `timestamp::DateTime`: observation timestamp
- `underlying::Underlying`: the underlying asset
- `points::Vector{VolPoint}`: the vol surface points
"""
struct VolatilitySurface
    spot::Float64
    timestamp::DateTime
    underlying::Underlying
    points::Vector{VolPoint}
end

"""
    time_to_expiry(expiry::DateTime, now::DateTime) -> Float64

Calculate time to expiry in years.
"""
function time_to_expiry(expiry::DateTime, now::DateTime)::Float64
    ms_diff = Dates.value(expiry - now)
    days = ms_diff / (1000 * 60 * 60 * 24)
    return days / DAYS_PER_YEAR
end

"""
    deribit_time_to_expiry(expiry::DateTime, now::DateTime) -> Float64

Calculate time to expiry in years using Deribit's convention.
Deribit options expire at 08:00 UTC on the expiry date.
"""
function deribit_time_to_expiry(expiry::DateTime, now::DateTime)::Float64
    # Deribit options expire at 08:00 UTC on the expiry date
    expiry_8am = DateTime(Dates.Date(expiry)) + Dates.Hour(8)
    ms_diff = Dates.value(expiry_8am - now)
    days = ms_diff / (1000 * 60 * 60 * 24)
    return days / DAYS_PER_YEAR
end

# ============================================================================
# Black-76 Pricing (Vol → Price)
# ============================================================================

"""
    call_put_sign(option_type::OptionType) -> Int

Return +1 for Call, -1 for Put. Used for unified Black formula.
"""
call_put_sign(option_type::OptionType)::Int = option_type == Call ? 1 : -1

"""
    black76_price(F, K, T, σ, option_type; r=0.0) -> Float64

Calculate the Black-76 option price.

The Black-76 model prices European options on futures/forwards.

# Arguments
- `F::Float64`: Forward/futures price (underlying_price from Deribit)
- `K::Float64`: Strike price
- `T::Float64`: Time to expiry in years
- `σ::Float64`: Implied volatility as decimal (e.g., 0.65 for 65%)
- `option_type::OptionType`: Call or Put
- `r::Float64`: Risk-free interest rate for discounting (default: 0.0)

# Returns
- Option price in absolute terms (same units as F and K)

# Formula
```
Price = D * cp * (F * N(cp*d₁) - K * N(cp*d₂))

where:
  cp = +1 for Call, -1 for Put
  D  = e^(-rT) (discount factor)
  d₁ = [ln(F/K) + (σ²/2)T] / (σ√T)
  d₂ = d₁ - σ√T
  N(·) = standard normal CDF
```
"""
function black76_price(F::Float64, K::Float64, T::Float64, σ::Float64, 
                       option_type::OptionType; r::Float64=0.0)::Float64
    cp = call_put_sign(option_type)
    D = exp(-r * T)  # Discount factor
    
    # Handle edge cases
    if T <= 0.0
        # At expiry: intrinsic value
        return max(cp * (F - K), 0.0)
    end
    
    if σ <= 0.0
        # Zero vol: deterministic forward
        return D * max(cp * (F - K), 0.0)
    end
    
    # Black-76 formula (unified for call/put)
    sqrtT = sqrt(T)
    d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * sqrtT)
    d2 = d1 - σ * sqrtT
    
    N = Normal()
    return D * cp * (F * cdf(N, cp * d1) - K * cdf(N, cp * d2))
end

"""
    vol_to_price(σ, F, K, T, option_type; r=0.0) -> Float64

Convert implied volatility to option price as a fraction of the underlying.
This matches Deribit's price quotation convention.

# Arguments
- `σ::Float64`: Implied volatility as decimal (e.g., 0.65 for 65%)
- `F::Float64`: Forward/futures price (underlying_price)
- `K::Float64`: Strike price  
- `T::Float64`: Time to expiry in years
- `option_type::OptionType`: Call or Put
- `r::Float64`: Interest rate for discounting (default: 0.0)

# Returns
- Option price as fraction of underlying (e.g., 0.05 for 5% of underlying)
"""
function vol_to_price(σ::Float64, F::Float64, K::Float64, T::Float64,
                      option_type::OptionType; r::Float64=0.0)::Float64
    abs_price = black76_price(F, K, T, σ, option_type; r=r)
    return abs_price / F
end

"""
    vol_to_price(record::VolRecord) -> Union{Float64, Missing}

Convert a VolRecord's mark_iv to the corresponding option price (as fraction of underlying).
Returns missing if mark_iv is missing.

Uses the record's underlying_price as forward, and interest_rate if available.
"""
function vol_to_price(record::VolRecord)::Union{Float64, Missing}
    ismissing(record.mark_iv) && return missing
    
    σ = record.mark_iv / 100.0  # Convert from percentage to decimal
    F = record.underlying_price
    K = record.strike
    T = time_to_expiry(record.expiry, record.timestamp)
    
    # Use interest_rate from record if available (stored as decimal, e.g., 0.05 for 5%)
    r = 0.0  # Default to 0 if not available
    
    if T <= 0.0
        # Expired option: intrinsic value
        if record.option_type == Call
            return max(F - K, 0.0) / F
        else
            return max(K - F, 0.0) / F
        end
    end
    
    return vol_to_price(σ, F, K, T, record.option_type; r=r)
end

"""
    black76_vega(F, K, T, σ, option_type; r=0.0) -> Float64

Calculate the Black-76 vega (derivative of price w.r.t. volatility).
Vega is the same for calls and puts.

# Returns
- Vega in absolute terms (∂Price/∂σ)
"""
function black76_vega(F::Float64, K::Float64, T::Float64, σ::Float64,
                      option_type::OptionType; r::Float64=0.0)::Float64
    if T <= 0.0 || σ <= 0.0
        return 0.0
    end
    
    sqrtT = sqrt(T)
    d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * sqrtT)
    
    D = exp(-r * T)
    N = Normal()
    
    # Vega = D * F * √T * φ(d1), where φ is standard normal PDF
    return D * F * sqrtT * pdf(N, d1)
end

"""
    price_to_iv(price, F, K, T, option_type; r=0.0, tol=1e-8, max_iter=100) -> Float64

Convert an option price (as fraction of underlying) to implied volatility.
Uses Newton-Raphson iteration with vega.

# Arguments
- `price::Float64`: Option price as fraction of underlying
- `F::Float64`: Forward/futures price (underlying_price)
- `K::Float64`: Strike price
- `T::Float64`: Time to expiry in years
- `option_type::OptionType`: Call or Put
- `r::Float64`: Interest rate for discounting (default: 0.0)
- `tol::Float64`: Convergence tolerance (default: 1e-8)
- `max_iter::Int`: Maximum iterations (default: 100)

# Returns
- Implied volatility as decimal (e.g., 0.65 for 65%)
- Returns NaN if no solution found
"""
function price_to_iv(price::Float64, F::Float64, K::Float64, T::Float64,
                     option_type::OptionType; r::Float64=0.0, 
                     tol::Float64=1e-8, max_iter::Int=100)::Float64
    # Handle edge cases
    if T <= 0.0
        return NaN
    end
    
    cp = call_put_sign(option_type)
    D = exp(-r * T)
    
    # Intrinsic value check
    intrinsic = max(cp * (F - K), 0.0) / F
    if price <= intrinsic + tol
        return 0.0  # Zero vol (at or below intrinsic)
    end
    
    # Upper bound check (price can't exceed forward for calls, or strike for puts)
    max_price = option_type == Call ? 1.0 : K / F
    if price >= max_price * D - tol
        return NaN  # Price too high
    end
    
    # Initial guess using Brenner-Subrahmanyam approximation for ATM
    σ = sqrt(2π / T) * price  # Good starting point
    σ = clamp(σ, 0.01, 5.0)   # Keep in reasonable range
    
    # Newton-Raphson iteration
    for _ in 1:max_iter
        computed_price = vol_to_price(σ, F, K, T, option_type; r=r)
        diff = computed_price - price
        
        if abs(diff) < tol
            return σ
        end
        
        # Vega (as fraction of F)
        vega = black76_vega(F, K, T, σ, option_type; r=r) / F
        
        if abs(vega) < 1e-15
            # Vega too small, switch to bisection
            break
        end
        
        # Newton step with damping
        step = diff / vega
        σ_new = σ - step
        
        # Keep σ positive and bounded
        σ_new = clamp(σ_new, 0.001, 10.0)
        
        # Check for convergence
        if abs(σ_new - σ) < tol
            return σ_new
        end
        
        σ = σ_new
    end
    
    # Fallback: bisection if Newton didn't converge
    σ_low, σ_high = 0.001, 5.0
    for _ in 1:100
        σ_mid = (σ_low + σ_high) / 2
        computed_price = vol_to_price(σ_mid, F, K, T, option_type; r=r)
        
        if abs(computed_price - price) < tol
            return σ_mid
        end
        
        if computed_price < price
            σ_low = σ_mid
        else
            σ_high = σ_mid
        end
        
        if σ_high - σ_low < tol
            return σ_mid
        end
    end
    
    return NaN  # Failed to converge
end

"""
    price_to_iv(record::VolRecord) -> Union{Float64, Missing}

Convert a VolRecord's mark_price to implied volatility.
Returns missing if mark_price is missing.

Uses Deribit's 08:00 UTC expiry convention for time calculation.
"""
function price_to_iv(record::VolRecord)::Union{Float64, Missing}
    ismissing(record.mark_price) && return missing
    
    price = record.mark_price
    F = record.underlying_price
    K = record.strike
    T = time_to_expiry(record.expiry, record.timestamp)
    
    if T <= 0.0
        return missing
    end
    
    σ = price_to_iv(price, F, K, T, record.option_type)
    return isnan(σ) ? missing : σ
end

"""
    bid_iv(record::VolRecord) -> Union{Float64, Missing}

Compute implied volatility from a VolRecord's bid_price.
Returns missing if bid_price is missing or IV cannot be computed.
"""
function bid_iv(record::VolRecord)::Union{Float64, Missing}
    ismissing(record.bid_price) && return missing
    
    price = record.bid_price
    F = record.underlying_price
    K = record.strike
    T = time_to_expiry(record.expiry, record.timestamp)
    
    if T <= 0.0
        return missing
    end
    
    σ = price_to_iv(price, F, K, T, record.option_type)
    return isnan(σ) ? missing : σ
end

"""
    ask_iv(record::VolRecord) -> Union{Float64, Missing}

Compute implied volatility from a VolRecord's ask_price.
Returns missing if ask_price is missing or IV cannot be computed.
"""
function ask_iv(record::VolRecord)::Union{Float64, Missing}
    ismissing(record.ask_price) && return missing
    
    price = record.ask_price
    F = record.underlying_price
    K = record.strike
    T = time_to_expiry(record.expiry, record.timestamp)
    
    if T <= 0.0
        return missing
    end
    
    σ = price_to_iv(price, F, K, T, record.option_type)
    return isnan(σ) ? missing : σ
end

"""
    is_itm_preferred(log_moneyness::Float64, option_type::OptionType) -> Bool

Check if this option type is the ITM (preferred) one for the given moneyness.
- Call is ITM when S > K, i.e., log(K/S) < 0
- Put is ITM when S < K, i.e., log(K/S) > 0
"""
function is_itm_preferred(log_moneyness::Float64, option_type::OptionType)::Bool
    if log_moneyness < 0
        return option_type == Call
    else
        return option_type == Put
    end
end

"""
    build_surface(records::Vector{VolRecord}) -> VolatilitySurface

Build a VolatilitySurface from a vector of VolRecords (assumed to be from a single timestamp).

For each (strike, expiry) pair, chooses the ITM option. At ATM, uses highest volume as tiebreaker.
Records with missing mark_iv are skipped.
"""
function build_surface(records::Vector{VolRecord})::VolatilitySurface
    isempty(records) && error("Cannot build surface from empty records")

    # Use first record for metadata (assuming all from same timestamp/underlying)
    first_rec = records[1]
    spot = first_rec.underlying_price
    timestamp = first_rec.timestamp
    underlying = first_rec.underlying

    # Group by (strike, expiry)
    grouped = Dict{Tuple{Float64, DateTime}, Vector{VolRecord}}()
    for rec in records
        ismissing(rec.mark_iv) && continue
        key = (rec.strike, rec.expiry)
        if !haskey(grouped, key)
            grouped[key] = VolRecord[]
        end
        push!(grouped[key], rec)
    end

    # Build vol points
    points = VolPoint[]

    for ((strike, expiry), recs) in grouped
        log_m = log(strike / spot)
        τ = time_to_expiry(expiry, timestamp)

        # Skip expired or near-expired options
        τ <= 0 && continue

        # Choose best record: prefer ITM, tiebreak by volume
        best = nothing
        best_score = -Inf

        for rec in recs
            itm = is_itm_preferred(log_m, rec.option_type)
            vol = coalesce(rec.volume, 0.0)
            # ITM gets priority (score +1e9), then volume
            score = (itm ? 1e9 : 0.0) + vol
            if score > best_score
                best_score = score
                best = rec
            end
        end

        if best !== nothing && !ismissing(best.mark_iv)
            vol = best.mark_iv / 100.0
            
            # Compute bid/ask IVs from prices
            bvol = bid_iv(best)
            avol = ask_iv(best)
            
            push!(points, VolPoint(log_m, τ, vol, bvol, avol))
        end
    end

    # Sort by τ, then by log_moneyness
    sort!(points, by = p -> (p.τ, p.log_moneyness))

    return VolatilitySurface(spot, timestamp, underlying, points)
end

# ============================================================================
# Term Structure (slice of vol surface at fixed moneyness)
# ============================================================================

"""
    TermStructure

A volatility term structure at a fixed moneyness level (typically ATM).
This is a 1D slice of the volatility surface: τ → vol.

# Fields
- `spot::Float64`: spot price of underlying
- `timestamp::DateTime`: observation timestamp
- `underlying::Underlying`: the underlying asset
- `moneyness::Float64`: the log-moneyness level (0.0 for ATM)
- `tenors::Vector{Float64}`: time to expiry in years
- `vols::Vector{Float64}`: implied volatilities (decimal)
"""
struct TermStructure
    spot::Float64
    timestamp::DateTime
    underlying::Underlying
    moneyness::Float64
    tenors::Vector{Float64}
    mark_vols::Vector{Union{Float64, Missing}}
    bid_vols::Vector{Union{Float64, Missing}}
    ask_vols::Vector{Union{Float64, Missing}}
end

"""
    atm_term_structure(surface::VolatilitySurface; atm_threshold::Float64=0.05) -> TermStructure

Extract the ATM term structure from a volatility surface.

For each unique tenor, selects the point closest to ATM (log_moneyness ≈ 0)
within the given threshold. Returns bid, ask, and mark vols for each tenor.

# Arguments
- `surface`: A VolatilitySurface
- `atm_threshold`: Maximum |log_moneyness| to consider as ATM (default: 0.05 ≈ 5%)

# Returns
- A TermStructure with moneyness=0.0, containing bid_vols, ask_vols, mark_vols
"""
function atm_term_structure(surface::VolatilitySurface; atm_threshold::Float64=0.05)::TermStructure
    # Group points by tenor (τ)
    by_tenor = Dict{Float64, Vector{VolPoint}}()
    for p in surface.points
        abs(p.log_moneyness) > atm_threshold && continue
        if !haskey(by_tenor, p.τ)
            by_tenor[p.τ] = VolPoint[]
        end
        push!(by_tenor[p.τ], p)
    end

    # For each tenor, pick the point closest to ATM
    tenors = Float64[]
    bid_vols = Union{Float64,Missing}[]
    ask_vols = Union{Float64,Missing}[]
    mark_vols = Union{Float64,Missing}[]

    for τ in sort(collect(keys(by_tenor)))
        points = by_tenor[τ]
        # Find point with smallest |log_moneyness|
        best = argmin(p -> abs(p.log_moneyness), points)
        push!(tenors, τ)
        push!(bid_vols, best.bid_vol)
        push!(ask_vols, best.ask_vol)
        push!(mark_vols, best.vol)
    end

    return TermStructure(
        surface.spot,
        surface.timestamp,
        surface.underlying,
        0.0,  # ATM moneyness
        tenors,
        mark_vols,
        bid_vols,
        ask_vols
    )
end

"""
    atm_term_structure(records::Vector{VolRecord}; atm_threshold::Float64=0.05) -> TermStructure

Build ATM term structure directly from records (convenience method).
"""
function atm_term_structure(records::Vector{VolRecord}; atm_threshold::Float64=0.05)::TermStructure
    surface = build_surface(records)
    return atm_term_structure(surface; atm_threshold=atm_threshold)
end

end # module
