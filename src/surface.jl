# Volatility Surface Representation
# Types and functions for building and working with volatility surfaces

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
    bid_vol::Union{Float64,Missing}
    ask_vol::Union{Float64,Missing}
end

"""
    VolatilitySurface

A volatility surface at a single point in time.

# Fields
- `spot::Float64`: spot price of underlying
- `timestamp::DateTime`: observation timestamp
- `underlying::Underlying`: the underlying asset
- `points::Vector{VolPoint}`: the vol surface points
- `records::Vector{OptionRecord}`: raw records for this timestamp
"""
struct VolatilitySurface
    spot::Float64
    timestamp::DateTime
    underlying::Underlying
    points::Vector{VolPoint}
    records::Vector{OptionRecord}
end

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
- `mark_vols::Vector{Union{Float64, Missing}}`: mark implied volatilities (decimal)
- `bid_vols::Vector{Union{Float64, Missing}}`: bid implied volatilities (decimal)
- `ask_vols::Vector{Union{Float64, Missing}}`: ask implied volatilities (decimal)
"""
struct TermStructure
    spot::Float64
    timestamp::DateTime
    underlying::Underlying
    moneyness::Float64
    tenors::Vector{Float64}
    mark_vols::Vector{Union{Float64,Missing}}
    bid_vols::Vector{Union{Float64,Missing}}
    ask_vols::Vector{Union{Float64,Missing}}
end

# ============================================================================
# OptionRecord convenience wrappers for pricing functions
# ============================================================================

"""
    vol_to_price(record::OptionRecord) -> Union{Float64, Missing}

Convert an OptionRecord's mark_iv to the corresponding option price (as fraction of spot).
Returns missing if mark_iv is missing.

Uses the record's spot as the forward/spot reference.
"""
function vol_to_price(record::OptionRecord)::Union{Float64,Missing}
    ismissing(record.mark_iv) && return missing

    σ = record.mark_iv / 100.0  # Convert from percentage to decimal
    F = record.spot
    K = record.strike
    T = time_to_expiry(record.expiry, record.timestamp)

    r = 0.0  # Default to 0 (no rate info in OptionRecord)

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
    price_to_iv(record::OptionRecord) -> Union{Float64, Missing}

Convert an OptionRecord's mark_price to implied volatility.
Returns missing if mark_price is missing.

"""
function price_to_iv(record::OptionRecord)::Union{Float64,Missing}
    ismissing(record.mark_price) && return missing

    price = record.mark_price
    F = record.spot
    K = record.strike
    T = time_to_expiry(record.expiry, record.timestamp)

    if T <= 0.0
        return missing
    end

    σ = price_to_iv(price, F, K, T, record.option_type)
    return isnan(σ) ? missing : σ
end

"""
    bid_iv(record::OptionRecord) -> Union{Float64, Missing}

Compute implied volatility from an OptionRecord's bid_price.
Returns missing if bid_price is missing or IV cannot be computed.
"""
function bid_iv(record::OptionRecord)::Union{Float64,Missing}
    ismissing(record.bid_price) && return missing

    price = record.bid_price
    F = record.spot
    K = record.strike
    T = time_to_expiry(record.expiry, record.timestamp)

    if T <= 0.0
        return missing
    end

    σ = price_to_iv(price, F, K, T, record.option_type)
    return isnan(σ) ? missing : σ
end

"""
    ask_iv(record::OptionRecord) -> Union{Float64, Missing}

Compute implied volatility from an OptionRecord's ask_price.
Returns missing if ask_price is missing or IV cannot be computed.
"""
function ask_iv(record::OptionRecord)::Union{Float64,Missing}
    ismissing(record.ask_price) && return missing

    price = record.ask_price
    F = record.spot
    K = record.strike
    T = time_to_expiry(record.expiry, record.timestamp)

    if T <= 0.0
        return missing
    end

    σ = price_to_iv(price, F, K, T, record.option_type)
    return isnan(σ) ? missing : σ
end

# ============================================================================
# Surface Building
# ============================================================================

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
    build_surface(records::Vector{OptionRecord}) -> VolatilitySurface

Build a VolatilitySurface from a vector of OptionRecords (assumed to be from a single timestamp).
Records with missing bid/ask are filtered out. Stores the remaining raw records on the surface
for exact bid/ask lookup.

For each (strike, expiry) pair, chooses the ITM option. At ATM, uses highest volume as tiebreaker.
Records with missing mark_iv are skipped.
"""
function build_surface(records::Vector{OptionRecord})::VolatilitySurface
    isempty(records) && error("Cannot build surface from empty records")

    # Filter out records without complete bid/ask quotes
    records = filter(r -> !ismissing(r.bid_price) && !ismissing(r.ask_price), records)
    isempty(records) && error("Cannot build surface: no records with bid/ask quotes")

    # Use first record for metadata (assuming all from same timestamp/underlying)
    first_rec = records[1]
    spot = first_rec.spot
    timestamp = first_rec.timestamp
    underlying = first_rec.underlying

    # Group by (strike, expiry)
    grouped = Dict{Tuple{Float64,DateTime},Vector{OptionRecord}}()
    for rec in records
        ismissing(rec.mark_iv) && continue
        key = (rec.strike, rec.expiry)
        if !haskey(grouped, key)
            grouped[key] = OptionRecord[]
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
    sort!(points, by=p -> (p.τ, p.log_moneyness))

    return VolatilitySurface(spot, timestamp, underlying, points, records)
end

# ============================================================================
# Raw Record Lookup
# ============================================================================

"""
    find_record(surface::VolatilitySurface, strike::Float64, expiry::DateTime, option_type::OptionType)
        -> Union{OptionRecord, Missing}

Find the raw OptionRecord for an exact strike/expiry/option_type at this surface.
Returns `missing` if no matching record exists.
"""
function find_record(
    surface::VolatilitySurface,
    strike::Float64,
    expiry::DateTime,
    option_type::OptionType
)::Union{OptionRecord, Missing}
    for rec in surface.records
        if rec.strike == strike && rec.expiry == expiry && rec.option_type == option_type
            return rec
        end
    end
    return missing
end

# ============================================================================
# Term Structure (slice of vol surface at fixed moneyness)
# ============================================================================

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
    by_tenor = Dict{Float64,Vector{VolPoint}}()
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
    atm_term_structure(records::Vector{OptionRecord}; atm_threshold::Float64=0.05) -> TermStructure

Build ATM term structure directly from records (convenience method).
"""
function atm_term_structure(records::Vector{OptionRecord}; atm_threshold::Float64=0.05)::TermStructure
    surface = build_surface(records)
    return atm_term_structure(surface; atm_threshold=atm_threshold)
end
