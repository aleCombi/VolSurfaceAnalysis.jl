# Build a VolatilitySurface from a raw option chain.
#
# v1 handles the mark-price convention only (the only one ParquetDataSource
# produces). Per-strike IV is inverted from the OTM-side mark (call if
# K >= spot, put otherwise), falling back to the ITM side if the OTM side
# is missing.

# Pick the OTM-side quote for a (call_quote, put_quote, strike, spot) group.
# Returns (mark, option_type) or nothing if neither side has a usable mark.
function _pick_otm(call_q::Union{OptionQuote,Nothing},
                   put_q::Union{OptionQuote,Nothing},
                   strike::Float64, spot::Float64)
    otm_is_call = strike >= spot
    primary = otm_is_call ? call_q : put_q
    fallback = otm_is_call ? put_q : call_q

    if primary !== nothing && !ismissing(primary.mark) && primary.mark > 0.0
        return (Float64(primary.mark), primary.option_type)
    end
    if fallback !== nothing && !ismissing(fallback.mark) && fallback.mark > 0.0
        return (Float64(fallback.mark), fallback.option_type)
    end
    return nothing
end

function _build_slice(expiry::DateTime, tau::Float64,
                     quotes::Vector{OptionQuote},
                     spot::Float64, rate::Float64, div::Float64)::Union{ExpirySlice,Nothing}
    by_strike = Dict{Float64,Tuple{Union{OptionQuote,Nothing},Union{OptionQuote,Nothing}}}()
    for q in quotes
        c, p = get(by_strike, q.strike, (nothing, nothing))
        if q.option_type == Call
            by_strike[q.strike] = (q, p)
        else
            by_strike[q.strike] = (c, q)
        end
    end

    strikes = Float64[]
    ivs = Float64[]
    for k in sort!(collect(keys(by_strike)))
        c, p = by_strike[k]
        picked = _pick_otm(c, p, k, spot)
        picked === nothing && continue
        mark, otype = picked
        sigma = implied_vol(mark, spot, k, tau, otype; r=rate, q=div)
        sigma === nothing && continue
        push!(strikes, k)
        push!(ivs, sigma)
    end

    isempty(strikes) && return nothing
    return ExpirySlice(expiry, tau, strikes, ivs)
end

"""
    build_surface(chain, spot, rate, div) -> RawSurface

Build a `RawSurface` from a chain (vector of `OptionQuote`s) at a single
timestamp. All quotes in `chain` are expected to share a `timestamp` and
`underlying`; the surface inherits both from `chain[1]`.

For each expiry, builds an `ExpirySlice` by inverting per-strike IV from
the OTM-side mark. Strikes where neither side has a usable mark, or where
IV inversion fails (price outside `[intrinsic, deep-vol limit]`), are
dropped. Expiries with no usable strikes are dropped.

Throws on an empty chain or if every slice ends up empty.
"""
function build_surface(chain::Vector{OptionQuote}, spot::Float64,
                       rate::Float64, div::Float64)::RawSurface
    isempty(chain) && throw(ArgumentError("cannot build surface from empty chain"))
    ts = chain[1].timestamp
    underlying = chain[1].underlying

    by_expiry = Dict{DateTime,Vector{OptionQuote}}()
    for q in chain
        push!(get!(() -> OptionQuote[], by_expiry, q.expiry), q)
    end

    slices = ExpirySlice[]
    for e in sort!(collect(keys(by_expiry)))
        tau = time_to_expiry(e, ts)
        tau <= 0.0 && continue
        sl = _build_slice(e, tau, by_expiry[e], spot, rate, div)
        sl === nothing && continue
        push!(slices, sl)
    end

    isempty(slices) &&
        throw(ArgumentError("no usable expiries in chain (all dropped during IV inversion)"))

    return RawSurface(underlying, ts, spot, rate, div, slices)
end
