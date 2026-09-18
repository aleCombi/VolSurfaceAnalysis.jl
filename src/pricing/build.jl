# `build_surface`: a RawSurface from an option chain quoted by mark price.

# The OTM side (call at or above spot, put below) first; the ITM side is a
# fallback rather than the strike being dropped. Returns (mark, option_type)
# or nothing when neither side has a positive mark.
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
    build_surface(chain, spot, rate, div) -> Union{RawSurface, Nothing}

A `RawSurface` from a chain of `OptionQuote`s at one instant, with one
slice per expiry and one IV per strike inverted from the OTM-side mark.
Every quote is expected to share a `timestamp` and `underlying`; the
surface takes both from `chain[1]`.

Dropped, silently: a strike with no positive mark on either side or
whose mark does not invert, and an expiry at or before the chain's
timestamp or with no surviving strike. Returns `nothing` when no expiry
survives. Throws `ArgumentError` on an empty chain.
"""
function build_surface(chain::Vector{OptionQuote}, spot::Float64,
                       rate::Float64, div::Float64)::Union{RawSurface,Nothing}
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

    isempty(slices) && return nothing

    return RawSurface(underlying, ts, spot, rate, div, slices)
end
