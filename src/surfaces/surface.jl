# Volatility surface: per-expiry slices of (strike, IV) plus the spot, rate,
# div, and timestamp used to build them. Queries return interpolated IVs and
# BS-derived prices / greeks.

"""
    ExpirySlice

One expiry's worth of inverted IVs. `strikes` is sorted ascending; `ivs`
is the same length, indexed by strike. `tau` is time-to-expiry in years,
cached at build time.
"""
struct ExpirySlice
    expiry  :: DateTime
    tau     :: Float64
    strikes :: Vector{Float64}
    ivs     :: Vector{Float64}

    function ExpirySlice(expiry::DateTime, tau::Float64,
                         strikes::AbstractVector{<:Real},
                         ivs::AbstractVector{<:Real})
        length(strikes) == length(ivs) ||
            throw(ArgumentError("strikes and ivs must have equal length"))
        isempty(strikes) &&
            throw(ArgumentError("ExpirySlice must have at least one strike"))
        issorted(strikes) ||
            throw(ArgumentError("strikes must be sorted"))
        allunique(strikes) ||
            throw(ArgumentError("strikes must be unique"))
        new(expiry, tau,
            collect(Float64, strikes), collect(Float64, ivs))
    end
end

"""
    VolatilitySurface

Abstract type for vol surfaces. Concrete subtypes carry their own
representation (raw grid, SVI, SABR, ...). Query methods (`iv`, `price`,
`delta`, `gamma`, `vega`, `forward`) work against any subtype that
implements `expiries`, `get_slice`, and exposes `spot`, `rate`, `div`.
"""
abstract type VolatilitySurface end

"""
    RawSurface <: VolatilitySurface

Raw per-expiry slices. `iv(s, expiry, strike)` interpolates linearly in
log-moneyness within a slice; cross-expiry queries are not supported in
v1 (calling code must hit a quoted expiry exactly).
"""
struct RawSurface <: VolatilitySurface
    underlying :: Underlying
    timestamp  :: DateTime
    spot       :: Float64
    rate       :: Float64
    div        :: Float64
    slices     :: Vector{ExpirySlice}

    function RawSurface(underlying::Underlying, timestamp::DateTime,
                        spot::Float64, rate::Float64, div::Float64,
                        slices::AbstractVector{ExpirySlice})
        isempty(slices) &&
            throw(ArgumentError("RawSurface must have at least one slice"))
        issorted(slices, by=s->s.expiry) ||
            throw(ArgumentError("slices must be sorted by expiry"))
        allunique(s.expiry for s in slices) ||
            throw(ArgumentError("slice expiries must be unique"))
        new(underlying, timestamp, spot, rate, div, collect(ExpirySlice, slices))
    end
end

"""
    expiries(s::VolatilitySurface) -> Vector{DateTime}
"""
expiries(s::RawSurface) = [sl.expiry for sl in s.slices]

"""
    get_slice(s::VolatilitySurface, expiry) -> Union{ExpirySlice, Nothing}
"""
function get_slice(s::RawSurface, expiry::DateTime)::Union{ExpirySlice,Nothing}
    for sl in s.slices
        sl.expiry == expiry && return sl
    end
    return nothing
end

"""
    forward(s::VolatilitySurface, expiry) -> Float64

`S * exp((r - q) * tau)` where `tau` comes from the matching slice. Errors
if the expiry is not present.
"""
function forward(s::RawSurface, expiry::DateTime)::Float64
    sl = get_slice(s, expiry)
    sl === nothing && throw(ArgumentError("expiry $expiry not in surface"))
    return s.spot * exp((s.rate - s.div) * sl.tau)
end

# Linear interpolation in log-moneyness within a slice.
# Out-of-range strikes flat-extrapolate at the endpoint IV.
function _interp_iv(sl::ExpirySlice, spot::Float64, strike::Float64)::Float64
    x  = log(strike / spot)
    xs = [log(k / spot) for k in sl.strikes]
    n  = length(xs)
    if n == 1 || x <= xs[1]
        return sl.ivs[1]
    end
    if x >= xs[end]
        return sl.ivs[end]
    end
    i = searchsortedlast(xs, x)
    w = (x - xs[i]) / (xs[i+1] - xs[i])
    return sl.ivs[i] + w * (sl.ivs[i+1] - sl.ivs[i])
end

"""
    iv(s::VolatilitySurface, expiry, strike) -> Float64

Implied vol at (`expiry`, `strike`). Errors if `expiry` is not present.
Strike is linearly interpolated in log-moneyness; out of range flat-
extrapolates.
"""
function iv(s::RawSurface, expiry::DateTime, strike::Float64)::Float64
    sl = get_slice(s, expiry)
    sl === nothing && throw(ArgumentError("expiry $expiry not in surface"))
    return _interp_iv(sl, s.spot, strike)
end

"""
    price(s::VolatilitySurface, expiry, strike, option_type) -> Float64
"""
function price(s::RawSurface, expiry::DateTime, strike::Float64,
               option_type::OptionType)::Float64
    sl = get_slice(s, expiry)
    sl === nothing && throw(ArgumentError("expiry $expiry not in surface"))
    sigma = _interp_iv(sl, s.spot, strike)
    return bs_price(s.spot, strike, sl.tau, sigma, option_type;
                    r=s.rate, q=s.div)
end

"""
    delta(s::VolatilitySurface, expiry, strike, option_type) -> Float64
"""
function delta(s::RawSurface, expiry::DateTime, strike::Float64,
               option_type::OptionType)::Float64
    sl = get_slice(s, expiry)
    sl === nothing && throw(ArgumentError("expiry $expiry not in surface"))
    sigma = _interp_iv(sl, s.spot, strike)
    return bs_delta(s.spot, strike, sl.tau, sigma, option_type;
                    r=s.rate, q=s.div)
end

"""
    gamma(s::VolatilitySurface, expiry, strike) -> Float64
"""
function gamma(s::RawSurface, expiry::DateTime, strike::Float64)::Float64
    sl = get_slice(s, expiry)
    sl === nothing && throw(ArgumentError("expiry $expiry not in surface"))
    sigma = _interp_iv(sl, s.spot, strike)
    return bs_gamma(s.spot, strike, sl.tau, sigma; r=s.rate, q=s.div)
end

"""
    vega(s::VolatilitySurface, expiry, strike) -> Float64
"""
function vega(s::RawSurface, expiry::DateTime, strike::Float64)::Float64
    sl = get_slice(s, expiry)
    sl === nothing && throw(ArgumentError("expiry $expiry not in surface"))
    sigma = _interp_iv(sl, s.spot, strike)
    return bs_vega(s.spot, strike, sl.tau, sigma; r=s.rate, q=s.div)
end
