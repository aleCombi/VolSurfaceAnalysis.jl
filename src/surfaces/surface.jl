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

"""
    invert_delta(s::VolatilitySurface, expiry, option_type, target_abs_delta;
                 tol=1e-6, maxiter=100) -> Union{Float64, Nothing}

Strike `K` such that `abs(delta(s, expiry, K, option_type)) == target_abs_delta`.

Bisects on `K` over the matching slice's observed strike range
`[slice.strikes[1], slice.strikes[end]]`. Returns `nothing` when the target
is outside the bracket -- no observed strike in the slice carries that
delta. Errors if the expiry is not in the surface (matching the rest of
the query API).

The slice-range bracket means the inversion never lands in the surface's
flat-extrapolation regime; richer surfaces with explicit extrapolation
policies can widen the bracket without changing the contract.
"""
function invert_delta(s::VolatilitySurface, expiry::DateTime,
                      option_type::OptionType, target_abs_delta::Float64;
                      tol::Float64=1e-6, maxiter::Int=100)::Union{Float64,Nothing}
    target_abs_delta > 0.0 ||
        throw(ArgumentError("target_abs_delta must be positive, got $target_abs_delta"))
    sl = get_slice(s, expiry)
    sl === nothing && throw(ArgumentError("expiry $expiry not in surface"))

    K_lo = sl.strikes[1]
    K_hi = sl.strikes[end]
    d_lo = abs(delta(s, expiry, K_lo, option_type))
    d_hi = abs(delta(s, expiry, K_hi, option_type))

    lo_d, hi_d = minmax(d_lo, d_hi)
    (target_abs_delta < lo_d || target_abs_delta > hi_d) && return nothing

    # |delta| is monotone in K under BS at a fixed sigma: increasing for puts,
    # decreasing for calls. Within the slice, IV is piecewise linear in
    # log-moneyness so |delta(K)| stays monotone in practice for SPY-style
    # smiles; pathological smiles could break it, in which case bisection
    # returns the last midpoint after maxiter.
    increasing_in_K = d_hi > d_lo
    a, b = K_lo, K_hi
    for _ in 1:maxiter
        m = 0.5 * (a + b)
        d_m = abs(delta(s, expiry, m, option_type))
        if abs(d_m - target_abs_delta) < tol || (b - a) < tol
            return m
        end
        if increasing_in_K
            d_m < target_abs_delta ? (a = m) : (b = m)
        else
            d_m > target_abs_delta ? (a = m) : (b = m)
        end
    end
    return 0.5 * (a + b)
end
