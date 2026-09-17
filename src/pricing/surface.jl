# Vol surface types and their queries: IV, price, greeks, forward, delta
# inversion.

"""
    ExpirySlice(expiry, tau, strikes, ivs)

One expiry's IVs by strike, with `tau` the time to expiry in years cached
at build time. Throws `ArgumentError` unless `strikes` is non-empty,
sorted, unique, and the same length as `ivs`.
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

Abstract supertype of surface representations. A representation carries
`underlying`, `timestamp`, `spot`, `rate` and `div`, and answers
`expiries`, `get_slice`, `iv`, `price`, `delta`, `gamma`, `vega` and
`forward`; `invert_delta` is derived from `get_slice` and `delta` for
any of them.
"""
abstract type VolatilitySurface end

"""
    RawSurface(underlying, timestamp, spot, rate, div, slices)

The slices stored directly, with no parametric form. Queries name a
quoted expiry exactly; within a slice the IV is interpolated linearly
in log-moneyness and held flat outside the observed strikes. Throws
`ArgumentError` unless `slices` is non-empty, sorted by expiry, and
unique in expiry.
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

The quoted expiries, ascending.
"""
expiries(s::RawSurface) = [sl.expiry for sl in s.slices]

"""
    get_slice(s::VolatilitySurface, expiry) -> Union{ExpirySlice, Nothing}

The slice at exactly `expiry`, or `nothing` when it is not quoted.
"""
function get_slice(s::RawSurface, expiry::DateTime)::Union{ExpirySlice,Nothing}
    for sl in s.slices
        sl.expiry == expiry && return sl
    end
    return nothing
end

"""
    forward(s::VolatilitySurface, expiry) -> Float64

`S * exp((r - q) * tau)` from the surface's own spot, rate and div and
the slice's cached `tau`. Throws `ArgumentError` if `expiry` is not
quoted.
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

Implied vol at (`expiry`, `strike`), interpolated within the slice.
Throws `ArgumentError` if `expiry` is not quoted.
"""
function iv(s::RawSurface, expiry::DateTime, strike::Float64)::Float64
    sl = get_slice(s, expiry)
    sl === nothing && throw(ArgumentError("expiry $expiry not in surface"))
    return _interp_iv(sl, s.spot, strike)
end

"""
    price(s::VolatilitySurface, expiry, strike, option_type) -> Float64

Black-Scholes price at `iv(s, expiry, strike)`, from the surface's own
spot, rate, div and the slice's `tau`. Throws `ArgumentError` if
`expiry` is not quoted; so do `delta`, `gamma` and `vega`.
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

Black-Scholes delta at the surface's IV; see [`price`](@ref).
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

Black-Scholes gamma at the surface's IV; see [`price`](@ref).
"""
function gamma(s::RawSurface, expiry::DateTime, strike::Float64)::Float64
    sl = get_slice(s, expiry)
    sl === nothing && throw(ArgumentError("expiry $expiry not in surface"))
    sigma = _interp_iv(sl, s.spot, strike)
    return bs_gamma(s.spot, strike, sl.tau, sigma; r=s.rate, q=s.div)
end

"""
    vega(s::VolatilitySurface, expiry, strike) -> Float64

Black-Scholes vega at the surface's IV, per 1.0 of vol; see
[`price`](@ref).
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

The strike `K` with `abs(delta(s, expiry, K, option_type)) ==
target_abs_delta`, found by bisection over the slice's observed strike
range. Returns `nothing` when no observed strike carries that delta.
Throws `ArgumentError` if `expiry` is not quoted or the target is not
positive.
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

    # Bisection assumes |delta| monotone in K. That holds at fixed sigma,
    # and across the smile because the slice interpolates linearly in
    # log-moneyness, which is monotone in strike order -- so changing the
    # interpolation can break this search. The direction is read off the
    # endpoints rather than the option type, so a smile that breaks the
    # assumption degrades to the last midpoint, not a wrong-way search.
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
