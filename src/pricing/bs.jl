# Black-Scholes pricing with continuous dividend yield, and IV inversion.
#
# Inputs throughout: S=spot, K=strike, T=time to expiry (years), sigma=vol,
# r=risk-free rate (cont. comp.), q=dividend yield (cont. comp.).

const _DAYS_PER_YEAR = 365.25

"""
    time_to_expiry(expiry::DateTime, now::DateTime) -> Float64

Years from `now` to `expiry`. Uses 365.25-day years.
"""
function time_to_expiry(expiry::DateTime, now::DateTime)::Float64
    ms = Dates.value(expiry - now)
    days = ms / (1000.0 * 60.0 * 60.0 * 24.0)
    return days / _DAYS_PER_YEAR
end

# Standard normal CDF via Abramowitz & Stegun 7.1.26 (max abs err ~1.5e-7).
# Sufficient for IV-inversion to 1e-6 tolerance in vol.
function _norm_cdf(x::Float64)::Float64
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    sgn = x < 0 ? -1.0 : 1.0
    z = abs(x) / sqrt(2.0)
    t = 1.0 / (1.0 + p * z)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-z * z)
    return 0.5 * (1.0 + sgn * y)
end

_norm_pdf(x::Float64)::Float64 = exp(-0.5 * x * x) / sqrt(2.0 * pi)

_cp_sign(o::OptionType)::Float64 = o == Call ? 1.0 : -1.0

"""
    bs_price(S, K, T, sigma, option_type; r, q) -> Float64

Black-Scholes price of a European option on a spot asset paying continuous
dividend yield `q`, discounted at rate `r`.
"""
function bs_price(S::Float64, K::Float64, T::Float64, sigma::Float64,
                  option_type::OptionType; r::Float64, q::Float64)::Float64
    cp = _cp_sign(option_type)
    if T <= 0.0
        return max(cp * (S - K), 0.0)
    end
    if sigma <= 0.0
        return max(cp * (S * exp(-q * T) - K * exp(-r * T)), 0.0)
    end
    sqrtT = sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return cp * (S * exp(-q * T) * _norm_cdf(cp * d1) -
                 K * exp(-r * T) * _norm_cdf(cp * d2))
end

"""
    bs_delta(S, K, T, sigma, option_type; r, q) -> Float64
"""
function bs_delta(S::Float64, K::Float64, T::Float64, sigma::Float64,
                  option_type::OptionType; r::Float64, q::Float64)::Float64
    cp = _cp_sign(option_type)
    if T <= 0.0 || sigma <= 0.0
        return cp * (cp * (S - K) > 0 ? 1.0 : 0.0)
    end
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    return cp * exp(-q * T) * _norm_cdf(cp * d1)
end

"""
    bs_gamma(S, K, T, sigma; r, q) -> Float64

Gamma is the same for calls and puts.
"""
function bs_gamma(S::Float64, K::Float64, T::Float64, sigma::Float64;
                  r::Float64, q::Float64)::Float64
    (T <= 0.0 || sigma <= 0.0) && return 0.0
    sqrtT = sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return exp(-q * T) * _norm_pdf(d1) / (S * sigma * sqrtT)
end

"""
    bs_vega(S, K, T, sigma; r, q) -> Float64

Vega is the same for calls and puts. Returns dPrice/dSigma (per 1.0 of vol,
not per 1%).
"""
function bs_vega(S::Float64, K::Float64, T::Float64, sigma::Float64;
                 r::Float64, q::Float64)::Float64
    (T <= 0.0 || sigma <= 0.0) && return 0.0
    sqrtT = sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return S * exp(-q * T) * _norm_pdf(d1) * sqrtT
end

"""
    implied_vol(price, S, K, T, option_type; r, q,
                lo=1e-6, hi=5.0, tol=1e-8, maxiter=100) -> Union{Float64,Nothing}

Invert Black-Scholes for sigma via bisection. Returns `nothing` if the price
is outside the bracket `[bs_price(lo), bs_price(hi)]` (typically: price
below intrinsic, or above the deep-vol limit).

Bracket defaults handle SPY-style equity options at any plausible IV.
"""
function implied_vol(price::Float64, S::Float64, K::Float64, T::Float64,
                     option_type::OptionType;
                     r::Float64, q::Float64,
                     lo::Float64=1e-6, hi::Float64=5.0,
                     tol::Float64=1e-8, maxiter::Int=100)::Union{Float64,Nothing}
    (T <= 0.0 || price <= 0.0) && return nothing
    p_lo = bs_price(S, K, T, lo, option_type; r=r, q=q)
    p_hi = bs_price(S, K, T, hi, option_type; r=r, q=q)
    (price < p_lo || price > p_hi) && return nothing
    a, b = lo, hi
    for _ in 1:maxiter
        m = 0.5 * (a + b)
        p_m = bs_price(S, K, T, m, option_type; r=r, q=q)
        if abs(p_m - price) < tol || (b - a) < tol
            return m
        end
        if p_m < price
            a = m
        else
            b = m
        end
    end
    return 0.5 * (a + b)
end
