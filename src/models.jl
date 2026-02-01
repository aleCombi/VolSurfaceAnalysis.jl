# Black-76 Pricing Model
# Pricing functions for European options on futures/forwards

const DAYS_PER_YEAR = 365.25

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
    black76_delta(F, K, T, σ, option_type; r=0.0) -> Float64

Calculate the Black-76 delta (derivative of price w.r.t. forward/spot).

# Returns
- Delta: ∂Price/∂F (ranges from 0 to 1 for calls, -1 to 0 for puts)
"""
function black76_delta(F::Float64, K::Float64, T::Float64, σ::Float64,
                       option_type::OptionType; r::Float64=0.0)::Float64
    if T <= 0.0
        # At expiry: delta is 1 (ITM) or 0 (OTM)
        cp = call_put_sign(option_type)
        return cp * (F - K) > 0 ? Float64(cp) : 0.0
    end
    
    if σ <= 0.0
        cp = call_put_sign(option_type)
        return cp * (F - K) > 0 ? Float64(cp) : 0.0
    end
    
    sqrtT = sqrt(T)
    d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * sqrtT)
    
    D = exp(-r * T)
    N = Normal()
    
    if option_type == Call
        return D * cdf(N, d1)
    else
        return D * (cdf(N, d1) - 1.0)
    end
end

"""
    black76_gamma(F, K, T, σ, option_type; r=0.0) -> Float64

Calculate the Black-76 gamma (second derivative of price w.r.t. forward/spot).
Gamma is the same for calls and puts.

# Returns
- Gamma: ∂²Price/∂F²
"""
function black76_gamma(F::Float64, K::Float64, T::Float64, σ::Float64,
                       option_type::OptionType; r::Float64=0.0)::Float64
    if T <= 0.0 || σ <= 0.0
        return 0.0
    end
    
    sqrtT = sqrt(T)
    d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * sqrtT)
    
    D = exp(-r * T)
    N = Normal()
    
    # Gamma = D * φ(d1) / (F * σ * √T)
    return D * pdf(N, d1) / (F * σ * sqrtT)
end

"""
    black76_theta(F, K, T, σ, option_type; r=0.0) -> Float64

Calculate the Black-76 theta (derivative of price w.r.t. time).
Returns the daily theta (price change per day).

# Returns
- Theta: ∂Price/∂t per day (typically negative for long options)
"""
function black76_theta(F::Float64, K::Float64, T::Float64, σ::Float64,
                       option_type::OptionType; r::Float64=0.0)::Float64
    if T <= 0.0 || σ <= 0.0
        return 0.0
    end
    
    cp = call_put_sign(option_type)
    sqrtT = sqrt(T)
    d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * sqrtT)
    d2 = d1 - σ * sqrtT
    
    D = exp(-r * T)
    N = Normal()
    
    # Theta components:
    # 1. Time decay from vega: -F * φ(d1) * σ / (2 * √T)
    # 2. Discounting effect: r * D * [F * N(cp*d1) - K * N(cp*d2)] (for r > 0)
    
    time_decay = -D * F * pdf(N, d1) * σ / (2 * sqrtT)
    discount_effect = r * D * cp * (F * cdf(N, cp * d1) - K * cdf(N, cp * d2))
    
    # Annual theta - convert to daily
    annual_theta = time_decay + discount_effect
    return annual_theta / DAYS_PER_YEAR
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
    price_to_iv(price, F, K, T, option_type; r=0.0, σ_min=0.001, σ_max=10.0, tol=1e-8, max_iter=100) -> Float64

Convert an option price (as fraction of underlying) to implied volatility.
Uses `Roots.find_zero` with a bracketing method (Brent).

# Arguments
- `price::Float64`: Option price as fraction of underlying
- `F::Float64`: Forward/futures price (underlying_price)
- `K::Float64`: Strike price
- `T::Float64`: Time to expiry in years
- `option_type::OptionType`: Call or Put
- `r::Float64`: Interest rate for discounting (default: 0.0)
- `σ_min::Float64`: Minimum volatility bound for solver (default: 0.001)
- `σ_max::Float64`: Maximum volatility bound for solver (default: 10.0)
- `tol::Float64`: Convergence tolerance (default: 1e-8)
- `max_iter::Int`: Maximum iterations (default: 100)

# Returns
- Implied volatility as decimal (e.g., 0.65 for 65%)
- Returns NaN if no solution found
"""
function price_to_iv(price::Float64, F::Float64, K::Float64, T::Float64,
                     option_type::OptionType; r::Float64=0.0,
                     σ_min::Float64=0.001, σ_max::Float64=10.0,
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

    f(σ) = vol_to_price(σ, F, K, T, option_type; r=r) - price

    f_low = f(σ_min)
    f_high = f(σ_max)
    if f_low > 0.0 || f_high < 0.0
        return NaN
    end

    try
        return find_zero(f, (σ_min, σ_max), Brent(); atol=tol, rtol=0.0, maxiters=max_iter)
    catch
        return NaN
    end

end

