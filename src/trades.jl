# Trade Representation and Pricing
# Types and functions for representing and pricing option trades

"""
    Trade

Represents a single option trade (long or short position).

# Fields
- `underlying::Underlying`: The underlying asset (BTC or ETH)
- `strike::Float64`: Strike price
- `expiry::DateTime`: Expiration date (08:00 UTC normalized)
- `option_type::OptionType`: Call or Put
- `direction::Int`: +1 for long, -1 for short
- `quantity::Float64`: Number of contracts
- `trade_date::DateTime`: Date/time when the trade was executed
"""
struct Trade
    underlying::Underlying
    strike::Float64
    expiry::DateTime
    option_type::OptionType
    direction::Int
    quantity::Float64
    trade_date::DateTime
end

"""
    Trade(underlying, strike, expiry, option_type, trade_date; direction=1, quantity=1.0)

Convenience constructor with default direction (long) and quantity (1.0).
"""
function Trade(underlying::Underlying, strike::Float64, expiry::DateTime,
               option_type::OptionType, trade_date::DateTime;
               direction::Int=1, quantity::Float64=1.0)
    return Trade(underlying, strike, expiry, option_type, direction, quantity, trade_date)
end

# ============================================================================
# Volatility Lookup
# ============================================================================

"""
    find_vol(surface::VolatilitySurface, strike::Float64, expiry::DateTime;
             side::Symbol=:mark, tol::Float64=1e-9) -> Union{Float64, Missing}

Find the implied volatility for a given strike and expiry on the surface.

Returns `missing` if no exact match is found (within floating-point tolerance).
The match is performed in (log_moneyness, τ) coordinates.

# Arguments
- `surface`: The volatility surface to search
- `strike`: Strike price
- `expiry`: Expiration datetime
- `side`: Which vol to return - `:mark` (default), `:bid`, or `:ask`
- `tol`: Tolerance for floating-point comparison (default: 1e-9)

# Notes
For realistic backtesting, use `:ask` when buying and `:bid` when selling.
If bid/ask is missing, falls back to mark vol.
"""
function find_vol(surface::VolatilitySurface, strike::Float64, expiry::DateTime;
                  side::Symbol=:mark, tol::Float64=1e-9)::Union{Float64,Missing}
    # Convert to surface coordinates
    log_m = log(strike / surface.spot)
    τ = time_to_expiry(expiry, surface.timestamp)

    # Search for matching point
    for p in surface.points
        if abs(p.log_moneyness - log_m) < tol && abs(p.τ - τ) < tol
            if side == :bid
                return coalesce(p.bid_vol, p.vol)
            elseif side == :ask
                return coalesce(p.ask_vol, p.vol)
            else
                return p.vol
            end
        end
    end

    return missing
end

# ============================================================================
# Pricing Functions
# ============================================================================

"""
    price(trade::Trade, surface::VolatilitySurface; r::Float64=0.0) -> Union{Float64, Missing}

Price a trade using the implied volatility from a volatility surface.

Returns `missing` if the trade's strike/expiry is not found on the surface.
The price is returned as an absolute value (not normalized by spot).

# Arguments
- `trade`: The trade to price
- `surface`: The volatility surface containing IV data
- `r`: Risk-free rate for discounting (default: 0.0)

# Returns
- Absolute option price × direction × quantity, or `missing` if IV not found
"""
function price(trade::Trade, surface::VolatilitySurface; r::Float64=0.0)::Union{Float64,Missing}
    # Verify underlying matches
    trade.underlying != surface.underlying && return missing

    # Find IV on the surface
    σ = find_vol(surface, trade.strike, trade.expiry)
    ismissing(σ) && return missing

    # Calculate time to expiry from surface timestamp
    T = time_to_expiry(trade.expiry, surface.timestamp)
    T <= 0.0 && return missing

    # Price using Black-76
    F = surface.spot
    K = trade.strike
    abs_price = black76_price(F, K, T, σ, trade.option_type; r=r)

    return abs_price * trade.direction * trade.quantity
end

"""
    price(trade::Trade, spot::Float64, σ::Float64, timestamp::DateTime;
          r::Float64=0.0) -> Union{Float64, Missing}

Price a trade using explicit spot price and implied volatility.

Useful for scenario analysis and what-if calculations.

# Arguments
- `trade`: The trade to price
- `spot`: Current spot/forward price
- `σ`: Implied volatility as decimal (e.g., 0.65 for 65%)
- `timestamp`: Current timestamp for time-to-expiry calculation
- `r`: Risk-free rate for discounting (default: 0.0)

# Returns
- Absolute option price × direction × quantity, or `missing` if expired
"""
function price(trade::Trade, spot::Float64, σ::Float64, timestamp::DateTime;
               r::Float64=0.0)::Union{Float64,Missing}
    T = time_to_expiry(trade.expiry, timestamp)
    T <= 0.0 && return missing

    abs_price = black76_price(spot, trade.strike, T, σ, trade.option_type; r=r)
    return abs_price * trade.direction * trade.quantity
end

# ============================================================================
# Payoff at Maturity
# ============================================================================

"""
    payoff(trade::Trade, spot_at_expiry::Float64) -> Float64

Calculate the payoff of a trade at expiration.

# Arguments
- `trade`: The trade
- `spot_at_expiry`: The spot price at expiration

# Returns
- Intrinsic value × direction × quantity

# Examples
```julia
# Long call with strike 100, spot at expiry 110
t = Trade(BTC, 100.0, expiry, Call, trade_date)
payoff(t, 110.0)  # → 10.0

# Short put with strike 100, spot at expiry 90
t = Trade(BTC, 100.0, expiry, Put, trade_date; direction=-1)
payoff(t, 90.0)   # → -10.0 (loss for short put)
```
"""
function payoff(trade::Trade, spot_at_expiry::Float64)::Float64
    intrinsic = if trade.option_type == Call
        max(spot_at_expiry - trade.strike, 0.0)
    else
        max(trade.strike - spot_at_expiry, 0.0)
    end

    return intrinsic * trade.direction * trade.quantity
end
