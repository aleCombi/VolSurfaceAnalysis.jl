# Position Management
# Pure types and functions for option positions

using Dates

"""
    Position

A pure record of an option position: what was traded and at what price.
This is a value type with no side effects.

# Fields
- `trade::Trade`: The contract specification
- `entry_price::Float64`: Price paid/received (fraction of underlying)
- `entry_spot::Float64`: Spot price at entry (USD)
- `entry_timestamp::DateTime`: When position was opened
"""
struct Position
    trade::Trade
    entry_price::Float64
    entry_spot::Float64
    entry_timestamp::DateTime
end

# ============================================================================
# Position Creation
# ============================================================================

"""
    open_position(trade, surface) -> Position

Create a position by pricing a trade against a surface. Pure function, no side effects.

# Arguments
- `trade::Trade`: Trade to open
- `surface::VolatilitySurface`: Current market for pricing

# Returns
- A new Position

# Throws
- Error if IV not found on surface or option expired
"""
function open_position(trade::Trade, surface::VolatilitySurface)::Position
    # Buy (direction=+1) uses ask, Sell (direction=-1) uses bid
    side = trade.direction > 0 ? :ask : :bid
    σ = find_vol(surface, trade.strike, trade.expiry; side=side)
    ismissing(σ) && error("IV not found for $(trade.strike)/$(trade.expiry)")

    T = time_to_expiry(trade.expiry, surface.timestamp)
    T <= 0.0 && error("Cannot open position on expired option")

    entry_price = vol_to_price(σ, surface.spot, trade.strike, T, trade.option_type)

    return Position(trade, entry_price, surface.spot, surface.timestamp)
end

"""
    _open_positions(trades, surface) -> Vector{Position}

Create positions for a vector of trades. Returns an empty vector if any trade
cannot be opened (missing IV or expired).
"""
function _open_positions(trades::Vector{Trade}, surface::VolatilitySurface)::Vector{Position}
    positions = Position[]
    for t in trades
        try
            push!(positions, open_position(t, surface))
        catch
            return Position[]
        end
    end
    return positions
end

# ============================================================================
# Settlement
# ============================================================================

"""
    entry_cost(position) -> Float64

Calculate the entry cost of a position in USD.
Positive for buys (paid), negative for sells (received premium).
"""
function entry_cost(position::Position)::Float64
    trade = position.trade
    return position.entry_price * trade.direction * trade.quantity * position.entry_spot
end

"""
    settle(position, settlement_spot) -> Float64

Calculate the P&L of a position at expiry. Pure function.

P&L = payoff at expiry - entry cost

# Arguments
- `position::Position`: The position to settle
- `settlement_spot::Float64`: Spot price at expiry

# Returns
- Realized P&L in USD (positive = profit, negative = loss)

# Example
```julia
trade = Trade(BTC, 100000.0, expiry, Put; direction=-1)  # short put
pos = open_position(trade, surface)
pnl = settle(pos, 95000.0)  # settlement at 95k
```
"""
function settle(position::Position, settlement_spot::Float64)::Float64
    payout = payoff(position.trade, settlement_spot)
    cost = entry_cost(position)
    return payout - cost
end

"""
    settle(positions::Vector{Position}, settlement_spot::Float64) -> Float64

Settle multiple positions at the same spot price. Returns total P&L.
"""
function settle(positions::Vector{Position}, settlement_spot::Float64)::Float64
    return sum(settle(pos, settlement_spot) for pos in positions; init=0.0)
end
