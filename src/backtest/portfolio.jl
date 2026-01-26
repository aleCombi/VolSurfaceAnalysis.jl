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
Uses raw bid/ask prices from the surface records (no volatility-based fallback).

# Arguments
- `trade::Trade`: Trade to open
- `surface::VolatilitySurface`: Current market for pricing

# Returns
- A new Position

# Throws
- Error if record or bid/ask price is missing
"""
function open_position(trade::Trade, surface::VolatilitySurface)::Position
    trade.underlying != surface.underlying && error("Underlying mismatch for trade")

    rec = find_record(surface, trade.strike, trade.expiry, trade.option_type)
    ismissing(rec) && error("Record not found for $(trade.strike)/$(trade.expiry)/$(trade.option_type)")

    price = trade.direction > 0 ? rec.ask_price : rec.bid_price
    ismissing(price) && error("Missing bid/ask price for $(trade.strike)/$(trade.expiry)/$(trade.option_type)")

    entry_price = price

    return Position(trade, entry_price, surface.spot, surface.timestamp)
end

"""
    _open_positions(trades, surface; debug=false) -> Vector{Position}

Create positions for a vector of trades. Returns an empty vector if any trade
cannot be opened (missing record or bid/ask). When `debug=true`, prints the failure.
"""
function _open_positions(
    trades::Vector{Trade},
    surface::VolatilitySurface;
    debug::Bool=false
)::Vector{Position}
    positions = Position[]
    for t in trades
        try
            push!(positions, open_position(t, surface))
        catch e
            if debug
                println("No entry: open_position failed for $(t.option_type) strike=$(t.strike) expiry=$(t.expiry) error=$(e)")
            end
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
