# Portfolio Management
# Position tracking and mark-to-market valuation for backtesting

using Dates

"""
    Position

Represents an open option position.

# Fields
- `trade::Trade`: The underlying trade details
- `entry_price::Float64`: Price paid (fraction of underlying)
- `entry_spot::Float64`: Spot price at entry (USD)
- `entry_timestamp::DateTime`: When position was opened
- `entry_vol::Float64`: Implied volatility at entry (decimal)
- `id::Int`: Unique position identifier
"""
struct Position
    trade::Trade
    entry_price::Float64
    entry_spot::Float64
    entry_timestamp::DateTime
    entry_vol::Float64
    id::Int
end

"""
    PortfolioSnapshot

A snapshot of portfolio state at a point in time.

# Fields
- `timestamp::DateTime`: Snapshot time
- `positions::Int`: Number of open positions
- `mtm_value::Float64`: Mark-to-market value (in underlying units)
- `unrealized_pnl::Float64`: Unrealized P&L
- `realized_pnl::Float64`: Cumulative realized P&L
- `cash::Float64`: Cash balance
- `delta::Float64`: Net portfolio delta
- `vega::Float64`: Net portfolio vega
"""
struct PortfolioSnapshot
    timestamp::DateTime
    positions::Int
    mtm_value::Float64
    unrealized_pnl::Float64
    realized_pnl::Float64
    cash::Float64
    delta::Float64
    vega::Float64
end

"""
    TradeRecord

Record of an executed trade for logging.

# Fields
- `position_id::Int`: Position identifier
- `action::Symbol`: :open or :close
- `timestamp::DateTime`: Execution time
- `trade::Trade`: Trade details
- `price::Float64`: Execution price (fraction of underlying)
- `vol::Float64`: IV at execution
- `pnl::Float64`: Realized P&L (for closes)
"""
struct TradeRecord
    position_id::Int
    action::Symbol
    timestamp::DateTime
    trade::Trade
    price::Float64
    vol::Float64
    pnl::Float64
end

"""
    Portfolio

Manages a collection of option positions with P&L tracking.

# Fields
- `positions::Vector{Position}`: Open positions
- `cash::Float64`: Cash balance (in underlying units)
- `realized_pnl::Float64`: Cumulative realized P&L
- `history::Vector{PortfolioSnapshot}`: Historical snapshots
- `trade_log::Vector{TradeRecord}`: All executed trades
- `next_id::Int`: Next position ID
"""
mutable struct Portfolio
    positions::Vector{Position}
    cash::Float64
    realized_pnl::Float64
    history::Vector{PortfolioSnapshot}
    trade_log::Vector{TradeRecord}
    next_id::Int
end

"""
    Portfolio(; initial_cash=0.0)

Create an empty portfolio with optional initial cash.
"""
function Portfolio(; initial_cash::Float64=0.0)
    Portfolio(Position[], initial_cash, 0.0, PortfolioSnapshot[], TradeRecord[], 1)
end

# ============================================================================
# Position Management
# ============================================================================

"""
    add_position!(portfolio, trade, surface) -> Position

Open a new position at current market prices.

# Arguments
- `portfolio::Portfolio`: Portfolio to add to
- `trade::Trade`: Trade to open
- `surface::VolatilitySurface`: Current market for pricing

# Returns
- The new Position

# Throws
- Error if IV not found on surface
"""
function add_position!(portfolio::Portfolio, trade::Trade,
                       surface::VolatilitySurface)::Position
    # Get current IV and price
    # Buy (direction=+1) uses ask, Sell (direction=-1) uses bid
    side = trade.direction > 0 ? :ask : :bid
    σ = find_vol(surface, trade.strike, trade.expiry; side=side)
    ismissing(σ) && error("IV not found for $(trade.strike)/$(trade.expiry)")

    T = time_to_expiry(trade.expiry, surface.timestamp)
    T <= 0.0 && error("Cannot open position on expired option")

    entry_price = vol_to_price(σ, surface.spot, trade.strike, T, trade.option_type)
    
    # Create position
    pos = Position(trade, entry_price, surface.spot, surface.timestamp, σ, portfolio.next_id)
    portfolio.next_id += 1
    
    # Update cash (buy = pay, sell = receive)
    cost = entry_price * trade.direction * trade.quantity * surface.spot
    portfolio.cash -= cost
    
    push!(portfolio.positions, pos)
    
    # Log trade
    push!(portfolio.trade_log, TradeRecord(
        pos.id, :open, surface.timestamp, trade, entry_price, σ, 0.0
    ))
    
    return pos
end

"""
    close_position!(portfolio, position, surface) -> Float64

Close an existing position at current market prices.

# Returns
- Realized P&L for this position
"""
function close_position!(portfolio::Portfolio, position::Position,
                         surface::VolatilitySurface)::Float64
    # Find position in portfolio
    idx = findfirst(p -> p.id == position.id, portfolio.positions)
    idx === nothing && error("Position not found in portfolio")

    trade = position.trade

    # Get exit price
    # Closing long (direction=+1) means selling → use bid
    # Closing short (direction=-1) means buying back → use ask
    side = trade.direction > 0 ? :bid : :ask
    σ = find_vol(surface, trade.strike, trade.expiry; side=side)
    T = time_to_expiry(trade.expiry, surface.timestamp)

    exit_price = if T <= 0.0 || ismissing(σ)
        # Option expired or no IV - use intrinsic
        intrinsic_payoff(trade, surface.spot) / surface.spot
    else
        vol_to_price(σ, surface.spot, trade.strike, T, trade.option_type)
    end
    
    exit_value = exit_price * trade.direction * trade.quantity * surface.spot
    entry_value = position.entry_price * trade.direction * trade.quantity * position.entry_spot
    pnl = exit_value - entry_value

    # Update cash (closing reverses direction)
    portfolio.cash += exit_value
    
    # Update realized P&L
    portfolio.realized_pnl += pnl
    
    # Remove from positions
    deleteat!(portfolio.positions, idx)
    
    # Log trade
    exit_vol = ismissing(σ) ? 0.0 : σ
    push!(portfolio.trade_log, TradeRecord(
        position.id, :close, surface.timestamp, trade, exit_price, exit_vol, pnl
    ))
    
    return pnl
end

"""
    close_all!(portfolio, surface) -> Float64

Close all open positions. Returns total realized P&L.
"""
function close_all!(portfolio::Portfolio, surface::VolatilitySurface)::Float64
    total_pnl = 0.0
    while !isempty(portfolio.positions)
        pos = portfolio.positions[1]
        total_pnl += close_position!(portfolio, pos, surface)
    end
    return total_pnl
end

"""
    intrinsic_value(trade, spot) -> Float64

Calculate intrinsic value of an option at given spot.
"""
function intrinsic_value(trade::Trade, spot::Float64)::Float64
    # Backwards-compatible helper: signed payoff (USD) for the whole position.
    return intrinsic_payoff(trade, spot) * trade.direction * trade.quantity
end

"""
    intrinsic_payoff(trade, spot) -> Float64

Intrinsic payoff (USD) per contract, independent of direction and quantity.
"""
function intrinsic_payoff(trade::Trade, spot::Float64)::Float64
    if trade.option_type == Call
        return max(spot - trade.strike, 0.0)
    else
        return max(trade.strike - spot, 0.0)
    end
end

# ============================================================================
# Mark-to-Market
# ============================================================================

"""
    position_value(position, surface) -> Float64

Calculate current market value of a position.
"""
function position_value(position::Position, surface::VolatilitySurface)::Float64
    trade = position.trade
    σ = find_vol(surface, trade.strike, trade.expiry)
    T = time_to_expiry(trade.expiry, surface.timestamp)
    
    if T <= 0.0 || ismissing(σ)
        return intrinsic_payoff(trade, surface.spot) * trade.direction * trade.quantity
    end
    
    price = vol_to_price(σ, surface.spot, trade.strike, T, trade.option_type)
    return price * trade.direction * trade.quantity * surface.spot
end

"""
    position_pnl(position, surface) -> Float64

Calculate unrealized P&L for a position.
"""
function position_pnl(position::Position, surface::VolatilitySurface)::Float64
    current_value = position_value(position, surface)
    entry_value = position.entry_price * position.trade.direction *
                  position.trade.quantity * position.entry_spot
    return current_value - entry_value
end

"""
    mark_to_market(portfolio, surface) -> PortfolioSnapshot

Calculate current portfolio state at given market prices.
"""
function mark_to_market(portfolio::Portfolio, 
                        surface::VolatilitySurface)::PortfolioSnapshot
    # Sum position values
    mtm_value = portfolio.cash
    unrealized_pnl = 0.0
    total_delta = 0.0
    total_vega = 0.0
    
    for pos in portfolio.positions
        mtm_value += position_value(pos, surface)
        unrealized_pnl += position_pnl(pos, surface)
        
        # Calculate Greeks (simplified - using mark vol)
        trade = pos.trade
        σ = find_vol(surface, trade.strike, trade.expiry)
        T = time_to_expiry(trade.expiry, surface.timestamp)
        
        if T > 0.0 && !ismissing(σ)
            # Delta approximation: ∂Price/∂S ≈ N(d1) for calls
            d = position_delta(pos, surface)
            v = position_vega(pos, surface)
            total_delta += d
            total_vega += v
        end
    end
    
    return PortfolioSnapshot(
        surface.timestamp,
        length(portfolio.positions),
        mtm_value,
        unrealized_pnl,
        portfolio.realized_pnl,
        portfolio.cash,
        total_delta,
        total_vega
    )
end

"""
    record_snapshot!(portfolio, surface)

Record a MTM snapshot in portfolio history.
"""
function record_snapshot!(portfolio::Portfolio, surface::VolatilitySurface)
    snapshot = mark_to_market(portfolio, surface)
    push!(portfolio.history, snapshot)
    return snapshot
end

# ============================================================================
# Greeks (simplified Black-76 approximations)
# ============================================================================

"""
    position_delta(position, surface) -> Float64

Calculate delta exposure for a position.
"""
function position_delta(position::Position, surface::VolatilitySurface)::Float64
    trade = position.trade
    σ = find_vol(surface, trade.strike, trade.expiry)
    T = time_to_expiry(trade.expiry, surface.timestamp)
    
    if T <= 0.0 || ismissing(σ)
        # At expiry: delta is 1 (ITM) or 0 (OTM)
        if trade.option_type == Call
            d = surface.spot > trade.strike ? 1.0 : 0.0
        else
            d = surface.spot < trade.strike ? -1.0 : 0.0
        end
        return d * trade.direction * trade.quantity
    end
    
    # Black-76 delta: N(d1) for calls, N(d1) - 1 for puts
    F = surface.spot
    K = trade.strike
    sqrtT = sqrt(T)
    d1 = (log(F / K) + 0.5 * σ^2 * T) / (σ * sqrtT)
    
    N = Normal()
    delta = if trade.option_type == Call
        cdf(N, d1)
    else
        cdf(N, d1) - 1.0
    end
    
    return delta * trade.direction * trade.quantity
end

"""
    position_vega(position, surface) -> Float64

Calculate vega exposure for a position.
"""
function position_vega(position::Position, surface::VolatilitySurface)::Float64
    trade = position.trade
    σ = find_vol(surface, trade.strike, trade.expiry)
    T = time_to_expiry(trade.expiry, surface.timestamp)
    
    if T <= 0.0 || ismissing(σ)
        return 0.0
    end
    
    vega = black76_vega(surface.spot, trade.strike, T, σ, trade.option_type)
    return vega * trade.direction * trade.quantity
end

# ============================================================================
# Queries
# ============================================================================

"""
    num_positions(portfolio) -> Int

Get number of open positions.
"""
num_positions(portfolio::Portfolio) = length(portfolio.positions)

"""
    total_value(portfolio, surface) -> Float64

Get total portfolio value at current market.
"""
function total_value(portfolio::Portfolio, surface::VolatilitySurface)::Float64
    return mark_to_market(portfolio, surface).mtm_value
end

"""
    get_positions(portfolio; underlying=nothing, option_type=nothing)

Get positions with optional filters.
"""
function get_positions(portfolio::Portfolio; 
                       underlying::Union{Underlying,Nothing}=nothing,
                       option_type::Union{OptionType,Nothing}=nothing)::Vector{Position}
    positions = portfolio.positions
    
    if underlying !== nothing
        positions = filter(p -> p.trade.underlying == underlying, positions)
    end
    if option_type !== nothing
        positions = filter(p -> p.trade.option_type == option_type, positions)
    end
    
    return positions
end

"""
    positions_expiring(portfolio, date) -> Vector{Position}

Get positions expiring on given date.
"""
function positions_expiring(portfolio::Portfolio, date::Date)::Vector{Position}
    filter(p -> Date(p.trade.expiry) == date, portfolio.positions)
end
