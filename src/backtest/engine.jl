# Backtest engine (minimal, functional)
# Strategy transforms a portfolio at time t into a portfolio at time t+epsilon.

using Dates

"""
    Strategy

Abstract strategy type. Implement `next_portfolio` to define how the portfolio
should evolve given the current portfolio, surface, and previous timestamp.
"""
abstract type Strategy end

"""
    next_portfolio(strategy, positions, surface, prev_timestamp) -> Vector{Position}

Return the portfolio at time t+epsilon given the portfolio at time t, the
market surface observed at time t, and the previous timestamp.

This function must be implemented by user strategies.
"""
function next_portfolio(
    ::Strategy,
    ::Vector{Position},
    ::VolatilitySurface,
    ::DateTime
)::Vector{Position}
    error("next_portfolio not implemented for this strategy")
end

"""
    ScheduledStrategy

Abstract strategy with a fixed entry schedule. Implement `entry_schedule` and
`entry_positions`. A default `next_portfolio` is provided.
"""
abstract type ScheduledStrategy <: Strategy end

"""
    entry_schedule(strategy) -> Vector{DateTime}

Return the list of timestamps at which this strategy should enter trades.
"""
function entry_schedule(::ScheduledStrategy)::Vector{DateTime}
    error("entry_schedule not implemented for this strategy")
end

"""
    entry_positions(strategy, surface) -> Vector{Position}

Return the positions to enter at the given surface. Positions can depend on both
the strategy and the surface.
"""
function entry_positions(::ScheduledStrategy, ::VolatilitySurface)::Vector{Position}
    error("entry_positions not implemented for this strategy")
end

"""
    next_portfolio(strategy, positions, surface, prev_timestamp) -> Vector{Position}

Default implementation for scheduled strategies. `prev_timestamp` must always be
provided; for the first step, pass the current timestamp (or an earlier time if
you want entries at the first snapshot).
"""
function next_portfolio(
    strategy::ScheduledStrategy,
    positions::Vector{Position},
    surface::VolatilitySurface,
    prev_timestamp::DateTime
)::Vector{Position}
    should_enter = any(
        t -> (t > prev_timestamp && t <= surface.timestamp),
        entry_schedule(strategy)
    )

    if should_enter
        new_positions = entry_positions(strategy, surface)
        return vcat(positions, new_positions)
    end
    return positions
end

"""
    BacktestResult

Minimal backtest result container.

# Fields
- `timestamps`: snapshot times
- `realized_pnl`: realized P&L at each step (from expired positions)
- `cumulative_pnl`: cumulative realized P&L
- `positions`: optional portfolio snapshots (when recording is enabled)
"""
struct BacktestResult
    timestamps::Vector{DateTime}
    realized_pnl::Vector{Float64}
    cumulative_pnl::Vector{Float64}
    positions::Union{Nothing, Vector{Vector{Position}}}
end

"""
    backtest_strategy(strategy, iter) -> (positions, pnl)

Event-driven backtest for scheduled strategies. For each scheduled entry time,
the strategy is executed at the next available surface tick (first timestamp
>= entry time). Positions are settled at their expiry using the surface spot at
that time.

Returns:
- `positions::Vector{Position}`: all entered positions (entry_timestamp reflects
  the actual tick used)
- `pnl::Vector{Union{Missing, Float64}}`: realized P&L per position; `missing`
  if no surface is available at expiry
"""
function backtest_strategy(strategy::ScheduledStrategy, iter::SurfaceIterator)
    ts = timestamps(iter)
    entry_times = entry_schedule(strategy)

    surfaces = filter(!isnothing, map(t -> _next_surface(iter, ts, t), entry_times))
    positions_by_surface = map(s -> entry_positions(strategy, s), surfaces)
    positions = reduce(vcat, positions_by_surface; init=Position[])
    pnl = map(pos -> _settle_at_expiry(iter, pos), positions)

    return positions, pnl
end

function _next_surface(
    iter::SurfaceIterator,
    ts::Vector{DateTime},
    entry_time::DateTime
)::Union{VolatilitySurface, Nothing}
    idx = findfirst(t -> t >= entry_time, ts)
    return idx === nothing ? nothing : surface_at(iter, idx)
end

function _settle_at_expiry(
    iter::SurfaceIterator,
    pos::Position
)::Union{Missing, Float64}
    expiry_surface = surface_at(iter, pos.trade.expiry)
    return expiry_surface === nothing ? missing : settle(pos, expiry_surface.spot)
end
