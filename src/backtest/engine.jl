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
    backtest_strategy(strategy, surfaces::Dict{DateTime, VolatilitySurface}) -> (positions, pnl)

Event-driven backtest for scheduled strategies. For each scheduled entry time,
the strategy is executed at the next available surface (first timestamp >= entry time).
Positions are settled at their expiry using the surface spot at that time.

# Arguments
- `strategy`: A ScheduledStrategy
- `surfaces`: Dict mapping timestamp → VolatilitySurface

# Returns
- `positions::Vector{Position}`: all entered positions
- `pnl::Vector{Union{Missing, Float64}}`: realized P&L per position; `missing` if no surface at expiry
"""
function backtest_strategy(
    strategy::ScheduledStrategy,
    surfaces::Dict{DateTime, VolatilitySurface}
)
    ts = sort(collect(keys(surfaces)))
    entry_times = entry_schedule(strategy)

    all_positions = Position[]
    all_pnl = Union{Missing, Float64}[]

    for entry_time in entry_times
        # Find next available surface
        idx = findfirst(t -> t >= entry_time, ts)
        idx === nothing && continue

        surface = surfaces[ts[idx]]
        positions = entry_positions(strategy, surface)

        for pos in positions
            # Settle at expiry
            expiry_surface = get(surfaces, pos.trade.expiry, nothing)
            pnl = expiry_surface === nothing ? missing : settle(pos, expiry_surface.spot)

            push!(all_positions, pos)
            push!(all_pnl, pnl)
        end
    end

    return all_positions, all_pnl
end

"""
    backtest_strategy(strategy, surfaces, spots) -> (positions, pnl)

Event-driven backtest for scheduled strategies using a spot time series for
settlement. This allows expiry settlement without requiring a full surface
at the expiry timestamp.

# Arguments
- `strategy`: A ScheduledStrategy
- `surfaces`: Dict mapping timestamp -> VolatilitySurface (for entries)
- `spots`: Dict mapping timestamp -> spot price (for settlement)

# Returns
- `positions::Vector{Position}`: all entered positions
- `pnl::Vector{Union{Missing, Float64}}`: realized P&L per position; `missing` if no spot at expiry
"""
function backtest_strategy(
    strategy::ScheduledStrategy,
    surfaces::Dict{DateTime, VolatilitySurface},
    spots::AbstractDict{DateTime, Float64}
)
    ts = sort(collect(keys(surfaces)))
    entry_times = entry_schedule(strategy)

    all_positions = Position[]
    all_pnl = Union{Missing, Float64}[]

    for entry_time in entry_times
        # Find next available surface
        idx = findfirst(t -> t >= entry_time, ts)
        idx === nothing && continue

        surface = surfaces[ts[idx]]
        positions = entry_positions(strategy, surface)

        for pos in positions
            settlement_spot = get(spots, pos.trade.expiry, missing)
            pnl = ismissing(settlement_spot) ? missing : settle(pos, settlement_spot)

            push!(all_positions, pos)
            push!(all_pnl, pnl)
        end
    end

    return all_positions, all_pnl
end
