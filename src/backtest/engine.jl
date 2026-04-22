# Backtest engine (minimal, functional)

using Dates

"""
    ScheduledStrategy

Abstract strategy with a fixed entry schedule. Implement `entry_schedule` and
`entry_positions`.
"""
abstract type ScheduledStrategy end

"""
    BacktestResult

Result container from `backtest_strategy`.

# Fields
- `positions::Vector{Position}`: all entered positions
- `pnl::Vector{Union{Missing, Float64}}`: realized P&L per position; `missing` if no settlement data
"""
struct BacktestResult
    positions::Vector{Position}
    pnl::Vector{Union{Missing, Float64}}
end

"""
    entry_schedule(strategy) -> Vector{DateTime}

Return the list of timestamps at which this strategy should enter trades.
"""
function entry_schedule(::ScheduledStrategy)::Vector{DateTime}
    error("entry_schedule not implemented for this strategy")
end

"""
    entry_positions(strategy, surface) -> Vector{Position}

Return the positions to enter at the given surface (legacy 2-arg form).
"""
function entry_positions(::ScheduledStrategy, ::VolatilitySurface)::Vector{Position}
    error("entry_positions not implemented for this strategy")
end

"""
    entry_positions(strategy, surface, history::BacktestDataSource) -> Vector{Position}

Return the positions to enter at the given surface. `history` is a
`BacktestDataSource` filtered to timestamps <= the current entry time,
providing access to historical surfaces and spot prices without look-ahead.

Strategies that don't need history can implement the 2-arg form instead;
the default 3-arg fallback delegates to it.
"""
entry_positions(s::ScheduledStrategy, surface::VolatilitySurface, ::BacktestDataSource) =
    entry_positions(s, surface)

"""
    each_entry(f, source, expiry_interval, schedule; clear_cache=false)

Iterate over scheduled entry timestamps, resolving each to a surface and expiry.
Calls `f(ctx::StrikeSelectionContext, settlement)` for each valid entry.
`settlement` is `Union{Float64, Missing}`.

When `clear_cache=true`, `clear_cache!(source)` is called after every callback
(success or exception), which prevents unbounded cache growth during long
single-pass sweeps. No-op for sources without caches.

This is the shared iteration core used by both `backtest_strategy` and
training data generation.
"""
function each_entry(
    f,
    source::BacktestDataSource,
    expiry_interval::Period,
    schedule::Vector{DateTime};
    clear_cache::Bool=false,
)
    ts = available_timestamps(source)
    for entry_time in schedule
        idx = findfirst(t -> t >= entry_time, ts)
        idx === nothing && continue

        surface = get_surface(source, ts[idx])
        surface === nothing && continue

        expiry_info = _select_expiry(expiry_interval, surface)
        expiry_info === nothing && continue
        expiry = expiry_info[1]

        history = HistoricalView(source, ts[idx])
        ctx = StrikeSelectionContext(surface, expiry, history)
        settlement = get_settlement_spot(source, expiry)
        try
            f(ctx, settlement)
        finally
            clear_cache && clear_cache!(source)
        end
    end
end

"""
    backtest_strategy(strategy, source::BacktestDataSource) -> BacktestResult

Event-driven backtest for scheduled strategies. For each scheduled entry time,
the strategy is executed at the next available surface (first timestamp >= entry time).
Positions are settled at their expiry using `get_settlement_spot`.

# Arguments
- `strategy`: A `ScheduledStrategy`
- `source`: A `BacktestDataSource` providing surfaces and settlement spots

# Returns
- `BacktestResult` with `positions` and `pnl` fields
"""
function backtest_strategy(
    strategy::ScheduledStrategy,
    source::BacktestDataSource
)
    ts = available_timestamps(source)
    entry_times = entry_schedule(strategy)

    all_positions = Position[]
    all_pnl = Union{Missing, Float64}[]

    for entry_time in entry_times
        idx = findfirst(t -> t >= entry_time, ts)
        idx === nothing && continue

        surface = get_surface(source, ts[idx])
        surface === nothing && continue
        history = HistoricalView(source, ts[idx])
        positions = entry_positions(strategy, surface, history)

        for pos in positions
            settlement_spot = get_settlement_spot(source, pos.trade.expiry)
            pnl = ismissing(settlement_spot) ? missing : settle(pos, settlement_spot)

            push!(all_positions, pos)
            push!(all_pnl, pnl)
        end
    end

    return BacktestResult(all_positions, all_pnl)
end

"""
    backtest_strategy(strategy, surfaces, spots) -> BacktestResult

Convenience method: wraps pre-loaded dictionaries in a `DictDataSource`.
"""
function backtest_strategy(
    strategy::ScheduledStrategy,
    surfaces::AbstractDict{DateTime, VolatilitySurface},
    spots::AbstractDict{DateTime, Float64}
)
    backtest_strategy(strategy, DictDataSource(surfaces, spots))
end
