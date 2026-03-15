# Iron Condor Strategy (scheduled)

using Dates

"""
    IronCondorStrategy

Scheduled iron condor strategy with pluggable strike selection.

# Fields
- `schedule::Vector{DateTime}`: Entry timestamps
- `expiry_interval::Period`: Time from entry to expiry (e.g., Day(1))
- `strike_selector`: Callable `f(ctx) -> (sp_K, sc_K, lp_K, lc_K) | nothing`.
  `ctx` is a named tuple `(surface, expiry, history)`.
- `quantity::Float64`: Contracts per leg
- `debug::Bool`: Emit diagnostics when entries fail
"""
struct IronCondorStrategy{F} <: ScheduledStrategy
    schedule::Vector{DateTime}
    expiry_interval::Period
    strike_selector::F
    quantity::Float64
    debug::Bool
end

entry_schedule(strategy::IronCondorStrategy)::Vector{DateTime} = strategy.schedule

function IronCondorStrategy(
    schedule::Vector{DateTime},
    expiry_interval::Period,
    strike_selector;
    quantity::Float64=1.0,
    debug::Bool=false
)
    return IronCondorStrategy(
        schedule,
        expiry_interval,
        strike_selector,
        quantity,
        debug
    )
end

function entry_positions(
    strategy::IronCondorStrategy,
    surface::VolatilitySurface,
    history::BacktestDataSource=DictDataSource(Dict{DateTime,VolatilitySurface}(), Dict{DateTime,Float64}())
)::Vector{Position}
    expiry_info = _select_expiry(strategy.expiry_interval, surface)
    if expiry_info === nothing
        if strategy.debug
            println("No entry: no valid expiry for timestamp=$(surface.timestamp)")
        end
        return Position[]
    end
    expiry = expiry_info[1]

    ctx = StrikeSelectionContext(surface, expiry, history)

    selector_result = strategy.strike_selector(ctx)

    if selector_result === nothing
        if strategy.debug
            println("No entry: invalid condor strikes (timestamp=$(surface.timestamp), spot=$(surface.spot))")
        end
        return Position[]
    end

    sp_K, sc_K, lp_K, lc_K = selector_result

    trades = Trade[
        Trade(surface.underlying, sp_K, expiry, Put; direction=-1, quantity=strategy.quantity),
        Trade(surface.underlying, sc_K, expiry, Call; direction=-1, quantity=strategy.quantity),
        Trade(surface.underlying, lp_K, expiry, Put; direction=1, quantity=strategy.quantity),
        Trade(surface.underlying, lc_K, expiry, Call; direction=1, quantity=strategy.quantity),
    ]

    return _open_positions(trades, surface; debug=strategy.debug)
end
