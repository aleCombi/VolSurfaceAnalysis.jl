# Iron Condor Strategy (scheduled)

using Dates

"""
    FixedSize(quantity)

Constant sizing policy — always returns the same quantity.
Default sizer for `IronCondorStrategy`.
"""
struct FixedSize
    quantity::Float64
end
(f::FixedSize)(::StrikeSelectionContext) = f.quantity

"""
    IronCondorStrategy

Scheduled iron condor strategy with pluggable strike selection and sizing.

# Fields
- `schedule::Vector{DateTime}`: Entry timestamps
- `expiry_interval::Period`: Time from entry to expiry (e.g., Day(1))
- `strike_selector`: Callable `f(ctx) -> (sp_K, sc_K, lp_K, lc_K) | nothing`.
  `ctx` is a `StrikeSelectionContext(surface, expiry, history)`.
- `sizer`: Callable `f(ctx) -> Float64` returning the quantity to trade.
  Defaults to `FixedSize(1.0)`. Use `MLSizer` for ML-modulated sizing.
- `debug::Bool`: Emit diagnostics when entries fail
"""
struct IronCondorStrategy{F,S} <: ScheduledStrategy
    schedule::Vector{DateTime}
    expiry_interval::Period
    strike_selector::F
    sizer::S
    debug::Bool
end

entry_schedule(strategy::IronCondorStrategy)::Vector{DateTime} = strategy.schedule

function IronCondorStrategy(
    schedule::Vector{DateTime},
    expiry_interval::Period,
    strike_selector;
    sizer=FixedSize(1.0),
    debug::Bool=false
)
    return IronCondorStrategy(
        schedule,
        expiry_interval,
        strike_selector,
        sizer,
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
        strategy.debug && println("No entry: no valid expiry for timestamp=$(surface.timestamp)")
        return Position[]
    end
    expiry = expiry_info[1]

    ctx = StrikeSelectionContext(surface, expiry, history)

    selector_result = strategy.strike_selector(ctx)
    if selector_result === nothing
        strategy.debug && println("No entry: invalid condor strikes (timestamp=$(surface.timestamp), spot=$(surface.spot))")
        return Position[]
    end
    sp_K, sc_K, lp_K, lc_K = selector_result

    quantity = strategy.sizer(ctx)
    if quantity <= 0.0
        strategy.debug && println("No entry: sizer returned q=$(quantity) (timestamp=$(surface.timestamp))")
        return Position[]
    end

    trades = Trade[
        Trade(surface.underlying, sp_K, expiry, Put; direction=-1, quantity=quantity),
        Trade(surface.underlying, sc_K, expiry, Call; direction=-1, quantity=quantity),
        Trade(surface.underlying, lp_K, expiry, Put; direction=1, quantity=quantity),
        Trade(surface.underlying, lc_K, expiry, Call; direction=1, quantity=quantity),
    ]

    return _open_positions(trades, surface; debug=strategy.debug)
end
