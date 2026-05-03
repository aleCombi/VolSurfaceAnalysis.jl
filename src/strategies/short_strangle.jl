# Short Strangle Strategy (scheduled, naked — no wings)

using Dates

"""
    ShortStrangleStrategy

Scheduled short strangle strategy: short OTM put + short OTM call, no long
wings. Pluggable strike selection and sizing.

# Fields
- `schedule::Vector{DateTime}`: Entry timestamps
- `expiry_interval::Period`: Time from entry to expiry (e.g., `Day(1)`)
- `strike_selector`: Callable `f(ctx) -> (sp_K, sc_K) | nothing`.
  `ctx` is a `StrikeSelectionContext(surface, expiry, history)`.
- `sizer`: Callable `f(ctx) -> Float64` returning the quantity to trade.
  Defaults to `FixedSize(1.0)`. Use `MLSizer` for ML-modulated sizing.

Selectors return a 2-tuple `(short_put_strike, short_call_strike)` rather
than the 4-tuple used by `IronCondorStrategy`. Use
`delta_strangle_selector(put_delta, call_delta)` for the standard
delta-based picker.
"""
struct ShortStrangleStrategy{F,S} <: ScheduledStrategy
    schedule::Vector{DateTime}
    expiry_interval::Period
    strike_selector::F
    sizer::S
end

entry_schedule(strategy::ShortStrangleStrategy)::Vector{DateTime} = strategy.schedule

function ShortStrangleStrategy(
    schedule::Vector{DateTime},
    expiry_interval::Period,
    strike_selector;
    sizer=FixedSize(1.0)
)
    return ShortStrangleStrategy(schedule, expiry_interval, strike_selector, sizer)
end

function entry_positions(
    strategy::ShortStrangleStrategy,
    surface::VolatilitySurface,
    history::BacktestDataSource=DictDataSource(Dict{DateTime,VolatilitySurface}(), Dict{DateTime,Float64}())
)::Vector{Position}
    expiry_info = _select_expiry(strategy.expiry_interval, surface)
    if expiry_info === nothing
        @debug "No entry: no valid expiry for timestamp=$(surface.timestamp)"
        return Position[]
    end
    expiry = expiry_info[1]

    ctx = StrikeSelectionContext(surface, expiry, history)

    selector_result = strategy.strike_selector(ctx)
    if selector_result === nothing
        @debug "No entry: invalid strangle strikes (timestamp=$(surface.timestamp), spot=$(surface.spot))"
        return Position[]
    end
    sp_K, sc_K = selector_result

    quantity = strategy.sizer(ctx)
    if quantity <= 0.0
        @debug "No entry: sizer returned q=$(quantity) (timestamp=$(surface.timestamp))"
        return Position[]
    end

    trades = Trade[
        Trade(surface.underlying, sp_K, expiry, Put;  direction=-1, quantity=quantity),
        Trade(surface.underlying, sc_K, expiry, Call; direction=-1, quantity=quantity),
    ]
    return _open_positions(trades, surface)
end

"""
    delta_strangle_selector(put_delta, call_delta; rate=0.0, div_yield=0.0)

Return a callable `ctx -> (sp_K, sc_K) | nothing` that picks short strikes
by absolute delta. Asymmetric deltas allowed (e.g. 0.20 put / 0.05 call).
"""
function delta_strangle_selector(put_delta::Float64, call_delta::Float64;
                                 rate::Float64=0.0,
                                 div_yield::Float64=0.0)
    return function(ctx)
        _delta_strangle_strikes_asymmetric(
            ctx, put_delta, call_delta; rate=rate, div_yield=div_yield,
        )
    end
end

"""
    open_strangle_positions(ctx, sp_K, sc_K; direction=-1, quantity=1.0)
        -> Vector{Position}

Open the two legs of a strangle at `ctx.surface` / `ctx.expiry`:
a put at `sp_K` and a call at `sc_K`, both with the same `direction`
(`-1` = short / sell at bid, `+1` = long / buy at ask) and `quantity`.
Returns an empty `Vector{Position}` if either leg cannot be opened.
Callers should check `length(positions) == 2`.
"""
function open_strangle_positions(
    ctx,
    sp_K::Float64, sc_K::Float64;
    direction::Int=-1,
    quantity::Float64=1.0,
)::Vector{Position}
    surface = ctx.surface
    expiry = ctx.expiry
    trades = Trade[
        Trade(surface.underlying, sp_K, expiry, Put;  direction=direction, quantity=quantity),
        Trade(surface.underlying, sc_K, expiry, Call; direction=direction, quantity=quantity),
    ]
    return _open_positions(trades, surface)
end
