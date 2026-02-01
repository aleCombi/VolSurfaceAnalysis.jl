# Short Strangle Strategy (scheduled)

using Dates

"""
    ShortStrangleStrategy

Scheduled strangle strategy with ATM-IV-proportional strike selection.
At entry, computes ATM implied vol from the surface and trades OTM put/call
at +/- N sigma from spot.

# Fields
- `schedule::Vector{DateTime}`: Entry timestamps
- `expiry_interval::Period`: Time from entry to expiry (e.g., Day(1))
- `short_sigmas::Float64`: Legs at +/- N sigma from spot
- `rate::Float64`: Risk-free rate for cost-of-carry forward
- `div_yield::Float64`: Dividend yield for cost-of-carry forward
- `quantity::Float64`: Base contracts per leg (scaled by |position_size|)
- `tau_tol::Float64`: Tolerance for expiry tenor mismatch (used for warnings)
- `debug::Bool`: Emit diagnostics when entries fail
- `strike_selector`: Optional strike selector. If `nothing`, uses default
  sigma-based logic. Otherwise, must be callable as `f(ctx)` and return:
  - `(put_K, call_K)` for fixed quantity short strangle, or
  - `(put_K, call_K, position_size)` for dynamic sizing where:
    - position_size > 0: short vol (sell strangle), quantity scaled by size
    - position_size < 0: long vol (buy strangle), quantity scaled by |size|
    - position_size ≈ 0: skip trade
  Returns `nothing` to skip entry.
  The `ctx` is a named tuple with:
  `surface`, `expiry`, `tau`, `recs`, `put_strikes`, `call_strikes`.
"""
struct ShortStrangleStrategy{F} <: ScheduledStrategy
    schedule::Vector{DateTime}
    expiry_interval::Period
    short_sigmas::Float64
    rate::Float64
    div_yield::Float64
    quantity::Float64
    tau_tol::Float64
    debug::Bool
    strike_selector::F
end

entry_schedule(strategy::ShortStrangleStrategy)::Vector{DateTime} = strategy.schedule

function ShortStrangleStrategy(
    schedule::Vector{DateTime},
    expiry_interval::Period,
    short_sigmas::Float64,
    rate::Float64,
    div_yield::Float64,
    quantity::Float64,
    tau_tol::Float64;
    strike_selector=nothing
)
    return ShortStrangleStrategy(
        schedule,
        expiry_interval,
        short_sigmas,
        rate,
        div_yield,
        quantity,
        tau_tol,
        false,
        strike_selector
    )
end

function ShortStrangleStrategy(
    schedule::Vector{DateTime},
    expiry_interval::Period,
    short_sigmas::Float64;
    rate::Float64=0.0,
    div_yield::Float64=0.0,
    quantity::Float64=1.0,
    tau_tol::Float64=1e-6,
    debug::Bool=false,
    strike_selector=nothing
)
    return ShortStrangleStrategy(
        schedule,
        expiry_interval,
        short_sigmas,
        rate,
        div_yield,
        quantity,
        tau_tol,
        debug,
        strike_selector
    )
end

function entry_positions(
    strategy::ShortStrangleStrategy,
    surface::VolatilitySurface
)::Vector{Position}
    expiry_info = _select_expiry(strategy.expiry_interval, surface)
    expiry_info === nothing && return Position[]
    expiry, tau_target, tau_closest = expiry_info

    if strategy.debug && abs(tau_closest - tau_target) > strategy.tau_tol
        println("Warning: large tenor mismatch target_tau=$(tau_target), closest_tau=$(tau_closest)")
    end

    recs = filter(r -> r.expiry == expiry, surface.records)
    isempty(recs) && return Position[]

    put_strikes = sort(unique(r.strike for r in recs if r.option_type == Put))
    call_strikes = sort(unique(r.strike for r in recs if r.option_type == Call))
    if isempty(put_strikes) || isempty(call_strikes)
        return Position[]
    end

    ctx = (
        surface=surface,
        expiry=expiry,
        tau=tau_closest,
        recs=recs,
        put_strikes=put_strikes,
        call_strikes=call_strikes
    )

    selector_result = if strategy.strike_selector === nothing
        _sigma_strangle_strikes(
            ctx,
            strategy.short_sigmas,
            strategy.rate,
            strategy.div_yield;
            debug=strategy.debug
        )
    else
        strategy.strike_selector(ctx)
    end

    if selector_result === nothing
        if strategy.debug
            println("No entry: invalid strangle strikes (timestamp=$(surface.timestamp), spot=$(surface.spot))")
        end
        return Position[]
    end

    # Handle both 2-tuple (strikes only) and 3-tuple (strikes + position size) returns
    put_K, call_K, position_size = if length(selector_result) == 3
        selector_result
    else
        (selector_result[1], selector_result[2], 1.0)
    end

    # Scale quantity by absolute position size
    # position_size > 0: short vol (sell strangle)
    # position_size < 0: long vol (buy strangle)
    effective_quantity = strategy.quantity * abs(position_size)

    # Skip trade if position size is effectively zero
    if effective_quantity < 1e-6
        if strategy.debug
            println("No entry: position_size=$(position_size) too small (timestamp=$(surface.timestamp))")
        end
        return Position[]
    end

    # Direction: -1 for short (sell), +1 for long (buy)
    direction = position_size >= 0 ? -1 : 1

    trades = Trade[
        Trade(surface.underlying, put_K, expiry, Put; direction=direction, quantity=effective_quantity),
        Trade(surface.underlying, call_K, expiry, Call; direction=direction, quantity=effective_quantity),
    ]

    return _open_positions(trades, surface; debug=strategy.debug)
end
