# Iron Condor Strategy (scheduled)

using Dates

"""
    IronCondorStrategy

Scheduled iron condor strategy with ATM-IV-proportional strike selection.
At entry, computes ATM implied vol from the surface and places legs at
configurable multiples of the implied sigma move.

# Fields
- `schedule::Vector{DateTime}`: Entry timestamps
- `expiry_interval::Period`: Time from entry to expiry (e.g., Day(1))
- `short_sigmas::Float64`: Inner legs at +/- N sigma from spot
- `long_sigmas::Float64`: Outer legs (wings) at +/- N sigma from spot
- `rate::Float64`: Risk-free rate for cost-of-carry forward
- `div_yield::Float64`: Dividend yield for cost-of-carry forward
- `quantity::Float64`: Base contracts per leg (scaled by |position_size|)
- `tau_tol::Float64`: Tolerance for expiry tenor mismatch (used for warnings)
- `debug::Bool`: Emit diagnostics when entries fail
- `strike_selector`: Optional strike selector. If `nothing`, uses default
  sigma-based logic. Otherwise, must be callable as `f(ctx)` and return:
  - `(inner_put, inner_call, outer_put, outer_call)` for fixed quantity, or
  - `(inner_put, inner_call, outer_put, outer_call, position_size)` where:
    - position_size > 0: short condor (sell inner, buy outer)
    - position_size < 0: long condor (buy inner, sell outer)
    - position_size ≈ 0: skip trade
  Returns `nothing` to skip entry.
  The `ctx` is a named tuple with:
  `surface`, `expiry`, `tau`, `recs`, `put_strikes`, `call_strikes`.
"""
struct IronCondorStrategy{F} <: ScheduledStrategy
    schedule::Vector{DateTime}
    expiry_interval::Period
    short_sigmas::Float64
    long_sigmas::Float64
    rate::Float64
    div_yield::Float64
    quantity::Float64
    tau_tol::Float64
    debug::Bool
    strike_selector::F
end

entry_schedule(strategy::IronCondorStrategy)::Vector{DateTime} = strategy.schedule

function IronCondorStrategy(
    schedule::Vector{DateTime},
    expiry_interval::Period,
    short_sigmas::Float64,
    long_sigmas::Float64,
    rate::Float64,
    div_yield::Float64,
    quantity::Float64,
    tau_tol::Float64;
    strike_selector=nothing
)
    return IronCondorStrategy(
        schedule,
        expiry_interval,
        short_sigmas,
        long_sigmas,
        rate,
        div_yield,
        quantity,
        tau_tol,
        false,
        strike_selector
    )
end

function IronCondorStrategy(
    schedule::Vector{DateTime},
    expiry_interval::Period,
    short_sigmas::Float64,
    long_sigmas::Float64;
    rate::Float64=0.0,
    div_yield::Float64=0.0,
    quantity::Float64=1.0,
    tau_tol::Float64=1e-6,
    debug::Bool=false,
    strike_selector=nothing
)
    return IronCondorStrategy(
        schedule,
        expiry_interval,
        short_sigmas,
        long_sigmas,
        rate,
        div_yield,
        quantity,
        tau_tol,
        debug,
        strike_selector
    )
end

function entry_positions(
    strategy::IronCondorStrategy,
    surface::VolatilitySurface
)::Vector{Position}
    expiry_info = _select_expiry(strategy.expiry_interval, surface)
    if expiry_info === nothing
        if strategy.debug
            println("No entry: no valid expiry for timestamp=$(surface.timestamp)")
        end
        return Position[]
    end
    expiry, tau_target, tau_closest = expiry_info

    if strategy.debug && abs(tau_closest - tau_target) > strategy.tau_tol
        println("Warning: large tenor mismatch target_tau=$(tau_target), closest_tau=$(tau_closest)")
    end

    recs = filter(r -> r.expiry == expiry, surface.records)
    if isempty(recs)
        if strategy.debug
            println("No entry: no records for expiry (timestamp=$(surface.timestamp), expiry=$(expiry))")
        end
        return Position[]
    end

    put_strikes = sort(unique(r.strike for r in recs if r.option_type == Put))
    call_strikes = sort(unique(r.strike for r in recs if r.option_type == Call))

    if isempty(put_strikes) || isempty(call_strikes)
        if strategy.debug
            println("No entry: missing puts or calls for expiry (timestamp=$(surface.timestamp), expiry=$(expiry))")
        end
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
        _sigma_condor_strikes(
            ctx,
            strategy.short_sigmas,
            strategy.long_sigmas,
            strategy.rate,
            strategy.div_yield;
            debug=strategy.debug
        )
    else
        strategy.strike_selector(ctx)
    end

    if selector_result === nothing
        if strategy.debug
            println("No entry: invalid condor strikes (timestamp=$(surface.timestamp), spot=$(surface.spot))")
        end
        return Position[]
    end

    # Handle both 4-tuple (strikes only) and 5-tuple (strikes + position size) returns
    inner_put_K, inner_call_K, outer_put_K, outer_call_K, position_size = if length(selector_result) == 5
        selector_result
    else
        (selector_result[1], selector_result[2], selector_result[3], selector_result[4], 1.0)
    end

    # Scale quantity by absolute position size
    # position_size > 0: short iron condor (sell inner, buy outer)
    # position_size < 0: long iron condor (buy inner, sell outer)
    effective_quantity = strategy.quantity * abs(position_size)

    # Skip trade if position size is effectively zero
    if effective_quantity < 1e-6
        if strategy.debug
            println("No entry: position_size=$(position_size) too small (timestamp=$(surface.timestamp))")
        end
        return Position[]
    end

    # Inner direction: -1 for short condor (sell inner), +1 for long condor (buy inner)
    inner_dir = position_size >= 0 ? -1 : 1
    outer_dir = -inner_dir  # Opposite direction for wings

    trades = Trade[
        Trade(surface.underlying, inner_put_K, expiry, Put; direction=inner_dir, quantity=effective_quantity),
        Trade(surface.underlying, inner_call_K, expiry, Call; direction=inner_dir, quantity=effective_quantity),
        Trade(surface.underlying, outer_put_K, expiry, Put; direction=outer_dir, quantity=effective_quantity),
        Trade(surface.underlying, outer_call_K, expiry, Call; direction=outer_dir, quantity=effective_quantity),
    ]

    return _open_positions(trades, surface; debug=strategy.debug)
end
