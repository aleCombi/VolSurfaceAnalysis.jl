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
- `short_sigmas::Float64`: Short legs at +/- N sigma from spot
- `long_sigmas::Float64`: Long legs (wings) at +/- N sigma from spot
- `rate::Float64`: Risk-free rate for cost-of-carry forward
- `div_yield::Float64`: Dividend yield for cost-of-carry forward
- `quantity::Float64`: Contracts per leg
- `tau_tol::Float64`: Tolerance for expiry tenor mismatch (used for warnings)
- `debug::Bool`: Emit diagnostics when entries fail
- `strike_selector`: Optional strike selector. If `nothing`, uses default
  sigma-based logic. Otherwise, must be callable as `f(ctx)` and return
  `(short_put, short_call, long_put, long_call)` or `nothing`.
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

    strikes_tuple = if strategy.strike_selector === nothing
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

    if strikes_tuple === nothing
        if strategy.debug
            println("No entry: invalid condor strikes (timestamp=$(surface.timestamp), spot=$(surface.spot))")
        end
        return Position[]
    end

    short_put_K, short_call_K, long_put_K, long_call_K = strikes_tuple

    trades = Trade[
        Trade(surface.underlying, short_put_K, expiry, Put; direction=-1, quantity=strategy.quantity),
        Trade(surface.underlying, short_call_K, expiry, Call; direction=-1, quantity=strategy.quantity),
        Trade(surface.underlying, long_put_K, expiry, Put; direction=1, quantity=strategy.quantity),
        Trade(surface.underlying, long_call_K, expiry, Call; direction=1, quantity=strategy.quantity),
    ]

    return _open_positions(trades, surface; debug=strategy.debug)
end
