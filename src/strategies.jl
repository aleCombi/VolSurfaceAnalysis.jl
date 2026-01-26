# Strategy implementations (scheduled)

using Dates

"""
    IronCondorStrategy

Scheduled iron condor strategy with fixed entry times and percentage-based
strike selection around spot.

# Fields
- `schedule::Vector{DateTime}`: Entry timestamps
- `expiry_interval::Period`: Time from entry to expiry (e.g., Day(1))
- `short_put_pct::Float64`: Spot multiplier for short put strike
- `short_call_pct::Float64`: Spot multiplier for short call strike
- `long_put_pct::Float64`: Spot multiplier for long put strike
- `long_call_pct::Float64`: Spot multiplier for long call strike
- `quantity::Float64`: Contracts per leg
- `tau_tol::Float64`: Tolerance for expiry tenor mismatch (used for warnings)
- `debug::Bool`: Emit diagnostics when entries fail
"""
struct IronCondorStrategy <: ScheduledStrategy
    schedule::Vector{DateTime}
    expiry_interval::Period
    short_put_pct::Float64
    short_call_pct::Float64
    long_put_pct::Float64
    long_call_pct::Float64
    quantity::Float64
    tau_tol::Float64
    debug::Bool
end

entry_schedule(strategy::IronCondorStrategy)::Vector{DateTime} = strategy.schedule

function IronCondorStrategy(
    schedule::Vector{DateTime},
    expiry_interval::Period,
    short_put_pct::Float64,
    short_call_pct::Float64,
    long_put_pct::Float64,
    long_call_pct::Float64,
    quantity::Float64,
    tau_tol::Float64
)
    return IronCondorStrategy(
        schedule,
        expiry_interval,
        short_put_pct,
        short_call_pct,
        long_put_pct,
        long_call_pct,
        quantity,
        tau_tol,
        false
    )
end

function IronCondorStrategy(
    schedule::Vector{DateTime},
    expiry_interval::Period,
    short_put_pct::Float64,
    short_call_pct::Float64,
    long_put_pct::Float64,
    long_call_pct::Float64;
    quantity::Float64=1.0,
    tau_tol::Float64=1e-6,
    debug::Bool=false
)
    return IronCondorStrategy(
        schedule,
        expiry_interval,
        short_put_pct,
        short_call_pct,
        long_put_pct,
        long_call_pct,
        quantity,
        tau_tol,
        debug
    )
end

function entry_positions(
    strategy::IronCondorStrategy,
    surface::VolatilitySurface
)::Vector{Position}
    expiry_target = surface.timestamp + strategy.expiry_interval
    tau_target = time_to_expiry(expiry_target, surface.timestamp)
    if tau_target <= 0.0
        if strategy.debug
            println("No entry: non-positive time to expiry (timestamp=$(surface.timestamp), expiry=$(expiry_target))")
        end
        return Position[]
    end

    expiries = unique(rec.expiry for rec in surface.records)
    if isempty(expiries)
        if strategy.debug
            println("No entry: surface has no records (timestamp=$(surface.timestamp))")
        end
        return Position[]
    end

    taus = [time_to_expiry(e, surface.timestamp) for e in expiries]
    idx = argmin(abs.(taus .- tau_target))
    expiry = expiries[idx]
    tau_closest = taus[idx]

    if strategy.debug && expiry != expiry_target
        println("Using closest expiry: target_tau=$(tau_target), closest_tau=$(tau_closest), expiry=$(expiry)")
    elseif strategy.debug && abs(tau_closest - tau_target) > strategy.tau_tol
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

    strikes_tuple = _condor_strikes(
        put_strikes,
        call_strikes,
        surface.spot,
        strategy.short_put_pct,
        strategy.short_call_pct,
        strategy.long_put_pct,
        strategy.long_call_pct
    )
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

function _condor_strikes(
    put_strikes::Vector{Float64},
    call_strikes::Vector{Float64},
    spot::Float64,
    short_put_pct::Float64,
    short_call_pct::Float64,
    long_put_pct::Float64,
    long_call_pct::Float64
)::Union{Nothing, Tuple{Float64,Float64,Float64,Float64}}
    target_short_put = spot * short_put_pct
    target_short_call = spot * short_call_pct
    target_long_put = spot * long_put_pct
    target_long_call = spot * long_call_pct

    otm_puts = filter(s -> s < spot, put_strikes)
    otm_calls = filter(s -> s > spot, call_strikes)

    short_put = !isempty(otm_puts) ? _nearest_strike(otm_puts, target_short_put) :
                _nearest_strike(put_strikes, target_short_put)
    short_call = !isempty(otm_calls) ? _nearest_strike(otm_calls, target_short_call) :
                 _nearest_strike(call_strikes, target_short_call)

    far_otm_puts = filter(s -> s < short_put, put_strikes)
    far_otm_calls = filter(s -> s > short_call, call_strikes)

    long_put = if !isempty(far_otm_puts)
        _nearest_strike(far_otm_puts, target_long_put)
    else
        !isempty(otm_puts) ? minimum(otm_puts) : _nearest_strike(put_strikes, target_long_put)
    end

    long_call = if !isempty(far_otm_calls)
        _nearest_strike(far_otm_calls, target_long_call)
    else
        !isempty(otm_calls) ? maximum(otm_calls) : _nearest_strike(call_strikes, target_long_call)
    end

    if !(long_put < short_put < spot < short_call < long_call)
        return nothing
    end

    return (short_put, short_call, long_put, long_call)
end

function _nearest_strike(strikes::Vector{Float64}, target::Float64)::Float64
    distances = abs.(strikes .- target)
    return strikes[argmin(distances)]
end
