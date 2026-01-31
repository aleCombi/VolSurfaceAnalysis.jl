# Short Strangle Strategy (scheduled)

using Dates

"""
    ShortStrangleStrategy

Scheduled short strangle strategy with ATM-IV-proportional strike selection.
At entry, computes ATM implied vol from the surface and sells OTM put/call
at +/- N sigma from spot.

# Fields
- `schedule::Vector{DateTime}`: Entry timestamps
- `expiry_interval::Period`: Time from entry to expiry (e.g., Day(1))
- `short_sigmas::Float64`: Short legs at +/- N sigma from spot
- `rate::Float64`: Risk-free rate for cost-of-carry forward
- `div_yield::Float64`: Dividend yield for cost-of-carry forward
- `quantity::Float64`: Contracts per leg
- `tau_tol::Float64`: Tolerance for expiry tenor mismatch (used for warnings)
- `debug::Bool`: Emit diagnostics when entries fail
"""
struct ShortStrangleStrategy <: ScheduledStrategy
    schedule::Vector{DateTime}
    expiry_interval::Period
    short_sigmas::Float64
    rate::Float64
    div_yield::Float64
    quantity::Float64
    tau_tol::Float64
    debug::Bool
end

entry_schedule(strategy::ShortStrangleStrategy)::Vector{DateTime} = strategy.schedule

function ShortStrangleStrategy(
    schedule::Vector{DateTime},
    expiry_interval::Period,
    short_sigmas::Float64,
    rate::Float64,
    div_yield::Float64,
    quantity::Float64,
    tau_tol::Float64
)
    return ShortStrangleStrategy(
        schedule,
        expiry_interval,
        short_sigmas,
        rate,
        div_yield,
        quantity,
        tau_tol,
        false
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
    debug::Bool=false
)
    return ShortStrangleStrategy(
        schedule,
        expiry_interval,
        short_sigmas,
        rate,
        div_yield,
        quantity,
        tau_tol,
        debug
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

    atm_rec = recs[argmin([abs(r.strike - surface.spot) for r in recs])]
    ismissing(atm_rec.mark_price) && return Position[]

    F_atm = surface.spot * exp((strategy.rate - strategy.div_yield) * tau_closest)
    atm_iv = price_to_iv(atm_rec.mark_price, F_atm, atm_rec.strike, tau_closest, atm_rec.option_type; r=strategy.rate)
    if isnan(atm_iv) || atm_iv <= 0.0
        return Position[]
    end

    sigma_move = atm_iv * sqrt(tau_closest)
    short_put_pct = 1.0 - strategy.short_sigmas * sigma_move
    short_call_pct = 1.0 + strategy.short_sigmas * sigma_move

    put_strikes = sort(unique(r.strike for r in recs if r.option_type == Put))
    call_strikes = sort(unique(r.strike for r in recs if r.option_type == Call))
    if isempty(put_strikes) || isempty(call_strikes)
        return Position[]
    end

    target_put = surface.spot * short_put_pct
    target_call = surface.spot * short_call_pct

    short_put_K = _pick_otm_strike(put_strikes, surface.spot, target_put; side=:put)
    short_call_K = _pick_otm_strike(call_strikes, surface.spot, target_call; side=:call)

    trades = Trade[
        Trade(surface.underlying, short_put_K, expiry, Put; direction=-1, quantity=strategy.quantity),
        Trade(surface.underlying, short_call_K, expiry, Call; direction=-1, quantity=strategy.quantity),
    ]

    return _open_positions(trades, surface; debug=strategy.debug)
end
