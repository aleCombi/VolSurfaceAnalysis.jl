# Strategy helpers

function _nearest_strike(strikes::Vector{Float64}, target::Float64)::Float64
    distances = abs.(strikes .- target)
    return strikes[argmin(distances)]
end

function _pick_otm_strike(
    strikes::Vector{Float64},
    spot::Float64,
    target::Float64;
    side::Symbol
)::Float64
    otm = side == :put ? filter(s -> s < spot, strikes) : filter(s -> s > spot, strikes)
    return !isempty(otm) ? _nearest_strike(otm, target) : _nearest_strike(strikes, target)
end

function _select_expiry(
    expiry_interval::Period,
    surface::VolatilitySurface
)::Union{Nothing,Tuple{DateTime,Float64,Float64}}
    expiry_target = surface.timestamp + expiry_interval
    tau_target = time_to_expiry(expiry_target, surface.timestamp)
    tau_target <= 0.0 && return nothing

    expiries = unique(rec.expiry for rec in surface.records)
    isempty(expiries) && return nothing

    taus = [time_to_expiry(e, surface.timestamp) for e in expiries]
    idx = argmin(abs.(taus .- tau_target))
    expiry = expiries[idx]
    tau_closest = taus[idx]

    return (expiry, tau_target, tau_closest)
end

function _atm_iv_from_records(
    recs::Vector{OptionRecord},
    spot::Float64,
    tau::Float64,
    rate::Float64,
    div_yield::Float64;
    debug::Bool=false,
    timestamp::Union{Nothing,DateTime}=nothing
)::Union{Nothing,Float64}
    isempty(recs) && return nothing

    atm_rec = recs[argmin([abs(r.strike - spot) for r in recs])]
    if ismissing(atm_rec.mark_price)
        if debug && timestamp !== nothing
            println("No entry: ATM record has no mark_price (timestamp=$(timestamp))")
        end
        return nothing
    end

    F_atm = spot * exp((rate - div_yield) * tau)
    atm_iv = price_to_iv(atm_rec.mark_price, F_atm, atm_rec.strike, tau, atm_rec.option_type; r=rate)
    if isnan(atm_iv) || atm_iv <= 0.0
        if debug && timestamp !== nothing
            println("No entry: could not compute ATM IV (timestamp=$(timestamp))")
        end
        return nothing
    end

    return atm_iv
end

function _sigma_strangle_strikes(
    ctx,
    short_sigmas::Float64,
    rate::Float64,
    div_yield::Float64;
    debug::Bool=false
)::Union{Nothing,Tuple{Float64,Float64}}
    tau = ctx.tau
    tau <= 0.0 && return nothing

    atm_iv = _atm_iv_from_records(
        ctx.recs,
        ctx.surface.spot,
        tau,
        rate,
        div_yield;
        debug=debug,
        timestamp=ctx.surface.timestamp
    )
    atm_iv === nothing && return nothing

    sigma_move = atm_iv * sqrt(tau)
    if debug
        println(
            "  ATM IV=$(round(atm_iv * 100; digits=1))%, sigma=$(round(sigma_move * 100; digits=2))%, " *
            "shorts +/-$(round(short_sigmas * sigma_move * 100; digits=2))%"
        )
    end
    short_put_pct = 1.0 - short_sigmas * sigma_move
    short_call_pct = 1.0 + short_sigmas * sigma_move

    put_strikes = ctx.put_strikes
    call_strikes = ctx.call_strikes
    if isempty(put_strikes) || isempty(call_strikes)
        return nothing
    end

    target_put = ctx.surface.spot * short_put_pct
    target_call = ctx.surface.spot * short_call_pct

    short_put_K = _pick_otm_strike(put_strikes, ctx.surface.spot, target_put; side=:put)
    short_call_K = _pick_otm_strike(call_strikes, ctx.surface.spot, target_call; side=:call)

    return (short_put_K, short_call_K)
end

function _sigma_condor_strikes(
    ctx,
    short_sigmas::Float64,
    long_sigmas::Float64,
    rate::Float64,
    div_yield::Float64;
    debug::Bool=false
)::Union{Nothing,Tuple{Float64,Float64,Float64,Float64}}
    tau = ctx.tau
    tau <= 0.0 && return nothing

    atm_iv = _atm_iv_from_records(
        ctx.recs,
        ctx.surface.spot,
        tau,
        rate,
        div_yield;
        debug=debug,
        timestamp=ctx.surface.timestamp
    )
    atm_iv === nothing && return nothing

    sigma_move = atm_iv * sqrt(tau)
    if debug
        println(
            "  ATM IV=$(round(atm_iv * 100; digits=1))%, sigma=$(round(sigma_move * 100; digits=2))%, " *
            "shorts +/-$(round(short_sigmas * sigma_move * 100; digits=2))%, " *
            "longs +/-$(round(long_sigmas * sigma_move * 100; digits=2))%"
        )
    end
    short_put_pct = 1.0 - short_sigmas * sigma_move
    short_call_pct = 1.0 + short_sigmas * sigma_move
    long_put_pct = 1.0 - long_sigmas * sigma_move
    long_call_pct = 1.0 + long_sigmas * sigma_move

    put_strikes = ctx.put_strikes
    call_strikes = ctx.call_strikes
    if isempty(put_strikes) || isempty(call_strikes)
        return nothing
    end

    return _condor_strikes(
        put_strikes,
        call_strikes,
        ctx.surface.spot,
        short_put_pct,
        short_call_pct,
        long_put_pct,
        long_call_pct
    )
end

function _delta_from_record(
    rec::OptionRecord,
    F::Float64,
    tau::Float64,
    rate::Float64
)::Union{Missing,Float64}
    vol = if !ismissing(rec.mark_iv) && rec.mark_iv > 0
        rec.mark_iv / 100.0
    elseif !ismissing(rec.mark_price)
        iv = price_to_iv(rec.mark_price, F, rec.strike, tau, rec.option_type; r=rate)
        isnan(iv) || iv <= 0.0 ? missing : iv
    else
        missing
    end
    vol === missing && return missing
    return black76_delta(F, rec.strike, tau, vol, rec.option_type; r=rate)
end

function _best_delta_strike(
    recs::Vector{OptionRecord},
    target::Float64,
    spot::Float64,
    side::Symbol,
    F::Float64,
    tau::Float64,
    rate::Float64;
    debug::Bool=false
)::Union{Nothing,Float64}
    candidates = if side == :put
        filter(r -> r.strike < spot, recs)
    else
        filter(r -> r.strike > spot, recs)
    end
    isempty(candidates) && (candidates = recs)

    best_strike = nothing
    best_delta = missing
    best_diff = Inf
    for rec in candidates
        delta = _delta_from_record(rec, F, tau, rate)
        delta === missing && continue
        diff = abs(delta - target)
        if diff < best_diff
            best_diff = diff
            best_strike = rec.strike
            best_delta = delta
        end
    end

    if debug && best_strike !== nothing
        println("  target delta=$(round(target, digits=2)) -> strike=$(round(best_strike, digits=2)) delta=$(round(best_delta, digits=3))")
    end

    return best_strike
end

function _delta_strangle_strikes(
    ctx,
    target_delta::Float64;
    rate::Float64=0.0,
    div_yield::Float64=0.0,
    debug::Bool=false
)::Union{Nothing,Tuple{Float64,Float64}}
    tau = ctx.tau
    tau <= 0.0 && return nothing

    put_recs = filter(r -> r.option_type == Put, ctx.recs)
    call_recs = filter(r -> r.option_type == Call, ctx.recs)
    isempty(put_recs) && return nothing
    isempty(call_recs) && return nothing

    F = ctx.surface.spot * exp((rate - div_yield) * tau)
    target = abs(target_delta)

    short_put = _best_delta_strike(
        put_recs,
        -target,
        ctx.surface.spot,
        :put,
        F,
        tau,
        rate;
        debug=debug
    )
    short_call = _best_delta_strike(
        call_recs,
        target,
        ctx.surface.spot,
        :call,
        F,
        tau,
        rate;
        debug=debug
    )

    if short_put === nothing || short_call === nothing
        return nothing
    end

    return (short_put, short_call)
end

"""
    _delta_strangle_strikes_asymmetric(ctx, put_delta_abs, call_delta_abs; rate, div_yield, debug)
        -> Union{Nothing, Tuple{Float64, Float64}}

Select strangle strikes with potentially different deltas for put and call legs.
This allows the ML model to predict asymmetric strangles.

# Arguments
- `ctx`: Strike selection context
- `put_delta_abs::Float64`: Absolute delta for short put (e.g., 0.15 for 15-delta)
- `call_delta_abs::Float64`: Absolute delta for short call (e.g., 0.20 for 20-delta)
- `rate::Float64`: Risk-free rate
- `div_yield::Float64`: Dividend yield
- `debug::Bool`: Print debug info

# Returns
- Tuple of (short_put_K, short_call_K) or nothing
"""
function _delta_strangle_strikes_asymmetric(
    ctx,
    put_delta_abs::Float64,
    call_delta_abs::Float64;
    rate::Float64=0.0,
    div_yield::Float64=0.0,
    debug::Bool=false
)::Union{Nothing,Tuple{Float64,Float64}}
    tau = ctx.tau
    tau <= 0.0 && return nothing

    put_recs = filter(r -> r.option_type == Put, ctx.recs)
    call_recs = filter(r -> r.option_type == Call, ctx.recs)
    isempty(put_recs) && return nothing
    isempty(call_recs) && return nothing

    F = ctx.surface.spot * exp((rate - div_yield) * tau)

    short_put = _best_delta_strike(
        put_recs,
        -abs(put_delta_abs),
        ctx.surface.spot,
        :put,
        F,
        tau,
        rate;
        debug=debug
    )
    short_call = _best_delta_strike(
        call_recs,
        abs(call_delta_abs),
        ctx.surface.spot,
        :call,
        F,
        tau,
        rate;
        debug=debug
    )

    if short_put === nothing || short_call === nothing
        return nothing
    end

    return (short_put, short_call)
end

function _filter_by_delta(
    recs::Vector{OptionRecord},
    target_abs::Float64,
    F::Float64,
    tau::Float64,
    rate::Float64
)::Vector{OptionRecord}
    filtered = OptionRecord[]
    for rec in recs
        delta = _delta_from_record(rec, F, tau, rate)
        delta === missing && continue
        abs(delta) <= target_abs && push!(filtered, rec)
    end
    return isempty(filtered) ? recs : filtered
end

function _find_rec_by_strike(
    recs::Vector{OptionRecord},
    strike::Float64
)::Union{Nothing,OptionRecord}
    for rec in recs
        rec.strike == strike && return rec
    end
    return nothing
end

"""
    _delta_condor_strikes(ctx, short_put_delta_abs, short_call_delta_abs,
                          long_put_delta_abs, long_call_delta_abs; ...) -> strikes

"16 delta" means |Delta| ~= 0.16. In Black-Scholes, call delta = N(d1) and
put delta = N(d1) - 1, so |put delta| = N(-d1). N(-1) ~= 0.1587, which implies
|d1| ~= 1. Delta is a risk-neutral, price-implied coordinate (not probability).
Selecting by delta is equivalent to selecting by |d1|. Greeks (vega/gamma)
scale with phi(d1), so delta selection places strikes where Greek density is desired.
"""
function _delta_condor_strikes(
    ctx,
    short_put_delta_abs::Float64,
    short_call_delta_abs::Float64,
    long_put_delta_abs::Float64,
    long_call_delta_abs::Float64;
    rate::Float64=0.0,
    div_yield::Float64=0.0,
    min_delta_gap::Float64=0.08,
    debug::Bool=false
)::Union{Nothing,Tuple{Float64,Float64,Float64,Float64}}
    tau = ctx.tau
    tau <= 0.0 && return nothing

    long_put_target = min(long_put_delta_abs, short_put_delta_abs - min_delta_gap)
    long_call_target = min(long_call_delta_abs, short_call_delta_abs - min_delta_gap)
    (long_put_target <= 0.0 || long_call_target <= 0.0) && return nothing

    F = ctx.surface.spot * exp((rate - div_yield) * tau)
    put_recs = filter(r -> r.option_type == Put, ctx.recs)
    call_recs = filter(r -> r.option_type == Call, ctx.recs)
    isempty(put_recs) && return nothing
    isempty(call_recs) && return nothing

    short_put = _best_delta_strike(
        put_recs,
        -short_put_delta_abs,
        ctx.surface.spot,
        :put,
        F,
        tau,
        rate;
        debug=debug
    )
    short_call = _best_delta_strike(
        call_recs,
        short_call_delta_abs,
        ctx.surface.spot,
        :call,
        F,
        tau,
        rate;
        debug=debug
    )
    (short_put === nothing || short_call === nothing) && return nothing

    long_put_recs = filter(r -> r.strike < short_put, put_recs)
    long_call_recs = filter(r -> r.strike > short_call, call_recs)
    isempty(long_put_recs) && return nothing
    isempty(long_call_recs) && return nothing

    long_put_recs = _filter_by_delta(long_put_recs, long_put_target, F, tau, rate)
    long_call_recs = _filter_by_delta(long_call_recs, long_call_target, F, tau, rate)

    long_put = _best_delta_strike(
        long_put_recs,
        -long_put_target,
        ctx.surface.spot,
        :put,
        F,
        tau,
        rate;
        debug=debug
    )
    long_call = _best_delta_strike(
        long_call_recs,
        long_call_target,
        ctx.surface.spot,
        :call,
        F,
        tau,
        rate;
        debug=debug
    )
    (long_put === nothing || long_call === nothing) && return nothing

    if !(long_put < short_put < ctx.surface.spot < short_call < long_call)
        return nothing
    end

    sp_rec = _find_rec_by_strike(put_recs, short_put)
    sc_rec = _find_rec_by_strike(call_recs, short_call)
    lp_rec = _find_rec_by_strike(put_recs, long_put)
    lc_rec = _find_rec_by_strike(call_recs, long_call)
    sp_rec === nothing && return nothing
    sc_rec === nothing && return nothing
    lp_rec === nothing && return nothing
    lc_rec === nothing && return nothing

    sp_delta = _delta_from_record(sp_rec, F, tau, rate)
    sc_delta = _delta_from_record(sc_rec, F, tau, rate)
    lp_delta = _delta_from_record(lp_rec, F, tau, rate)
    lc_delta = _delta_from_record(lc_rec, F, tau, rate)
    if sp_delta === missing || sc_delta === missing || lp_delta === missing || lc_delta === missing
        return nothing
    end

    if abs(lp_delta) > abs(sp_delta) - min_delta_gap
        long_put = _best_delta_strike(
            long_put_recs,
            0.0,
            ctx.surface.spot,
            :put,
            F,
            tau,
            rate;
            debug=debug
        )
        long_put === nothing && return nothing
    end
    if abs(lc_delta) > abs(sc_delta) - min_delta_gap
        long_call = _best_delta_strike(
            long_call_recs,
            0.0,
            ctx.surface.spot,
            :call,
            F,
            tau,
            rate;
            debug=debug
        )
        long_call === nothing && return nothing
    end

    if !(long_put < short_put < ctx.surface.spot < short_call < long_call)
        return nothing
    end

    return (short_put, short_call, long_put, long_call)
end

"""
    _condor_wings_by_objective(ctx, short_put_K, short_call_K; objective, target_max_loss, max_loss_min, max_loss_max, min_credit, min_delta_gap, prefer_symmetric, roi_eps, rate, div_yield, debug)
        -> Union{Nothing, Tuple{Float64, Float64}}

Select wing strikes (long put and long call) for a short iron condor using a configurable objective.

Supported objectives:
- `:target_max_loss`: match a target max loss (legacy behavior)
- `:roi`: maximize entry credit / max loss
- `:pnl`: maximize entry credit

Risk controls:
- `max_loss_min`, `max_loss_max` in dollars
- `min_credit` in dollars

If `prefer_symmetric=true`, an asymmetry penalty is applied to the objective score.
"""
function _condor_wings_by_objective(
    ctx,
    short_put_K::Float64,
    short_call_K::Float64;
    objective::Symbol=:target_max_loss,
    target_max_loss::Union{Nothing,Float64}=nothing,
    max_loss_min::Float64=0.0,
    max_loss_max::Float64=Inf,
    min_credit::Float64=0.0,
    min_delta_gap::Float64=0.08,
    prefer_symmetric::Bool=true,
    roi_eps::Float64=1e-6,
    rate::Float64=0.0,
    div_yield::Float64=0.0,
    debug::Bool=false
)::Union{Nothing,Tuple{Float64,Float64}}
    tau = ctx.tau
    tau <= 0.0 && return nothing

    objective in (:target_max_loss, :roi, :pnl) || error("Unknown condor wing objective: $objective")
    if objective == :target_max_loss && target_max_loss === nothing
        return nothing
    end
    max_loss_max < max_loss_min && return nothing

    spot = ctx.surface.spot
    target_max_loss_norm = target_max_loss === nothing ? nothing : target_max_loss / spot
    max_loss_min_norm = max(0.0, max_loss_min / spot)
    max_loss_max_norm = isfinite(max_loss_max) ? max_loss_max / spot : Inf
    min_credit_norm = min_credit / spot

    F = spot * exp((rate - div_yield) * tau)
    put_recs = filter(r -> r.option_type == Put, ctx.recs)
    call_recs = filter(r -> r.option_type == Call, ctx.recs)

    # Get short strike records for pricing (we sell at bid)
    short_put_rec = _find_rec_by_strike(put_recs, short_put_K)
    short_call_rec = _find_rec_by_strike(call_recs, short_call_K)
    (short_put_rec === nothing || short_call_rec === nothing) && return nothing

    short_put_price = if !ismissing(short_put_rec.bid_price)
        short_put_rec.bid_price
    elseif !ismissing(short_put_rec.mark_price)
        short_put_rec.mark_price
    else
        return nothing
    end

    short_call_price = if !ismissing(short_call_rec.bid_price)
        short_call_rec.bid_price
    elseif !ismissing(short_call_rec.mark_price)
        short_call_rec.mark_price
    else
        return nothing
    end

    long_put_candidates = filter(r -> r.strike < short_put_K &&
                                      (!ismissing(r.ask_price) || !ismissing(r.mark_price)),
                                 put_recs)
    long_call_candidates = filter(r -> r.strike > short_call_K &&
                                       (!ismissing(r.ask_price) || !ismissing(r.mark_price)),
                                  call_recs)
    isempty(long_put_candidates) && return nothing
    isempty(long_call_candidates) && return nothing

    # Enforce a minimum delta gap between short and long wings.
    if min_delta_gap > 0.0
        short_put_delta = _delta_from_record(short_put_rec, F, tau, rate)
        short_call_delta = _delta_from_record(short_call_rec, F, tau, rate)

        if short_put_delta !== missing && short_call_delta !== missing
            max_long_put_delta = abs(short_put_delta) - min_delta_gap
            max_long_call_delta = abs(short_call_delta) - min_delta_gap

            long_put_candidates = [r for r in long_put_candidates
                                   if (_delta_from_record(r, F, tau, rate) |>
                                       (d -> d === missing || abs(d) <= max_long_put_delta))]
            long_call_candidates = [r for r in long_call_candidates
                                    if (_delta_from_record(r, F, tau, rate) |>
                                        (d -> d === missing || abs(d) <= max_long_call_delta))]
        end
    end
    isempty(long_put_candidates) && return nothing
    isempty(long_call_candidates) && return nothing

    best_combo = nothing
    best_score = -Inf

    for long_put_rec in long_put_candidates
        for long_call_rec in long_call_candidates
            long_put_price = if !ismissing(long_put_rec.ask_price)
                long_put_rec.ask_price
            elseif !ismissing(long_put_rec.mark_price)
                long_put_rec.mark_price
            else
                continue
            end

            long_call_price = if !ismissing(long_call_rec.ask_price)
                long_call_rec.ask_price
            elseif !ismissing(long_call_rec.mark_price)
                long_call_rec.mark_price
            else
                continue
            end

            put_spread_width = (short_put_K - long_put_rec.strike) / spot
            call_spread_width = (long_call_rec.strike - short_call_K) / spot
            net_credit = (short_put_price + short_call_price) - (long_put_price + long_call_price)
            put_max_loss = put_spread_width - net_credit
            call_max_loss = call_spread_width - net_credit
            max_loss = max(put_max_loss, call_max_loss)

            # Reject invalid or out-of-band risk/credit profiles.
            max_loss > 0.0 || continue
            max_loss < max_loss_min_norm && continue
            max_loss > max_loss_max_norm && continue
            net_credit < min_credit_norm && continue

            width_diff = abs(put_spread_width - call_spread_width)
            penalty = prefer_symmetric ? width_diff * 0.5 : 0.0

            score = if objective == :target_max_loss
                -(abs(max_loss - target_max_loss_norm) + penalty)
            elseif objective == :roi
                (net_credit / max(max_loss, roi_eps)) - penalty
            else
                net_credit - penalty
            end

            if score > best_score
                best_score = score
                best_combo = (
                    long_put_rec.strike,
                    long_call_rec.strike,
                    max_loss,
                    net_credit,
                    put_spread_width,
                    call_spread_width
                )
            end
        end
    end

    best_combo === nothing && return nothing
    long_put_K, long_call_K, achieved_max_loss, net_credit, put_width, call_width = best_combo

    if !(long_put_K < short_put_K < spot < short_call_K < long_call_K)
        return nothing
    end

    if debug
        if objective == :target_max_loss
            println("  Objective=target_max_loss, target=$(round(target_max_loss_norm * 100, digits=2)), achieved=$(round(achieved_max_loss * 100, digits=2))")
        elseif objective == :roi
            roi = net_credit / max(achieved_max_loss, roi_eps)
            println("  Objective=roi, achieved ROI=$(round(roi, digits=4)), max_loss=$(round(achieved_max_loss * 100, digits=2))")
        else
            println("  Objective=pnl, credit=$(round(net_credit * 100, digits=2)), max_loss=$(round(achieved_max_loss * 100, digits=2))")
        end
        println("  Put width=$(round(put_width * 100, digits=2))%, call width=$(round(call_width * 100, digits=2))%")
        println("  Wings: put=$(round(long_put_K, digits=2)), call=$(round(long_call_K, digits=2))")
    end

    return (long_put_K, long_call_K)
end

"""
    _max_loss_condor_wings(ctx, short_put_K, short_call_K, target_max_loss; rate, div_yield, min_delta_gap, prefer_symmetric, debug)
        -> Union{Nothing, Tuple{Float64, Float64}}

Select wing strikes (long put and long call) to achieve a target maximum loss for an iron condor,
given fixed short strikes.

For a short iron condor, the maximum loss is calculated as:
    max_loss = spread_width - net_credit

where:
- spread_width = max(short_put - long_put, long_call - short_call)
- net_credit = (short_put_price + short_call_price) - (long_put_price + long_call_price)

# Arguments
- `ctx`: Strike selection context with surface, tau, recs, etc.
- `short_put_K::Float64`: Short put strike (inner)
- `short_call_K::Float64`: Short call strike (inner)
- `target_max_loss::Float64`: Desired maximum loss per contract (in dollars, e.g., 5.0 for \$500 max loss)
- `rate::Float64`: Risk-free rate
- `div_yield::Float64`: Dividend yield
- `min_delta_gap::Float64`: Minimum delta gap between short and long strikes
- `prefer_symmetric::Bool`: If true, prefer equal width spreads on both sides
- `debug::Bool`: Print debug information

# Returns
- Tuple of (long_put_K, long_call_K) or nothing if no valid wings found
"""
function _max_loss_condor_wings(
    ctx,
    short_put_K::Float64,
    short_call_K::Float64,
    target_max_loss::Float64;
    rate::Float64=0.0,
    div_yield::Float64=0.0,
    min_delta_gap::Float64=0.08,
    prefer_symmetric::Bool=true,
    debug::Bool=false
)::Union{Nothing,Tuple{Float64,Float64}}
    return _condor_wings_by_objective(
        ctx,
        short_put_K,
        short_call_K;
        objective=:target_max_loss,
        target_max_loss=target_max_loss,
        max_loss_min=0.0,
        max_loss_max=Inf,
        min_credit=-Inf,
        min_delta_gap=min_delta_gap,
        prefer_symmetric=prefer_symmetric,
        roi_eps=1e-6,
        rate=rate,
        div_yield=div_yield,
        debug=debug
    )
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
