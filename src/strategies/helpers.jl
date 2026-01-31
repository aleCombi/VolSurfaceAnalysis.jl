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
