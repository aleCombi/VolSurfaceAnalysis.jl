# Strike selection primitives and selector factories for iron condors.
#
# The selector ctx is a minimal named tuple: (surface, expiry, history).
# Internal helpers derive tau and records from these.

# =============================================================================
# Primitives
# =============================================================================

function _nearest_strike(strikes::Vector{Float64}, target::Float64)::Float64
    distances = abs.(strikes .- target)
    return strikes[argmin(distances)]
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

"""
    _ctx_tau(ctx) -> Float64

Derive time-to-expiry from a selector context.
"""
_ctx_tau(ctx) = time_to_expiry(ctx.expiry, ctx.surface.timestamp)

"""
    _ctx_recs(ctx) -> Vector{OptionRecord}

Derive option records for the target expiry from a selector context.
"""
_ctx_recs(ctx) = filter(r -> r.expiry == ctx.expiry, ctx.surface.records)

# =============================================================================
# Record helpers
# =============================================================================

function _find_rec_by_strike(
    recs::Vector{OptionRecord},
    strike::Float64
)::Union{Nothing,OptionRecord}
    for rec in recs
        rec.strike == strike && return rec
    end
    return nothing
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

# =============================================================================
# Asymmetric delta strangle strikes (used by condor selectors)
# =============================================================================

"""
    _delta_strangle_strikes_asymmetric(ctx, put_delta_abs, call_delta_abs; rate, div_yield, debug)
        -> Union{Nothing, Tuple{Float64, Float64}}

Select strangle strikes with potentially different deltas for put and call legs.
"""
function _delta_strangle_strikes_asymmetric(
    ctx,
    put_delta_abs::Float64,
    call_delta_abs::Float64;
    rate::Float64=0.0,
    div_yield::Float64=0.0,
    debug::Bool=false
)::Union{Nothing,Tuple{Float64,Float64}}
    tau = _ctx_tau(ctx)
    tau <= 0.0 && return nothing

    recs = _ctx_recs(ctx)
    put_recs = filter(r -> r.option_type == Put, recs)
    call_recs = filter(r -> r.option_type == Call, recs)
    isempty(put_recs) && return nothing
    isempty(call_recs) && return nothing

    F = ctx.surface.spot * exp((rate - div_yield) * tau)

    short_put = _best_delta_strike(
        put_recs, -abs(put_delta_abs), ctx.surface.spot, :put, F, tau, rate; debug=debug
    )
    short_call = _best_delta_strike(
        call_recs, abs(call_delta_abs), ctx.surface.spot, :call, F, tau, rate; debug=debug
    )

    if short_put === nothing || short_call === nothing
        return nothing
    end

    return (short_put, short_call)
end

# =============================================================================
# Sigma-based condor strikes
# =============================================================================

function _sigma_condor_strikes(
    ctx,
    short_sigmas::Float64,
    long_sigmas::Float64,
    rate::Float64,
    div_yield::Float64;
    debug::Bool=false
)::Union{Nothing,Tuple{Float64,Float64,Float64,Float64}}
    tau = _ctx_tau(ctx)
    tau <= 0.0 && return nothing

    recs = _ctx_recs(ctx)
    atm_iv = _atm_iv_from_records(
        recs, ctx.surface.spot, tau, rate, div_yield;
        debug=debug, timestamp=ctx.surface.timestamp
    )
    atm_iv === nothing && return nothing

    sigma_move = atm_iv * sqrt(tau)
    if debug
        println("  ATM IV=$(round(atm_iv * 100; digits=1))%, sigma=$(round(sigma_move * 100; digits=2))%, " *
              "shorts +/-$(round(short_sigmas * sigma_move * 100; digits=2))%, " *
              "longs +/-$(round(long_sigmas * sigma_move * 100; digits=2))%")
    end

    short_put_pct = 1.0 - short_sigmas * sigma_move
    short_call_pct = 1.0 + short_sigmas * sigma_move
    long_put_pct = 1.0 - long_sigmas * sigma_move
    long_call_pct = 1.0 + long_sigmas * sigma_move

    put_strikes = sort(unique(r.strike for r in recs if r.option_type == Put))
    call_strikes = sort(unique(r.strike for r in recs if r.option_type == Call))
    if isempty(put_strikes) || isempty(call_strikes)
        return nothing
    end

    spot = ctx.surface.spot
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

# =============================================================================
# Objective-driven wing selection
# =============================================================================

"""
    _condor_wings_by_objective(ctx, short_put_K, short_call_K; objective, ...)
        -> Union{Nothing, Tuple{Float64, Float64}}

Select wing strikes (long put and long call) for an iron condor using a configurable objective.

Supported objectives:
- `:target_max_loss`: match a target max loss
- `:roi`: maximize entry credit / max loss
- `:pnl`: maximize entry credit
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
    tau = _ctx_tau(ctx)
    tau <= 0.0 && return nothing

    objective in (:target_max_loss, :roi, :pnl) || error("Unknown condor wing objective: $objective")
    if objective == :target_max_loss && target_max_loss === nothing
        return nothing
    end
    max_loss_max < max_loss_min && return nothing

    recs = _ctx_recs(ctx)
    spot = ctx.surface.spot
    target_max_loss_norm = target_max_loss === nothing ? nothing : target_max_loss / spot
    max_loss_min_norm = max(0.0, max_loss_min / spot)
    max_loss_max_norm = isfinite(max_loss_max) ? max_loss_max / spot : Inf
    min_credit_norm = min_credit / spot

    F = spot * exp((rate - div_yield) * tau)
    put_recs = filter(r -> r.option_type == Put, recs)
    call_recs = filter(r -> r.option_type == Call, recs)

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

# =============================================================================
# Delta-based condor strikes
# =============================================================================

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
    tau = _ctx_tau(ctx)
    tau <= 0.0 && return nothing

    long_put_target = min(long_put_delta_abs, short_put_delta_abs - min_delta_gap)
    long_call_target = min(long_call_delta_abs, short_call_delta_abs - min_delta_gap)
    (long_put_target <= 0.0 || long_call_target <= 0.0) && return nothing

    recs = _ctx_recs(ctx)
    F = ctx.surface.spot * exp((rate - div_yield) * tau)
    put_recs = filter(r -> r.option_type == Put, recs)
    call_recs = filter(r -> r.option_type == Call, recs)
    isempty(put_recs) && return nothing
    isempty(call_recs) && return nothing

    short_put = _best_delta_strike(
        put_recs, -short_put_delta_abs, ctx.surface.spot, :put, F, tau, rate; debug=debug
    )
    short_call = _best_delta_strike(
        call_recs, short_call_delta_abs, ctx.surface.spot, :call, F, tau, rate; debug=debug
    )
    (short_put === nothing || short_call === nothing) && return nothing

    long_put_recs = filter(r -> r.strike < short_put, put_recs)
    long_call_recs = filter(r -> r.strike > short_call, call_recs)
    isempty(long_put_recs) && return nothing
    isempty(long_call_recs) && return nothing

    long_put = _best_delta_strike(
        long_put_recs, -long_put_target, ctx.surface.spot, :put, F, tau, rate; debug=debug
    )
    long_call = _best_delta_strike(
        long_call_recs, long_call_target, ctx.surface.spot, :call, F, tau, rate; debug=debug
    )
    (long_put === nothing || long_call === nothing) && return nothing

    if !(long_put < short_put < ctx.surface.spot < short_call < long_call)
        return nothing
    end

    return (short_put, short_call, long_put, long_call)
end

# =============================================================================
# Selector factories
# =============================================================================

"""
    sigma_selector(short_sigmas, long_sigmas; rate=0.0, div_yield=0.0)

Return a callable `ctx -> (sp_K, sc_K, lp_K, lc_K) | nothing` that selects
iron condor strikes at sigma multiples of ATM implied vol.
"""
function sigma_selector(short_sigmas::Float64, long_sigmas::Float64;
                        rate::Float64=0.0, div_yield::Float64=0.0)
    return function(ctx)
        _sigma_condor_strikes(ctx, short_sigmas, long_sigmas, rate, div_yield)
    end
end

"""
    delta_selector(short_delta; rate=0.0, div_yield=0.0, wing_objective=:roi, kwargs...)

Return a callable `ctx -> (sp_K, sc_K, lp_K, lc_K) | nothing` that selects
short strikes by delta and optimizes wings using `_condor_wings_by_objective`.
"""
function delta_selector(short_delta::Float64;
                        rate::Float64=0.0,
                        div_yield::Float64=0.0,
                        wing_objective::Symbol=:roi,
                        wing_kwargs...)
    return function(ctx)
        shorts = _delta_strangle_strikes_asymmetric(
            ctx, short_delta, short_delta; rate=rate, div_yield=div_yield
        )
        shorts === nothing && return nothing
        sp_K, sc_K = shorts
        wings = _condor_wings_by_objective(
            ctx, sp_K, sc_K;
            objective=wing_objective,
            rate=rate,
            div_yield=div_yield,
            wing_kwargs...
        )
        wings === nothing && return nothing
        lp_K, lc_K = wings
        return (sp_K, sc_K, lp_K, lc_K)
    end
end

"""
    delta_condor_selector(sp_delta, sc_delta, lp_delta, lc_delta; rate=0.0, div_yield=0.0, min_delta_gap=0.08)

Return a callable `ctx -> (sp_K, sc_K, lp_K, lc_K) | nothing` that selects
all 4 condor strikes by delta.
"""
function delta_condor_selector(sp_delta::Float64, sc_delta::Float64,
                               lp_delta::Float64, lc_delta::Float64;
                               rate::Float64=0.0,
                               div_yield::Float64=0.0,
                               min_delta_gap::Float64=0.08)
    return function(ctx)
        _delta_condor_strikes(
            ctx, sp_delta, sc_delta, lp_delta, lc_delta;
            rate=rate, div_yield=div_yield, min_delta_gap=min_delta_gap
        )
    end
end

# =============================================================================
# Spread helpers
# =============================================================================

"""
    _relative_spread(rec::OptionRecord) -> Union{Nothing, Float64}

Compute (ask - bid) / mid for a record. Returns `nothing` if bid/ask missing or mid <= 0.
"""
function _relative_spread(rec::OptionRecord)::Union{Nothing,Float64}
    (ismissing(rec.bid_price) || ismissing(rec.ask_price)) && return nothing
    bid = Float64(rec.bid_price)
    ask = Float64(rec.ask_price)
    mid = (bid + ask) / 2.0
    mid <= 0.0 && return nothing
    return (ask - bid) / mid
end

# =============================================================================
# Constrained selector
# =============================================================================

"""
    constrained_delta_selector(put_delta, call_delta;
        rate=0.0, div_yield=0.0, max_loss, max_spread_rel=Inf,
        min_delta_gap=0.08, prefer_symmetric=true)

Return a callable `ctx -> (sp_K, sc_K, lp_K, lc_K) | nothing` that:
1. Selects short strikes at the given deltas
2. Checks relative bid-ask spread on the short strikes against `max_spread_rel`
3. Finds wings where max loss ≤ `max_loss` (USD), maximizing ROI within that constraint

Emits `@warn` and returns `nothing` when constraints cannot be met.
"""
function constrained_delta_selector(put_delta::Float64, call_delta::Float64;
                                    rate::Float64=0.0,
                                    div_yield::Float64=0.0,
                                    max_loss::Float64,
                                    max_spread_rel::Float64=Inf,
                                    min_delta_gap::Float64=0.08,
                                    prefer_symmetric::Bool=true)
    return function(ctx)
        tau = _ctx_tau(ctx)
        tau <= 0.0 && return nothing

        recs = _ctx_recs(ctx)
        spot = ctx.surface.spot
        ts = ctx.surface.timestamp

        # Find short strikes by delta
        shorts = _delta_strangle_strikes_asymmetric(
            ctx, put_delta, call_delta; rate=rate, div_yield=div_yield
        )
        if shorts === nothing
            @warn "No valid short strikes at deltas ($put_delta, $call_delta)" timestamp=ts
            return nothing
        end
        sp_K, sc_K = shorts

        # Check bid-ask spread on short strikes
        if isfinite(max_spread_rel)
            put_recs = filter(r -> r.option_type == Put, recs)
            call_recs = filter(r -> r.option_type == Call, recs)
            sp_rec = _find_rec_by_strike(put_recs, sp_K)
            sc_rec = _find_rec_by_strike(call_recs, sc_K)

            for (label, rec) in [("short put", sp_rec), ("short call", sc_rec)]
                rec === nothing && continue
                spread = _relative_spread(rec)
                if spread !== nothing && spread > max_spread_rel
                    @warn "Spread too wide on $label" timestamp=ts strike=rec.strike spread=round(spread, digits=3) max=max_spread_rel
                    return nothing
                end
            end
        end

        # Find wings with max loss constraint, maximizing ROI
        wings = _condor_wings_by_objective(
            ctx, sp_K, sc_K;
            objective=:roi,
            max_loss_max=max_loss,
            min_delta_gap=min_delta_gap,
            prefer_symmetric=prefer_symmetric,
            rate=rate,
            div_yield=div_yield
        )
        if wings === nothing
            @warn "No wings satisfying max_loss ≤ $max_loss" timestamp=ts short_put=sp_K short_call=sc_K
            return nothing
        end
        lp_K, lc_K = wings

        return (sp_K, sc_K, lp_K, lc_K)
    end
end
