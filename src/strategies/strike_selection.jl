# Strike selection primitives and selector factories for iron condors.

# =============================================================================
# Price extraction helper
# =============================================================================

"""Extract bid or ask price from record, falling back to mark_price. Returns `nothing` if neither available."""
function _extract_price(rec, side::Symbol)
    primary = side == :bid ? rec.bid_price : rec.ask_price
    !ismissing(primary) && return primary
    !ismissing(rec.mark_price) && return rec.mark_price
    return nothing
end

# =============================================================================
# StrikeSelectionContext
# =============================================================================

"""
    StrikeSelectionContext

Formalized context passed to strike selectors. Contains the volatility surface,
target expiry, and a look-ahead-safe historical view.

Selectors are callables: `f(ctx::StrikeSelectionContext) -> (sp_K, sc_K, lp_K, lc_K) | nothing`
"""
struct StrikeSelectionContext
    surface::VolatilitySurface
    expiry::DateTime
    history::BacktestDataSource
end

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

    best_strike !== nothing && @debug "  target delta=$(round(target, digits=2)) -> strike=$(round(best_strike, digits=2)) delta=$(round(best_delta, digits=3))"

    return best_strike
end

function _atm_iv_from_records(
    recs::Vector{OptionRecord},
    spot::Float64,
    tau::Float64,
    rate::Float64,
    div_yield::Float64;
    timestamp::Union{Nothing,DateTime}=nothing
)::Union{Nothing,Float64}
    isempty(recs) && return nothing

    atm_rec = recs[argmin([abs(r.strike - spot) for r in recs])]
    if ismissing(atm_rec.mark_price)
        timestamp !== nothing && @debug "No entry: ATM record has no mark_price (timestamp=$(timestamp))"
        return nothing
    end

    F_atm = spot * exp((rate - div_yield) * tau)
    atm_iv = price_to_iv(atm_rec.mark_price, F_atm, atm_rec.strike, tau, atm_rec.option_type; r=rate)
    if isnan(atm_iv) || atm_iv <= 0.0
        timestamp !== nothing && @debug "No entry: could not compute ATM IV (timestamp=$(timestamp))"
        return nothing
    end

    return atm_iv
end

# =============================================================================
# Public delta-strike helpers (for scripts that need per-entry primitives)
# =============================================================================

"""
    delta_context(ctx; rate=0.0, div_yield=0.0)
        -> (; put_recs, call_recs, spot, F, tau, rate) | nothing

Resolve the per-entry quantities needed for delta-based strike lookup:
records filtered by option type, spot, forward `F = spot * exp((rate-div_yield)*tau)`,
and tau. Returns `nothing` if `tau <= 0` or either put or call records are empty.

Pair with `delta_strike(dctx, target_delta, opt_type)` to pick strikes.
"""
function delta_context(
    ctx;
    rate::Float64=0.0,
    div_yield::Float64=0.0,
)
    tau = _ctx_tau(ctx)
    tau <= 0.0 && return nothing
    recs = _ctx_recs(ctx)
    put_recs  = filter(r -> r.option_type == Put,  recs)
    call_recs = filter(r -> r.option_type == Call, recs)
    (isempty(put_recs) || isempty(call_recs)) && return nothing
    spot = ctx.surface.spot
    F = spot * exp((rate - div_yield) * tau)
    return (put_recs=put_recs, call_recs=call_recs, spot=spot, F=F, tau=tau, rate=rate)
end

"""
    delta_strike(dctx, target_delta, opt_type::OptionType) -> Float64 | nothing

Return the OTM strike whose Black-76 delta is closest to `target_delta` for the
given option type (`Put` or `Call`). `target_delta` follows the standard sign
convention (negative for puts, positive for calls). `dctx` is the NamedTuple
returned by `delta_context`. Returns `nothing` when no viable record has a
computable delta.
"""
function delta_strike(
    dctx,
    target_delta::Float64,
    opt_type::OptionType,
)::Union{Nothing,Float64}
    recs = opt_type == Put ? dctx.put_recs : dctx.call_recs
    side = opt_type == Put ? :put : :call
    return _best_delta_strike(recs, target_delta, dctx.spot, side, dctx.F, dctx.tau, dctx.rate)
end

"""
    nearest_otm_strike(dctx, reference_K, width, opt_type::OptionType)
        -> Float64 | nothing

Wing-pick helper: return the strike strictly OTM relative to `reference_K`
whose distance from `reference_K ± width` is minimal. For `Put`, the wing lies
below `reference_K` at target `reference_K - width`; for `Call`, above at
`reference_K + width`. `width` is a positive distance in strike-price units.
Returns `nothing` when no strictly-OTM record exists on the requested side.
"""
function nearest_otm_strike(
    dctx,
    reference_K::Float64,
    width::Float64,
    opt_type::OptionType,
)::Union{Nothing,Float64}
    if opt_type == Put
        otm = filter(r -> r.strike < reference_K, dctx.put_recs)
        target = reference_K - width
    else
        otm = filter(r -> r.strike > reference_K, dctx.call_recs)
        target = reference_K + width
    end
    isempty(otm) && return nothing
    return otm[argmin(abs.([r.strike - target for r in otm]))].strike
end

"""
    extract_price(rec, side::Symbol) -> Float64 | nothing

Return `rec.bid_price` (`side=:bid`) or `rec.ask_price` (`side=:ask`), falling
back to `rec.mark_price` when the primary is missing. Returns `nothing` if
neither is available.
"""
extract_price(rec, side::Symbol) = _extract_price(rec, side)

# =============================================================================
# Asymmetric delta strangle strikes (used by condor selectors)
# =============================================================================

"""
    _delta_strangle_strikes_asymmetric(ctx, put_delta_abs, call_delta_abs; rate, div_yield)
        -> Union{Nothing, Tuple{Float64, Float64}}

Select strangle strikes with potentially different deltas for put and call legs.
"""
function _delta_strangle_strikes_asymmetric(
    ctx,
    put_delta_abs::Float64,
    call_delta_abs::Float64;
    rate::Float64=0.0,
    div_yield::Float64=0.0
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
        put_recs, -abs(put_delta_abs), ctx.surface.spot, :put, F, tau, rate
    )
    short_call = _best_delta_strike(
        call_recs, abs(call_delta_abs), ctx.surface.spot, :call, F, tau, rate
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
    div_yield::Float64
)::Union{Nothing,Tuple{Float64,Float64,Float64,Float64}}
    tau = _ctx_tau(ctx)
    tau <= 0.0 && return nothing

    recs = _ctx_recs(ctx)
    atm_iv = _atm_iv_from_records(
        recs, ctx.surface.spot, tau, rate, div_yield;
        timestamp=ctx.surface.timestamp
    )
    atm_iv === nothing && return nothing

    sigma_move = atm_iv * sqrt(tau)
    @debug "  ATM IV=$(round(atm_iv * 100; digits=1))%, sigma=$(round(sigma_move * 100; digits=2))%, " *
          "shorts +/-$(round(short_sigmas * sigma_move * 100; digits=2))%, " *
          "longs +/-$(round(long_sigmas * sigma_move * 100; digits=2))%"

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
    _select_condor_wings(ctx, short_put_K, short_call_K; max_loss, max_spread_rel, rate, div_yield)
        -> Union{Nothing, Tuple{Float64, Float64}}

Select wing strikes (long put and long call) for an iron condor.
Picks the widest liquid wings that keep max loss within the cap.

Candidates are sorted widest-first; the first valid combo is returned.
Wings with bid-ask spread exceeding `max_spread_rel` are skipped.
"""
function _select_condor_wings(
    ctx,
    short_put_K::Float64,
    short_call_K::Float64;
    max_loss::Float64=Inf,
    max_spread_rel::Float64=Inf,
    rate::Float64=0.0,
    div_yield::Float64=0.0,
)::Union{Nothing,Tuple{Float64,Float64}}
    recs = _ctx_recs(ctx)
    spot = ctx.surface.spot
    max_loss_norm = isfinite(max_loss) ? max_loss / spot : Inf

    put_recs = filter(r -> r.option_type == Put, recs)
    call_recs = filter(r -> r.option_type == Call, recs)

    short_put_price = let rec = _find_rec_by_strike(put_recs, short_put_K)
        rec === nothing ? nothing : _extract_price(rec, :bid)
    end
    short_call_price = let rec = _find_rec_by_strike(call_recs, short_call_K)
        rec === nothing ? nothing : _extract_price(rec, :bid)
    end
    (short_put_price === nothing || short_call_price === nothing) && return nothing

    # Candidates: have a price, pass spread filter. Sorted widest-first.
    long_put_candidates = sort(
        filter(r -> r.strike < short_put_K &&
                    _extract_price(r, :ask) !== nothing &&
                    _passes_spread(r, max_spread_rel),
               put_recs);
        by=r -> r.strike  # ascending = widest first (lowest strike)
    )
    long_call_candidates = sort(
        filter(r -> r.strike > short_call_K &&
                    _extract_price(r, :ask) !== nothing &&
                    _passes_spread(r, max_spread_rel),
               call_recs);
        by=r -> -r.strike  # descending = widest first (highest strike)
    )
    isempty(long_put_candidates) && return nothing
    isempty(long_call_candidates) && return nothing

    for lp_rec in long_put_candidates
        for lc_rec in long_call_candidates
            put_width = (short_put_K - lp_rec.strike) / spot
            call_width = (lc_rec.strike - short_call_K) / spot
            credit = (short_put_price + short_call_price) -
                     (_extract_price(lp_rec, :ask) + _extract_price(lc_rec, :ask))
            ml = max(put_width, call_width) - credit
            ml > 0.0 || continue
            ml <= max_loss_norm && return (lp_rec.strike, lc_rec.strike)
        end
    end

    return nothing
end

"""Check that a record's bid-ask spread is within threshold."""
_passes_spread(rec, max_spread_rel) =
    !isfinite(max_spread_rel) || let s = _relative_spread(rec); s === nothing || s <= max_spread_rel end

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
    min_delta_gap::Float64=0.08
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
        put_recs, -short_put_delta_abs, ctx.surface.spot, :put, F, tau, rate
    )
    short_call = _best_delta_strike(
        call_recs, short_call_delta_abs, ctx.surface.spot, :call, F, tau, rate
    )
    (short_put === nothing || short_call === nothing) && return nothing

    long_put_recs = filter(r -> r.strike < short_put, put_recs)
    long_call_recs = filter(r -> r.strike > short_call, call_recs)
    isempty(long_put_recs) && return nothing
    isempty(long_call_recs) && return nothing

    long_put = _best_delta_strike(
        long_put_recs, -long_put_target, ctx.surface.spot, :put, F, tau, rate
    )
    long_call = _best_delta_strike(
        long_call_recs, long_call_target, ctx.surface.spot, :call, F, tau, rate
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
    delta_selector(short_delta; rate=0.0, div_yield=0.0, kwargs...)

Return a callable `ctx -> (sp_K, sc_K, lp_K, lc_K) | nothing` that selects
short strikes by delta and picks the widest valid wings.
"""
function delta_selector(short_delta::Float64;
                        rate::Float64=0.0,
                        div_yield::Float64=0.0,
                        wing_kwargs...)
    return function(ctx)
        shorts = _delta_strangle_strikes_asymmetric(
            ctx, short_delta, short_delta; rate=rate, div_yield=div_yield
        )
        shorts === nothing && return nothing
        sp_K, sc_K = shorts
        wings = _select_condor_wings(
            ctx, sp_K, sc_K;
            rate=rate, div_yield=div_yield, wing_kwargs...
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

"""Check that short leg bid-ask spreads are within `max_spread_rel`. Returns `false` if any leg exceeds the threshold."""
function _check_short_spreads(ctx::StrikeSelectionContext, sp_K, sc_K, max_spread_rel)
    isfinite(max_spread_rel) || return true
    recs = _ctx_recs(ctx)
    for (opt_type, strike) in ((Put, sp_K), (Call, sc_K))
        rec = _find_rec_by_strike(filter(r -> r.option_type == opt_type, recs), strike)
        rec === nothing && continue
        spread = _relative_spread(rec)
        spread !== nothing && spread > max_spread_rel && return false
    end
    return true
end

# =============================================================================
# Constrained selector
# =============================================================================

"""
    constrained_delta_selector(put_delta, call_delta;
        rate=0.0, div_yield=0.0, max_loss, max_spread_rel=Inf)

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
                                    max_spread_rel::Float64=Inf)
    return function(ctx)
        ts = ctx.surface.timestamp

        shorts = _delta_strangle_strikes_asymmetric(
            ctx, put_delta, call_delta; rate=rate, div_yield=div_yield
        )
        if shorts === nothing
            @warn "No valid short strikes at deltas ($put_delta, $call_delta)" timestamp=ts
            return nothing
        end
        sp_K, sc_K = shorts

        _check_short_spreads(ctx, sp_K, sc_K, max_spread_rel) || begin
            @warn "Spread too wide on short legs" timestamp=ts
            return nothing
        end

        wings = _select_condor_wings(
            ctx, sp_K, sc_K;
            max_loss=max_loss, max_spread_rel=max_spread_rel,
            rate=rate, div_yield=div_yield
        )
        if wings === nothing
            @warn "No wings satisfying max_loss ≤ $max_loss" timestamp=ts short_put=sp_K short_call=sc_K
            return nothing
        end
        lp_K, lc_K = wings

        return (sp_K, sc_K, lp_K, lc_K)
    end
end
