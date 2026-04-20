# Stateful condor selector: picks short legs by delta, long-wing width selected
# by rolling-window training Sharpe.
#
# Mirrors the logic in `scripts/condor_rolling_wing.jl` so the result is
# identical when run via `IronCondorStrategy + backtest_strategy`.
#
# Design: split immutable `Config` from mutable `State`. The selector wrapper
# holds both; only `state` changes during a backtest.

"""Configuration for `RollingWingCondorSelector` — immutable."""
struct RollingWingCondorConfig
    put_delta::Float64
    call_delta::Float64
    wing_widths::Vector{Float64}
    train_days::Int
    test_days::Int
    step_days::Int            # advance windows by this; usually = test_days (no overlap)
    rate::Float64
    div_yield::Float64
    max_tau_days::Float64     # skip entries whose closest expiry is further than this
end

"""Mutable rolling state — grows with each accepted entry."""
mutable struct RollingWingCondorState
    history::Vector{NamedTuple}
    current_wing::Float64
    first_entry_date::Date    # date of first accepted entry (anchors window grid)
    last_window_idx::Int      # last computed window index; -1 means uninitialized
end

RollingWingCondorState(init_wing::Float64) =
    RollingWingCondorState(NamedTuple[], init_wing, Date(1900, 1, 1), -1)

"""
    RollingWingCondorSelector(config) | RollingWingCondorSelector(; kwargs...)

Selector for [`IronCondorStrategy`](@ref). Picks short legs by delta and selects
the long-wing width that maximised trailing-Sharpe over the previous training
window. Calendar-aligned recompute every `step_days`.
"""
struct RollingWingCondorSelector
    config::RollingWingCondorConfig
    state::RollingWingCondorState
end

function RollingWingCondorSelector(;
    put_delta::Float64,
    call_delta::Float64,
    wing_widths::Vector{Float64},
    train_days::Int,
    test_days::Int,
    step_days::Int = test_days,
    rate::Float64,
    div_yield::Float64,
    max_tau_days::Float64 = Inf,
)
    cfg = RollingWingCondorConfig(
        put_delta, call_delta, wing_widths, train_days, test_days, step_days,
        rate, div_yield, max_tau_days,
    )
    return RollingWingCondorSelector(cfg, RollingWingCondorState(maximum(wing_widths)))
end

# Annualized Sharpe assuming the input is a sample of per-trade PnLs.
_sharpe_of(v::AbstractVector{<:Real}) =
    isempty(v) ? 0.0 : (std(v) > 0 ? mean(v) / std(v) * sqrt(252) : 0.0)

# Recompute the active wing from past `train_days` of settled entries.
# `window_start` is the date that anchors the test window (boundary).
function _refresh_wing!(sel::RollingWingCondorSelector,
                       ctx::StrikeSelectionContext,
                       window_start::Date)
    cfg, state = sel.config, sel.state
    today = window_start
    cutoff = today - Day(cfg.train_days)

    n_w = length(cfg.wing_widths)
    pnls_per_wing = [Float64[] for _ in 1:n_w]

    for entry in state.history
        edate = Date(entry.entry_ts)
        edate < cutoff && continue
        edate >= today && continue
        spot_settle = get_settlement_spot(ctx.history, entry.expiry)
        ismissing(spot_settle) && continue
        for wi in 1:n_w
            credit_frac = entry.entry_credits_per_wing[wi]
            isnan(credit_frac) && continue
            sp_K = entry.sp_K
            sc_K = entry.sc_K
            lp_K = entry.wing_long_strikes[wi]
            lc_K = entry.wing_call_long_strikes[wi]
            credit_usd = credit_frac * entry.spot_at_entry
            intrinsic_usd =
                -max(sp_K - Float64(spot_settle), 0.0) +
                -max(Float64(spot_settle) - sc_K, 0.0) +
                +max(lp_K - Float64(spot_settle), 0.0) +
                +max(Float64(spot_settle) - lc_K, 0.0)
            push!(pnls_per_wing[wi], credit_usd + intrinsic_usd)
        end
    end

    sharpes = [_sharpe_of(pnls_per_wing[wi]) for wi in 1:n_w]
    if all(isempty, pnls_per_wing) || all(==(0.0), sharpes)
        return  # not enough data — keep current wing
    end
    best_wi = argmax(sharpes)
    state.current_wing = cfg.wing_widths[best_wi]
    nothing
end

function (sel::RollingWingCondorSelector)(ctx::StrikeSelectionContext)
    cfg, state = sel.config, sel.state

    today = Date(ctx.surface.timestamp)
    recs = _ctx_recs(ctx)
    tau = _ctx_tau(ctx)
    tau <= 0.0 && return nothing
    tau * 365.25 > cfg.max_tau_days && return nothing

    spot = ctx.surface.spot
    F = spot * exp((cfg.rate - cfg.div_yield) * tau)
    put_recs  = filter(r -> r.option_type == Put,  recs)
    call_recs = filter(r -> r.option_type == Call, recs)
    isempty(put_recs) && return nothing
    isempty(call_recs) && return nothing

    sp_K = _best_delta_strike(put_recs,  -cfg.put_delta,  spot, :put,  F, tau, cfg.rate)
    sc_K = _best_delta_strike(call_recs,  cfg.call_delta, spot, :call, F, tau, cfg.rate)
    (sp_K === nothing || sc_K === nothing) && return nothing

    otm_put_recs  = filter(r -> r.strike <  sp_K, put_recs)
    otm_call_recs = filter(r -> r.strike >  sc_K, call_recs)
    (isempty(otm_put_recs) || isempty(otm_call_recs)) && return nothing

    # All preconditions passed — anchor window grid to this date if not already.
    if state.first_entry_date == Date(1900, 1, 1)
        state.first_entry_date = today
    end

    # Calendar-aligned window grid: refresh wing on window boundaries.
    days_after_train_start = (today - state.first_entry_date).value - cfg.train_days
    if days_after_train_start >= 0
        wi = days_after_train_start ÷ cfg.step_days
        if wi != state.last_window_idx
            window_start = state.first_entry_date + Day(cfg.train_days + wi * cfg.step_days)
            _refresh_wing!(sel, ctx, window_start)
            state.last_window_idx = wi
        end
    end

    # Snap wings for ALL widths (so we can compute shadow PnL later).
    n_w = length(cfg.wing_widths)
    wing_long_strikes      = Vector{Float64}(undef, n_w)
    wing_call_long_strikes = Vector{Float64}(undef, n_w)
    entry_credits_per_wing = Vector{Float64}(undef, n_w)

    sp_rec = nothing
    for r in put_recs
        if r.strike == sp_K
            sp_rec = r; break
        end
    end
    sc_rec = nothing
    for r in call_recs
        if r.strike == sc_K
            sc_rec = r; break
        end
    end
    (sp_rec === nothing || sc_rec === nothing) && return nothing
    short_put_bid  = _extract_price(sp_rec, :bid)
    short_call_bid = _extract_price(sc_rec, :bid)
    (short_put_bid === nothing || short_call_bid === nothing) && return nothing

    for wi in 1:n_w
        ww = cfg.wing_widths[wi]
        target_lp = sp_K - ww
        target_lc = sc_K + ww
        lp_rec = otm_put_recs[argmin(abs.([r.strike - target_lp for r in otm_put_recs]))]
        lc_rec = otm_call_recs[argmin(abs.([r.strike - target_lc for r in otm_call_recs]))]
        wing_long_strikes[wi]      = lp_rec.strike
        wing_call_long_strikes[wi] = lc_rec.strike
        long_put_ask  = _extract_price(lp_rec, :ask)
        long_call_ask = _extract_price(lc_rec, :ask)
        entry_credits_per_wing[wi] = if long_put_ask === nothing || long_call_ask === nothing
            NaN
        else
            short_put_bid + short_call_bid - long_put_ask - long_call_ask
        end
    end

    # Active wing for THIS entry
    active_wi = findfirst(==(state.current_wing), cfg.wing_widths)
    active_wi === nothing && (active_wi = length(cfg.wing_widths))

    lp_K = wing_long_strikes[active_wi]
    lc_K = wing_call_long_strikes[active_wi]

    push!(state.history, (
        entry_ts = ctx.surface.timestamp,
        expiry   = ctx.expiry,
        sp_K     = sp_K,
        sc_K     = sc_K,
        wing_long_strikes      = wing_long_strikes,
        wing_call_long_strikes = wing_call_long_strikes,
        entry_credits_per_wing = entry_credits_per_wing,
        spot_at_entry          = spot,
        chosen_wing            = state.current_wing,
    ))

    return (sp_K, sc_K, lp_K, lc_K)
end
