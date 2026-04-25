# Stateful strangle selector: picks short put + short call by absolute delta,
# selecting the (put_delta, call_delta) combo that maximised a rolling-window
# score over its trailing training history.
#
# Mirrors `RollingWingCondorSelector` for naked short strangles. Records
# per-entry shadow credits across all candidate combos and per-window
# selection decisions (full score grid + chosen) so post-hoc diagnostics
# (rank-IC, IS/OOS Sharpe per combo, baseline rank, etc.) can be computed
# entirely from `selector.state` after a backtest completes.

"""Configuration for `RollingDeltaStrangleSelector` — immutable."""
struct RollingDeltaStrangleConfig{F}
    put_deltas::Vector{Float64}
    call_deltas::Vector{Float64}
    train_days::Int
    test_days::Int
    step_days::Int            # advance windows by this; usually = test_days (no overlap)
    rate::Float64
    div_yield::Float64
    max_tau_days::Float64
    score::F                  # AbstractVector{<:Real} -> Float64; default annualized Sharpe
end

"""Mutable rolling state — grows with each accepted entry / window boundary."""
mutable struct RollingDeltaStrangleState
    history::Vector{NamedTuple}            # per-entry shadow data
    fold_choices::Vector{NamedTuple}       # per-window-boundary decisions
    current_combo::Tuple{Float64,Float64}
    first_entry_date::Date
    last_window_idx::Int
end

RollingDeltaStrangleState(init_combo::Tuple{Float64,Float64}) =
    RollingDeltaStrangleState(NamedTuple[], NamedTuple[], init_combo,
                              Date(1900, 1, 1), -1)

"""
    RollingDeltaStrangleSelector(; put_deltas, call_deltas,
                                 train_days, test_days, step_days=test_days,
                                 rate, div_yield,
                                 max_tau_days=Inf, score=_ann_sharpe)

Selector for [`ShortStrangleStrategy`](@ref). For each entry, picks
`(put_delta, call_delta)` from `put_deltas × call_deltas` that maximised
`score(pnl_history)` over the trailing `train_days` of accepted entries.
Recomputes the active combo at calendar-aligned window boundaries every
`step_days`. Returns the strikes for the active combo (`(sp_K, sc_K)`).

Post-backtest analysis: read `selector.state.history` (per-entry shadow
credits across all combos) and `selector.state.fold_choices` (per-window
chosen combo + full score grid). The downstream runner functions
(`run_strangle_rolling`, `report_strangle_rolling`) consume these.

The default `score` is annualized Sharpe; pass any `AbstractVector{<:Real}
-> Float64` for CVaR-regularized scoring etc.
"""
struct RollingDeltaStrangleSelector{F}
    config::RollingDeltaStrangleConfig{F}
    state::RollingDeltaStrangleState
end

# Annualized Sharpe assuming the input is a sample of per-trade PnLs.
_ann_sharpe(v::AbstractVector{<:Real}) =
    isempty(v) ? 0.0 : (std(v) > 0 ? mean(v) / std(v) * sqrt(252) : 0.0)

function RollingDeltaStrangleSelector(;
    put_deltas::Vector{Float64},
    call_deltas::Vector{Float64},
    train_days::Int,
    test_days::Int,
    step_days::Int = test_days,
    rate::Float64,
    div_yield::Float64,
    max_tau_days::Float64 = Inf,
    score = _ann_sharpe,
)
    isempty(put_deltas)  && error("put_deltas must be non-empty")
    isempty(call_deltas) && error("call_deltas must be non-empty")
    cfg = RollingDeltaStrangleConfig(
        put_deltas, call_deltas, train_days, test_days, step_days,
        rate, div_yield, max_tau_days, score,
    )
    init = (put_deltas[1], call_deltas[1])
    return RollingDeltaStrangleSelector(cfg, RollingDeltaStrangleState(init))
end

# Compute per-combo PnL vectors over the trailing training window.
# Returns a (n_put × n_call) matrix of Float64 vectors (settlement PnL in USD).
function _trailing_window_pnls(sel::RollingDeltaStrangleSelector,
                              ctx::StrikeSelectionContext,
                              window_start::Date)
    cfg, state = sel.config, sel.state
    cutoff = window_start - Day(cfg.train_days)
    n_p, n_c = length(cfg.put_deltas), length(cfg.call_deltas)
    pnls = [Float64[] for _ in 1:n_p, _ in 1:n_c]

    for entry in state.history
        edate = Date(entry.entry_ts)
        edate < cutoff && continue
        edate >= window_start && continue
        spot_settle = get_settlement_spot(ctx.history, entry.expiry)
        ismissing(spot_settle) && continue
        @inbounds for i in 1:n_p, j in 1:n_c
            credit_frac = entry.credit_frac[i, j]
            isnan(credit_frac) && continue
            sp_K = entry.put_strikes[i]
            sc_K = entry.call_strikes[j]
            credit_usd    = credit_frac * entry.spot_at_entry
            intrinsic_usd = max(sp_K - Float64(spot_settle), 0.0) +
                            max(Float64(spot_settle) - sc_K, 0.0)
            push!(pnls[i, j], credit_usd - intrinsic_usd)
        end
    end
    return pnls
end

# Score the per-combo PnL vectors and pick argmax.
# Returns (scores::Matrix, best_idx::CartesianIndex) or (nothing, nothing) if
# nothing is scorable.
function _score_grid(sel::RollingDeltaStrangleSelector, pnls::Matrix{Vector{Float64}})
    cfg = sel.config
    n_p, n_c = size(pnls)
    scores = fill(-Inf, n_p, n_c)
    @inbounds for i in 1:n_p, j in 1:n_c
        isempty(pnls[i, j]) && continue
        scores[i, j] = cfg.score(pnls[i, j])
    end
    all(==(-Inf), scores) && return (nothing, nothing)
    return (scores, argmax(scores))
end

# Refresh the active combo from the past `train_days` of settled entries.
# Records the decision in `state.fold_choices`.
function _refresh_combo!(sel::RollingDeltaStrangleSelector,
                        ctx::StrikeSelectionContext,
                        window_start::Date)
    cfg, state = sel.config, sel.state
    pnls = _trailing_window_pnls(sel, ctx, window_start)
    scores, best = _score_grid(sel, pnls)
    if scores === nothing
        return  # nothing scorable — keep current combo, no fold record
    end
    state.current_combo = (cfg.put_deltas[best[1]], cfg.call_deltas[best[2]])
    push!(state.fold_choices, (
        idx          = length(state.fold_choices) + 1,
        window_start = window_start,
        train_start  = window_start - Day(cfg.train_days),
        train_end    = window_start - Day(1),
        scores       = scores,
        chosen       = state.current_combo,
        chosen_idx   = (best[1], best[2]),
        n_settled    = sum(length, pnls),
    ))
    return
end

function (sel::RollingDeltaStrangleSelector)(ctx::StrikeSelectionContext)
    cfg, state = sel.config, sel.state

    today = Date(ctx.surface.timestamp)
    dctx = delta_context(ctx; rate=cfg.rate, div_yield=cfg.div_yield)
    dctx === nothing && return nothing
    dctx.tau * 365.25 > cfg.max_tau_days && return nothing

    n_p, n_c = length(cfg.put_deltas), length(cfg.call_deltas)
    put_strikes  = fill(NaN, n_p)
    call_strikes = fill(NaN, n_c)
    put_bids     = fill(NaN, n_p)
    call_bids    = fill(NaN, n_c)

    # Resolve per-pd put strike/bid (NaN where unavailable, don't abort entry)
    for i in 1:n_p
        K = delta_strike(dctx, -cfg.put_deltas[i], Put)
        K === nothing && continue
        rec = find_record_at_strike(dctx.put_recs, K)
        rec === nothing && continue
        bid = extract_price(rec, :bid)
        bid === nothing && continue
        put_strikes[i] = K
        put_bids[i]    = bid
    end
    for j in 1:n_c
        K = delta_strike(dctx, cfg.call_deltas[j], Call)
        K === nothing && continue
        rec = find_record_at_strike(dctx.call_recs, K)
        rec === nothing && continue
        bid = extract_price(rec, :bid)
        bid === nothing && continue
        call_strikes[j] = K
        call_bids[j]    = bid
    end

    # Skip entry only if no put or no call resolved on any candidate
    !any(isfinite, put_bids)  && return nothing
    !any(isfinite, call_bids) && return nothing

    # Anchor window grid to first accepted entry
    if state.first_entry_date == Date(1900, 1, 1)
        state.first_entry_date = today
    end

    # Calendar-aligned window grid: refresh combo on window boundaries
    days_after_train_start = (today - state.first_entry_date).value - cfg.train_days
    if days_after_train_start >= 0
        wi = days_after_train_start ÷ cfg.step_days
        if wi != state.last_window_idx
            window_start = state.first_entry_date + Day(cfg.train_days + wi * cfg.step_days)
            _refresh_combo!(sel, ctx, window_start)
            state.last_window_idx = wi
        end
    end

    # Shadow credits per (pd, cd) — fraction of spot, NaN if either leg failed.
    credit_frac = Matrix{Float64}(undef, n_p, n_c)
    @inbounds for i in 1:n_p, j in 1:n_c
        credit_frac[i, j] = isfinite(put_bids[i]) && isfinite(call_bids[j]) ?
            put_bids[i] + call_bids[j] : NaN
    end

    push!(state.history, (
        entry_ts      = ctx.surface.timestamp,
        expiry        = ctx.expiry,
        spot_at_entry = dctx.spot,
        put_strikes   = put_strikes,
        call_strikes  = call_strikes,
        credit_frac   = credit_frac,
        chosen_combo  = state.current_combo,
    ))

    # Active (pd, cd) for THIS entry — fall back to first valid combo if the
    # active one's strikes failed.
    pd_idx = findfirst(==(state.current_combo[1]), cfg.put_deltas)
    cd_idx = findfirst(==(state.current_combo[2]), cfg.call_deltas)
    pd_idx === nothing && (pd_idx = findfirst(isfinite, put_strikes))
    cd_idx === nothing && (cd_idx = findfirst(isfinite, call_strikes))
    if !isfinite(put_strikes[pd_idx])
        pd_idx = findfirst(isfinite, put_strikes)
    end
    if !isfinite(call_strikes[cd_idx])
        cd_idx = findfirst(isfinite, call_strikes)
    end
    return (put_strikes[pd_idx], call_strikes[cd_idx])
end
