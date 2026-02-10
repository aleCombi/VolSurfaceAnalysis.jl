# ML Training Pipeline for Strike Selection
# Generates training data and trains the neural network

using Flux.Optimise: Adam

const DEFAULT_RATE = 0.045
const DEFAULT_DIV_YIELD = 0.013

"""
    TrainingDataset

Container for training data.

# Fields
- `features::Matrix{Float32}`: Input features (n_features × n_samples)
- `labels::Matrix{Float32}`: Target deltas scaled to [0,1] (2 × n_samples)
- `raw_deltas::Matrix{Float32}`: Original delta values (2 × n_samples)
- `pnls::Vector{Float32}`: Best P&L for each sample
- `size_labels::Vector{Float32}`: Position size targets in [0, 1]
- `timestamps::Vector{DateTime}`: Entry timestamps
"""
struct TrainingDataset
    features::Matrix{Float32}
    labels::Matrix{Float32}
    raw_deltas::Matrix{Float32}
    pnls::Vector{Float32}
    size_labels::Vector{Float32}
    timestamps::Vector{DateTime}
end

"""
    CondorScoringDataset

Dataset for candidate-scoring models (state-action utility regression).

# Fields
- `features::Matrix{Float32}`: Combined state+candidate feature matrix
- `utilities::Vector{Float32}`: Training utility target (e.g., realized ROI)
- `pnls::Vector{Float32}`: Realized PnL per candidate
- `max_losses::Vector{Float32}`: Max loss per candidate
- `timestamps::Vector{DateTime}`: Entry timestamps (repeated across candidates)
"""
struct CondorScoringDataset
    features::Matrix{Float32}
    utilities::Vector{Float32}
    pnls::Vector{Float32}
    max_losses::Vector{Float32}
    timestamps::Vector{DateTime}
end

# Delta grid for searching optimal strikes
const DELTA_GRID = collect(0.05:0.025:0.35)

# Long-wing delta grid for 4-delta condor search
const LONG_DELTA_GRID = collect(0.01:0.02:0.15)

"""Resolve expiry, filter records by type, and compute forward rate for simulation functions."""
function _prepare_simulation_env(
    surface::VolatilitySurface,
    expiry_interval::Period,
    rate::Float64,
    div_yield::Float64;
    compute_strikes::Bool=false
)
    expiry_result = _select_expiry(expiry_interval, surface)
    expiry_result === nothing && return nothing

    expiry, _, tau = expiry_result
    tau <= 0.0 && return nothing

    recs = filter(r -> r.expiry == expiry, surface.records)
    isempty(recs) && return nothing

    put_recs = filter(r -> r.option_type == Put, recs)
    call_recs = filter(r -> r.option_type == Call, recs)
    isempty(put_recs) && return nothing
    isempty(call_recs) && return nothing

    F = surface.spot * exp((rate - div_yield) * tau)

    ctx = nothing
    if compute_strikes
        put_strikes = sort(unique(r.strike for r in put_recs))
        call_strikes = sort(unique(r.strike for r in call_recs))
        (isempty(put_strikes) || isempty(call_strikes)) && return nothing
        ctx = (surface=surface, expiry=expiry, tau=tau, recs=recs,
               put_strikes=put_strikes, call_strikes=call_strikes)
    end

    return (expiry=expiry, tau=tau, recs=recs, put_recs=put_recs,
            call_recs=call_recs, F=F, ctx=ctx)
end

"""
    simulate_strangle_pnl(surface, settlement_spot, put_delta, call_delta; rate, div_yield, expiry_interval) -> Union{Float64, Nothing}

Simulate P&L for a short strangle with given deltas.

# Arguments
- `surface::VolatilitySurface`: Entry surface
- `settlement_spot::Float64`: Settlement spot price
- `put_delta::Float64`: Absolute delta for short put
- `call_delta::Float64`: Absolute delta for short call
- `rate::Float64`: Risk-free rate
- `div_yield::Float64`: Dividend yield
- `expiry_interval::Period`: Time to expiry

# Returns
- P&L in dollars, or nothing if strikes cannot be found
"""
function simulate_strangle_pnl(
    surface::VolatilitySurface,
    settlement_spot::Float64,
    put_delta::Float64,
    call_delta::Float64;
    rate::Float64=DEFAULT_RATE,
    div_yield::Float64=DEFAULT_DIV_YIELD,
    expiry_interval::Period=Day(1)
)::Union{Float64,Nothing}
    env = _prepare_simulation_env(surface, expiry_interval, rate, div_yield)
    env === nothing && return nothing
    (; put_recs, call_recs, F) = env

    # Find strikes by delta
    short_put_K = _best_delta_strike(put_recs, -put_delta, surface.spot, :put, F, env.tau, rate)
    short_call_K = _best_delta_strike(call_recs, call_delta, surface.spot, :call, F, env.tau, rate)

    (short_put_K === nothing || short_call_K === nothing) && return nothing

    # Get entry prices (bid for shorts)
    put_rec = _find_rec_by_strike(put_recs, short_put_K)
    call_rec = _find_rec_by_strike(call_recs, short_call_K)

    (put_rec === nothing || call_rec === nothing) && return nothing
    (ismissing(put_rec.bid_price) || ismissing(call_rec.bid_price)) && return nothing

    # Entry cost (negative for shorts = premium received)
    entry_put = put_rec.bid_price * (-1) * surface.spot
    entry_call = call_rec.bid_price * (-1) * surface.spot
    entry_cost = entry_put + entry_call

    # Settlement payoff
    put_payoff = max(short_put_K - settlement_spot, 0.0) * (-1)
    call_payoff = max(settlement_spot - short_call_K, 0.0) * (-1)
    total_payoff = put_payoff + call_payoff

    pnl = total_payoff - entry_cost
    return pnl
end

"""Grid search over delta pairs, calling `simulate_fn(put_delta, call_delta)` for each."""
function _find_optimal_deltas_grid(
    delta_grid::Vector{Float64},
    simulate_fn::Function
)::Union{Tuple{Float64,Float64,Float64},Nothing}
    best_pnl = -Inf
    best_put_delta = 0.15
    best_call_delta = 0.15
    n_valid = 0

    for put_delta in delta_grid
        for call_delta in delta_grid
            pnl = simulate_fn(put_delta, call_delta)
            pnl === nothing && continue
            n_valid += 1

            if pnl > best_pnl
                best_pnl = pnl
                best_put_delta = put_delta
                best_call_delta = call_delta
            end
        end
    end

    n_valid < 1 && return nothing
    return (best_put_delta, best_call_delta, best_pnl)
end

"""
    find_optimal_deltas(surface, settlement_spot; rate, div_yield, expiry_interval, delta_grid)
        -> Union{Tuple{Float64, Float64, Float64}, Nothing}

Find optimal delta values via grid search.

# Returns
- Tuple of (best_put_delta, best_call_delta, best_pnl) or nothing
"""
function find_optimal_deltas(
    surface::VolatilitySurface,
    settlement_spot::Float64;
    rate::Float64=DEFAULT_RATE,
    div_yield::Float64=DEFAULT_DIV_YIELD,
    expiry_interval::Period=Day(1),
    delta_grid::Vector{Float64}=DELTA_GRID
)::Union{Tuple{Float64,Float64,Float64},Nothing}
    sim_fn = (pd, cd) -> simulate_strangle_pnl(
        surface, settlement_spot, pd, cd;
        rate=rate, div_yield=div_yield, expiry_interval=expiry_interval
    )
    return _find_optimal_deltas_grid(delta_grid, sim_fn)
end

"""
    simulate_condor_pnl(surface, settlement_spot, short_put_delta, short_call_delta;
        wing_delta_abs, target_max_loss, min_delta_gap, prefer_symmetric, rate, div_yield, expiry_interval)
        -> Union{Float64, Nothing}

Simulate P&L for a short iron condor. Supports two wing modes:
1. Fixed-delta wings via `wing_delta_abs`
2. Target max-loss wings via `target_max_loss`

Wing mode behavior:
- If `wing_objective=:target_max_loss` and `target_max_loss` is provided, uses target-max-loss wings.
- If `wing_objective` is `:roi` or `:pnl`, uses objective-driven wings with configurable risk band.
- If `wing_objective=:target_max_loss` and `target_max_loss` is not provided, falls back to fixed-delta wings when `wing_delta_abs` is provided.

# Arguments
- `surface::VolatilitySurface`: Entry surface
- `settlement_spot::Float64`: Settlement spot price
- `short_put_delta::Float64`: Absolute delta for short put
- `short_call_delta::Float64`: Absolute delta for short call
- `wing_delta_abs::Union{Nothing,Float64}`: Absolute delta for long wings (fixed mode)
- `target_max_loss::Union{Nothing,Float64}`: Target maximum loss per contract in dollars (max-loss mode)
- `wing_objective::Symbol`: Wing objective (`:target_max_loss`, `:roi`, or `:pnl`)
- `max_loss_min::Float64`: Minimum allowed max loss in dollars (objective mode)
- `max_loss_max::Float64`: Maximum allowed max loss in dollars (objective mode)
- `min_credit::Float64`: Minimum net credit in dollars (objective mode)
- `min_delta_gap::Float64`: Minimum delta gap between short and long legs
- `prefer_symmetric::Bool`: Prefer symmetric wing widths in max-loss mode
- `rate::Float64`: Risk-free rate
- `div_yield::Float64`: Dividend yield
- `expiry_interval::Period`: Time to expiry

# Returns
- P&L in dollars, or nothing if strikes cannot be found
"""
function simulate_condor_pnl(
    surface::VolatilitySurface,
    settlement_spot::Float64,
    short_put_delta::Float64,
    short_call_delta::Float64;
    wing_delta_abs::Union{Nothing,Float64}=0.05,
    target_max_loss::Union{Nothing,Float64}=nothing,
    wing_objective::Symbol=:target_max_loss,
    max_loss_min::Float64=0.0,
    max_loss_max::Float64=Inf,
    min_credit::Float64=0.0,
    min_delta_gap::Float64=0.08,
    prefer_symmetric::Bool=true,
    rate::Float64=DEFAULT_RATE,
    div_yield::Float64=DEFAULT_DIV_YIELD,
    expiry_interval::Period=Day(1)
)::Union{Float64,Nothing}
    env = _prepare_simulation_env(surface, expiry_interval, rate, div_yield; compute_strikes=true)
    env === nothing && return nothing
    (; put_recs, call_recs, ctx) = env

    use_fixed_delta_wings = (
        wing_objective == :target_max_loss &&
        target_max_loss === nothing &&
        wing_delta_abs !== nothing
    )

    strikes = if use_fixed_delta_wings
        _delta_condor_strikes(
            ctx,
            short_put_delta,
            short_call_delta,
            wing_delta_abs,
            wing_delta_abs;
            rate=rate,
            div_yield=div_yield,
            min_delta_gap=min_delta_gap
        )
    else
        short_strikes = _delta_strangle_strikes_asymmetric(
            ctx,
            short_put_delta,
            short_call_delta;
            rate=rate,
            div_yield=div_yield
        )
        short_strikes === nothing && return nothing
        short_put_K, short_call_K = short_strikes

        wings = _condor_wings_by_objective(
            ctx,
            short_put_K,
            short_call_K;
            objective=wing_objective,
            target_max_loss=target_max_loss,
            max_loss_min=max_loss_min,
            max_loss_max=max_loss_max,
            min_credit=min_credit,
            rate=rate,
            div_yield=div_yield,
            min_delta_gap=min_delta_gap,
            prefer_symmetric=prefer_symmetric,
            debug=false
        )
        wings === nothing && return nothing
        long_put_K, long_call_K = wings
        (short_put_K, short_call_K, long_put_K, long_call_K)
    end
    strikes === nothing && return nothing

    short_put_K, short_call_K, long_put_K, long_call_K = strikes

    short_put_rec = _find_rec_by_strike(put_recs, short_put_K)
    short_call_rec = _find_rec_by_strike(call_recs, short_call_K)
    long_put_rec = _find_rec_by_strike(put_recs, long_put_K)
    long_call_rec = _find_rec_by_strike(call_recs, long_call_K)
    (short_put_rec === nothing || short_call_rec === nothing ||
        long_put_rec === nothing || long_call_rec === nothing) && return nothing

    if ismissing(short_put_rec.bid_price) || ismissing(short_call_rec.bid_price) ||
       ismissing(long_put_rec.ask_price) || ismissing(long_call_rec.ask_price)
        return nothing
    end

    # Entry cost (short inner legs = bid, long outer legs = ask)
    entry_short_put = short_put_rec.bid_price * (-1) * surface.spot
    entry_short_call = short_call_rec.bid_price * (-1) * surface.spot
    entry_long_put = long_put_rec.ask_price * (1) * surface.spot
    entry_long_call = long_call_rec.ask_price * (1) * surface.spot
    entry_cost = entry_short_put + entry_short_call + entry_long_put + entry_long_call

    # Settlement payoff
    put_payoff = max(short_put_K - settlement_spot, 0.0) * (-1) +
                 max(long_put_K - settlement_spot, 0.0) * (1)
    call_payoff = max(settlement_spot - short_call_K, 0.0) * (-1) +
                  max(settlement_spot - long_call_K, 0.0) * (1)
    total_payoff = put_payoff + call_payoff

    pnl = total_payoff - entry_cost
    return pnl
end

"""
    find_optimal_condor_deltas(surface, settlement_spot;
        wing_delta_abs, target_max_loss, wing_objective, max_loss_min, max_loss_max, min_credit,
        min_delta_gap, prefer_symmetric, rate, div_yield, expiry_interval, delta_grid)
        -> Union{Tuple{Float64, Float64, Float64}, Nothing}

Find optimal inner deltas for an iron condor via grid search under the configured wing policy.

# Returns
- Tuple of (best_put_delta, best_call_delta, best_pnl) or nothing
"""
function find_optimal_condor_deltas(
    surface::VolatilitySurface,
    settlement_spot::Float64;
    wing_delta_abs::Union{Nothing,Float64}=0.05,
    target_max_loss::Union{Nothing,Float64}=nothing,
    wing_objective::Symbol=:target_max_loss,
    max_loss_min::Float64=0.0,
    max_loss_max::Float64=Inf,
    min_credit::Float64=0.0,
    min_delta_gap::Float64=0.08,
    prefer_symmetric::Bool=true,
    rate::Float64=DEFAULT_RATE,
    div_yield::Float64=DEFAULT_DIV_YIELD,
    expiry_interval::Period=Day(1),
    delta_grid::Vector{Float64}=DELTA_GRID
)::Union{Tuple{Float64,Float64,Float64},Nothing}
    sim_fn = (pd, cd) -> simulate_condor_pnl(
        surface, settlement_spot, pd, cd;
        wing_delta_abs=wing_delta_abs,
        target_max_loss=target_max_loss,
        wing_objective=wing_objective,
        max_loss_min=max_loss_min,
        max_loss_max=max_loss_max,
        min_credit=min_credit,
        min_delta_gap=min_delta_gap,
        prefer_symmetric=prefer_symmetric,
        rate=rate,
        div_yield=div_yield,
        expiry_interval=expiry_interval
    )
    return _find_optimal_deltas_grid(delta_grid, sim_fn)
end

"""
    build_condor_ctx(surface; expiry_interval=Day(1))
        -> Union{Nothing, NamedTuple}

Build a strike-selection context for the expiry closest to `surface.timestamp + expiry_interval`.
"""
function build_condor_ctx(
    surface::VolatilitySurface;
    expiry_interval::Period=Day(1),
    rate::Float64=DEFAULT_RATE,
    div_yield::Float64=DEFAULT_DIV_YIELD
)::Union{Nothing,NamedTuple}
    env = _prepare_simulation_env(surface, expiry_interval, rate, div_yield; compute_strikes=true)
    env === nothing && return nothing
    return (ctx=env.ctx, expiry=env.expiry, tau=env.tau)
end

"""
    build_condor_ctx(surface, expiry, tau) -> Union{Nothing, NamedTuple}

Build a condor context from an explicit expiry and tau (no search needed).
Returns a NamedTuple with `(surface, expiry, tau, recs, put_strikes, call_strikes)`
or `nothing` if insufficient data.
"""
function build_condor_ctx(
    surface::VolatilitySurface,
    expiry::DateTime,
    tau::Float64
)::Union{Nothing,NamedTuple}
    recs = filter(r -> r.expiry == expiry, surface.records)
    isempty(recs) && return nothing
    put_strikes = sort(unique(r.strike for r in recs if r.option_type == Put))
    call_strikes = sort(unique(r.strike for r in recs if r.option_type == Call))
    (isempty(put_strikes) || isempty(call_strikes)) && return nothing
    return (surface=surface, expiry=expiry, tau=tau, recs=recs,
            put_strikes=put_strikes, call_strikes=call_strikes)
end

function _leg_relative_spread(rec::OptionRecord)::Float64
    if ismissing(rec.bid_price) || ismissing(rec.ask_price)
        return 0.0
    end
    mid = (rec.bid_price + rec.ask_price) / 2
    mid <= 0.0 && return 0.0
    return (rec.ask_price - rec.bid_price) / mid
end

"""
    condor_entry_metrics_from_strikes(ctx, short_put_K, short_call_K, long_put_K, long_call_K; rate, div_yield)
        -> Union{Nothing, NamedTuple}

Compute entry-side condor metrics (credit, max loss, widths, per-leg spreads, and leg deltas).
"""
function condor_entry_metrics_from_strikes(
    ctx,
    short_put_K::Float64,
    short_call_K::Float64,
    long_put_K::Float64,
    long_call_K::Float64;
    rate::Float64=DEFAULT_RATE,
    div_yield::Float64=DEFAULT_DIV_YIELD
)::Union{Nothing,NamedTuple}
    put_recs = filter(r -> r.option_type == Put, ctx.recs)
    call_recs = filter(r -> r.option_type == Call, ctx.recs)

    sp = _find_rec_by_strike(put_recs, short_put_K)
    sc = _find_rec_by_strike(call_recs, short_call_K)
    lp = _find_rec_by_strike(put_recs, long_put_K)
    lc = _find_rec_by_strike(call_recs, long_call_K)
    (sp === nothing || sc === nothing || lp === nothing || lc === nothing) && return nothing

    if ismissing(sp.bid_price) || ismissing(sc.bid_price) || ismissing(lp.ask_price) || ismissing(lc.ask_price)
        return nothing
    end

    spot = ctx.surface.spot
    net_credit = (sp.bid_price + sc.bid_price - lp.ask_price - lc.ask_price) * spot
    width_put = short_put_K - long_put_K
    width_call = long_call_K - short_call_K
    max_loss = max(width_put, width_call) - net_credit
    max_loss <= 0.0 && return nothing

    tau = ctx.tau
    tau <= 0.0 && return nothing
    F = spot * exp((rate - div_yield) * tau)

    to_f64_or_zero(x) = x === missing ? 0.0 : Float64(x)
    short_put_delta = to_f64_or_zero(_delta_from_record(sp, F, tau, rate))
    short_call_delta = to_f64_or_zero(_delta_from_record(sc, F, tau, rate))
    long_put_delta = to_f64_or_zero(_delta_from_record(lp, F, tau, rate))
    long_call_delta = to_f64_or_zero(_delta_from_record(lc, F, tau, rate))

    entry_roi = net_credit / max_loss
    avg_leg_spread = (
        _leg_relative_spread(sp) +
        _leg_relative_spread(sc) +
        _leg_relative_spread(lp) +
        _leg_relative_spread(lc)
    ) / 4

    return (
        net_credit=net_credit,
        max_loss=max_loss,
        entry_roi=entry_roi,
        width_put=width_put,
        width_call=width_call,
        short_put_rel_spread=_leg_relative_spread(sp),
        short_call_rel_spread=_leg_relative_spread(sc),
        long_put_rel_spread=_leg_relative_spread(lp),
        long_call_rel_spread=_leg_relative_spread(lc),
        avg_leg_rel_spread=avg_leg_spread,
        short_put_delta=short_put_delta,
        short_call_delta=short_call_delta,
        long_put_delta=long_put_delta,
        long_call_delta=long_call_delta
    )
end

"""
    condor_metrics_from_strikes(ctx, settlement_spot, short_put_K, short_call_K, long_put_K, long_call_K; rate, div_yield)
        -> Union{Nothing, NamedTuple}

Compute realized PnL/ROI for a specific condor, net of bid-ask execution.
"""
function condor_metrics_from_strikes(
    ctx,
    settlement_spot::Float64,
    short_put_K::Float64,
    short_call_K::Float64,
    long_put_K::Float64,
    long_call_K::Float64;
    rate::Float64=DEFAULT_RATE,
    div_yield::Float64=DEFAULT_DIV_YIELD
)::Union{Nothing,NamedTuple}
    entry = condor_entry_metrics_from_strikes(
        ctx,
        short_put_K,
        short_call_K,
        long_put_K,
        long_call_K;
        rate=rate,
        div_yield=div_yield
    )
    entry === nothing && return nothing

    put_payoff = -max(short_put_K - settlement_spot, 0.0) + max(long_put_K - settlement_spot, 0.0)
    call_payoff = -max(settlement_spot - short_call_K, 0.0) + max(settlement_spot - long_call_K, 0.0)
    pnl = put_payoff + call_payoff + entry.net_credit
    roi = pnl / entry.max_loss

    return (
        pnl=pnl,
        roi=roi,
        max_loss=entry.max_loss,
        net_credit=entry.net_credit,
        entry_roi=entry.entry_roi,
        width_put=entry.width_put,
        width_call=entry.width_call,
        short_put_rel_spread=entry.short_put_rel_spread,
        short_call_rel_spread=entry.short_call_rel_spread,
        long_put_rel_spread=entry.long_put_rel_spread,
        long_call_rel_spread=entry.long_call_rel_spread,
        avg_leg_rel_spread=entry.avg_leg_rel_spread,
        short_put_delta=entry.short_put_delta,
        short_call_delta=entry.short_call_delta,
        long_put_delta=entry.long_put_delta,
        long_call_delta=entry.long_call_delta
    )
end

"""
    resolve_condor_from_deltas(ctx, settlement_spot, put_delta, call_delta; ...) -> Union{Nothing, NamedTuple}

Convenience function: given short-leg deltas, find short strikes via
`_delta_strangle_strikes_asymmetric`, select wings via `_condor_wings_by_objective`,
and compute realized P&L via `condor_metrics_from_strikes`.

Returns `(short_put_K, short_call_K, long_put_K, long_call_K, pnl, max_loss)` or `nothing`.
"""
function resolve_condor_from_deltas(
    ctx,
    settlement_spot::Float64,
    put_delta::Float64,
    call_delta::Float64;
    wing_objective::Symbol=:roi,
    target_max_loss::Union{Nothing,Float64}=nothing,
    max_loss_min::Float64=0.0,
    max_loss_max::Float64=Inf,
    min_credit::Float64=0.0,
    min_delta_gap::Float64=0.08,
    prefer_symmetric::Bool=false,
    rate::Float64=DEFAULT_RATE,
    div_yield::Float64=DEFAULT_DIV_YIELD
)::Union{Nothing,NamedTuple}
    shorts = _delta_strangle_strikes_asymmetric(
        ctx, put_delta, call_delta;
        rate=rate, div_yield=div_yield
    )
    shorts === nothing && return nothing
    short_put_K, short_call_K = shorts

    wings = _condor_wings_by_objective(
        ctx, short_put_K, short_call_K;
        objective=wing_objective,
        target_max_loss=target_max_loss,
        max_loss_min=max_loss_min,
        max_loss_max=max_loss_max,
        min_credit=min_credit,
        rate=rate,
        div_yield=div_yield,
        min_delta_gap=min_delta_gap,
        prefer_symmetric=prefer_symmetric,
        debug=false
    )
    wings === nothing && return nothing
    long_put_K, long_call_K = wings

    metrics = condor_metrics_from_strikes(
        ctx, settlement_spot,
        short_put_K, short_call_K, long_put_K, long_call_K;
        rate=rate, div_yield=div_yield
    )
    metrics === nothing && return nothing

    return (
        short_put_K=short_put_K,
        short_call_K=short_call_K,
        long_put_K=long_put_K,
        long_call_K=long_call_K,
        pnl=metrics.pnl,
        max_loss=metrics.max_loss
    )
end

"""
    enumerate_condor_candidates(ctx; ...) -> Vector{NamedTuple}

Enumerate candidate condor structures for one state by scanning short-delta pairs
and resolving wings under the configured policy.
"""
function enumerate_condor_candidates(
    ctx;
    delta_grid::Vector{Float64}=collect(0.05:0.015:0.35),
    max_candidates::Int=400,
    wing_delta_abs::Union{Nothing,Float64}=nothing,
    target_max_loss::Union{Nothing,Float64}=nothing,
    wing_objective::Symbol=:roi,
    max_loss_min::Float64=0.0,
    max_loss_max::Float64=Inf,
    min_credit::Float64=0.0,
    min_delta_gap::Float64=0.08,
    prefer_symmetric::Bool=true,
    rate::Float64=DEFAULT_RATE,
    div_yield::Float64=DEFAULT_DIV_YIELD
)::Vector{NamedTuple}
    if !(wing_objective in (:target_max_loss, :roi, :pnl))
        error("wing_objective must be one of :target_max_loss, :roi, :pnl")
    end

    candidates = NamedTuple[]
    seen = Set{NTuple{4,Float64}}()

    use_fixed_delta_wings = (
        wing_objective == :target_max_loss &&
        target_max_loss === nothing &&
        wing_delta_abs !== nothing
    )

    for put_delta in delta_grid
        for call_delta in delta_grid
            strikes = if use_fixed_delta_wings
                _delta_condor_strikes(
                    ctx,
                    put_delta,
                    call_delta,
                    wing_delta_abs,
                    wing_delta_abs;
                    rate=rate,
                    div_yield=div_yield,
                    min_delta_gap=min_delta_gap
                )
            else
                shorts = _delta_strangle_strikes_asymmetric(
                    ctx,
                    put_delta,
                    call_delta;
                    rate=rate,
                    div_yield=div_yield
                )
                shorts === nothing && continue
                short_put_K, short_call_K = shorts

                wings = _condor_wings_by_objective(
                    ctx,
                    short_put_K,
                    short_call_K;
                    objective=wing_objective,
                    target_max_loss=target_max_loss,
                    max_loss_min=max_loss_min,
                    max_loss_max=max_loss_max,
                    min_credit=min_credit,
                    rate=rate,
                    div_yield=div_yield,
                    min_delta_gap=min_delta_gap,
                    prefer_symmetric=prefer_symmetric,
                    debug=false
                )
                wings === nothing && continue
                long_put_K, long_call_K = wings
                (short_put_K, short_call_K, long_put_K, long_call_K)
            end
            strikes === nothing && continue

            short_put_K, short_call_K, long_put_K, long_call_K = strikes
            key = (short_put_K, short_call_K, long_put_K, long_call_K)
            key in seen && continue
            push!(seen, key)

            push!(candidates, (
                short_put_K=short_put_K,
                short_call_K=short_call_K,
                long_put_K=long_put_K,
                long_call_K=long_call_K,
                short_put_delta=put_delta,
                short_call_delta=call_delta
            ))
        end
    end

    if max_candidates > 0 && length(candidates) > max_candidates
        max_candidates = min(max_candidates, length(candidates))
        raw_idx = collect(round.(Int, range(1, length(candidates), length=max_candidates)))
        idx = unique(clamp.(raw_idx, 1, length(candidates)))
        candidates = candidates[idx]
    end

    return candidates
end

"""
    condor_scoring_feature_vector(state_features, ctx, candidate; rate, div_yield, implied_move_floor)
        -> Union{Nothing, Vector{Float32}}

Build combined state-action features for one condor candidate.
"""
function condor_scoring_feature_vector(
    state_features::Vector{Float32},
    ctx,
    candidate;
    rate::Float64=DEFAULT_RATE,
    div_yield::Float64=DEFAULT_DIV_YIELD,
    implied_move_floor::Float64=1e-6
)::Union{Nothing,Vector{Float32}}
    entry = condor_entry_metrics_from_strikes(
        ctx,
        candidate.short_put_K,
        candidate.short_call_K,
        candidate.long_put_K,
        candidate.long_call_K;
        rate=rate,
        div_yield=div_yield
    )
    entry === nothing && return nothing

    spot = ctx.surface.spot
    spot <= 0.0 && return nothing

    atm_iv = _atm_iv_from_records(
        ctx.recs,
        spot,
        ctx.tau,
        rate,
        div_yield;
        debug=false,
        timestamp=nothing
    )
    implied_move = (atm_iv === nothing || !isfinite(atm_iv)) ? implied_move_floor : max(atm_iv * sqrt(ctx.tau), implied_move_floor)

    short_put_distance = (spot - candidate.short_put_K) / spot
    short_call_distance = (candidate.short_call_K - spot) / spot
    put_width_norm = entry.width_put / spot
    call_width_norm = entry.width_call / spot
    width_asymmetry = abs(put_width_norm - call_width_norm)
    inner_width_norm = (candidate.short_call_K - candidate.short_put_K) / spot
    net_credit_norm = entry.net_credit / spot
    max_loss_norm = entry.max_loss / spot

    put_implied_move_distance = short_put_distance / implied_move
    call_implied_move_distance = short_call_distance / implied_move
    put_delta_gap = abs(abs(entry.short_put_delta) - abs(entry.long_put_delta))
    call_delta_gap = abs(abs(entry.short_call_delta) - abs(entry.long_call_delta))

    candidate_features = Float32[
        candidate.short_put_delta,
        candidate.short_call_delta,
        short_put_distance,
        short_call_distance,
        put_width_norm,
        call_width_norm,
        width_asymmetry,
        inner_width_norm,
        net_credit_norm,
        max_loss_norm,
        entry.entry_roi,
        entry.short_put_rel_spread,
        entry.short_call_rel_spread,
        entry.long_put_rel_spread,
        entry.long_call_rel_spread,
        entry.avg_leg_rel_spread,
        put_implied_move_distance,
        call_implied_move_distance,
        put_delta_gap,
        call_delta_gap
    ]

    length(candidate_features) == N_CONDOR_CANDIDATE_FEATURES || return nothing
    return vcat(state_features, candidate_features)
end

"""
    condor_realized_utility(pnl, max_loss; objective=:roi) -> Float64

Utility target for candidate-scoring models.
"""
function condor_realized_utility(
    pnl::Float64,
    max_loss::Float64;
    objective::Symbol=:roi
)::Float64
    eps = 1e-6
    if objective == :roi
        return pnl / max(max_loss, eps)
    elseif objective == :pnl
        return pnl
    elseif objective == :risk_adjusted
        return pnl / sqrt(max(max_loss, eps))
    else
        error("Unknown utility objective: $objective (expected :roi, :pnl, or :risk_adjusted)")
    end
end

"""
    generate_condor_candidate_training_data(surfaces, settlement_spots, spot_history_dict; ...)
        -> CondorScoringDataset

Generate candidate-level training data for condor action scoring.
"""
function generate_condor_candidate_training_data(
    surfaces::Dict{DateTime,VolatilitySurface},
    settlement_spots::Dict{DateTime,Float64},
    spot_history_dict::Dict{DateTime,SpotHistory};
    rate::Float64=DEFAULT_RATE,
    div_yield::Float64=DEFAULT_DIV_YIELD,
    expiry_interval::Period=Day(1),
    utility_objective::Symbol=:roi,
    candidate_delta_grid::Vector{Float64}=collect(0.05:0.015:0.35),
    max_candidates_per_day::Int=400,
    wing_delta_abs::Union{Nothing,Float64}=nothing,
    target_max_loss::Union{Nothing,Float64}=nothing,
    wing_objective::Symbol=:roi,
    max_loss_min::Float64=0.0,
    max_loss_max::Float64=Inf,
    min_credit::Float64=0.0,
    min_delta_gap::Float64=0.08,
    prefer_symmetric::Bool=true,
    use_logsig::Bool=false,
    prev_surfaces::Union{Nothing,Dict{DateTime,VolatilitySurface}}=nothing,
    verbose::Bool=true
)::CondorScoringDataset
    utility_objective in (:roi, :pnl, :risk_adjusted) || error("utility_objective must be one of :roi, :pnl, :risk_adjusted")

    feature_rows = Vector{Float32}[]
    utilities = Float32[]
    pnls = Float32[]
    max_losses = Float32[]
    timestamps = DateTime[]

    sorted_ts = sort(collect(keys(surfaces)))
    n_total_days = length(sorted_ts)
    n_skipped_days = 0
    n_valid_days = 0
    n_valid_candidates = 0

    for (i, ts) in enumerate(sorted_ts)
        surface = surfaces[ts]
        ctx_info = build_condor_ctx(surface; expiry_interval=expiry_interval)
        if ctx_info === nothing
            n_skipped_days += 1
            continue
        end
        ctx = ctx_info.ctx
        expiry = ctx_info.expiry
        tau = ctx_info.tau

        settlement = get(settlement_spots, expiry, nothing)
        if settlement === nothing
            n_skipped_days += 1
            continue
        end

        spot_history = get(spot_history_dict, ts, nothing)
        prev_surface = prev_surfaces !== nothing ? get(prev_surfaces, ts, nothing) : nothing
        feats = extract_features(surface, tau; spot_history=spot_history, use_logsig=use_logsig, prev_surface=prev_surface)
        if feats === nothing
            n_skipped_days += 1
            continue
        end
        state_features = features_to_vector(feats)

        candidates = enumerate_condor_candidates(
            ctx;
            delta_grid=candidate_delta_grid,
            max_candidates=max_candidates_per_day,
            wing_delta_abs=wing_delta_abs,
            target_max_loss=target_max_loss,
            wing_objective=wing_objective,
            max_loss_min=max_loss_min,
            max_loss_max=max_loss_max,
            min_credit=min_credit,
            min_delta_gap=min_delta_gap,
            prefer_symmetric=prefer_symmetric,
            rate=rate,
            div_yield=div_yield
        )

        if isempty(candidates)
            n_skipped_days += 1
            continue
        end

        n_day_candidates = 0
        for candidate in candidates
            x = condor_scoring_feature_vector(
                state_features,
                ctx,
                candidate;
                rate=rate,
                div_yield=div_yield
            )
            x === nothing && continue

            metrics = condor_metrics_from_strikes(
                ctx,
                settlement,
                candidate.short_put_K,
                candidate.short_call_K,
                candidate.long_put_K,
                candidate.long_call_K;
                rate=rate,
                div_yield=div_yield
            )
            metrics === nothing && continue

            utility = condor_realized_utility(metrics.pnl, metrics.max_loss; objective=utility_objective)
            isfinite(utility) || continue

            push!(feature_rows, x)
            push!(utilities, Float32(utility))
            push!(pnls, Float32(metrics.pnl))
            push!(max_losses, Float32(metrics.max_loss))
            push!(timestamps, ts)
            n_day_candidates += 1
        end

        if n_day_candidates > 0
            n_valid_days += 1
            n_valid_candidates += n_day_candidates
        else
            n_skipped_days += 1
        end

        if verbose && (i % 20 == 0 || i == n_total_days)
            avg_candidates = n_valid_days == 0 ? 0.0 : n_valid_candidates / n_valid_days
            println("  Processed $i / $n_total_days days (valid_days: $n_valid_days, skipped_days: $n_skipped_days, avg_candidates/day: $(round(avg_candidates, digits=1)))")
        end
    end

    isempty(feature_rows) && error("No valid candidate-level training samples generated")

    X = reduce(hcat, feature_rows)
    return CondorScoringDataset(X, utilities, pnls, max_losses, timestamps)
end

"""Core loop for generating delta-prediction training data via grid-search optimization."""
function _generate_delta_training_data_core(
    surfaces::Dict{DateTime,VolatilitySurface},
    settlement_spots::Dict{DateTime,Float64},
    spot_history_dict::Dict{DateTime,SpotHistory},
    optimize_fn::Function;
    expiry_interval::Period=Day(1),
    use_logsig::Bool=false,
    prev_surfaces::Union{Nothing,Dict{DateTime,VolatilitySurface}}=nothing,
    verbose::Bool=true
)::TrainingDataset
    features_list = Vector{Float32}[]
    labels_list = Vector{Float32}[]
    raw_deltas_list = Vector{Float32}[]
    pnls = Float32[]
    timestamps = DateTime[]

    sorted_ts = sort(collect(keys(surfaces)))
    n_total = length(sorted_ts)
    n_skipped = 0
    n_processed = 0

    for (i, ts) in enumerate(sorted_ts)
        surface = surfaces[ts]

        expiry_target = ts + expiry_interval
        expiries = unique(rec.expiry for rec in surface.records)
        isempty(expiries) && continue

        taus = [time_to_expiry(e, ts) for e in expiries]
        tau_target = time_to_expiry(expiry_target, ts)
        idx = argmin(abs.(taus .- tau_target))
        expiry = expiries[idx]
        tau = taus[idx]

        settlement = get(settlement_spots, expiry, nothing)
        if settlement === nothing
            n_skipped += 1
            continue
        end

        spot_history = get(spot_history_dict, ts, nothing)
        prev_surface = prev_surfaces !== nothing ? get(prev_surfaces, ts, nothing) : nothing
        feats = extract_features(surface, tau; spot_history=spot_history, use_logsig=use_logsig, prev_surface=prev_surface)
        if feats === nothing
            n_skipped += 1
            continue
        end

        result = optimize_fn(surface, settlement)
        if result === nothing
            n_skipped += 1
            continue
        end

        best_put_delta, best_call_delta, best_pnl = result

        push!(features_list, features_to_vector(feats))
        push!(raw_deltas_list, Float32[best_put_delta, best_call_delta])
        scaled = unscale_deltas(Float32[best_put_delta, best_call_delta])
        push!(labels_list, scaled)
        push!(pnls, Float32(best_pnl))
        push!(timestamps, ts)

        n_processed += 1

        if verbose && (i % 20 == 0 || i == n_total)
            println("  Processed $i / $n_total (valid: $n_processed, skipped: $n_skipped)")
        end
    end

    isempty(features_list) && error("No valid training samples generated")

    X = reduce(hcat, features_list)
    Y = reduce(hcat, labels_list)
    raw_Y = reduce(hcat, raw_deltas_list)

    size_labels = compute_size_labels(pnls)

    if verbose
        n_positive = count(p -> p > 0, pnls)
        avg_size = mean(size_labels)
        println("  Position sizing: $(n_positive)/$(length(pnls)) profitable, avg_size=$(round(avg_size, digits=3))")
    end

    return TrainingDataset(X, Y, raw_Y, pnls, size_labels, timestamps)
end

"""
    generate_condor_training_data(surfaces, settlement_spots, spot_history_dict;
        rate, div_yield, expiry_interval, wing_delta_abs, target_max_loss, wing_objective,
        max_loss_min, max_loss_max, min_credit, min_delta_gap, prefer_symmetric, use_logsig, verbose)
        -> TrainingDataset

Generate training data for iron condor strike selection by grid searching
optimal inner deltas under the configured wing policy.
"""
function generate_condor_training_data(
    surfaces::Dict{DateTime,VolatilitySurface},
    settlement_spots::Dict{DateTime,Float64},
    spot_history_dict::Dict{DateTime,SpotHistory};
    rate::Float64=DEFAULT_RATE,
    div_yield::Float64=DEFAULT_DIV_YIELD,
    expiry_interval::Period=Day(1),
    wing_delta_abs::Union{Nothing,Float64}=0.05,
    target_max_loss::Union{Nothing,Float64}=nothing,
    wing_objective::Symbol=:target_max_loss,
    max_loss_min::Float64=0.0,
    max_loss_max::Float64=Inf,
    min_credit::Float64=0.0,
    min_delta_gap::Float64=0.08,
    prefer_symmetric::Bool=true,
    use_logsig::Bool=false,
    prev_surfaces::Union{Nothing,Dict{DateTime,VolatilitySurface}}=nothing,
    verbose::Bool=true
)::TrainingDataset
    opt_fn = (surface, settlement) -> find_optimal_condor_deltas(
        surface, settlement;
        wing_delta_abs=wing_delta_abs,
        target_max_loss=target_max_loss,
        wing_objective=wing_objective,
        max_loss_min=max_loss_min,
        max_loss_max=max_loss_max,
        min_credit=min_credit,
        min_delta_gap=min_delta_gap,
        prefer_symmetric=prefer_symmetric,
        rate=rate,
        div_yield=div_yield,
        expiry_interval=expiry_interval
    )
    return _generate_delta_training_data_core(
        surfaces, settlement_spots, spot_history_dict, opt_fn;
        expiry_interval=expiry_interval,
        use_logsig=use_logsig,
        prev_surfaces=prev_surfaces,
        verbose=verbose
    )
end

"""
    simulate_condor_pnl_4d(surface, settlement_spot, sp_delta, sc_delta, lp_delta, lc_delta;
        min_delta_gap, rate, div_yield, expiry_interval)
        -> Union{Float64, Nothing}

Simulate P&L for a short iron condor given all 4 leg deltas directly.
Uses `_delta_condor_strikes` to resolve deltas to strikes — no wing search.

# Arguments
- `surface::VolatilitySurface`: Entry surface
- `settlement_spot::Float64`: Settlement spot price
- `sp_delta::Float64`: Absolute delta for short put
- `sc_delta::Float64`: Absolute delta for short call
- `lp_delta::Float64`: Absolute delta for long put
- `lc_delta::Float64`: Absolute delta for long call
- `min_delta_gap::Float64`: Minimum delta gap between short and long legs
- `rate::Float64`: Risk-free rate
- `div_yield::Float64`: Dividend yield
- `expiry_interval::Period`: Time to expiry

# Returns
- P&L in dollars, or nothing if strikes cannot be found
"""
function simulate_condor_pnl_4d(
    surface::VolatilitySurface,
    settlement_spot::Float64,
    sp_delta::Float64,
    sc_delta::Float64,
    lp_delta::Float64,
    lc_delta::Float64;
    min_delta_gap::Float64=0.02,
    rate::Float64=DEFAULT_RATE,
    div_yield::Float64=DEFAULT_DIV_YIELD,
    expiry_interval::Period=Day(1)
)::Union{NamedTuple{(:pnl, :max_loss),Tuple{Float64,Float64}},Nothing}
    env = _prepare_simulation_env(surface, expiry_interval, rate, div_yield; compute_strikes=true)
    env === nothing && return nothing
    (; put_recs, call_recs, ctx) = env

    strikes = _delta_condor_strikes(
        ctx, sp_delta, sc_delta, lp_delta, lc_delta;
        rate=rate, div_yield=div_yield, min_delta_gap=min_delta_gap
    )
    strikes === nothing && return nothing

    short_put_K, short_call_K, long_put_K, long_call_K = strikes

    short_put_rec = _find_rec_by_strike(put_recs, short_put_K)
    short_call_rec = _find_rec_by_strike(call_recs, short_call_K)
    long_put_rec = _find_rec_by_strike(put_recs, long_put_K)
    long_call_rec = _find_rec_by_strike(call_recs, long_call_K)
    (short_put_rec === nothing || short_call_rec === nothing ||
        long_put_rec === nothing || long_call_rec === nothing) && return nothing

    if ismissing(short_put_rec.bid_price) || ismissing(short_call_rec.bid_price) ||
       ismissing(long_put_rec.ask_price) || ismissing(long_call_rec.ask_price)
        return nothing
    end

    entry_cost = (short_put_rec.bid_price * (-1) + short_call_rec.bid_price * (-1) +
                  long_put_rec.ask_price * (1) + long_call_rec.ask_price * (1)) * surface.spot

    put_payoff = -max(short_put_K - settlement_spot, 0.0) + max(long_put_K - settlement_spot, 0.0)
    call_payoff = -max(settlement_spot - short_call_K, 0.0) + max(settlement_spot - long_call_K, 0.0)

    pnl = put_payoff + call_payoff - entry_cost

    # Max loss = wing width - credit received
    credit = -entry_cost
    wing_width = max(short_put_K - long_put_K, long_call_K - short_call_K)
    max_loss = wing_width - credit
    max_loss = max(max_loss, 1e-6)  # Guard against zero/negative max_loss

    return (pnl=pnl, max_loss=max_loss)
end

"""Grid search over all 4 delta dimensions for iron condor optimization.
Optimizes ROI (pnl/max_loss) to find tight, efficient condors."""
function _find_optimal_condor_deltas_4d_grid(
    short_delta_grid::Vector{Float64},
    long_delta_grid::Vector{Float64},
    simulate_fn::Function
)::Union{Tuple{Float64,Float64,Float64,Float64,Float64},Nothing}
    best_roi = -Inf
    best_pnl = -Inf
    best_sp = 0.15
    best_sc = 0.15
    best_lp = 0.05
    best_lc = 0.05
    n_valid = 0

    for sp_delta in short_delta_grid
        for sc_delta in short_delta_grid
            for lp_delta in long_delta_grid
                for lc_delta in long_delta_grid
                    result = simulate_fn(sp_delta, sc_delta, lp_delta, lc_delta)
                    result === nothing && continue
                    n_valid += 1

                    roi = result.pnl / result.max_loss
                    if roi > best_roi
                        best_roi = roi
                        best_pnl = result.pnl
                        best_sp = sp_delta
                        best_sc = sc_delta
                        best_lp = lp_delta
                        best_lc = lc_delta
                    end
                end
            end
        end
    end

    n_valid < 1 && return nothing
    return (best_sp, best_sc, best_lp, best_lc, best_pnl)
end

"""
    find_optimal_condor_deltas_4d(surface, settlement_spot;
        min_delta_gap, rate, div_yield, expiry_interval, short_delta_grid, long_delta_grid)
        -> Union{Tuple{Float64, Float64, Float64, Float64, Float64}, Nothing}

Find optimal 4-delta iron condor via grid search over all legs.

# Returns
- Tuple of (best_sp_delta, best_sc_delta, best_lp_delta, best_lc_delta, best_pnl) or nothing
"""
function find_optimal_condor_deltas_4d(
    surface::VolatilitySurface,
    settlement_spot::Float64;
    min_delta_gap::Float64=0.02,
    rate::Float64=DEFAULT_RATE,
    div_yield::Float64=DEFAULT_DIV_YIELD,
    expiry_interval::Period=Day(1),
    short_delta_grid::Vector{Float64}=DELTA_GRID,
    long_delta_grid::Vector{Float64}=LONG_DELTA_GRID
)::Union{Tuple{Float64,Float64,Float64,Float64,Float64},Nothing}
    sim_fn = (sp, sc, lp, lc) -> simulate_condor_pnl_4d(
        surface, settlement_spot, sp, sc, lp, lc;
        min_delta_gap=min_delta_gap, rate=rate, div_yield=div_yield,
        expiry_interval=expiry_interval
    )
    return _find_optimal_condor_deltas_4d_grid(short_delta_grid, long_delta_grid, sim_fn)
end

"""Core loop for generating 4-delta condor training data via grid-search optimization."""
function _generate_condor_4d_training_data_core(
    surfaces::Dict{DateTime,VolatilitySurface},
    settlement_spots::Dict{DateTime,Float64},
    spot_history_dict::Dict{DateTime,SpotHistory},
    optimize_fn::Function;
    expiry_interval::Period=Day(1),
    use_logsig::Bool=false,
    prev_surfaces::Union{Nothing,Dict{DateTime,VolatilitySurface}}=nothing,
    verbose::Bool=true
)::TrainingDataset
    features_list = Vector{Float32}[]
    labels_list = Vector{Float32}[]
    raw_deltas_list = Vector{Float32}[]
    pnls = Float32[]
    timestamps = DateTime[]

    sorted_ts = sort(collect(keys(surfaces)))
    n_total = length(sorted_ts)
    n_skipped = 0
    n_processed = 0

    for (i, ts) in enumerate(sorted_ts)
        surface = surfaces[ts]

        expiry_target = ts + expiry_interval
        expiries = unique(rec.expiry for rec in surface.records)
        isempty(expiries) && continue

        taus = [time_to_expiry(e, ts) for e in expiries]
        tau_target = time_to_expiry(expiry_target, ts)
        idx = argmin(abs.(taus .- tau_target))
        expiry = expiries[idx]
        tau = taus[idx]

        settlement = get(settlement_spots, expiry, nothing)
        if settlement === nothing
            n_skipped += 1
            continue
        end

        spot_history = get(spot_history_dict, ts, nothing)
        prev_surface = prev_surfaces !== nothing ? get(prev_surfaces, ts, nothing) : nothing
        feats = extract_features(surface, tau; spot_history=spot_history, use_logsig=use_logsig, prev_surface=prev_surface)
        if feats === nothing
            n_skipped += 1
            continue
        end

        result = optimize_fn(surface, settlement)
        if result === nothing
            n_skipped += 1
            continue
        end

        best_sp, best_sc, best_lp, best_lc, best_pnl = result

        push!(features_list, features_to_vector(feats))
        raw = Float32[best_sp, best_sc, best_lp, best_lc]
        push!(raw_deltas_list, raw)
        scaled = unscale_deltas_4d(reshape(raw, 4, 1))
        push!(labels_list, vec(scaled))
        push!(pnls, Float32(best_pnl))
        push!(timestamps, ts)

        n_processed += 1

        if verbose && (i % 20 == 0 || i == n_total)
            println("  Processed $i / $n_total (valid: $n_processed, skipped: $n_skipped)")
        end
    end

    isempty(features_list) && error("No valid 4D condor training samples generated")

    X = reduce(hcat, features_list)
    Y = reduce(hcat, labels_list)
    raw_Y = reduce(hcat, raw_deltas_list)

    # No position sizing for 4D models — fill with zeros
    size_labels = zeros(Float32, length(pnls))

    if verbose
        n_positive = count(p -> p > 0, pnls)
        println("  4D condor data: $(n_processed) samples, $(n_positive)/$(length(pnls)) profitable")
    end

    return TrainingDataset(X, Y, raw_Y, pnls, size_labels, timestamps)
end

"""
    generate_condor_4d_training_data(surfaces, settlement_spots, spot_history_dict;
        min_delta_gap, rate, div_yield, expiry_interval, short_delta_grid, long_delta_grid,
        use_logsig, prev_surfaces, verbose)
        -> TrainingDataset

Generate training data for 4-delta iron condor strike selection by grid searching
all 4 leg deltas independently.
"""
function generate_condor_4d_training_data(
    surfaces::Dict{DateTime,VolatilitySurface},
    settlement_spots::Dict{DateTime,Float64},
    spot_history_dict::Dict{DateTime,SpotHistory};
    min_delta_gap::Float64=0.02,
    rate::Float64=DEFAULT_RATE,
    div_yield::Float64=DEFAULT_DIV_YIELD,
    expiry_interval::Period=Day(1),
    short_delta_grid::Vector{Float64}=DELTA_GRID,
    long_delta_grid::Vector{Float64}=LONG_DELTA_GRID,
    use_logsig::Bool=false,
    prev_surfaces::Union{Nothing,Dict{DateTime,VolatilitySurface}}=nothing,
    verbose::Bool=true
)::TrainingDataset
    opt_fn = (surface, settlement) -> find_optimal_condor_deltas_4d(
        surface, settlement;
        min_delta_gap=min_delta_gap, rate=rate, div_yield=div_yield,
        expiry_interval=expiry_interval,
        short_delta_grid=short_delta_grid, long_delta_grid=long_delta_grid
    )
    return _generate_condor_4d_training_data_core(
        surfaces, settlement_spots, spot_history_dict, opt_fn;
        expiry_interval=expiry_interval,
        use_logsig=use_logsig,
        prev_surfaces=prev_surfaces,
        verbose=verbose
    )
end

"""
    compute_size_labels(pnls; temperature, center_on_zero) -> Vector{Float32}

Compute position size labels from P&L values using tanh normalization.

Position sizing intuition:
- Profitable trades → positive size (short vol / sell strangle)
- Losing trades → negative size (long vol / buy strangle)
- Temperature controls how aggressive the sizing is (lower = more extreme)

# Arguments
- `pnls::Vector{Float32}`: Raw P&L values (positive = profit when selling vol)
- `temperature::Float64`: Scaling factor for tanh (default: 1.0)
- `center_on_zero::Bool`: If true, center on 0 (profitable=sell, losing=buy).
  If false, center on mean P&L (default: true)

# Returns
- Vector of position sizes in [-1, 1]
  - +1: Strong signal to sell vol (short strangle)
  - 0: No trade / neutral
  - -1: Strong signal to buy vol (long strangle)
"""
function compute_size_labels(
    pnls::Vector{Float32};
    temperature::Float64=1.0,
    center_on_zero::Bool=true
)::Vector{Float32}
    isempty(pnls) && return Float32[]

    σ = std(pnls)
    σ = max(σ, 1e-6)  # Avoid division by zero

    if center_on_zero
        # Center on 0: positive P&L → sell, negative P&L → buy
        z_scores = pnls ./ (σ * temperature)
    else
        # Center on mean (relative performance)
        μ = mean(pnls)
        z_scores = (pnls .- μ) ./ (σ * temperature)
    end

    size_labels = Float32.(tanh.(z_scores))

    return size_labels
end

"""
    generate_training_data(surfaces, settlement_spots, spot_history_dict; rate, div_yield, expiry_interval, use_logsig, verbose)
        -> TrainingDataset

Generate training data from surfaces by grid searching for optimal deltas.

# Arguments
- `surfaces::Dict{DateTime,VolatilitySurface}`: Entry surfaces
- `settlement_spots::Dict{DateTime,Float64}`: Settlement prices by expiry time
- `spot_history_dict::Dict{DateTime,SpotHistory}`: Historical spots for each entry time
- `rate::Float64`: Risk-free rate
- `div_yield::Float64`: Dividend yield
- `expiry_interval::Period`: Time to expiry
- `use_logsig::Bool`: Use log-signature features instead of signature
- `verbose::Bool`: Print progress

# Returns
- `TrainingDataset` with features, labels, and metadata
"""
function generate_training_data(
    surfaces::Dict{DateTime,VolatilitySurface},
    settlement_spots::Dict{DateTime,Float64},
    spot_history_dict::Dict{DateTime,SpotHistory};
    rate::Float64=DEFAULT_RATE,
    div_yield::Float64=DEFAULT_DIV_YIELD,
    expiry_interval::Period=Day(1),
    use_logsig::Bool=false,
    prev_surfaces::Union{Nothing,Dict{DateTime,VolatilitySurface}}=nothing,
    verbose::Bool=true
)::TrainingDataset
    opt_fn = (surface, settlement) -> find_optimal_deltas(
        surface, settlement;
        rate=rate, div_yield=div_yield, expiry_interval=expiry_interval
    )
    return _generate_delta_training_data_core(
        surfaces, settlement_spots, spot_history_dict, opt_fn;
        expiry_interval=expiry_interval,
        use_logsig=use_logsig,
        prev_surfaces=prev_surfaces,
        verbose=verbose
    )
end

"""Generic training loop with early stopping, used by both `train_model!` and `train_scoring_model!`."""
function _train_loop!(
    model,
    n_samples::Int,
    get_batch::Function,
    compute_loss::Function,
    compute_val_loss::Union{Nothing,Function},
    history_keys::Vector{String};
    on_batch_end::Union{Nothing,Function}=nothing,
    format_epoch::Union{Nothing,Function}=nothing,
    epochs::Int=100,
    batch_size::Int=32,
    learning_rate::Float64=1e-3,
    patience::Int=10,
    verbose::Bool=true
)
    opt_state = Flux.setup(Adam(learning_rate), model)
    best_val_loss = Inf
    patience_counter = 0
    best_model_state = nothing
    history = Dict{String,Vector{Float64}}(k => Float64[] for k in history_keys)

    for epoch in 1:epochs
        perm = randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        accum = Dict{String,Float64}()

        Flux.trainmode!(model)

        for batch_start in 1:batch_size:n_samples
            batch_end = min(batch_start + batch_size - 1, n_samples)
            batch_idx = perm[batch_start:batch_end]
            batch_data = get_batch(batch_idx)

            loss, grads = Flux.withgradient(model) do m
                compute_loss(m, batch_data)
            end
            Flux.update!(opt_state, model, grads[1])

            epoch_loss += loss
            n_batches += 1

            if on_batch_end !== nothing
                on_batch_end(model, batch_data, accum)
            end
        end

        avg_train_loss = epoch_loss / n_batches
        push!(history["train_loss"], Float64(avg_train_loss))

        Flux.testmode!(model)
        val_loss = if compute_val_loss !== nothing
            compute_val_loss(model)
        else
            avg_train_loss
        end
        push!(history["val_loss"], Float64(val_loss))

        extra_str = ""
        if format_epoch !== nothing
            extra_str = format_epoch(accum, n_batches, history)
        end

        if val_loss < best_val_loss
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = Flux.state(model)
        else
            patience_counter += 1
        end

        if verbose && (epoch % 10 == 0 || epoch == 1 || patience_counter >= patience)
            val_str = compute_val_loss !== nothing ? ", val=$(round(val_loss, digits=6))" : ""
            println("  Epoch $epoch: train=$(round(avg_train_loss, digits=6))$val_str$extra_str")
        end

        if patience_counter >= patience
            verbose && println("  Early stopping at epoch $epoch")
            break
        end
    end

    if best_model_state !== nothing && compute_val_loss !== nothing
        Flux.loadmodel!(model, best_model_state)
    end

    return model, history
end

"""
    train_model!(model, train_data; val_data, epochs, batch_size, learning_rate, patience, delta_weight, size_weight, verbose)
        -> (model, history)

Train the neural network model on delta predictions, and optionally position size.

Dispatches on model output dimension:
- 2-output (strangle deltas) or 4-output (condor deltas): trains with delta_loss only (MSE on all outputs)
- 3-output (legacy): includes position size loss weighted by `size_weight`

# Arguments
- `model`: Flux model to train (2, 3, or 4 outputs)
- `train_data::TrainingDataset`: Training data
- `val_data::Union{Nothing,TrainingDataset}`: Validation data (optional)
- `epochs::Int`: Maximum training epochs
- `batch_size::Int`: Batch size
- `learning_rate::Float64`: Learning rate for Adam optimizer
- `patience::Int`: Early stopping patience
- `delta_weight::Float32`: Weight for delta prediction loss (default: 1.0)
- `size_weight::Float32`: Weight for position size prediction loss (default: 1.0, only for 3-output)
- `verbose::Bool`: Print progress

# Returns
- Trained model and training history dict
"""
function train_model!(
    model,
    train_data::TrainingDataset;
    val_data::Union{Nothing,TrainingDataset}=nothing,
    epochs::Int=100,
    batch_size::Int=32,
    learning_rate::Float64=1e-3,
    patience::Int=10,
    delta_weight::Float32=1.0f0,
    size_weight::Float32=1.0f0,
    verbose::Bool=true
)
    n_samples = size(train_data.features, 2)

    # Detect model output dimension from a test forward pass
    test_output = model(train_data.features[:, 1:1])
    n_model_outputs = size(test_output, 1)

    if n_model_outputs == 3
        # Legacy 3-output: delta + position size
        get_batch = batch_idx -> (
            x=train_data.features[:, batch_idx],
            y_deltas=train_data.labels[:, batch_idx],
            y_sizes=train_data.size_labels[batch_idx]
        )

        loss_fn = (m, bd) -> combined_loss(m, bd.x, bd.y_deltas, bd.y_sizes;
            delta_weight=delta_weight, size_weight=size_weight)

        val_fn = val_data !== nothing ? (m -> begin
            combined_loss(m, val_data.features, val_data.labels, val_data.size_labels;
                delta_weight=delta_weight, size_weight=size_weight)
        end) : nothing

        batch_end_fn = (m, bd, accum) -> begin
            preds = m(bd.x)
            accum["delta"] = get(accum, "delta", 0.0) + Float64(mse(preds[1:2, :], bd.y_deltas))
            accum["size"] = get(accum, "size", 0.0) + Float64(mse(preds[3, :], bd.y_sizes))
        end

        fmt_fn = (accum, n_batches, history) -> begin
            avg_d = get(accum, "delta", 0.0) / n_batches
            avg_s = get(accum, "size", 0.0) / n_batches
            push!(history["train_delta_loss"], avg_d)
            push!(history["train_size_loss"], avg_s)
            " (δ=$(round(avg_d, digits=4)), sz=$(round(avg_s, digits=4)))"
        end

        return _train_loop!(
            model, n_samples, get_batch, loss_fn, val_fn,
            ["train_loss", "val_loss", "train_delta_loss", "train_size_loss"];
            on_batch_end=batch_end_fn, format_epoch=fmt_fn,
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            patience=patience, verbose=verbose
        )
    else
        # 2-output (strangle) or 4-output (condor): pure delta loss
        get_batch = batch_idx -> (
            x=train_data.features[:, batch_idx],
            y_deltas=train_data.labels[:, batch_idx]
        )

        loss_fn = (m, bd) -> delta_loss(m, bd.x, bd.y_deltas)

        val_fn = val_data !== nothing ? (m -> begin
            delta_loss(m, val_data.features, val_data.labels)
        end) : nothing

        batch_end_fn = (m, bd, accum) -> begin
            preds = m(bd.x)
            accum["delta"] = get(accum, "delta", 0.0) + Float64(mse(preds, bd.y_deltas))
        end

        fmt_fn = (accum, n_batches, history) -> begin
            avg_d = get(accum, "delta", 0.0) / n_batches
            push!(history["train_delta_loss"], avg_d)
            " (δ=$(round(avg_d, digits=4)))"
        end

        return _train_loop!(
            model, n_samples, get_batch, loss_fn, val_fn,
            ["train_loss", "val_loss", "train_delta_loss"];
            on_batch_end=batch_end_fn, format_epoch=fmt_fn,
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            patience=patience, verbose=verbose
        )
    end
end

"""
    evaluate_model(model, data; min_delta, max_delta) -> Dict

Evaluate model on a dataset. Dispatches on model output dimension:
- 4-output: uses `scale_deltas_4d` for per-leg scaling
- 3-output (legacy): uses `scale_deltas` for short deltas + position size eval
- 2-output: uses `scale_deltas` for short deltas only

# Returns
Dict with:
- `delta_mse`: Mean squared error on delta predictions
- `delta_mae`: Mean absolute error on deltas
- `size_mse`: Mean squared error on position sizes (0.0 for 2/4-output)
- `size_mae`: Mean absolute error on position sizes (0.0 for 2/4-output)
- `pred_deltas`: Predicted deltas (n_outputs × n_samples)
- `true_deltas`: True deltas (n_outputs × n_samples)
- `pred_sizes`: Predicted position sizes (n_samples,) — zeros for 2/4-output
- `true_sizes`: True position sizes (n_samples,) — zeros for 2/4-output
"""
function evaluate_model(
    model,
    data::TrainingDataset;
    min_delta::Float32=0.05f0,
    max_delta::Float32=0.35f0
)::Dict
    Flux.testmode!(model)

    raw_output = model(data.features)
    n_outputs = size(raw_output, 1)
    n_samples = size(raw_output, 2)
    true_deltas = data.raw_deltas

    if n_outputs == 4
        pred_deltas = scale_deltas_4d(raw_output)
        delta_mse = mean((pred_deltas .- true_deltas).^2)
        delta_mae = mean(abs.(pred_deltas .- true_deltas))

        return Dict(
            "delta_mse" => delta_mse,
            "delta_mae" => delta_mae,
            "size_mse" => 0.0,
            "size_mae" => 0.0,
            "pred_deltas" => pred_deltas,
            "true_deltas" => true_deltas,
            "pred_sizes" => zeros(Float32, n_samples),
            "true_sizes" => zeros(Float32, n_samples)
        )
    elseif n_outputs == 3
        pred_deltas = scale_deltas(raw_output[1:2, :]; min_delta=min_delta, max_delta=max_delta)
        delta_mse = mean((pred_deltas .- true_deltas).^2)
        delta_mae = mean(abs.(pred_deltas .- true_deltas))

        pred_sizes = vec(raw_output[3, :])
        true_sizes = data.size_labels
        size_mse = mean((pred_sizes .- true_sizes).^2)
        size_mae = mean(abs.(pred_sizes .- true_sizes))

        return Dict(
            "delta_mse" => delta_mse,
            "delta_mae" => delta_mae,
            "size_mse" => size_mse,
            "size_mae" => size_mae,
            "pred_deltas" => pred_deltas,
            "true_deltas" => true_deltas,
            "pred_sizes" => pred_sizes,
            "true_sizes" => true_sizes
        )
    else
        # 2-output (strangle)
        pred_deltas = scale_deltas(raw_output[1:2, :]; min_delta=min_delta, max_delta=max_delta)
        delta_mse = mean((pred_deltas .- true_deltas).^2)
        delta_mae = mean(abs.(pred_deltas .- true_deltas))

        return Dict(
            "delta_mse" => delta_mse,
            "delta_mae" => delta_mae,
            "size_mse" => 0.0,
            "size_mae" => 0.0,
            "pred_deltas" => pred_deltas,
            "true_deltas" => true_deltas,
            "pred_sizes" => zeros(Float32, n_samples),
            "true_sizes" => zeros(Float32, n_samples)
        )
    end
end

"""
    train_scoring_model!(model, train_data; val_data, epochs, batch_size, learning_rate, patience, verbose)
        -> (model, history)

Train a scalar regression model for candidate utility prediction.
"""
function train_scoring_model!(
    model,
    train_data::CondorScoringDataset;
    val_data::Union{Nothing,CondorScoringDataset}=nothing,
    epochs::Int=100,
    batch_size::Int=256,
    learning_rate::Float64=1e-3,
    patience::Int=10,
    verbose::Bool=true
)
    n_samples = size(train_data.features, 2)

    get_batch = batch_idx -> (
        x=train_data.features[:, batch_idx],
        y=train_data.utilities[batch_idx]
    )

    loss_fn = (m, bd) -> begin
        preds = vec(m(bd.x))
        mse(preds, bd.y)
    end

    val_fn = val_data !== nothing ? (m -> begin
        preds = vec(m(val_data.features))
        mse(preds, val_data.utilities)
    end) : nothing

    return _train_loop!(
        model, n_samples, get_batch, loss_fn, val_fn,
        ["train_loss", "val_loss"];
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        patience=patience, verbose=verbose
    )
end

"""
    evaluate_scoring_model(model, data) -> Dict

Evaluate a trained candidate-scoring model.
"""
function evaluate_scoring_model(
    model,
    data::CondorScoringDataset
)::Dict
    Flux.testmode!(model)
    preds = vec(model(data.features))
    y = data.utilities

    mse_val = mean((preds .- y).^2)
    mae_val = mean(abs.(preds .- y))

    return Dict(
        "utility_mse" => mse_val,
        "utility_mae" => mae_val,
        "pred_utilities" => preds,
        "true_utilities" => y
    )
end
