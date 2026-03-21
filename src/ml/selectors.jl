# ML Selectors — abstract type and concrete implementations

"""
    MLSelector

Abstract type for ML-based strike selectors. All subtypes must be callable:
`(sel::MLSelector)(ctx::StrikeSelectionContext) -> (sp_K, sc_K, lp_K, lc_K) | nothing`
"""
abstract type MLSelector end

# =============================================================================
# ScoredCandidateSelector — enumerate candidates, score with model, pick best
# =============================================================================

"""
    ScoredCandidateSelector

Enumerates candidate condors from a delta grid, scores each with a trained
neural network, and returns the highest-scoring candidate.

Plugs into `IronCondorStrategy` as a strike selector:
`(sel::ScoredCandidateSelector)(ctx::StrikeSelectionContext) -> (sp_K, sc_K, lp_K, lc_K) | nothing`

# Fields
- `model`: trained Flux model
- `feature_means`, `feature_stds`: normalization parameters
- `surface_features`: which surface features to extract
- `candidate_features`: which candidate features to extract
- `delta_grid`: candidate generation grid
- `rate`, `div_yield`: pricing parameters
- `max_loss`: max loss constraint (USD)
- `max_spread_rel`: bid-ask spread filter
"""
struct ScoredCandidateSelector{M} <: MLSelector
    model::M
    feature_means::Vector{Float32}
    feature_stds::Vector{Float32}
    surface_features::Vector{<:Feature}
    candidate_features::Vector{<:CandidateFeature}
    delta_grid
    rate::Float64
    div_yield::Float64
    max_loss::Float64
    max_spread_rel::Float64
end

function ScoredCandidateSelector(
    model,
    feature_means::Vector{Float32},
    feature_stds::Vector{Float32};
    surface_features::Vector{<:Feature}=default_surface_features(),
    candidate_features::Vector{<:CandidateFeature}=default_candidate_features(),
    delta_grid=0.08:0.02:0.30,
    rate::Float64=0.0,
    div_yield::Float64=0.0,
    max_loss::Float64=Inf,
    max_spread_rel::Float64=Inf
)
    ScoredCandidateSelector(
        model, feature_means, feature_stds,
        surface_features, candidate_features,
        delta_grid, rate, div_yield, max_loss, max_spread_rel
    )
end

function (sel::ScoredCandidateSelector)(ctx::StrikeSelectionContext)
    # 1. Extract surface features (handles multi-output features like SpotLogSig)
    sf_vec = extract_surface_features(ctx, sel.surface_features)
    sf_vec === nothing && return nothing

    # 2. Generate candidates from delta grid
    candidates = _enumerate_condor_candidates(
        ctx, sel.delta_grid, sel.candidate_features;
        rate=sel.rate, div_yield=sel.div_yield,
        max_spread_rel=sel.max_spread_rel,
        max_loss_max=sel.max_loss
    )
    isempty(candidates) && return nothing

    # 3. Score all candidates
    cf_matrix = hcat([c[2] for c in candidates]...)
    scores = score_candidates(
        sel.model, sf_vec, cf_matrix;
        feature_means=sel.feature_means,
        feature_stds=sel.feature_stds
    )

    # 4. Return highest-scoring
    return candidates[argmax(scores)][1]
end

# =============================================================================
# DirectDeltaSelector — predict optimal deltas from surface features
# =============================================================================

"""
    DirectDeltaSelector

Predicts optimal (put_delta, call_delta) directly from surface features using a
trained regression model, then resolves to strikes and wings mechanically.

No candidate enumeration — the model output is the delta pair.

# Fields
- `model`: trained Flux model (input: surface features, output: 2 deltas)
- `feature_means`, `feature_stds`: normalization parameters (surface features only)
- `surface_features`: which surface features to extract
- `rate`, `div_yield`: pricing parameters
- `max_loss`: max loss constraint for wing selection
- `max_spread_rel`: bid-ask spread filter on short legs
- `delta_clamp`: (min, max) to clamp predicted deltas (default (0.05, 0.40))
"""
struct DirectDeltaSelector{M} <: MLSelector
    model::M
    feature_means::Vector{Float32}
    feature_stds::Vector{Float32}
    surface_features::Vector{<:Feature}
    rate::Float64
    div_yield::Float64
    max_loss::Float64
    max_spread_rel::Float64
    delta_clamp::Tuple{Float64,Float64}
end

function DirectDeltaSelector(
    model,
    feature_means::Vector{Float32},
    feature_stds::Vector{Float32};
    surface_features::Vector{<:Feature}=default_surface_features(),
    rate::Float64=0.0,
    div_yield::Float64=0.0,
    max_loss::Float64=Inf,
    max_spread_rel::Float64=Inf,
    delta_clamp::Tuple{Float64,Float64}=(0.05, 0.40)
)
    DirectDeltaSelector(
        model, feature_means, feature_stds,
        surface_features, rate, div_yield, max_loss, max_spread_rel, delta_clamp
    )
end

function (sel::DirectDeltaSelector)(ctx::StrikeSelectionContext)
    # 1. Extract surface features (handles multi-output features like SpotLogSig)
    sf_vec = extract_surface_features(ctx, sel.surface_features)
    sf_vec === nothing && return nothing

    # 2. Normalize and predict deltas
    safe_stds = max.(sel.feature_stds, Float32(1e-8))
    x_norm = (sf_vec .- sel.feature_means) ./ safe_stds
    raw_output = vec(sel.model(reshape(x_norm, :, 1)))

    # Clamp to valid delta range
    lo, hi = sel.delta_clamp
    pd = clamp(Float64(raw_output[1]), lo, hi)
    cd = clamp(Float64(raw_output[2]), lo, hi)

    # 3. Resolve deltas to short strikes
    shorts = _delta_strangle_strikes_asymmetric(
        ctx, pd, cd; rate=sel.rate, div_yield=sel.div_yield
    )
    shorts === nothing && return nothing
    sp_K, sc_K = shorts

    # 4. Optional spread filter
    if isfinite(sel.max_spread_rel)
        recs = _ctx_recs(ctx)
        put_recs = filter(r -> r.option_type == Put, recs)
        call_recs = filter(r -> r.option_type == Call, recs)
        sp_rec = _find_rec_by_strike(put_recs, sp_K)
        sc_rec = _find_rec_by_strike(call_recs, sc_K)
        for rec in (sp_rec, sc_rec)
            rec === nothing && continue
            spread = _relative_spread(rec)
            if spread !== nothing && spread > sel.max_spread_rel
                return nothing
            end
        end
    end

    # 5. Select wings
    wings = _condor_wings_by_objective(
        ctx, sp_K, sc_K;
        objective=:roi,
        max_loss_max=sel.max_loss,
        rate=sel.rate,
        div_yield=sel.div_yield
    )
    wings === nothing && return nothing
    lp_K, lc_K = wings

    return (sp_K, sc_K, lp_K, lc_K)
end

# Backward compat alias
const MLCondorSelector = ScoredCandidateSelector

# =============================================================================
# SizedIronCondorStrategy — ML-modulated quantity
# =============================================================================

"""
    SizedIronCondorStrategy

Iron condor strategy where quantity is modulated by an ML model that predicts
the baseline condor's PnL from surface features.

# Fields
- `schedule::Vector{DateTime}`: Entry timestamps
- `expiry_interval::Period`: Time from entry to expiry
- `strike_selector`: Callable `f(ctx) -> (sp_K, sc_K, lp_K, lc_K) | nothing`
- `model`: Trained Flux model (surface features → predicted PnL)
- `feature_means`, `feature_stds`: Normalization parameters
- `surface_features`: Which surface features to extract
- `sizing_policy`: Callable `(predicted_pnl) -> quantity`
- `debug::Bool`: Emit diagnostics
"""
struct SizedIronCondorStrategy{F,M,P} <: ScheduledStrategy
    schedule::Vector{DateTime}
    expiry_interval::Period
    strike_selector::F
    model::M
    feature_means::Vector{Float32}
    feature_stds::Vector{Float32}
    surface_features::Vector{<:Feature}
    sizing_policy::P
    debug::Bool
end

function SizedIronCondorStrategy(
    schedule::Vector{DateTime},
    expiry_interval::Period,
    strike_selector,
    model,
    feature_means::Vector{Float32},
    feature_stds::Vector{Float32};
    surface_features::Vector{<:Feature}=default_surface_features(),
    sizing_policy=linear_sizing(),
    debug::Bool=false
)
    return SizedIronCondorStrategy(
        schedule, expiry_interval, strike_selector,
        model, feature_means, feature_stds, surface_features,
        sizing_policy, debug
    )
end

entry_schedule(strategy::SizedIronCondorStrategy)::Vector{DateTime} = strategy.schedule

function entry_positions(
    strategy::SizedIronCondorStrategy,
    surface::VolatilitySurface,
    history::BacktestDataSource=DictDataSource(Dict{DateTime,VolatilitySurface}(), Dict{DateTime,Float64}())
)::Vector{Position}
    expiry_info = _select_expiry(strategy.expiry_interval, surface)
    if expiry_info === nothing
        strategy.debug && println("No entry: no valid expiry for timestamp=$(surface.timestamp)")
        return Position[]
    end
    expiry = expiry_info[1]

    ctx = StrikeSelectionContext(surface, expiry, history)

    # 1. Select strikes via baseline selector
    selector_result = strategy.strike_selector(ctx)
    if selector_result === nothing
        strategy.debug && println("No entry: invalid condor strikes (timestamp=$(surface.timestamp))")
        return Position[]
    end
    sp_K, sc_K, lp_K, lc_K = selector_result

    # 2. Extract surface features → normalize → predict PnL
    sf_vec = extract_surface_features(ctx, strategy.surface_features)
    if sf_vec === nothing
        strategy.debug && println("No entry: feature extraction failed (timestamp=$(surface.timestamp))")
        return Position[]
    end

    safe_stds = max.(strategy.feature_stds, Float32(1e-8))
    x_norm = (sf_vec .- strategy.feature_means) ./ safe_stds
    predicted_pnl = Float64(vec(strategy.model(reshape(x_norm, :, 1)))[1])

    # 3. Apply sizing policy
    quantity = strategy.sizing_policy(predicted_pnl)
    if quantity <= 0.0
        strategy.debug && println("No entry: sizing policy returned q=$(quantity) (pred_pnl=$(predicted_pnl))")
        return Position[]
    end

    trades = Trade[
        Trade(surface.underlying, sp_K, expiry, Put; direction=-1, quantity=quantity),
        Trade(surface.underlying, sc_K, expiry, Call; direction=-1, quantity=quantity),
        Trade(surface.underlying, lp_K, expiry, Put; direction=1, quantity=quantity),
        Trade(surface.underlying, lc_K, expiry, Call; direction=1, quantity=quantity),
    ]

    return _open_positions(trades, surface; debug=strategy.debug)
end
