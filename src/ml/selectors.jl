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
    # 1. Extract surface features
    sf = map(f -> f(ctx), sel.surface_features)
    any(isnothing, sf) && return nothing
    sf_vec = Float32.(sf)

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
    # 1. Extract surface features
    sf = map(f -> f(ctx), sel.surface_features)
    any(isnothing, sf) && return nothing
    sf_vec = Float32.(sf)

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
