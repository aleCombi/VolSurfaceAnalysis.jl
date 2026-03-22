# ML Training — data generation, training loop, and ScoredCandidateSelector

# =============================================================================
# Utility functions (training labels)
# =============================================================================

"""ROI utility: pnl / max_loss."""
roi_utility(pnl::Float64, max_loss::Float64) = pnl / max_loss

"""Raw PnL utility."""
pnl_utility(pnl::Float64, max_loss::Float64) = pnl

# =============================================================================
# Sizing policies
# =============================================================================

"""
    linear_sizing(; threshold=1.0, max_q=3.0, skip_negative=true)

Returns a sizing policy `(predicted_pnl) -> quantity`.
Linearly scales from 0 to `max_q` based on `predicted_pnl / threshold`.
If `skip_negative`, returns 0 when predicted PnL ≤ 0.
"""
function linear_sizing(; threshold::Float64=1.0, max_q::Float64=3.0, skip_negative::Bool=true)
    return function(predicted_pnl::Float64)
        if skip_negative && predicted_pnl <= 0.0
            return 0.0
        end
        return clamp(predicted_pnl / threshold, 0.0, max_q)
    end
end

"""
    binary_sizing(; threshold=0.0, quantity=1.0)

Returns a sizing policy that trades `quantity` when predicted PnL > threshold, else 0.
Simple regime filter: trade or don't trade.
"""
function binary_sizing(; threshold::Float64=0.0, quantity::Float64=1.0)
    return function(predicted_pnl::Float64)
        return predicted_pnl > threshold ? quantity : 0.0
    end
end

"""
    probability_sizing(; threshold=0.5, quantity=1.0)

Returns a sizing policy for classifier models that output logits.
Applies sigmoid to get probability, trades `quantity` when P(win) > threshold.
"""
function probability_sizing(; threshold::Float64=0.5, quantity::Float64=1.0)
    return function(logit::Float64)
        prob = 1.0 / (1.0 + exp(-logit))
        return prob > threshold ? quantity : 0.0
    end
end

"""
    sigmoid_sizing(; scale=1.0, max_q=3.0)

Returns a smooth sizing policy `(predicted_pnl) -> quantity`.
Always trades (never returns 0): `max_q / (1 + exp(-pnl/scale))`.
"""
function sigmoid_sizing(; scale::Float64=1.0, max_q::Float64=3.0)
    return function(predicted_pnl::Float64)
        return max_q / (1.0 + exp(-predicted_pnl / scale))
    end
end

"""
    risk_adjusted_utility(λ=1.0)

Returns a utility function `(pnl, max_loss) -> roi - λ * roi²`.
Penalizes extreme returns (positive or negative), favoring consistency.
Higher λ = more penalty on variance.
"""
risk_adjusted_utility(λ::Float64=1.0) = (pnl::Float64, max_loss::Float64) -> begin
    roi = pnl / max_loss
    roi - λ * roi^2
end

# =============================================================================
# Shared condor settlement helper
# =============================================================================

"""
    _settle_condor(ctx, sp_K, sc_K, lp_K, lc_K, settlement) -> NamedTuple | nothing

Build 4 condor trades, open positions, settle, and compute credit/max_loss.
Returns `nothing` if positions can't be opened.
Returns `(pnl, credit, max_loss)` otherwise — caller decides whether to filter on max_loss.
"""
function _settle_condor(
    ctx::StrikeSelectionContext, sp_K, sc_K, lp_K, lc_K, settlement::Float64
)
    trades = Trade[
        Trade(ctx.surface.underlying, sp_K, ctx.expiry, Put;  direction=-1, quantity=1.0),
        Trade(ctx.surface.underlying, sc_K, ctx.expiry, Call; direction=-1, quantity=1.0),
        Trade(ctx.surface.underlying, lp_K, ctx.expiry, Put;  direction=1, quantity=1.0),
        Trade(ctx.surface.underlying, lc_K, ctx.expiry, Call; direction=1, quantity=1.0),
    ]

    positions = _open_positions(trades, ctx.surface)
    isempty(positions) && return nothing

    pnl = settle(positions, settlement)
    credit = -sum(entry_cost(p) for p in positions)
    max_loss = max(sp_K - lp_K, lc_K - sc_K) - credit

    return (pnl=pnl, credit=credit, max_loss=max_loss)
end

# =============================================================================
# Training data
# =============================================================================

"""
    TrainingExample

A single (features, label) pair for training the scoring model.
"""
struct TrainingExample
    surface_features::Vector{Float32}
    candidate_features::Vector{Float32}
    label::Float32
end

"""
    _enumerate_condor_candidates(ctx, delta_grid, candidate_features; kwargs...)

Enumerate candidate condors from a delta grid, returning `(strikes, feature_vec)` tuples.
Shared by `generate_training_data` and `ScoredCandidateSelector`.

Returns `Vector{Tuple{NTuple{4,Float64}, Vector{Float32}}}`.
"""
function _enumerate_condor_candidates(
    ctx::StrikeSelectionContext,
    delta_grid,
    candidate_features::Vector{<:CandidateFeature};
    rate::Float64=0.0,
    div_yield::Float64=0.0,
    max_spread_rel::Float64=Inf,
    wing_kwargs...
)
    recs = _ctx_recs(ctx)
    candidates = Tuple{NTuple{4,Float64}, Vector{Float32}}[]

    for pd in delta_grid, cd in delta_grid
        shorts = _delta_strangle_strikes_asymmetric(
            ctx, Float64(pd), Float64(cd); rate=rate, div_yield=div_yield
        )
        shorts === nothing && continue
        sp_K, sc_K = shorts

        _check_short_spreads(ctx, sp_K, sc_K, max_spread_rel) || continue

        wings = _select_condor_wings(
            ctx, sp_K, sc_K;
            max_spread_rel=max_spread_rel,
            rate=rate, div_yield=div_yield, wing_kwargs...
        )
        wings === nothing && continue
        lp_K, lc_K = wings

        cf = map(f -> f(ctx, sp_K, sc_K, lp_K, lc_K), candidate_features)
        any(isnothing, cf) && continue
        push!(candidates, ((sp_K, sc_K, lp_K, lc_K), Float32.(cf)))
    end

    return candidates
end

"""
    generate_training_data(source, expiry_interval, schedule; kwargs...) -> Vector{TrainingExample}

Generate training examples by iterating over entry timestamps and enumerating
candidate condors from a delta grid. Uses `each_entry` for the timestamp/expiry loop.

# Keyword Arguments
- `delta_grid`: range of delta values for candidate generation (default 0.08:0.02:0.30)
- `rate`, `div_yield`: pricing parameters
- `utility`: label function `(pnl, max_loss) -> Float64` (default `roi_utility`)
- `surface_features`: vector of `Feature` instances
- `candidate_features`: vector of `CandidateFeature` instances
- `wing_kwargs...`: passed to `_select_condor_wings`
"""
function generate_training_data(
    source::BacktestDataSource,
    expiry_interval::Period,
    schedule::Vector{DateTime};
    delta_grid=0.08:0.02:0.30,
    rate::Float64=0.0,
    div_yield::Float64=0.0,
    utility=roi_utility,
    surface_features::Vector{<:Feature}=default_surface_features(; rate, div_yield),
    candidate_features::Vector{<:CandidateFeature}=default_candidate_features(; rate, div_yield),
    wing_kwargs...
)
    examples = TrainingExample[]

    each_entry(source, expiry_interval, schedule) do ctx, settlement
        ismissing(settlement) && return

        # Extract surface features (handles multi-output features like SpotLogSig)
        sf_vec = extract_surface_features(ctx, surface_features)
        sf_vec === nothing && return

        candidates = _enumerate_condor_candidates(
            ctx, delta_grid, candidate_features;
            rate=rate, div_yield=div_yield, wing_kwargs...
        )

        for ((sp_K, sc_K, lp_K, lc_K), cf_vec) in candidates
            result = _settle_condor(ctx, sp_K, sc_K, lp_K, lc_K, Float64(settlement))
            result === nothing && continue
            result.max_loss <= 0.0 && continue

            label = utility(result.pnl, result.max_loss)
            push!(examples, TrainingExample(sf_vec, cf_vec, Float32(label)))
        end
    end

    return examples
end

# =============================================================================
# Sizing training data
# =============================================================================

"""
    SizingTrainingExample

A single (surface_features, pnl) pair for training the sizing model.
One example per entry: the PnL of the baseline condor at that surface state.
"""
struct SizingTrainingExample
    surface_features::Vector{Float32}
    pnl::Float32
end

"""
    generate_sizing_training_data(source, expiry_interval, schedule, strike_selector; kwargs...)

Generate sizing training data: run the baseline `strike_selector` at each entry,
open positions, settle, and record `(surface_features, pnl)`.

Returns `Vector{SizingTrainingExample}`.

# Keyword Arguments
- `rate`, `div_yield`: pricing parameters
- `surface_features`: vector of `Feature` instances
"""
function generate_sizing_training_data(
    source::BacktestDataSource,
    expiry_interval::Period,
    schedule::Vector{DateTime},
    strike_selector;
    rate::Float64=0.0,
    div_yield::Float64=0.0,
    surface_features::Vector{<:Feature}=default_surface_features(; rate, div_yield)
)
    examples = SizingTrainingExample[]

    each_entry(source, expiry_interval, schedule) do ctx, settlement
        ismissing(settlement) && return

        # Extract surface features
        sf_vec = extract_surface_features(ctx, surface_features)
        sf_vec === nothing && return

        # Run baseline selector
        selector_result = strike_selector(ctx)
        selector_result === nothing && return
        sp_K, sc_K, lp_K, lc_K = selector_result

        result = _settle_condor(ctx, sp_K, sc_K, lp_K, lc_K, Float64(settlement))
        result === nothing && return

        push!(examples, SizingTrainingExample(sf_vec, Float32(result.pnl)))
    end

    return examples
end

# =============================================================================
# Delta regression training data
# =============================================================================

"""
    DeltaTrainingExample

A single (surface_features, optimal_deltas) pair for training a delta regression model.
"""
struct DeltaTrainingExample
    surface_features::Vector{Float32}
    put_delta::Float32
    call_delta::Float32
end

"""
    _best_delta_pair(ctx, delta_pairs, settlement, utility; kwargs...) -> (pd, cd) | nothing

Search delta_pairs for the condor with highest utility. Returns the best
(put_delta, call_delta) or `nothing` if no valid condor found.
"""
function _best_delta_pair(
    ctx::StrikeSelectionContext, delta_pairs, settlement::Float64, utility;
    rate::Float64=0.0, div_yield::Float64=0.0, wing_kwargs...
)
    best_utility = -Inf
    best_pd = 0.0
    best_cd = 0.0

    for (pd, cd) in delta_pairs
        shorts = _delta_strangle_strikes_asymmetric(
            ctx, Float64(pd), Float64(cd); rate=rate, div_yield=div_yield
        )
        shorts === nothing && continue
        sp_K, sc_K = shorts

        wings = _select_condor_wings(
            ctx, sp_K, sc_K;
            rate=rate, div_yield=div_yield, wing_kwargs...
        )
        wings === nothing && continue
        lp_K, lc_K = wings

        result = _settle_condor(ctx, sp_K, sc_K, lp_K, lc_K, settlement)
        result === nothing && continue
        result.max_loss <= 0.0 && continue

        u = utility(result.pnl, result.max_loss)
        if u > best_utility
            best_utility = u
            best_pd = pd
            best_cd = cd
        end
    end

    return isfinite(best_utility) ? (best_pd, best_cd) : nothing
end

"""
    generate_delta_training_data(source, expiry_interval, schedule; kwargs...) -> Vector{DeltaTrainingExample}

Generate training examples for delta regression: for each entry timestamp, enumerate
all candidate condors, evaluate their ROI, and record the delta pair of the best one.

# Keyword Arguments
- `delta_grid`: range of delta values for candidate generation (default 0.08:0.02:0.30)
- `rate`, `div_yield`: pricing parameters
- `utility`: label function `(pnl, max_loss) -> Float64` (default `roi_utility`)
- `surface_features`: vector of `Feature` instances
- `symmetric`: if true, only consider pd == cd pairs (1D search) (default false)
- `wing_kwargs...`: passed to `_select_condor_wings`
"""
function generate_delta_training_data(
    source::BacktestDataSource,
    expiry_interval::Period,
    schedule::Vector{DateTime};
    delta_grid=0.08:0.02:0.30,
    rate::Float64=0.0,
    div_yield::Float64=0.0,
    utility=roi_utility,
    surface_features::Vector{<:Feature}=default_surface_features(; rate, div_yield),
    symmetric::Bool=false,
    wing_kwargs...
)
    examples = DeltaTrainingExample[]

    delta_pairs = symmetric ?
        [(d, d) for d in delta_grid] :
        [(pd, cd) for pd in delta_grid for cd in delta_grid]

    each_entry(source, expiry_interval, schedule) do ctx, settlement
        ismissing(settlement) && return

        sf_vec = extract_surface_features(ctx, surface_features)
        sf_vec === nothing && return

        best = _best_delta_pair(ctx, delta_pairs, Float64(settlement), utility;
            rate=rate, div_yield=div_yield, wing_kwargs...)
        best === nothing && return

        push!(examples, DeltaTrainingExample(sf_vec, Float32(best[1]), Float32(best[2])))
    end

    return examples
end

# =============================================================================
# Element-wise loss functions
# =============================================================================

"""Element-wise squared error: `(ŷ .- y) .^ 2`."""
mse_loss(ŷ, y) = (ŷ .- y) .^ 2

"""Element-wise logit binary cross-entropy (numerically stable, applies sigmoid internally)."""
bce_loss(ŷ, y) = Flux.logitbinarycrossentropy.(ŷ, y)

# =============================================================================
# Generic training loop (works with any X → Y matrices)
# =============================================================================

"""
    train_model!(model, X, Y; loss_fn, sample_weight_fn, epochs, lr, batch_size, val_fraction, patience)

Train a Flux model on feature matrix X (dims × samples) and target matrix Y
(output_dim × samples).

# Keyword Arguments
- `loss_fn`: element-wise loss `(ŷ, y) -> array` (default `mse_loss`)
- `sample_weight_fn`: optional `Y_train -> W_train` for class balancing (default `nothing`)
- `epochs`, `lr`, `batch_size`, `val_fraction`, `patience`: training hyperparameters

Returns `(model, feature_means, feature_stds, history)`.
"""
function train_model!(
    model,
    X::Matrix{Float32},
    Y::Matrix{Float32};
    loss_fn=mse_loss,
    sample_weight_fn=nothing,
    epochs::Int=100,
    lr::Float64=1e-3,
    batch_size::Int=64,
    val_fraction::Float64=0.2,
    patience::Int=10
)
    n = size(X, 2)
    n_val = max(1, round(Int, n * val_fraction))
    n_train = n - n_val

    perm = randperm(n)
    X_train, Y_train = X[:, perm[1:n_train]], Y[:, perm[1:n_train]]
    X_val, Y_val = X[:, perm[n_train+1:end]], Y[:, perm[n_train+1:end]]

    feature_means = vec(mean(X_train; dims=2))
    feature_stds = vec(std(X_train; dims=2))

    X_train_norm = _normalize(X_train, feature_means, feature_stds)
    X_val_norm = _normalize(X_val, feature_means, feature_stds)

    W_train = sample_weight_fn !== nothing ?
        sample_weight_fn(Y_train) : ones(Float32, size(Y_train))

    opt_state = Flux.setup(Adam(lr), model)

    train_losses = Float64[]
    val_losses = Float64[]
    best_val_loss = Inf
    best_epoch = 0

    for epoch in 1:epochs
        perm_train = randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in 1:batch_size:n_train
            stop = min(start + batch_size - 1, n_train)
            batch_idx = perm_train[start:stop]
            X_batch = X_train_norm[:, batch_idx]
            Y_batch = Y_train[:, batch_idx]
            W_batch = W_train[:, batch_idx]

            loss, grads = Flux.withgradient(model) do m
                element_losses = loss_fn(m(X_batch), Y_batch)
                sum(W_batch .* element_losses) / sum(W_batch)
            end

            Flux.update!(opt_state, model, grads[1])
            epoch_loss += loss
            n_batches += 1
        end

        push!(train_losses, epoch_loss / n_batches)

        # Validation: always unweighted mean
        val_loss = mean(loss_fn(model(X_val_norm), Y_val))
        push!(val_losses, val_loss)

        if val_loss < best_val_loss
            best_val_loss = val_loss
            best_epoch = epoch
        elseif epoch - best_epoch >= patience
            break
        end
    end

    return (model, feature_means, feature_stds,
            (train_loss=train_losses, val_loss=val_losses))
end

"""
    train_classifier!(model, X, Y; pos_weight=1.0, kwargs...)

Train a binary classifier with logit BCE loss and class-balanced weighting.
Thin wrapper around `train_model!`.

`pos_weight` upweights the positive class: samples with Y=1 get weight `pos_weight`,
samples with Y=0 get weight 1.0.

Returns `(model, feature_means, feature_stds, history)`.
"""
function train_classifier!(
    model,
    X::Matrix{Float32},
    Y::Matrix{Float32};
    pos_weight::Float64=1.0,
    kwargs...
)
    weight_fn = Y_train -> Float32.(1.0 .+ (pos_weight - 1.0) .* Y_train)
    train_model!(model, X, Y;
        loss_fn=bce_loss,
        sample_weight_fn=weight_fn,
        kwargs...
    )
end

# =============================================================================
# Training loop (TrainingExample-based, for ScoredCandidateSelector)
# =============================================================================

"""
    train_scoring_model!(model, data; kwargs...)

Train a scoring model on `TrainingExample` data with MSE loss.
Builds X/Y matrices from the examples, then delegates to `train_model!`.

Returns `(model, feature_means, feature_stds, history)`.
"""
function train_scoring_model!(
    model,
    data::Vector{TrainingExample};
    kwargs...
)
    n = length(data)
    sf_dim = length(data[1].surface_features)
    cf_dim = length(data[1].candidate_features)
    total_dim = sf_dim + cf_dim

    X = Matrix{Float32}(undef, total_dim, n)
    Y = Matrix{Float32}(undef, 1, n)
    for (j, ex) in enumerate(data)
        X[1:sf_dim, j] = ex.surface_features
        X[sf_dim+1:end, j] = ex.candidate_features
        Y[1, j] = ex.label
    end

    return train_model!(model, X, Y; kwargs...)
end

