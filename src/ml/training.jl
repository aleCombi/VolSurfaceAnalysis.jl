# ML Training — data generation, training loop, and MLCondorSelector

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
Shared by `generate_training_data` and `MLCondorSelector`.

Returns `Vector{Tuple{NTuple{4,Float64}, Vector{Float32}}}`.
"""
function _enumerate_condor_candidates(
    ctx::StrikeSelectionContext,
    delta_grid,
    candidate_features::Vector{<:CandidateFeature};
    rate::Float64=0.0,
    div_yield::Float64=0.0,
    wing_objective::Symbol=:roi,
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

        # Optional spread filter on short legs
        if isfinite(max_spread_rel)
            put_recs = filter(r -> r.option_type == Put, recs)
            call_recs = filter(r -> r.option_type == Call, recs)
            sp_rec = _find_rec_by_strike(put_recs, sp_K)
            sc_rec = _find_rec_by_strike(call_recs, sc_K)
            skip = false
            for rec in (sp_rec, sc_rec)
                rec === nothing && continue
                spread = _relative_spread(rec)
                if spread !== nothing && spread > max_spread_rel
                    skip = true
                    break
                end
            end
            skip && continue
        end

        wings = _condor_wings_by_objective(
            ctx, sp_K, sc_K;
            objective=wing_objective,
            rate=rate,
            div_yield=div_yield,
            wing_kwargs...
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
- `wing_objective`: objective for wing selection (default `:roi`)
- `wing_kwargs...`: passed to `_condor_wings_by_objective`
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
    wing_objective::Symbol=:roi,
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
            rate=rate, div_yield=div_yield, wing_objective=wing_objective, wing_kwargs...
        )

        for ((sp_K, sc_K, lp_K, lc_K), cf_vec) in candidates
            # Build trades and open positions
            trades = Trade[
                Trade(ctx.surface.underlying, sp_K, ctx.expiry, Put; direction=-1, quantity=1.0),
                Trade(ctx.surface.underlying, sc_K, ctx.expiry, Call; direction=-1, quantity=1.0),
                Trade(ctx.surface.underlying, lp_K, ctx.expiry, Put; direction=1, quantity=1.0),
                Trade(ctx.surface.underlying, lc_K, ctx.expiry, Call; direction=1, quantity=1.0),
            ]

            positions = _open_positions(trades, ctx.surface)
            isempty(positions) && continue

            pnl = settle(positions, Float64(settlement))

            # Compute max loss from spread widths and credit
            put_spread_width = (sp_K - lp_K)
            call_spread_width = (lc_K - sc_K)
            credit = -sum(entry_cost(p) for p in positions)
            max_loss = max(put_spread_width, call_spread_width) - credit
            max_loss <= 0.0 && continue

            label = utility(pnl, max_loss)
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

        # Build trades and open positions
        trades = Trade[
            Trade(ctx.surface.underlying, sp_K, ctx.expiry, Put; direction=-1, quantity=1.0),
            Trade(ctx.surface.underlying, sc_K, ctx.expiry, Call; direction=-1, quantity=1.0),
            Trade(ctx.surface.underlying, lp_K, ctx.expiry, Put; direction=1, quantity=1.0),
            Trade(ctx.surface.underlying, lc_K, ctx.expiry, Call; direction=1, quantity=1.0),
        ]

        positions = _open_positions(trades, ctx.surface)
        isempty(positions) && return

        pnl = settle(positions, Float64(settlement))
        push!(examples, SizingTrainingExample(sf_vec, Float32(pnl)))
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
    generate_delta_training_data(source, expiry_interval, schedule; kwargs...) -> Vector{DeltaTrainingExample}

Generate training examples for delta regression: for each entry timestamp, enumerate
all candidate condors, evaluate their ROI, and record the delta pair of the best one.

# Keyword Arguments
- `delta_grid`: range of delta values for candidate generation (default 0.08:0.02:0.30)
- `rate`, `div_yield`: pricing parameters
- `utility`: label function `(pnl, max_loss) -> Float64` (default `roi_utility`)
- `surface_features`: vector of `Feature` instances
- `wing_objective`: objective for wing selection (default `:roi`)
- `symmetric`: if true, only consider pd == cd pairs (1D search) (default false)
- `wing_kwargs...`: passed to `_condor_wings_by_objective`
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
    wing_objective::Symbol=:roi,
    symmetric::Bool=false,
    wing_kwargs...
)
    examples = DeltaTrainingExample[]

    delta_pairs = symmetric ?
        [(d, d) for d in delta_grid] :
        [(pd, cd) for pd in delta_grid for cd in delta_grid]

    each_entry(source, expiry_interval, schedule) do ctx, settlement
        ismissing(settlement) && return

        # Extract surface features (handles multi-output features like SpotLogSig)
        sf_vec = extract_surface_features(ctx, surface_features)
        sf_vec === nothing && return

        best_utility = -Inf
        best_pd = 0.0
        best_cd = 0.0

        for (pd, cd) in delta_pairs
            shorts = _delta_strangle_strikes_asymmetric(
                ctx, Float64(pd), Float64(cd); rate=rate, div_yield=div_yield
            )
            shorts === nothing && continue
            sp_K, sc_K = shorts

            wings = _condor_wings_by_objective(
                ctx, sp_K, sc_K;
                objective=wing_objective,
                rate=rate,
                div_yield=div_yield,
                wing_kwargs...
            )
            wings === nothing && continue
            lp_K, lc_K = wings

            # Build trades and open positions
            trades = Trade[
                Trade(ctx.surface.underlying, sp_K, ctx.expiry, Put; direction=-1, quantity=1.0),
                Trade(ctx.surface.underlying, sc_K, ctx.expiry, Call; direction=-1, quantity=1.0),
                Trade(ctx.surface.underlying, lp_K, ctx.expiry, Put; direction=1, quantity=1.0),
                Trade(ctx.surface.underlying, lc_K, ctx.expiry, Call; direction=1, quantity=1.0),
            ]

            positions = _open_positions(trades, ctx.surface)
            isempty(positions) && continue

            pnl = settle(positions, Float64(settlement))

            put_spread_width = (sp_K - lp_K)
            call_spread_width = (lc_K - sc_K)
            credit = -sum(entry_cost(p) for p in positions)
            max_loss = max(put_spread_width, call_spread_width) - credit
            max_loss <= 0.0 && continue

            u = utility(pnl, max_loss)
            if u > best_utility
                best_utility = u
                best_pd = pd
                best_cd = cd
            end
        end

        if isfinite(best_utility)
            push!(examples, DeltaTrainingExample(sf_vec, Float32(best_pd), Float32(best_cd)))
        end
    end

    return examples
end

# =============================================================================
# Generic training loop (works with any X → Y matrices)
# =============================================================================

"""
    train_model!(model, X, Y; epochs=100, lr=1e-3, batch_size=64, val_fraction=0.2, patience=10)

Train a Flux model on feature matrix X (dims × samples) and target matrix Y
(output_dim × samples) with MSE loss.

Returns `(model, feature_means, feature_stds, history)`.
"""
function train_model!(
    model,
    X::Matrix{Float32},
    Y::Matrix{Float32};
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
    safe_stds = max.(feature_stds, Float32(1e-8))

    X_train_norm = (X_train .- feature_means) ./ safe_stds
    X_val_norm = (X_val .- feature_means) ./ safe_stds

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

            loss, grads = Flux.withgradient(model) do m
                Flux.mse(m(X_batch), Y_batch)
            end

            Flux.update!(opt_state, model, grads[1])
            epoch_loss += loss
            n_batches += 1
        end

        push!(train_losses, epoch_loss / n_batches)

        val_loss = Flux.mse(model(X_val_norm), Y_val)
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

# =============================================================================
# Training loop (TrainingExample-based, for ScoredCandidateSelector)
# =============================================================================

"""
    train_scoring_model!(model, data; epochs=100, lr=1e-3, batch_size=64, val_fraction=0.2)

Train a scoring model on `TrainingExample` data with MSE loss.

Returns `(model, feature_means, feature_stds, history)` where `history` is a
NamedTuple with `train_loss` and `val_loss` vectors.
"""
function train_scoring_model!(
    model,
    data::Vector{TrainingExample};
    epochs::Int=100,
    lr::Float64=1e-3,
    batch_size::Int=64,
    val_fraction::Float64=0.2,
    patience::Int=10
)
    n = length(data)
    n_val = max(1, round(Int, n * val_fraction))
    n_train = n - n_val

    # Shuffle
    perm = randperm(n)
    train_idx = perm[1:n_train]
    val_idx = perm[n_train+1:end]

    # Build feature matrices
    function build_matrix(indices)
        sf_dim = length(data[1].surface_features)
        cf_dim = length(data[1].candidate_features)
        total_dim = sf_dim + cf_dim
        X = Matrix{Float32}(undef, total_dim, length(indices))
        Y = Vector{Float32}(undef, length(indices))
        for (j, i) in enumerate(indices)
            X[1:sf_dim, j] = data[i].surface_features
            X[sf_dim+1:end, j] = data[i].candidate_features
            Y[j] = data[i].label
        end
        return X, Y
    end

    X_train, Y_train = build_matrix(train_idx)
    X_val, Y_val = build_matrix(val_idx)

    # Compute normalization stats from training set
    feature_means = vec(mean(X_train; dims=2))
    feature_stds = vec(std(X_train; dims=2))
    safe_stds = max.(feature_stds, Float32(1e-8))

    # Normalize
    X_train_norm = (X_train .- feature_means) ./ safe_stds
    X_val_norm = (X_val .- feature_means) ./ safe_stds

    opt_state = Flux.setup(Adam(lr), model)

    train_losses = Float64[]
    val_losses = Float64[]
    best_val_loss = Inf
    best_epoch = 0

    for epoch in 1:epochs
        # Mini-batch training
        perm_train = randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in 1:batch_size:n_train
            stop = min(start + batch_size - 1, n_train)
            batch_idx = perm_train[start:stop]
            X_batch = X_train_norm[:, batch_idx]
            Y_batch = Y_train[batch_idx]

            loss, grads = Flux.withgradient(model) do m
                pred = vec(m(X_batch))
                Flux.mse(pred, Y_batch)
            end

            Flux.update!(opt_state, model, grads[1])
            epoch_loss += loss
            n_batches += 1
        end

        avg_train_loss = epoch_loss / n_batches
        push!(train_losses, avg_train_loss)

        # Validation loss
        val_pred = vec(model(X_val_norm))
        val_loss = Flux.mse(val_pred, Y_val)
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

