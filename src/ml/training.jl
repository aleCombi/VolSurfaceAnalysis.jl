# ML Training — data generation, training loop, and MLCondorSelector

# =============================================================================
# Utility functions (training labels)
# =============================================================================

"""ROI utility: pnl / max_loss."""
roi_utility(pnl::Float64, max_loss::Float64) = pnl / max_loss

"""Raw PnL utility."""
pnl_utility(pnl::Float64, max_loss::Float64) = pnl

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

        # Extract surface features
        sf = map(f -> f(ctx), surface_features)
        any(isnothing, sf) && return
        sf_vec = Float32.(sf)

        recs = _ctx_recs(ctx)
        tau = _ctx_tau(ctx)
        spot = ctx.surface.spot

        for pd in delta_grid, cd in delta_grid
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

            # Compute max loss from spread widths and credit
            put_spread_width = (sp_K - lp_K)
            call_spread_width = (lc_K - sc_K)
            credit = -sum(entry_cost(p) for p in positions)
            max_loss = max(put_spread_width, call_spread_width) - credit
            max_loss <= 0.0 && continue

            label = utility(pnl, max_loss)

            # Extract candidate features
            cf = map(f -> f(ctx, sp_K, sc_K, lp_K, lc_K), candidate_features)
            any(isnothing, cf) && continue

            push!(examples, TrainingExample(sf_vec, Float32.(cf), Float32(label)))
        end
    end

    return examples
end

# =============================================================================
# Training loop
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

# =============================================================================
# MLCondorSelector — trained selector that plugs into IronCondorStrategy
# =============================================================================

"""
    MLCondorSelector

A trained strike selector that scores candidate condors using a neural network.
Plugs directly into `IronCondorStrategy` as a strike selector:
`(sel::MLCondorSelector)(ctx::StrikeSelectionContext) -> (sp_K, sc_K, lp_K, lc_K) | nothing`

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
struct MLCondorSelector{M}
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

function MLCondorSelector(
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
    MLCondorSelector(
        model, feature_means, feature_stds,
        surface_features, candidate_features,
        delta_grid, rate, div_yield, max_loss, max_spread_rel
    )
end

function (sel::MLCondorSelector)(ctx::StrikeSelectionContext)
    # 1. Extract surface features
    sf = map(f -> f(ctx), sel.surface_features)
    any(isnothing, sf) && return nothing
    sf_vec = Float32.(sf)

    # 2. Generate candidates from delta grid
    recs = _ctx_recs(ctx)
    tau = _ctx_tau(ctx)
    spot = ctx.surface.spot

    candidates = Tuple{NTuple{4,Float64}, Vector{Float32}}[]

    for pd in sel.delta_grid, cd in sel.delta_grid
        shorts = _delta_strangle_strikes_asymmetric(
            ctx, Float64(pd), Float64(cd); rate=sel.rate, div_yield=sel.div_yield
        )
        shorts === nothing && continue
        sp_K, sc_K = shorts

        # Optional spread check on shorts
        if isfinite(sel.max_spread_rel)
            put_recs = filter(r -> r.option_type == Put, recs)
            call_recs = filter(r -> r.option_type == Call, recs)
            sp_rec = _find_rec_by_strike(put_recs, sp_K)
            sc_rec = _find_rec_by_strike(call_recs, sc_K)
            skip = false
            for rec in (sp_rec, sc_rec)
                rec === nothing && continue
                spread = _relative_spread(rec)
                if spread !== nothing && spread > sel.max_spread_rel
                    skip = true
                    break
                end
            end
            skip && continue
        end

        wings = _condor_wings_by_objective(
            ctx, sp_K, sc_K;
            objective=:roi,
            max_loss_max=sel.max_loss,
            rate=sel.rate,
            div_yield=sel.div_yield
        )
        wings === nothing && continue
        lp_K, lc_K = wings

        cf = map(f -> f(ctx, sp_K, sc_K, lp_K, lc_K), sel.candidate_features)
        any(isnothing, cf) && continue
        push!(candidates, ((sp_K, sc_K, lp_K, lc_K), Float32.(cf)))
    end

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
