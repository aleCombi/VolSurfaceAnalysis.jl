# ML Training Pipeline for Strike Selection
# Generates training data and trains the neural network

using Flux.Optimise: Adam

"""
    TrainingDataset

Container for training data.

# Fields
- `features::Matrix{Float32}`: Input features (n_features × n_samples)
- `labels::Matrix{Float32}`: Target deltas scaled to [0,1] (2 × n_samples)
- `raw_deltas::Matrix{Float32}`: Original delta values (2 × n_samples)
- `pnls::Vector{Float32}`: Best P&L for each sample
- `timestamps::Vector{DateTime}`: Entry timestamps
"""
struct TrainingDataset
    features::Matrix{Float32}
    labels::Matrix{Float32}
    raw_deltas::Matrix{Float32}
    pnls::Vector{Float32}
    timestamps::Vector{DateTime}
end

# Delta grid for searching optimal strikes
const DELTA_GRID = collect(0.05:0.025:0.35)

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
    rate::Float64=0.045,
    div_yield::Float64=0.013,
    expiry_interval::Period=Day(1)
)::Union{Float64,Nothing}
    # Find expiry
    expiry_result = _select_expiry(expiry_interval, surface)
    expiry_result === nothing && return nothing

    expiry, _, tau = expiry_result
    tau <= 0.0 && return nothing

    # Get records for this expiry
    recs = filter(r -> r.expiry == expiry, surface.records)
    isempty(recs) && return nothing

    put_recs = filter(r -> r.option_type == Put, recs)
    call_recs = filter(r -> r.option_type == Call, recs)
    isempty(put_recs) && return nothing
    isempty(call_recs) && return nothing

    F = surface.spot * exp((rate - div_yield) * tau)

    # Find strikes by delta
    short_put_K = _best_delta_strike(put_recs, -put_delta, surface.spot, :put, F, tau, rate)
    short_call_K = _best_delta_strike(call_recs, call_delta, surface.spot, :call, F, tau, rate)

    (short_put_K === nothing || short_call_K === nothing) && return nothing

    # Get entry prices (bid for shorts)
    put_rec = nothing
    call_rec = nothing
    for rec in put_recs
        if rec.strike == short_put_K
            put_rec = rec
            break
        end
    end
    for rec in call_recs
        if rec.strike == short_call_K
            call_rec = rec
            break
        end
    end

    (put_rec === nothing || call_rec === nothing) && return nothing
    (ismissing(put_rec.bid_price) || ismissing(call_rec.bid_price)) && return nothing

    # Entry cost (negative for shorts = premium received)
    entry_put = put_rec.bid_price * (-1) * surface.spot  # Short put
    entry_call = call_rec.bid_price * (-1) * surface.spot  # Short call
    entry_cost = entry_put + entry_call

    # Settlement payoff
    put_payoff = max(short_put_K - settlement_spot, 0.0) * (-1)  # Short put
    call_payoff = max(settlement_spot - short_call_K, 0.0) * (-1)  # Short call
    total_payoff = put_payoff + call_payoff

    # P&L = payoff - entry_cost
    pnl = total_payoff - entry_cost

    return pnl
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
    rate::Float64=0.045,
    div_yield::Float64=0.013,
    expiry_interval::Period=Day(1),
    delta_grid::Vector{Float64}=DELTA_GRID
)::Union{Tuple{Float64,Float64,Float64},Nothing}
    best_pnl = -Inf
    best_put_delta = 0.15
    best_call_delta = 0.15

    n_valid = 0

    for put_delta in delta_grid
        for call_delta in delta_grid
            pnl = simulate_strangle_pnl(
                surface, settlement_spot, put_delta, call_delta;
                rate=rate, div_yield=div_yield, expiry_interval=expiry_interval
            )
            pnl === nothing && continue
            n_valid += 1

            if pnl > best_pnl
                best_pnl = pnl
                best_put_delta = put_delta
                best_call_delta = call_delta
            end
        end
    end

    # Need at least some valid combinations
    n_valid < 10 && return nothing

    return (best_put_delta, best_call_delta, best_pnl)
end

"""
    generate_training_data(surfaces, settlement_spots, spot_history_dict; rate, div_yield, expiry_interval, verbose)
        -> TrainingDataset

Generate training data from surfaces by grid searching for optimal deltas.

# Arguments
- `surfaces::Dict{DateTime,VolatilitySurface}`: Entry surfaces
- `settlement_spots::Dict{DateTime,Float64}`: Settlement prices by expiry time
- `spot_history_dict::Dict{DateTime,SpotHistory}`: Historical spots for each entry time
- `rate::Float64`: Risk-free rate
- `div_yield::Float64`: Dividend yield
- `expiry_interval::Period`: Time to expiry
- `verbose::Bool`: Print progress

# Returns
- `TrainingDataset` with features, labels, and metadata
"""
function generate_training_data(
    surfaces::Dict{DateTime,VolatilitySurface},
    settlement_spots::Dict{DateTime,Float64},
    spot_history_dict::Dict{DateTime,SpotHistory};
    rate::Float64=0.045,
    div_yield::Float64=0.013,
    expiry_interval::Period=Day(1),
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

        # Compute expiry time
        expiry_target = ts + expiry_interval
        expiries = unique(rec.expiry for rec in surface.records)
        isempty(expiries) && continue

        taus = [time_to_expiry(e, ts) for e in expiries]
        tau_target = time_to_expiry(expiry_target, ts)
        idx = argmin(abs.(taus .- tau_target))
        expiry = expiries[idx]
        tau = taus[idx]

        # Get settlement spot
        settlement = get(settlement_spots, expiry, nothing)
        if settlement === nothing
            n_skipped += 1
            continue
        end

        # Get spot history for this timestamp
        spot_history = get(spot_history_dict, ts, nothing)

        # Extract features
        feats = extract_features(surface, tau; spot_history=spot_history)
        if feats === nothing
            n_skipped += 1
            continue
        end

        # Find optimal deltas
        result = find_optimal_deltas(
            surface, settlement;
            rate=rate, div_yield=div_yield, expiry_interval=expiry_interval
        )
        if result === nothing
            n_skipped += 1
            continue
        end

        best_put_delta, best_call_delta, best_pnl = result

        # Store data
        push!(features_list, features_to_vector(feats))
        push!(raw_deltas_list, Float32[best_put_delta, best_call_delta])
        # Scale deltas to [0, 1] for model training
        scaled = unscale_deltas(Float32[best_put_delta, best_call_delta])
        push!(labels_list, scaled)
        push!(pnls, Float32(best_pnl))
        push!(timestamps, ts)

        n_processed += 1

        if verbose && (i % 20 == 0 || i == n_total)
            println("  Processed $i / $n_total (valid: $n_processed, skipped: $n_skipped)")
        end
    end

    if isempty(features_list)
        error("No valid training samples generated")
    end

    # Stack into matrices
    X = reduce(hcat, features_list)
    Y = reduce(hcat, labels_list)
    raw_Y = reduce(hcat, raw_deltas_list)

    return TrainingDataset(X, Y, raw_Y, pnls, timestamps)
end

"""
    train_model!(model, train_data; val_data, epochs, batch_size, learning_rate, patience, verbose)
        -> (model, history)

Train the neural network model.

# Arguments
- `model`: Flux model to train
- `train_data::TrainingDataset`: Training data
- `val_data::Union{Nothing,TrainingDataset}`: Validation data (optional)
- `epochs::Int`: Maximum training epochs
- `batch_size::Int`: Batch size
- `learning_rate::Float64`: Learning rate for Adam optimizer
- `patience::Int`: Early stopping patience
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
    verbose::Bool=true
)
    opt_state = Flux.setup(Adam(learning_rate), model)

    n_samples = size(train_data.features, 2)
    best_val_loss = Inf
    patience_counter = 0
    best_model_state = nothing

    history = Dict{String,Vector{Float64}}(
        "train_loss" => Float64[],
        "val_loss" => Float64[]
    )

    for epoch in 1:epochs
        # Shuffle data
        perm = randperm(n_samples)

        epoch_loss = 0.0
        n_batches = 0

        # Training mode
        Flux.trainmode!(model)

        for batch_start in 1:batch_size:n_samples
            batch_end = min(batch_start + batch_size - 1, n_samples)
            batch_idx = perm[batch_start:batch_end]

            x_batch = train_data.features[:, batch_idx]
            y_batch = train_data.labels[:, batch_idx]

            # Compute gradients and update
            loss, grads = Flux.withgradient(model) do m
                delta_loss(m, x_batch, y_batch)
            end

            Flux.update!(opt_state, model, grads[1])

            epoch_loss += loss
            n_batches += 1
        end

        avg_train_loss = epoch_loss / n_batches
        push!(history["train_loss"], avg_train_loss)

        # Validation
        val_loss = if val_data !== nothing
            Flux.testmode!(model)
            mse(model(val_data.features), val_data.labels)
        else
            avg_train_loss
        end
        push!(history["val_loss"], Float64(val_loss))

        # Early stopping
        if val_loss < best_val_loss
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = Flux.state(model)
        else
            patience_counter += 1
        end

        if verbose && (epoch % 10 == 0 || epoch == 1 || patience_counter >= patience)
            val_str = val_data !== nothing ? ", val=$(round(val_loss, digits=6))" : ""
            println("  Epoch $epoch: train=$(round(avg_train_loss, digits=6))$val_str")
        end

        if patience_counter >= patience
            verbose && println("  Early stopping at epoch $epoch")
            break
        end
    end

    # Restore best model if we have validation
    if best_model_state !== nothing && val_data !== nothing
        Flux.loadmodel!(model, best_model_state)
    end

    return model, history
end

"""
    evaluate_model(model, data; min_delta, max_delta) -> Dict

Evaluate model on a dataset.

# Returns
Dict with:
- `mse`: Mean squared error on delta predictions
- `mae`: Mean absolute error
- `pred_deltas`: Predicted deltas
- `true_deltas`: True deltas
"""
function evaluate_model(
    model,
    data::TrainingDataset;
    min_delta::Float32=0.05f0,
    max_delta::Float32=0.35f0
)::Dict
    Flux.testmode!(model)

    pred_scaled = model(data.features)
    pred_deltas = scale_deltas(pred_scaled; min_delta=min_delta, max_delta=max_delta)

    true_deltas = data.raw_deltas

    # MSE on actual deltas
    mse_val = mean((pred_deltas .- true_deltas).^2)
    mae_val = mean(abs.(pred_deltas .- true_deltas))

    return Dict(
        "mse" => mse_val,
        "mae" => mae_val,
        "pred_deltas" => pred_deltas,
        "true_deltas" => true_deltas
    )
end
