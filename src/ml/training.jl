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
    simulate_condor_pnl(surface, settlement_spot, short_put_delta, short_call_delta; wing_delta_abs, min_delta_gap, rate, div_yield, expiry_interval)
        -> Union{Float64, Nothing}

Simulate P&L for a short iron condor with fixed wings (long deltas).

# Arguments
- `surface::VolatilitySurface`: Entry surface
- `settlement_spot::Float64`: Settlement spot price
- `short_put_delta::Float64`: Absolute delta for short put
- `short_call_delta::Float64`: Absolute delta for short call
- `wing_delta_abs::Float64`: Absolute delta for long wings (fixed)
- `min_delta_gap::Float64`: Minimum delta gap between short and long legs
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
    wing_delta_abs::Float64=0.05,
    min_delta_gap::Float64=0.08,
    rate::Float64=0.045,
    div_yield::Float64=0.013,
    expiry_interval::Period=Day(1)
)::Union{Float64,Nothing}
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

    put_strikes = sort(unique(r.strike for r in put_recs))
    call_strikes = sort(unique(r.strike for r in call_recs))
    isempty(put_strikes) && return nothing
    isempty(call_strikes) && return nothing

    ctx = (
        surface=surface,
        expiry=expiry,
        tau=tau,
        recs=recs,
        put_strikes=put_strikes,
        call_strikes=call_strikes
    )

    strikes = _delta_condor_strikes(
        ctx,
        short_put_delta,
        short_call_delta,
        wing_delta_abs,
        wing_delta_abs;
        rate=rate,
        div_yield=div_yield,
        min_delta_gap=min_delta_gap
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
    find_optimal_condor_deltas(surface, settlement_spot; wing_delta_abs, min_delta_gap, rate, div_yield, expiry_interval, delta_grid)
        -> Union{Tuple{Float64, Float64, Float64}, Nothing}

Find optimal inner deltas for an iron condor with fixed wings via grid search.

# Returns
- Tuple of (best_put_delta, best_call_delta, best_pnl) or nothing
"""
function find_optimal_condor_deltas(
    surface::VolatilitySurface,
    settlement_spot::Float64;
    wing_delta_abs::Float64=0.05,
    min_delta_gap::Float64=0.08,
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
            pnl = simulate_condor_pnl(
                surface, settlement_spot, put_delta, call_delta;
                wing_delta_abs=wing_delta_abs,
                min_delta_gap=min_delta_gap,
                rate=rate,
                div_yield=div_yield,
                expiry_interval=expiry_interval
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

    n_valid < 10 && return nothing
    return (best_put_delta, best_call_delta, best_pnl)
end

"""
    generate_condor_training_data(surfaces, settlement_spots, spot_history_dict;
        rate, div_yield, expiry_interval, wing_delta_abs, min_delta_gap, use_logsig, verbose)
        -> TrainingDataset

Generate training data for iron condor strike selection by grid searching
optimal inner deltas with fixed wings.
"""
function generate_condor_training_data(
    surfaces::Dict{DateTime,VolatilitySurface},
    settlement_spots::Dict{DateTime,Float64},
    spot_history_dict::Dict{DateTime,SpotHistory};
    rate::Float64=0.045,
    div_yield::Float64=0.013,
    expiry_interval::Period=Day(1),
    wing_delta_abs::Float64=0.05,
    min_delta_gap::Float64=0.08,
    use_logsig::Bool=false,
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
        feats = extract_features(surface, tau; spot_history=spot_history, use_logsig=use_logsig)
        if feats === nothing
            n_skipped += 1
            continue
        end

        result = find_optimal_condor_deltas(
            surface, settlement;
            wing_delta_abs=wing_delta_abs,
            min_delta_gap=min_delta_gap,
            rate=rate,
            div_yield=div_yield,
            expiry_interval=expiry_interval
        )
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
    rate::Float64=0.045,
    div_yield::Float64=0.013,
    expiry_interval::Period=Day(1),
    use_logsig::Bool=false,
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
        feats = extract_features(surface, tau; spot_history=spot_history, use_logsig=use_logsig)
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

    # Compute position size labels from P&L distribution
    size_labels = compute_size_labels(pnls)

    if verbose
        n_positive = count(p -> p > 0, pnls)
        avg_size = mean(size_labels)
        println("  Position sizing: $(n_positive)/$(length(pnls)) profitable, avg_size=$(round(avg_size, digits=3))")
    end

    return TrainingDataset(X, Y, raw_Y, pnls, size_labels, timestamps)
end

"""
    train_model!(model, train_data; val_data, epochs, batch_size, learning_rate, patience, delta_weight, size_weight, verbose)
        -> (model, history)

Train the neural network model with combined delta and position size loss.

# Arguments
- `model`: Flux model to train (should have 3 outputs: put_delta, call_delta, position_size)
- `train_data::TrainingDataset`: Training data
- `val_data::Union{Nothing,TrainingDataset}`: Validation data (optional)
- `epochs::Int`: Maximum training epochs
- `batch_size::Int`: Batch size
- `learning_rate::Float64`: Learning rate for Adam optimizer
- `patience::Int`: Early stopping patience
- `delta_weight::Float32`: Weight for delta prediction loss (default: 1.0)
- `size_weight::Float32`: Weight for position size prediction loss (default: 1.0)
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
    opt_state = Flux.setup(Adam(learning_rate), model)

    n_samples = size(train_data.features, 2)
    best_val_loss = Inf
    patience_counter = 0
    best_model_state = nothing

    history = Dict{String,Vector{Float64}}(
        "train_loss" => Float64[],
        "val_loss" => Float64[],
        "train_delta_loss" => Float64[],
        "train_size_loss" => Float64[]
    )

    for epoch in 1:epochs
        # Shuffle data
        perm = randperm(n_samples)

        epoch_loss = 0.0
        epoch_delta_loss = 0.0
        epoch_size_loss = 0.0
        n_batches = 0

        # Training mode
        Flux.trainmode!(model)

        for batch_start in 1:batch_size:n_samples
            batch_end = min(batch_start + batch_size - 1, n_samples)
            batch_idx = perm[batch_start:batch_end]

            x_batch = train_data.features[:, batch_idx]
            y_deltas_batch = train_data.labels[:, batch_idx]
            y_sizes_batch = train_data.size_labels[batch_idx]

            # Compute gradients and update
            loss, grads = Flux.withgradient(model) do m
                combined_loss(m, x_batch, y_deltas_batch, y_sizes_batch;
                    delta_weight=delta_weight, size_weight=size_weight)
            end

            Flux.update!(opt_state, model, grads[1])

            # Track individual losses for monitoring
            preds = model(x_batch)
            d_loss = mse(preds[1:2, :], y_deltas_batch)
            s_loss = mse(preds[3, :], y_sizes_batch)

            epoch_loss += loss
            epoch_delta_loss += d_loss
            epoch_size_loss += s_loss
            n_batches += 1
        end

        avg_train_loss = epoch_loss / n_batches
        avg_delta_loss = epoch_delta_loss / n_batches
        avg_size_loss = epoch_size_loss / n_batches
        push!(history["train_loss"], avg_train_loss)
        push!(history["train_delta_loss"], Float64(avg_delta_loss))
        push!(history["train_size_loss"], Float64(avg_size_loss))

        # Validation
        val_loss = if val_data !== nothing
            Flux.testmode!(model)
            combined_loss(model, val_data.features, val_data.labels, val_data.size_labels;
                delta_weight=delta_weight, size_weight=size_weight)
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
            println("  Epoch $epoch: train=$(round(avg_train_loss, digits=6))$val_str (δ=$(round(avg_delta_loss, digits=4)), sz=$(round(avg_size_loss, digits=4)))")
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
- `delta_mse`: Mean squared error on delta predictions
- `delta_mae`: Mean absolute error on deltas
- `size_mse`: Mean squared error on position size predictions
- `size_mae`: Mean absolute error on position sizes
- `pred_deltas`: Predicted deltas (2 × n_samples)
- `true_deltas`: True deltas (2 × n_samples)
- `pred_sizes`: Predicted position sizes (n_samples,)
- `true_sizes`: True position sizes (n_samples,)
"""
function evaluate_model(
    model,
    data::TrainingDataset;
    min_delta::Float32=0.05f0,
    max_delta::Float32=0.35f0
)::Dict
    Flux.testmode!(model)

    raw_output = model(data.features)

    # Delta predictions (first 2 outputs)
    pred_deltas = scale_deltas(raw_output[1:2, :]; min_delta=min_delta, max_delta=max_delta)
    true_deltas = data.raw_deltas

    delta_mse = mean((pred_deltas .- true_deltas).^2)
    delta_mae = mean(abs.(pred_deltas .- true_deltas))

    # Position size predictions (3rd output)
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
end
