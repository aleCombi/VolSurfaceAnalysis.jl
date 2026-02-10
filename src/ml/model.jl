# ML Model Definition for Strike Selection
# Uses Flux.jl for neural network implementation

using Flux: Chain, Dense, relu, sigmoid, Dropout
using Flux.Losses: mse

# Delta bounds for 4-output condor models (per-output scaling)
const SHORT_MIN_DELTA = 0.05f0
const SHORT_MAX_DELTA = 0.35f0
const LONG_MIN_DELTA = 0.01f0
const LONG_MAX_DELTA = 0.15f0

"""
    mixed_output_activation(x) -> Array

Custom activation for strike selection model:
- 2-output or 4-output: all sigmoid → [0, 1] for deltas
- 3-output (legacy): sigmoid for deltas, tanh for position size
"""
function mixed_output_activation(x)
    n_outputs = size(x, 1)
    if n_outputs == 3
        # Legacy 3-output model: sigmoid for deltas, tanh for size
        deltas = sigmoid.(x[1:2, :])
        sizing = tanh.(x[3:end, :])
        return vcat(deltas, sizing)
    else
        # 2-output (strangle) or 4-output (condor): all sigmoid
        return sigmoid.(x)
    end
end

"""
    create_strike_model(; input_dim, hidden_dims, output_dim, dropout_rate) -> Chain

Create a neural network for strike selection.

Output activation depends on output_dim:
- 2 outputs: sigmoid for (short_put_delta, short_call_delta)
- 3 outputs (legacy): sigmoid for deltas, tanh for position_size
- 4 outputs: sigmoid for (short_put_delta, short_call_delta, long_put_delta, long_call_delta)

# Arguments
- `input_dim::Int`: Number of input features (default: N_FEATURES)
- `hidden_dims::Vector{Int}`: Hidden layer dimensions (default: [64, 32, 16])
- `output_dim::Int`: Number of outputs (default: 4 for condor)
- `dropout_rate::Float64`: Dropout probability (default: 0.2)
"""
function _build_mlp_layers(input_dim::Int, hidden_dims::Vector{Int}, dropout_rate::Float64)::Vector
    layers = []
    push!(layers, Dense(input_dim => hidden_dims[1], relu))
    if dropout_rate > 0
        push!(layers, Dropout(dropout_rate))
    end
    for i in 1:(length(hidden_dims)-1)
        push!(layers, Dense(hidden_dims[i] => hidden_dims[i+1], relu))
        if dropout_rate > 0
            push!(layers, Dropout(dropout_rate))
        end
    end
    return layers
end

function create_strike_model(;
    input_dim::Int=N_FEATURES,
    hidden_dims::Vector{Int}=[64, 32, 16],
    output_dim::Int=4,
    dropout_rate::Float64=0.2
)::Chain
    layers = _build_mlp_layers(input_dim, hidden_dims, dropout_rate)
    push!(layers, Dense(hidden_dims[end] => output_dim, identity))
    push!(layers, mixed_output_activation)
    return Chain(layers...)
end

"""
    create_scoring_model(; input_dim, hidden_dims, dropout_rate) -> Chain

Create a scalar-regression neural network for candidate scoring.
The model outputs one value per candidate (predicted utility).
"""
function create_scoring_model(;
    input_dim::Int=N_FEATURES,
    hidden_dims::Vector{Int}=[128, 64, 32],
    dropout_rate::Float64=0.2
)::Chain
    layers = _build_mlp_layers(input_dim, hidden_dims, dropout_rate)
    push!(layers, Dense(hidden_dims[end] => 1, identity))
    return Chain(layers...)
end

"""
    scale_deltas(raw_output; min_delta, max_delta)

Scale sigmoid output [0, 1] to valid delta range [min_delta, max_delta].
"""
function scale_deltas(
    raw_output;
    min_delta::Float32=0.05f0,
    max_delta::Float32=0.35f0
)
    return min_delta .+ raw_output .* (max_delta - min_delta)
end

"""
    unscale_deltas(deltas; min_delta, max_delta)

Convert delta values to [0, 1] range for model targets.
"""
function unscale_deltas(
    deltas;
    min_delta::Float32=0.05f0,
    max_delta::Float32=0.35f0
)
    return (deltas .- min_delta) ./ (max_delta - min_delta)
end

"""
    scale_deltas_4d(raw_output; short_min_delta, short_max_delta, long_min_delta, long_max_delta)

Scale 4-output sigmoid [0,1] to delta ranges with per-output bounds.
Outputs 1-2 (short legs): [short_min_delta, short_max_delta]
Outputs 3-4 (long legs): [long_min_delta, long_max_delta]
"""
function scale_deltas_4d(
    raw_output;
    short_min_delta::Float32=SHORT_MIN_DELTA,
    short_max_delta::Float32=SHORT_MAX_DELTA,
    long_min_delta::Float32=LONG_MIN_DELTA,
    long_max_delta::Float32=LONG_MAX_DELTA
)
    short_scaled = short_min_delta .+ raw_output[1:2, :] .* (short_max_delta - short_min_delta)
    long_scaled = long_min_delta .+ raw_output[3:4, :] .* (long_max_delta - long_min_delta)
    return vcat(short_scaled, long_scaled)
end

"""
    unscale_deltas_4d(deltas; short_min_delta, short_max_delta, long_min_delta, long_max_delta)

Convert 4-delta values to [0, 1] range for model targets.
"""
function unscale_deltas_4d(
    deltas;
    short_min_delta::Float32=SHORT_MIN_DELTA,
    short_max_delta::Float32=SHORT_MAX_DELTA,
    long_min_delta::Float32=LONG_MIN_DELTA,
    long_max_delta::Float32=LONG_MAX_DELTA
)
    short_unscaled = (deltas[1:2, :] .- short_min_delta) ./ (short_max_delta - short_min_delta)
    long_unscaled = (deltas[3:4, :] .- long_min_delta) ./ (long_max_delta - long_min_delta)
    return vcat(short_unscaled, long_unscaled)
end

"""
    delta_loss(model, x, y) -> Float32

Compute MSE loss between model predictions and targets. Works for any output dimension.
"""
function delta_loss(model, x, y)
    predictions = model(x)
    return mse(predictions, y)
end

"""
    predict_deltas(model, x; min_delta, max_delta) -> Matrix{Float32}

Get delta predictions from model (first 2 outputs only).
"""
function predict_deltas(
    model,
    x;
    min_delta::Float32=0.05f0,
    max_delta::Float32=0.35f0
)::Matrix{Float32}
    raw = model(x)
    delta_raw = raw[1:2, :]
    return scale_deltas(delta_raw; min_delta=min_delta, max_delta=max_delta)
end

"""
    predict_condor_deltas(model, x; short_min_delta, short_max_delta, long_min_delta, long_max_delta) -> Matrix{Float32}

Get 4-delta predictions from a 4-output condor model.
Returns (4 × n_samples) matrix: [short_put, short_call, long_put, long_call].
"""
function predict_condor_deltas(
    model,
    x;
    short_min_delta::Float32=SHORT_MIN_DELTA,
    short_max_delta::Float32=SHORT_MAX_DELTA,
    long_min_delta::Float32=LONG_MIN_DELTA,
    long_max_delta::Float32=LONG_MAX_DELTA
)::Matrix{Float32}
    raw = model(x)
    @assert size(raw, 1) >= 4 "Model must have at least 4 outputs for predict_condor_deltas"
    return scale_deltas_4d(raw[1:4, :];
        short_min_delta=short_min_delta, short_max_delta=short_max_delta,
        long_min_delta=long_min_delta, long_max_delta=long_max_delta
    )
end

"""
    predict_with_sizing(model, x; min_delta, max_delta)

Legacy: get delta predictions and position sizes from a 3-output model.
"""
function predict_with_sizing(
    model,
    x;
    min_delta::Float32=0.05f0,
    max_delta::Float32=0.35f0
)
    raw = model(x)
    @assert size(raw, 1) >= 3 "Model must have at least 3 outputs for predict_with_sizing"

    delta_raw = raw[1:2, :]
    deltas = scale_deltas(delta_raw; min_delta=min_delta, max_delta=max_delta)
    position_sizes = vec(raw[3, :])

    return deltas, position_sizes
end

"""
    combined_loss(model, x, y_deltas, y_sizes; delta_weight, size_weight)

Legacy: combined MSE loss for 3-output models (deltas + position sizes).
"""
function combined_loss(
    model, x, y_deltas, y_sizes;
    delta_weight::Float32=1.0f0,
    size_weight::Float32=1.0f0
)
    predictions = model(x)
    pred_deltas = predictions[1:2, :]
    delta_loss_val = mse(pred_deltas, y_deltas)
    pred_sizes = predictions[3, :]
    size_loss_val = mse(pred_sizes, y_sizes)
    return delta_weight * delta_loss_val + size_weight * size_loss_val
end
