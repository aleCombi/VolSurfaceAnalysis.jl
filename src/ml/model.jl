# ML Model Definition for Strike Selection
# Uses Flux.jl for neural network implementation

using Flux: Chain, Dense, relu, sigmoid, Dropout
using Flux.Losses: mse

"""
    create_strike_model(; input_dim, hidden_dims, output_dim, dropout_rate) -> Chain

Create a neural network for strike selection.

The model outputs values in [0, 1] via sigmoid, which are then scaled to valid delta range.

# Arguments
- `input_dim::Int`: Number of input features (default: 15)
- `hidden_dims::Vector{Int}`: Hidden layer dimensions (default: [64, 32, 16])
- `output_dim::Int`: Number of outputs (default: 2 for put_delta, call_delta)
- `dropout_rate::Float64`: Dropout probability (default: 0.2)

# Returns
- `Chain`: Flux neural network model
"""
function create_strike_model(;
    input_dim::Int=N_FEATURES,
    hidden_dims::Vector{Int}=[64, 32, 16],
    output_dim::Int=2,
    dropout_rate::Float64=0.2
)::Chain
    layers = []

    # Input layer
    push!(layers, Dense(input_dim => hidden_dims[1], relu))
    if dropout_rate > 0
        push!(layers, Dropout(dropout_rate))
    end

    # Hidden layers
    for i in 1:(length(hidden_dims)-1)
        push!(layers, Dense(hidden_dims[i] => hidden_dims[i+1], relu))
        if dropout_rate > 0
            push!(layers, Dropout(dropout_rate))
        end
    end

    # Output layer with sigmoid to bound output in [0, 1]
    push!(layers, Dense(hidden_dims[end] => output_dim, sigmoid))

    return Chain(layers...)
end

"""
    scale_deltas(raw_output; min_delta, max_delta) -> Vector{Float32}

Scale sigmoid output [0, 1] to valid delta range [min_delta, max_delta].

# Arguments
- `raw_output`: Model output (sigmoid values in [0, 1])
- `min_delta::Float32`: Minimum delta value (default: 0.05)
- `max_delta::Float32`: Maximum delta value (default: 0.35)
"""
function scale_deltas(
    raw_output;
    min_delta::Float32=0.05f0,
    max_delta::Float32=0.35f0
)
    return min_delta .+ raw_output .* (max_delta - min_delta)
end

"""
    unscale_deltas(deltas; min_delta, max_delta) -> Vector{Float32}

Convert delta values to [0, 1] range for model targets.

# Arguments
- `deltas`: Delta values in [min_delta, max_delta]
- `min_delta::Float32`: Minimum delta value (default: 0.05)
- `max_delta::Float32`: Maximum delta value (default: 0.35)
"""
function unscale_deltas(
    deltas;
    min_delta::Float32=0.05f0,
    max_delta::Float32=0.35f0
)
    return (deltas .- min_delta) ./ (max_delta - min_delta)
end

"""
    delta_loss(model, x, y) -> Float32

Compute MSE loss between model predictions and target deltas.

# Arguments
- `model`: Neural network model
- `x`: Input features (n_features × n_samples)
- `y`: Target deltas in [0, 1] (2 × n_samples)
"""
function delta_loss(model, x, y)
    predictions = model(x)
    return mse(predictions, y)
end

"""
    predict_deltas(model, x; min_delta, max_delta) -> Matrix{Float32}

Get delta predictions from model.

# Arguments
- `model`: Trained neural network
- `x`: Input features (n_features × n_samples)
- `min_delta`, `max_delta`: Delta scaling bounds

# Returns
- Matrix of predicted deltas (2 × n_samples)
"""
function predict_deltas(
    model,
    x;
    min_delta::Float32=0.05f0,
    max_delta::Float32=0.35f0
)::Matrix{Float32}
    raw = model(x)
    return scale_deltas(raw; min_delta=min_delta, max_delta=max_delta)
end
