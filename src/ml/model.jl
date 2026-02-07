# ML Model Definition for Strike Selection
# Uses Flux.jl for neural network implementation

using Flux: Chain, Dense, relu, sigmoid, Dropout
using Flux.Losses: mse

"""
    mixed_output_activation(x) -> Array

Custom activation for strike selection model:
- First 2 outputs: sigmoid → [0, 1] for deltas
- 3rd output (if present): tanh → [-1, 1] for position size

Position size interpretation:
- +1: Full size short vol (sell strangle)
- 0: No trade
- -1: Full size long vol (buy strangle)
"""
function mixed_output_activation(x)
    n_outputs = size(x, 1)
    if n_outputs <= 2
        # Legacy 2-output model: all sigmoid
        return sigmoid.(x)
    else
        # 3+ output model: sigmoid for deltas, tanh for size
        deltas = sigmoid.(x[1:2, :])
        sizing = tanh.(x[3:end, :])
        return vcat(deltas, sizing)
    end
end

"""
    create_strike_model(; input_dim, hidden_dims, output_dim, dropout_rate) -> Chain

Create a neural network for strike selection and position sizing.

Output activation depends on output_dim:
- Outputs 1-2: sigmoid [0, 1], scaled to delta range [min_delta, max_delta]
- Output 3 (if output_dim=3): tanh [-1, 1] for position size
  - Positive: short vol (sell strangle)
  - Negative: long vol (buy strangle)

# Arguments
- `input_dim::Int`: Number of input features (default: N_FEATURES)
- `hidden_dims::Vector{Int}`: Hidden layer dimensions (default: [64, 32, 16])
- `output_dim::Int`: Number of outputs (default: 3 for put_delta, call_delta, position_size)
- `dropout_rate::Float64`: Dropout probability (default: 0.2)

# Returns
- `Chain`: Flux neural network model
"""
function create_strike_model(;
    input_dim::Int=N_FEATURES,
    hidden_dims::Vector{Int}=[64, 32, 16],
    output_dim::Int=3,
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

    # Output layer with mixed activation (sigmoid for deltas, tanh for size)
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

    push!(layers, Dense(hidden_dims[end] => 1, identity))
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

Get delta predictions from model (first 2 outputs only).

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
    # Only take first 2 rows (deltas), ignore position size if present
    delta_raw = raw[1:2, :]
    return scale_deltas(delta_raw; min_delta=min_delta, max_delta=max_delta)
end

"""
    predict_with_sizing(model, x; min_delta, max_delta) -> Tuple{Matrix{Float32}, Vector{Float32}}

Get delta predictions and position sizes from model.

# Arguments
- `model`: Trained neural network (must have 3 outputs)
- `x`: Input features (n_features × n_samples)
- `min_delta`, `max_delta`: Delta scaling bounds

# Returns
- Tuple of (deltas matrix (2 × n_samples), position_sizes vector (n_samples,))
  Position sizes are in [-1, 1] where:
  - Positive: short vol (sell strangle)
  - Negative: long vol (buy strangle)
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

    # Position size is in [-1, 1] from tanh
    position_sizes = vec(raw[3, :])

    return deltas, position_sizes
end

"""
    combined_loss(model, x, y_deltas, y_sizes; delta_weight, size_weight) -> Float32

Compute combined MSE loss for deltas and position sizes.

# Arguments
- `model`: Neural network model
- `x`: Input features (n_features × n_samples)
- `y_deltas`: Target deltas in [0, 1] (2 × n_samples)
- `y_sizes`: Target position sizes in [-1, 1] (n_samples,)
  - Positive: short vol (sell), Negative: long vol (buy)
- `delta_weight::Float32`: Weight for delta loss (default: 1.0)
- `size_weight::Float32`: Weight for position size loss (default: 1.0)
"""
function combined_loss(
    model, x, y_deltas, y_sizes;
    delta_weight::Float32=1.0f0,
    size_weight::Float32=1.0f0
)
    predictions = model(x)

    # Delta loss (first 2 outputs)
    pred_deltas = predictions[1:2, :]
    delta_loss_val = mse(pred_deltas, y_deltas)

    # Position size loss (3rd output)
    pred_sizes = predictions[3, :]
    size_loss_val = mse(pred_sizes, y_sizes)

    return delta_weight * delta_loss_val + size_weight * size_loss_val
end
