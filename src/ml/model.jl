# ML Model — Flux-based scoring MLP
#
# Creates a small neural network that predicts utility (e.g., ROI) for a
# candidate condor given surface + candidate features.

"""
    create_scoring_model(; input_dim, hidden_dims=[64, 32])

Create a Flux MLP for scoring condor candidates. Output is a scalar
(predicted utility, no output activation).
"""
function create_scoring_model(; input_dim::Int, hidden_dims::Vector{Int}=[64, 32])
    layers = []
    prev_dim = input_dim
    for h in hidden_dims
        push!(layers, Dense(prev_dim => h, relu))
        prev_dim = h
    end
    push!(layers, Dense(prev_dim => 1))
    return Chain(layers...)
end

"""
    score_candidates(model, surface_feats, candidate_feats_matrix; feature_means, feature_stds)

Score a set of candidate condors.

# Arguments
- `model`: Flux model (Chain)
- `surface_feats::Vector{Float32}`: surface features (shared across candidates)
- `candidate_feats_matrix::Matrix{Float32}`: candidate features, one column per candidate
- `feature_means::Vector{Float32}`: normalization means
- `feature_stds::Vector{Float32}`: normalization stds

# Returns
- `Vector{Float32}`: one score per candidate
"""
function score_candidates(
    model,
    surface_feats::Vector{Float32},
    candidate_feats_matrix::Matrix{Float32};
    feature_means::Vector{Float32},
    feature_stds::Vector{Float32}
)
    n_candidates = size(candidate_feats_matrix, 2)
    # Broadcast surface features across all candidates
    sf_matrix = repeat(surface_feats, 1, n_candidates)
    features = vcat(sf_matrix, candidate_feats_matrix)

    # Normalize
    safe_stds = max.(feature_stds, Float32(1e-8))
    normalized = (features .- feature_means) ./ safe_stds

    # Forward pass
    scores = model(normalized)
    return vec(scores)
end
