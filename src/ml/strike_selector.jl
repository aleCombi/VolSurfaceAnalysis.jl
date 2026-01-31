# ML Strike Selector Wrapper
# Callable struct that integrates trained model with backtest system

"""
    MLStrikeSelector

A callable struct that uses a trained neural network to select strikes.
Can be used as the `strike_selector` parameter in ShortStrangleStrategy.

# Fields
- `model`: Trained Flux neural network
- `feature_means::Vector{Float32}`: Feature means for normalization
- `feature_stds::Vector{Float32}`: Feature stds for normalization
- `min_delta::Float32`: Minimum delta bound
- `max_delta::Float32`: Maximum delta bound
- `rate::Float64`: Risk-free rate for delta calculation
- `div_yield::Float64`: Dividend yield for delta calculation
- `spot_history::Dict{DateTime,SpotHistory}`: Spot history for features
"""
struct MLStrikeSelector
    model::Any  # Flux Chain
    feature_means::Vector{Float32}
    feature_stds::Vector{Float32}
    min_delta::Float32
    max_delta::Float32
    rate::Float64
    div_yield::Float64
    spot_history::Dict{DateTime,SpotHistory}
end

"""
    MLStrikeSelector(model, feature_means, feature_stds; min_delta, max_delta, rate, div_yield)

Create an MLStrikeSelector with default parameters.
"""
function MLStrikeSelector(
    model,
    feature_means::Vector{Float32},
    feature_stds::Vector{Float32};
    min_delta::Float32=0.05f0,
    max_delta::Float32=0.35f0,
    rate::Float64=0.045,
    div_yield::Float64=0.013,
    spot_history::Dict{DateTime,SpotHistory}=Dict{DateTime,SpotHistory}()
)
    return MLStrikeSelector(
        model, feature_means, feature_stds,
        min_delta, max_delta, rate, div_yield, spot_history
    )
end

"""
    (selector::MLStrikeSelector)(ctx) -> Union{Nothing, Tuple{Float64, Float64}}

Make MLStrikeSelector callable for use as strike_selector in strategies.

# Arguments
- `ctx`: Strike selection context with surface, tau, recs, etc.

# Returns
- Tuple of (short_put_K, short_call_K) or nothing
"""
function (selector::MLStrikeSelector)(ctx)::Union{Nothing,Tuple{Float64,Float64}}
    surface = ctx.surface
    tau = ctx.tau

    # Get spot history for this timestamp if available
    spot_history = get(selector.spot_history, surface.timestamp, nothing)

    # Extract features
    feats = extract_features(surface, tau; spot_history=spot_history)
    feats === nothing && return nothing

    # Convert to vector and normalize
    x = features_to_vector(feats)
    x_norm = (x .- selector.feature_means) ./ selector.feature_stds

    # Run model inference (need to reshape for Flux)
    Flux.testmode!(selector.model)
    x_input = reshape(x_norm, :, 1)
    raw_output = selector.model(x_input)

    # Scale to delta range
    deltas = scale_deltas(raw_output; min_delta=selector.min_delta, max_delta=selector.max_delta)
    put_delta = Float64(deltas[1])
    call_delta = Float64(deltas[2])

    # Convert deltas to strikes using asymmetric helper
    return _delta_strangle_strikes_asymmetric(
        ctx, put_delta, call_delta;
        rate=selector.rate, div_yield=selector.div_yield
    )
end

"""
    save_ml_selector(path, model, feature_means, feature_stds; min_delta, max_delta)

Save a trained ML selector to a BSON file.
"""
function save_ml_selector(
    path::String,
    model,
    feature_means::Vector{Float32},
    feature_stds::Vector{Float32};
    min_delta::Float32=0.05f0,
    max_delta::Float32=0.35f0
)
    BSON.@save path model feature_means feature_stds min_delta max_delta
    return path
end

"""
    load_ml_selector(path; rate, div_yield, spot_history) -> MLStrikeSelector

Load a trained ML selector from a BSON file.

# Arguments
- `path::String`: Path to BSON file
- `rate::Float64`: Risk-free rate for strike conversion
- `div_yield::Float64`: Dividend yield for strike conversion
- `spot_history::Dict{DateTime,SpotHistory}`: Spot history for feature extraction
"""
function load_ml_selector(
    path::String;
    rate::Float64=0.045,
    div_yield::Float64=0.013,
    spot_history::Dict{DateTime,SpotHistory}=Dict{DateTime,SpotHistory}()
)::MLStrikeSelector
    data = BSON.load(path)
    return MLStrikeSelector(
        data[:model],
        data[:feature_means],
        data[:feature_stds];
        min_delta=get(data, :min_delta, 0.05f0),
        max_delta=get(data, :max_delta, 0.35f0),
        rate=rate,
        div_yield=div_yield,
        spot_history=spot_history
    )
end
