# ML Strike Selector Wrapper
# Callable struct that integrates trained model with backtest system

"""
    MLStrikeSelector

A callable struct that uses a trained neural network to select strikes and position sizing.
Can be used as the `strike_selector` parameter in ShortStrangleStrategy.

When the model has 3 outputs (put_delta, call_delta, position_size), returns a 3-tuple.
When the model has 2 outputs (legacy), returns a 2-tuple.

Position size interpretation:
- Positive: short vol (sell strangle)
- Negative: long vol (buy strangle)
- Near zero: skip trade

# Fields
- `model`: Trained Flux neural network
- `feature_means::Vector{Float32}`: Feature means for normalization
- `feature_stds::Vector{Float32}`: Feature stds for normalization
- `min_delta::Float32`: Minimum delta bound
- `max_delta::Float32`: Maximum delta bound
- `rate::Float64`: Risk-free rate for delta calculation
- `div_yield::Float64`: Dividend yield for delta calculation
- `spot_history::Dict{DateTime,SpotHistory}`: Spot history for features
- `return_sizing::Bool`: If true and model has 3 outputs, return (put_K, call_K, size)
- `use_logsig::Bool`: Use log-signature features instead of signature
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
    return_sizing::Bool
    use_logsig::Bool
end

"""
    MLStrikeSelector(model, feature_means, feature_stds; min_delta, max_delta, rate, div_yield, return_sizing)

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
    spot_history::Dict{DateTime,SpotHistory}=Dict{DateTime,SpotHistory}(),
    return_sizing::Bool=true,
    use_logsig::Bool=false
)
    return MLStrikeSelector(
        model, feature_means, feature_stds,
        min_delta, max_delta, rate, div_yield, spot_history, return_sizing, use_logsig
    )
end

"""
    (selector::MLStrikeSelector)(ctx) -> Union{Nothing, Tuple{Float64, Float64}, Tuple{Float64, Float64, Float64}}

Make MLStrikeSelector callable for use as strike_selector in strategies.

# Arguments
- `ctx`: Strike selection context with surface, tau, recs, etc.

# Returns
- If return_sizing=false or model has 2 outputs: Tuple of (short_put_K, short_call_K) or nothing
- If return_sizing=true and model has 3 outputs: Tuple of (short_put_K, short_call_K, position_size) or nothing
"""
function (selector::MLStrikeSelector)(ctx)
    surface = ctx.surface
    tau = ctx.tau

    # Get spot history for this timestamp if available
    spot_history = get(selector.spot_history, surface.timestamp, nothing)

    # Extract features
    feats = extract_features(surface, tau; spot_history=spot_history, use_logsig=selector.use_logsig)
    feats === nothing && return nothing

    # Convert to vector and normalize
    x = features_to_vector(feats)
    x_norm = (x .- selector.feature_means) ./ selector.feature_stds

    # Run model inference (need to reshape for Flux)
    Flux.testmode!(selector.model)
    x_input = reshape(x_norm, :, 1)
    raw_output = selector.model(x_input)

    # Check if model has position sizing output
    has_sizing = size(raw_output, 1) >= 3

    # Scale deltas (first 2 outputs)
    delta_raw = raw_output[1:2, :]
    deltas = scale_deltas(delta_raw; min_delta=selector.min_delta, max_delta=selector.max_delta)
    put_delta = Float64(deltas[1])
    call_delta = Float64(deltas[2])

    # Convert deltas to strikes using asymmetric helper
    strikes = _delta_strangle_strikes_asymmetric(
        ctx, put_delta, call_delta;
        rate=selector.rate, div_yield=selector.div_yield
    )

    strikes === nothing && return nothing

    # Return with or without sizing
    if selector.return_sizing && has_sizing
        position_size = Float64(raw_output[3, 1])  # In [-1, 1] from tanh
        return (strikes[1], strikes[2], position_size)
    else
        return strikes
    end
end

"""
    MLCondorStrikeSelector

Callable selector for iron condors that predicts inner deltas and uses fixed wings.

Fields are similar to MLStrikeSelector with additional:
- `wing_delta_abs::Float32`: Absolute delta for long wings (fixed)
- `min_delta_gap::Float32`: Minimum delta gap between short and long legs
"""
struct MLCondorStrikeSelector
    model::Any
    feature_means::Vector{Float32}
    feature_stds::Vector{Float32}
    min_delta::Float32
    max_delta::Float32
    wing_delta_abs::Float32
    min_delta_gap::Float32
    rate::Float64
    div_yield::Float64
    spot_history::Dict{DateTime,SpotHistory}
    return_sizing::Bool
    use_logsig::Bool
end

"""
    MLCondorStrikeSelector(model, feature_means, feature_stds; wing_delta_abs, min_delta_gap, min_delta, max_delta, rate, div_yield, return_sizing)
"""
function MLCondorStrikeSelector(
    model,
    feature_means::Vector{Float32},
    feature_stds::Vector{Float32};
    wing_delta_abs::Float32=0.05f0,
    min_delta_gap::Float32=0.08f0,
    min_delta::Float32=0.05f0,
    max_delta::Float32=0.35f0,
    rate::Float64=0.045,
    div_yield::Float64=0.013,
    spot_history::Dict{DateTime,SpotHistory}=Dict{DateTime,SpotHistory}(),
    return_sizing::Bool=true,
    use_logsig::Bool=false
)
    return MLCondorStrikeSelector(
        model,
        feature_means,
        feature_stds,
        min_delta,
        max_delta,
        wing_delta_abs,
        min_delta_gap,
        rate,
        div_yield,
        spot_history,
        return_sizing,
        use_logsig
    )
end

"""
    (selector::MLCondorStrikeSelector)(ctx) -> Union{Nothing, Tuple{Float64, Float64, Float64, Float64},
                                                    Tuple{Float64, Float64, Float64, Float64, Float64}}

Returns inner/outer strikes, and optionally a position size.
"""
function (selector::MLCondorStrikeSelector)(ctx)
    surface = ctx.surface
    tau = ctx.tau

    spot_history = get(selector.spot_history, surface.timestamp, nothing)
    feats = extract_features(surface, tau; spot_history=spot_history, use_logsig=selector.use_logsig)
    feats === nothing && return nothing

    x = features_to_vector(feats)
    x_norm = (x .- selector.feature_means) ./ selector.feature_stds

    Flux.testmode!(selector.model)
    x_input = reshape(x_norm, :, 1)
    raw_output = selector.model(x_input)

    has_sizing = size(raw_output, 1) >= 3

    delta_raw = raw_output[1:2, :]
    deltas = scale_deltas(delta_raw; min_delta=selector.min_delta, max_delta=selector.max_delta)
    put_delta = Float64(deltas[1])
    call_delta = Float64(deltas[2])

    strikes = _delta_condor_strikes(
        ctx,
        put_delta,
        call_delta,
        Float64(selector.wing_delta_abs),
        Float64(selector.wing_delta_abs);
        rate=selector.rate,
        div_yield=selector.div_yield,
        min_delta_gap=Float64(selector.min_delta_gap)
    )
    strikes === nothing && return nothing

    if selector.return_sizing && has_sizing
        position_size = Float64(raw_output[3, 1])
        return (strikes[1], strikes[2], strikes[3], strikes[4], position_size)
    else
        return strikes
    end
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
    load_ml_selector(path; rate, div_yield, spot_history, return_sizing) -> MLStrikeSelector

Load a trained ML selector from a BSON file.

# Arguments
- `path::String`: Path to BSON file
- `rate::Float64`: Risk-free rate for strike conversion
- `div_yield::Float64`: Dividend yield for strike conversion
- `spot_history::Dict{DateTime,SpotHistory}`: Spot history for feature extraction
- `return_sizing::Bool`: Whether to return position sizing (default: true)
"""
function load_ml_selector(
    path::String;
    rate::Float64=0.045,
    div_yield::Float64=0.013,
    spot_history::Dict{DateTime,SpotHistory}=Dict{DateTime,SpotHistory}(),
    return_sizing::Bool=true,
    use_logsig::Bool=false
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
        spot_history=spot_history,
        return_sizing=return_sizing,
        use_logsig=use_logsig
    )
end
