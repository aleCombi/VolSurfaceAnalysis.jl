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

Callable selector for iron condors that predicts inner deltas and selects wings.

Wing selection modes:
1. Fixed delta mode: Uses `wing_delta_abs` to select wings at fixed deltas
2. Objective mode: Uses `_condor_wings_by_objective` with `wing_objective`:
   - `:target_max_loss` (legacy): match `target_max_loss`
   - `:roi`: maximize entry credit / max loss
   - `:pnl`: maximize entry credit

Fields are similar to MLStrikeSelector with additional:
- `wing_delta_abs::Union{Nothing,Float32}`: Absolute delta for long wings (fixed delta mode)
- `target_max_loss::Union{Nothing,Float32}`: Target maximum loss per contract (max loss mode, in dollars)
- `wing_objective::Symbol`: Objective for selecting wings in objective mode
- `max_loss_min::Float32`: Minimum allowed max loss in dollars (objective mode)
- `max_loss_max::Float32`: Maximum allowed max loss in dollars (objective mode)
- `min_credit::Float32`: Minimum net credit in dollars (objective mode)
- `min_delta_gap::Float32`: Minimum delta gap between short and long legs
- `prefer_symmetric::Bool`: If true in max loss mode, prefer equal width spreads
"""
struct MLCondorStrikeSelector
    model::Any
    feature_means::Vector{Float32}
    feature_stds::Vector{Float32}
    min_delta::Float32
    max_delta::Float32
    wing_delta_abs::Union{Nothing,Float32}
    target_max_loss::Union{Nothing,Float32}
    wing_objective::Symbol
    max_loss_min::Float32
    max_loss_max::Float32
    min_credit::Float32
    min_delta_gap::Float32
    prefer_symmetric::Bool
    rate::Float64
    div_yield::Float64
    spot_history::Dict{DateTime,SpotHistory}
    return_sizing::Bool
    use_logsig::Bool
end

"""
    MLCondorStrikeSelector(model, feature_means, feature_stds; wing_delta_abs, target_max_loss, wing_objective, max_loss_min, max_loss_max, min_credit, min_delta_gap, prefer_symmetric, min_delta, max_delta, rate, div_yield, return_sizing)

Create an MLCondorStrikeSelector.

Mode selection precedence:
- Fixed mode if `wing_delta_abs !== nothing`, `target_max_loss===nothing`, and `wing_objective==:target_max_loss`
- Objective mode otherwise (requires valid objective configuration)
"""
function MLCondorStrikeSelector(
    model,
    feature_means::Vector{Float32},
    feature_stds::Vector{Float32};
    wing_delta_abs::Union{Nothing,Float32}=0.05f0,
    target_max_loss::Union{Nothing,Float32}=nothing,
    wing_objective::Symbol=:target_max_loss,
    max_loss_min::Float32=0.0f0,
    max_loss_max::Float32=Float32(Inf),
    min_credit::Float32=0.0f0,
    min_delta_gap::Float32=0.08f0,
    prefer_symmetric::Bool=true,
    min_delta::Float32=0.05f0,
    max_delta::Float32=0.35f0,
    rate::Float64=0.045,
    div_yield::Float64=0.013,
    spot_history::Dict{DateTime,SpotHistory}=Dict{DateTime,SpotHistory}(),
    return_sizing::Bool=true,
    use_logsig::Bool=false
)
    if !(wing_objective in (:target_max_loss, :roi, :pnl))
        error("wing_objective must be one of :target_max_loss, :roi, :pnl")
    end

    if max_loss_max < max_loss_min
        error("max_loss_max must be >= max_loss_min")
    end

    # If objective mode is selected with :target_max_loss, a target is required.
    uses_objective_mode = !(wing_delta_abs !== nothing && target_max_loss === nothing && wing_objective == :target_max_loss)
    if uses_objective_mode && wing_objective == :target_max_loss && target_max_loss === nothing
        error("target_max_loss must be provided when wing_objective=:target_max_loss in objective mode")
    end

    return MLCondorStrikeSelector(
        model,
        feature_means,
        feature_stds,
        min_delta,
        max_delta,
        wing_delta_abs,
        target_max_loss,
        wing_objective,
        max_loss_min,
        max_loss_max,
        min_credit,
        min_delta_gap,
        prefer_symmetric,
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
Uses either fixed-delta wings or objective-driven wings depending on configuration.
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

    # Determine short strikes using ML predicted deltas
    short_strikes = _delta_strangle_strikes_asymmetric(
        ctx,
        put_delta,
        call_delta;
        rate=selector.rate,
        div_yield=selector.div_yield
    )
    short_strikes === nothing && return nothing
    short_put_K, short_call_K = short_strikes

    # Select wings based on mode
    use_fixed_delta_wings = (
        selector.wing_delta_abs !== nothing &&
        selector.target_max_loss === nothing &&
        selector.wing_objective == :target_max_loss
    )

    wings = if use_fixed_delta_wings
        # Fixed delta mode
        long_put_K = _best_delta_strike(
            filter(r -> r.option_type == Put && r.strike < short_put_K, ctx.recs),
            -Float64(selector.wing_delta_abs),
            ctx.surface.spot,
            :put,
            ctx.surface.spot * exp((selector.rate - selector.div_yield) * tau),
            tau,
            selector.rate;
            debug=false
        )
        long_call_K = _best_delta_strike(
            filter(r -> r.option_type == Call && r.strike > short_call_K, ctx.recs),
            Float64(selector.wing_delta_abs),
            ctx.surface.spot,
            :call,
            ctx.surface.spot * exp((selector.rate - selector.div_yield) * tau),
            tau,
            selector.rate;
            debug=false
        )
        (long_put_K === nothing || long_call_K === nothing) ? nothing : (long_put_K, long_call_K)
    else
        _condor_wings_by_objective(
            ctx,
            short_put_K,
            short_call_K;
            objective=selector.wing_objective,
            target_max_loss=(selector.target_max_loss === nothing ? nothing : Float64(selector.target_max_loss)),
            max_loss_min=Float64(selector.max_loss_min),
            max_loss_max=Float64(selector.max_loss_max),
            min_credit=Float64(selector.min_credit),
            rate=selector.rate,
            div_yield=selector.div_yield,
            min_delta_gap=Float64(selector.min_delta_gap),
            prefer_symmetric=selector.prefer_symmetric,
            debug=false
        )
    end

    wings === nothing && return nothing
    long_put_K, long_call_K = wings

    # Validate ordering
    if !(long_put_K < short_put_K < ctx.surface.spot < short_call_K < long_call_K)
        return nothing
    end

    strikes = (short_put_K, short_call_K, long_put_K, long_call_K)

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
