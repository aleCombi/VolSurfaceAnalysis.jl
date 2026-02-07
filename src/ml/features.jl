# ML Feature Extraction from Volatility Surfaces
# Extracts features for ML-based strike selection
# Includes path signature features from recent spot behavior

# Signature level and path construction settings.
# We build a 2D base path (normalized time, cumulative log-return) and apply
# a lead-lag transform before computing signatures.
const SIGNATURE_LEVEL = 3
const _BASE_PATH_DIM = 2
const _LEADLAG_PATH_DIM = 2 * _BASE_PATH_DIM
const _LOGSIG_BASIS_CACHE = Dict{Tuple{Int,Int},Any}()

function _logsig_basis(path_dim::Int, level::Int)
    key = (path_dim, level)
    return get!(_LOGSIG_BASIS_CACHE, key) do
        prepare(path_dim, level)
    end
end

_sig_feature_dim(path_dim::Int, level::Int) = sum(path_dim^k for k in 1:level)
_logsig_feature_dim(path_dim::Int, level::Int) = length(_logsig_basis(path_dim, level).lynds)

# Signature dims for lead-lag transformed base path.
const SIGNATURE_DIM = _sig_feature_dim(_LEADLAG_PATH_DIM, SIGNATURE_LEVEL)
const LOGSIGNATURE_DIM = _logsig_feature_dim(_LEADLAG_PATH_DIM, SIGNATURE_LEVEL)

# Dead log-signature indices (constant or near-constant across training data).
# These are pruned from the feature vector to avoid noise.
const LOGSIG_DEAD_INDICES = [1, 3, 6, 12, 18, 24, 28]

# Moneyness proxies for multi-delta sampling: z = quantile(Normal(), 1-delta)
const _DELTA_Z_10 = 1.2815515655446004  # quantile(Normal(), 0.9)
const _DELTA_Z_25 = 0.6744897501960817  # quantile(Normal(), 0.75)
const _DELTA_Z_40 = 0.2533471031357997  # quantile(Normal(), 0.6)

"""
    path_feature_dim(; use_logsig=true, level=SIGNATURE_LEVEL) -> Int

Number of path-derived features produced by the configured signature transform.
"""
function path_feature_dim(; use_logsig::Bool=true, level::Int=SIGNATURE_LEVEL)::Int
    return use_logsig ?
        _logsig_feature_dim(_LEADLAG_PATH_DIM, level) :
        _sig_feature_dim(_LEADLAG_PATH_DIM, level)
end

"""
    pruned_logsig_dim(; level=SIGNATURE_LEVEL) -> Int

Number of logsig features after removing dead indices.
"""
function pruned_logsig_dim(; level::Int=SIGNATURE_LEVEL)::Int
    return _logsig_feature_dim(_LEADLAG_PATH_DIM, level) - length(LOGSIG_DEAD_INDICES)
end

# 36 base features + pruned logsig
const N_BASE_FEATURES = 36

"""
    n_features(; use_logsig=true, level=SIGNATURE_LEVEL) -> Int

Total input feature count: N_BASE_FEATURES + pruned path-derived features.
"""
function n_features(; use_logsig::Bool=true, level::Int=SIGNATURE_LEVEL)::Int
    if use_logsig
        return N_BASE_FEATURES + pruned_logsig_dim(; level=level)
    else
        return N_BASE_FEATURES + _sig_feature_dim(_LEADLAG_PATH_DIM, level)
    end
end

# Candidate-specific condor features appended to state features in scoring mode.
const N_CONDOR_CANDIDATE_FEATURES = 20

"""
    n_condor_scoring_features(; use_logsig=true, level=SIGNATURE_LEVEL) -> Int

Total feature count for condor candidate scoring:
state features + candidate-specific condor features.
"""
function n_condor_scoring_features(; use_logsig::Bool=true, level::Int=SIGNATURE_LEVEL)::Int
    return n_features(; use_logsig=use_logsig, level=level) + N_CONDOR_CANDIDATE_FEATURES
end

"""
    SurfaceFeatures

Features extracted from a volatility surface for ML-based strike selection.

36 base features + path signature vector (pruned to 23 for logsig) = 59 total.
"""
struct SurfaceFeatures
    # Vol surface (6)
    atm_iv::Float64
    atm_bid_ask_spread::Float64
    term_slope::Float64
    put_skew_25d::Float64
    call_skew_25d::Float64
    skew_asymmetry::Float64
    # Richer smile (10)
    put_skew_10d::Float64
    call_skew_10d::Float64
    put_skew_40d::Float64
    call_skew_40d::Float64
    risk_reversal_25d::Float64
    butterfly_25d::Float64
    smile_slope_put::Float64
    smile_slope_call::Float64
    wing_convexity_put::Float64
    wing_convexity_call::Float64
    # Spread by moneyness (6)
    spread_10d_put::Float64
    spread_25d_put::Float64
    spread_40d_put::Float64
    spread_10d_call::Float64
    spread_25d_call::Float64
    spread_40d_call::Float64
    # Surface dynamics (3)
    delta_atm_iv_1d::Float64
    delta_skew_1d::Float64
    delta_term_slope_1d::Float64
    # Volume (3)
    atm_volume::Float64
    total_volume::Float64
    volume_weighted_spread::Float64
    # Market context (8)
    spot::Float64
    spot_return_1d::Float64
    spot_return_5d::Float64
    tau::Float64
    realized_vol_5d::Float64
    iv_rv_ratio::Float64
    avg_bid_ask_spread::Float64
    n_strikes::Float64
    # Path signature (pruned)
    path_signature::Vector{Float64}
end

"""
    SpotHistory

Container for a timestamped spot price path (oldest first).
"""
struct SpotHistory
    timestamps::Vector{DateTime}
    prices::Vector{Float64}
end

# Default feature count uses log-signature path features.
const N_FEATURES = n_features()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

function _build_time_logreturn_path(prices::Vector{Float64})::Matrix{Float64}
    log_returns = diff(log.(prices))
    cum_log_returns = cumsum(log_returns)
    times = range(0.0, 1.0, length=length(cum_log_returns))
    return hcat(collect(times), cum_log_returns)
end

"""Remove dead logsig indices from a raw logsig vector."""
function _prune_logsig(raw::Vector{Float64})::Vector{Float64}
    return [raw[i] for i in eachindex(raw) if i ∉ LOGSIG_DEAD_INDICES]
end

"""
Extract IV skew and bid-ask spread at a given moneyness level on one side.
Returns (skew, spread) where skew = vol - atm_iv, spread = (ask-bid)/vol.
`side` is :put (positive log_moneyness) or :call (negative log_moneyness).
"""
function _iv_and_spread_at_moneyness(
    points::Vector, atm_iv::Float64, tau::Float64, tau_tol::Float64,
    delta_z::Float64, side::Symbol
)::Tuple{Float64,Float64}
    moneyness = delta_z * atm_iv * sqrt(tau)
    target_m = side == :put ? moneyness : -moneyness
    side_points = side == :put ?
        filter(p -> p.log_moneyness > 0 && abs(p.τ - tau) < tau_tol, points) :
        filter(p -> p.log_moneyness < 0 && abs(p.τ - tau) < tau_tol, points)
    isempty(side_points) && return (0.0, 0.0)
    closest = argmin(p -> abs(p.log_moneyness - target_m), side_points)
    skew = closest.vol - atm_iv
    spread = if !ismissing(closest.bid_vol) && !ismissing(closest.ask_vol) && closest.vol > 0
        (closest.ask_vol - closest.bid_vol) / closest.vol
    else
        0.0
    end
    return (skew, spread)
end

"""
Lightweight snapshot of surface for dynamics computation.
Returns (atm_iv, risk_reversal, term_slope) or nothing.
"""
function _surface_snapshot(
    surface::VolatilitySurface, target_tau::Float64
)::Union{Nothing,Tuple{Float64,Float64,Float64}}
    tau_tol = 0.01
    tau_points = filter(p -> abs(p.τ - target_tau) < tau_tol, surface.points)
    isempty(tau_points) && return nothing

    atm_point = argmin(p -> abs(p.log_moneyness), tau_points)
    atm_iv = atm_point.vol
    atm_iv <= 0.0 && return nothing

    # Put/call skew at 25d
    sigma_25d = _DELTA_Z_25 * atm_iv * sqrt(target_tau)
    put_pts = filter(p -> p.log_moneyness > 0 && abs(p.τ - target_tau) < tau_tol, surface.points)
    call_pts = filter(p -> p.log_moneyness < 0 && abs(p.τ - target_tau) < tau_tol, surface.points)
    ps25 = isempty(put_pts) ? 0.0 : (argmin(p -> abs(p.log_moneyness - sigma_25d), put_pts).vol - atm_iv)
    cs25 = isempty(call_pts) ? 0.0 : (argmin(p -> abs(p.log_moneyness + sigma_25d), call_pts).vol - atm_iv)
    rr = ps25 - cs25

    # Term slope
    atm_all = filter(p -> abs(p.log_moneyness) < 0.05, surface.points)
    ts = if length(atm_all) >= 2
        taus = [p.τ for p in atm_all]
        vols = [p.vol for p in atm_all]
        if maximum(taus) - minimum(taus) > 0.01
            tau_mean = mean(taus); vol_mean = mean(vols)
            num = sum((t - tau_mean) * (v - vol_mean) for (t, v) in zip(taus, vols))
            den = sum((t - tau_mean)^2 for t in taus)
            den > 0 ? num / den : 0.0
        else
            0.0
        end
    else
        0.0
    end
    return (atm_iv, rr, ts)
end

# ---------------------------------------------------------------------------
# Path signature computation
# ---------------------------------------------------------------------------

"""
    compute_path_signature(spot_history, current_spot; level) -> Vector{Float64}

Compute the path signature of the recent spot price path after lead-lag augmentation.
"""
function compute_path_signature(
    spot_history::Vector{Float64},
    current_spot::Float64;
    level::Int=SIGNATURE_LEVEL
)::Vector{Float64}
    expected_dim = path_feature_dim(; use_logsig=false, level=level)
    prices = vcat(spot_history, current_spot)
    n = length(prices)
    if n < 3
        return zeros(Float64, expected_dim)
    end
    path = _build_time_logreturn_path(prices)
    try
        return sig_leadlag(path, level)
    catch e
        return zeros(Float64, expected_dim)
    end
end

"""
    compute_logsig_features(spot_history, current_spot; level) -> Vector{Float64}

Compute the log-signature of the recent spot price path after lead-lag augmentation.
"""
function compute_logsig_features(
    spot_history::Vector{Float64},
    current_spot::Float64;
    level::Int=SIGNATURE_LEVEL
)::Vector{Float64}
    expected_dim = path_feature_dim(; use_logsig=true, level=level)
    prices = vcat(spot_history, current_spot)
    n = length(prices)
    if n < 3
        return zeros(Float64, expected_dim)
    end
    path = _build_time_logreturn_path(prices)
    try
        basis = _logsig_basis(_LEADLAG_PATH_DIM, level)
        return logsig_leadlag(path, basis)
    catch e
        return zeros(Float64, expected_dim)
    end
end

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

"""
    extract_features(surface, target_tau; spot_history, use_logsig, prev_surface)
        -> Union{SurfaceFeatures, Nothing}

Extract ML features from a volatility surface, including path signature features.
When `prev_surface` is provided, surface dynamics features are computed.
"""
function extract_features(
    surface::VolatilitySurface,
    target_tau::Float64;
    spot_history::Union{Nothing,SpotHistory}=nothing,
    use_logsig::Bool=false,
    prev_surface::Union{Nothing,VolatilitySurface}=nothing
)::Union{SurfaceFeatures,Nothing}
    tau_tol = 0.01  # ~3.6 days tolerance
    tau_points = filter(p -> abs(p.τ - target_tau) < tau_tol, surface.points)
    isempty(tau_points) && return nothing

    # ATM IV
    atm_point = argmin(p -> abs(p.log_moneyness), tau_points)
    atm_iv = atm_point.vol
    atm_iv <= 0.0 && return nothing

    # ATM bid-ask spread
    atm_bid_ask_spread = if !ismissing(atm_point.bid_vol) && !ismissing(atm_point.ask_vol) && atm_point.vol > 0
        (atm_point.ask_vol - atm_point.bid_vol) / atm_point.vol
    else
        0.1
    end

    # Term structure slope
    atm_all_tenors = filter(p -> abs(p.log_moneyness) < 0.05, surface.points)
    term_slope = if length(atm_all_tenors) >= 2
        taus = [p.τ for p in atm_all_tenors]
        vols = [p.vol for p in atm_all_tenors]
        if maximum(taus) - minimum(taus) > 0.01
            tau_mean = mean(taus); vol_mean = mean(vols)
            num = sum((t - tau_mean) * (v - vol_mean) for (t, v) in zip(taus, vols))
            den = sum((t - tau_mean)^2 for t in taus)
            den > 0 ? num / den : 0.0
        else
            0.0
        end
    else
        0.0
    end

    # Multi-delta IV and spread extraction
    put_skew_10d, spread_10d_put = _iv_and_spread_at_moneyness(surface.points, atm_iv, target_tau, tau_tol, _DELTA_Z_10, :put)
    put_skew_25d, spread_25d_put = _iv_and_spread_at_moneyness(surface.points, atm_iv, target_tau, tau_tol, _DELTA_Z_25, :put)
    put_skew_40d, spread_40d_put = _iv_and_spread_at_moneyness(surface.points, atm_iv, target_tau, tau_tol, _DELTA_Z_40, :put)
    call_skew_10d, spread_10d_call = _iv_and_spread_at_moneyness(surface.points, atm_iv, target_tau, tau_tol, _DELTA_Z_10, :call)
    call_skew_25d, spread_25d_call = _iv_and_spread_at_moneyness(surface.points, atm_iv, target_tau, tau_tol, _DELTA_Z_25, :call)
    call_skew_40d, spread_40d_call = _iv_and_spread_at_moneyness(surface.points, atm_iv, target_tau, tau_tol, _DELTA_Z_40, :call)

    skew_asymmetry = abs(put_skew_25d) - abs(call_skew_25d)

    # Derived smile shape features
    risk_reversal_25d = put_skew_25d - call_skew_25d
    butterfly_25d = (put_skew_25d + call_skew_25d) / 2

    # Smile slope: IV gradient from 40d to 10d (normalized by moneyness gap)
    put_moneyness_gap = (_DELTA_Z_10 - _DELTA_Z_40) * atm_iv * sqrt(target_tau)
    call_moneyness_gap = put_moneyness_gap  # symmetric by construction
    smile_slope_put = put_moneyness_gap > 1e-8 ? (put_skew_10d - put_skew_40d) / put_moneyness_gap : 0.0
    smile_slope_call = call_moneyness_gap > 1e-8 ? (call_skew_10d - call_skew_40d) / call_moneyness_gap : 0.0

    # Wing convexity: second difference
    wing_convexity_put = put_skew_10d - 2 * put_skew_25d + put_skew_40d
    wing_convexity_call = call_skew_10d - 2 * call_skew_25d + call_skew_40d

    # Surface dynamics (vs previous day)
    delta_atm_iv_1d = 0.0
    delta_skew_1d = 0.0
    delta_term_slope_1d = 0.0
    if prev_surface !== nothing
        snap = _surface_snapshot(prev_surface, target_tau)
        if snap !== nothing
            prev_atm, prev_rr, prev_ts = snap
            delta_atm_iv_1d = atm_iv - prev_atm
            delta_skew_1d = risk_reversal_25d - prev_rr
            delta_term_slope_1d = term_slope - prev_ts
        end
    end

    # Volume features (from records at target expiry)
    target_recs = filter(r -> abs(time_to_expiry(r.expiry, surface.timestamp) - target_tau) < tau_tol, surface.records)
    atm_volume = 0.0
    total_volume = 0.0
    volume_weighted_spread = 0.0
    if !isempty(target_recs)
        atm_moneyness_tol = 0.02
        for r in target_recs
            v = coalesce(r.volume, 0.0)
            total_volume += v
            lm = log(r.strike / surface.spot)
            if abs(lm) < atm_moneyness_tol
                atm_volume += v
            end
        end
        # Volume-weighted spread
        w_sum = 0.0
        s_sum = 0.0
        for r in target_recs
            v = coalesce(r.volume, 0.0)
            if v > 0 && !ismissing(r.bid_price) && !ismissing(r.ask_price) && r.ask_price > 0
                spread = (r.ask_price - r.bid_price) / r.ask_price
                w_sum += v
                s_sum += v * spread
            end
        end
        volume_weighted_spread = w_sum > 0 ? s_sum / w_sum : 0.0
    end

    # Spot features
    spot = surface.spot
    spot_return_1d = 0.0
    spot_return_5d = 0.0
    realized_vol_5d = 0.0
    raw_sig = zeros(Float64, path_feature_dim(; use_logsig=use_logsig))

    if spot_history !== nothing && length(spot_history.prices) >= 2
        times = spot_history.timestamps
        prices = spot_history.prices

        idx_1d = searchsortedlast(times, surface.timestamp - Day(1))
        if idx_1d >= 1
            spot_return_1d = (spot - prices[idx_1d]) / prices[idx_1d]
        end

        idx_5d = searchsortedlast(times, surface.timestamp - Day(5))
        if idx_5d >= 1
            spot_return_5d = (spot - prices[idx_5d]) / prices[idx_5d]
        end

        daily_prices = Float64[spot]
        for k in 1:5
            idx = searchsortedlast(times, surface.timestamp - Day(k))
            idx < 1 && break
            push!(daily_prices, prices[idx])
        end
        if length(daily_prices) >= 2
            returns = Float64[]
            for i in 1:(length(daily_prices) - 1)
                r = (daily_prices[i] - daily_prices[i+1]) / daily_prices[i+1]
                push!(returns, r)
            end
            if length(returns) >= 2
                realized_vol_5d = std(returns) * sqrt(252)
            end
        end

        if use_logsig
            raw_sig = compute_logsig_features(prices, spot)
        else
            raw_sig = compute_path_signature(prices, spot)
        end
    end

    # Prune dead logsig features
    path_signature = use_logsig ? _prune_logsig(raw_sig) : raw_sig

    iv_rv_ratio = realized_vol_5d > 0.01 ? atm_iv / realized_vol_5d : 1.0

    # Average bid-ask spread across surface
    valid_spreads = Float64[]
    for p in tau_points
        if !ismissing(p.bid_vol) && !ismissing(p.ask_vol) && p.vol > 0
            push!(valid_spreads, (p.ask_vol - p.bid_vol) / p.vol)
        end
    end
    avg_bid_ask_spread = isempty(valid_spreads) ? 0.1 : mean(valid_spreads)

    n_strikes = Float64(length(tau_points))

    return SurfaceFeatures(
        # Vol surface (6)
        atm_iv, atm_bid_ask_spread, term_slope,
        put_skew_25d, call_skew_25d, skew_asymmetry,
        # Richer smile (10)
        put_skew_10d, call_skew_10d, put_skew_40d, call_skew_40d,
        risk_reversal_25d, butterfly_25d,
        smile_slope_put, smile_slope_call,
        wing_convexity_put, wing_convexity_call,
        # Spread by moneyness (6)
        spread_10d_put, spread_25d_put, spread_40d_put,
        spread_10d_call, spread_25d_call, spread_40d_call,
        # Surface dynamics (3)
        delta_atm_iv_1d, delta_skew_1d, delta_term_slope_1d,
        # Volume (3)
        atm_volume, total_volume, volume_weighted_spread,
        # Market context (8)
        spot, spot_return_1d, spot_return_5d, target_tau,
        realized_vol_5d, iv_rv_ratio, avg_bid_ask_spread, n_strikes,
        # Path signature (pruned)
        path_signature
    )
end

"""
    features_to_vector(f::SurfaceFeatures) -> Vector{Float32}

Convert SurfaceFeatures struct to a Float32 vector for model input.
"""
function features_to_vector(f::SurfaceFeatures)::Vector{Float32}
    base_features = Float32[
        # Vol surface (6)
        f.atm_iv,
        f.atm_bid_ask_spread,
        f.term_slope,
        f.put_skew_25d,
        f.call_skew_25d,
        f.skew_asymmetry,
        # Richer smile (10)
        f.put_skew_10d,
        f.call_skew_10d,
        f.put_skew_40d,
        f.call_skew_40d,
        f.risk_reversal_25d,
        f.butterfly_25d,
        f.smile_slope_put,
        f.smile_slope_call,
        f.wing_convexity_put,
        f.wing_convexity_call,
        # Spread by moneyness (6)
        f.spread_10d_put,
        f.spread_25d_put,
        f.spread_40d_put,
        f.spread_10d_call,
        f.spread_25d_call,
        f.spread_40d_call,
        # Surface dynamics (3)
        f.delta_atm_iv_1d,
        f.delta_skew_1d,
        f.delta_term_slope_1d,
        # Volume (3)
        f.atm_volume / 1000.0,
        f.total_volume / 10000.0,
        f.volume_weighted_spread,
        # Market context (8)
        f.spot / 500.0,
        f.spot_return_1d,
        f.spot_return_5d,
        f.tau,
        f.realized_vol_5d,
        f.iv_rv_ratio,
        f.avg_bid_ask_spread,
        f.n_strikes / 100.0,
    ]

    sig_features = Float32.(f.path_signature)
    return vcat(base_features, sig_features)
end

"""
    normalize_features(X::Matrix{Float32}) -> (Matrix{Float32}, Vector{Float32}, Vector{Float32})

Normalize features to zero mean and unit variance.
Returns normalized matrix, means, and stds.
"""
function normalize_features(X::Matrix{Float32})
    means = vec(mean(X, dims=2))
    stds = vec(std(X, dims=2))
    # Avoid division by zero
    stds[stds .== 0] .= 1.0f0
    X_norm = (X .- means) ./ stds
    return X_norm, means, stds
end

"""
    apply_normalization(X::Matrix{Float32}, means::Vector{Float32}, stds::Vector{Float32}) -> Matrix{Float32}

Apply pre-computed normalization to features.
"""
function apply_normalization(
    X::Matrix{Float32},
    means::Vector{Float32},
    stds::Vector{Float32}
)::Matrix{Float32}
    return (X .- means) ./ stds
end
