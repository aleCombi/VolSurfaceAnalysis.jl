# ML Feature Extraction from Volatility Surfaces
# Extracts features for ML-based strike selection
# Includes path signature features from recent spot behavior

# Signature level for path features (level 3 on 2D path gives 14 features)
const SIGNATURE_LEVEL = 3
const SIGNATURE_DIM = 14  # For 2D path at level 3: 2 + 4 + 8 = 14
const _LOGSIG_PATH_DIM = 2
const _LOGSIG_BASIS_CACHE = Dict{Tuple{Int,Int},Any}()

function _logsig_basis(path_dim::Int, level::Int)
    key = (path_dim, level)
    return get!(_LOGSIG_BASIS_CACHE, key) do
        prepare(path_dim, level)
    end
end

"""
    SurfaceFeatures

Features extracted from a volatility surface for ML-based strike selection.

# Fields
## Vol Surface Features
- `atm_iv::Float64`: ATM implied volatility (decimal)
- `atm_bid_ask_spread::Float64`: Bid-ask spread at ATM (as fraction of mid vol)
- `term_slope::Float64`: Slope of term structure (vol change per year of tau)
- `put_skew_25d::Float64`: 25-delta put IV minus ATM IV
- `call_skew_25d::Float64`: 25-delta call IV minus ATM IV
- `skew_asymmetry::Float64`: |put_skew| - |call_skew|
- `smile_curvature::Float64`: (put_skew + call_skew) / 2

## Market Context Features
- `spot::Float64`: Current spot price
- `spot_return_1d::Float64`: 1-day spot return
- `spot_return_5d::Float64`: 5-day spot return
- `tau::Float64`: Time to expiry in years
- `realized_vol_5d::Float64`: 5-day realized volatility of spot
- `iv_rv_ratio::Float64`: ATM IV / realized vol (vol risk premium)
- `avg_bid_ask_spread::Float64`: Average bid-ask spread across surface
- `n_strikes::Float64`: Number of available strikes (normalized)

## Path Signature Features (from recent spot path)
- `path_signature::Vector{Float64}`: Path signature of (time, log-price) path
"""
struct SurfaceFeatures
    # Vol surface features (7)
    atm_iv::Float64
    atm_bid_ask_spread::Float64
    term_slope::Float64
    put_skew_25d::Float64
    call_skew_25d::Float64
    skew_asymmetry::Float64
    smile_curvature::Float64
    # Market context (8)
    spot::Float64
    spot_return_1d::Float64
    spot_return_5d::Float64
    tau::Float64
    realized_vol_5d::Float64
    iv_rv_ratio::Float64
    avg_bid_ask_spread::Float64
    n_strikes::Float64
    # Path signature (14 for level 3 on 2D)
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

# 15 base features + 14 signature features = 29 total
const N_FEATURES = 15 + SIGNATURE_DIM

"""
    compute_path_signature(spot_history::Vector{Float64}, current_spot::Float64; level=SIGNATURE_LEVEL)
        -> Vector{Float64}

Compute the path signature of the recent spot price path.

Uses a 2D path representation: (normalized_time, log_return)
- Time is normalized to [0, 1]
- Log returns capture the price dynamics

# Arguments
- `spot_history::Vector{Float64}`: Historical spot prices (oldest first, most recent last)
- `current_spot::Float64`: Current spot price to append
- `level::Int`: Truncation level for signature (default: 3)

# Returns
- Vector of signature features (14 for level 3 on 2D path)
"""
function compute_path_signature(
    spot_history::Vector{Float64},
    current_spot::Float64;
    level::Int=SIGNATURE_LEVEL
)::Vector{Float64}
    # Build full price path including current spot
    prices = vcat(spot_history, current_spot)
    n = length(prices)

    if n < 3
        # Not enough data for meaningful signature
        return zeros(Float64, SIGNATURE_DIM)
    end

    # Create 2D path: (normalized_time, cumulative_log_return)
    # Using cumulative log returns captures both direction and volatility
    log_returns = diff(log.(prices))
    cum_log_returns = cumsum(log_returns)

    # Normalize time to [0, 1]
    times = collect(0.0:(1.0/(n-2)):(1.0))

    # Ensure same length
    if length(times) != length(cum_log_returns)
        times = range(0.0, 1.0, length=length(cum_log_returns))
    end

    # Build path matrix: rows are time points, columns are dimensions
    path = hcat(collect(times), cum_log_returns)

    # Compute signature
    try
        signature = sig(path, level)
        # Ensure we have exactly SIGNATURE_DIM features
        if length(signature) >= SIGNATURE_DIM
            return signature[1:SIGNATURE_DIM]
        else
            # Pad with zeros if needed
            return vcat(signature, zeros(Float64, SIGNATURE_DIM - length(signature)))
        end
    catch e
        # Return zeros on error
        return zeros(Float64, SIGNATURE_DIM)
    end
end

"""
    compute_logsig_features(spot_history::Vector{Float64}, current_spot::Float64; level=SIGNATURE_LEVEL)
        -> Vector{Float64}

Compute the log-signature of the recent spot price path.
Log-signature is more numerically stable and has nicer statistical properties.

# Arguments
- `spot_history::Vector{Float64}`: Historical spot prices
- `current_spot::Float64`: Current spot price
- `level::Int`: Truncation level

# Returns
- Vector of log-signature features padded/truncated to `SIGNATURE_DIM`
"""
function compute_logsig_features(
    spot_history::Vector{Float64},
    current_spot::Float64;
    level::Int=SIGNATURE_LEVEL
)::Vector{Float64}
    prices = vcat(spot_history, current_spot)
    n = length(prices)

    if n < 3
        return zeros(Float64, SIGNATURE_DIM)
    end

    log_returns = diff(log.(prices))
    cum_log_returns = cumsum(log_returns)
    times = range(0.0, 1.0, length=length(cum_log_returns))
    path = hcat(collect(times), cum_log_returns)

    try
        # ChenSignatures.logsig expects a precomputed BasisCache, not just level.
        basis = _logsig_basis(_LOGSIG_PATH_DIM, level)
        lsig = logsig(path, basis)
        if length(lsig) >= SIGNATURE_DIM
            return lsig[1:SIGNATURE_DIM]
        else
            return vcat(lsig, zeros(Float64, SIGNATURE_DIM - length(lsig)))
        end
    catch e
        return zeros(Float64, SIGNATURE_DIM)
    end
end

"""
    extract_features(surface, target_tau; spot_history=nothing, use_logsig=false)
        -> Union{SurfaceFeatures, Nothing}

Extract ML features from a volatility surface, including path signature features.

# Arguments
- `surface::VolatilitySurface`: The volatility surface
- `target_tau::Float64`: Target time to expiry in years
- `spot_history::Union{Nothing,SpotHistory}`: Historical spot prices with timestamps
  (oldest first), should have at least 6 values for proper signature computation
- `use_logsig::Bool`: Use log-signature instead of signature (default: false)

# Returns
- `SurfaceFeatures` if extraction succeeds, `nothing` otherwise
"""
function extract_features(
    surface::VolatilitySurface,
    target_tau::Float64;
    spot_history::Union{Nothing,SpotHistory}=nothing,
    use_logsig::Bool=false
)::Union{SurfaceFeatures,Nothing}
    # Filter points near target tau
    tau_tol = 0.01  # ~3.6 days tolerance
    tau_points = filter(p -> abs(p.τ - target_tau) < tau_tol, surface.points)
    isempty(tau_points) && return nothing

    # ATM IV: point closest to log_moneyness = 0
    atm_point = argmin(p -> abs(p.log_moneyness), tau_points)
    atm_iv = atm_point.vol
    atm_iv <= 0.0 && return nothing

    # ATM bid-ask spread
    atm_bid_ask_spread = if !ismissing(atm_point.bid_vol) && !ismissing(atm_point.ask_vol) && atm_point.vol > 0
        (atm_point.ask_vol - atm_point.bid_vol) / atm_point.vol
    else
        0.1  # Default to 10% of vol if missing
    end

    # Term structure slope (use all points near ATM)
    atm_threshold = 0.05
    atm_all_tenors = filter(p -> abs(p.log_moneyness) < atm_threshold, surface.points)
    term_slope = if length(atm_all_tenors) >= 2
        taus = [p.τ for p in atm_all_tenors]
        vols = [p.vol for p in atm_all_tenors]
        if maximum(taus) - minimum(taus) > 0.01  # At least ~4 days spread
            tau_mean = mean(taus)
            vol_mean = mean(vols)
            num = sum((t - tau_mean) * (v - vol_mean) for (t, v) in zip(taus, vols))
            den = sum((t - tau_mean)^2 for t in taus)
            den > 0 ? num / den : 0.0
        else
            0.0
        end
    else
        0.0
    end

    # Skew: approximate 25-delta points as ~0.67 sigma from ATM
    sigma_25d = 0.67 * atm_iv * sqrt(target_tau)

    # Put skew (OTM puts have positive log_moneyness in our convention)
    put_points = filter(p -> p.log_moneyness > 0 && abs(p.τ - target_tau) < tau_tol, surface.points)
    put_skew_25d = if !isempty(put_points)
        target_moneyness = sigma_25d
        closest_put = argmin(p -> abs(p.log_moneyness - target_moneyness), put_points)
        closest_put.vol - atm_iv
    else
        0.0
    end

    # Call skew (OTM calls have negative log_moneyness)
    call_points = filter(p -> p.log_moneyness < 0 && abs(p.τ - target_tau) < tau_tol, surface.points)
    call_skew_25d = if !isempty(call_points)
        target_moneyness = -sigma_25d
        closest_call = argmin(p -> abs(p.log_moneyness - target_moneyness), call_points)
        closest_call.vol - atm_iv
    else
        0.0
    end

    skew_asymmetry = abs(put_skew_25d) - abs(call_skew_25d)
    smile_curvature = (put_skew_25d + call_skew_25d) / 2

    # Spot features
    spot = surface.spot
    spot_return_1d = 0.0
    spot_return_5d = 0.0
    realized_vol_5d = 0.0
    path_signature = zeros(Float64, SIGNATURE_DIM)

    if spot_history !== nothing && length(spot_history.prices) >= 2
        times = spot_history.timestamps
        prices = spot_history.prices

        # 1-day and 5-day returns based on time offsets
        idx_1d = searchsortedlast(times, surface.timestamp - Day(1))
        if idx_1d >= 1
            spot_return_1d = (spot - prices[idx_1d]) / prices[idx_1d]
        end

        idx_5d = searchsortedlast(times, surface.timestamp - Day(5))
        if idx_5d >= 1
            spot_return_5d = (spot - prices[idx_5d]) / prices[idx_5d]
        end

        # Realized vol from daily returns (up to last 5 days)
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

        # Compute path signature from spot history (use full minute path)
        if use_logsig
            path_signature = compute_logsig_features(prices, spot)
        else
            path_signature = compute_path_signature(prices, spot)
        end
    end

    # IV/RV ratio (vol risk premium indicator)
    iv_rv_ratio = if realized_vol_5d > 0.01
        atm_iv / realized_vol_5d
    else
        1.0
    end

    # Liquidity features
    valid_spreads = Float64[]
    for p in tau_points
        if !ismissing(p.bid_vol) && !ismissing(p.ask_vol) && p.vol > 0
            spread = (p.ask_vol - p.bid_vol) / p.vol
            push!(valid_spreads, spread)
        end
    end
    avg_bid_ask_spread = isempty(valid_spreads) ? 0.1 : mean(valid_spreads)

    n_strikes = Float64(length(tau_points))

    return SurfaceFeatures(
        atm_iv,
        atm_bid_ask_spread,
        term_slope,
        put_skew_25d,
        call_skew_25d,
        skew_asymmetry,
        smile_curvature,
        spot,
        spot_return_1d,
        spot_return_5d,
        target_tau,
        realized_vol_5d,
        iv_rv_ratio,
        avg_bid_ask_spread,
        n_strikes,
        path_signature
    )
end

"""
    features_to_vector(f::SurfaceFeatures) -> Vector{Float32}

Convert SurfaceFeatures struct to a Float32 vector for model input.
Includes both base features and path signature features.
"""
function features_to_vector(f::SurfaceFeatures)::Vector{Float32}
    base_features = Float32[
        f.atm_iv,
        f.atm_bid_ask_spread,
        f.term_slope,
        f.put_skew_25d,
        f.call_skew_25d,
        f.skew_asymmetry,
        f.smile_curvature,
        f.spot / 500.0,  # Normalize spot (assuming SPY ~500)
        f.spot_return_1d,
        f.spot_return_5d,
        f.tau,
        f.realized_vol_5d,
        f.iv_rv_ratio,
        f.avg_bid_ask_spread,
        f.n_strikes / 100.0  # Normalize strike count
    ]

    # Append path signature features
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
