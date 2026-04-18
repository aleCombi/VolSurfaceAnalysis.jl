# ML Feature System
#
# Features are callable structs that extract numeric values from a
# StrikeSelectionContext. Two abstract types:
#   Feature          — one value per entry timestamp (surface-level)
#   CandidateFeature — one value per candidate condor (strike-level)

# =============================================================================
# Abstract types
# =============================================================================

"""
    Feature

Abstract type for surface-level features.
Subtypes are callable: `(f::Feature)(ctx::StrikeSelectionContext) -> Union{Float64, Nothing}`
"""
abstract type Feature end

"""
    CandidateFeature

Abstract type for candidate-level features (depend on chosen strikes).
Subtypes are callable:
`(f::CandidateFeature)(ctx::StrikeSelectionContext, sp_K, sc_K, lp_K, lc_K) -> Union{Float64, Nothing}`
"""
abstract type CandidateFeature end

"""
    extract_surface_features(ctx, features) -> Union{Nothing, Vector{Float32}}

Extract and flatten surface features from a context. Handles both scalar
features (returning Float64) and multi-output features (returning Vector{Float64}).
Returns `nothing` if any feature returns `nothing`.
"""
function extract_surface_features(ctx::StrikeSelectionContext, features::Vector{<:Feature})::Union{Nothing, Vector{Float32}}
    result = Float32[]
    for f in features
        val = f(ctx)
        val === nothing && return nothing
        if val isa AbstractVector
            append!(result, Float32.(val))
        else
            push!(result, Float32(val))
        end
    end
    return result
end

"""
    surface_feature_dim(features) -> Int

Compute the total dimension of the surface feature vector, accounting for
multi-output features like SpotLogSig.
"""
function surface_feature_dim(features::Vector{<:Feature})::Int
    dim = 0
    for f in features
        if f isa SpotLogSig || f isa SpotMinuteLogSig || f isa IntradayLogSig
            dim += logsig_dim(f)
        else
            dim += 1
        end
    end
    return dim
end

# =============================================================================
# Surface features
# =============================================================================

"""ATM implied volatility for the target expiry."""
struct ATMImpliedVol <: Feature
    rate::Float64
    div_yield::Float64
end
ATMImpliedVol(; rate::Float64=0.0, div_yield::Float64=0.0) = ATMImpliedVol(rate, div_yield)

function (f::ATMImpliedVol)(ctx::StrikeSelectionContext)::Union{Float64,Nothing}
    tau = _ctx_tau(ctx)
    tau <= 0.0 && return nothing
    recs = _ctx_recs(ctx)
    return _atm_iv_from_records(recs, ctx.surface.spot, tau, f.rate, f.div_yield)
end

"""IV at a given delta minus ATM IV (skew measure)."""
struct DeltaSkew <: Feature
    delta::Float64
    side::Symbol   # :put or :call
    rate::Float64
    div_yield::Float64
end
DeltaSkew(delta::Float64, side::Symbol; rate::Float64=0.0, div_yield::Float64=0.0) =
    DeltaSkew(delta, side, rate, div_yield)

function (f::DeltaSkew)(ctx::StrikeSelectionContext)::Union{Float64,Nothing}
    tau = _ctx_tau(ctx)
    tau <= 0.0 && return nothing
    recs = _ctx_recs(ctx)
    spot = ctx.surface.spot
    F = spot * exp((f.rate - f.div_yield) * tau)

    atm_iv = _atm_iv_from_records(recs, spot, tau, f.rate, f.div_yield)
    atm_iv === nothing && return nothing

    side_recs = filter(r -> r.option_type == (f.side == :put ? Put : Call), recs)
    isempty(side_recs) && return nothing

    target_delta = f.side == :put ? -f.delta : f.delta
    strike = _best_delta_strike(side_recs, target_delta, spot, f.side, F, tau, f.rate)
    strike === nothing && return nothing

    rec = _find_rec_by_strike(side_recs, strike)
    rec === nothing && return nothing

    iv = if !ismissing(rec.mark_iv) && rec.mark_iv > 0
        rec.mark_iv / 100.0
    elseif !ismissing(rec.mark_price)
        computed = price_to_iv(rec.mark_price, F, rec.strike, tau, rec.option_type; r=f.rate)
        (isnan(computed) || computed <= 0.0) ? nothing : computed
    else
        nothing
    end
    iv === nothing && return nothing
    return iv - atm_iv
end

"""Risk reversal: call skew minus put skew at a given delta."""
struct RiskReversal <: Feature
    delta::Float64
    rate::Float64
    div_yield::Float64
end
RiskReversal(delta::Float64; rate::Float64=0.0, div_yield::Float64=0.0) =
    RiskReversal(delta, rate, div_yield)

"""Compute (call_skew, put_skew) at a given delta. Returns `nothing` if either side fails."""
function _dual_skew(ctx::StrikeSelectionContext, delta, rate, div_yield)
    call_skew = DeltaSkew(delta, :call; rate=rate, div_yield=div_yield)(ctx)
    put_skew = DeltaSkew(delta, :put; rate=rate, div_yield=div_yield)(ctx)
    (call_skew === nothing || put_skew === nothing) && return nothing
    return (call_skew, put_skew)
end

function (f::RiskReversal)(ctx::StrikeSelectionContext)::Union{Float64,Nothing}
    s = _dual_skew(ctx, f.delta, f.rate, f.div_yield)
    s === nothing ? nothing : s[1] - s[2]
end

"""Butterfly: average of put and call skew at a given delta."""
struct Butterfly <: Feature
    delta::Float64
    rate::Float64
    div_yield::Float64
end
Butterfly(delta::Float64; rate::Float64=0.0, div_yield::Float64=0.0) =
    Butterfly(delta, rate, div_yield)

function (f::Butterfly)(ctx::StrikeSelectionContext)::Union{Float64,Nothing}
    s = _dual_skew(ctx, f.delta, f.rate, f.div_yield)
    s === nothing ? nothing : (s[2] + s[1]) / 2.0
end

"""ATM IV slope across expiries (term structure slope)."""
struct TermSlope <: Feature
    rate::Float64
    div_yield::Float64
end
TermSlope(; rate::Float64=0.0, div_yield::Float64=0.0) = TermSlope(rate, div_yield)

function (f::TermSlope)(ctx::StrikeSelectionContext)::Union{Float64,Nothing}
    tau = _ctx_tau(ctx)
    tau <= 0.0 && return nothing

    all_expiries = unique(r.expiry for r in ctx.surface.records)
    length(all_expiries) < 2 && return nothing

    target_recs = _ctx_recs(ctx)
    atm_iv_target = _atm_iv_from_records(target_recs, ctx.surface.spot, tau, f.rate, f.div_yield)
    atm_iv_target === nothing && return nothing

    # Find the next available expiry after the target
    other_expiries = filter(e -> e != ctx.expiry, all_expiries)
    isempty(other_expiries) && return nothing

    # Pick the closest expiry that is different from target
    taus_other = [time_to_expiry(e, ctx.surface.timestamp) for e in other_expiries]
    idx = argmin(abs.(taus_other .- tau))
    next_expiry = other_expiries[idx]
    next_tau = taus_other[idx]
    (next_tau <= 0.0 || next_tau == tau) && return nothing

    next_recs = filter(r -> r.expiry == next_expiry, ctx.surface.records)
    atm_iv_next = _atm_iv_from_records(next_recs, ctx.surface.spot, next_tau, f.rate, f.div_yield)
    atm_iv_next === nothing && return nothing

    return (atm_iv_next - atm_iv_target) / (next_tau - tau)
end

"""Relative bid-ask spread on the nearest-to-ATM record."""
struct ATMSpread <: Feature end

function (::ATMSpread)(ctx::StrikeSelectionContext)::Union{Float64,Nothing}
    recs = _ctx_recs(ctx)
    isempty(recs) && return nothing
    atm_rec = recs[argmin([abs(r.strike - ctx.surface.spot) for r in recs])]
    return _relative_spread(atm_rec)
end

"""Relative bid-ask spread at a given delta."""
struct DeltaSpread <: Feature
    delta::Float64
    side::Symbol
    rate::Float64
    div_yield::Float64
end
DeltaSpread(delta::Float64, side::Symbol; rate::Float64=0.0, div_yield::Float64=0.0) =
    DeltaSpread(delta, side, rate, div_yield)

function (f::DeltaSpread)(ctx::StrikeSelectionContext)::Union{Float64,Nothing}
    tau = _ctx_tau(ctx)
    tau <= 0.0 && return nothing
    recs = _ctx_recs(ctx)
    spot = ctx.surface.spot
    F = spot * exp((f.rate - f.div_yield) * tau)
    side_recs = filter(r -> r.option_type == (f.side == :put ? Put : Call), recs)
    isempty(side_recs) && return nothing
    target_delta = f.side == :put ? -f.delta : f.delta
    strike = _best_delta_strike(side_recs, target_delta, spot, f.side, F, tau, f.rate)
    strike === nothing && return nothing
    rec = _find_rec_by_strike(side_recs, strike)
    rec === nothing && return nothing
    return _relative_spread(rec)
end

"""Total volume across all records for the target expiry."""
struct TotalVolume <: Feature end

function (::TotalVolume)(ctx::StrikeSelectionContext)::Union{Float64,Nothing}
    recs = _ctx_recs(ctx)
    isempty(recs) && return nothing
    return Float64(sum(coalesce(r.volume, 0.0) for r in recs))
end

"""Ratio of put volume to call volume."""
struct PutCallVolumeRatio <: Feature end

function (::PutCallVolumeRatio)(ctx::StrikeSelectionContext)::Union{Float64,Nothing}
    recs = _ctx_recs(ctx)
    put_vol = sum(coalesce(r.volume, 0.0) for r in recs if r.option_type == Put; init=0.0)
    call_vol = sum(coalesce(r.volume, 0.0) for r in recs if r.option_type == Call; init=0.0)
    call_vol <= 0.0 && return nothing
    return put_vol / call_vol
end

"""Hour of day as fraction [0, 1)."""
struct HourOfDay <: Feature end

function (::HourOfDay)(ctx::StrikeSelectionContext)::Union{Float64,Nothing}
    return Dates.hour(ctx.surface.timestamp) / 24.0
end

"""Day of week: Mon=0, Fri=1, scaled to [0, 1]."""
struct DayOfWeek <: Feature end

function (::DayOfWeek)(ctx::StrikeSelectionContext)::Union{Float64,Nothing}
    return (Dates.dayofweek(ctx.surface.timestamp) - 1) / 4.0
end

# =============================================================================
# History-based features (backward-looking, use ctx.history)
# =============================================================================

"""
    _historical_spots(ctx, n) -> Union{Nothing, Vector{Float64}}

Extract the most recent `n` daily spot prices from historical surfaces,
ordered oldest-to-newest. Returns `nothing` if fewer than `n` are available.
"""
function _historical_spots(ctx::StrikeSelectionContext, n::Int)::Union{Nothing,Vector{Float64}}
    hist_ts = available_timestamps(ctx.history)
    length(hist_ts) < n && return nothing
    recent_ts = hist_ts[max(1, end-n+1):end]
    spots = Float64[]
    for ts in recent_ts
        surf = get_surface(ctx.history, ts)
        surf === nothing && continue
        push!(spots, surf.spot)
    end
    length(spots) < n && return nothing
    return spots
end

"""
    _historical_atm_ivs(ctx, n; rate, div_yield) -> Union{Nothing, Vector{Float64}}

Extract the most recent `n` daily ATM IVs from historical surfaces.
"""
function _historical_atm_ivs(
    ctx::StrikeSelectionContext, n::Int;
    rate::Float64=0.0, div_yield::Float64=0.0
)::Union{Nothing,Vector{Float64}}
    hist_ts = available_timestamps(ctx.history)
    length(hist_ts) < n && return nothing
    recent_ts = hist_ts[max(1, end-n+1):end]
    ivs = Float64[]
    for ts in recent_ts
        surf = get_surface(ctx.history, ts)
        surf === nothing && continue
        # Use a 1-day tau for ATM IV computation on historical surfaces
        recs = surf.records
        isempty(recs) && continue
        # Find the closest expiry to 1 day out
        expiries = unique(r.expiry for r in recs)
        taus = [time_to_expiry(e, ts) for e in expiries]
        valid = findall(t -> t > 0.0, taus)
        isempty(valid) && continue
        best = valid[argmin(taus[valid])]
        exp_recs = filter(r -> r.expiry == expiries[best], recs)
        iv = _atm_iv_from_records(exp_recs, surf.spot, taus[best], rate, div_yield)
        iv === nothing && continue
        push!(ivs, iv)
    end
    length(ivs) < n && return nothing
    return ivs
end

"""Realized volatility from daily log returns over `lookback` days (annualized)."""
struct RealizedVol <: Feature
    lookback::Int
end
RealizedVol(; lookback::Int=20) = RealizedVol(lookback)

function (f::RealizedVol)(ctx::StrikeSelectionContext)::Union{Float64,Nothing}
    spots = _historical_spots(ctx, f.lookback + 1)
    spots === nothing && return nothing
    log_returns = diff(log.(spots))
    length(log_returns) < 2 && return nothing
    return std(log_returns) * sqrt(252.0)
end

"""Variance risk premium: ATM IV minus recent realized vol."""
struct VarianceRiskPremium <: Feature
    lookback::Int
    rate::Float64
    div_yield::Float64
end
VarianceRiskPremium(; lookback::Int=20, rate::Float64=0.0, div_yield::Float64=0.0) =
    VarianceRiskPremium(lookback, rate, div_yield)

function (f::VarianceRiskPremium)(ctx::StrikeSelectionContext)::Union{Float64,Nothing}
    atm_iv = ATMImpliedVol(f.rate, f.div_yield)(ctx)
    atm_iv === nothing && return nothing
    rv = RealizedVol(f.lookback)(ctx)
    rv === nothing && return nothing
    return atm_iv - rv
end

"""Spot return over `lookback` days: (spot_now / spot_then) - 1."""
struct SpotMomentum <: Feature
    lookback::Int
end
SpotMomentum(; lookback::Int=5) = SpotMomentum(lookback)

function (f::SpotMomentum)(ctx::StrikeSelectionContext)::Union{Float64,Nothing}
    spots = _historical_spots(ctx, f.lookback + 1)
    spots === nothing && return nothing
    return spots[end] / spots[1] - 1.0
end

"""Change in ATM IV over `lookback` days (current IV - past IV)."""
struct IVChange <: Feature
    lookback::Int
    rate::Float64
    div_yield::Float64
end
IVChange(; lookback::Int=5, rate::Float64=0.0, div_yield::Float64=0.0) =
    IVChange(lookback, rate, div_yield)

function (f::IVChange)(ctx::StrikeSelectionContext)::Union{Float64,Nothing}
    ivs = _historical_atm_ivs(ctx, f.lookback + 1; rate=f.rate, div_yield=f.div_yield)
    ivs === nothing && return nothing
    return ivs[end] - ivs[1]
end

"""Current ATM IV percentile within the last `lookback` days."""
struct IVPercentile <: Feature
    lookback::Int
    rate::Float64
    div_yield::Float64
end
IVPercentile(; lookback::Int=20, rate::Float64=0.0, div_yield::Float64=0.0) =
    IVPercentile(lookback, rate, div_yield)

function (f::IVPercentile)(ctx::StrikeSelectionContext)::Union{Float64,Nothing}
    ivs = _historical_atm_ivs(ctx, f.lookback; rate=f.rate, div_yield=f.div_yield)
    ivs === nothing && return nothing
    current_iv = ATMImpliedVol(f.rate, f.div_yield)(ctx)
    current_iv === nothing && return nothing
    return count(iv -> iv <= current_iv, ivs) / length(ivs)
end

# =============================================================================
# Signature features
# =============================================================================

"""
    SpotLogSig

Log-signature of the time-augmented spot return path over a lookback window.
Returns a vector of features (one per logsig component) rather than a single Float64.

The path has 2 channels: normalized time ∈ [0,1] and cumulative log return.
At depth `m` this yields `logsig_dim(2, m)` features.

This is a *multi-output* feature — it returns a `Vector{Float64}`, not a scalar.
Use `extract_multi_features` or flatten manually.
"""
struct SpotLogSig <: Feature
    lookback::Int
    depth::Int
    basis::Any  # ChenSignatures.BasisCache
end

function SpotLogSig(; lookback::Int=20, depth::Int=3)
    basis = prepare(2, depth)
    SpotLogSig(lookback, depth, basis)
end

"""Number of logsig components for this feature."""
logsig_dim(f::SpotLogSig) = length(f.basis.lynds)

function (f::SpotLogSig)(ctx::StrikeSelectionContext)::Union{Vector{Float64},Nothing}
    spots = _historical_spots(ctx, f.lookback + 1)
    spots === nothing && return nothing

    # Build time-augmented path: (normalized_time, cumulative_log_return)
    n = length(spots)
    path = Matrix{Float64}(undef, n, 2)
    for i in 1:n
        path[i, 1] = (i - 1) / (n - 1)           # time ∈ [0, 1]
        path[i, 2] = log(spots[i] / spots[1])     # cumulative log return
    end

    return Vector{Float64}(logsig(path, f.basis))
end

"""
    SpotMinuteLogSig

Log-signature of minute-level spot prices over the last `lookback_hours` hours.
Uses `get_spots` to pull high-frequency spot data from the data source,
providing much richer path information than the daily `SpotLogSig`.

This is a *multi-output* feature — it returns a `Vector{Float64}`.
"""
struct SpotMinuteLogSig <: Feature
    lookback_hours::Int
    depth::Int
    min_points::Int
    basis::Any  # ChenSignatures.BasisCache
end

function SpotMinuteLogSig(; lookback_hours::Int=6, depth::Int=3, min_points::Int=30)
    basis = prepare(2, depth)
    SpotMinuteLogSig(lookback_hours, depth, min_points, basis)
end

logsig_dim(f::SpotMinuteLogSig) = length(f.basis.lynds)

function (f::SpotMinuteLogSig)(ctx::StrikeSelectionContext)::Union{Vector{Float64},Nothing}
    ts = ctx.surface.timestamp
    from = ts - Hour(f.lookback_hours)
    spot_dict = get_spots(ctx.history, from, ts)
    isempty(spot_dict) && return nothing

    # Sort by timestamp
    sorted_ts = sort(collect(keys(spot_dict)))
    length(sorted_ts) < f.min_points && return nothing

    prices = [spot_dict[t] for t in sorted_ts]

    # Build time-augmented path: (normalized_time, cumulative_log_return)
    n = length(prices)
    path = Matrix{Float64}(undef, n, 2)
    for i in 1:n
        path[i, 1] = (i - 1) / (n - 1)
        path[i, 2] = log(prices[i] / prices[1])
    end

    return Vector{Float64}(logsig(path, f.basis))
end

# =============================================================================
# Intraday ATM IV path helper
# =============================================================================

"""
    BarCache

Per-date cache of option bars grouped by timestamp. Loads the full day's
parquet once, then slices in Julia for any time range — no repeated DuckDB queries.
"""
struct BarCache
    store::LocalDataStore
    data::Dict{Tuple{Date, String}, Dict{DateTime, Vector{PolygonBar}}}
end
BarCache(store::LocalDataStore) = BarCache(store, Dict{Tuple{Date,String}, Dict{DateTime, Vector{PolygonBar}}}())

function _get_bars(cache::BarCache, date::Date, underlying::Union{Underlying, AbstractString};
                   spot_hint::Float64=NaN, half_range::Float64=0.06)
    sym = underlying isa Underlying ? ticker(underlying) : uppercase(String(underlying))
    key = (date, sym)
    if !haskey(cache.data, key)
        path = polygon_options_path(cache.store, date, sym)
        by_ts = Dict{DateTime, Vector{PolygonBar}}()
        if isfile(path)
            open_utc = et_to_utc(date, Time(9, 30))
            close_utc = et_to_utc(date, Time(16, 0))
            from_str = Dates.format(open_utc, "yyyy-mm-dd HH:MM:SS")
            to_str = Dates.format(close_utc, "yyyy-mm-dd HH:MM:SS")
            where = "timestamp BETWEEN '$from_str' AND '$to_str'"
            if !isnan(spot_hint) && spot_hint > 0
                lo = round(spot_hint * (1.0 - half_range), digits=0)
                hi = round(spot_hint * (1.0 + half_range), digits=0)
                try
                    for bar in read_polygon_parquet(path;
                            where=where * " AND parsed_strike BETWEEN $lo AND $hi")
                        push!(get!(by_ts, bar.timestamp, PolygonBar[]), bar)
                    end
                catch  # parsed_strike column missing in older files
                    empty!(by_ts)
                    for bar in read_polygon_parquet(path; where=where)
                        push!(get!(by_ts, bar.timestamp, PolygonBar[]), bar)
                    end
                end
            else
                for bar in read_polygon_parquet(path; where=where)
                    push!(get!(by_ts, bar.timestamp, PolygonBar[]), bar)
                end
            end
        end
        cache.data[key] = by_ts
    end
    return cache.data[key]
end

"""
    _bulk_put_skew(cache, underlying, date, spot_dict, from, to; rate, div_yield, target_moneyness)

Compute put skew (OTM put IV - ATM IV) at each minute in [from, to].
`target_moneyness` is strike/spot for the OTM put (e.g. 0.95 for 5% OTM).
Shares the same bar cache as `_bulk_atm_iv` — one DuckDB read per date.
"""
function _bulk_put_skew(
    cache::BarCache,
    underlying::Union{Underlying, AbstractString},
    date::Date,
    spot_dict::Dict{DateTime, Float64},
    from::DateTime,
    to::DateTime;
    rate::Float64=0.0,
    div_yield::Float64=0.0,
    spot_hint::Float64=NaN,
    target_moneyness::Float64=0.95
)::Dict{DateTime, Float64}
    all_bars = _get_bars(cache, date, underlying; spot_hint=spot_hint)
    isempty(all_bars) && return Dict{DateTime, Float64}()

    skews = Dict{DateTime, Float64}()
    for (ts, ts_bars) in all_bars
        (from <= ts <= to) || continue
        spot = get(spot_dict, ts, nothing)
        spot === nothing && continue

        # Pick nearest expiry (shortest positive tau)
        best_tau = Inf
        best_expiry = DateTime(0)
        for bar in ts_bars
            tau = time_to_expiry(bar.expiry, ts)
            if 0.0 < tau < best_tau
                best_tau = tau
                best_expiry = bar.expiry
            end
        end
        best_tau == Inf && continue

        F = spot * exp((rate - div_yield) * best_tau)
        target_strike = spot * target_moneyness

        # Find ATM bar and OTM put bar at that expiry
        atm_bar = nothing
        otm_bar = nothing
        atm_dist = Inf
        otm_dist = Inf
        for bar in ts_bars
            bar.expiry == best_expiry || continue
            dist_atm = abs(bar.strike - spot)
            dist_otm = abs(bar.strike - target_strike)
            if dist_atm < atm_dist
                atm_dist = dist_atm
                atm_bar = bar
            end
            if bar.option_type == Put && bar.strike < spot && dist_otm < otm_dist
                otm_dist = dist_otm
                otm_bar = bar
            end
        end
        (atm_bar === nothing || otm_bar === nothing) && continue

        # Compute IVs
        atm_price = atm_bar.close / spot
        atm_iv = price_to_iv(atm_price, F, atm_bar.strike, best_tau, atm_bar.option_type; r=rate)
        (isnan(atm_iv) || atm_iv <= 0.0 || atm_iv > 5.0) && continue

        otm_price = otm_bar.close / spot
        otm_iv = price_to_iv(otm_price, F, otm_bar.strike, best_tau, Put; r=rate)
        (isnan(otm_iv) || otm_iv <= 0.0 || otm_iv > 5.0) && continue

        skews[ts] = otm_iv - atm_iv
    end
    return skews
end

"""
    _bulk_atm_iv(cache, underlying, date, spot_dict, from, to; rate, div_yield)

Compute ATM IV at each minute in [from, to] using cached bars. ONE DuckDB read
per (date, symbol), reused across all entry times on the same date.
"""
function _bulk_atm_iv(
    cache::BarCache,
    underlying::Union{Underlying, AbstractString},
    date::Date,
    spot_dict::Dict{DateTime, Float64},
    from::DateTime,
    to::DateTime;
    rate::Float64=0.0,
    div_yield::Float64=0.0,
    spot_hint::Float64=NaN
)::Dict{DateTime, Float64}
    all_bars = _get_bars(cache, date, underlying; spot_hint=spot_hint)
    isempty(all_bars) && return Dict{DateTime, Float64}()

    ivs = Dict{DateTime, Float64}()
    for (ts, ts_bars) in all_bars
        (from <= ts <= to) || continue
        spot = get(spot_dict, ts, nothing)
        spot === nothing && continue

        # Pick nearest expiry (shortest positive tau)
        best_tau = Inf
        best_expiry = DateTime(0)
        for bar in ts_bars
            tau = time_to_expiry(bar.expiry, ts)
            if 0.0 < tau < best_tau
                best_tau = tau
                best_expiry = bar.expiry
            end
        end
        best_tau == Inf && continue

        # Find nearest-ATM bar at that expiry
        best_bar = nothing
        best_dist = Inf
        for bar in ts_bars
            bar.expiry == best_expiry || continue
            dist = abs(bar.strike - spot)
            if dist < best_dist
                best_dist = dist
                best_bar = bar
            end
        end
        best_bar === nothing && continue

        # IV from close price (close is USD, divide by spot for fraction)
        mark_price = best_bar.close / spot
        F = spot * exp((rate - div_yield) * best_tau)
        iv = price_to_iv(mark_price, F, best_bar.strike, best_tau, best_bar.option_type; r=rate)
        (isnan(iv) || iv <= 0.0 || iv > 5.0) && continue

        ivs[ts] = iv
    end
    return ivs
end

# =============================================================================
# Intraday (time, spot, ATM vol) log-signature
# =============================================================================

const MARKET_OPEN_ET = Time(9, 30)

"""
    IntradayLogSig(; depth=3, rate=0.0, div_yield=0.0, min_points=20, store=DEFAULT_STORE)

Log-signature of the intraday 2-channel path (spot, ATM implied vol)
from market open (9:30 ET) to the current entry timestamp.

# Channels
1. **spot**: cumulative log-return from market open
2. **ATM vol**: change in ATM IV from market open (IV_t - IV_0)

# What the signature captures
- Level 1: net spot return, net IV change
- Level 2: spot-IV cross-area (leverage effect — does vol rise when spot drops?)
- Level 3: higher-order dynamics of the spot-vol relationship

For depth=3 with 2 channels: **5 components**.
"""
struct IntradayLogSig <: Feature
    depth::Int
    channels::Int
    rate::Float64
    div_yield::Float64
    min_points::Int
    cache::BarCache
    basis::Any  # ChenSignatures.BasisCache
    skew_moneyness::Float64  # only used when channels == 3
end

function IntradayLogSig(; depth::Int=3, channels::Int=2,
                          rate::Float64=0.0, div_yield::Float64=0.0,
                          min_points::Int=20, store::LocalDataStore=DEFAULT_STORE,
                          skew_moneyness::Float64=0.95)
    @assert channels in (2, 3) "IntradayLogSig supports 2 or 3 channels"
    basis = prepare(channels, depth)
    IntradayLogSig(depth, channels, rate, div_yield, min_points, BarCache(store), basis, skew_moneyness)
end

logsig_dim(f::IntradayLogSig) = length(f.basis.lynds)

function (f::IntradayLogSig)(ctx::StrikeSelectionContext)::Union{Vector{Float64},Nothing}
    ts = ctx.surface.timestamp
    open_utc = et_to_utc(Date(ts), MARKET_OPEN_ET)

    # Minute spots from history (lazy per-date)
    spot_dict = get_spots(ctx.history, open_utc, ts)
    isempty(spot_dict) && return nothing

    # ATM IV: cached per (date, symbol), one DuckDB read per date
    iv_dict = _bulk_atm_iv(f.cache, ctx.surface.underlying, Date(ts), spot_dict,
                           open_utc, ts; rate=f.rate, div_yield=f.div_yield,
                           spot_hint=ctx.surface.spot)
    isempty(iv_dict) && return nothing

    # Optional skew channel
    skew_dict = Dict{DateTime, Float64}()
    if f.channels == 3
        skew_dict = _bulk_put_skew(f.cache, ctx.surface.underlying, Date(ts), spot_dict,
                                   open_utc, ts; rate=f.rate, div_yield=f.div_yield,
                                   spot_hint=ctx.surface.spot,
                                   target_moneyness=f.skew_moneyness)
        isempty(skew_dict) && return nothing
    end

    # Timestamps where we have all channels
    common_ts = sort(intersect(collect(keys(spot_dict)), collect(keys(iv_dict))))
    if f.channels == 3
        common_ts = sort(intersect(common_ts, collect(keys(skew_dict))))
    end
    length(common_ts) < f.min_points && return nothing

    spots = [spot_dict[t] for t in common_ts]
    ivs = [iv_dict[t] for t in common_ts]

    n = length(common_ts)
    path = Matrix{Float64}(undef, n, f.channels)
    for i in 1:n
        path[i, 1] = log(spots[i] / spots[1])     # cumulative log return
        path[i, 2] = ivs[i] - ivs[1]              # IV change from open
    end
    if f.channels == 3
        skews = [skew_dict[t] for t in common_ts]
        for i in 1:n
            path[i, 3] = skews[i] - skews[1]      # skew change from open
        end
    end

    return Vector{Float64}(logsig(path, f.basis))
end

# =============================================================================
# Default surface feature set
# =============================================================================

function default_surface_features(; rate::Float64=0.0, div_yield::Float64=0.0)
    Feature[
        # Current surface snapshot
        ATMImpliedVol(; rate, div_yield),
        DeltaSkew(0.25, :put; rate, div_yield),
        ATMSpread(),
        # Backward-looking dynamics
        RealizedVol(; lookback=20),
        VarianceRiskPremium(; lookback=20, rate, div_yield),
        SpotMomentum(; lookback=5),
        SpotMomentum(; lookback=20),
        IVChange(; lookback=5, rate, div_yield),
        IVPercentile(; lookback=20, rate, div_yield),
    ]
end

const DEFAULT_SURFACE_FEATURES = default_surface_features()

# =============================================================================
# Candidate features
# =============================================================================

"""Compute absolute delta for a single leg given option type and strike."""
function _short_leg_delta(ctx::StrikeSelectionContext, opt_type::OptionType, strike_K, rate, div_yield)
    tau = _ctx_tau(ctx)
    tau <= 0.0 && return nothing
    recs = _ctx_recs(ctx)
    F = ctx.surface.spot * exp((rate - div_yield) * tau)
    rec = _find_rec_by_strike(filter(r -> r.option_type == opt_type, recs), Float64(strike_K))
    rec === nothing && return nothing
    d = _delta_from_record(rec, F, tau, rate)
    d === missing && return nothing
    return abs(d)
end

"""Absolute delta of the short put leg."""
struct ShortPutDelta <: CandidateFeature
    rate::Float64
    div_yield::Float64
end
ShortPutDelta(; rate::Float64=0.0, div_yield::Float64=0.0) = ShortPutDelta(rate, div_yield)

(f::ShortPutDelta)(ctx::StrikeSelectionContext, sp_K, sc_K, lp_K, lc_K) =
    _short_leg_delta(ctx, Put, sp_K, f.rate, f.div_yield)

"""Absolute delta of the short call leg."""
struct ShortCallDelta <: CandidateFeature
    rate::Float64
    div_yield::Float64
end
ShortCallDelta(; rate::Float64=0.0, div_yield::Float64=0.0) = ShortCallDelta(rate, div_yield)

(f::ShortCallDelta)(ctx::StrikeSelectionContext, sp_K, sc_K, lp_K, lc_K) =
    _short_leg_delta(ctx, Call, sc_K, f.rate, f.div_yield)

"""Net entry credit as fraction of spot."""
struct EntryCredit <: CandidateFeature end

function (::EntryCredit)(ctx::StrikeSelectionContext, sp_K, sc_K, lp_K, lc_K)::Union{Float64,Nothing}
    recs = _ctx_recs(ctx)
    put_recs = filter(r -> r.option_type == Put, recs)
    call_recs = filter(r -> r.option_type == Call, recs)

    sp_rec = _find_rec_by_strike(put_recs, Float64(sp_K))
    sc_rec = _find_rec_by_strike(call_recs, Float64(sc_K))
    lp_rec = _find_rec_by_strike(put_recs, Float64(lp_K))
    lc_rec = _find_rec_by_strike(call_recs, Float64(lc_K))

    (sp_rec === nothing || sc_rec === nothing || lp_rec === nothing || lc_rec === nothing) && return nothing

    sp_bid = ismissing(sp_rec.bid_price) ? (ismissing(sp_rec.mark_price) ? (return nothing) : sp_rec.mark_price) : sp_rec.bid_price
    sc_bid = ismissing(sc_rec.bid_price) ? (ismissing(sc_rec.mark_price) ? (return nothing) : sc_rec.mark_price) : sc_rec.bid_price
    lp_ask = ismissing(lp_rec.ask_price) ? (ismissing(lp_rec.mark_price) ? (return nothing) : lp_rec.mark_price) : lp_rec.ask_price
    lc_ask = ismissing(lc_rec.ask_price) ? (ismissing(lc_rec.mark_price) ? (return nothing) : lc_rec.mark_price) : lc_rec.ask_price

    return (sp_bid + sc_bid) - (lp_ask + lc_ask)
end

"""Max loss as fraction of spot."""
struct MaxLoss <: CandidateFeature end

function (::MaxLoss)(ctx::StrikeSelectionContext, sp_K, sc_K, lp_K, lc_K)::Union{Float64,Nothing}
    credit = EntryCredit()(ctx, sp_K, sc_K, lp_K, lc_K)
    credit === nothing && return nothing
    spot = ctx.surface.spot
    put_spread_width = (Float64(sp_K) - Float64(lp_K)) / spot
    call_spread_width = (Float64(lc_K) - Float64(sc_K)) / spot
    max_loss = max(put_spread_width, call_spread_width) - credit
    max_loss <= 0.0 && return nothing
    return max_loss
end

"""Credit-to-max-loss ratio (ROI at entry)."""
struct CreditToMaxLoss <: CandidateFeature end

function (::CreditToMaxLoss)(ctx::StrikeSelectionContext, sp_K, sc_K, lp_K, lc_K)::Union{Float64,Nothing}
    credit = EntryCredit()(ctx, sp_K, sc_K, lp_K, lc_K)
    credit === nothing && return nothing
    ml = MaxLoss()(ctx, sp_K, sc_K, lp_K, lc_K)
    ml === nothing && return nothing
    return credit / ml
end

# =============================================================================
# Default candidate feature set
# =============================================================================

function default_candidate_features(; rate::Float64=0.0, div_yield::Float64=0.0)
    CandidateFeature[
        ShortPutDelta(; rate, div_yield),
        ShortCallDelta(; rate, div_yield),
        EntryCredit(),
        MaxLoss(),
    ]
end

const DEFAULT_CANDIDATE_FEATURES = default_candidate_features()
