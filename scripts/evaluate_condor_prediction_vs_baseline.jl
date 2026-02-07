# Evaluate how well ML predictions recover condor quality
# against both a grid oracle and a strike-exhaustive super-oracle baseline.


using Pkg
Pkg.activate(@__DIR__)

using VolSurfaceAnalysis
using Dates
using CSV, DataFrames
using Statistics
using Flux
using Printf
using BSON

# =============================================================================
# Configuration
# =============================================================================
const UNDERLYING_SYMBOL = "SPXW"
const SPOT_SYMBOL = "SPY"         # Proxy for SPX index spot
const SPOT_MULTIPLIER = 10.0      # SPX ≈ SPY × 10

# Evaluation period defaults (override with CLI args: start end [model_path] [model_mode] [symbol])
const DEFAULT_EVAL_START = Date(2025, 2, 7)
const DEFAULT_EVAL_END = Date(2025, 8, 7)
const ENTRY_TIME_ET = Time(10, 0)

# Strategy and data settings
const EXPIRY_INTERVAL = Day(1)
const RISK_FREE_RATE = 0.045
const DIV_YIELD = 0.013
const MIN_VOLUME = 0
const SPREAD_LAMBDA = 0.0
const SPOT_HISTORY_LOOKBACK_DAYS = 5
const USE_LOGSIG = true

# Condor policy (kept aligned with deployment)
const TARGET_MAX_LOSS = nothing  # Not used with :roi objective
const MIN_DELTA_GAP = 0.01
const PREFER_SYMMETRIC_WINGS = false
const CONSTRAINED_SUPER_MAX_LOSS_TOL = 10.0

# Candidate scoring policy (used when evaluating score-model checkpoints)
const SCORE_WING_OBJECTIVE = :roi
const SCORE_MAX_LOSS_MIN = 50.0
const SCORE_MAX_LOSS_MAX = 300.0
const SCORE_MIN_CREDIT = 1.0
const SCORE_DELTA_GRID = collect(0.05:0.015:0.35)
const SCORE_MAX_CANDIDATES_PER_DAY = 400

# Delta scaling bounds used by the model
const MIN_DELTA = 0.05f0
const MAX_DELTA = 0.35f0

# Constant baseline condor deltas
const BASELINE_PUT_DELTA = 0.16
const BASELINE_CALL_DELTA = 0.16

# Paths
const RUN_ID = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
const RUN_DIR = joinpath(@__DIR__, "runs", "condor_prediction_eval_$(RUN_ID)")
const LATEST_DIR = joinpath(@__DIR__, "latest_runs", "condor_prediction_eval")
const DEFAULT_MODEL_PATH = joinpath(@__DIR__, "latest_runs", "ml_strike_selector", "strike_selector.bson")


# =============================================================================
# Helpers
# =============================================================================

function build_entry_timestamps(dates::Vector{Date}, entry_time::Time)::Vector{DateTime}
    ts = DateTime[]
    for date in dates
        push!(ts, et_to_utc(date, entry_time))
    end
    return sort(ts)
end

function load_minute_spots(
    start_date::Date,
    end_date::Date;
    lookback_days::Union{Nothing,Int}=SPOT_HISTORY_LOOKBACK_DAYS,
    symbol::String=SPOT_SYMBOL,
    multiplier::Float64=SPOT_MULTIPLIER
)::Dict{DateTime,Float64}
    all_dates = available_polygon_dates(DEFAULT_STORE, symbol)
    isempty(all_dates) && error("No spot dates found for $symbol")

    min_date = lookback_days === nothing ? minimum(all_dates) : start_date - Day(lookback_days)
    filtered_dates = filter(d -> d >= min_date && d <= end_date, all_dates)

    spots = Dict{DateTime,Float64}()
    for d in filtered_dates
        path = polygon_spot_path(DEFAULT_STORE, d, symbol)
        isfile(path) || continue
        dict = read_polygon_spot_prices(path; underlying=symbol)
        merge!(spots, dict)
    end

    if multiplier != 1.0
        for (k, v) in spots
            spots[k] = v * multiplier
        end
    end

    return spots
end

function load_surfaces_and_spots(
    start_date::Date,
    end_date::Date;
    symbol::String=UNDERLYING_SYMBOL,
    spot_symbol::String=SPOT_SYMBOL,
    spot_multiplier::Float64=SPOT_MULTIPLIER,
    entry_time::Time=ENTRY_TIME_ET
)
    println("  Loading dates from $start_date to $end_date...")
    all_dates = available_polygon_dates(DEFAULT_STORE, symbol)
    filtered_dates = filter(d -> d >= start_date && d <= end_date, all_dates)
    isempty(filtered_dates) && error("No dates found in range $start_date to $end_date")
    println("  Found $(length(filtered_dates)) trading days")

    entry_ts = build_entry_timestamps(filtered_dates, entry_time)

    # Load entry spots (from spot proxy symbol, scaled)
    entry_spots = read_polygon_spot_prices_for_timestamps(
        polygon_spot_root(DEFAULT_STORE),
        entry_ts;
        symbol=spot_symbol
    )
    if spot_multiplier != 1.0
        for (k, v) in entry_spots
            entry_spots[k] = v * spot_multiplier
        end
    end
    println("  Loaded $(length(entry_spots)) entry spots (via $spot_symbol × $spot_multiplier)")

    # Build surfaces (options from symbol, spots already scaled)
    path_for_ts = ts -> polygon_options_path(DEFAULT_STORE, Date(ts), symbol)
    read_records = (path; where="") -> read_polygon_option_records(
        path,
        entry_spots;
        where=where,
        min_volume=MIN_VOLUME,
        warn=false,
        spread_lambda=SPREAD_LAMBDA
    )
    surfaces = build_surfaces_for_timestamps(
        entry_ts;
        path_for_timestamp=path_for_ts,
        read_records=read_records
    )
    println("  Built $(length(surfaces)) surfaces")

    # Load settlement spots (need to compute expiry times first)
    expiry_ts = DateTime[]
    for (ts, surface) in surfaces
        expiries = unique(rec.expiry for rec in surface.records)
        tau_target = time_to_expiry(ts + EXPIRY_INTERVAL, ts)
        for exp in expiries
            tau = time_to_expiry(exp, ts)
            if abs(tau - tau_target) < 0.1
                push!(expiry_ts, exp)
            end
        end
    end
    expiry_ts = unique(expiry_ts)

    settlement_spots = read_polygon_spot_prices_for_timestamps(
        polygon_spot_root(DEFAULT_STORE),
        expiry_ts;
        symbol=spot_symbol
    )
    if spot_multiplier != 1.0
        for (k, v) in settlement_spots
            settlement_spots[k] = v * spot_multiplier
        end
    end
    println("  Loaded $(length(settlement_spots)) settlement spots (via $spot_symbol × $spot_multiplier)")

    return surfaces, entry_spots, settlement_spots
end

function build_spot_history_dict(
    timestamps::Vector{DateTime},
    all_spots::Dict{DateTime,Float64};
    lookback_days::Union{Nothing,Int}=SPOT_HISTORY_LOOKBACK_DAYS
)::Dict{DateTime,SpotHistory}
    history_dict = Dict{DateTime,SpotHistory}()
    sorted_pairs = sort(collect(all_spots); by=first)
    isempty(sorted_pairs) && return history_dict

    ts_vec = [p[1] for p in sorted_pairs]
    price_vec = [p[2] for p in sorted_pairs]
    first_ts = ts_vec[1]

    for ts in timestamps
        start_ts = lookback_days === nothing ? first_ts : ts - Day(lookback_days)
        i = searchsortedfirst(ts_vec, start_ts)
        j = searchsortedfirst(ts_vec, ts) - 1
        if j >= i && (j - i + 1) >= 2
            history_dict[ts] = SpotHistory(ts_vec[i:j], price_vec[i:j])
        end
    end
    return history_dict
end

function build_prev_surfaces_dict(
    surfaces::Dict{DateTime,VolatilitySurface};
    symbol::String=UNDERLYING_SYMBOL
)::Dict{DateTime,VolatilitySurface}
    all_option_dates = sort(available_polygon_dates(DEFAULT_STORE, symbol))
    by_date = Dict{Date,DateTime}()
    for ts in keys(surfaces)
        d = Date(ts)
        if !haskey(by_date, d) || ts < by_date[d]
            by_date[d] = ts
        end
    end
    prev_dict = Dict{DateTime,VolatilitySurface}()
    for ts in keys(surfaces)
        d = Date(ts)
        idx = searchsortedlast(all_option_dates, d - Day(1))
        idx < 1 && continue
        prev_date = all_option_dates[idx]
        if haskey(by_date, prev_date)
            prev_dict[ts] = surfaces[by_date[prev_date]]
        end
    end
    return prev_dict
end

function nearest_expiry_and_tau(surface::VolatilitySurface, ts::DateTime, expiry_interval::Period)
    expiries = unique(rec.expiry for rec in surface.records)
    isempty(expiries) && return nothing

    tau_target = time_to_expiry(ts + expiry_interval, ts)
    taus = [time_to_expiry(e, ts) for e in expiries]
    idx = argmin(abs.(taus .- tau_target))
    return expiries[idx], taus[idx]
end

safe_mean(v) = isempty(v) ? missing : mean(v)
safe_diff(a, b) = (ismissing(a) || ismissing(b)) ? missing : (a - b)

function safe_risk_return(
    pnl::Union{Missing,Float64},
    max_loss::Union{Missing,Float64}
)::Union{Missing,Float64}
    if ismissing(pnl) || ismissing(max_loss)
        return missing
    end
    max_loss <= 0 && return missing
    return pnl / max_loss
end

function safe_weighted_roi(
    pnls,
    max_losses
)::Union{Missing,Float64}
    total_pnl = 0.0
    total_risk = 0.0
    for (pnl, risk) in zip(pnls, max_losses)
        if ismissing(pnl) || ismissing(risk)
            continue
        end
        risk <= 0 && continue
        total_pnl += pnl
        total_risk += risk
    end
    return total_risk > 0 ? total_pnl / total_risk : missing
end

function condor_candidate_feature_names()::Vector{String}
    return String[
        "cand_short_put_delta",
        "cand_short_call_delta",
        "cand_short_put_distance",
        "cand_short_call_distance",
        "cand_put_width_norm",
        "cand_call_width_norm",
        "cand_width_asymmetry",
        "cand_inner_width_norm",
        "cand_net_credit_norm",
        "cand_max_loss_norm",
        "cand_entry_roi",
        "cand_short_put_rel_spread",
        "cand_short_call_rel_spread",
        "cand_long_put_rel_spread",
        "cand_long_call_rel_spread",
        "cand_avg_leg_rel_spread",
        "cand_put_implied_move_distance",
        "cand_call_implied_move_distance",
        "cand_put_delta_gap",
        "cand_call_delta_gap"
    ]
end

function feature_names(; use_logsig::Bool=USE_LOGSIG, include_candidate_features::Bool=false)
    base = String[
        # Vol surface (6)
        "atm_iv", "atm_bid_ask_spread", "term_slope",
        "put_skew_25d", "call_skew_25d", "skew_asymmetry",
        # Richer smile (10)
        "put_skew_10d", "call_skew_10d", "put_skew_40d", "call_skew_40d",
        "risk_reversal_25d", "butterfly_25d",
        "smile_slope_put", "smile_slope_call",
        "wing_convexity_put", "wing_convexity_call",
        # Spread by moneyness (6)
        "spread_10d_put", "spread_25d_put", "spread_40d_put",
        "spread_10d_call", "spread_25d_call", "spread_40d_call",
        # Surface dynamics (3)
        "delta_atm_iv_1d", "delta_skew_1d", "delta_term_slope_1d",
        # Volume (3)
        "atm_volume_scaled", "total_volume_scaled", "volume_weighted_spread",
        # Market context (8)
        "spot_scaled", "spot_return_1d", "spot_return_5d", "tau",
        "realized_vol_5d", "iv_rv_ratio", "avg_bid_ask_spread", "n_strikes_scaled"
    ]
    # Use pruned logsig dimension (dead features removed)
    if use_logsig
        n_path = pruned_logsig_dim()
        all_indices = setdiff(1:path_feature_dim(; use_logsig=true), LOGSIG_DEAD_INDICES)
        path = ["logsig_$(i)" for i in all_indices]
    else
        n_path = path_feature_dim(; use_logsig=false)
        path = ["sig_$(i)" for i in 1:n_path]
    end
    names = vcat(base, path)
    if include_candidate_features
        names = vcat(names, condor_candidate_feature_names())
    end
    return names
end

function build_feature_stats_df(
    X_raw::Matrix{Float64},
    X_norm::Matrix{Float64};
    use_logsig::Bool=USE_LOGSIG,
    include_candidate_features::Bool=false
)::DataFrame
    n_features = size(X_raw, 1)
    n_obs = size(X_raw, 2)
    n_obs == size(X_norm, 2) || error("Raw and normalized feature matrices must align by column")

    names = feature_names(; use_logsig=use_logsig, include_candidate_features=include_candidate_features)
    n_features == length(names) || error("Feature name count mismatch: $(length(names)) vs $n_features")

    raw_mean = Float64[]
    raw_std = Float64[]
    raw_min = Float64[]
    raw_max = Float64[]
    raw_pct_zero = Float64[]
    raw_pct_near_zero = Float64[]
    norm_mean = Float64[]
    norm_std = Float64[]
    norm_abs_mean = Float64[]
    constant_raw = Bool[]
    near_constant_raw = Bool[]
    shifted_norm_mean = Bool[]
    off_unit_norm_std = Bool[]

    for i in 1:n_features
        rv = collect(@view X_raw[i, :])
        nv = collect(@view X_norm[i, :])

        rmean = mean(rv)
        rstd = std(rv)
        nmean = mean(nv)
        nstd = std(nv)
        abs_nmean = abs(nmean)

        push!(raw_mean, rmean)
        push!(raw_std, rstd)
        push!(raw_min, minimum(rv))
        push!(raw_max, maximum(rv))
        push!(raw_pct_zero, count(x -> x == 0.0, rv) / n_obs)
        push!(raw_pct_near_zero, count(x -> abs(x) <= 1e-8, rv) / n_obs)
        push!(norm_mean, nmean)
        push!(norm_std, nstd)
        push!(norm_abs_mean, abs_nmean)
        push!(constant_raw, rstd <= 1e-10)
        push!(near_constant_raw, rstd <= 1e-4)
        push!(shifted_norm_mean, abs_nmean > 0.5)
        push!(off_unit_norm_std, abs(nstd - 1.0) > 0.5)
    end

    return DataFrame(
        FeatureIndex=1:n_features,
        FeatureName=names,
        RawMean=raw_mean,
        RawStd=raw_std,
        RawMin=raw_min,
        RawMax=raw_max,
        RawPctZero=raw_pct_zero,
        RawPctNearZero=raw_pct_near_zero,
        NormMean=norm_mean,
        NormStd=norm_std,
        NormAbsMean=norm_abs_mean,
        IsConstantRaw=constant_raw,
        IsNearConstantRaw=near_constant_raw,
        IsShiftedNormMean=shifted_norm_mean,
        IsOffUnitNormStd=off_unit_norm_std
    )
end

function build_condor_ctx(
    surface::VolatilitySurface,
    expiry::DateTime,
    tau::Float64
)::Union{Nothing,NamedTuple}
    recs = filter(r -> r.expiry == expiry, surface.records)
    isempty(recs) && return nothing

    put_strikes = sort(unique(r.strike for r in recs if r.option_type == Put))
    call_strikes = sort(unique(r.strike for r in recs if r.option_type == Call))
    (isempty(put_strikes) || isempty(call_strikes)) && return nothing

    return (
        surface=surface,
        expiry=expiry,
        tau=tau,
        recs=recs,
        put_strikes=put_strikes,
        call_strikes=call_strikes
    )
end

function find_rec_by_strike(recs::Vector{OptionRecord}, K::Float64)::Union{Nothing,OptionRecord}
    for r in recs
        r.strike == K && return r
    end
    return nothing
end

function condor_metrics_from_strikes(
    ctx,
    settlement_spot::Float64,
    short_put_K::Float64,
    short_call_K::Float64,
    long_put_K::Float64,
    long_call_K::Float64
)::Union{Nothing,NamedTuple}
    put_recs = filter(r -> r.option_type == Put, ctx.recs)
    call_recs = filter(r -> r.option_type == Call, ctx.recs)

    sp = find_rec_by_strike(put_recs, short_put_K)
    sc = find_rec_by_strike(call_recs, short_call_K)
    lp = find_rec_by_strike(put_recs, long_put_K)
    lc = find_rec_by_strike(call_recs, long_call_K)
    (sp === nothing || sc === nothing || lp === nothing || lc === nothing) && return nothing

    if ismissing(sp.bid_price) || ismissing(sc.bid_price) || ismissing(lp.ask_price) || ismissing(lc.ask_price)
        return nothing
    end

    net_credit = (sp.bid_price + sc.bid_price - lp.ask_price - lc.ask_price) * ctx.surface.spot
    put_payoff = -max(short_put_K - settlement_spot, 0.0) + max(long_put_K - settlement_spot, 0.0)
    call_payoff = -max(settlement_spot - short_call_K, 0.0) + max(settlement_spot - long_call_K, 0.0)
    pnl = put_payoff + call_payoff + net_credit

    width_put = short_put_K - long_put_K
    width_call = long_call_K - short_call_K
    max_loss = max(width_put, width_call) - net_credit

    return (
        pnl=pnl,
        max_loss=max_loss,
        net_credit=net_credit,
        width_put=width_put,
        width_call=width_call
    )
end

function resolve_condor_from_deltas(
    ctx,
    settlement_spot::Float64,
    put_delta::Float64,
    call_delta::Float64;
    target_max_loss::Union{Nothing,Float64}=TARGET_MAX_LOSS,
    wing_objective::Symbol=SCORE_WING_OBJECTIVE,
    max_loss_min::Float64=SCORE_MAX_LOSS_MIN,
    max_loss_max::Float64=SCORE_MAX_LOSS_MAX,
    min_credit::Float64=SCORE_MIN_CREDIT,
    min_delta_gap::Float64=MIN_DELTA_GAP,
    prefer_symmetric::Bool=PREFER_SYMMETRIC_WINGS,
    rate::Float64=RISK_FREE_RATE,
    div_yield::Float64=DIV_YIELD
)::Union{Nothing,NamedTuple}
    shorts = VolSurfaceAnalysis._delta_strangle_strikes_asymmetric(
        ctx,
        put_delta,
        call_delta;
        rate=rate,
        div_yield=div_yield
    )
    shorts === nothing && return nothing
    short_put_K, short_call_K = shorts

    wings = VolSurfaceAnalysis._condor_wings_by_objective(
        ctx,
        short_put_K,
        short_call_K;
        objective=wing_objective,
        target_max_loss=target_max_loss,
        max_loss_min=max_loss_min,
        max_loss_max=max_loss_max,
        min_credit=min_credit,
        rate=rate,
        div_yield=div_yield,
        min_delta_gap=min_delta_gap,
        prefer_symmetric=prefer_symmetric,
        debug=false
    )
    wings === nothing && return nothing
    long_put_K, long_call_K = wings

    metrics = condor_metrics_from_strikes(
        ctx,
        settlement_spot,
        short_put_K,
        short_call_K,
        long_put_K,
        long_call_K
    )
    metrics === nothing && return nothing

    return (
        short_put_K=short_put_K,
        short_call_K=short_call_K,
        long_put_K=long_put_K,
        long_call_K=long_call_K,
        pnl=metrics.pnl,
        max_loss=metrics.max_loss
    )
end

function best_put_side_component(
    put_recs::Vector{OptionRecord},
    spot::Float64,
    settlement_spot::Float64
)::Union{Nothing,NamedTuple}
    best_component = -Inf
    best_short_put = 0.0
    best_long_put = 0.0
    found = false

    for short_put in put_recs
        short_put.strike < spot || continue
        ismissing(short_put.bid_price) && continue
        short_bid = short_put.bid_price

        for long_put in put_recs
            long_put.strike < short_put.strike || continue
            ismissing(long_put.ask_price) && continue
            long_ask = long_put.ask_price

            payoff = -max(short_put.strike - settlement_spot, 0.0) +
                     max(long_put.strike - settlement_spot, 0.0)
            entry_credit = (short_bid - long_ask) * spot
            component = payoff + entry_credit

            if component > best_component
                best_component = component
                best_short_put = short_put.strike
                best_long_put = long_put.strike
                found = true
            end
        end
    end

    found || return nothing
    return (
        component=best_component,
        short_put_K=best_short_put,
        long_put_K=best_long_put
    )
end

function best_call_side_component(
    call_recs::Vector{OptionRecord},
    spot::Float64,
    settlement_spot::Float64
)::Union{Nothing,NamedTuple}
    best_component = -Inf
    best_short_call = 0.0
    best_long_call = 0.0
    found = false

    for short_call in call_recs
        short_call.strike > spot || continue
        ismissing(short_call.bid_price) && continue
        short_bid = short_call.bid_price

        for long_call in call_recs
            long_call.strike > short_call.strike || continue
            ismissing(long_call.ask_price) && continue
            long_ask = long_call.ask_price

            payoff = -max(settlement_spot - short_call.strike, 0.0) +
                     max(settlement_spot - long_call.strike, 0.0)
            entry_credit = (short_bid - long_ask) * spot
            component = payoff + entry_credit

            if component > best_component
                best_component = component
                best_short_call = short_call.strike
                best_long_call = long_call.strike
                found = true
            end
        end
    end

    found || return nothing
    return (
        component=best_component,
        short_call_K=best_short_call,
        long_call_K=best_long_call
    )
end

function find_super_oracle_condor(
    ctx,
    settlement_spot::Float64
)::Union{Nothing,NamedTuple}
    put_recs = filter(r -> r.option_type == Put, ctx.recs)
    call_recs = filter(r -> r.option_type == Call, ctx.recs)
    isempty(put_recs) && return nothing
    isempty(call_recs) && return nothing

    put_side = best_put_side_component(put_recs, ctx.surface.spot, settlement_spot)
    call_side = best_call_side_component(call_recs, ctx.surface.spot, settlement_spot)
    (put_side === nothing || call_side === nothing) && return nothing

    super_pnl = put_side.component + call_side.component
    return (
        SuperShortPutK=put_side.short_put_K,
        SuperLongPutK=put_side.long_put_K,
        SuperShortCallK=call_side.short_call_K,
        SuperLongCallK=call_side.long_call_K,
        SuperPnL=super_pnl
    )
end

function find_constrained_super_oracle_condor(
    ctx,
    settlement_spot::Float64;
    target_max_loss::Union{Nothing,Float64}=TARGET_MAX_LOSS,
    max_loss_tol::Float64=CONSTRAINED_SUPER_MAX_LOSS_TOL,
    wing_objective::Symbol=SCORE_WING_OBJECTIVE,
    max_loss_min::Float64=SCORE_MAX_LOSS_MIN,
    max_loss_max::Float64=SCORE_MAX_LOSS_MAX,
    min_credit::Float64=SCORE_MIN_CREDIT,
    min_delta_gap::Float64=MIN_DELTA_GAP,
    prefer_symmetric::Bool=PREFER_SYMMETRIC_WINGS,
    rate::Float64=RISK_FREE_RATE,
    div_yield::Float64=DIV_YIELD
)::Union{Nothing,NamedTuple}
    put_recs = filter(r -> r.option_type == Put, ctx.recs)
    call_recs = filter(r -> r.option_type == Call, ctx.recs)

    short_put_candidates = filter(
        r -> r.strike < ctx.surface.spot && !ismissing(r.bid_price),
        put_recs
    )
    short_call_candidates = filter(
        r -> r.strike > ctx.surface.spot && !ismissing(r.bid_price),
        call_recs
    )
    (isempty(short_put_candidates) || isempty(short_call_candidates)) && return nothing

    best = nothing
    best_pnl = -Inf

    for sp in short_put_candidates
        for sc in short_call_candidates
            wings = VolSurfaceAnalysis._condor_wings_by_objective(
                ctx,
                sp.strike,
                sc.strike;
                objective=wing_objective,
                target_max_loss=target_max_loss,
                max_loss_min=max_loss_min,
                max_loss_max=max_loss_max,
                min_credit=min_credit,
                rate=rate,
                div_yield=div_yield,
                min_delta_gap=min_delta_gap,
                prefer_symmetric=prefer_symmetric,
                debug=false
            )
            wings === nothing && continue
            lpK, lcK = wings

            metrics = condor_metrics_from_strikes(
                ctx,
                settlement_spot,
                sp.strike,
                sc.strike,
                lpK,
                lcK
            )
            metrics === nothing && continue

            # For ROI mode, constrain max_loss within tolerance of the range midpoint
            if max_loss_min > 0.0 && max_loss_max < Inf
                mid_max_loss = (max_loss_min + max_loss_max) / 2.0
                abs(metrics.max_loss - mid_max_loss) <= max_loss_tol || continue
            end

            if metrics.pnl > best_pnl
                best_pnl = metrics.pnl
                best = (
                    short_put_K=sp.strike,
                    short_call_K=sc.strike,
                    long_put_K=lpK,
                    long_call_K=lcK,
                    pnl=metrics.pnl,
                    max_loss=metrics.max_loss
                )
            end
        end
    end

    return best
end

function parse_args()
    eval_start = DEFAULT_EVAL_START
    eval_end = DEFAULT_EVAL_END
    model_path = DEFAULT_MODEL_PATH
    model_mode_override = nothing
    symbol = UNDERLYING_SYMBOL

    if length(ARGS) >= 1
        eval_start = Date(ARGS[1])
    end
    if length(ARGS) >= 2
        eval_end = Date(ARGS[2])
    end
    if length(ARGS) >= 3
        model_path = ARGS[3]
    end
    if length(ARGS) >= 4
        mode_arg = lowercase(String(ARGS[4]))
        if mode_arg == "delta"
            model_mode_override = :delta
        elseif mode_arg == "score"
            model_mode_override = :score
        elseif mode_arg == "auto"
            model_mode_override = nothing
        else
            error("Unknown model mode '$mode_arg'. Use one of: auto, delta, score")
        end
    end
    if length(ARGS) >= 5
        symbol = uppercase(String(ARGS[5]))
    end

    return eval_start, eval_end, model_path, model_mode_override, symbol
end

function main()
    eval_start, eval_end, model_path, model_mode_override, symbol = parse_args()
    isfile(model_path) || error("Model file not found: $model_path")
    mkpath(RUN_DIR)

    println("=" ^ 90)
    println("CONDOR PREDICTION QUALITY VS CONSTANT BASELINE")
    println("=" ^ 90)
    println("Underlying: $symbol (spot via $SPOT_SYMBOL × $SPOT_MULTIPLIER)")
    println("Eval period: $eval_start to $eval_end")
    println("Model path: $model_path")
    println("Wing policy: objective=$SCORE_WING_OBJECTIVE, max_loss=[$SCORE_MAX_LOSS_MIN,$SCORE_MAX_LOSS_MAX], min_credit=$SCORE_MIN_CREDIT, min_delta_gap=$MIN_DELTA_GAP")
    println("Score candidate policy: objective=$SCORE_WING_OBJECTIVE, max_loss=[$SCORE_MAX_LOSS_MIN,$SCORE_MAX_LOSS_MAX], min_credit=$SCORE_MIN_CREDIT, max_candidates/day=$SCORE_MAX_CANDIDATES_PER_DAY")
    println("Constrained super-oracle max-loss tolerance: +/-$CONSTRAINED_SUPER_MAX_LOSS_TOL")
    println("Baseline deltas: put=$BASELINE_PUT_DELTA, call=$BASELINE_CALL_DELTA")
    println("Output directory: $RUN_DIR")
    println()

    println("PHASE 1: Load Model")
    println("-" ^ 50)
    model_data = BSON.load(model_path)
    model = model_data[:model]
    feature_means = model_data[:feature_means]
    feature_stds = model_data[:feature_stds]
    state_input_dim = n_features(; use_logsig=USE_LOGSIG)
    score_input_dim = n_condor_scoring_features(; use_logsig=USE_LOGSIG)
    length(feature_means) == length(feature_stds) || error("Model normalization vectors must have equal length")

    detected_model_mode = if length(feature_means) == state_input_dim
        :delta
    elseif length(feature_means) == score_input_dim
        :score
    else
        error("Model normalization dimension $(length(feature_means)) not recognized. Expected state dim $state_input_dim or score dim $score_input_dim for use_logsig=$USE_LOGSIG.")
    end

    model_mode = model_mode_override === nothing ? detected_model_mode : model_mode_override
    if model_mode != detected_model_mode
        error("Requested model mode $model_mode but checkpoint normalization dimension indicates $detected_model_mode mode.")
    end
    println("  Loaded model and normalization stats (mode=$model_mode, input_dim=$(length(feature_means)))")
    println()

    println("PHASE 2: Load Evaluation Data")
    println("-" ^ 50)
    surfaces, _, settlement_spots = load_surfaces_and_spots(
        eval_start,
        eval_end;
        symbol=symbol,
        entry_time=ENTRY_TIME_ET
    )
    timestamps = sort(collect(keys(surfaces)))
    println("  Loading minute spot history...")
    minute_spots = load_minute_spots(
        eval_start,
        eval_end;
        lookback_days=SPOT_HISTORY_LOOKBACK_DAYS
    )
    spot_history = build_spot_history_dict(timestamps, minute_spots; lookback_days=SPOT_HISTORY_LOOKBACK_DAYS)
    println("  Built spot history for $(length(spot_history)) timestamps")

    prev_surfaces = build_prev_surfaces_dict(surfaces; symbol=symbol)
    println("  Built prev-day surface map for $(length(prev_surfaces))/$(length(surfaces)) timestamps")
    println()

    println("PHASE 3: Score ML vs Grid Oracle vs Super Oracle vs Constrained Super Oracle vs Baseline")
    println("-" ^ 50)
    rows = NamedTuple[]
    feature_vectors_raw = Vector{Vector{Float64}}()
    feature_vectors_norm = Vector{Vector{Float64}}()
    n_skipped = 0

    Flux.testmode!(model)

    for (i, ts) in enumerate(timestamps)
        surface = surfaces[ts]

        expiry_info = nearest_expiry_and_tau(surface, ts, EXPIRY_INTERVAL)
        if expiry_info === nothing
            n_skipped += 1
            continue
        end
        expiry, tau = expiry_info
        tau <= 0.0 && (n_skipped += 1; continue)

        ctx = build_condor_ctx(surface, expiry, tau)
        ctx === nothing && (n_skipped += 1; continue)

        settlement = get(settlement_spots, expiry, nothing)
        settlement === nothing && (n_skipped += 1; continue)

        hist = get(spot_history, ts, nothing)
        prev_surf = get(prev_surfaces, ts, nothing)
        feats = extract_features(surface, tau; spot_history=hist, use_logsig=USE_LOGSIG, prev_surface=prev_surf)
        feats === nothing && (n_skipped += 1; continue)

        state_x = features_to_vector(feats)
        x_for_stats = Float64[]
        x_norm_for_stats = Float64[]
        pred_put_delta = missing
        pred_call_delta = missing
        pred_size = missing
        pred_score = missing
        pred_condor = nothing

        if model_mode == :delta
            x_norm = (state_x .- feature_means) ./ feature_stds
            raw = model(reshape(x_norm, :, 1))
            pred_deltas = scale_deltas(raw[1:2, :]; min_delta=MIN_DELTA, max_delta=MAX_DELTA)
            pred_put_delta = Float64(pred_deltas[1])
            pred_call_delta = Float64(pred_deltas[2])
            pred_size = size(raw, 1) >= 3 ? Float64(raw[3, 1]) : missing
            pred_condor = resolve_condor_from_deltas(
                ctx,
                settlement,
                pred_put_delta,
                pred_call_delta
            )
            x_for_stats = Float64.(state_x)
            x_norm_for_stats = Float64.(x_norm)
        else
            candidates = enumerate_condor_candidates(
                ctx;
                delta_grid=SCORE_DELTA_GRID,
                max_candidates=SCORE_MAX_CANDIDATES_PER_DAY,
                wing_delta_abs=nothing,
                target_max_loss=TARGET_MAX_LOSS,
                wing_objective=SCORE_WING_OBJECTIVE,
                max_loss_min=SCORE_MAX_LOSS_MIN,
                max_loss_max=SCORE_MAX_LOSS_MAX,
                min_credit=SCORE_MIN_CREDIT,
                min_delta_gap=MIN_DELTA_GAP,
                prefer_symmetric=PREFER_SYMMETRIC_WINGS,
                rate=RISK_FREE_RATE,
                div_yield=DIV_YIELD
            )
            isempty(candidates) && (n_skipped += 1; continue)

            candidate_features = Vector{Float32}[]
            valid_candidates = NamedTuple[]
            for candidate in candidates
                x_candidate = condor_scoring_feature_vector(
                    state_x,
                    ctx,
                    candidate;
                    rate=RISK_FREE_RATE,
                    div_yield=DIV_YIELD
                )
                x_candidate === nothing && continue
                push!(candidate_features, x_candidate)
                push!(valid_candidates, candidate)
            end
            isempty(candidate_features) && (n_skipped += 1; continue)

            X_candidates = reduce(hcat, candidate_features)
            X_candidates_norm = (X_candidates .- feature_means) ./ feature_stds
            raw_scores = model(X_candidates_norm)
            scores = vec(raw_scores)
            isempty(scores) && (n_skipped += 1; continue)

            best_idx = argmax(scores)
            best_candidate = valid_candidates[best_idx]

            pred_put_delta = best_candidate.short_put_delta
            pred_call_delta = best_candidate.short_call_delta
            pred_score = Float64(scores[best_idx])
            pred_metrics = condor_metrics_from_strikes(
                ctx,
                settlement,
                best_candidate.short_put_K,
                best_candidate.short_call_K,
                best_candidate.long_put_K,
                best_candidate.long_call_K
            )
            pred_metrics !== nothing && (pred_condor = (
                short_put_K=best_candidate.short_put_K,
                short_call_K=best_candidate.short_call_K,
                long_put_K=best_candidate.long_put_K,
                long_call_K=best_candidate.long_call_K,
                pnl=pred_metrics.pnl,
                max_loss=pred_metrics.max_loss
            ))

            x_for_stats = Float64.(X_candidates[:, best_idx])
            x_norm_for_stats = Float64.(X_candidates_norm[:, best_idx])
        end

        grid_oracle = find_optimal_condor_deltas(
            surface, settlement;
            wing_delta_abs=nothing,
            target_max_loss=TARGET_MAX_LOSS,
            wing_objective=SCORE_WING_OBJECTIVE,
            max_loss_min=SCORE_MAX_LOSS_MIN,
            max_loss_max=SCORE_MAX_LOSS_MAX,
            min_credit=SCORE_MIN_CREDIT,
            min_delta_gap=MIN_DELTA_GAP,
            prefer_symmetric=PREFER_SYMMETRIC_WINGS,
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            expiry_interval=EXPIRY_INTERVAL
        )
        if grid_oracle === nothing
            n_skipped += 1
            continue
        end
        grid_best_put_delta, grid_best_call_delta, grid_best_pnl = grid_oracle

        super_oracle = find_super_oracle_condor(ctx, settlement)
        if super_oracle === nothing
            n_skipped += 1
            continue
        end
        super_pnl = super_oracle.SuperPnL

        constrained_super = find_constrained_super_oracle_condor(
            ctx,
            settlement
        )

        grid_condor = resolve_condor_from_deltas(
            ctx,
            settlement,
            grid_best_put_delta,
            grid_best_call_delta
        )
        grid_max_loss = grid_condor === nothing ? missing : grid_condor.max_loss

        pred_pnl = pred_condor === nothing ? missing : pred_condor.pnl
        pred_max_loss = pred_condor === nothing ? missing : pred_condor.max_loss

        base_condor = resolve_condor_from_deltas(
            ctx,
            settlement,
            BASELINE_PUT_DELTA,
            BASELINE_CALL_DELTA
        )
        base_pnl = base_condor === nothing ? missing : base_condor.pnl
        base_max_loss = base_condor === nothing ? missing : base_condor.max_loss

        constrained_super_pnl = constrained_super === nothing ? missing : constrained_super.pnl
        constrained_super_max_loss = constrained_super === nothing ? missing : constrained_super.max_loss

        super_metrics = condor_metrics_from_strikes(
            ctx,
            settlement,
            super_oracle.SuperShortPutK,
            super_oracle.SuperShortCallK,
            super_oracle.SuperLongPutK,
            super_oracle.SuperLongCallK
        )
        super_max_loss = super_metrics === nothing ? missing : super_metrics.max_loss

        pred_regret_grid = ismissing(pred_pnl) ? missing : grid_best_pnl - pred_pnl
        base_regret_grid = ismissing(base_pnl) ? missing : grid_best_pnl - base_pnl
        pred_regret_super = ismissing(pred_pnl) ? missing : super_pnl - pred_pnl
        base_regret_super = ismissing(base_pnl) ? missing : super_pnl - base_pnl
        grid_regret_super = super_pnl - grid_best_pnl
        pred_regret_constrained = (ismissing(pred_pnl) || ismissing(constrained_super_pnl)) ? missing : (constrained_super_pnl - pred_pnl)
        base_regret_constrained = (ismissing(base_pnl) || ismissing(constrained_super_pnl)) ? missing : (constrained_super_pnl - base_pnl)
        grid_regret_constrained = ismissing(constrained_super_pnl) ? missing : (constrained_super_pnl - grid_best_pnl)
        constrained_regret_super = ismissing(constrained_super_pnl) ? missing : (super_pnl - constrained_super_pnl)
        pred_beats_base = (ismissing(pred_pnl) || ismissing(base_pnl)) ? missing : (pred_pnl > base_pnl)
        grid_roi = safe_risk_return(grid_best_pnl, grid_max_loss)
        super_roi = safe_risk_return(super_pnl, super_max_loss)
        constrained_super_roi = safe_risk_return(constrained_super_pnl, constrained_super_max_loss)
        pred_roi = safe_risk_return(pred_pnl, pred_max_loss)
        base_roi = safe_risk_return(base_pnl, base_max_loss)

        push!(feature_vectors_raw, x_for_stats)
        push!(feature_vectors_norm, x_norm_for_stats)

        push!(rows, (
            Timestamp=ts,
            Expiry=expiry,
            SettlementSpot=settlement,
            GridBestPutDelta=grid_best_put_delta,
            GridBestCallDelta=grid_best_call_delta,
            GridBestPnL=grid_best_pnl,
            SuperShortPutK=super_oracle.SuperShortPutK,
            SuperLongPutK=super_oracle.SuperLongPutK,
            SuperShortCallK=super_oracle.SuperShortCallK,
            SuperLongCallK=super_oracle.SuperLongCallK,
            SuperPnL=super_pnl,
            SuperMaxLoss=super_max_loss,
            ConstrainedSuperShortPutK=(constrained_super === nothing ? missing : constrained_super.short_put_K),
            ConstrainedSuperLongPutK=(constrained_super === nothing ? missing : constrained_super.long_put_K),
            ConstrainedSuperShortCallK=(constrained_super === nothing ? missing : constrained_super.short_call_K),
            ConstrainedSuperLongCallK=(constrained_super === nothing ? missing : constrained_super.long_call_K),
            ConstrainedSuperPnL=constrained_super_pnl,
            ConstrainedSuperMaxLoss=constrained_super_max_loss,
            PredPutDelta=pred_put_delta,
            PredCallDelta=pred_call_delta,
            PredSize=pred_size,
            PredScore=pred_score,
            PredPnL=pred_pnl,
            PredMaxLoss=pred_max_loss,
            PredROI=pred_roi,
            BasePutDelta=BASELINE_PUT_DELTA,
            BaseCallDelta=BASELINE_CALL_DELTA,
            BasePnL=base_pnl,
            BaseMaxLoss=base_max_loss,
            BaseROI=base_roi,
            GridBestMaxLoss=grid_max_loss,
            GridBestROI=grid_roi,
            SuperROI=super_roi,
            ConstrainedSuperROI=constrained_super_roi,
            PredRegretVsGrid=pred_regret_grid,
            BaseRegretVsGrid=base_regret_grid,
            PredRegretVsSuper=pred_regret_super,
            BaseRegretVsSuper=base_regret_super,
            GridRegretVsSuper=grid_regret_super,
            PredRegretVsConstrainedSuper=pred_regret_constrained,
            BaseRegretVsConstrainedSuper=base_regret_constrained,
            GridRegretVsConstrainedSuper=grid_regret_constrained,
            ConstrainedRegretVsSuper=constrained_regret_super,
            PredBeatsBase=pred_beats_base
        ))

        if i % 25 == 0 || i == length(timestamps)
            println("  Processed $i / $(length(timestamps)) (kept: $(length(rows)), skipped: $n_skipped)")
        end
    end
    println()

    isempty(rows) && error("No valid evaluation rows produced")

    details_df = DataFrame(rows)
    details_path = joinpath(RUN_DIR, "condor_prediction_details.csv")
    CSV.write(details_path, details_df)
    println("Details saved to: $details_path")

    valid_pred = .!ismissing.(details_df.PredPnL)
    valid_base = .!ismissing.(details_df.BasePnL)
    valid_both = valid_pred .& valid_base

    pred_pnls = collect(skipmissing(details_df.PredPnL))
    base_pnls = collect(skipmissing(details_df.BasePnL))
    pred_rois = collect(skipmissing(details_df.PredROI))
    base_rois = collect(skipmissing(details_df.BaseROI))
    grid_rois = collect(skipmissing(details_df.GridBestROI))
    super_rois = collect(skipmissing(details_df.SuperROI))
    constrained_super_rois = collect(skipmissing(details_df.ConstrainedSuperROI))
    valid_super = .!ismissing.(details_df.SuperPnL)
    valid_constrained_super = .!ismissing.(details_df.ConstrainedSuperPnL)

    pred_regrets_grid = collect(skipmissing(details_df.PredRegretVsGrid))
    base_regrets_grid = collect(skipmissing(details_df.BaseRegretVsGrid))
    pred_regrets_super = collect(skipmissing(details_df.PredRegretVsSuper))
    base_regrets_super = collect(skipmissing(details_df.BaseRegretVsSuper))
    grid_regrets_super = collect(skipmissing(details_df.GridRegretVsSuper))
    super_pnls = collect(skipmissing(details_df.SuperPnL))
    constrained_super_pnls = collect(skipmissing(details_df.ConstrainedSuperPnL))
    pred_regrets_constrained = collect(skipmissing(details_df.PredRegretVsConstrainedSuper))
    base_regrets_constrained = collect(skipmissing(details_df.BaseRegretVsConstrainedSuper))
    grid_regrets_constrained = collect(skipmissing(details_df.GridRegretVsConstrainedSuper))
    constrained_regrets_super = collect(skipmissing(details_df.ConstrainedRegretVsSuper))
    grid_max_losses = collect(skipmissing(details_df.GridBestMaxLoss))
    super_max_losses = collect(skipmissing(details_df.SuperMaxLoss))
    constrained_super_max_losses = collect(skipmissing(details_df.ConstrainedSuperMaxLoss))
    beat_flags = [details_df.PredPnL[i] > details_df.BasePnL[i] for i in eachindex(valid_both) if valid_both[i]]
    grid_weighted_roi = safe_weighted_roi(details_df.GridBestPnL, details_df.GridBestMaxLoss)
    super_weighted_roi = safe_weighted_roi(details_df.SuperPnL, details_df.SuperMaxLoss)
    constrained_super_weighted_roi = safe_weighted_roi(details_df.ConstrainedSuperPnL, details_df.ConstrainedSuperMaxLoss)
    pred_weighted_roi = safe_weighted_roi(details_df.PredPnL, details_df.PredMaxLoss)
    base_weighted_roi = safe_weighted_roi(details_df.BasePnL, details_df.BaseMaxLoss)

    isempty(feature_vectors_raw) && error("No feature vectors captured for feature stats")
    X_raw = reduce(hcat, feature_vectors_raw)
    X_norm = reduce(hcat, feature_vectors_norm)
    feature_stats_df = build_feature_stats_df(
        X_raw,
        X_norm;
        use_logsig=USE_LOGSIG,
        include_candidate_features=(model_mode == :score)
    )
    flagged_features = filter(
        row ->
            row.IsConstantRaw ||
            row.IsNearConstantRaw ||
            row.RawPctNearZero >= 0.95 ||
            row.IsShiftedNormMean ||
            row.IsOffUnitNormStd,
        feature_stats_df
    )

    delta_abs_err_put = abs.(details_df.PredPutDelta .- details_df.GridBestPutDelta)
    delta_abs_err_call = abs.(details_df.PredCallDelta .- details_df.GridBestCallDelta)
    delta_mae_put = safe_mean(delta_abs_err_put)
    delta_mae_call = safe_mean(delta_abs_err_call)
    delta_rmse_put = sqrt(mean((details_df.PredPutDelta .- details_df.GridBestPutDelta).^2))
    delta_rmse_call = sqrt(mean((details_df.PredCallDelta .- details_df.GridBestCallDelta).^2))
    pred_minus_base_avg_roi = safe_diff(safe_mean(pred_rois), safe_mean(base_rois))
    pred_minus_base_weighted_roi = safe_diff(pred_weighted_roi, base_weighted_roi)

    summary_df = DataFrame(
        Metric = String[
            "underlying_symbol",
            "model_mode",
            "model_input_dim",
            "n_rows",
            "n_with_pred_pnl",
            "n_with_base_pnl",
            "n_with_both_pnl",
            "n_with_super_pnl",
            "n_with_constrained_super_pnl",
            "grid_oracle_avg_pnl",
            "super_oracle_avg_pnl",
            "constrained_super_oracle_avg_pnl",
            "grid_oracle_avg_roi",
            "super_oracle_avg_roi",
            "constrained_super_oracle_avg_roi",
            "super_minus_grid_avg_gap",
            "constrained_minus_grid_avg_gap",
            "super_minus_constrained_avg_gap",
            "pred_avg_pnl",
            "base_avg_pnl",
            "pred_avg_roi",
            "base_avg_roi",
            "pred_minus_base_avg_roi",
            "grid_oracle_weighted_roi",
            "super_oracle_weighted_roi",
            "constrained_super_oracle_weighted_roi",
            "pred_weighted_roi",
            "base_weighted_roi",
            "pred_minus_base_weighted_roi",
            "pred_avg_regret_vs_grid_oracle",
            "base_avg_regret_vs_grid_oracle",
            "pred_avg_regret_vs_super_oracle",
            "base_avg_regret_vs_super_oracle",
            "grid_oracle_avg_regret_vs_super_oracle",
            "pred_avg_regret_vs_constrained_super_oracle",
            "base_avg_regret_vs_constrained_super_oracle",
            "grid_oracle_avg_regret_vs_constrained_super_oracle",
            "constrained_super_avg_regret_vs_super_oracle",
            "grid_oracle_avg_max_loss",
            "super_oracle_avg_max_loss",
            "constrained_super_oracle_avg_max_loss",
            "pred_beats_base_rate",
            "pred_win_rate",
            "base_win_rate",
            "delta_mae_put_vs_grid_oracle",
            "delta_mae_call_vs_grid_oracle",
            "delta_rmse_put_vs_grid_oracle",
            "delta_rmse_call_vs_grid_oracle",
            "n_features",
            "n_constant_raw_features",
            "n_near_constant_raw_features",
            "n_high_zero_raw_features",
            "n_shifted_norm_mean_features",
            "n_off_unit_norm_std_features"
        ],
        Value = Any[
            symbol,
            string(model_mode),
            length(feature_means),
            nrow(details_df),
            count(valid_pred),
            count(valid_base),
            count(valid_both),
            count(valid_super),
            count(valid_constrained_super),
            mean(details_df.GridBestPnL),
            safe_mean(super_pnls),
            safe_mean(constrained_super_pnls),
            safe_mean(grid_rois),
            safe_mean(super_rois),
            safe_mean(constrained_super_rois),
            safe_mean(details_df.GridRegretVsSuper),
            safe_mean(grid_regrets_constrained),
            safe_mean(constrained_regrets_super),
            safe_mean(pred_pnls),
            safe_mean(base_pnls),
            safe_mean(pred_rois),
            safe_mean(base_rois),
            pred_minus_base_avg_roi,
            grid_weighted_roi,
            super_weighted_roi,
            constrained_super_weighted_roi,
            pred_weighted_roi,
            base_weighted_roi,
            pred_minus_base_weighted_roi,
            safe_mean(pred_regrets_grid),
            safe_mean(base_regrets_grid),
            safe_mean(pred_regrets_super),
            safe_mean(base_regrets_super),
            safe_mean(grid_regrets_super),
            safe_mean(pred_regrets_constrained),
            safe_mean(base_regrets_constrained),
            safe_mean(grid_regrets_constrained),
            safe_mean(constrained_regrets_super),
            safe_mean(grid_max_losses),
            safe_mean(super_max_losses),
            safe_mean(constrained_super_max_losses),
            safe_mean(beat_flags),
            safe_mean(pred_pnls .> 0),
            safe_mean(base_pnls .> 0),
            delta_mae_put,
            delta_mae_call,
            delta_rmse_put,
            delta_rmse_call,
            nrow(feature_stats_df),
            count(feature_stats_df.IsConstantRaw),
            count(feature_stats_df.IsNearConstantRaw),
            count(feature_stats_df.RawPctNearZero .>= 0.95),
            count(feature_stats_df.IsShiftedNormMean),
            count(feature_stats_df.IsOffUnitNormStd)
        ]
    )

    summary_path = joinpath(RUN_DIR, "condor_prediction_summary.csv")
    CSV.write(summary_path, summary_df)
    println("Summary saved to: $summary_path")
    feature_stats_path = joinpath(RUN_DIR, "feature_stats.csv")
    CSV.write(feature_stats_path, feature_stats_df)
    println("Feature stats saved to: $feature_stats_path")
    flagged_features_path = joinpath(RUN_DIR, "feature_flags.csv")
    CSV.write(flagged_features_path, flagged_features)
    println("Feature flags saved to: $flagged_features_path")
    println()

    println("=" ^ 90)
    println("RESULT SNAPSHOT")
    println("=" ^ 90)
    @printf("Model mode: %s\n", string(model_mode))
    @printf("Rows kept: %d\n", nrow(details_df))
    @printf("Rows with both PnLs: %d\n", count(valid_both))
    @printf("Rows with constrained super PnL: %d\n", count(valid_constrained_super))
    @printf("Grid oracle avg PnL: %.4f\n", mean(details_df.GridBestPnL))
    @printf("Super oracle avg PnL: %.4f\n", coalesce(safe_mean(super_pnls), NaN))
    @printf("Constrained super avg PnL: %.4f\n", coalesce(safe_mean(constrained_super_pnls), NaN))
    @printf("Grid/Super/Constrained avg ROI: %.4f / %.4f / %.4f\n",
        coalesce(safe_mean(grid_rois), NaN),
        coalesce(safe_mean(super_rois), NaN),
        coalesce(safe_mean(constrained_super_rois), NaN))
    @printf("Pred/Base avg ROI: %.4f / %.4f\n",
        coalesce(safe_mean(pred_rois), NaN),
        coalesce(safe_mean(base_rois), NaN))
    @printf("Pred - Base avg ROI: %.4f\n", coalesce(pred_minus_base_avg_roi, NaN))
    @printf("Pred/Base weighted ROI: %.4f / %.4f\n",
        coalesce(pred_weighted_roi, NaN),
        coalesce(base_weighted_roi, NaN))
    @printf("Pred - Base weighted ROI: %.4f\n", coalesce(pred_minus_base_weighted_roi, NaN))
    @printf("Grid oracle regret vs super: %.4f\n", coalesce(safe_mean(grid_regrets_super), NaN))
    @printf("Grid oracle regret vs constrained super: %.4f\n", coalesce(safe_mean(grid_regrets_constrained), NaN))
    @printf("Pred   avg PnL: %.4f\n", coalesce(safe_mean(pred_pnls), NaN))
    @printf("Base   avg PnL: %.4f\n", coalesce(safe_mean(base_pnls), NaN))
    @printf("Pred regret vs grid oracle: %.4f\n", coalesce(safe_mean(pred_regrets_grid), NaN))
    @printf("Base regret vs grid oracle: %.4f\n", coalesce(safe_mean(base_regrets_grid), NaN))
    @printf("Pred regret vs super oracle: %.4f\n", coalesce(safe_mean(pred_regrets_super), NaN))
    @printf("Base regret vs super oracle: %.4f\n", coalesce(safe_mean(base_regrets_super), NaN))
    @printf("Pred regret vs constrained super: %.4f\n", coalesce(safe_mean(pred_regrets_constrained), NaN))
    @printf("Base regret vs constrained super: %.4f\n", coalesce(safe_mean(base_regrets_constrained), NaN))
    @printf("Grid/Super/Constrained avg max loss: %.3f / %.3f / %.3f\n",
        coalesce(safe_mean(grid_max_losses), NaN),
        coalesce(safe_mean(super_max_losses), NaN),
        coalesce(safe_mean(constrained_super_max_losses), NaN))
    @printf("Pred beats base rate: %.3f\n", coalesce(safe_mean(beat_flags), NaN))
    @printf("Put delta MAE vs grid oracle: %.4f\n", coalesce(delta_mae_put, NaN))
    @printf("Call delta MAE vs grid oracle: %.4f\n", coalesce(delta_mae_call, NaN))
    @printf("Feature flags (constant / near-constant / high-zero / norm-mean-shift / norm-std-off): %d / %d / %d / %d / %d\n",
        count(feature_stats_df.IsConstantRaw),
        count(feature_stats_df.IsNearConstantRaw),
        count(feature_stats_df.RawPctNearZero .>= 0.95),
        count(feature_stats_df.IsShiftedNormMean),
        count(feature_stats_df.IsOffUnitNormStd))
    println("=" ^ 90)

    isdir(LATEST_DIR) && rm(LATEST_DIR; recursive=true)
    cp(RUN_DIR, LATEST_DIR)
    println("Latest run: $LATEST_DIR")
end

main()
