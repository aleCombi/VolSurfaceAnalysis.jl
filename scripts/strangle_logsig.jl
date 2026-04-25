# scripts/strangle_logsig.jl
#
# Tail-day filter / ranking for 1-DTE short strangle (Δp=0.20, Δc=0.05).
# Two methods, selected via METHOD env:
#
#   METHOD=classifier (default) — replaces strangle_logsig_filter_1d.jl
#       GLMNet logistic on (training PnL <= 10th-percentile) tail label,
#       skip when P(loss) > τ. Sweep τ ∈ {0.30, 0.50, 0.70, 0.90}.
#       Defaults: ENTRY=14:00, EXPIRY=Hour(2), MAX_TAU_DAYS=0.5,
#       SpotMinuteLogSig(lookback=5h, min_points=150).
#
#   METHOD=ranking — replaces strangle_logsig_ranking_1d.jl
#       Ridge regression on PnL, skip days whose predicted PnL is
#       in the bottom-q% of the test fold's predictions. Sweep q ∈ {10, 20, 30, 50}.
#       Defaults: ENTRY=12:00, EXPIRY=Day(1), MAX_TAU_DAYS=2.0,
#       SpotMinuteLogSig(lookback=3h, min_points=100).
#
# Other knobs (TRAIN_DAYS, TEST_DAYS, STEP_DAYS, LOSS_QUANTILE, TAU_GRID,
# SKIP_PCTS, etc.) are ENV-overridable.

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, Plots
using Flux: sigmoid

# =============================================================================
# Configuration
# =============================================================================

METHOD = lowercase(get(ENV, "METHOD", "classifier"))
METHOD in ("classifier", "ranking") || error("METHOD must be classifier|ranking, got $METHOD")

SYMBOL        = get(ENV, "SYM", "SPY")
START_DATE    = Date(get(ENV, "START_DATE", "2014-06-02"))
END_DATE      = Date(get(ENV, "END_DATE",   "2026-03-27"))

# Method-specific defaults
_default_entry_hour = METHOD == "classifier" ? "14" : "12"
_default_tenor      = METHOD == "classifier" ? "2h" : "1d"
_default_max_tau    = METHOD == "classifier" ? "0.5" : "2.0"
_default_lookback   = METHOD == "classifier" ? "5"   : "3"
_default_min_pts    = METHOD == "classifier" ? "150" : "100"

ENTRY_TIME    = Time(parse(Int, get(ENV, "ENTRY_HOUR", _default_entry_hour)), 0)
_tenor_str    = lowercase(get(ENV, "TENOR", _default_tenor))
EXPIRY_INTERVAL = if endswith(_tenor_str, "d")
    Day(parse(Int, _tenor_str[1:end-1]))
elseif endswith(_tenor_str, "h")
    Hour(parse(Int, _tenor_str[1:end-1]))
else
    error("Unknown TENOR: $_tenor_str")
end
MAX_TAU_DAYS  = parse(Float64, get(ENV, "MAX_TAU_DAYS", _default_max_tau))
SPREAD_LAMBDA = parse(Float64, get(ENV, "SPREAD_LAMBDA", "0.7"))
RATE          = parse(Float64, get(ENV, "RATE", "0.045"))
DIV_YIELD     = parse(Float64, get(ENV, "DIV", "0.013"))

PUT_DELTA     = parse(Float64, get(ENV, "PUT_DELTA",  "0.20"))
CALL_DELTA    = parse(Float64, get(ENV, "CALL_DELTA", "0.05"))

TRAIN_DAYS    = parse(Int, get(ENV, "TRAIN_DAYS", "90"))
TEST_DAYS     = parse(Int, get(ENV, "TEST_DAYS",  "30"))
STEP_DAYS     = parse(Int, get(ENV, "STEP_DAYS",  "30"))

LOSS_QUANTILE = parse(Float64, get(ENV, "LOSS_QUANTILE", "0.10"))
TAU_GRID      = [parse(Float64, strip(t))
                 for t in split(get(ENV, "TAU_GRID", "0.30,0.50,0.70,0.90"), ",")]
SKIP_PCTS     = [parse(Int, strip(s))
                 for s in split(get(ENV, "SKIP_PCTS", "10,20,30,50"), ",")]

LOOKBACK_HRS  = parse(Int, get(ENV, "LOOKBACK", _default_lookback))
MIN_POINTS    = parse(Int, get(ENV, "MIN_PTS",  _default_min_pts))

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "strangle_logsig_$(METHOD)_$(SYMBOL)_$run_ts")
mkpath(run_dir)
println("Output: $run_dir   METHOD=$METHOD")
println("\n  $SYMBOL  $START_DATE → $END_DATE   shorts (Δp=$PUT_DELTA, Δc=$CALL_DELTA)")
println("  entry=$ENTRY_TIME tenor=$EXPIRY_INTERVAL max_tau=$(MAX_TAU_DAYS)d")
if METHOD == "classifier"
    println("  τ grid: $TAU_GRID   loss quantile: $LOSS_QUANTILE")
else
    println("  Skip bottom percentile grid: $SKIP_PCTS")
end

# =============================================================================
# Data source
# =============================================================================

println("\nLoading $SYMBOL …")
(; source, sched) = polygon_parquet_source(SYMBOL;
    start_date=START_DATE, end_date=END_DATE, entry_time=ENTRY_TIME,
    rate=RATE, div_yield=DIV_YIELD, spread_lambda=SPREAD_LAMBDA,
)

# =============================================================================
# Features
# =============================================================================

logsig_feat = SpotMinuteLogSig(; lookback_hours=LOOKBACK_HRS, depth=3, min_points=MIN_POINTS)
n_logsig = logsig_dim(logsig_feat)
println("  SpotMinuteLogSig dim: $n_logsig (lookback $(LOOKBACK_HRS)h)")

base_feats = Feature[
    ATMImpliedVol(; rate=RATE, div_yield=DIV_YIELD),
    DeltaSkew(0.25, :put; rate=RATE, div_yield=DIV_YIELD),
    ATMSpread(),
    RealizedVol(; lookback=20),
]
n_base = length(base_feats)
n_feat = n_logsig + n_base
println("  base feats: $n_base   total feature dim: $n_feat")

# =============================================================================
# Build dataset
# =============================================================================

dates = Date[]; pnls = Float64[]; features = Vector{Vector{Float64}}()
n_total = 0; n_skip_pnl = 0; n_skip_feat = 0

# classifier variant uses surface-cache eviction in finally; ranking uses clear_cache=true
function _evict!(src::ParquetDataSource)
    empty!(src.surface_cache); nothing
end

println("\nBuilding feature + PnL dataset...")
if METHOD == "classifier"
    each_entry(source, EXPIRY_INTERVAL, sched) do ctx, settlement
        try
            ismissing(settlement) && return
            global n_total += 1
            dctx = delta_context(ctx; rate=RATE, div_yield=DIV_YIELD)
            dctx === nothing && (global n_skip_pnl += 1; return)
            dctx.tau * 365.25 > MAX_TAU_DAYS && (global n_skip_pnl += 1; return)
            spot = dctx.spot

            sp_K = delta_strike(dctx, -PUT_DELTA,  Put)
            sc_K = delta_strike(dctx,  CALL_DELTA, Call)
            (sp_K === nothing || sc_K === nothing) && (global n_skip_pnl += 1; return)
            sp_rec = find_record_at_strike(dctx.put_recs,  sp_K)
            sc_rec = find_record_at_strike(dctx.call_recs, sc_K)
            (sp_rec === nothing || sc_rec === nothing) && (global n_skip_pnl += 1; return)
            sp_bid = extract_price(sp_rec, :bid)
            sc_bid = extract_price(sc_rec, :bid)
            (sp_bid === nothing || sc_bid === nothing) && (global n_skip_pnl += 1; return)
            spot_settle = Float64(settlement)
            credit_usd = (sp_bid + sc_bid) * spot
            intrinsic_usd = max(sp_K - spot_settle, 0.0) + max(spot_settle - sc_K, 0.0)
            pnl = credit_usd - intrinsic_usd

            ls = logsig_feat(ctx)
            ls === nothing && (global n_skip_feat += 1; return)
            feat_vec = Float64[]
            append!(feat_vec, ls)
            ok = true
            for f in base_feats
                v = f(ctx)
                if v === nothing
                    ok = false; break
                end
                if v isa AbstractVector
                    append!(feat_vec, v)
                else
                    push!(feat_vec, Float64(v))
                end
            end
            ok || (global n_skip_feat += 1; return)
            length(feat_vec) == n_feat || (global n_skip_feat += 1; return)
            any(!isfinite, feat_vec) && (global n_skip_feat += 1; return)

            push!(dates, Date(ctx.surface.timestamp))
            push!(pnls, pnl)
            push!(features, feat_vec)
        finally
            _evict!(source)
        end
    end
else  # ranking
    each_entry(source, EXPIRY_INTERVAL, sched; clear_cache=true) do ctx, settlement
        ismissing(settlement) && return
        global n_total += 1
        dctx = delta_context(ctx; rate=RATE, div_yield=DIV_YIELD)
        dctx === nothing && (global n_skip_pnl += 1; return)
        dctx.tau * 365.25 > MAX_TAU_DAYS && (global n_skip_pnl += 1; return)
        spot = dctx.spot

        sp_K = delta_strike(dctx, -PUT_DELTA,  Put)
        sc_K = delta_strike(dctx,  CALL_DELTA, Call)
        (sp_K === nothing || sc_K === nothing) && (global n_skip_pnl += 1; return)
        sp_rec = find_record_at_strike(dctx.put_recs,  sp_K)
        sc_rec = find_record_at_strike(dctx.call_recs, sc_K)
        (sp_rec === nothing || sc_rec === nothing) && (global n_skip_pnl += 1; return)
        sp_bid = extract_price(sp_rec, :bid)
        sc_bid = extract_price(sc_rec, :bid)
        (sp_bid === nothing || sc_bid === nothing) && (global n_skip_pnl += 1; return)
        spot_settle = Float64(settlement)
        credit_usd = (sp_bid + sc_bid) * spot
        intrinsic_usd = max(sp_K - spot_settle, 0.0) + max(spot_settle - sc_K, 0.0)
        pnl = credit_usd - intrinsic_usd

        ls = logsig_feat(ctx)
        ls === nothing && (global n_skip_feat += 1; return)
        feat_vec = Float64[]
        append!(feat_vec, ls)
        ok = true
        for f in base_feats
            v = f(ctx)
            if v === nothing
                ok = false; break
            end
            if v isa AbstractVector
                append!(feat_vec, v)
            else
                push!(feat_vec, Float64(v))
            end
        end
        ok || (global n_skip_feat += 1; return)
        length(feat_vec) == n_feat || (global n_skip_feat += 1; return)
        any(!isfinite, feat_vec) && (global n_skip_feat += 1; return)

        push!(dates, Date(ctx.surface.timestamp))
        push!(pnls, pnl)
        push!(features, feat_vec)
    end
end

ord = sortperm(dates); dates = dates[ord]; pnls = pnls[ord]; features = features[ord]
@printf "\n  total %d   kept %d   skip(pnl) %d   skip(feat) %d\n" n_total length(dates) n_skip_pnl n_skip_feat
length(dates) < 200 && error("Too few entries")

X = reduce(hcat, features)
@printf "  Feature matrix: %d × %d\n" size(X, 1) size(X, 2)

# =============================================================================
# Rolling fit + filter (per-method)
# =============================================================================

function rolling_classifier_filter(dates, pnls, X, train_days, test_days, step_days, loss_q, tau_grid)
    out = Dict(t => (kept_pnls=Float64[], kept_dates=Date[], n_skipped=0) for t in tau_grid)
    fold_log = NamedTuple[]

    test_start = dates[1] + Day(train_days); last_d = dates[end]
    fidx = 0
    while test_start <= last_d
        te_end = test_start + Day(test_days) - Day(1)
        tr_start = test_start - Day(train_days); tr_end = test_start - Day(1)
        tr_idx = findall(d -> tr_start <= d <= tr_end, dates)
        te_idx = findall(d -> test_start <= d <= te_end, dates)
        if length(tr_idx) < 30 || length(te_idx) < 5
            test_start += Day(step_days); continue
        end

        tr_pnls = pnls[tr_idx]
        loss_threshold = quantile(tr_pnls, loss_q)
        y_train = Float32.(tr_pnls .<= loss_threshold)
        n_pos = sum(y_train)
        if n_pos < 3 || n_pos > length(y_train) - 3
            for t in tau_grid
                for i in te_idx
                    push!(out[t].kept_pnls, pnls[i])
                    push!(out[t].kept_dates, dates[i])
                end
            end
            test_start += Day(step_days); continue
        end

        X_train = Float32.(X[:, tr_idx])
        Y_train = reshape(y_train, 1, :)
        model, means, stds, _ = train_glmnet_classifier!(nothing, X_train, Y_train;
            alpha=0.5, val_fraction=0.2)

        X_test = Float32.(X[:, te_idx])
        X_test_norm = (X_test .- means) ./ max.(stds, 1f-6)
        logits_test = vec(model(X_test_norm))
        probs = sigmoid.(logits_test)

        for (j, i) in enumerate(te_idx)
            for t in tau_grid
                if probs[j] > t
                    out[t] = (kept_pnls=out[t].kept_pnls, kept_dates=out[t].kept_dates,
                              n_skipped=out[t].n_skipped + 1)
                else
                    push!(out[t].kept_pnls, pnls[i])
                    push!(out[t].kept_dates, dates[i])
                end
            end
        end
        fidx += 1
        push!(fold_log, (idx=fidx, test=(test_start, te_end),
                          n_tr=length(tr_idx), n_te=length(te_idx),
                          loss_threshold=loss_threshold, n_pos=Int(n_pos),
                          mean_prob=mean(probs)))
        test_start += Day(step_days)
    end
    return out, fold_log
end

function rolling_ranking_filter(dates, pnls, X, train_days, test_days, step_days, skip_pcts)
    out = Dict(p => (kept_pnls=Float64[], kept_dates=Date[], n_skipped=0) for p in skip_pcts)
    fold_log = NamedTuple[]

    test_start = dates[1] + Day(train_days); last_d = dates[end]
    fidx = 0
    while test_start <= last_d
        te_end = test_start + Day(test_days) - Day(1)
        tr_start = test_start - Day(train_days); tr_end = test_start - Day(1)
        tr_idx = findall(d -> tr_start <= d <= tr_end, dates)
        te_idx = findall(d -> test_start <= d <= te_end, dates)
        if length(tr_idx) < 30 || length(te_idx) < 5
            test_start += Day(step_days); continue
        end

        X_train = Float32.(X[:, tr_idx])
        Y_train = reshape(Float32.(pnls[tr_idx]), 1, :)
        model, means, stds, _ = train_ridge!(nothing, X_train, Y_train;
            alpha=0.0, val_fraction=0.2)

        X_test = Float32.(X[:, te_idx])
        X_test_norm = (X_test .- means) ./ max.(stds, 1f-6)
        preds = vec(model(X_test_norm))

        for p in skip_pcts
            cutoff = quantile(preds, p / 100.0)
            for (j, i) in enumerate(te_idx)
                if preds[j] < cutoff
                    out[p] = (kept_pnls=out[p].kept_pnls, kept_dates=out[p].kept_dates,
                              n_skipped=out[p].n_skipped + 1)
                else
                    push!(out[p].kept_pnls, pnls[i])
                    push!(out[p].kept_dates, dates[i])
                end
            end
        end
        fidx += 1
        push!(fold_log, (idx=fidx, test=(test_start, te_end),
                          n_tr=length(tr_idx), n_te=length(te_idx),
                          mean_pred=mean(preds), spearman_check=cor(preds, pnls[te_idx])))
        test_start += Day(step_days)
    end
    return out, fold_log
end

if METHOD == "classifier"
    println("\nRunning rolling classifier...")
    out, folds = rolling_classifier_filter(dates, pnls, X, TRAIN_DAYS, TEST_DAYS, STEP_DAYS, LOSS_QUANTILE, TAU_GRID)
    sweep_keys = TAU_GRID
    label_fn = t -> "τ=$(round(t; digits=2))"
else
    println("\nRunning rolling ridge + percentile filter...")
    out, folds = rolling_ranking_filter(dates, pnls, X, TRAIN_DAYS, TEST_DAYS, STEP_DAYS, SKIP_PCTS)
    sweep_keys = SKIP_PCTS
    label_fn = p -> "skip-$p%"
end
println("  Folds: $(length(folds))")

# Baseline (no filter, same OOS dates)
all_test_dates = Set{Date}()
for f in folds
    for i in findall(d -> f.test[1] <= d <= f.test[2], dates)
        push!(all_test_dates, dates[i])
    end
end
base_pnls = Float64[]; base_dates = Date[]
for (i, d) in enumerate(dates)
    d in all_test_dates || continue
    push!(base_pnls, pnls[i]); push!(base_dates, d)
end

# =============================================================================
# Reports
# =============================================================================

function annual_sharpe(p, ds, year)
    mask = Dates.year.(ds) .== year
    v = filter(!isnan, p[mask])
    isempty(v) || std(v) == 0 ? NaN : mean(v)/std(v)*sqrt(252)
end
function annual_total(p, ds, year)
    mask = Dates.year.(ds) .== year
    v = filter(!isnan, p[mask])
    isempty(v) ? 0.0 : sum(v)
end

oos_years = sort(unique(Dates.year.(base_dates)))

println("\n" * "=" ^ 80)
println("  Per-year Sharpe by sweep parameter")
println("=" ^ 80)
@printf "  %-5s" "year"
for k in sweep_keys; @printf "  %-8s" label_fn(k); end
@printf "    %s\n" "no filter"
println("  " * "─"^(8 + 10 * length(sweep_keys) + 12))
for y in oos_years
    @printf "  %-5d" y
    for k in sweep_keys
        sh = annual_sharpe(out[k].kept_pnls, out[k].kept_dates, y)
        @printf "  %+7.2f " sh
    end
    bsh = annual_sharpe(base_pnls, base_dates, y)
    @printf "    %+7.2f\n" bsh
end

println("\n" * "=" ^ 80)
println("  Per-year total \$ PnL by sweep parameter")
println("=" ^ 80)
@printf "  %-5s" "year"
for k in sweep_keys; @printf "  %-9s" label_fn(k); end
@printf "    %s\n" "no filter"
println("  " * "─"^(8 + 11 * length(sweep_keys) + 12))
for y in oos_years
    @printf "  %-5d" y
    for k in sweep_keys
        total = annual_total(out[k].kept_pnls, out[k].kept_dates, y)
        @printf "  %+9.0f" total
    end
    bt = annual_total(base_pnls, base_dates, y)
    @printf "    %+7.0f\n" bt
end

println("\n" * "=" ^ 80)
println("  Full-period summary")
println("=" ^ 80)
@printf "  %-12s  %5s  %5s  %+10s  %+8s  %+8s  %+8s  %5s\n" "variant" "n_kept" "n_skip" "total" "Sharpe" "MeanYr" "MinYr" "+yrs"
function summarize(p, ds, n_skip, label, oos_years)
    full_sh = isempty(p) || std(p) == 0 ? NaN : mean(p)/std(p)*sqrt(252)
    yr_sh = filter(!isnan, [annual_sharpe(p, ds, y) for y in oos_years])
    @printf "  %-12s  %5d  %5d  %+10.0f  %+8.2f  %+8.2f  %+8.2f  %2d/%-2d\n" label length(p) n_skip sum(p) full_sh mean(yr_sh) minimum(yr_sh) count(>(0), yr_sh) length(yr_sh)
end
for k in sweep_keys
    summarize(out[k].kept_pnls, out[k].kept_dates, out[k].n_skipped, label_fn(k), oos_years)
end
summarize(base_pnls, base_dates, 0, "no filter", oos_years)

# 2018 zoom
println("\n  ── 2018 zoom: skipped vs taken ──")
for k in sweep_keys
    skip_2018_dates = filter(d -> Dates.year(d) == 2018 && !(d in Set(out[k].kept_dates)), base_dates)
    p_kept = [out[k].kept_pnls[i] for (i, dd) in enumerate(out[k].kept_dates) if Dates.year(dd) == 2018]
    p_skipped = [pnls[i] for (i, d) in enumerate(dates) if d in Set(skip_2018_dates)]
    sh = isempty(p_kept) || std(p_kept) == 0 ? NaN : mean(p_kept)/std(p_kept)*sqrt(252)
    if isempty(p_skipped)
        @printf "    %-10s  kept %3d (sh %+5.2f, total %+6.1f)   skipped 0\n" label_fn(k) length(p_kept) sh sum(p_kept)
    else
        @printf "    %-10s  kept %3d (sh %+5.2f, total %+6.1f)   skipped %3d (mean %+5.2f, min %+5.2f, max %+5.2f)\n" label_fn(k) length(p_kept) sh sum(p_kept) length(p_skipped) mean(p_skipped) minimum(p_skipped) maximum(p_skipped)
    end
end

# Plots
plt_title = METHOD == "classifier" ?
    "$SYMBOL strangle (0.20, 0.05) — logsig classifier tail-day filter" :
    "$SYMBOL strangle (0.20, 0.05) — logsig ridge + skip-bottom-q%"
plt = plot(; xlabel="date", ylabel="cumulative PnL (USD)",
    title=plt_title, size=(1100, 600), legend=:topleft)
colors = [:steelblue, :seagreen, :darkorange, :firebrick]
for (ki, k) in enumerate(sweep_keys)
    od = sortperm(out[k].kept_dates)
    plot!(plt, out[k].kept_dates[od], cumsum(out[k].kept_pnls[od]);
        label="$(label_fn(k)) (kept $(length(out[k].kept_pnls)))",
        lw=2, color=colors[mod1(ki, length(colors))])
end
ord_b = sortperm(base_dates)
plot!(plt, base_dates[ord_b], cumsum(base_pnls[ord_b]);
    label="no filter (n=$(length(base_pnls)))", lw=2, color=:black, ls=:dash)
hline!(plt, [0]; color=:gray, ls=:dash, label="")
savefig(plt, joinpath(run_dir, "equity_curves.png"))
println("\n  Saved: equity_curves.png")
