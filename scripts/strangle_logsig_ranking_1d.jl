# scripts/strangle_logsig_ranking_1d.jl
#
# Tail-day ranking filter for SPY 1-DTE (0.20, 0.05) short strangles.
# Uses RIDGE REGRESSION + percentile-based selection (the approach that
# worked in earlier session).
#
# Pipeline:
#   - Cache features per entry day (SpotMinuteLogSig 3h + 4 base features)
#   - Cache (0.20, 0.05) PnL per day
#   - Rolling 90d train / 30d test:
#       * Fit ridge regressor: features → PnL
#       * Predict on test features
#       * Skip days where predicted PnL is in the bottom-q% of test predictions
#   - Sweep skip percentile q ∈ {10, 20, 30, 50}; report per-year + overall.

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, Plots

SYMBOL = "SPY"
START_DATE = Date(2014, 6, 2)
END_DATE   = Date(2026, 3, 27)
ENTRY_TIME = Time(12, 0)        # noon — match the working logsig setup
EXPIRY_INTERVAL = Day(1)
MAX_TAU_DAYS = 2.0
SPREAD_LAMBDA = 0.7
RATE = 0.045
DIV_YIELD = 0.013

PUT_DELTA  = 0.20
CALL_DELTA = 0.05

TRAIN_DAYS = 90
TEST_DAYS  = 30
STEP_DAYS  = 30
SKIP_PCTS  = [10, 20, 30, 50]   # skip bottom-q% of predicted PnL

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "strangle_logsig_ranking_1d_$(SYMBOL)_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")
println("\n  $SYMBOL  $START_DATE → $END_DATE   1-DTE noon   shorts (Δp=$PUT_DELTA, Δc=$CALL_DELTA)")
println("  Skip bottom percentile grid: $SKIP_PCTS")

# --- Data source ------------------------------------------------------------
println("\nLoading $SYMBOL  $START_DATE → $END_DATE ...")
(; source, sched) = polygon_parquet_source(SYMBOL;
    start_date=START_DATE, end_date=END_DATE, entry_time=ENTRY_TIME,
    rate=RATE, div_yield=DIV_YIELD, spread_lambda=SPREAD_LAMBDA,
)

# --- Features --------------------------------------------------------------
logsig_feat = SpotMinuteLogSig(; lookback_hours=3, depth=3, min_points=100)
n_logsig = logsig_dim(logsig_feat)
println("  SpotMinuteLogSig dim: $n_logsig (lookback 3h)")

base_feats = Feature[
    ATMImpliedVol(; rate=RATE, div_yield=DIV_YIELD),
    DeltaSkew(0.25, :put; rate=RATE, div_yield=DIV_YIELD),
    ATMSpread(),
    RealizedVol(; lookback=20),
]
n_base = length(base_feats)
n_feat = n_logsig + n_base
println("  base feats: $n_base   total feature dim: $n_feat")

# --- Build dataset ---------------------------------------------------------
dates = Date[]; pnls = Float64[]; features = Vector{Vector{Float64}}()
n_total = 0; n_skip_pnl = 0; n_skip_feat = 0

println("\nBuilding feature + PnL dataset...")
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
    sp_rec = nothing; sc_rec = nothing
    for r in dctx.put_recs;  r.strike == sp_K && (sp_rec = r; break); end
    for r in dctx.call_recs; r.strike == sc_K && (sc_rec = r; break); end
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

ord = sortperm(dates); dates = dates[ord]; pnls = pnls[ord]; features = features[ord]
@printf "\n  total %d   kept %d   skip(pnl) %d   skip(feat) %d\n" n_total length(dates) n_skip_pnl n_skip_feat
length(dates) < 200 && error("Too few entries")

X = reduce(hcat, features)
@printf "  Feature matrix: %d × %d\n" size(X, 1) size(X, 2)

# --- Rolling ridge + percentile filter --------------------------------------
function rolling_ridge_filter(dates, pnls, X, train_days, test_days, step_days, skip_pcts)
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
            cutoff = quantile(preds, p / 100.0)   # bottom-p% threshold of predictions
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

println("\nRunning rolling ridge + percentile filter...")
out, folds = rolling_ridge_filter(dates, pnls, X, TRAIN_DAYS, TEST_DAYS, STEP_DAYS, SKIP_PCTS)
println("  Folds: $(length(folds))")

# Baseline: no-filter over the same OOS window
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

# --- Reports ----------------------------------------------------------------
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
println("  Per-year Sharpe by skip%")
println("=" ^ 80)
@printf "  %-5s" "year"
for p in SKIP_PCTS; @printf "  skip-%-3d" p; end
@printf "    %s\n" "no filter"
println("  " * "─"^(8 + 9 * length(SKIP_PCTS) + 12))
for y in oos_years
    @printf "  %-5d" y
    for p in SKIP_PCTS
        sh = annual_sharpe(out[p].kept_pnls, out[p].kept_dates, y)
        @printf "  %+7.2f" sh
    end
    bsh = annual_sharpe(base_pnls, base_dates, y)
    @printf "    %+7.2f\n" bsh
end

println("\n" * "=" ^ 80)
println("  Per-year total \$ PnL by skip%")
println("=" ^ 80)
@printf "  %-5s" "year"
for p in SKIP_PCTS; @printf "  skip-%-5d" p; end
@printf "    %s\n" "no filter"
println("  " * "─"^(8 + 11 * length(SKIP_PCTS) + 12))
for y in oos_years
    @printf "  %-5d" y
    for p in SKIP_PCTS
        total = annual_total(out[p].kept_pnls, out[p].kept_dates, y)
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
for p in SKIP_PCTS
    summarize(out[p].kept_pnls, out[p].kept_dates, out[p].n_skipped, "skip-$p%", oos_years)
end
summarize(base_pnls, base_dates, 0, "no filter", oos_years)

# --- 2018 zoom --------------------------------------------------------------
println("\n  ── 2018 zoom: skipped vs taken ──")
for p in SKIP_PCTS
    keep_2018 = filter(d -> Dates.year(d) == 2018, out[p].kept_dates)
    skip_2018_dates = filter(d -> Dates.year(d) == 2018 && !(d in Set(out[p].kept_dates)), base_dates)
    p_kept = [out[p].kept_pnls[i] for (i, dd) in enumerate(out[p].kept_dates) if Dates.year(dd) == 2018]
    p_skipped = [pnls[i] for (i, d) in enumerate(dates) if d in Set(skip_2018_dates)]
    sh = isempty(p_kept) || std(p_kept) == 0 ? NaN : mean(p_kept)/std(p_kept)*sqrt(252)
    if isempty(p_skipped)
        @printf "    skip-%2d%%  kept %3d (sh %+5.2f, total %+6.1f)   skipped 0\n" p length(p_kept) sh sum(p_kept)
    else
        @printf "    skip-%2d%%  kept %3d (sh %+5.2f, total %+6.1f)   skipped %3d (mean %+5.2f, min %+5.2f, max %+5.2f)\n" p length(p_kept) sh sum(p_kept) length(p_skipped) mean(p_skipped) minimum(p_skipped) maximum(p_skipped)
    end
end

# --- Plots ------------------------------------------------------------------
plt = plot(; xlabel="date", ylabel="cumulative PnL (USD)",
    title="$SYMBOL 1-DTE strangle (0.20, 0.05) — logsig ridge + skip-bottom-q%",
    size=(1100, 600), legend=:topleft)
colors = [:steelblue, :seagreen, :darkorange, :firebrick]
for (pi, p) in enumerate(SKIP_PCTS)
    od = sortperm(out[p].kept_dates)
    plot!(plt, out[p].kept_dates[od], cumsum(out[p].kept_pnls[od]);
        label="skip-$p% (kept $(length(out[p].kept_pnls)))", lw=2, color=colors[mod1(pi, length(colors))])
end
ord_b = sortperm(base_dates)
plot!(plt, base_dates[ord_b], cumsum(base_pnls[ord_b]);
    label="no filter (n=$(length(base_pnls)))", lw=2, color=:black, ls=:dash)
hline!(plt, [0]; color=:gray, ls=:dash, label="")
savefig(plt, joinpath(run_dir, "equity_curves.png"))
println("\n  Saved: equity_curves.png")
