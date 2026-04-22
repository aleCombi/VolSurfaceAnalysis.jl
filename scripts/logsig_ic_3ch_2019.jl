# scripts/logsig_ic_test.jl
#
# Test: does intraday spot-path information predict realized PnL of a fixed
# 20p/5c short strangle on SPY?
#
# - Fixed strangle: sell 20Δ put + sell 5Δ call, expire next day, fill at bid.
# - Features: configurable Vector{<:Feature}; extracted via `extract_surface_features`.
# - Period: 2019-01-01 → 2024-01-31 (post-2018 for liquidity sanity).
# - Validation: rolling walk-forward — train on last `TRAIN_WINDOW_DAYS`,
#   test on next `TEST_WINDOW_DAYS`, step `STEP_DAYS`. Report per-window IC,
#   aggregated IC, and pooled-test top-quintile metrics.
# - Model: ridge (`train_ridge!` with internal chronological val split).

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, Random, Plots

# =============================================================================
# Configuration
# =============================================================================

SYMBOL              = "SPY"
START_DATE          = Date(2019, 1, 1)
END_DATE            = Date(2024, 1, 31)
ENTRY_TIME          = Time(12, 0)
EXPIRY_INTERVAL     = Day(1)
SPREAD_LAMBDA       = 0.7
RATE                = 0.045
DIV_YIELD           = 0.013

# Fixed strangle
PUT_DELTA           = 0.20
CALL_DELTA          = 0.05

# Feature(s) — swap freely
FEATURES            = Feature[
    IntradayLogSig(
        channels       = 3,         # log_return, IV_change, put_skew_change from open
        depth          = 3,
        rate           = RATE,
        div_yield      = DIV_YIELD,
        min_points     = 30,        # need ≥30 minute bars common across all 3 channels
        skew_moneyness = 0.95,      # OTM put at 5% below spot
    ),
]

# Walk-forward windows
TRAIN_WINDOW_DAYS   = 3 * 365   # rolling 3-year train
TEST_WINDOW_DAYS    = 90        # 1-quarter test
STEP_DAYS           = 90        # advance one test window per fold

# =============================================================================
# Output dir
# =============================================================================

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "logsig_ic_3ch_2019_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")

# =============================================================================
# Data source
# =============================================================================

println("\nLoading $SYMBOL  $START_DATE → $END_DATE ...")
(; source, sched) = polygon_parquet_source(SYMBOL;
    start_date=START_DATE, end_date=END_DATE, entry_time=ENTRY_TIME,
    rate=RATE, div_yield=DIV_YIELD, spread_lambda=SPREAD_LAMBDA,
)
println("  $(length(sched)) entry timestamps")

n_features = surface_feature_dim(FEATURES)
println("\nFeature stack:")
for f in FEATURES
    println("  ", typeof(f), "  →  dim = ", f isa SpotMinuteLogSig || f isa SpotLogSig || f isa IntradayLogSig ? logsig_dim(f) : 1)
end
println("  total feature dim = $n_features")

# =============================================================================
# Build dataset (one entry per day): (date, features, pnl_usd, spot)
# Clear caches after each entry — features extracted, originals discarded.
# =============================================================================

dates  = Date[]
feats  = Vector{Vector{Float32}}()
pnls   = Float64[]
spots  = Float64[]

n_total       = 0
n_skip_feat   = 0
n_skip_strk   = 0

# Cache-eviction helper. ParquetDataSource and IntradayLogSig.BarCache both
# accumulate per-date data forever; clear after each entry.
function _evict_caches!(src::ParquetDataSource, feats::Vector{<:Feature})
    empty!(src.surface_cache)
    empty!(src.spot_date_cache)
    for f in feats
        if f isa IntradayLogSig
            empty!(f.cache.data)
        end
    end
    nothing
end

println("\nBuilding dataset...")
report_every = 100
each_entry(source, EXPIRY_INTERVAL, sched; clear_cache=true) do ctx, settlement
    try
        ismissing(settlement) && return
        global n_total += 1

        fvec = extract_surface_features(ctx, FEATURES)
        if fvec === nothing
            global n_skip_feat += 1
            return
        end

        dctx = delta_context(ctx; rate=RATE, div_yield=DIV_YIELD)
dctx === nothing && return
spot = dctx.spot

p_K = delta_strike(dctx, -PUT_DELTA,  Put)
c_K = delta_strike(dctx,  CALL_DELTA, Call)
        if p_K === nothing || c_K === nothing
            global n_skip_strk += 1
            return
        end

        short_pos = Position[]
        for t in (Trade(ctx.surface.underlying, p_K, ctx.expiry, Put;  direction=-1, quantity=1.0),
                  Trade(ctx.surface.underlying, c_K, ctx.expiry, Call; direction=-1, quantity=1.0))
            p = open_position(t, ctx.surface)
            p === nothing && continue
            push!(short_pos, p)
        end
        if length(short_pos) != 2
            global n_skip_strk += 1
            return
        end

        pnl_usd = settle(short_pos, Float64(settlement)) * spot

        push!(dates, Date(ctx.surface.timestamp))
        push!(feats, fvec)
        push!(pnls,  pnl_usd)
        push!(spots, spot)

        if n_total % report_every == 0
            @printf "  %5d entries processed  (kept %d, mem freed each step)\r" n_total length(dates)
        end
    finally
        _evict_caches!(source, FEATURES)
    end
end
println()

n_kept = length(dates)
@printf "  %d entries → kept %d  (skipped: %d feature, %d strikes)\n" n_total n_kept n_skip_feat n_skip_strk
n_kept == 0 && error("No usable entries")

# =============================================================================
# Helpers
# =============================================================================

function rank_vec(v::AbstractVector{<:Real})
    perm = sortperm(v)
    ranks = similar(v, Int)
    @inbounds for i in eachindex(perm)
        ranks[perm[i]] = i
    end
    return ranks
end
spearman_corr(x, y) = cor(rank_vec(x), rank_vec(y))

function fit_predict(X_train::Matrix{Float32}, Y_train::Matrix{Float32},
                    X_test::Matrix{Float32}; alpha::Float64=0.0)
    Random.seed!(42)
    model, μ, σ, _ = train_ridge!(nothing, X_train, Y_train; alpha=alpha, val_fraction=0.2)
    σ_safe = [s > 1e-8 ? s : one(s) for s in σ]
    X_test_norm = (X_test .- Float32.(μ)) ./ Float32.(σ_safe)
    return vec(model(X_test_norm))
end

# =============================================================================
# Walk-forward folds
# =============================================================================

X_all = hcat(feats...)                              # (n_features, n_kept)
Y_all = reshape(Float32.(pnls), 1, length(pnls))    # (1, n_kept)

# Sort by date (defensive — should already be sorted by each_entry order)
sort_perm = sortperm(dates)
dates_sorted = dates[sort_perm]
X_all = Float32.(X_all[:, sort_perm])
Y_all = Y_all[:, sort_perm]
spots_sorted = spots[sort_perm]

# Generate fold start dates: first test window starts after TRAIN_WINDOW_DAYS
first_test_start = dates_sorted[1] + Day(TRAIN_WINDOW_DAYS)
last_date = dates_sorted[end]

function build_folds(dates_sorted, first_test_start, last_date,
                    train_window_days, test_window_days, step_days)
    folds = NamedTuple[]
    test_start = first_test_start
    fold_idx = 0
    while test_start <= last_date
        test_end = test_start + Day(test_window_days) - Day(1)
        train_start = test_start - Day(train_window_days)
        train_end = test_start - Day(1)

        train_mask = (dates_sorted .>= train_start) .& (dates_sorted .<= train_end)
        test_mask  = (dates_sorted .>= test_start)  .& (dates_sorted .<= test_end)
        n_tr = sum(train_mask)
        n_te = sum(test_mask)
        if n_tr < 200 || n_te < 20
            test_start += Day(step_days)
            continue
        end

        fold_idx += 1
        push!(folds, (
            idx=fold_idx,
            train_start=train_start, train_end=train_end,
            test_start=test_start,   test_end=test_end,
            train_mask=train_mask,   test_mask=test_mask,
            n_train=n_tr,            n_test=n_te,
        ))
        test_start += Day(step_days)
    end
    return folds
end

folds = build_folds(dates_sorted, first_test_start, last_date,
                    TRAIN_WINDOW_DAYS, TEST_WINDOW_DAYS, STEP_DAYS)

println("\nWalk-forward folds: $(length(folds))")
@printf "  train_window=%d days  test_window=%d days  step=%d days\n" TRAIN_WINDOW_DAYS TEST_WINDOW_DAYS STEP_DAYS

# =============================================================================
# Run folds
# =============================================================================

fold_results = NamedTuple[]
all_preds  = Float64[]
all_y      = Float64[]
all_dates  = Date[]
all_spots  = Float64[]

for f in folds
    X_tr = X_all[:, f.train_mask]
    Y_tr = Y_all[:, f.train_mask]
    X_te = X_all[:, f.test_mask]
    Y_te = Y_all[:, f.test_mask]

    preds = fit_predict(X_tr, Y_tr, X_te; alpha=0.0)
    y     = vec(Y_te)
    @assert all(isfinite, preds) "Non-finite predictions in fold $(f.idx)"

    # Per-fold metrics
    pic = cor(preds, y)
    sic = spearman_corr(preds, y)
    n   = length(y)
    n_q = max(1, n ÷ 5)
    sorted = sortperm(preds; rev=true)
    top    = sorted[1:n_q]
    bot    = sorted[end-n_q+1:end]
    top_pnl = mean(y[top])
    bot_pnl = mean(y[bot])
    all_pnl = mean(y)
    push!(fold_results, (
        idx=f.idx, train=(f.train_start, f.train_end), test=(f.test_start, f.test_end),
        n_train=f.n_train, n_test=n,
        pearson=pic, spearman=sic,
        all_pnl=all_pnl, top_pnl=top_pnl, bot_pnl=bot_pnl,
    ))

    # Pool for global stats
    append!(all_preds, preds)
    append!(all_y,     y)
    append!(all_dates, dates_sorted[f.test_mask])
    append!(all_spots, spots_sorted[f.test_mask])
end

# =============================================================================
# Per-fold table
# =============================================================================

println("\n", "=" ^ 78)
println("  WALK-FORWARD FOLD METRICS")
println("=" ^ 78)
@printf "  %-3s  %-23s  %-5s  %-5s  %-7s  %-7s  %-7s  %-7s  %-7s\n" (
    "f", "test window", "n_tr", "n_te", "Pear IC", "Spr IC", "AllPnL", "TopPnL", "BotPnL"
)...
println("  ", "-" ^ 76)
for r in fold_results
    @printf "  %-3d  %s → %s  %5d  %5d  %+7.4f  %+7.4f  %+7.1f  %+7.1f  %+7.1f\n" (
        r.idx, r.test[1], r.test[2], r.n_train, r.n_test,
        r.pearson, r.spearman, r.all_pnl, r.top_pnl, r.bot_pnl,
    )...
end

# =============================================================================
# Aggregate metrics
# =============================================================================

n_folds = length(fold_results)
mean_pearson  = mean(r.pearson  for r in fold_results)
mean_spearman = mean(r.spearman for r in fold_results)
std_spearman  = std([r.spearman for r in fold_results])
positive_folds = count(r -> r.spearman > 0, fold_results)

# Pooled out-of-sample
pooled_pearson  = cor(all_preds, all_y)
pooled_spearman = spearman_corr(all_preds, all_y)
n_pool          = length(all_y)
y_mean          = mean(all_y)
ss_tot          = sum((all_y .- y_mean) .^ 2)
ss_res          = sum((all_y .- all_preds) .^ 2)
pooled_r2       = 1 - ss_res / ss_tot

# Pooled top-quintile (across all OOS folds, ranked together)
n_q_pool        = max(1, n_pool ÷ 5)
sorted_pool     = sortperm(all_preds; rev=true)
top_pool        = sorted_pool[1:n_q_pool]
bot_pool        = sorted_pool[end-n_q_pool+1:end]
all_mean_pool   = mean(all_y)
top_mean_pool   = mean(all_y[top_pool])
bot_mean_pool   = mean(all_y[bot_pool])

sharpe(v) = std(v) > 0 ? mean(v) / std(v) * sqrt(252) : 0.0
all_sharpe = sharpe(all_y)
top_sharpe = sharpe(all_y[top_pool])
bot_sharpe = sharpe(all_y[bot_pool])

println("\n", "=" ^ 78)
println("  AGGREGATE — across $n_folds folds, $n_pool pooled OOS samples")
println("=" ^ 78)
@printf "  Mean per-fold Spearman = %+.4f   std = %.4f   #folds with IC>0 = %d/%d\n" mean_spearman std_spearman positive_folds n_folds
@printf "  Mean per-fold Pearson  = %+.4f\n" mean_pearson
println()
@printf "  Pooled Spearman IC      = %+.4f\n" pooled_spearman
@printf "  Pooled Pearson  IC      = %+.4f\n" pooled_pearson
@printf "  Pooled OOS R^2          = %+.4f\n" pooled_r2
println()
@printf "  Pooled all-trades AvgPnL = %+8.2f USD   Sharpe = %+.2f\n" all_mean_pool all_sharpe
@printf "  Pooled top quintile      = %+8.2f USD   Sharpe = %+.2f   n=%d\n" top_mean_pool top_sharpe n_q_pool
@printf "  Pooled bot quintile      = %+8.2f USD   Sharpe = %+.2f   n=%d\n" bot_mean_pool bot_sharpe n_q_pool
@printf "  Spread (top - bot)       = %+8.2f USD/trade\n" (top_mean_pool - bot_mean_pool)

# =============================================================================
# Plots
# =============================================================================

println("\n  Saving plots...")

# Per-fold Spearman bar chart
p1 = bar([r.idx for r in fold_results], [r.spearman for r in fold_results];
    xlabel="fold", ylabel="Spearman IC",
    title="walk-forward Spearman IC per fold (mean=$(round(mean_spearman, digits=3)))",
    color=[r.spearman > 0 ? :seagreen : :firebrick for r in fold_results],
    label="", size=(900, 400),
)
hline!(p1, [0]; color=:gray, ls=:dash, label="")
hline!(p1, [mean_spearman]; color=:black, ls=:dot, label="mean")
savefig(p1, joinpath(run_dir, "walkforward_ic.png"))
println("    walkforward_ic.png")

# Pooled cumulative PnL: all vs predicted top quintile vs bot quintile
order = sortperm(all_dates)
d_sort   = all_dates[order]
y_sort   = all_y[order]
p_sort   = all_preds[order]

in_top = falses(n_pool); in_top[top_pool] .= true
in_bot = falses(n_pool); in_bot[bot_pool] .= true

cum_all = cumsum(y_sort)
cum_top = cumsum([in_top[order[i]] ? y_sort[i] : 0.0 for i in 1:n_pool])
cum_bot = cumsum([in_bot[order[i]] ? y_sort[i] : 0.0 for i in 1:n_pool])

p2 = plot(d_sort, cum_all;
    xlabel="date", ylabel="cumulative PnL (USD)",
    title="cumulative PnL — pooled walk-forward (all vs predicted top/bot quintile)",
    label="all trades", lw=2, color=:gray, size=(1200, 500),
)
plot!(p2, d_sort, cum_top; label="top quintile (by pred)", lw=2, color=:seagreen)
plot!(p2, d_sort, cum_bot; label="bot quintile (by pred)", lw=2, color=:firebrick)
hline!(p2, [0]; color=:black, ls=:dash, label="")
savefig(p2, joinpath(run_dir, "cumulative_pnl_walkforward.png"))
println("    cumulative_pnl_walkforward.png")

# =============================================================================
# Verdict
# =============================================================================

println("\n", "=" ^ 78)
println("  VERDICT")
println("=" ^ 78)
verdict = if abs(pooled_spearman) > 0.10 && positive_folds >= n_folds * 0.7
    "STRONG signal (pooled |IC|>0.10 AND ≥70% folds positive)"
elseif abs(pooled_spearman) > 0.05 && positive_folds >= n_folds * 0.6
    "WEAK but consistent signal (pooled |IC|>0.05 AND ≥60% folds positive)"
elseif abs(pooled_spearman) > 0.02
    "MARGINAL (pooled |IC|>0.02 but inconsistent across folds)"
else
    "NO SIGNAL (pooled |IC|<=0.02)"
end
println("  $verdict")
println("\n  Output: $run_dir")
