# scripts/logsig_ic.jl
#
# Information-coefficient (IC) test: does an intraday spot-path log-signature
# feature predict realized PnL of a fixed delta short strangle?
#
# - Strangle: sell PUT_DELTA put + sell CALL_DELTA call, fill at bid, settle at expiry.
# - Feature: SpotMinuteLogSig(lookback_hours=L, depth=3, min_points=M).
# - Validation: rolling walk-forward — train TRAIN_WINDOW_DAYS, test TEST_WINDOW_DAYS,
#   step STEP_DAYS. Report per-fold Pearson/Spearman IC, pooled stats, top/bot quintile,
#   verdict.
# - Model: ridge (`train_ridge!` with internal chronological val split).
#
# Replaces 16 prior scripts (logsig_ic_2ch_*_{2016|2019|intraday|intraday_late}.jl).
# All knobs below can be overridden via ENV. Common presets:
#
#   # SPY 1-DTE baseline (2016+)
#   SYM=SPY START_YEAR=2016 ENTRY_HOUR=12 TENOR=1d TRAIN_DAYS=365 LOOKBACK=3 MIN_PTS=100
#
#   # SPY intraday 12:00 / 4h
#   SYM=SPY ENTRY_HOUR=12 TENOR=4h MAX_TAU_DAYS=0.5 LOOKBACK=5 MIN_PTS=150
#
#   # SPY intraday late 14:00 / 2h (canonical)
#   SYM=SPY ENTRY_HOUR=14 TENOR=2h MAX_TAU_DAYS=0.5 LOOKBACK=5 MIN_PTS=150
#
#   # HYG 1-DTE (div-heavy: DIV=0.055, longer window allowed)
#   SYM=HYG DIV=0.055 MAX_TAU_DAYS=5 TENOR=1d LOOKBACK=3 MIN_PTS=100
#
#   # Other symbol-specific DIV_YIELDs used previously:
#   #   GLD=0.00  TSLA=0.00  SMH=0.00  XLE=0.03  TLT=0.036  QQQ=0.006  IWM=0.013

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, Random, Plots

# =============================================================================
# Configuration (override via ENV)
# =============================================================================

SYMBOL              = get(ENV, "SYM", "SPY")
START_DATE          = Date(parse(Int, get(ENV, "START_YEAR", "2017")), 1, 1)
END_DATE            = Date(parse(Int, get(ENV, "END_YEAR", "2024")),
                           parse(Int, get(ENV, "END_MONTH", "1")),
                           parse(Int, get(ENV, "END_DAY",   "31")))
ENTRY_TIME          = Time(parse(Int, get(ENV, "ENTRY_HOUR", "14")), 0)

# Tenor: "1d" → Day(1), "2h" → Hour(2), "4h" → Hour(4), etc.
_tenor_str          = lowercase(get(ENV, "TENOR", "2h"))
EXPIRY_INTERVAL     = if endswith(_tenor_str, "d")
    Day(parse(Int, _tenor_str[1:end-1]))
elseif endswith(_tenor_str, "h")
    Hour(parse(Int, _tenor_str[1:end-1]))
else
    error("Unknown TENOR: $_tenor_str (use e.g. \"1d\" or \"2h\")")
end

SPREAD_LAMBDA       = parse(Float64, get(ENV, "SPREAD_LAMBDA", "0.7"))
RATE                = parse(Float64, get(ENV, "RATE", "0.045"))
DIV_YIELD           = parse(Float64, get(ENV, "DIV", "0.013"))

# Skip entries whose picked expiry is further than MAX_TAU_DAYS from now.
# Use a large number (e.g. 5) for daily tenors; 0.5 for intraday ≤ 12h.
MAX_TAU_DAYS        = parse(Float64, get(ENV, "MAX_TAU_DAYS", "0.5"))

# Fixed strangle deltas
PUT_DELTA           = parse(Float64, get(ENV, "PUT_DELTA",  "0.20"))
CALL_DELTA          = parse(Float64, get(ENV, "CALL_DELTA", "0.05"))

# Feature: SpotMinuteLogSig. Daily scripts used (3, 100); intraday (5, 150).
LOGSIG_LOOKBACK_HRS = parse(Int, get(ENV, "LOOKBACK",  "5"))
LOGSIG_MIN_POINTS   = parse(Int, get(ENV, "MIN_PTS",   "150"))
FEATURES            = Feature[
    SpotMinuteLogSig(lookback_hours=LOGSIG_LOOKBACK_HRS, depth=3, min_points=LOGSIG_MIN_POINTS),
]

# Walk-forward windows
TRAIN_WINDOW_DAYS   = parse(Int, get(ENV, "TRAIN_DAYS", "365"))
TEST_WINDOW_DAYS    = parse(Int, get(ENV, "TEST_DAYS",  "90"))
STEP_DAYS           = parse(Int, get(ENV, "STEP_DAYS",  "90"))
MIN_TRAIN_ENTRIES   = parse(Int, get(ENV, "MIN_TRAIN",  "80"))
MIN_TEST_ENTRIES    = parse(Int, get(ENV, "MIN_TEST",   "20"))

# =============================================================================
# Output dir
# =============================================================================

run_tag = get(ENV, "TAG", "$(SYMBOL)_$(_tenor_str)_$(ENTRY_TIME)")
run_ts  = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "logsig_ic_$(run_tag)_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")
println("\n  $SYMBOL  $START_DATE → $END_DATE   entry $(ENTRY_TIME)  tenor=$(EXPIRY_INTERVAL)")
println("  Δ_put=$PUT_DELTA Δ_call=$CALL_DELTA  max_tau=$(MAX_TAU_DAYS)d")
println("  logsig lookback=$(LOGSIG_LOOKBACK_HRS)h depth=3 min_points=$(LOGSIG_MIN_POINTS)")
println("  walk-forward train=$(TRAIN_WINDOW_DAYS)d test=$(TEST_WINDOW_DAYS)d step=$(STEP_DAYS)d")

# =============================================================================
# Data source
# =============================================================================

println("\nLoading $SYMBOL …")
(; source, sched) = polygon_parquet_source(SYMBOL;
    start_date=START_DATE, end_date=END_DATE, entry_time=ENTRY_TIME,
    rate=RATE, div_yield=DIV_YIELD, spread_lambda=SPREAD_LAMBDA,
)
println("  $(length(sched)) entry timestamps")

n_features = surface_feature_dim(FEATURES)
println("\nFeature stack:")
for f in FEATURES
    println("  ", typeof(f), "  →  dim = ",
            f isa SpotMinuteLogSig || f isa SpotLogSig || f isa IntradayLogSig ? logsig_dim(f) : 1)
end
println("  total feature dim = $n_features")

# =============================================================================
# Build dataset: (date, features, pnl_usd, spot) per entry
# =============================================================================

dates  = Date[]
feats  = Vector{Vector{Float32}}()
pnls   = Float64[]
spots  = Float64[]

n_total       = 0
n_skip_feat   = 0
n_skip_strk   = 0

println("\nBuilding dataset...")
report_every = 100
each_entry(source, EXPIRY_INTERVAL, sched; clear_cache=true) do ctx, settlement
    ismissing(settlement) && return
    global n_total += 1

    fvec = extract_surface_features(ctx, FEATURES)
    if fvec === nothing
        global n_skip_feat += 1
        return
    end

    dctx = delta_context(ctx; rate=RATE, div_yield=DIV_YIELD)
    dctx === nothing && return
    dctx.tau * 365.25 > MAX_TAU_DAYS && (global n_skip_strk += 1; return)
    spot = dctx.spot

    p_K = delta_strike(dctx, -PUT_DELTA,  Put)
    c_K = delta_strike(dctx,  CALL_DELTA, Call)
    if p_K === nothing || c_K === nothing
        global n_skip_strk += 1
        return
    end

    short_pos = open_strangle_positions(ctx, p_K, c_K; direction=-1)
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
        @printf "  %5d entries processed  (kept %d)\r" n_total length(dates)
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

sort_perm    = sortperm(dates)
dates_sorted = dates[sort_perm]
X_all        = Float32.(X_all[:, sort_perm])
Y_all        = Y_all[:, sort_perm]
spots_sorted = spots[sort_perm]

include(joinpath(@__DIR__, "lib", "experiment.jl"))
folds = build_folds(dates_sorted;
    train_days=TRAIN_WINDOW_DAYS, test_days=TEST_WINDOW_DAYS, step_days=STEP_DAYS,
    min_train=MIN_TRAIN_ENTRIES, min_test=MIN_TEST_ENTRIES,
)

println("\nWalk-forward folds: $(length(folds))")

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

    pic = cor(preds, y)
    sic = spearman_corr(preds, y)
    n   = length(y)
    n_q = max(1, n ÷ 5)
    sorted = sortperm(preds; rev=true)
    top    = sorted[1:n_q]
    bot    = sorted[end-n_q+1:end]
    push!(fold_results, (
        idx=f.idx, train=(f.train_start, f.train_end), test=(f.test_start, f.test_end),
        n_train=sum(f.train_mask), n_test=n,
        pearson=pic, spearman=sic,
        all_pnl=mean(y), top_pnl=mean(y[top]), bot_pnl=mean(y[bot]),
    ))

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

n_folds        = length(fold_results)
mean_pearson   = mean(r.pearson  for r in fold_results)
mean_spearman  = mean(r.spearman for r in fold_results)
std_spearman   = std([r.spearman for r in fold_results])
positive_folds = count(r -> r.spearman > 0, fold_results)

pooled_pearson  = cor(all_preds, all_y)
pooled_spearman = spearman_corr(all_preds, all_y)
n_pool          = length(all_y)
y_mean          = mean(all_y)
ss_tot          = sum((all_y .- y_mean) .^ 2)
ss_res          = sum((all_y .- all_preds) .^ 2)
pooled_r2       = 1 - ss_res / ss_tot

n_q_pool      = max(1, n_pool ÷ 5)
sorted_pool   = sortperm(all_preds; rev=true)
top_pool      = sorted_pool[1:n_q_pool]
bot_pool      = sorted_pool[end-n_q_pool+1:end]
all_mean_pool = mean(all_y)
top_mean_pool = mean(all_y[top_pool])
bot_mean_pool = mean(all_y[bot_pool])

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

order    = sortperm(all_dates)
d_sort   = all_dates[order]
y_sort   = all_y[order]

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
