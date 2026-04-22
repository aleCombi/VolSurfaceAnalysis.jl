# scripts/spy_sizing_logsig.jl
#
# Translate the SPY 1DTE logsig signal into a position-sizing strategy.
# Walk-forward: rolling 3yr train, 1Q test, step 1Q.
# Compare cumulative PnL of fixed-size baseline vs ridge-driven sizing policies.

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, Random, Plots

# =============================================================================
# Configuration  (matches the SPY 1DTE setup that gave mean IC +0.095)
# =============================================================================

SYMBOL              = "SPY"
START_DATE          = Date(2017, 1, 1)
END_DATE            = Date(2024, 1, 31)
ENTRY_TIME          = Time(12, 0)        # 12pm ET
EXPIRY_INTERVAL     = Hour(4)            # 4pm ET expiry → 4h tenor
SPREAD_LAMBDA       = 0.7
RATE                = 0.045
DIV_YIELD           = 0.013
MAX_TAU_DAYS        = 0.5                # filter to intraday only

PUT_DELTA           = 0.20
CALL_DELTA          = 0.05

FEATURES            = Feature[
    SpotMinuteLogSig(lookback_hours=3, depth=3, min_points=100),
]

TRAIN_WINDOW_DAYS   = 365
TEST_WINDOW_DAYS    = 90
STEP_DAYS           = 90

MAX_SIZE            = 2.0   # cap for linear / sigmoid policies

# =============================================================================
# Output dir
# =============================================================================

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "spy_sizing_logsig_4h_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")

# =============================================================================
# Data source + dataset build (clone from logsig_ic_test)
# =============================================================================

println("\nLoading $SYMBOL  $START_DATE → $END_DATE ...")
(; source, sched) = polygon_parquet_source(SYMBOL;
    start_date=START_DATE, end_date=END_DATE, entry_time=ENTRY_TIME,
    rate=RATE, div_yield=DIV_YIELD, spread_lambda=SPREAD_LAMBDA,
)

dates  = Date[]
feats  = Vector{Vector{Float32}}()
pnls   = Float64[]
spots  = Float64[]
n_total       = 0
n_skip_feat   = 0
n_skip_strk   = 0

println("\nBuilding dataset...")
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
    (p_K === nothing || c_K === nothing) && (global n_skip_strk += 1; return)
    short_pos = Position[]
    for t in (Trade(ctx.surface.underlying, p_K, ctx.expiry, Put;  direction=-1, quantity=1.0),
              Trade(ctx.surface.underlying, c_K, ctx.expiry, Call; direction=-1, quantity=1.0))
        p = open_position(t, ctx.surface)
        p === nothing && continue
        push!(short_pos, p)
    end
    length(short_pos) == 2 || (global n_skip_strk += 1; return)
    pnl_usd = settle(short_pos, Float64(settlement)) * spot
    push!(dates, Date(ctx.surface.timestamp))
    push!(feats, fvec)
    push!(pnls,  pnl_usd)
    push!(spots, spot)
end
println()
n_kept = length(dates)
@printf "  %d entries → kept %d (skipped: %d feat, %d strikes)\n" n_total n_kept n_skip_feat n_skip_strk

# =============================================================================
# Walk-forward fit + predict
# =============================================================================

# Sort by date (defensive)
ord = sortperm(dates)
dates = dates[ord]
feats = feats[ord]
pnls  = pnls[ord]
spots = spots[ord]

X_all = Float32.(hcat(feats...))                # (n_features, n_kept)
Y_all = reshape(Float32.(pnls), 1, length(pnls))

function build_folds(d, train_days, test_days, step)
    first_test = d[1] + Day(train_days)
    last_d = d[end]
    folds = NamedTuple[]
    ts_start = first_test
    while ts_start <= last_d
        te_end = ts_start + Day(test_days) - Day(1)
        tr_start = ts_start - Day(train_days)
        tr_end = ts_start - Day(1)
        tr_mask = (d .>= tr_start) .& (d .<= tr_end)
        te_mask = (d .>= ts_start) .& (d .<= te_end)
        if sum(tr_mask) >= 80 && sum(te_mask) >= 20
            push!(folds, (
                train_mask=tr_mask, test_mask=te_mask,
                train_start=tr_start, train_end=tr_end,
                test_start=ts_start, test_end=te_end,
            ))
        end
        ts_start += Day(step)
    end
    return folds
end

folds = build_folds(dates, TRAIN_WINDOW_DAYS, TEST_WINDOW_DAYS, STEP_DAYS)
println("\nFolds: $(length(folds))   (train_window=$TRAIN_WINDOW_DAYS d, test_window=$TEST_WINDOW_DAYS d, step=$STEP_DAYS d)")

# Pooled OOS arrays + per-fold cache of train-set normalization stats for sizing
all_dates_oos  = Date[]
all_y_oos      = Float64[]
all_pred_oos   = Float64[]
all_train_μ    = Float64[]   # per-OOS-sample, mean of TRAIN preds for that fold
all_train_σ    = Float64[]   # per-OOS-sample, std of TRAIN preds for that fold
all_train_med  = Float64[]
all_train_q75  = Float64[]

for (i, f) in enumerate(folds)
    Random.seed!(42)
    Xtr = X_all[:, f.train_mask]
    Ytr = Y_all[:, f.train_mask]
    Xte = X_all[:, f.test_mask]
    Yte = Y_all[:, f.test_mask]
    Dte = dates[f.test_mask]

    model, μ, σ, _ = train_ridge!(nothing, Xtr, Ytr; alpha=0.0, val_fraction=0.2)
    σ_safe = [s > 1e-8 ? s : one(s) for s in σ]

    # In-sample (train) predictions for normalization stats
    Xtr_norm = (Xtr .- Float32.(μ)) ./ Float32.(σ_safe)
    train_preds = Float64.(vec(model(Xtr_norm)))
    tμ = mean(train_preds)
    tσ = std(train_preds)
    tmed = median(train_preds)
    tq75 = sort(train_preds)[max(1, round(Int, 0.75 * length(train_preds)))]

    Xte_norm = (Xte .- Float32.(μ)) ./ Float32.(σ_safe)
    preds_te = Float64.(vec(model(Xte_norm)))

    n_te = length(preds_te)
    append!(all_dates_oos, Dte)
    append!(all_y_oos, Float64.(vec(Yte)))
    append!(all_pred_oos, preds_te)
    append!(all_train_μ,   fill(tμ,   n_te))
    append!(all_train_σ,   fill(tσ,   n_te))
    append!(all_train_med, fill(tmed, n_te))
    append!(all_train_q75, fill(tq75, n_te))
end

n_oos = length(all_y_oos)
println("Pooled OOS: $n_oos samples  ($(all_dates_oos[1]) → $(all_dates_oos[end]))")

# =============================================================================
# Sizing policies — applied to OOS predictions, normalized using TRAIN stats
# =============================================================================

# Each policy: (name, size_fn(pred, μ, σ, med, q75) → size)
sigmoid_fn(z) = 1 / (1 + exp(-z))

policies = [
    ("baseline (size=1)",     (p, μ, σ, m, q) -> 1.0),
    ("binary skip-below-med", (p, μ, σ, m, q) -> p > m ? 1.0 : 0.0),
    ("binary top-quartile",   (p, μ, σ, m, q) -> p > q ? 1.0 : 0.0),
    ("linear z-clip[0,$MAX_SIZE]", (p, μ, σ, m, q) ->
        clamp((p - μ) / max(σ, 1e-9), 0.0, MAX_SIZE)),
    ("sigmoid 2·σ(z)", (p, μ, σ, m, q) ->
        MAX_SIZE * sigmoid_fn((p - μ) / max(σ, 1e-9))),
]

# Compute sized PnL series per policy
sized_results = NamedTuple[]
for (name, fn) in policies
    sizes = [fn(all_pred_oos[i], all_train_μ[i], all_train_σ[i],
                all_train_med[i], all_train_q75[i]) for i in 1:n_oos]
    sized_pnls = sizes .* all_y_oos

    n_trades = count(s -> s > 0, sizes)
    avg_size_when_traded = n_trades > 0 ? mean(sizes[sizes .> 0]) : 0.0
    total_pnl = sum(sized_pnls)
    avg_pnl_per_day = mean(sized_pnls)               # per calendar trade-slot
    avg_pnl_per_trade = n_trades > 0 ? mean(sized_pnls[sizes .> 0]) : 0.0
    s = std(sized_pnls)
    sharpe = s > 0 ? mean(sized_pnls) / s * sqrt(252) : 0.0

    # Max drawdown on cumulative PnL
    cum = cumsum(sized_pnls)
    running_max = accumulate(max, cum)
    drawdowns = cum .- running_max
    max_dd = -minimum(drawdowns)

    push!(sized_results, (
        name=name, sizes=sizes, sized_pnls=sized_pnls,
        n_trades=n_trades, trade_rate=n_trades / n_oos,
        avg_size=avg_size_when_traded,
        total=total_pnl, avg_per_day=avg_pnl_per_day, avg_per_trade=avg_pnl_per_trade,
        sharpe=sharpe, max_dd=max_dd,
    ))
end

# =============================================================================
# Print comparison
# =============================================================================

println("\n", "=" ^ 100)
println("  SIZING COMPARISON — pooled OOS over $(length(folds)) walk-forward folds, $n_oos samples")
println("=" ^ 100)
@printf "  %-28s  %7s  %5s  %8s  %10s  %8s  %8s  %8s\n" "policy" "trades" "rate" "avg_sz" "total_PnL" "/day" "Sharpe" "maxDD"
println("  ", "-" ^ 96)
for r in sized_results
    @printf "  %-28s  %7d  %4.0f%%  %8.3f  %+10.0f  %+8.2f  %+8.2f  %8.0f\n" (
        r.name, r.n_trades, 100*r.trade_rate, r.avg_size,
        r.total, r.avg_per_day, r.sharpe, r.max_dd,
    )...
end

# =============================================================================
# Plot — cumulative PnL by policy
# =============================================================================

println("\n  Saving plots...")

ord_d = sortperm(all_dates_oos)
d_sorted = all_dates_oos[ord_d]

p_cum = plot(;
    xlabel="date", ylabel="cumulative PnL (USD)",
    title="SPY 1DTE 20p/5c — sizing policies (pooled walk-forward OOS)",
    legend=:topleft, size=(1300, 600),
)
colors = [:gray, :steelblue, :purple, :seagreen, :darkorange]
for (i, r) in enumerate(sized_results)
    cum = cumsum(r.sized_pnls[ord_d])
    plot!(p_cum, d_sorted, cum;
        label=@sprintf("%-26s  Sharpe=%+.2f", r.name, r.sharpe),
        lw=(r.name == "baseline (size=1)" ? 3 : 2),
        color=colors[i], ls=(r.name == "baseline (size=1)" ? :dash : :solid),
    )
end
hline!(p_cum, [0]; color=:black, ls=:dot, label="")
savefig(p_cum, joinpath(run_dir, "cumulative_pnl.png"))
println("    cumulative_pnl.png")

# Size histogram for the linear and sigmoid policies
p_sz = plot(layout=(1, 2), size=(1200, 400),
    title=["sizes — linear" "sizes — sigmoid"],
)
linear_idx = findfirst(r -> startswith(r.name, "linear"), sized_results)
sigmoid_idx = findfirst(r -> startswith(r.name, "sigmoid"), sized_results)
if linear_idx !== nothing
    histogram!(p_sz[1], sized_results[linear_idx].sizes;
        bins=30, label="", color=:seagreen, alpha=0.7)
end
if sigmoid_idx !== nothing
    histogram!(p_sz[2], sized_results[sigmoid_idx].sizes;
        bins=30, label="", color=:darkorange, alpha=0.7)
end
savefig(p_sz, joinpath(run_dir, "size_histograms.png"))
println("    size_histograms.png")

println("\n  Output: $run_dir")
