using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Flux, Statistics, Plots, Random

# =============================================================================
# Parameters
# =============================================================================

ENTRY_TIME      = Time(10, 0)
TRAIN_ENTRY_TIMES = [Time(10, 0), Time(12, 0), Time(14, 0)]
EXPIRY_INTERVAL = Day(1)
SPREAD_LAMBDA   = 0.7

TRAIN_START = Date(2024, 2, 1)
TRAIN_END   = Date(2024, 12, 31)
TEST_START  = Date(2025, 1, 1)
TEST_END    = Date(2025, 12, 31)

RATE           = 0.045
DIV_YIELD      = 0.013
MAX_LOSS       = 5.0
MAX_SPREAD_REL = 0.50
MIN_DELTA_GAP  = 0.08
PUT_DELTA      = 0.16
CALL_DELTA     = 0.16

SYMBOL         = "SPY"
SPOT_SYMBOL    = "SPY"
SPOT_MULTIPLIER = 1.0

# =============================================================================
# Output directory
# =============================================================================

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "sizing_features_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")

# =============================================================================
# 1. Build ParquetDataSource
# =============================================================================

println("\n--- Building ParquetDataSource ---")
store = DEFAULT_STORE

all_dates = available_polygon_dates(store, SYMBOL)
filtered_dates = filter(d -> d >= TRAIN_START && d <= TEST_END, all_dates)
println("  $(length(filtered_dates)) trading days ($(TRAIN_START) to $(TEST_END))")

all_entry_times = sort(unique([ENTRY_TIME; TRAIN_ENTRY_TIMES]))
entry_ts = build_entry_timestamps(filtered_dates, all_entry_times)

entry_spots = read_polygon_spot_prices_for_timestamps(
    polygon_spot_root(store), entry_ts; symbol=SPOT_SYMBOL
)
if SPOT_MULTIPLIER != 1.0
    for (k, v) in entry_spots
        entry_spots[k] = v * SPOT_MULTIPLIER
    end
end
println("  $(length(entry_spots)) entry spots loaded")

path_for_ts = ts -> polygon_options_path(store, Date(ts), SYMBOL)
read_records = (path; where="") -> read_polygon_option_records(
    path, entry_spots; where=where,
    min_volume=0, warn=false, spread_lambda=SPREAD_LAMBDA
)

source = ParquetDataSource(
    entry_ts;
    path_for_timestamp=path_for_ts,
    read_records=read_records,
    spot_root=polygon_spot_root(store),
    spot_symbol=SPOT_SYMBOL,
    spot_multiplier=SPOT_MULTIPLIER
)

all_timestamps = available_timestamps(source)

test_dates = filter(d -> d >= TEST_START, filtered_dates)
test_entry_ts = build_entry_timestamps(test_dates, ENTRY_TIME)
test_entry_set = Set(test_entry_ts)

train_schedule = filter(t -> Date(t) <= TRAIN_END, all_timestamps)
test_schedule  = filter(t -> t in test_entry_set, all_timestamps)
println("  Train: $(length(train_schedule)) timestamps")
println("  Test:  $(length(test_schedule)) timestamps")

# =============================================================================
# 2. Baseline selector + strategy
# =============================================================================

baseline_selector = constrained_delta_selector(
    PUT_DELTA, CALL_DELTA;
    rate=RATE, div_yield=DIV_YIELD,
    max_loss=MAX_LOSS,
    max_spread_rel=MAX_SPREAD_REL,
    min_delta_gap=MIN_DELTA_GAP
)

baseline_strategy = IronCondorStrategy(test_schedule, EXPIRY_INTERVAL, baseline_selector)

# =============================================================================
# 3. Define feature sets
# =============================================================================

feature_configs = [
    ("Minimal (8d)" => Feature[
        ATMImpliedVol(; rate=RATE, div_yield=DIV_YIELD),
        DeltaSkew(0.25, :put; rate=RATE, div_yield=DIV_YIELD),
        ATMSpread(),
        SpotLogSig(; lookback=20, depth=3),
    ]),
    ("Full scalar + daily logsig (20d)" => Feature[
        ATMImpliedVol(; rate=RATE, div_yield=DIV_YIELD),
        DeltaSkew(0.25, :put; rate=RATE, div_yield=DIV_YIELD),
        RiskReversal(0.25; rate=RATE, div_yield=DIV_YIELD),
        Butterfly(0.25; rate=RATE, div_yield=DIV_YIELD),
        ATMSpread(),
        TotalVolume(),
        PutCallVolumeRatio(),
        HourOfDay(),
        DayOfWeek(),
        RealizedVol(; lookback=20),
        VarianceRiskPremium(; lookback=20, rate=RATE, div_yield=DIV_YIELD),
        SpotMomentum(; lookback=5),
        SpotMomentum(; lookback=20),
        IVChange(; lookback=5, rate=RATE, div_yield=DIV_YIELD),
        IVPercentile(; lookback=20, rate=RATE, div_yield=DIV_YIELD),
        SpotLogSig(; lookback=20, depth=3),
    ]),
    ("Full scalar + minute logsig (25d)" => Feature[
        ATMImpliedVol(; rate=RATE, div_yield=DIV_YIELD),
        DeltaSkew(0.25, :put; rate=RATE, div_yield=DIV_YIELD),
        RiskReversal(0.25; rate=RATE, div_yield=DIV_YIELD),
        Butterfly(0.25; rate=RATE, div_yield=DIV_YIELD),
        ATMSpread(),
        TotalVolume(),
        PutCallVolumeRatio(),
        HourOfDay(),
        DayOfWeek(),
        RealizedVol(; lookback=20),
        VarianceRiskPremium(; lookback=20, rate=RATE, div_yield=DIV_YIELD),
        SpotMomentum(; lookback=5),
        SpotMomentum(; lookback=20),
        IVChange(; lookback=5, rate=RATE, div_yield=DIV_YIELD),
        IVPercentile(; lookback=20, rate=RATE, div_yield=DIV_YIELD),
        SpotMinuteLogSig(; lookback_hours=1, depth=3),
        SpotMinuteLogSig(; lookback_hours=6, depth=3),
    ]),
    ("Full scalar + all logsig (30d)" => Feature[
        ATMImpliedVol(; rate=RATE, div_yield=DIV_YIELD),
        DeltaSkew(0.25, :put; rate=RATE, div_yield=DIV_YIELD),
        RiskReversal(0.25; rate=RATE, div_yield=DIV_YIELD),
        Butterfly(0.25; rate=RATE, div_yield=DIV_YIELD),
        ATMSpread(),
        TotalVolume(),
        PutCallVolumeRatio(),
        HourOfDay(),
        DayOfWeek(),
        RealizedVol(; lookback=20),
        VarianceRiskPremium(; lookback=20, rate=RATE, div_yield=DIV_YIELD),
        SpotMomentum(; lookback=5),
        SpotMomentum(; lookback=20),
        IVChange(; lookback=5, rate=RATE, div_yield=DIV_YIELD),
        IVPercentile(; lookback=20, rate=RATE, div_yield=DIV_YIELD),
        SpotLogSig(; lookback=20, depth=3),
        SpotMinuteLogSig(; lookback_hours=1, depth=3),
        SpotMinuteLogSig(; lookback_hours=6, depth=3),
    ]),
]

# =============================================================================
# 4. Run baseline once
# =============================================================================

println("\n--- Backtesting Baseline ---")
base_result = backtest_strategy(baseline_strategy, source)
base_margin = condor_max_loss_by_key(base_result.positions)
base_metrics = performance_metrics(base_result.positions, base_result.pnl; margin_by_key=base_margin)

# =============================================================================
# 5. Train + evaluate binary sizing for each feature set (3 seeds each)
# =============================================================================

SEEDS = [42, 123, 7]

struct FeatureResult
    name::String
    dim::Int
    n_examples::Int
    val_loss::Float64
    metrics::PerformanceMetrics
    positions::Vector{Position}
    pnl::Vector{Union{Missing, Float64}}
end

all_results = FeatureResult[]

for (name, sf) in feature_configs
    input_dim = surface_feature_dim(sf)
    println("\n" * "=" ^ 70)
    println("  Feature set: $name ($input_dim dims)")
    println("=" ^ 70)

    # Generate training data (same for all seeds)
    examples = generate_sizing_training_data(
        source, EXPIRY_INTERVAL, train_schedule, baseline_selector;
        rate=RATE, div_yield=DIV_YIELD,
        surface_features=sf
    )
    println("  $(length(examples)) training examples")

    if isempty(examples)
        println("  SKIP: no training data")
        continue
    end

    X = hcat([e.surface_features for e in examples]...)
    Y = reshape(Float32[e.pnl for e in examples], 1, :)

    for seed in SEEDS
        Random.seed!(seed)

        model = Chain(
            Dense(input_dim => 32, relu),
            Dense(32 => 16, relu),
            Dense(16 => 1),
        )

        model, feat_means, feat_stds, history = train_model!(
            model, X, Y;
            epochs=200, lr=1e-3, batch_size=32, val_fraction=0.2, patience=20
        )
        vl = history.val_loss[end]

        strategy = SizedIronCondorStrategy(
            test_schedule, EXPIRY_INTERVAL, baseline_selector,
            model, feat_means, feat_stds;
            surface_features=sf,
            sizing_policy=binary_sizing(; threshold=0.0, quantity=1.0),
            debug=false
        )

        result = backtest_strategy(strategy, source)
        margin = condor_max_loss_by_key(result.positions)
        metrics = performance_metrics(result.positions, result.pnl; margin_by_key=margin)

        push!(all_results, FeatureResult(
            name, input_dim, length(examples), vl,
            metrics, result.positions, result.pnl
        ))

        @printf("  seed=%d  val_loss=%.3f  trades=%d  sharpe=%.2f  sortino=%.2f  roi=%s  pnl=%s\n",
            seed, vl, metrics.count, metrics.sharpe, metrics.sortino,
            fmt_metric(metrics.total_roi; pct=true), fmt_currency(metrics.total_pnl))
    end
end

# =============================================================================
# 6. Summary table (averaged over seeds)
# =============================================================================

println("\n" * "=" ^ 90)
println("  BINARY SIZING COMPARISON — OOS $(TEST_START) to $(TEST_END) (avg over $(length(SEEDS)) seeds)")
println("=" ^ 90)
@printf("  %-32s  %4s  %6s  %8s  %8s  %8s  %8s  %8s\n",
    "Feature Set", "Dim", "Trades", "ROI", "Sharpe", "Sortino", "WinRate", "PnL")
println("  ", "-" ^ 86)

# Baseline row
@printf("  %-32s  %4s  %6d  %8s  %8.2f  %8.2f  %8s  %8s\n",
    "Baseline (no ML)", "-", base_metrics.count,
    fmt_metric(base_metrics.total_roi; pct=true),
    base_metrics.sharpe, base_metrics.sortino,
    fmt_pct(base_metrics.win_rate), fmt_currency(base_metrics.total_pnl))

# Group by feature set name
for (name, _) in feature_configs
    rows = filter(r -> r.name == name, all_results)
    isempty(rows) && continue
    n = length(rows)
    avg_trades = mean(r.metrics.count for r in rows)
    avg_roi = mean(r.metrics.total_roi for r in rows)
    avg_sharpe = mean(r.metrics.sharpe for r in rows)
    avg_sortino = mean(r.metrics.sortino for r in rows)
    avg_wr = mean(r.metrics.win_rate for r in rows)
    avg_pnl = mean(r.metrics.total_pnl for r in rows)
    avg_vl = mean(r.val_loss for r in rows)
    dim = rows[1].dim

    @printf("  %-32s  %4d  %6.0f  %7.1f%%  %8.2f  %8.2f  %7.1f%%  %8s\n",
        name, dim, avg_trades, avg_roi * 100, avg_sharpe, avg_sortino,
        avg_wr * 100, fmt_currency(avg_pnl))
end
println("=" ^ 90)

# =============================================================================
# 7. Plots
# =============================================================================

function condor_pnl_curve(positions, pnl)
    by_date = Dict{Date, Float64}()
    for (pos, p) in zip(positions, pnl)
        ismissing(p) && continue
        d = Date(pos.entry_timestamp)
        by_date[d] = get(by_date, d, 0.0) + Float64(p)
    end
    dates = sort(collect(keys(by_date)))
    pnls = [by_date[d] for d in dates]
    return dates, pnls
end

println("\n--- Generating plots ---")
colors = [:blue, :red, :green, :orange]

# Helper: pick best seed by Sharpe for each feature set
function best_result(name)
    rows = filter(r -> r.name == name, all_results)
    isempty(rows) && return nothing
    return rows[argmax([r.metrics.sharpe for r in rows])]
end

# --- Plot 1: Equity curves (cumulative PnL) ---
p_eq = plot(; title="Cumulative Condor P&L (best seed per feature set)",
    xlabel="Date", ylabel="Cumulative P&L (USD)", legend=:topleft, size=(1000, 550))

base_ds, base_ps = condor_pnl_curve(base_result.positions, base_result.pnl)
plot!(p_eq, base_ds, cumsum(base_ps); label="Baseline (n=$(length(base_ps)))",
    linewidth=2.5, color=:black)

for (i, (name, _)) in enumerate(feature_configs)
    best = best_result(name)
    best === nothing && continue
    ds, ps = condor_pnl_curve(best.positions, best.pnl)
    isempty(ds) && continue
    short_name = split(name, " (")[1]
    plot!(p_eq, ds, cumsum(ps); label="$short_name (n=$(length(ps)))",
        linewidth=1.5, color=colors[i], linestyle=:dash)
end
eq_path = joinpath(run_dir, "equity_curves.png")
savefig(p_eq, eq_path)
println("  Saved: $eq_path")

# --- Plot 2: Per-condor PnL scatter over time ---
p_scatter = plot(; title="Per-Condor P&L Over Time (best seed per feature set)",
    xlabel="Date", ylabel="P&L per condor (USD)", legend=:topright, size=(1000, 550))

# Baseline dots
scatter!(p_scatter, base_ds, base_ps; label="Baseline", color=:black,
    alpha=0.3, markersize=3, markerstrokewidth=0)

for (i, (name, _)) in enumerate(feature_configs)
    best = best_result(name)
    best === nothing && continue
    ds, ps = condor_pnl_curve(best.positions, best.pnl)
    isempty(ds) && continue
    short_name = split(name, " (")[1]
    scatter!(p_scatter, ds, ps; label=short_name, color=colors[i],
        alpha=0.5, markersize=4, markerstrokewidth=0)
end
hline!(p_scatter, [0.0]; label="", color=:gray, linestyle=:dash, linewidth=0.5)
scatter_path = joinpath(run_dir, "pnl_scatter.png")
savefig(p_scatter, scatter_path)
println("  Saved: $scatter_path")

# --- Plot 3: CDF of per-condor PnL ---
p_cdf = plot(; title="Per-Condor P&L — Cumulative Distribution (best seed)",
    xlabel="P&L per condor (USD)", ylabel="Cumulative probability",
    legend=:bottomright, size=(1000, 550))

function plot_cdf!(p, values, label; kwargs...)
    sorted = sort(values)
    n = length(sorted)
    ys = (1:n) ./ n
    plot!(p, sorted, ys; label=label, linewidth=2, kwargs...)
end

plot_cdf!(p_cdf, base_ps, "Baseline (n=$(length(base_ps)))"; color=:black)

for (i, (name, _)) in enumerate(feature_configs)
    best = best_result(name)
    best === nothing && continue
    ds, ps = condor_pnl_curve(best.positions, best.pnl)
    isempty(ps) && continue
    short_name = split(name, " (")[1]
    plot_cdf!(p_cdf, ps, "$short_name (n=$(length(ps)))"; color=colors[i], linestyle=:dash)
end
vline!(p_cdf, [0.0]; label="", color=:gray, linestyle=:dot, linewidth=0.5)
hline!(p_cdf, [0.5]; label="", color=:gray, linestyle=:dot, linewidth=0.5)
cdf_path = joinpath(run_dir, "pnl_cdf.png")
savefig(p_cdf, cdf_path)
println("  Saved: $cdf_path")

# --- Plot 4: Metrics bar chart (averaged over seeds, with error bars) ---
names_short = ["Baseline"; [split(name, " (")[1] for (name, _) in feature_configs]]

avg_sharpe = [base_metrics.sharpe]
std_sharpe = [0.0]
avg_sortino = [base_metrics.sortino]
std_sortino = [0.0]
avg_winrate = [base_metrics.win_rate * 100]
std_winrate = [0.0]

for (name, _) in feature_configs
    rows = filter(r -> r.name == name, all_results)
    if isempty(rows)
        push!(avg_sharpe, 0.0); push!(std_sharpe, 0.0)
        push!(avg_sortino, 0.0); push!(std_sortino, 0.0)
        push!(avg_winrate, 0.0); push!(std_winrate, 0.0)
        continue
    end
    sharpes = [r.metrics.sharpe for r in rows]
    sortinos = [r.metrics.sortino for r in rows]
    winrates = [r.metrics.win_rate * 100 for r in rows]
    push!(avg_sharpe, mean(sharpes)); push!(std_sharpe, std(sharpes))
    push!(avg_sortino, mean(sortinos)); push!(std_sortino, std(sortinos))
    push!(avg_winrate, mean(winrates)); push!(std_winrate, std(winrates))
end

bar_colors = [:black; colors]

p_metrics = plot(layout=(1, 3), size=(1200, 450),
    title=["Sharpe (avg ± std)" "Sortino (avg ± std)" "Win Rate % (avg ± std)"])

bar!(p_metrics[1], names_short, avg_sharpe; yerr=std_sharpe, color=bar_colors,
    label="", xrotation=15)
bar!(p_metrics[2], names_short, avg_sortino; yerr=std_sortino, color=bar_colors,
    label="", xrotation=15)
bar!(p_metrics[3], names_short, avg_winrate; yerr=std_winrate, color=bar_colors,
    label="", xrotation=15)

metrics_path = joinpath(run_dir, "metrics_comparison.png")
savefig(p_metrics, metrics_path)
println("  Saved: $metrics_path")

println("\nDone.")
