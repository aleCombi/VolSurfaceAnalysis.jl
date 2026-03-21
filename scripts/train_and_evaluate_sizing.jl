using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Flux, Statistics

# =============================================================================
# Parameters
# =============================================================================

ENTRY_TIME      = Time(10, 0)
TRAIN_ENTRY_TIMES = [Time(10, 0), Time(12, 0), Time(14, 0)]
EXPIRY_INTERVAL = Day(1)
SPREAD_LAMBDA   = 0.7

# Train/test split
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

# Sizing parameters
SIZING_THRESHOLD_QUANTILE = 0.5  # use median PnL as threshold
MAX_QUANTITY = 3.0

# =============================================================================
# Output directory
# =============================================================================

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "sizing_oos_$run_ts")
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

# Build schedules
test_dates = filter(d -> d >= TEST_START, filtered_dates)
test_entry_ts = build_entry_timestamps(test_dates, ENTRY_TIME)
test_entry_set = Set(test_entry_ts)

train_schedule = filter(t -> Date(t) <= TRAIN_END, all_timestamps)
test_schedule  = filter(t -> t in test_entry_set, all_timestamps)
println("  Train: $(length(train_schedule)) timestamps ($(length(all_entry_times)) entries/day)")
println("  Test:  $(length(test_schedule)) timestamps (1 entry/day)")

# =============================================================================
# 2. Baseline selector
# =============================================================================

baseline_selector = constrained_delta_selector(
    PUT_DELTA, CALL_DELTA;
    rate=RATE, div_yield=DIV_YIELD,
    max_loss=MAX_LOSS,
    max_spread_rel=MAX_SPREAD_REL,
    min_delta_gap=MIN_DELTA_GAP
)

# =============================================================================
# 3. Generate sizing training data
# =============================================================================

println("\n--- Generating sizing training data (train period only) ---")
sf = Feature[
    # Surface snapshot
    ATMImpliedVol(; rate=RATE, div_yield=DIV_YIELD),
    DeltaSkew(0.25, :put; rate=RATE, div_yield=DIV_YIELD),
    RiskReversal(0.25; rate=RATE, div_yield=DIV_YIELD),
    Butterfly(0.25; rate=RATE, div_yield=DIV_YIELD),
    ATMSpread(),
    TotalVolume(),
    PutCallVolumeRatio(),
    HourOfDay(),
    DayOfWeek(),
    # History-based
    RealizedVol(; lookback=20),
    VarianceRiskPremium(; lookback=20, rate=RATE, div_yield=DIV_YIELD),
    SpotMomentum(; lookback=5),
    SpotMomentum(; lookback=20),
    IVChange(; lookback=5, rate=RATE, div_yield=DIV_YIELD),
    IVPercentile(; lookback=20, rate=RATE, div_yield=DIV_YIELD),
    # Path signature
    SpotLogSig(; lookback=20, depth=3),              # 5 dim (daily, 20-day)
]
input_dim = surface_feature_dim(sf)
println("  Features: $(input_dim - 5) scalar + 5 logsig = $input_dim total")

examples = generate_sizing_training_data(
    source, EXPIRY_INTERVAL, train_schedule, baseline_selector;
    rate=RATE, div_yield=DIV_YIELD,
    surface_features=sf
)
println("  $(length(examples)) training examples generated")

if isempty(examples)
    println("No training data generated. Check data availability.")
    exit(1)
end

# Build X, Y matrices
X = hcat([e.surface_features for e in examples]...)
Y = reshape(Float32[e.pnl for e in examples], 1, :)
@printf("  PnL targets — mean=%.4f std=%.4f min=%.4f max=%.4f\n",
    mean(Y), std(Y), minimum(Y), maximum(Y))

# Compute sizing threshold from training data
median_pnl = Float64(Statistics.median(filter(x -> x > 0, vec(Y))))
@printf("  Median positive PnL (sizing threshold): %.4f\n", median_pnl)

# =============================================================================
# 4. Train model (surface features → PnL)
# =============================================================================

println("\n--- Training PnL prediction model ---")
model = Chain(
    Dense(input_dim => 32, relu),
    Dense(32 => 16, relu),
    Dense(16 => 1),  # raw PnL output, no activation
)

model, feat_means, feat_stds, history = train_model!(
    model, X, Y;
    epochs=200, lr=1e-3, batch_size=32, val_fraction=0.2, patience=20
)
println("  Training completed: $(length(history.train_loss)) epochs")
@printf("  Final train loss: %.6f\n", history.train_loss[end])
@printf("  Final val loss:   %.6f\n", history.val_loss[end])

# =============================================================================
# 5. Build strategies
# =============================================================================

baseline_strategy = IronCondorStrategy(test_schedule, EXPIRY_INTERVAL, baseline_selector)

# Binary: trade or skip based on predicted PnL sign
binary_strategy = SizedIronCondorStrategy(
    test_schedule, EXPIRY_INTERVAL, baseline_selector,
    model, feat_means, feat_stds;
    surface_features=sf,
    sizing_policy=binary_sizing(; threshold=0.0, quantity=1.0),
    debug=false
)

# Linear: proportional sizing
linear_strategy = SizedIronCondorStrategy(
    test_schedule, EXPIRY_INTERVAL, baseline_selector,
    model, feat_means, feat_stds;
    surface_features=sf,
    sizing_policy=linear_sizing(; threshold=median_pnl, max_q=MAX_QUANTITY),
    debug=false
)

# Sigmoid: smooth, always trades
sigmoid_strategy = SizedIronCondorStrategy(
    test_schedule, EXPIRY_INTERVAL, baseline_selector,
    model, feat_means, feat_stds;
    surface_features=sf,
    sizing_policy=sigmoid_sizing(; scale=median_pnl, max_q=MAX_QUANTITY),
    debug=false
)

# =============================================================================
# 6. Backtest on TEST period (out-of-sample)
# =============================================================================

function run_backtest(strategy, source, label)
    println("--- Backtesting $label ---")
    result = backtest_strategy(strategy, source)
    margin = condor_max_loss_by_key(result.positions)
    metrics = performance_metrics(result.positions, result.pnl; margin_by_key=margin)
    return (result=result, metrics=metrics)
end

function print_comparison(results, labels, title)
    println("\n", "=" ^ 78)
    println("  $title")
    println("=" ^ 78)

    header = @sprintf("  %-20s", "Metric")
    for l in labels
        header *= @sprintf("  %12s", l)
    end
    println(header)
    println("  ", "-" ^ (20 + 14 * length(labels)))

    metrics = [r.metrics for r in results]

    row(name, accessor; fmt=x -> fmt_metric(x; pct=true)) = begin
        s = @sprintf("  %-20s", name)
        for m in metrics
            s *= @sprintf("  %12s", fmt(accessor(m)))
        end
        println(s)
    end

    row("Trades", m -> m.count; fmt=x -> string(x))
    row("Total ROI", m -> m.total_roi)
    row("Avg Return", m -> m.avg_return)
    row("Sharpe", m -> m.sharpe; fmt=fmt_ratio)
    row("Sortino", m -> m.sortino; fmt=fmt_ratio)
    row("Win Rate", m -> m.win_rate; fmt=fmt_pct)
    row("Total PnL", m -> m.total_pnl; fmt=fmt_currency)
    println("=" ^ 78)
end

println("\n=== OUT-OF-SAMPLE EVALUATION ($(TEST_START) to $(TEST_END)) ===\n")
base_oos    = run_backtest(baseline_strategy, source, "Baseline (q=1)")
binary_oos  = run_backtest(binary_strategy, source, "Binary sizing")
linear_oos  = run_backtest(linear_strategy, source, "Linear sizing")
sigmoid_oos = run_backtest(sigmoid_strategy, source, "Sigmoid sizing")

print_comparison(
    [base_oos, binary_oos, linear_oos, sigmoid_oos],
    ["Baseline", "Binary", "Linear", "Sigmoid"],
    "$SYMBOL OOS: Baseline vs ML-Sized ($(TEST_START) to $(TEST_END))"
)

# =============================================================================
# 7. Plots: equity curves + PnL distribution
# =============================================================================

using Plots

"""Aggregate per-position PnL into per-condor (entry date) PnL."""
function condor_pnl_curve(result)
    by_date = Dict{Date, Float64}()
    for (pos, pnl) in zip(result.positions, result.pnl)
        ismissing(pnl) && continue
        d = Date(pos.entry_timestamp)
        by_date[d] = get(by_date, d, 0.0) + Float64(pnl)
    end
    dates = sort(collect(keys(by_date)))
    pnls = [by_date[d] for d in dates]
    return dates, pnls
end

# -- Equity curves (cumulative condor PnL) --
println("\n--- Generating plots ---")
p_eq = plot(; title="Cumulative Condor P&L — OOS $(TEST_START) to $(TEST_END)",
    xlabel="Date", ylabel="Cumulative P&L (USD)", legend=:topleft, size=(900, 500))

for (res, label, style) in [
    (base_oos, "Baseline", :solid),
    (binary_oos, "Binary", :dash),
]
    ds, ps = condor_pnl_curve(res.result)
    isempty(ds) && continue
    plot!(p_eq, ds, cumsum(ps); label=label, linestyle=style, linewidth=2)
end
eq_path = joinpath(run_dir, "equity_curves.png")
savefig(p_eq, eq_path)
println("  Saved: $eq_path")

# -- Per-condor PnL distribution overlay --
base_ds, base_ps = condor_pnl_curve(base_oos.result)
bin_ds, bin_ps = condor_pnl_curve(binary_oos.result)

edges = range(minimum(vcat(base_ps, bin_ps)) - 0.5, maximum(vcat(base_ps, bin_ps)) + 0.5, length=40)
p_dist = histogram(base_ps; bins=edges, alpha=0.5, label="Baseline (n=$(length(base_ps)))",
    title="Per-Condor P&L Distribution — OOS",
    xlabel="P&L per condor entry (USD)", ylabel="Frequency",
    size=(900, 500), normalize=:probability)
histogram!(p_dist, bin_ps; bins=edges, alpha=0.5, label="Binary (n=$(length(bin_ps)))",
    normalize=:probability)
vline!(p_dist, [0.0]; label="", color=:black, linestyle=:dash, linewidth=1)
dist_path = joinpath(run_dir, "pnl_distribution.png")
savefig(p_dist, dist_path)
println("  Saved: $dist_path")

# =============================================================================
# 8. Predicted quantities over time
# =============================================================================

println("\n--- Predicted quantities over time (OOS) ---")
pred_timestamps = DateTime[]
pred_pnls = Float64[]
linear_qs = Float64[]
sigmoid_qs = Float64[]

safe_stds = max.(feat_stds, Float32(1e-8))
linear_pol = linear_sizing(; threshold=median_pnl, max_q=MAX_QUANTITY)
sigmoid_pol = sigmoid_sizing(; scale=median_pnl, max_q=MAX_QUANTITY)

each_entry(source, EXPIRY_INTERVAL, test_schedule) do ctx, settlement
    sf_vec = extract_surface_features(ctx, sf)
    sf_vec === nothing && return
    x_norm = (sf_vec .- feat_means) ./ safe_stds
    pred = Float64(vec(model(reshape(x_norm, :, 1)))[1])
    push!(pred_timestamps, ctx.surface.timestamp)
    push!(pred_pnls, pred)
    push!(linear_qs, linear_pol(pred))
    push!(sigmoid_qs, sigmoid_pol(pred))
end

# Save CSV
pred_path = joinpath(run_dir, "predicted_sizing.csv")
open(pred_path, "w") do io
    println(io, "timestamp,predicted_pnl,linear_q,sigmoid_q")
    for (ts, p, lq, sq) in zip(pred_timestamps, pred_pnls, linear_qs, sigmoid_qs)
        @printf(io, "%s,%.6f,%.3f,%.3f\n", ts, p, lq, sq)
    end
end
println("  Saved to: $pred_path")

if !isempty(pred_pnls)
    @printf("  Predicted PnL — n=%d  mean=%.4f  std=%.4f  min=%.4f  max=%.4f\n",
        length(pred_pnls), mean(pred_pnls), std(pred_pnls),
        minimum(pred_pnls), maximum(pred_pnls))
    @printf("  Linear q — mean=%.2f  skip_rate=%.1f%%\n",
        mean(linear_qs), 100.0 * count(q -> q == 0.0, linear_qs) / length(linear_qs))
    @printf("  Sigmoid q — mean=%.2f  min=%.3f  max=%.3f\n",
        mean(sigmoid_qs), minimum(sigmoid_qs), maximum(sigmoid_qs))
end

# =============================================================================
# 9. Save model and results
# =============================================================================

using BSON
model_path = joinpath(run_dir, "model.bson")
BSON.@save model_path model feat_means feat_stds
println("\nModel saved to: $model_path")

open(joinpath(run_dir, "loss_history.csv"), "w") do io
    println(io, "epoch,train_loss,val_loss")
    for (i, (tl, vl)) in enumerate(zip(history.train_loss, history.val_loss))
        @printf(io, "%d,%.6f,%.6f\n", i, tl, vl)
    end
end

println("Done.")
