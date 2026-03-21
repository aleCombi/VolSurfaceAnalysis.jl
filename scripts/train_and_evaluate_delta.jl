using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Flux, Statistics

# =============================================================================
# Parameters
# =============================================================================

ENTRY_TIME      = Time(10, 0)           # backtest entry time
TRAIN_ENTRY_TIMES = [Time(10, 0), Time(12, 0), Time(14, 0)]  # extra entries for training only
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
DELTA_GRID     = 0.08:0.02:0.30

SYMBOL         = "SPY"
SPOT_SYMBOL    = "SPY"
SPOT_MULTIPLIER = 1.0

# =============================================================================
# Output directory
# =============================================================================

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "delta_oos_$run_ts")
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

# Use all entry times (train + test) for the data source
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

# Build separate test timestamps (single entry time only, for fair comparison)
test_dates = filter(d -> d >= TEST_START, filtered_dates)
test_entry_ts = build_entry_timestamps(test_dates, ENTRY_TIME)
test_entry_set = Set(test_entry_ts)

# Training: all entry times × train dates (more data)
# Testing: single entry time × test dates (fair comparison)
train_schedule = filter(t -> Date(t) <= TRAIN_END, all_timestamps)
test_schedule  = filter(t -> t in test_entry_set, all_timestamps)
println("  Train: $(length(train_schedule)) timestamps ($(length(all_entry_times)) entries/day)")
println("  Test:  $(length(test_schedule)) timestamps (1 entry/day)")

# =============================================================================
# 2. Generate delta regression training data — TRAIN period only
# =============================================================================

println("\n--- Generating delta training data (train period only) ---")
sf = Feature[
    # Surface snapshot
    ATMImpliedVol(; rate=RATE, div_yield=DIV_YIELD),
    DeltaSkew(0.25, :put; rate=RATE, div_yield=DIV_YIELD),
    ATMSpread(),
    # Path signature (replaces all backward-looking features)
    SpotLogSig(; lookback=20, depth=3),
]
println("  Features: 3 snapshot + 5 logsig = $(surface_feature_dim(sf)) total")

examples = generate_delta_training_data(
    source, EXPIRY_INTERVAL, train_schedule;
    delta_grid=DELTA_GRID,
    rate=RATE, div_yield=DIV_YIELD,
    utility=risk_adjusted_utility(1.0),
    surface_features=sf,
    wing_objective=:roi,
    symmetric=true,
    max_loss_max=MAX_LOSS
)
println("  $(length(examples)) training examples generated")

if isempty(examples)
    println("No training data generated. Check data availability.")
    exit(1)
end

input_dim = length(examples[1].surface_features)
println("  Feature dimension: $input_dim (surface only)")

# Build X, Y matrices — symmetric so put_delta == call_delta, use 1D target
X = hcat([e.surface_features for e in examples]...)
Y = reshape(Float32[e.put_delta for e in examples], 1, :)
@printf("  Delta targets — mean=%.3f std=%.3f min=%.3f max=%.3f\n",
    mean(Y), std(Y), minimum(Y), maximum(Y))

# =============================================================================
# 3. Train model
# =============================================================================

println("\n--- Training 1D symmetric delta regression model ---")
model = Chain(
    Dense(input_dim => 32, relu),
    Dense(32 => 16, relu),
    Dense(16 => 1, sigmoid),  # output in (0, 1), rescaled to delta range
)

# Scale targets to (0, 1) for sigmoid output
delta_lo, delta_hi = Float64(first(DELTA_GRID)), Float64(last(DELTA_GRID))
Y_scaled = Float32.((Y .- delta_lo) ./ (delta_hi - delta_lo))

model, feat_means, feat_stds, history = train_model!(
    model, X, Y_scaled;
    epochs=200, lr=1e-3, batch_size=32, val_fraction=0.2, patience=20
)
println("  Training completed: $(length(history.train_loss)) epochs")
@printf("  Final train loss: %.6f\n", history.train_loss[end])
@printf("  Final val loss:   %.6f\n", history.val_loss[end])

# Wrap model: rescale (0,1) -> delta range, then duplicate to (delta, delta) for DirectDeltaSelector
rescaled_model = Chain(
    model,
    x -> x .* Float32(delta_hi - delta_lo) .+ Float32(delta_lo),
    x -> vcat(x, x)  # symmetric: same delta for put and call
)

# =============================================================================
# 4. Build selectors
# =============================================================================

delta_selector = DirectDeltaSelector(
    rescaled_model, feat_means, feat_stds;
    surface_features=sf,
    rate=RATE, div_yield=DIV_YIELD,
    max_loss=MAX_LOSS,
    max_spread_rel=MAX_SPREAD_REL,
    delta_clamp=(delta_lo, delta_hi)
)

baseline_selector = constrained_delta_selector(
    PUT_DELTA, CALL_DELTA;
    rate=RATE, div_yield=DIV_YIELD,
    max_loss=MAX_LOSS,
    max_spread_rel=MAX_SPREAD_REL,
    min_delta_gap=MIN_DELTA_GAP
)

# =============================================================================
# 5. Backtest on TEST period (out-of-sample)
# =============================================================================

function run_backtest(selector, schedule, source, label)
    println("--- Backtesting $label ($(length(schedule)) dates) ---")
    strategy = IronCondorStrategy(schedule, EXPIRY_INTERVAL, selector)
    result = backtest_strategy(strategy, source)
    margin = condor_max_loss_by_key(result.positions)
    metrics = performance_metrics(result.positions, result.pnl; margin_by_key=margin)
    return (result=result, metrics=metrics)
end

function print_comparison(ml_m, base_m, title)
    println("\n", "=" ^ 70)
    println("  $title")
    println("=" ^ 70)
    @printf("  %-20s  %12s  %12s\n", "Metric", "DeltaReg", "Baseline")
    println("  ", "-" ^ 48)
    @printf("  %-20s  %12d  %12d\n", "Trades", ml_m.count, base_m.count)
    @printf("  %-20s  %12s  %12s\n", "Total ROI", fmt_metric(ml_m.total_roi; pct=true), fmt_metric(base_m.total_roi; pct=true))
    @printf("  %-20s  %12s  %12s\n", "Avg Return", fmt_metric(ml_m.avg_return; pct=true), fmt_metric(base_m.avg_return; pct=true))
    @printf("  %-20s  %12s  %12s\n", "Sharpe", fmt_ratio(ml_m.sharpe), fmt_ratio(base_m.sharpe))
    @printf("  %-20s  %12s  %12s\n", "Sortino", fmt_ratio(ml_m.sortino), fmt_ratio(base_m.sortino))
    @printf("  %-20s  %12s  %12s\n", "Win Rate", fmt_pct(ml_m.win_rate), fmt_pct(base_m.win_rate))
    @printf("  %-20s  %12s  %12s\n", "Total PnL", fmt_currency(ml_m.total_pnl), fmt_currency(base_m.total_pnl))
    println("=" ^ 70)
end

println("\n=== OUT-OF-SAMPLE EVALUATION ($(TEST_START) to $(TEST_END)) ===\n")
ml_oos   = run_backtest(delta_selector, test_schedule, source, "DeltaReg")
base_oos = run_backtest(baseline_selector, test_schedule, source, "Baseline")
print_comparison(ml_oos.metrics, base_oos.metrics,
    "$SYMBOL OUT-OF-SAMPLE: DeltaReg vs Baseline ($(TEST_START) to $(TEST_END))")

# In-sample: also single entry time for fair comparison
train_dates_single = filter(d -> d >= TRAIN_START && d <= TRAIN_END, filtered_dates)
train_schedule_single = build_entry_timestamps(train_dates_single, ENTRY_TIME)
train_schedule_single = filter(t -> t in Set(all_timestamps), train_schedule_single)

println("\n=== IN-SAMPLE REFERENCE ($(TRAIN_START) to $(TRAIN_END)) ===\n")
ml_is   = run_backtest(delta_selector, train_schedule_single, source, "DeltaReg (in-sample)")
base_is = run_backtest(baseline_selector, train_schedule_single, source, "Baseline (in-sample)")
print_comparison(ml_is.metrics, base_is.metrics,
    "$SYMBOL IN-SAMPLE: DeltaReg vs Baseline ($(TRAIN_START) to $(TRAIN_END))")

# =============================================================================
# 6. Save model and results
# =============================================================================

# =============================================================================
# 6. Predicted deltas over time
# =============================================================================

println("\n--- Predicted deltas over time (OOS) ---")
pred_timestamps = DateTime[]
pred_deltas = Float64[]
safe_stds = max.(feat_stds, Float32(1e-8))

each_entry(source, EXPIRY_INTERVAL, test_schedule) do ctx, settlement
    sf_vec = extract_surface_features(ctx, sf)
    sf_vec === nothing && return
    x_norm = (sf_vec .- feat_means) ./ safe_stds
    raw = vec(rescaled_model(reshape(x_norm, :, 1)))
    pred_delta = clamp(Float64(raw[1]), delta_lo, delta_hi)
    push!(pred_timestamps, ctx.surface.timestamp)
    push!(pred_deltas, pred_delta)
end

# Save CSV
pred_path = joinpath(run_dir, "predicted_deltas.csv")
open(pred_path, "w") do io
    println(io, "timestamp,predicted_delta")
    for (ts, d) in zip(pred_timestamps, pred_deltas)
        @printf(io, "%s,%.4f\n", ts, d)
    end
end
println("  Saved to: $pred_path")

# Print summary + table
if !isempty(pred_deltas)
    @printf("  n=%d  mean=%.3f  std=%.3f  min=%.3f  max=%.3f\n",
        length(pred_deltas), mean(pred_deltas), std(pred_deltas),
        minimum(pred_deltas), maximum(pred_deltas))
    @printf("  Baseline fixed delta: %.3f\n", PUT_DELTA)
    println("\n  Date                 Delta   vs 0.16")
    println("  ", "-" ^ 42)
    for (ts, d) in zip(pred_timestamps, pred_deltas)
        diff = d - PUT_DELTA
        @printf("  %-22s %.4f  %+.4f\n", ts, d, diff)
    end
end

# =============================================================================
# 7. Save model and results
# =============================================================================

using BSON
model_path = joinpath(run_dir, "model.bson")
BSON.@save model_path rescaled_model feat_means feat_stds
println("\nModel saved to: $model_path")

open(joinpath(run_dir, "loss_history.csv"), "w") do io
    println(io, "epoch,train_loss,val_loss")
    for (i, (tl, vl)) in enumerate(zip(history.train_loss, history.val_loss))
        @printf(io, "%d,%.6f,%.6f\n", i, tl, vl)
    end
end

println("Done.")
