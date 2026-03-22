using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, DataFrames, Flux

# =============================================================================
# Parameters
# =============================================================================

ENTRY_TIME      = Time(10, 0)
EXPIRY_INTERVAL = Day(1)
SPREAD_LAMBDA   = 0.5

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
run_dir = joinpath(@__DIR__, "runs", "ml_oos_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")

# =============================================================================
# 1. Build ParquetDataSource (lazy, full spot access for history features)
# =============================================================================

println("\n--- Building ParquetDataSource ---")
store = DEFAULT_STORE

all_dates = available_polygon_dates(store, SYMBOL)
filtered_dates = filter(d -> d >= TRAIN_START && d <= TEST_END, all_dates)
println("  $(length(filtered_dates)) trading days ($(TRAIN_START) to $(TEST_END))")

entry_ts = build_entry_timestamps(filtered_dates, ENTRY_TIME)

# Entry spots needed for Polygon record conversion (synthetic bid/ask)
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
train_schedule = filter(t -> Date(t) <= TRAIN_END, all_timestamps)
test_schedule  = filter(t -> Date(t) >= TEST_START, all_timestamps)
println("  Train: $(length(train_schedule)) timestamps")
println("  Test:  $(length(test_schedule)) timestamps")

# =============================================================================
# 2. Generate training data — TRAIN period only
# =============================================================================

println("\n--- Generating training data (train period only) ---")
sf = default_surface_features(; rate=RATE, div_yield=DIV_YIELD)
cf = default_candidate_features(; rate=RATE, div_yield=DIV_YIELD)

examples = generate_training_data(
    source, EXPIRY_INTERVAL, train_schedule;
    delta_grid=DELTA_GRID,
    rate=RATE, div_yield=DIV_YIELD,
    utility=roi_utility,
    surface_features=sf,
    candidate_features=cf,
    wing_objective=:roi,
    max_loss_max=MAX_LOSS
)
println("  $(length(examples)) training examples generated")

if isempty(examples)
    println("No training data generated. Check data availability.")
    exit(1)
end

input_dim = length(examples[1].surface_features) + length(examples[1].candidate_features)
println("  Feature dimension: $input_dim ($(length(examples[1].surface_features)) surface + $(length(examples[1].candidate_features)) candidate)")

# =============================================================================
# 3. Train model
# =============================================================================

println("\n--- Training model ---")
model = create_scoring_model(input_dim=input_dim, hidden_dims=[16])
model, feat_means, feat_stds, history = train_scoring_model!(
    model, examples;
    epochs=100, lr=1e-3, batch_size=64, val_fraction=0.2, patience=15
)
println("  Training completed: $(length(history.train_loss)) epochs")
@printf("  Final train loss: %.6f\n", history.train_loss[end])
@printf("  Final val loss:   %.6f\n", history.val_loss[end])

# =============================================================================
# 4. Build selectors
# =============================================================================

ml_selector = ScoredCandidateSelector(
    model, feat_means, feat_stds;
    surface_features=sf,
    candidate_features=cf,
    delta_grid=DELTA_GRID,
    rate=RATE, div_yield=DIV_YIELD,
    max_loss=MAX_LOSS,
    max_spread_rel=MAX_SPREAD_REL
)

baseline_selector = constrained_delta_selector(
    PUT_DELTA, CALL_DELTA;
    rate=RATE, div_yield=DIV_YIELD,
    max_loss=MAX_LOSS,
    max_spread_rel=MAX_SPREAD_REL
)

# =============================================================================
# 5. Backtest on TEST period (out-of-sample)
# =============================================================================

function run_backtest(selector, schedule, source, label)
    println("--- Backtesting $label ($(length(schedule)) dates) ---")
    strategy = IronCondorStrategy(schedule, EXPIRY_INTERVAL, selector)
    result = backtest_strategy(strategy, source)
    metrics = performance_metrics(result)
    return (result=result, metrics=metrics)
end

function print_comparison(ml_m, base_m, title)
    println("\n", "=" ^ 70)
    println("  $title")
    println("=" ^ 70)
    @printf("  %-20s  %12s  %12s\n", "Metric", "ML", "Baseline")
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
ml_oos   = run_backtest(ml_selector, test_schedule, source, "ML")
base_oos = run_backtest(baseline_selector, test_schedule, source, "Baseline")
print_comparison(ml_oos.metrics, base_oos.metrics,
    "$SYMBOL OUT-OF-SAMPLE: ML vs Baseline ($(TEST_START) to $(TEST_END))")

println("\n=== IN-SAMPLE REFERENCE ($(TRAIN_START) to $(TRAIN_END)) ===\n")
ml_is   = run_backtest(ml_selector, train_schedule, source, "ML (in-sample)")
base_is = run_backtest(baseline_selector, train_schedule, source, "Baseline (in-sample)")
print_comparison(ml_is.metrics, base_is.metrics,
    "$SYMBOL IN-SAMPLE: ML vs Baseline ($(TRAIN_START) to $(TRAIN_END))")

# =============================================================================
# 6. Save model and results
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
