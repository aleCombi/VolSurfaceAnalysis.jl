using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, DataFrames, Flux

# =============================================================================
# Parameters
# =============================================================================

ENTRY_TIME      = Time(10, 0)
EXPIRY_INTERVAL = Day(1)
SPREAD_LAMBDA   = 0.5
START_DATE      = Date(2024, 2, 1)
END_DATE        = Date(2025, 12, 31)

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
run_dir = joinpath(@__DIR__, "runs", "ml_train_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")

# =============================================================================
# 1. Load data
# =============================================================================

println("\n--- Loading data ---")
surfaces, _, settlement_spots = load_surfaces_and_spots(
    START_DATE, END_DATE;
    symbol=SYMBOL,
    spot_symbol=SPOT_SYMBOL,
    spot_multiplier=SPOT_MULTIPLIER,
    entry_times=ENTRY_TIME,
    spread_lambda=SPREAD_LAMBDA,
    expiry_interval=EXPIRY_INTERVAL
)
schedule = sort(collect(keys(surfaces)))
println("  $(length(schedule)) entry timestamps loaded")

source = DictDataSource(surfaces, settlement_spots)

# =============================================================================
# 2. Generate training data
# =============================================================================

println("\n--- Generating training data ---")
sf = default_surface_features(; rate=RATE, div_yield=DIV_YIELD)
cf = default_candidate_features(; rate=RATE, div_yield=DIV_YIELD)

examples = generate_training_data(
    source, EXPIRY_INTERVAL, schedule;
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
model = create_scoring_model(input_dim=input_dim)
model, feat_means, feat_stds, history = train_scoring_model!(
    model, examples;
    epochs=100, lr=1e-3, batch_size=64, val_fraction=0.2, patience=15
)
println("  Training completed: $(length(history.train_loss)) epochs")
@printf("  Final train loss: %.6f\n", history.train_loss[end])
@printf("  Final val loss:   %.6f\n", history.val_loss[end])

# =============================================================================
# 4. Build ML selector and baseline selector
# =============================================================================

ml_selector = MLCondorSelector(
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
    max_spread_rel=MAX_SPREAD_REL,
    min_delta_gap=MIN_DELTA_GAP
)

# =============================================================================
# 5. Backtest both
# =============================================================================

println("\n--- Backtesting ML selector ---")
ml_strategy = IronCondorStrategy(schedule, EXPIRY_INTERVAL, ml_selector)
ml_result = backtest_strategy(ml_strategy, source)

println("--- Backtesting baseline selector ---")
base_strategy = IronCondorStrategy(schedule, EXPIRY_INTERVAL, baseline_selector)
base_result = backtest_strategy(base_strategy, source)

# =============================================================================
# 6. Compare
# =============================================================================

function compute_metrics(result, label)
    margin = condor_max_loss_by_key(result.positions)
    metrics = performance_metrics(result.positions, result.pnl; margin_by_key=margin)
    return metrics
end

ml_metrics = compute_metrics(ml_result, "ML")
base_metrics = compute_metrics(base_result, "Baseline")

println("\n", "=" ^ 70)
println("  $SYMBOL: ML vs Baseline Comparison")
println("=" ^ 70)
@printf("  %-20s  %12s  %12s\n", "Metric", "ML", "Baseline")
println("  ", "-" ^ 48)
@printf("  %-20s  %12d  %12d\n", "Trades", ml_metrics.count, base_metrics.count)
@printf("  %-20s  %12s  %12s\n", "Total ROI", fmt_metric(ml_metrics.total_roi; pct=true), fmt_metric(base_metrics.total_roi; pct=true))
@printf("  %-20s  %12s  %12s\n", "Mean ROI", fmt_metric(ml_metrics.mean_roi; pct=true), fmt_metric(base_metrics.mean_roi; pct=true))
@printf("  %-20s  %12s  %12s\n", "Sharpe", fmt_ratio(ml_metrics.sharpe), fmt_ratio(base_metrics.sharpe))
@printf("  %-20s  %12s  %12s\n", "Sortino", fmt_ratio(ml_metrics.sortino), fmt_ratio(base_metrics.sortino))
@printf("  %-20s  %12s  %12s\n", "Win Rate", fmt_pct(ml_metrics.win_rate), fmt_pct(base_metrics.win_rate))
println("=" ^ 70)

# =============================================================================
# 7. Save model and results
# =============================================================================

using BSON
model_path = joinpath(run_dir, "model.bson")
BSON.@save model_path model feat_means feat_stds
println("\nModel saved to: $model_path")

# Save loss history
open(joinpath(run_dir, "loss_history.csv"), "w") do io
    println(io, "epoch,train_loss,val_loss")
    for (i, (tl, vl)) in enumerate(zip(history.train_loss, history.val_loss))
        @printf(io, "%d,%.6f,%.6f\n", i, tl, vl)
    end
end

println("Done.")
