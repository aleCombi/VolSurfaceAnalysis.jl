# ML Strike Selector Training and Evaluation
# Trains a neural network to learn optimal strike selection for short strangles
# Train on 6 months, test on subsequent 6 months

using Pkg
Pkg.activate(@__DIR__)

using VolSurfaceAnalysis
using Dates
using CSV, DataFrames
using Statistics
using BSON
using Flux
using Printf

# =============================================================================
# Configuration
# =============================================================================
const POLYGON_ROOT = raw"C:\repos\DeribitVols\data\massive_parquet\minute_aggs"
const SPOT_ROOT = raw"C:\repos\DeribitVols\data\massive_parquet\spot_1min"
const UNDERLYING_SYMBOL = "SPY"

# Data periods
const TRAIN_START = Date(2024, 1, 29)
const TRAIN_END = Date(2024, 6, 30)
const TEST_START = Date(2024, 7, 1)
const TEST_END = Date(2024, 12, 31)

# Strategy parameters
const ENTRY_TIME_ET = Time(10, 0)  # 10:00 AM Eastern Time
const EXPIRY_INTERVAL = Day(1)
const RISK_FREE_RATE = 0.045
const DIV_YIELD = 0.013
const QUANTITY = 1.0
const TAU_TOL = 1e-6
const MIN_VOLUME = 5
const SPOT_HISTORY_LOOKBACK_DAYS = 30  # set to `nothing` to use all available history

# Training parameters
const EPOCHS = 100
const BATCH_SIZE = 32
const LEARNING_RATE = 1e-3
const PATIENCE = 15
const HIDDEN_DIMS = [64, 32, 16]
const DROPOUT_RATE = 0.2

# Output paths
const RUN_ID = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
const RUN_DIR = joinpath(@__DIR__, "runs", "ml_strike_selector_$(RUN_ID)")
const MODEL_PATH = joinpath(RUN_DIR, "strike_selector.bson")

# =============================================================================
# Data Loading Helpers
# =============================================================================

function available_dates(root::String, symbol::String)::Vector{Date}
    dirs = readdir(root)
    dates = Date[]
    for dir in dirs
        m = match(r"date=(\d{4})-(\d{2})-(\d{2})", dir)
        m === nothing && continue
        date = Date(parse(Int, m[1]), parse(Int, m[2]), parse(Int, m[3]))
        path = joinpath(root, dir, "underlying=$(symbol)", "data.parquet")
        isfile(path) && push!(dates, date)
    end
    return sort(dates)
end

function available_spot_dates(root::String, symbol::String)::Vector{Date}
    dirs = readdir(root)
    dates = Date[]
    sym = uppercase(symbol)
    for dir in dirs
        m = match(r"date=(\d{4})-(\d{2})-(\d{2})", dir)
        m === nothing && continue
        date = Date(parse(Int, m[1]), parse(Int, m[2]), parse(Int, m[3]))
        path = joinpath(root, dir, "symbol=$sym", "data.parquet")
        isfile(path) && push!(dates, date)
    end
    return sort(dates)
end

function build_entry_timestamps(dates::Vector{Date})::Vector{DateTime}
    ts = DateTime[]
    for date in dates
        push!(ts, et_to_utc(date, ENTRY_TIME_ET))
    end
    return ts
end

function load_minute_spots(
    start_date::Date,
    end_date::Date;
    lookback_days::Union{Nothing,Int}=SPOT_HISTORY_LOOKBACK_DAYS,
    symbol::String=UNDERLYING_SYMBOL
)::Dict{DateTime,Float64}
    all_dates = available_spot_dates(SPOT_ROOT, symbol)
    isempty(all_dates) && error("No spot dates found in $SPOT_ROOT for $symbol")

    min_date = lookback_days === nothing ? minimum(all_dates) : start_date - Day(lookback_days)
    filtered_dates = filter(d -> d >= min_date && d <= end_date, all_dates)

    spots = Dict{DateTime,Float64}()
    for d in filtered_dates
        date_str = Dates.format(d, "yyyy-mm-dd")
        path = joinpath(SPOT_ROOT, "date=$date_str", "symbol=$(uppercase(symbol))", "data.parquet")
        isfile(path) || continue
        dict = read_polygon_spot_prices(path; underlying=symbol)
        merge!(spots, dict)
    end

    return spots
end

function load_surfaces_and_spots(
    start_date::Date,
    end_date::Date;
    symbol::String=UNDERLYING_SYMBOL
)
    println("  Loading dates from $start_date to $end_date...")
    all_dates = available_dates(POLYGON_ROOT, symbol)
    filtered_dates = filter(d -> d >= start_date && d <= end_date, all_dates)

    if isempty(filtered_dates)
        error("No dates found in range $start_date to $end_date")
    end
    println("  Found $(length(filtered_dates)) trading days")

    entry_ts = build_entry_timestamps(filtered_dates)

    # Load entry spots
    entry_spots = read_polygon_spot_prices_for_timestamps(
        SPOT_ROOT,
        entry_ts;
        symbol=symbol
    )
    println("  Loaded $(length(entry_spots)) entry spots")

    # Build surfaces
    path_for_ts = ts -> begin
        date_str = Dates.format(Date(ts), "yyyy-mm-dd")
        joinpath(POLYGON_ROOT, "date=$date_str", "underlying=$symbol", "data.parquet")
    end
    read_records = (path; where="") -> read_polygon_option_records(
        path,
        entry_spots;
        where=where,
        min_volume=MIN_VOLUME,
        warn=false
    )
    surfaces = build_surfaces_for_timestamps(
        entry_ts;
        path_for_timestamp=path_for_ts,
        read_records=read_records
    )
    println("  Built $(length(surfaces)) surfaces")

    # Load settlement spots (need to compute expiry times first)
    expiry_ts = DateTime[]
    for (ts, surface) in surfaces
        expiries = unique(rec.expiry for rec in surface.records)
        for exp in expiries
            tau = time_to_expiry(exp, ts)
            tau_target = time_to_expiry(ts + EXPIRY_INTERVAL, ts)
            if abs(tau - tau_target) < 0.1  # Within ~36 days
                push!(expiry_ts, exp)
            end
        end
    end
    expiry_ts = unique(expiry_ts)

    settlement_spots = read_polygon_spot_prices_for_timestamps(
        SPOT_ROOT,
        expiry_ts;
        symbol=symbol
    )
    println("  Loaded $(length(settlement_spots)) settlement spots")

    return surfaces, entry_spots, settlement_spots
end

function build_spot_history_dict(
    timestamps::Vector{DateTime},
    all_spots::Dict{DateTime,Float64};
    lookback_days::Union{Nothing,Int}=SPOT_HISTORY_LOOKBACK_DAYS
)::Dict{DateTime,SpotHistory}
    """Build timestamped spot history (minute bars) for each entry timestamp."""
    history_dict = Dict{DateTime,SpotHistory}()

    sorted_pairs = sort(collect(all_spots); by=first)
    isempty(sorted_pairs) && return history_dict

    ts_vec = [p[1] for p in sorted_pairs]
    price_vec = [p[2] for p in sorted_pairs]
    first_ts = ts_vec[1]

    for ts in timestamps
        start_ts = lookback_days === nothing ? first_ts : ts - Day(lookback_days)
        i = searchsortedfirst(ts_vec, start_ts)
        j = searchsortedfirst(ts_vec, ts) - 1  # strictly before ts
        if j >= i && (j - i + 1) >= 2
            history_ts = ts_vec[i:j]
            history_prices = price_vec[i:j]
            history_dict[ts] = SpotHistory(history_ts, history_prices)
        end
    end

    return history_dict
end

# =============================================================================
# Main Training and Evaluation
# =============================================================================

function main()
    mkpath(RUN_DIR)

    println("=" ^ 80)
    println("ML STRIKE SELECTOR - TRAINING AND EVALUATION")
    println("=" ^ 80)
    println("Output directory: $RUN_DIR")
    println()

    # -------------------------------------------------------------------------
    # Load Training Data
    # -------------------------------------------------------------------------
    println("PHASE 1: Loading Training Data")
    println("-" ^ 40)

    train_surfaces, train_entry_spots, train_settlement_spots = load_surfaces_and_spots(
        TRAIN_START, TRAIN_END
    )

    # Build spot history for training timestamps
    train_timestamps = sort(collect(keys(train_surfaces)))
    println("  Loading minute spot history for training...")
    train_minute_spots = load_minute_spots(
        TRAIN_START, TRAIN_END;
        lookback_days=SPOT_HISTORY_LOOKBACK_DAYS
    )
    println("  Loaded $(length(train_minute_spots)) minute spot points")
    train_spot_history = build_spot_history_dict(
        train_timestamps,
        train_minute_spots;
        lookback_days=SPOT_HISTORY_LOOKBACK_DAYS
    )
    println("  Built spot history for $(length(train_spot_history)) timestamps")
    println()

    # -------------------------------------------------------------------------
    # Generate Training Labels
    # -------------------------------------------------------------------------
    println("PHASE 2: Generating Training Labels (Grid Search)")
    println("-" ^ 40)
    println("  This may take a while...")

    train_data = generate_training_data(
        train_surfaces,
        train_settlement_spots,
        train_spot_history;
        rate=RISK_FREE_RATE,
        div_yield=DIV_YIELD,
        expiry_interval=EXPIRY_INTERVAL,
        verbose=true
    )
    println()
    println("  Training samples: $(length(train_data.timestamps))")
    println("  Feature matrix: $(size(train_data.features))")
    println("  Label range: put_delta=[$(minimum(train_data.raw_deltas[1,:])), $(maximum(train_data.raw_deltas[1,:]))]")
    println("              call_delta=[$(minimum(train_data.raw_deltas[2,:])), $(maximum(train_data.raw_deltas[2,:]))]")
    println()

    # -------------------------------------------------------------------------
    # Load Test Data
    # -------------------------------------------------------------------------
    println("PHASE 3: Loading Test Data")
    println("-" ^ 40)

    test_surfaces, test_entry_spots, test_settlement_spots = load_surfaces_and_spots(
        TEST_START, TEST_END
    )

    test_timestamps = sort(collect(keys(test_surfaces)))
    println("  Loading minute spot history for testing...")
    test_minute_spots = load_minute_spots(
        TEST_START, TEST_END;
        lookback_days=SPOT_HISTORY_LOOKBACK_DAYS
    )
    println("  Loaded $(length(test_minute_spots)) minute spot points")
    test_spot_history = build_spot_history_dict(
        test_timestamps,
        test_minute_spots;
        lookback_days=SPOT_HISTORY_LOOKBACK_DAYS
    )
    println("  Built spot history for $(length(test_spot_history)) timestamps")
    println()

    # Generate test labels (for evaluation)
    println("  Generating test labels...")
    test_data = generate_training_data(
        test_surfaces,
        test_settlement_spots,
        test_spot_history;
        rate=RISK_FREE_RATE,
        div_yield=DIV_YIELD,
        expiry_interval=EXPIRY_INTERVAL,
        verbose=true
    )
    println()
    println("  Test samples: $(length(test_data.timestamps))")
    println()

    # -------------------------------------------------------------------------
    # Normalize Features
    # -------------------------------------------------------------------------
    println("PHASE 4: Normalizing Features")
    println("-" ^ 40)

    X_train_norm, feature_means, feature_stds = normalize_features(train_data.features)
    X_test_norm = apply_normalization(test_data.features, feature_means, feature_stds)

    # Create normalized training data
    train_data_norm = TrainingDataset(
        X_train_norm,
        train_data.labels,
        train_data.raw_deltas,
        train_data.pnls,
        train_data.timestamps
    )
    test_data_norm = TrainingDataset(
        X_test_norm,
        test_data.labels,
        test_data.raw_deltas,
        test_data.pnls,
        test_data.timestamps
    )

    println("  Feature means: $(round.(feature_means, digits=4))")
    println("  Feature stds:  $(round.(feature_stds, digits=4))")
    println()

    # -------------------------------------------------------------------------
    # Train Model
    # -------------------------------------------------------------------------
    println("PHASE 5: Training Neural Network")
    println("-" ^ 40)
    println("  Architecture: $(N_FEATURES) -> $(HIDDEN_DIMS) -> 2")
    println("  Epochs: $EPOCHS, Batch size: $BATCH_SIZE, LR: $LEARNING_RATE")
    println()

    model = create_strike_model(
        input_dim=N_FEATURES,
        hidden_dims=HIDDEN_DIMS,
        output_dim=2,
        dropout_rate=DROPOUT_RATE
    )

    model, history = train_model!(
        model,
        train_data_norm;
        val_data=test_data_norm,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        patience=PATIENCE,
        verbose=true
    )
    println()

    # Save model
    save_ml_selector(MODEL_PATH, model, feature_means, feature_stds)
    println("  Model saved to: $MODEL_PATH")
    println()

    # -------------------------------------------------------------------------
    # Evaluate Model
    # -------------------------------------------------------------------------
    println("PHASE 6: Evaluating Model")
    println("-" ^ 40)

    train_eval = evaluate_model(model, train_data_norm)
    test_eval = evaluate_model(model, test_data_norm)

    println("  Training Set:")
    println("    MSE: $(round(train_eval["mse"], digits=6))")
    println("    MAE: $(round(train_eval["mae"], digits=4))")

    println("  Test Set:")
    println("    MSE: $(round(test_eval["mse"], digits=6))")
    println("    MAE: $(round(test_eval["mae"], digits=4))")
    println()

    # -------------------------------------------------------------------------
    # Backtest Comparison
    # -------------------------------------------------------------------------
    println("PHASE 7: Backtest Comparison (Test Period)")
    println("-" ^ 40)

    # Create ML selector
    ml_selector = MLStrikeSelector(
        model, feature_means, feature_stds;
        min_delta=0.05f0,
        max_delta=0.35f0,
        rate=RISK_FREE_RATE,
        div_yield=DIV_YIELD,
        spot_history=test_spot_history
    )

    # Create strategies
    schedule = sort(collect(keys(test_surfaces)))

    # ML Strategy
    strategy_ml = ShortStrangleStrategy(
        schedule,
        EXPIRY_INTERVAL,
        1.0;  # sigmas not used when selector provided
        rate=RISK_FREE_RATE,
        div_yield=DIV_YIELD,
        quantity=QUANTITY,
        tau_tol=TAU_TOL,
        debug=false,
        strike_selector=ml_selector
    )

    # Baseline: Fixed 15-delta
    baseline_selector = ctx -> VolSurfaceAnalysis._delta_strangle_strikes(
        ctx, 0.15; rate=RISK_FREE_RATE, div_yield=DIV_YIELD
    )
    strategy_baseline = ShortStrangleStrategy(
        schedule,
        EXPIRY_INTERVAL,
        1.0;
        rate=RISK_FREE_RATE,
        div_yield=DIV_YIELD,
        quantity=QUANTITY,
        tau_tol=TAU_TOL,
        debug=false,
        strike_selector=baseline_selector
    )

    # Baseline: 0.8 Sigma
    strategy_sigma = ShortStrangleStrategy(
        schedule,
        EXPIRY_INTERVAL,
        0.8;
        rate=RISK_FREE_RATE,
        div_yield=DIV_YIELD,
        quantity=QUANTITY,
        tau_tol=TAU_TOL,
        debug=false,
        strike_selector=nothing
    )

    # Run backtests
    println("  Running ML strategy backtest...")
    pos_ml, pnl_ml = backtest_strategy(strategy_ml, test_surfaces, test_settlement_spots)

    println("  Running baseline (15-delta) backtest...")
    pos_baseline, pnl_baseline = backtest_strategy(strategy_baseline, test_surfaces, test_settlement_spots)

    println("  Running baseline (0.8 sigma) backtest...")
    pos_sigma, pnl_sigma = backtest_strategy(strategy_sigma, test_surfaces, test_settlement_spots)
    println()

    # Compute metrics
    margin = 12000.0
    metrics_ml = performance_metrics(pos_ml, pnl_ml; margin_per_trade=margin)
    metrics_baseline = performance_metrics(pos_baseline, pnl_baseline; margin_per_trade=margin)
    metrics_sigma = performance_metrics(pos_sigma, pnl_sigma; margin_per_trade=margin)

    # -------------------------------------------------------------------------
    # Results Summary
    # -------------------------------------------------------------------------
    println("=" ^ 80)
    println("RESULTS SUMMARY")
    println("=" ^ 80)
    println()
    println("Test Period: $TEST_START to $TEST_END")
    println("Margin per trade: \$$(Int(margin))")
    println()

    println("-" ^ 70)
    println("Strategy             | Total P&L  |  Sharpe  | Win Rate |   Avg P&L")
    println("-" ^ 70)

    function fmt_pnl(v)
        ismissing(v) ? "n/a" : @sprintf("\$%.0f", v)
    end
    function fmt_sharpe(v)
        ismissing(v) ? "n/a" : @sprintf("%.2f", v)
    end
    function fmt_winrate(v)
        ismissing(v) ? "n/a" : @sprintf("%.1f%%", v * 100)
    end
    function fmt_avgpnl(v)
        ismissing(v) ? "n/a" : @sprintf("\$%.2f", v)
    end

    println("ML Selector          | $(rpad(fmt_pnl(metrics_ml.total_pnl), 10)) | $(rpad(fmt_sharpe(metrics_ml.sharpe), 8)) | $(rpad(fmt_winrate(metrics_ml.win_rate), 8)) | $(fmt_avgpnl(metrics_ml.avg_pnl))")
    println("Fixed 15-Delta       | $(rpad(fmt_pnl(metrics_baseline.total_pnl), 10)) | $(rpad(fmt_sharpe(metrics_baseline.sharpe), 8)) | $(rpad(fmt_winrate(metrics_baseline.win_rate), 8)) | $(fmt_avgpnl(metrics_baseline.avg_pnl))")
    println("0.8 Sigma            | $(rpad(fmt_pnl(metrics_sigma.total_pnl), 10)) | $(rpad(fmt_sharpe(metrics_sigma.sharpe), 8)) | $(rpad(fmt_winrate(metrics_sigma.win_rate), 8)) | $(fmt_avgpnl(metrics_sigma.avg_pnl))")
    println("-" ^ 70)
    println()

    # Save results
    results_df = DataFrame(
        Strategy = ["ML Selector", "Fixed 15-Delta", "0.8 Sigma"],
        TotalPnL = [metrics_ml.total_pnl, metrics_baseline.total_pnl, metrics_sigma.total_pnl],
        AvgPnL = [metrics_ml.avg_pnl, metrics_baseline.avg_pnl, metrics_sigma.avg_pnl],
        WinRate = [metrics_ml.win_rate, metrics_baseline.win_rate, metrics_sigma.win_rate],
        Sharpe = [metrics_ml.sharpe, metrics_baseline.sharpe, metrics_sigma.sharpe],
        Sortino = [metrics_ml.sortino, metrics_baseline.sortino, metrics_sigma.sortino],
        Count = [metrics_ml.count, metrics_baseline.count, metrics_sigma.count]
    )
    results_path = joinpath(RUN_DIR, "comparison_results.csv")
    CSV.write(results_path, results_df)
    println("Results saved to: $results_path")

    # Save training history
    history_df = DataFrame(
        Epoch = 1:length(history["train_loss"]),
        TrainLoss = history["train_loss"],
        ValLoss = history["val_loss"]
    )
    history_path = joinpath(RUN_DIR, "training_history.csv")
    CSV.write(history_path, history_df)
    println("Training history saved to: $history_path")

    # Save predicted vs actual deltas for analysis
    predictions_df = DataFrame(
        Timestamp = test_data.timestamps,
        TruePutDelta = test_data.raw_deltas[1, :],
        TrueCallDelta = test_data.raw_deltas[2, :],
        PredPutDelta = test_eval["pred_deltas"][1, :],
        PredCallDelta = test_eval["pred_deltas"][2, :],
        BestPnL = test_data.pnls
    )
    predictions_path = joinpath(RUN_DIR, "predictions.csv")
    CSV.write(predictions_path, predictions_df)
    println("Predictions saved to: $predictions_path")

    println()
    println("=" ^ 80)
    println("DONE")
    println("=" ^ 80)
end

main()
