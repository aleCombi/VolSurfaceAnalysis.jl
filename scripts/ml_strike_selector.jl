# ML Strike Selector Training and Evaluation
# Trains a neural network to learn optimal strike selection for short strangles
# Train on 6 months, validate on subsequent period

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
const UNDERLYING_SYMBOL = "SPY"
const STRATEGY = "condor"  # "strangle" or "condor"

# Data periods
const TRAIN_START = Date(2024, 1, 29)
const TRAIN_END = Date(2024, 6, 30)
const VAL_START = Date(2024, 7, 1)
const VAL_END = Date(2025, 6, 1)

# Strategy parameters
# Multiple entry times for training (more data), single time for validation (fair comparison)
const TRAIN_ENTRY_TIMES_ET = [Time(10, 0), Time(11, 0), Time(12, 0), Time(13, 0), Time(14, 0)]
const VAL_ENTRY_TIME_ET = Time(10, 0)  # 10:00 AM Eastern Time
const EXPIRY_INTERVAL = Day(1)
const RISK_FREE_RATE = 0.045
const DIV_YIELD = 0.013
const QUANTITY = 1.0
const TAU_TOL = 1e-6
const MIN_VOLUME = 5
const SPREAD_LAMBDA = 0.5
const SPOT_HISTORY_LOOKBACK_DAYS = 5  # set to `nothing` to use all available history
const USE_LOGSIG = true
const SAVE_PLOTS = true
const SHORT_DELTA_ABS = 0.16
const WING_DELTA_ABS = 0.10
const MIN_DELTA_GAP = 0.01
const SHORT_SIGMAS = 0.7
const LONG_SIGMAS = 1.5
const TARGET_MAX_LOSS = 10.0  # Target maximum loss per contract in dollars (e.g., 10.0 = $1000 max loss)
const PREFER_SYMMETRIC_WINGS = false

# Training parameters
const EPOCHS = 100
const BATCH_SIZE = 32
const LEARNING_RATE = 1e-3
const PATIENCE = 15
const HIDDEN_DIMS = [64, 32, 16]
const DROPOUT_RATE = 0.2

# Output paths
const RUN_ID = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
const RUN_DIR    = joinpath(@__DIR__, "runs", "ml_strike_selector_$(RUN_ID)")
const LATEST_DIR = joinpath(@__DIR__, "latest_runs", "ml_strike_selector")
const MODEL_PATH = joinpath(RUN_DIR, "strike_selector.bson")

# =============================================================================
# Data Loading Helpers
# =============================================================================

function build_entry_timestamps(dates::Vector{Date}, entry_times::Vector{Time})::Vector{DateTime}
    ts = DateTime[]
    for date in dates
        for t in entry_times
            push!(ts, et_to_utc(date, t))
        end
    end
    return sort(ts)
end

# Convenience for single time
function build_entry_timestamps(dates::Vector{Date}, entry_time::Time)::Vector{DateTime}
    return build_entry_timestamps(dates, [entry_time])
end

function load_minute_spots(
    start_date::Date,
    end_date::Date;
    lookback_days::Union{Nothing,Int}=SPOT_HISTORY_LOOKBACK_DAYS,
    symbol::String=UNDERLYING_SYMBOL
)::Dict{DateTime,Float64}
    all_dates = available_polygon_dates(DEFAULT_STORE, symbol)
    isempty(all_dates) && error("No spot dates found for $symbol")

    min_date = lookback_days === nothing ? minimum(all_dates) : start_date - Day(lookback_days)
    filtered_dates = filter(d -> d >= min_date && d <= end_date, all_dates)

    spots = Dict{DateTime,Float64}()
    for d in filtered_dates
        path = polygon_spot_path(DEFAULT_STORE, d, symbol)
        isfile(path) || continue
        dict = read_polygon_spot_prices(path; underlying=symbol)
        merge!(spots, dict)
    end

    return spots
end

function load_surfaces_and_spots(
    start_date::Date,
    end_date::Date;
    symbol::String=UNDERLYING_SYMBOL,
    entry_times::Union{Time,Vector{Time}}=VAL_ENTRY_TIME_ET
)
    println("  Loading dates from $start_date to $end_date...")
    all_dates = available_polygon_dates(DEFAULT_STORE, symbol)
    filtered_dates = filter(d -> d >= start_date && d <= end_date, all_dates)

    if isempty(filtered_dates)
        error("No dates found in range $start_date to $end_date")
    end
    n_times = entry_times isa Vector ? length(entry_times) : 1
    println("  Found $(length(filtered_dates)) trading days x $(n_times) entry times")

    entry_ts = build_entry_timestamps(filtered_dates, entry_times isa Vector ? entry_times : [entry_times])

    # Load entry spots
    entry_spots = read_polygon_spot_prices_for_timestamps(
        polygon_spot_root(DEFAULT_STORE),
        entry_ts;
        symbol=symbol
    )
    println("  Loaded $(length(entry_spots)) entry spots")

    # Build surfaces
    path_for_ts = ts -> polygon_options_path(DEFAULT_STORE, Date(ts), symbol)
    read_records = (path; where="") -> read_polygon_option_records(
        path,
        entry_spots;
        where=where,
        min_volume=MIN_VOLUME,
        warn=false,
        spread_lambda=SPREAD_LAMBDA
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
        polygon_spot_root(DEFAULT_STORE),
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
    strategy_label = STRATEGY == "condor" ? "ML IRON CONDOR SELECTOR" : "ML STRIKE SELECTOR"
    println("$strategy_label - TRAINING AND EVALUATION")
    println("=" ^ 80)
    println("Output directory: $RUN_DIR")
    println()

    # -------------------------------------------------------------------------
    # Load Training Data
    # -------------------------------------------------------------------------
    println("PHASE 1: Loading Training Data")
    println("-" ^ 40)

    train_surfaces, train_entry_spots, train_settlement_spots = load_surfaces_and_spots(
        TRAIN_START, TRAIN_END;
        entry_times=TRAIN_ENTRY_TIMES_ET
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

    train_data = if STRATEGY == "condor"
        generate_condor_training_data(
            train_surfaces,
            train_settlement_spots,
            train_spot_history;
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            expiry_interval=EXPIRY_INTERVAL,
            wing_delta_abs=nothing,
            target_max_loss=TARGET_MAX_LOSS,
            min_delta_gap=MIN_DELTA_GAP,
            prefer_symmetric=PREFER_SYMMETRIC_WINGS,
            use_logsig=USE_LOGSIG,
            verbose=true
        )
    else
        generate_training_data(
            train_surfaces,
            train_settlement_spots,
            train_spot_history;
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            expiry_interval=EXPIRY_INTERVAL,
            use_logsig=USE_LOGSIG,
            verbose=true
        )
    end
    println()
    println("  Training samples: $(length(train_data.timestamps))")
    println("  Feature matrix: $(size(train_data.features))")
    println("  Label range: put_delta=[$(minimum(train_data.raw_deltas[1,:])), $(maximum(train_data.raw_deltas[1,:]))]")
    println("              call_delta=[$(minimum(train_data.raw_deltas[2,:])), $(maximum(train_data.raw_deltas[2,:]))]")
    println()

    # -------------------------------------------------------------------------
    # Load Validation Data
    # -------------------------------------------------------------------------
    println("PHASE 3: Loading Validation Data")
    println("-" ^ 40)

    val_surfaces, val_entry_spots, val_settlement_spots = load_surfaces_and_spots(
        VAL_START, VAL_END;
        entry_times=VAL_ENTRY_TIME_ET
    )

    val_timestamps = sort(collect(keys(val_surfaces)))
    println("  Loading minute spot history for validation...")
    val_minute_spots = load_minute_spots(
        VAL_START, VAL_END;
        lookback_days=SPOT_HISTORY_LOOKBACK_DAYS
    )
    println("  Loaded $(length(val_minute_spots)) minute spot points")
    val_spot_history = build_spot_history_dict(
        val_timestamps,
        val_minute_spots;
        lookback_days=SPOT_HISTORY_LOOKBACK_DAYS
    )
    println("  Built spot history for $(length(val_spot_history)) timestamps")
    println()

    # Generate validation labels (for evaluation)
    println("  Generating validation labels...")
    val_data = if STRATEGY == "condor"
        generate_condor_training_data(
            val_surfaces,
            val_settlement_spots,
            val_spot_history;
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            expiry_interval=EXPIRY_INTERVAL,
            wing_delta_abs=nothing,
            target_max_loss=TARGET_MAX_LOSS,
            min_delta_gap=MIN_DELTA_GAP,
            prefer_symmetric=PREFER_SYMMETRIC_WINGS,
            use_logsig=USE_LOGSIG,
            verbose=true
        )
    else
        generate_training_data(
            val_surfaces,
            val_settlement_spots,
            val_spot_history;
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            expiry_interval=EXPIRY_INTERVAL,
            use_logsig=USE_LOGSIG,
            verbose=true
        )
    end
    println()
    println("  Validation samples: $(length(val_data.timestamps))")
    println()

    # -------------------------------------------------------------------------
    # Normalize Features
    # -------------------------------------------------------------------------
    println("PHASE 4: Normalizing Features")
    println("-" ^ 40)

    X_train_norm, feature_means, feature_stds = normalize_features(train_data.features)
    X_val_norm = apply_normalization(val_data.features, feature_means, feature_stds)

    # Create normalized training data
    train_data_norm = TrainingDataset(
        X_train_norm,
        train_data.labels,
        train_data.raw_deltas,
        train_data.pnls,
        train_data.size_labels,
        train_data.timestamps
    )
    val_data_norm = TrainingDataset(
        X_val_norm,
        val_data.labels,
        val_data.raw_deltas,
        val_data.pnls,
        val_data.size_labels,
        val_data.timestamps
    )

    println("  Feature means: $(round.(feature_means, digits=4))")
    println("  Feature stds:  $(round.(feature_stds, digits=4))")
    println()

    # -------------------------------------------------------------------------
    # Train Model
    # -------------------------------------------------------------------------
    println("PHASE 5: Training Neural Network")
    println("-" ^ 40)
    println("  Architecture: $(N_FEATURES) -> $(HIDDEN_DIMS) -> 3 (deltas + sizing)")
    println("  Epochs: $EPOCHS, Batch size: $BATCH_SIZE, LR: $LEARNING_RATE")
    println()

    model = create_strike_model(
        input_dim=N_FEATURES,
        hidden_dims=HIDDEN_DIMS,
        output_dim=3,  # put_delta, call_delta, position_size
        dropout_rate=DROPOUT_RATE
    )

    model, history = train_model!(
        model,
        train_data_norm;
        val_data=val_data_norm,
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
    val_eval = evaluate_model(model, val_data_norm)

    println("  Training Set:")
    println("    Delta MSE: $(round(train_eval["delta_mse"], digits=6)), MAE: $(round(train_eval["delta_mae"], digits=4))")
    println("    Size  MSE: $(round(train_eval["size_mse"], digits=6)), MAE: $(round(train_eval["size_mae"], digits=4))")

    println("  Validation Set:")
    println("    Delta MSE: $(round(val_eval["delta_mse"], digits=6)), MAE: $(round(val_eval["delta_mae"], digits=4))")
    println("    Size  MSE: $(round(val_eval["size_mse"], digits=6)), MAE: $(round(val_eval["size_mae"], digits=4))")
    println()

    # -------------------------------------------------------------------------
    # Backtest Comparison
    # -------------------------------------------------------------------------
    println("PHASE 7: Backtest Comparison (Validation Period)")
    println("-" ^ 40)

    # Create ML selector
    ml_selector = if STRATEGY == "condor"
        MLCondorStrikeSelector(
            model, feature_means, feature_stds;
            min_delta=0.05f0,
            max_delta=0.35f0,
            wing_delta_abs=nothing,  # Disable fixed delta wings
            target_max_loss=Float32(TARGET_MAX_LOSS),  # Use max loss wings
            min_delta_gap=Float32(MIN_DELTA_GAP),
            prefer_symmetric=PREFER_SYMMETRIC_WINGS,
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            spot_history=val_spot_history,
            use_logsig=USE_LOGSIG
        )
    else
        MLStrikeSelector(
            model, feature_means, feature_stds;
            min_delta=0.05f0,
            max_delta=0.35f0,
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            spot_history=val_spot_history,
            use_logsig=USE_LOGSIG
        )
    end

    # Create strategies
    schedule = sort(collect(keys(val_surfaces)))

    strategy_ml = if STRATEGY == "condor"
        IronCondorStrategy(
            schedule,
            EXPIRY_INTERVAL,
            SHORT_SIGMAS,
            LONG_SIGMAS;
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            quantity=QUANTITY,
            tau_tol=TAU_TOL,
            debug=false,
            strike_selector=ml_selector
        )
    else
        ShortStrangleStrategy(
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
    end

    strategy_baseline = if STRATEGY == "condor"
        baseline_selector = ctx -> VolSurfaceAnalysis._delta_condor_strikes(
            ctx,
            SHORT_DELTA_ABS,
            SHORT_DELTA_ABS,
            WING_DELTA_ABS,
            WING_DELTA_ABS;
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            min_delta_gap=MIN_DELTA_GAP,
            debug=false
        )
        IronCondorStrategy(
            schedule,
            EXPIRY_INTERVAL,
            SHORT_SIGMAS,
            LONG_SIGMAS;
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            quantity=QUANTITY,
            tau_tol=TAU_TOL,
            debug=false,
            strike_selector=baseline_selector
        )
    else
        baseline_selector = ctx -> VolSurfaceAnalysis._delta_strangle_strikes(
            ctx, 0.15; rate=RISK_FREE_RATE, div_yield=DIV_YIELD
        )
        ShortStrangleStrategy(
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
    end

    strategy_sigma = if STRATEGY == "condor"
        IronCondorStrategy(
            schedule,
            EXPIRY_INTERVAL,
            SHORT_SIGMAS,
            LONG_SIGMAS;
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            quantity=QUANTITY,
            tau_tol=TAU_TOL,
            debug=false,
            strike_selector=nothing
        )
    else
        ShortStrangleStrategy(
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
    end

    # Run backtests
    println("  Running ML strategy backtest...")
    pos_ml, pnl_ml = backtest_strategy(strategy_ml, val_surfaces, val_settlement_spots)

    println("  Running baseline (fixed delta) backtest...")
    pos_baseline, pnl_baseline = backtest_strategy(strategy_baseline, val_surfaces, val_settlement_spots)

    println("  Running baseline (0.8 sigma) backtest...")
    pos_sigma, pnl_sigma = backtest_strategy(strategy_sigma, val_surfaces, val_settlement_spots)
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
    println("Validation Period: $VAL_START to $VAL_END")
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

    ml_label = STRATEGY == "condor" ? "ML Condor" : "ML Selector"
    base_label = STRATEGY == "condor" ? "Fixed Delta Condor" : "Fixed 15-Delta"
    sigma_label = STRATEGY == "condor" ? "Sigma Condor" : "0.8 Sigma"
    println("$(rpad(ml_label, 20)) | $(rpad(fmt_pnl(metrics_ml.total_pnl), 10)) | $(rpad(fmt_sharpe(metrics_ml.sharpe), 8)) | $(rpad(fmt_winrate(metrics_ml.win_rate), 8)) | $(fmt_avgpnl(metrics_ml.avg_pnl))")
    println("$(rpad(base_label, 20)) | $(rpad(fmt_pnl(metrics_baseline.total_pnl), 10)) | $(rpad(fmt_sharpe(metrics_baseline.sharpe), 8)) | $(rpad(fmt_winrate(metrics_baseline.win_rate), 8)) | $(fmt_avgpnl(metrics_baseline.avg_pnl))")
    println("$(rpad(sigma_label, 20)) | $(rpad(fmt_pnl(metrics_sigma.total_pnl), 10)) | $(rpad(fmt_sharpe(metrics_sigma.sharpe), 8)) | $(rpad(fmt_winrate(metrics_sigma.win_rate), 8)) | $(fmt_avgpnl(metrics_sigma.avg_pnl))")
    println("-" ^ 70)
    println()

    # Save results
    results_df = DataFrame(
        Strategy = [ml_label, base_label, sigma_label],
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

    # Condor-specific per-trade report
    if STRATEGY == "condor"
        condor_df = condor_trade_table(pos_ml, pnl_ml)
        condor_path = joinpath(RUN_DIR, "condor_trades.csv")
        CSV.write(condor_path, condor_df)
        println("Condor trade report saved to: $condor_path")

        pnl_vals = collect(skipmissing(condor_df.PnL))
        max_loss_vals = collect(skipmissing(condor_df.MaxLoss))
        credit_vals = collect(skipmissing(condor_df.Credit))
        ror_vals = collect(skipmissing(condor_df.ReturnOnRisk))

        condor_summary = DataFrame(
            Metric = [
                "count",
                "avg_pnl_per_condor",
                "min_pnl_per_condor",
                "max_pnl_per_condor",
                "win_rate",
                "avg_credit",
                "avg_max_loss",
                "avg_return_on_risk"
            ],
            Value = [
                nrow(condor_df),
                isempty(pnl_vals) ? missing : mean(pnl_vals),
                isempty(pnl_vals) ? missing : minimum(pnl_vals),
                isempty(pnl_vals) ? missing : maximum(pnl_vals),
                isempty(pnl_vals) ? missing : count(x -> x > 0, pnl_vals) / length(pnl_vals),
                isempty(credit_vals) ? missing : mean(credit_vals),
                isempty(max_loss_vals) ? missing : mean(max_loss_vals),
                isempty(ror_vals) ? missing : mean(ror_vals)
            ]
        )
        summary_path = joinpath(RUN_DIR, "condor_summary.csv")
        CSV.write(summary_path, condor_summary)
        println("Condor summary saved to: $summary_path")
    end

    # Save training history
    history_df = DataFrame(
        Epoch = 1:length(history["train_loss"]),
        TrainLoss = history["train_loss"],
        ValLoss = history["val_loss"],
        TrainDeltaLoss = history["train_delta_loss"],
        TrainSizeLoss = history["train_size_loss"]
    )
    history_path = joinpath(RUN_DIR, "training_history.csv")
    CSV.write(history_path, history_df)
    println("Training history saved to: $history_path")

    # Save predicted vs actual deltas and sizes for analysis
    predictions_df = DataFrame(
        Timestamp = val_data.timestamps,
        TruePutDelta = val_data.raw_deltas[1, :],
        TrueCallDelta = val_data.raw_deltas[2, :],
        PredPutDelta = val_eval["pred_deltas"][1, :],
        PredCallDelta = val_eval["pred_deltas"][2, :],
        TrueSize = val_data.size_labels,
        PredSize = val_eval["pred_sizes"],
        BestPnL = val_data.pnls
    )
    predictions_path = joinpath(RUN_DIR, "predictions.csv")
    CSV.write(predictions_path, predictions_df)
    println("Predictions saved to: $predictions_path")

    # Save ML positions with P&L details
    positions_data = []
    for (pos, pnl_val) in zip(pos_ml, pnl_ml)
        entry_price_usd = pos.entry_price * pos.entry_spot
        entry_bid_usd = ismissing(pos.entry_bid) ? missing : pos.entry_bid * pos.entry_spot
        entry_ask_usd = ismissing(pos.entry_ask) ? missing : pos.entry_ask * pos.entry_spot
        entry_cost_usd = entry_cost(pos)
        push!(positions_data, (
            EntryTimestamp = pos.entry_timestamp,
            Expiry = pos.trade.expiry,
            OptionType = string(pos.trade.option_type),
            Strike = pos.trade.strike,
            Direction = pos.trade.direction,
            Quantity = pos.trade.quantity,
            EntrySpot = pos.entry_spot,
            EntryPrice = pos.entry_price,
            EntryBid = pos.entry_bid,
            EntryAsk = pos.entry_ask,
            EntryPriceUsd = entry_price_usd,
            EntryBidUsd = entry_bid_usd,
            EntryAskUsd = entry_ask_usd,
            EntryCostUsd = entry_cost_usd,
            PnL = pnl_val
        ))
    end
    positions_df = DataFrame(positions_data)
    positions_path = joinpath(RUN_DIR, "ml_positions.csv")
    CSV.write(positions_path, positions_df)
    println("ML positions saved to: $positions_path")

    # Save plots for the ML strategy
    if SAVE_PLOTS
        pnl_by_key, _ = aggregate_pnl(pos_ml, pnl_ml)
        if !isempty(pnl_by_key)
            keys_sorted = sort(collect(keys(pnl_by_key)); by=k -> k[1])
            trade_dates = Date.(getindex.(keys_sorted, 1))
            trade_pnls = [pnl_by_key[k] for k in keys_sorted]

            plots_dir = joinpath(RUN_DIR, "plots")
            title_prefix = STRATEGY == "condor" ? "ML Condor" : "ML Strangle"

            save_pnl_and_equity_curve(
                trade_dates,
                trade_pnls,
                joinpath(plots_dir, "ml_$(STRATEGY)_pnl_distribution.png");
                title_prefix=title_prefix
            )
            save_profit_curve(
                trade_dates,
                trade_pnls,
                joinpath(plots_dir, "ml_$(STRATEGY)_profit_curve.png");
                title="$title_prefix - Profit per Trade"
            )
        end

        if !isempty(val_entry_spots)
            plots_dir = joinpath(RUN_DIR, "plots")
            save_spot_curve(
                val_entry_spots,
                joinpath(plots_dir, "ml_$(STRATEGY)_spot_curve.png");
                title="Spot Curve $(UNDERLYING_SYMBOL)"
            )
        end
    end

    # Overwrite the no-timestamp "latest" copy
    isdir(LATEST_DIR) && rm(LATEST_DIR; recursive=true)
    cp(RUN_DIR, LATEST_DIR)
    println("Latest run: $LATEST_DIR")

    println()
    println("=" ^ 80)
    println("DONE")
    println("=" ^ 80)
end

main()
