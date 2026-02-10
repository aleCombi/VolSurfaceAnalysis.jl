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

include(joinpath(@__DIR__, "configurations.jl"))

# =============================================================================
# Configuration
# =============================================================================
const UNDERLYING_SYMBOL = "SPXW"
const STRATEGY = "condor"  # "strangle" or "condor"
const MODEL_MODE = :delta   # :delta (current), :score (candidate scoring), :hybrid (score with delta fallback)

# Data periods
const TRAIN_START = Date(2024, 2, 7)
const TRAIN_END = Date(2024, 11, 7)
const VAL_START = Date(2024, 11, 7)
const VAL_END = Date(2025, 2, 7)
const TEST_START = Date(2025, 2, 7)
const TEST_END = Date(2025, 5, 7)

# Strategy parameters
# Multiple entry times for training (more data), single time for validation (fair comparison)
const TRAIN_ENTRY_TIMES_ET = [Time(10, 0), Time(11, 0), Time(12, 0), Time(13, 0), Time(14, 0)]
const VAL_ENTRY_TIME_ET = Time(10, 0)  # 10:00 AM Eastern Time
const TEST_ENTRY_TIME_ET = Time(10, 0)
const EXPIRY_INTERVAL = Day(1)
const QUANTITY = 1.0
const TAU_TOL = 1e-6
const MIN_VOLUME = 0
const DEFAULT_SPREAD_LAMBDA = 0.0  # overridable via ARGS[2]
const SPOT_HISTORY_LOOKBACK_DAYS = 5  # set to `nothing` to use all available history
const USE_LOGSIG = true
const SAVE_PLOTS = true
const SHORT_DELTA_ABS = 0.16
const WING_DELTA_ABS = 0.10
const MIN_DELTA_GAP = 0.01
const SHORT_SIGMAS = 0.7
const LONG_SIGMAS = 1.5
const TARGET_MAX_LOSS = nothing  # Not used with :roi objective
const PREFER_SYMMETRIC_WINGS = false
const CONDOR_WING_OBJECTIVE = :roi  # :target_max_loss, :roi, or :pnl

# Candidate scoring parameters (used when MODEL_MODE == :score or :hybrid)
const SCORE_UTILITY_OBJECTIVE = :roi
const SCORE_DELTA_GRID = collect(0.05:0.015:0.35)
const SCORE_MAX_CANDIDATES_PER_DAY = 400
const SCORE_HIDDEN_DIMS = [128, 64, 32]

# Training parameters
const EPOCHS = 100
const BATCH_SIZE = 32
const LEARNING_RATE = 1e-3
const PATIENCE = 15
const HIDDEN_DIMS = [64, 32, 16]
const DROPOUT_RATE = 0.2

# Output paths (set in main() after symbol is known)

# =============================================================================
# Main Training and Evaluation
# =============================================================================

function parse_args()
    symbol = UNDERLYING_SYMBOL
    if length(ARGS) >= 1
        symbol = uppercase(String(ARGS[1]))
    end
    spread_lambda = DEFAULT_SPREAD_LAMBDA
    if length(ARGS) >= 2
        spread_lambda = parse(Float64, ARGS[2])
    end
    cfg = load_symbol_config(symbol)
    return symbol, spread_lambda, cfg
end

function main()
    symbol, SPREAD_LAMBDA, cfg = parse_args()

    # Output paths (include symbol + lambda so parallel runs don't collide)
    lambda_tag = @sprintf("lambda%.1f", SPREAD_LAMBDA)
    RUN_ID = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    RUN_DIR    = joinpath(@__DIR__, "runs", "ml_strike_selector_$(symbol)_$(lambda_tag)_$(RUN_ID)")
    LATEST_DIR = joinpath(@__DIR__, "latest_runs", "ml_strike_selector_$(symbol)_$(lambda_tag)")
    MODEL_PATH = joinpath(RUN_DIR, "strike_selector.bson")

    # Unpack per-symbol config
    SPOT_SYMBOL = cfg.spot_symbol
    SPOT_MULTIPLIER = cfg.spot_multiplier
    DIV_YIELD = cfg.div_yield
    RISK_FREE_RATE = cfg.risk_free_rate
    CONDOR_MAX_LOSS_MIN = cfg.condor_max_loss_min
    CONDOR_MAX_LOSS_MAX = cfg.condor_max_loss_max
    CONDOR_MIN_CREDIT = cfg.condor_min_credit

    if STRATEGY != "condor" && MODEL_MODE != :delta
        error("MODEL_MODE=$(MODEL_MODE) is only supported for STRATEGY=\"condor\"")
    end
    MODEL_MODE in (:delta, :score, :hybrid) || error("MODEL_MODE must be one of :delta, :score, :hybrid")

    mkpath(RUN_DIR)

    println("=" ^ 80)
    strategy_label = STRATEGY == "condor" ? "ML IRON CONDOR SELECTOR" : "ML STRIKE SELECTOR"
    println("$strategy_label - TRAINING AND EVALUATION")
    println("=" ^ 80)
    println("Underlying: $symbol")
    println("Output directory: $RUN_DIR")
    println("Model mode: $MODEL_MODE")
    println("Training:   $TRAIN_START to $TRAIN_END")
    println("Validation: $VAL_START to $VAL_END (early stopping)")
    println("Test:       $TEST_START to $TEST_END (backtest eval)")
    if MODEL_MODE != :delta && STRATEGY == "condor"
        println("Scoring objective: $SCORE_UTILITY_OBJECTIVE, max candidates/day: $SCORE_MAX_CANDIDATES_PER_DAY")
    end
    println()

    # -------------------------------------------------------------------------
    # Load Training Data
    # -------------------------------------------------------------------------
    println("PHASE 1: Loading Training Data")
    println("-" ^ 40)

    train_surfaces, train_entry_spots, train_settlement_spots = load_surfaces_and_spots(
        TRAIN_START, TRAIN_END;
        symbol=symbol,
        spot_symbol=SPOT_SYMBOL,
        spot_multiplier=SPOT_MULTIPLIER,
        entry_times=TRAIN_ENTRY_TIMES_ET,
        min_volume=MIN_VOLUME,
        spread_lambda=SPREAD_LAMBDA,
        expiry_interval=EXPIRY_INTERVAL
    )

    # Build spot history for training timestamps
    train_timestamps = sort(collect(keys(train_surfaces)))
    println("  Loading minute spot history for training...")
    train_minute_spots = load_minute_spots(
        TRAIN_START, TRAIN_END;
        lookback_days=SPOT_HISTORY_LOOKBACK_DAYS,
        symbol=SPOT_SYMBOL,
        multiplier=SPOT_MULTIPLIER
    )
    println("  Loaded $(length(train_minute_spots)) minute spot points")
    train_spot_history = build_spot_history_dict(
        train_timestamps,
        train_minute_spots;
        lookback_days=SPOT_HISTORY_LOOKBACK_DAYS
    )
    println("  Built spot history for $(length(train_spot_history)) timestamps")

    println("  Building prev-day surface mappings...")
    train_prev_surfaces = build_prev_surfaces_dict(train_surfaces; symbol=symbol)
    println("  Prev-day surfaces: $(length(train_prev_surfaces))/$(length(train_surfaces))")
    println()

    # -------------------------------------------------------------------------
    # Generate Training Labels
    # -------------------------------------------------------------------------
    println("PHASE 2: Generating Training Labels (Grid Search)")
    println("-" ^ 40)
    println("  This may take a while...")

    train_data = if STRATEGY == "condor" && MODEL_MODE != :delta
        generate_condor_candidate_training_data(
            train_surfaces,
            train_settlement_spots,
            train_spot_history;
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            expiry_interval=EXPIRY_INTERVAL,
            utility_objective=SCORE_UTILITY_OBJECTIVE,
            candidate_delta_grid=SCORE_DELTA_GRID,
            max_candidates_per_day=SCORE_MAX_CANDIDATES_PER_DAY,
            wing_delta_abs=nothing,
            target_max_loss=(CONDOR_WING_OBJECTIVE == :target_max_loss ? TARGET_MAX_LOSS : nothing),
            wing_objective=CONDOR_WING_OBJECTIVE,
            max_loss_min=CONDOR_MAX_LOSS_MIN,
            max_loss_max=CONDOR_MAX_LOSS_MAX,
            min_credit=CONDOR_MIN_CREDIT,
            min_delta_gap=MIN_DELTA_GAP,
            prefer_symmetric=PREFER_SYMMETRIC_WINGS,
            use_logsig=USE_LOGSIG,
            prev_surfaces=train_prev_surfaces,
            verbose=true
        )
    elseif STRATEGY == "condor"
        generate_condor_4d_training_data(
            train_surfaces,
            train_settlement_spots,
            train_spot_history;
            min_delta_gap=MIN_DELTA_GAP,
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            expiry_interval=EXPIRY_INTERVAL,
            use_logsig=USE_LOGSIG,
            prev_surfaces=train_prev_surfaces,
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
            prev_surfaces=train_prev_surfaces,
            verbose=true
        )
    end
    println()
    println("  Training samples: $(length(train_data.timestamps))")
    println("  Feature matrix: $(size(train_data.features))")
    if STRATEGY == "condor" && MODEL_MODE != :delta
        println("  Utility range: [$(minimum(train_data.utilities)), $(maximum(train_data.utilities))]")
        println("  Candidate PnL range: [$(minimum(train_data.pnls)), $(maximum(train_data.pnls))]")
    else
        n_label_dims = size(train_data.raw_deltas, 1)
        println("  Label range: short_put_delta=[$(minimum(train_data.raw_deltas[1,:])), $(maximum(train_data.raw_deltas[1,:]))]")
        println("              short_call_delta=[$(minimum(train_data.raw_deltas[2,:])), $(maximum(train_data.raw_deltas[2,:]))]")
        if n_label_dims >= 4
            println("              long_put_delta=[$(minimum(train_data.raw_deltas[3,:])), $(maximum(train_data.raw_deltas[3,:]))]")
            println("              long_call_delta=[$(minimum(train_data.raw_deltas[4,:])), $(maximum(train_data.raw_deltas[4,:]))]")
        end
    end
    println()

    # -------------------------------------------------------------------------
    # Load Validation Data
    # -------------------------------------------------------------------------
    println("PHASE 3: Loading Validation Data")
    println("-" ^ 40)

    val_surfaces, val_entry_spots, val_settlement_spots = load_surfaces_and_spots(
        VAL_START, VAL_END;
        symbol=symbol,
        spot_symbol=SPOT_SYMBOL,
        spot_multiplier=SPOT_MULTIPLIER,
        entry_times=VAL_ENTRY_TIME_ET,
        min_volume=MIN_VOLUME,
        spread_lambda=SPREAD_LAMBDA,
        expiry_interval=EXPIRY_INTERVAL
    )

    val_timestamps = sort(collect(keys(val_surfaces)))
    println("  Loading minute spot history for validation...")
    val_minute_spots = load_minute_spots(
        VAL_START, VAL_END;
        lookback_days=SPOT_HISTORY_LOOKBACK_DAYS,
        symbol=SPOT_SYMBOL,
        multiplier=SPOT_MULTIPLIER
    )
    println("  Loaded $(length(val_minute_spots)) minute spot points")
    val_spot_history = build_spot_history_dict(
        val_timestamps,
        val_minute_spots;
        lookback_days=SPOT_HISTORY_LOOKBACK_DAYS
    )
    println("  Built spot history for $(length(val_spot_history)) timestamps")

    val_prev_surfaces = build_prev_surfaces_dict(val_surfaces; symbol=symbol)
    println("  Built prev-day surface map for $(length(val_prev_surfaces))/$(length(val_surfaces)) timestamps")
    println()

    # Generate validation labels (for evaluation)
    println("  Generating validation labels...")
    val_data = if STRATEGY == "condor" && MODEL_MODE != :delta
        generate_condor_candidate_training_data(
            val_surfaces,
            val_settlement_spots,
            val_spot_history;
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            expiry_interval=EXPIRY_INTERVAL,
            utility_objective=SCORE_UTILITY_OBJECTIVE,
            candidate_delta_grid=SCORE_DELTA_GRID,
            max_candidates_per_day=SCORE_MAX_CANDIDATES_PER_DAY,
            wing_delta_abs=nothing,
            target_max_loss=(CONDOR_WING_OBJECTIVE == :target_max_loss ? TARGET_MAX_LOSS : nothing),
            wing_objective=CONDOR_WING_OBJECTIVE,
            max_loss_min=CONDOR_MAX_LOSS_MIN,
            max_loss_max=CONDOR_MAX_LOSS_MAX,
            min_credit=CONDOR_MIN_CREDIT,
            min_delta_gap=MIN_DELTA_GAP,
            prefer_symmetric=PREFER_SYMMETRIC_WINGS,
            use_logsig=USE_LOGSIG,
            prev_surfaces=val_prev_surfaces,
            verbose=true
        )
    elseif STRATEGY == "condor"
        generate_condor_4d_training_data(
            val_surfaces,
            val_settlement_spots,
            val_spot_history;
            min_delta_gap=MIN_DELTA_GAP,
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            expiry_interval=EXPIRY_INTERVAL,
            use_logsig=USE_LOGSIG,
            prev_surfaces=val_prev_surfaces,
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
            prev_surfaces=val_prev_surfaces,
            verbose=true
        )
    end
    println()
    println("  Validation samples: $(length(val_data.timestamps))")
    println()

    # -------------------------------------------------------------------------
    # Load Test Data
    # -------------------------------------------------------------------------
    println("PHASE 3b: Loading Test Data")
    println("-" ^ 40)

    test_surfaces, test_entry_spots, test_settlement_spots = load_surfaces_and_spots(
        TEST_START, TEST_END;
        symbol=symbol,
        spot_symbol=SPOT_SYMBOL,
        spot_multiplier=SPOT_MULTIPLIER,
        entry_times=TEST_ENTRY_TIME_ET,
        min_volume=MIN_VOLUME,
        spread_lambda=SPREAD_LAMBDA,
        expiry_interval=EXPIRY_INTERVAL
    )

    test_timestamps = sort(collect(keys(test_surfaces)))
    println("  Loading minute spot history for test...")
    test_minute_spots = load_minute_spots(
        TEST_START, TEST_END;
        lookback_days=SPOT_HISTORY_LOOKBACK_DAYS,
        symbol=SPOT_SYMBOL,
        multiplier=SPOT_MULTIPLIER
    )
    println("  Loaded $(length(test_minute_spots)) minute spot points")
    test_spot_history = build_spot_history_dict(
        test_timestamps,
        test_minute_spots;
        lookback_days=SPOT_HISTORY_LOOKBACK_DAYS
    )
    println("  Built spot history for $(length(test_spot_history)) timestamps")

    test_prev_surfaces = build_prev_surfaces_dict(test_surfaces; symbol=symbol)
    println("  Built prev-day surface map for $(length(test_prev_surfaces))/$(length(test_surfaces)) timestamps")
    println()

    # Generate test labels
    println("  Generating test labels...")
    test_data = if STRATEGY == "condor" && MODEL_MODE != :delta
        generate_condor_candidate_training_data(
            test_surfaces,
            test_settlement_spots,
            test_spot_history;
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            expiry_interval=EXPIRY_INTERVAL,
            utility_objective=SCORE_UTILITY_OBJECTIVE,
            candidate_delta_grid=SCORE_DELTA_GRID,
            max_candidates_per_day=SCORE_MAX_CANDIDATES_PER_DAY,
            wing_delta_abs=nothing,
            target_max_loss=(CONDOR_WING_OBJECTIVE == :target_max_loss ? TARGET_MAX_LOSS : nothing),
            wing_objective=CONDOR_WING_OBJECTIVE,
            max_loss_min=CONDOR_MAX_LOSS_MIN,
            max_loss_max=CONDOR_MAX_LOSS_MAX,
            min_credit=CONDOR_MIN_CREDIT,
            min_delta_gap=MIN_DELTA_GAP,
            prefer_symmetric=PREFER_SYMMETRIC_WINGS,
            use_logsig=USE_LOGSIG,
            prev_surfaces=test_prev_surfaces,
            verbose=true
        )
    elseif STRATEGY == "condor"
        generate_condor_4d_training_data(
            test_surfaces,
            test_settlement_spots,
            test_spot_history;
            min_delta_gap=MIN_DELTA_GAP,
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            expiry_interval=EXPIRY_INTERVAL,
            use_logsig=USE_LOGSIG,
            prev_surfaces=test_prev_surfaces,
            verbose=true
        )
    else
        generate_training_data(
            test_surfaces,
            test_settlement_spots,
            test_spot_history;
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            expiry_interval=EXPIRY_INTERVAL,
            use_logsig=USE_LOGSIG,
            prev_surfaces=test_prev_surfaces,
            verbose=true
        )
    end
    println()
    println("  Test samples: $(length(test_data.timestamps))")
    println()

    # -------------------------------------------------------------------------
    # Normalize Features
    # -------------------------------------------------------------------------
    println("PHASE 4: Normalizing Features")
    println("-" ^ 40)

    X_train_norm, feature_means, feature_stds = normalize_features(train_data.features)
    X_val_norm = apply_normalization(val_data.features, feature_means, feature_stds)
    X_test_norm = apply_normalization(test_data.features, feature_means, feature_stds)

    train_data_norm = if STRATEGY == "condor" && MODEL_MODE != :delta
        CondorScoringDataset(
            X_train_norm,
            train_data.utilities,
            train_data.pnls,
            train_data.max_losses,
            train_data.timestamps
        )
    else
        TrainingDataset(
            X_train_norm,
            train_data.labels,
            train_data.raw_deltas,
            train_data.pnls,
            train_data.size_labels,
            train_data.timestamps
        )
    end
    val_data_norm = if STRATEGY == "condor" && MODEL_MODE != :delta
        CondorScoringDataset(
            X_val_norm,
            val_data.utilities,
            val_data.pnls,
            val_data.max_losses,
            val_data.timestamps
        )
    else
        TrainingDataset(
            X_val_norm,
            val_data.labels,
            val_data.raw_deltas,
            val_data.pnls,
            val_data.size_labels,
            val_data.timestamps
        )
    end
    test_data_norm = if STRATEGY == "condor" && MODEL_MODE != :delta
        CondorScoringDataset(
            X_test_norm,
            test_data.utilities,
            test_data.pnls,
            test_data.max_losses,
            test_data.timestamps
        )
    else
        TrainingDataset(
            X_test_norm,
            test_data.labels,
            test_data.raw_deltas,
            test_data.pnls,
            test_data.size_labels,
            test_data.timestamps
        )
    end

    println("  Feature means: $(round.(feature_means, digits=4))")
    println("  Feature stds:  $(round.(feature_stds, digits=4))")
    println()

    # -------------------------------------------------------------------------
    # Train Model
    # -------------------------------------------------------------------------
    println("PHASE 5: Training Neural Network")
    println("-" ^ 40)
    input_dim = (STRATEGY == "condor" && MODEL_MODE != :delta) ?
        n_condor_scoring_features(; use_logsig=USE_LOGSIG) :
        n_features(; use_logsig=USE_LOGSIG)
    if size(train_data.features, 1) != input_dim
        error("Feature dimension mismatch: training matrix has $(size(train_data.features, 1)) rows, expected $input_dim")
    end

    if STRATEGY == "condor" && MODEL_MODE != :delta
        println("  Architecture: $(input_dim) -> $(SCORE_HIDDEN_DIMS) -> 1 (utility score)")
    else
        println("  Architecture: $(input_dim) -> $(HIDDEN_DIMS) -> 4 (4-delta condor)")
    end
    println("  Epochs: $EPOCHS, Batch size: $BATCH_SIZE, LR: $LEARNING_RATE")
    println()

    model, history = if STRATEGY == "condor" && MODEL_MODE != :delta
        scoring_model = create_scoring_model(
            input_dim=input_dim,
            hidden_dims=SCORE_HIDDEN_DIMS,
            dropout_rate=DROPOUT_RATE
        )
        train_scoring_model!(
            scoring_model,
            train_data_norm;
            val_data=val_data_norm,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            patience=PATIENCE,
            verbose=true
        )
    else
        strike_model = create_strike_model(
            input_dim=input_dim,
            hidden_dims=HIDDEN_DIMS,
            output_dim=4,  # short_put, short_call, long_put, long_call deltas
            dropout_rate=DROPOUT_RATE
        )
        train_model!(
            strike_model,
            train_data_norm;
            val_data=val_data_norm,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            patience=PATIENCE,
            verbose=true
        )
    end
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

    train_eval, val_eval, test_eval = if STRATEGY == "condor" && MODEL_MODE != :delta
        evaluate_scoring_model(model, train_data_norm), evaluate_scoring_model(model, val_data_norm), evaluate_scoring_model(model, test_data_norm)
    else
        evaluate_model(model, train_data_norm), evaluate_model(model, val_data_norm), evaluate_model(model, test_data_norm)
    end

    if STRATEGY == "condor" && MODEL_MODE != :delta
        println("  Training Set:")
        println("    Utility MSE: $(round(train_eval["utility_mse"], digits=6)), MAE: $(round(train_eval["utility_mae"], digits=4))")
        println("  Validation Set (early stopping):")
        println("    Utility MSE: $(round(val_eval["utility_mse"], digits=6)), MAE: $(round(val_eval["utility_mae"], digits=4))")
        println("  Test Set:")
        println("    Utility MSE: $(round(test_eval["utility_mse"], digits=6)), MAE: $(round(test_eval["utility_mae"], digits=4))")
    else
        println("  Training Set:")
        println("    Delta MSE: $(round(train_eval["delta_mse"], digits=6)), MAE: $(round(train_eval["delta_mae"], digits=4))")
        if train_eval["size_mse"] > 0
            println("    Size  MSE: $(round(train_eval["size_mse"], digits=6)), MAE: $(round(train_eval["size_mae"], digits=4))")
        end

        println("  Validation Set (early stopping):")
        println("    Delta MSE: $(round(val_eval["delta_mse"], digits=6)), MAE: $(round(val_eval["delta_mae"], digits=4))")
        if val_eval["size_mse"] > 0
            println("    Size  MSE: $(round(val_eval["size_mse"], digits=6)), MAE: $(round(val_eval["size_mae"], digits=4))")
        end

        println("  Test Set:")
        println("    Delta MSE: $(round(test_eval["delta_mse"], digits=6)), MAE: $(round(test_eval["delta_mae"], digits=4))")
        if test_eval["size_mse"] > 0
            println("    Size  MSE: $(round(test_eval["size_mse"], digits=6)), MAE: $(round(test_eval["size_mae"], digits=4))")
        end
    end
    println()

    # -------------------------------------------------------------------------
    # Backtest Comparison
    # -------------------------------------------------------------------------
    println("PHASE 7: Backtest Comparison (Test Period)")
    println("-" ^ 40)

    # Create ML selector
    ml_selector = if STRATEGY == "condor" && MODEL_MODE != :delta
        fallback_selector = if MODEL_MODE == :hybrid
            ctx -> begin
                shorts = VolSurfaceAnalysis._delta_strangle_strikes_asymmetric(
                    ctx,
                    SHORT_DELTA_ABS,
                    SHORT_DELTA_ABS;
                    rate=RISK_FREE_RATE,
                    div_yield=DIV_YIELD
                )
                shorts === nothing && return nothing
                short_put_K, short_call_K = shorts

                wings = VolSurfaceAnalysis._condor_wings_by_objective(
                    ctx,
                    short_put_K,
                    short_call_K;
                    objective=CONDOR_WING_OBJECTIVE,
                    target_max_loss=(CONDOR_WING_OBJECTIVE == :target_max_loss ? TARGET_MAX_LOSS : nothing),
                    max_loss_min=CONDOR_MAX_LOSS_MIN,
                    max_loss_max=CONDOR_MAX_LOSS_MAX,
                    min_credit=CONDOR_MIN_CREDIT,
                    rate=RISK_FREE_RATE,
                    div_yield=DIV_YIELD,
                    min_delta_gap=MIN_DELTA_GAP,
                    prefer_symmetric=PREFER_SYMMETRIC_WINGS,
                    debug=false
                )
                wings === nothing && return nothing
                long_put_K, long_call_K = wings
                return (short_put_K, short_call_K, long_put_K, long_call_K)
            end
        else
            nothing
        end

        MLCondorScoreSelector(
            model, feature_means, feature_stds;
            candidate_delta_grid=SCORE_DELTA_GRID,
            max_candidates=SCORE_MAX_CANDIDATES_PER_DAY,
            wing_delta_abs=nothing,
            target_max_loss=(CONDOR_WING_OBJECTIVE == :target_max_loss ? Float32(TARGET_MAX_LOSS) : nothing),
            wing_objective=CONDOR_WING_OBJECTIVE,
            max_loss_min=Float32(CONDOR_MAX_LOSS_MIN),
            max_loss_max=Float32(CONDOR_MAX_LOSS_MAX),
            min_credit=Float32(CONDOR_MIN_CREDIT),
            min_delta_gap=Float32(MIN_DELTA_GAP),
            prefer_symmetric=PREFER_SYMMETRIC_WINGS,
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            use_logsig=USE_LOGSIG,
            fallback_selector=fallback_selector
        )
    elseif STRATEGY == "condor"
        MLCondorStrikeSelector(
            model, feature_means, feature_stds;
            min_delta=0.05f0,
            max_delta=0.35f0,
            wing_delta_abs=nothing,
            target_max_loss=(CONDOR_WING_OBJECTIVE == :target_max_loss ? Float32(TARGET_MAX_LOSS) : nothing),
            wing_objective=CONDOR_WING_OBJECTIVE,
            max_loss_min=Float32(CONDOR_MAX_LOSS_MIN),
            max_loss_max=Float32(CONDOR_MAX_LOSS_MAX),
            min_credit=Float32(CONDOR_MIN_CREDIT),
            min_delta_gap=Float32(MIN_DELTA_GAP),
            prefer_symmetric=PREFER_SYMMETRIC_WINGS,
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            use_logsig=USE_LOGSIG
        )
    else
        MLStrikeSelector(
            model, feature_means, feature_stds;
            min_delta=0.05f0,
            max_delta=0.35f0,
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            use_logsig=USE_LOGSIG
        )
    end

    # Create strategies
    schedule = sort(collect(keys(test_surfaces)))

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
        baseline_selector = ctx -> begin
            shorts = VolSurfaceAnalysis._delta_strangle_strikes_asymmetric(
                ctx,
                SHORT_DELTA_ABS,
                SHORT_DELTA_ABS;
                rate=RISK_FREE_RATE,
                div_yield=DIV_YIELD
            )
            shorts === nothing && return nothing
            short_put_K, short_call_K = shorts

            wings = VolSurfaceAnalysis._condor_wings_by_objective(
                ctx,
                short_put_K,
                short_call_K;
                objective=CONDOR_WING_OBJECTIVE,
                target_max_loss=(CONDOR_WING_OBJECTIVE == :target_max_loss ? TARGET_MAX_LOSS : nothing),
                max_loss_min=CONDOR_MAX_LOSS_MIN,
                max_loss_max=CONDOR_MAX_LOSS_MAX,
                min_credit=CONDOR_MIN_CREDIT,
                rate=RISK_FREE_RATE,
                div_yield=DIV_YIELD,
                min_delta_gap=MIN_DELTA_GAP,
                prefer_symmetric=PREFER_SYMMETRIC_WINGS,
                debug=false
            )
            wings === nothing && return nothing
            long_put_K, long_call_K = wings
            return (short_put_K, short_call_K, long_put_K, long_call_K)
        end
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
    # Merge minute spots with settlement spots so DictDataSource serves both
    # historical queries (for ML features) and settlement lookups.
    all_spots = merge(test_minute_spots, test_settlement_spots)
    test_source = DictDataSource(test_surfaces, all_spots)

    println("  Running ML strategy backtest...")
    pos_ml, pnl_ml = backtest_strategy(strategy_ml, test_source)

    println("  Running baseline (fixed delta) backtest...")
    pos_baseline, pnl_baseline = backtest_strategy(strategy_baseline, test_source)

    println("  Running baseline (0.8 sigma) backtest...")
    pos_sigma, pnl_sigma = backtest_strategy(strategy_sigma, test_source)
    println()

    # Compute metrics
    if STRATEGY == "condor"
        metrics_ml = performance_metrics(pos_ml, pnl_ml; margin_by_key=condor_max_loss_by_key(pos_ml))
        metrics_baseline = performance_metrics(pos_baseline, pnl_baseline; margin_by_key=condor_max_loss_by_key(pos_baseline))
        metrics_sigma = performance_metrics(pos_sigma, pnl_sigma; margin_by_key=condor_max_loss_by_key(pos_sigma))
    else
        margin = 12000.0
        metrics_ml = performance_metrics(pos_ml, pnl_ml; margin_per_trade=margin)
        metrics_baseline = performance_metrics(pos_baseline, pnl_baseline; margin_per_trade=margin)
        metrics_sigma = performance_metrics(pos_sigma, pnl_sigma; margin_per_trade=margin)
    end

    # -------------------------------------------------------------------------
    # Results Summary
    # -------------------------------------------------------------------------
    println("=" ^ 80)
    println("RESULTS SUMMARY")
    println("=" ^ 80)
    println()
    println("Test Period: $TEST_START to $TEST_END (val for early stopping: $VAL_START to $VAL_END)")
    if STRATEGY == "condor"
        println("Return basis: per-trade max loss")
    else
        println("Margin per trade: \$$(Int(margin))")
    end
    println()

    println("-" ^ 90)
    println("Strategy             | Trades | Total P&L  | Avg ROI  |  Sharpe  | Win Rate |   Avg P&L")
    println("-" ^ 90)

    ml_label = if STRATEGY == "condor" && MODEL_MODE == :score
        "ML Condor Score"
    elseif STRATEGY == "condor" && MODEL_MODE == :hybrid
        "ML Condor Hybrid"
    elseif STRATEGY == "condor"
        "ML Condor"
    else
        "ML Selector"
    end
    base_label = STRATEGY == "condor" ? "Fixed Delta Condor" : "Fixed 15-Delta"
    sigma_label = STRATEGY == "condor" ? "Sigma Condor" : "0.8 Sigma"

    function _fmt_row(label, m)
        roi_str = ismissing(m.avg_return) ? "   N/A  " : @sprintf("%7.1f%%", m.avg_return * 100)
        "$(rpad(label, 20)) | $(lpad(string(m.count), 6)) | $(rpad(fmt_pnl(m.total_pnl), 10)) | $(roi_str) | $(rpad(fmt_ratio(m.sharpe), 8)) | $(rpad(fmt_pct(m.win_rate), 8)) | $(fmt_currency(m.avg_pnl))"
    end
    println(_fmt_row(ml_label, metrics_ml))
    println(_fmt_row(base_label, metrics_baseline))
    println(_fmt_row(sigma_label, metrics_sigma))
    println("-" ^ 90)
    println()

    # Save results
    results_df = DataFrame(
        Strategy = [ml_label, base_label, sigma_label],
        Count = [metrics_ml.count, metrics_baseline.count, metrics_sigma.count],
        TotalPnL = [metrics_ml.total_pnl, metrics_baseline.total_pnl, metrics_sigma.total_pnl],
        AvgPnL = [metrics_ml.avg_pnl, metrics_baseline.avg_pnl, metrics_sigma.avg_pnl],
        AvgROI = [metrics_ml.avg_return, metrics_baseline.avg_return, metrics_sigma.avg_return],
        WinRate = [metrics_ml.win_rate, metrics_baseline.win_rate, metrics_sigma.win_rate],
        Sharpe = [metrics_ml.sharpe, metrics_baseline.sharpe, metrics_sigma.sharpe],
        Sortino = [metrics_ml.sortino, metrics_baseline.sortino, metrics_sigma.sortino]
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
    history_df = if haskey(history, "train_size_loss")
        DataFrame(
            Epoch = 1:length(history["train_loss"]),
            TrainLoss = history["train_loss"],
            ValLoss = history["val_loss"],
            TrainDeltaLoss = history["train_delta_loss"],
            TrainSizeLoss = history["train_size_loss"]
        )
    elseif haskey(history, "train_delta_loss")
        DataFrame(
            Epoch = 1:length(history["train_loss"]),
            TrainLoss = history["train_loss"],
            ValLoss = history["val_loss"],
            TrainDeltaLoss = history["train_delta_loss"]
        )
    else
        DataFrame(
            Epoch = 1:length(history["train_loss"]),
            TrainLoss = history["train_loss"],
            ValLoss = history["val_loss"]
        )
    end
    history_path = joinpath(RUN_DIR, "training_history.csv")
    CSV.write(history_path, history_df)
    println("Training history saved to: $history_path")

    # Save predicted vs actual deltas and sizes for analysis (test set)
    predictions_df = if STRATEGY == "condor" && MODEL_MODE != :delta
        DataFrame(
            Timestamp = test_data.timestamps,
            TrueUtility = test_data.utilities,
            PredUtility = test_eval["pred_utilities"],
            CandidatePnL = test_data.pnls,
            CandidateMaxLoss = test_data.max_losses
        )
    elseif size(test_data.raw_deltas, 1) == 4
        DataFrame(
            Timestamp = test_data.timestamps,
            TrueShortPutDelta = test_data.raw_deltas[1, :],
            TrueShortCallDelta = test_data.raw_deltas[2, :],
            TrueLongPutDelta = test_data.raw_deltas[3, :],
            TrueLongCallDelta = test_data.raw_deltas[4, :],
            PredShortPutDelta = test_eval["pred_deltas"][1, :],
            PredShortCallDelta = test_eval["pred_deltas"][2, :],
            PredLongPutDelta = test_eval["pred_deltas"][3, :],
            PredLongCallDelta = test_eval["pred_deltas"][4, :],
            BestPnL = test_data.pnls
        )
    else
        DataFrame(
            Timestamp = test_data.timestamps,
            TruePutDelta = test_data.raw_deltas[1, :],
            TrueCallDelta = test_data.raw_deltas[2, :],
            PredPutDelta = test_eval["pred_deltas"][1, :],
            PredCallDelta = test_eval["pred_deltas"][2, :],
            TrueSize = test_data.size_labels,
            PredSize = test_eval["pred_sizes"],
            BestPnL = test_data.pnls
        )
    end
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
                title="Spot Curve $(symbol)"
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
