module VolSurfaceAnalysis

using Dates
using Distributions: Normal, cdf, pdf
using Roots: Brent, find_zero
using Statistics: mean, std, median
using Random
using Flux
using BSON
using ChenSignatures: sig, logsig, prepare, sig_leadlag, logsig_leadlag

# ============================================================================
# Core Types & Models
# ============================================================================
include("data/option_records.jl")  # Types & common utilities
include("models.jl")               # Black-76 pricing functions

# ============================================================================
# Data Layer (source conversions)
# ============================================================================
include("data/duckdb.jl")          # DuckDB parquet helpers
include("data/deribit.jl")         # Deribit source logic
include("data/polygon.jl")         # Polygon source logic
include("data/local_store.jl")     # Local data-store path derivation

# ============================================================================
# Core Library
# ============================================================================
include("surface.jl")              # Volatility surface representation
include("trades.jl")               # Trade representation and pricing

# ============================================================================
# Backtesting
# ============================================================================
include("backtest/portfolio.jl")   # Position management (pure)
include("backtest/engine.jl")      # Minimal backtest engine
include("backtest/metrics.jl")     # Backtest metrics
include("backtest/plots.jl")       # Backtest plots
include("strategies.jl")           # Strategy implementations

# ============================================================================
# Machine Learning Module
# ============================================================================
include("ml/features.jl")          # Feature extraction from vol surfaces
include("ml/model.jl")             # Flux.jl neural network definition
include("ml/training.jl")          # Training loop and data generation
include("ml/strike_selector.jl")   # ML-based strike selector

# ============================================================================
# Exports: Data Types
# ============================================================================
# Core types
export OptionType, Call, Put
export Underlying, ticker

# Source types (match parquet schemas)
export DeribitQuote, PolygonBar, DeribitDelivery
export SpotPrice

# Internal type (unified)
export OptionRecord


# ============================================================================
# Exports: Data Utilities
# ============================================================================
export parse_polygon_ticker
export to_option_record
export read_deribit_parquet, read_deribit_option_records
export read_deribit_spot_parquet, read_deribit_spot_prices
export read_polygon_parquet, read_polygon_option_records
export read_polygon_spot_parquet, read_polygon_spot_prices
export read_polygon_spot_prices_for_timestamps
export read_polygon_spot_prices_dir
export et_to_utc
export spot_dict

# Local data store
export LocalDataStore, DEFAULT_STORE
export polygon_options_root, polygon_spot_root
export polygon_options_path, polygon_spot_path
export deribit_history_path, deribit_delivery_path
export available_polygon_dates, available_deribit_dates

# ============================================================================
# Exports: Pricing Models
# ============================================================================
export black76_price, vol_to_price, black76_vega, price_to_iv
export black76_delta, black76_gamma, black76_theta
export time_to_expiry

# ============================================================================
# Exports: Volatility Surface
# ============================================================================
export VolPoint, VolatilitySurface, build_surface
export build_surfaces_for_timestamps
export find_record
export TermStructure, atm_term_structure
export bid_iv, ask_iv

# ============================================================================
# Exports: Trades
# ============================================================================
export Trade, price, payoff, find_vol

# ============================================================================
# Exports: Backtesting
# ============================================================================
# Position management
export Position, open_position, entry_cost, settle

# Engine
export Strategy, ScheduledStrategy
export next_portfolio, entry_schedule, entry_positions
export BacktestResult, backtest_strategy
export BacktestMetrics, PerformanceMetrics
export aggregate_pnl, backtest_metrics, performance_metrics, profit_curve, average_entry_spread
export condor_group_stats, condor_trade_table, condor_max_loss_by_key
export settlement_zone_analysis, settlement_zone_summary
export save_pnl_distribution, save_equity_curve, save_pnl_and_equity_curve, save_profit_curve, save_spot_curve

# Strategies
export IronCondorStrategy
export ShortStrangleStrategy

# ============================================================================
# Exports: Machine Learning
# ============================================================================
# Feature extraction
export SurfaceFeatures, SpotHistory, N_FEATURES, SIGNATURE_LEVEL, SIGNATURE_DIM, LOGSIGNATURE_DIM
export path_feature_dim, n_features, N_CONDOR_CANDIDATE_FEATURES, n_condor_scoring_features
export extract_features, features_to_vector
export compute_path_signature, compute_logsig_features
export normalize_features, apply_normalization

# Model
export create_strike_model, scale_deltas, unscale_deltas
export create_scoring_model
export delta_loss, predict_deltas
export predict_with_sizing, combined_loss

# Training
export TrainingDataset, DELTA_GRID
export CondorScoringDataset
export simulate_strangle_pnl, find_optimal_deltas
export generate_training_data, train_model!, evaluate_model
export compute_size_labels
export simulate_condor_pnl, find_optimal_condor_deltas, generate_condor_training_data
export build_condor_ctx, condor_entry_metrics_from_strikes, condor_metrics_from_strikes
export enumerate_condor_candidates, condor_scoring_feature_vector
export condor_realized_utility, generate_condor_candidate_training_data
export train_scoring_model!, evaluate_scoring_model

# Strike selector
export MLStrikeSelector
export MLCondorStrikeSelector
export MLCondorScoreSelector
export save_ml_selector, load_ml_selector

# Asymmetric delta helper (also used internally)
export _delta_strangle_strikes_asymmetric

end # module
