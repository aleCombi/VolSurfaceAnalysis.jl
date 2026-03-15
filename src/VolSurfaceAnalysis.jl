module VolSurfaceAnalysis

using Dates
using Distributions: Normal, cdf, pdf
using Flux
using Random: randperm
using Roots: Brent, find_zero
using Statistics: mean, std

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
include("backtest/positions.jl")   # Position management (pure)
include("backtest/data_source.jl") # BacktestDataSource protocol
include("backtest/engine.jl")      # Minimal backtest engine
include("backtest/metrics.jl")     # Backtest metrics
include("backtest/plots.jl")       # Backtest plots
include("strategies.jl")           # Strategy implementations

# ============================================================================
# ML Module
# ============================================================================
include("ml/features.jl")         # Feature types & implementations
include("ml/model.jl")            # Flux MLP + scoring
include("ml/training.jl")         # Data gen, training, MLCondorSelector

# ============================================================================
# Data helpers (for scripts)
# ============================================================================
include("data/helpers.jl")         # Shared data-loading helpers for scripts

# ============================================================================
# Exports: Data Types
# ============================================================================
# Core types
export OptionType, Call, Put
export Underlying, ticker

# Source types (match parquet schemas)
export DeribitQuote, PolygonBar
export SpotPrice

# Internal type (unified)
export OptionRecord

# ============================================================================
# Exports: Data Utilities
# ============================================================================
export parse_polygon_ticker
export to_option_record
export read_deribit_parquet, read_deribit_option_records
export read_deribit_spot_prices
export read_polygon_parquet, read_polygon_option_records
export read_polygon_spot_parquet, read_polygon_spot_prices
export read_polygon_spot_prices_for_timestamps
export et_to_utc
export spot_dict

# Local data store
export LocalDataStore, DEFAULT_STORE
export polygon_options_root, polygon_spot_root
export polygon_options_path, polygon_spot_path
export deribit_history_path, deribit_delivery_path
export available_polygon_dates, available_deribit_dates
export build_entry_timestamps, load_minute_spots, load_surfaces_and_spots

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
export ScheduledStrategy, BacktestResult
export entry_schedule, entry_positions
export backtest_strategy, each_entry

# Data sources
export BacktestDataSource, DictDataSource, ParquetDataSource, HistoricalView
export available_timestamps, get_surface, get_settlement_spot, get_spot, get_spots
export BacktestMetrics, PerformanceMetrics
export aggregate_pnl, backtest_metrics, performance_metrics, profit_curve, average_entry_spread
export condor_group_stats, condor_trade_table, condor_max_loss_by_key
export settlement_zone_analysis, settlement_zone_summary
export fmt_pnl, fmt_ratio, fmt_pct, fmt_currency, fmt_metric
export metrics_to_dataframe, pnl_results_dataframe
export format_backtest_report
export save_pnl_distribution, save_equity_curve, save_pnl_and_equity_curve, save_profit_curve, save_spot_curve

# Strategies
export IronCondorStrategy, StrikeSelectionContext
export sigma_selector, delta_selector, delta_condor_selector, constrained_delta_selector

# Strike selection helpers (used by scripts)
export _delta_strangle_strikes_asymmetric

# ML features
export Feature, CandidateFeature
export ATMImpliedVol, DeltaSkew, RiskReversal, Butterfly, TermSlope
export ATMSpread, DeltaSpread, TotalVolume, PutCallVolumeRatio, HourOfDay, DayOfWeek
export ShortPutDelta, ShortCallDelta, EntryCredit, MaxLoss, CreditToMaxLoss
export DEFAULT_SURFACE_FEATURES, DEFAULT_CANDIDATE_FEATURES
export default_surface_features, default_candidate_features

# ML model & training
export roi_utility, pnl_utility
export create_scoring_model, score_candidates
export generate_training_data, train_scoring_model!, TrainingExample
export MLCondorSelector

end # module
