module VolSurfaceAnalysis

using Dates
using DataFrames
using Parquet2
using Distributions: Normal, cdf, pdf
using Statistics: mean, std
using HTTP
using JSON3

# Include submodules in dependency order
include("data.jl")      # Types: OptionType, Underlying, VolRecord
include("models.jl")    # Black-76 pricing functions
include("surface.jl")   # Volatility surface representation
include("trades.jl")    # Trade representation and pricing
include("api.jl")       # Deribit API functions
include("local_data.jl")           # Local data store
include("backtest/iterator.jl")    # Surface iterator for backtesting
include("backtest/portfolio.jl")   # Portfolio management
include("backtest/engine.jl")      # Backtest engine

# Data types
export VolRecord, OptionType, Underlying
export Call, Put, BTC, ETH

# Data import functions
export read_vol_records, split_by_timestamp

# Model functions
export black76_price, vol_to_price, black76_vega, price_to_iv
export black76_delta, black76_gamma, black76_theta
export time_to_expiry

# Surface types and functions
export VolPoint, VolatilitySurface, build_surface
export TermStructure, atm_term_structure
export bid_iv, ask_iv

# Trade types and functions
export Trade, price, payoff, find_vol

# API functions
export DeliveryPrice, fetch_delivery_prices, fetch_delivery_prices_df
export get_delivery_price, save_delivery_prices

# Local data and backtesting
export LocalDataStore, list_parquet_files, available_dates
export load_file, load_all, load_date, load_range, get_timestamps
export SurfaceIterator, surface_at, first_timestamp, last_timestamp
export date_range, timestamps, filter_timestamps

# Portfolio management
export Position, PortfolioSnapshot, TradeRecord, Portfolio
export add_position!, close_position!, close_all!
export position_value, position_pnl, mark_to_market, record_snapshot!
export position_delta, position_vega
export num_positions, total_value, get_positions, positions_expiring

# Backtest engine
export Strategy, Order, on_snapshot, on_expiry
export PerformanceMetrics, BacktestResult
export run_backtest, execute_order!
export pnl_series, equity_curve, trades_summary
export compute_metrics

end # module
