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
include("backtest/portfolio.jl")   # Position management (pure)
include("backtest/engine.jl")      # Minimal backtest engine
include("strategies.jl")           # Strategy implementations

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

# Position management (pure functions)
export Position, open_position, entry_cost, settle

# Backtest engine (minimal)
export Strategy, ScheduledStrategy
export next_portfolio, entry_schedule, entry_positions
export BacktestResult, backtest_strategy

# Strategies
export IronCondorStrategy

end # module
