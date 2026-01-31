module VolSurfaceAnalysis

using Dates
using Distributions: Normal, cdf, pdf
using Roots: Brent, find_zero
using Statistics: mean, std, median

# ============================================================================
# Data Layer (types and conversions)
# ============================================================================
include("data/option_records.jl")  # Types & common utilities
include("data/deribit.jl")         # Deribit source logic
include("data/polygon.jl")         # Polygon source logic

# ============================================================================
# Core Library
# ============================================================================
include("models.jl")               # Black-76 pricing functions
include("surface.jl")              # Volatility surface representation
include("trades.jl")               # Trade representation and pricing

# ============================================================================
# Backtesting
# ============================================================================
include("backtest/portfolio.jl")   # Position management (pure)
include("backtest/engine.jl")      # Minimal backtest engine
include("strategies.jl")           # Strategy implementations

# ============================================================================
# Exports: Data Types
# ============================================================================
# Core types
export OptionType, Call, Put
export Underlying, ticker

# Source types (match parquet schemas)
export DeribitQuote, PolygonBar, DeribitDelivery

# Internal type (unified)
export OptionRecord


# ============================================================================
# Exports: Data Utilities
# ============================================================================
export parse_polygon_ticker
export to_option_record

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

# Strategies
export IronCondorStrategy

end # module
