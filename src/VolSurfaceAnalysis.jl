module VolSurfaceAnalysis

using Dates
using DataFrames
using Parquet2
using Distributions: Normal, cdf, pdf
using HTTP
using JSON3

# Include submodules in dependency order
include("data.jl")      # Types: OptionType, Underlying, VolRecord
include("models.jl")    # Black-76 pricing functions
include("surface.jl")   # Volatility surface representation
include("trades.jl")    # Trade representation and pricing
include("api.jl")       # Deribit API functions

# Data types
export VolRecord, OptionType, Underlying
export Call, Put, BTC, ETH

# Data import functions
export read_vol_records, split_by_timestamp

# Model functions
export black76_price, vol_to_price, black76_vega, price_to_iv
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

end # module
