# Per-underlying configuration for ML training and evaluation scripts.
# Configs are stored as TOML files in scripts/configs/<SYMBOL>.toml.
#
# Usage:
#   include("configurations.jl")
#   cfg = load_symbol_config("SPY")  # loads configs/SPY.toml

using Configurations

@option struct SymbolConfig
    # Spot proxy
    spot_symbol::String
    spot_multiplier::Float64 = 1.0

    # Pricing parameters
    div_yield::Float64 = 0.013
    risk_free_rate::Float64 = 0.045

    # Condor thresholds (scaled to underlying price level)
    condor_max_loss_min::Float64 = 5.0
    condor_max_loss_max::Float64 = 30.0
    condor_min_credit::Float64 = 0.10
    constrained_super_max_loss_tol::Float64 = 1.0
end

const CONFIGS_DIR = joinpath(@__DIR__, "configs")

"""
    load_symbol_config(symbol::String) -> SymbolConfig

Load per-underlying config from `configs/<SYMBOL>.toml`.
"""
function load_symbol_config(symbol::String)
    path = joinpath(CONFIGS_DIR, "$(uppercase(symbol)).toml")
    isfile(path) || error("No config file for '$symbol' at $path. Available: $(list_available_configs())")
    return from_toml(SymbolConfig, path)
end

"""
    list_available_configs() -> Vector{String}

Return sorted list of symbols with config files.
"""
function list_available_configs()
    isdir(CONFIGS_DIR) || return String[]
    return sort([splitext(f)[1] for f in readdir(CONFIGS_DIR) if endswith(f, ".toml")])
end
