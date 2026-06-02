using VolSurfaceAnalysis
using Test
using Dates

@testset "VolSurfaceAnalysis" begin
    include("data/test_quotes.jl")
    include("data/test_synth.jl")
    include("data/test_source.jl")
    include("data/test_polygon.jl")
    include("data/test_parquet_source.jl")
    include("model_data/test_curves.jl")
    include("surfaces/test_bs.jl")
    include("surfaces/test_surface.jl")
    include("model_data/test_source.jl")
    include("positions/test_trade.jl")
    include("positions/test_position.jl")
    include("backtest/test_time_cut.jl")
    include("policies/test_policy.jl")
    include("agents/test_agent.jl")
    include("backtest/test_engine.jl")
    include("metrics/test_pnl_series.jl")
    include("metrics/test_core.jl")
    include("metrics/test_optional.jl")
    include("metrics/test_dispatch.jl")
    include("experiment/test_experiment.jl")
    include("experiment/test_config.jl")
    include("persistence/test_store.jl")
    include("viz/test_spot.jl")
    include("viz/test_pnl.jl")
end
