using VolSurfaceAnalysis
using Test
using Dates

@testset "VolSurfaceAnalysis.jl" begin
    include("test_black76.jl")
    include("test_data_ingestion.jl")
    include("test_portfolio.jl")
    include("test_engine.jl")
    include("test_metrics.jl")
    include("test_local_data.jl")
    include("test_ml.jl")
    include("test_viz.jl")
    include("test_short_strangle.jl")
end

