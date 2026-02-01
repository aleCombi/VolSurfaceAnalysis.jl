using VolSurfaceAnalysis
using Test
using Dates

@testset "VolSurfaceAnalysis.jl" begin
    include("test_black76.jl")
    include("test_data_ingestion.jl")
    include("test_portfolio.jl")
    include("test_engine.jl")
end

