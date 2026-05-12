using VolSurfaceAnalysis
using Test
using Dates

@testset "VolSurfaceAnalysis" begin
    include("data/test_quotes.jl")
    include("data/test_source.jl")
    include("data/test_polygon.jl")
    include("data/test_parquet_source.jl")
    include("model_data/test_curves.jl")
    include("surfaces/test_bs.jl")
    include("surfaces/test_surface.jl")
    include("model_data/test_source.jl")
    include("viz/test_spot.jl")
end
