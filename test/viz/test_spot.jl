using Plots

@testset "Vector{SpotPrice} recipe: basic" begin
    ts = [DateTime(2024, 1, 15, 9, 30) + Minute(i) for i in 0:9]
    spots = [SpotPrice(Underlying("SPY"), 480.0 + i, ts[i + 1]) for i in 0:9]
    p = plot(spots)
    @test p isa Plots.Plot
end

@testset "Vector{SpotPrice} recipe: empty errors" begin
    @test_throws Exception plot(SpotPrice[])
end

@testset "Vector{SpotPrice} recipe: multiple underlyings errors" begin
    spots = [
        SpotPrice(Underlying("SPY"), 480.0, DateTime(2024, 1, 15, 9, 30)),
        SpotPrice(Underlying("QQQ"), 400.0, DateTime(2024, 1, 15, 9, 30)),
    ]
    @test_throws Exception plot(spots)
end

@testset "Vector{SpotPrice} recipe: kwargs override" begin
    ts = [DateTime(2024, 1, 15, 9, 30) + Minute(i) for i in 0:4]
    spots = [SpotPrice(Underlying("SPY"), 480.0 + i, ts[i + 1]) for i in 0:4]
    p = plot(spots; title="custom")
    @test p isa Plots.Plot
end
