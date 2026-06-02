using Plots

@testset "PnLSeries recipe: basic equity curve" begin
    ts = [DateTime(2024, 1, 15, 16, 0) + Day(i) for i in 0:4]
    pnl = [1.5, -0.5, 2.0, -1.0, 0.5]
    series = PnLSeries(ts, pnl, 480.0, 5, 5, 0)
    p = plot(series)
    @test p isa Plots.Plot
end

@testset "PnLSeries recipe: empty errors" begin
    series = PnLSeries(DateTime[], Float64[], 480.0, 0, 0, 0)
    @test_throws Exception plot(series)
end

@testset "PnLSeries recipe: kwargs override" begin
    ts = [DateTime(2024, 1, 15, 16, 0) + Day(i) for i in 0:2]
    series = PnLSeries(ts, [1.0, 2.0, 3.0], 480.0, 3, 3, 0)
    p = plot(series; title="custom", ylabel="USD")
    @test p isa Plots.Plot
end
