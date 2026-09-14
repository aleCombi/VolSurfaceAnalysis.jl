using Plots

_vz_curve(profit) = MarkedCurve(
    [DateTime(2024, 1, 15, 21, 0) + Day(i - 1) for i in 1:length(profit)],
    profit, DateTime[], Symbol[])

@testset "MarkedCurve recipe: the marked profit path" begin
    p = plot(_vz_curve([1.5, 1.0, 3.0, 2.0, 2.5]))
    @test p isa Plots.Plot
end

@testset "MarkedCurve recipe: no marked session errors" begin
    @test_throws Exception plot(_vz_curve(Float64[]))
end

@testset "MarkedCurve recipe: the path breaks at an unmarked session" begin
    # The chart must not draw a straight line through a valuation failure:
    # that is exactly the stretch the curve refuses to claim anything about.
    c = MarkedCurve([DateTime(2024, 1, 16, 21), DateTime(2024, 1, 18, 21)],
                    [1.0, 3.0],
                    [DateTime(2024, 1, 17, 21)], [:no_mark])
    p = plot(c)
    ys = p.series_list[1][:y]
    @test length(ys) == 3
    @test isnan(ys[2])                     # the gap sits between the two marks
    @test !any(isnan, (ys[1], ys[3]))
end

@testset "MarkedCurve recipe: kwargs override" begin
    p = plot(_vz_curve([1.0, 2.0, 3.0]); title="custom", ylabel="USD")
    @test p isa Plots.Plot
end
