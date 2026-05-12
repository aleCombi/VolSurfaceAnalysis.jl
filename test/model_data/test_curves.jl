@testset "FlatCurve" begin
    c = FlatCurve(0.04)
    @test c(DateTime(2024, 1, 1)) == 0.04
    @test c(DateTime(2030, 6, 15, 9, 30)) == 0.04
    @test c isa Curve
end

@testset "PCCurve: basic lookup" begin
    knots = [DateTime(2024, 1, 1), DateTime(2024, 4, 1), DateTime(2024, 7, 1)]
    values = [0.04, 0.045, 0.05]
    c = PCCurve(knots, values)

    # At knots
    @test c(knots[1]) == 0.04
    @test c(knots[2]) == 0.045
    @test c(knots[3]) == 0.05

    # Between knots: takes the prior knot's value
    @test c(DateTime(2024, 2, 15)) == 0.04
    @test c(DateTime(2024, 5, 15)) == 0.045
end

@testset "PCCurve: out-of-range flat-extrapolates" begin
    knots = [DateTime(2024, 1, 1), DateTime(2024, 4, 1)]
    values = [0.04, 0.045]
    c = PCCurve(knots, values)

    @test c(DateTime(2023, 1, 1)) == 0.04   # before first
    @test c(DateTime(2025, 1, 1)) == 0.045  # after last
end

@testset "PCCurve: single knot" begin
    c = PCCurve([DateTime(2024, 1, 1)], [0.03])
    @test c(DateTime(2020, 1, 1)) == 0.03
    @test c(DateTime(2024, 1, 1)) == 0.03
    @test c(DateTime(2030, 1, 1)) == 0.03
end

@testset "PCCurve: constructor validation" begin
    @test_throws ArgumentError PCCurve(DateTime[], Float64[])

    @test_throws ArgumentError PCCurve(
        [DateTime(2024, 1, 1), DateTime(2024, 2, 1)],
        [0.04],
    )

    @test_throws ArgumentError PCCurve(
        [DateTime(2024, 2, 1), DateTime(2024, 1, 1)],
        [0.04, 0.045],
    )

    @test_throws ArgumentError PCCurve(
        [DateTime(2024, 1, 1), DateTime(2024, 1, 1)],
        [0.04, 0.045],
    )
end

@testset "Curve: unimplemented subtype errors" begin
    struct _DummyCurve <: Curve end
    @test_throws ErrorException _DummyCurve()(DateTime(2024, 1, 1))
end
