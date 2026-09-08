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

# ---------- curve kinds ----------

@testset "RateCurve / DivCurve: kinds with a visibility time and a selector" begin
    rc = RateCurve(_MD_USD, FlatCurve(0.04))
    dc = DivCurve(_MD_SPY, FlatCurve(0.015))
    @test rc.timestamp == typemin(DateTime) && dc.timestamp == typemin(DateTime)
    @test rc.curve(_MD_T1) == 0.04 && dc.curve(_MD_T1) == 0.015
    @test selector(rc) === _MD_USD && selector(dc) === _MD_SPY
    @test selector_type(RateCurve) === Currency && selector_type(DivCurve) === Underlying
    stamped = RateCurve(_MD_USD, PCCurve([_MD_T1], [0.05]), _MD_T2)
    @test stamped.timestamp == _MD_T2
    @test_throws MethodError RateCurve(_MD_SPY, FlatCurve(0.04))       # selector type enforced by the field
    @test Clock{RateCurve}(_MD_USD) isa Clock{RateCurve,Currency}
    @test_throws ArgumentError Clock{RateCurve}(_MD_SPY)
end

@testset "Constant{RateCurve}: the flat-curve case on the protocol" begin
    m = MarketData(Constant(RateCurve(_MD_USD, FlatCurve(0.04))), Constant(DivCurve(_MD_SPY, FlatCurve(0.015))))
    @test only_or_missing(asof(m, RateCurve, _MD_USD, _MD_T1)).curve(_MD_T1) == 0.04
    @test_throws UnservedSelector asof(m, RateCurve, Currency("EUR"), _MD_T1)
    @test only_or_missing(asof(m, DivCurve, _MD_SPY, _MD_T1)).curve(_MD_T3) == 0.015
    @test_throws UnservedSelector asof(m, DivCurve, _MD_SPX, _MD_T1)
    @test timestamps(m, RateCurve, _MD_USD, _MD_T1, _MD_T3) == DateTime[]
    @test collect(between(m, DivCurve, _MD_SPY, _MD_T1, _MD_T3)) == DivCurve[]
    @test asof(TimeCut(m, _MD_T1), RateCurve, _MD_USD, _MD_T3) == asof(m, RateCurve, _MD_USD, _MD_T1)
end
