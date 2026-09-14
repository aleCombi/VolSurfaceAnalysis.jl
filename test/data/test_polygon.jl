using VolSurfaceAnalysis: parse_polygon_ticker, et_to_utc, Call, Put

@testset "parse_polygon_ticker: SPY call standard" begin
    u, expiry, otype, strike = parse_polygon_ticker("O:SPY240129C00406000")
    @test u == "SPY"
    @test otype == Call
    @test strike == 406.0
    @test expiry == DateTime(2024, 1, 29, 21, 0)
end

@testset "parse_polygon_ticker: DST boundary" begin
    _, expiry_winter, _, _ = parse_polygon_ticker("O:SPY240301C00400000")
    @test expiry_winter == DateTime(2024, 3, 1, 21, 0)

    _, expiry_summer, _, _ = parse_polygon_ticker("O:SPY240715C00400000")
    @test expiry_summer == DateTime(2024, 7, 15, 20, 0)
end

@testset "parse_polygon_ticker: put + malformed" begin
    _, _, otype, strike = parse_polygon_ticker("O:QQQ240115P00350500")
    @test otype == Put
    @test strike == 350.5

    @test_throws ArgumentError parse_polygon_ticker("O:SPY24")
    @test_throws ArgumentError parse_polygon_ticker("not_a_ticker")
end

@testset "et_to_utc: EST vs EDT" begin
    @test et_to_utc(Date(2024, 12, 15), Time(10, 0)) == DateTime(2024, 12, 15, 15, 0)
    @test et_to_utc(Date(2024, 7, 15), Time(10, 0)) == DateTime(2024, 7, 15, 14, 0)
end

@testset "bar stamps: a minute bar is visible at its end, not its open" begin
    # The convention, spelled out here rather than derived from the code:
    # a row stamped at 19:29 is the 19:29-19:30 minute, and its close, high
    # and low are knowable at 19:30. A decision at 19:30 therefore reads it,
    # and a decision at 19:29 does not.
    @test VolSurfaceAnalysis.BAR_INTERVAL == Minute(1)
    @test VolSurfaceAnalysis.bar_visible_at(DateTime(2024, 1, 15, 19, 29)) ==
          DateTime(2024, 1, 15, 19, 30)
    @test VolSurfaceAnalysis.bar_row_time(DateTime(2024, 1, 15, 19, 30)) ==
          DateTime(2024, 1, 15, 19, 29)
    # Inverses, and a whole-minute shift carries sub-second bounds through
    # untouched -- which is what lets a range predicate be translated into
    # the stored clock without widening or narrowing it.
    for t in (DateTime(2024, 1, 15, 23, 59), DateTime(2024, 1, 15, 19, 29, 30),
              DateTime(2024, 1, 15, 19, 29) + Millisecond(1))
        @test VolSurfaceAnalysis.bar_row_time(VolSurfaceAnalysis.bar_visible_at(t)) == t
        @test VolSurfaceAnalysis.bar_visible_at(VolSurfaceAnalysis.bar_row_time(t)) == t
    end
    # A 23:59 row crosses midnight into the next calendar date.
    @test VolSurfaceAnalysis.bar_visible_at(DateTime(2024, 1, 15, 23, 59)) ==
          DateTime(2024, 1, 16, 0, 0)
end
