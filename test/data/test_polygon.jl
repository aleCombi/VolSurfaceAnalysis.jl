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
