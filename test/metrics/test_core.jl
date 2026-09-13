# Tests for the always-on core metrics: total_pnl, n_round_trips, hit_rate.
# The series is built directly: the metrics are pure functions over it.

_cr_series(ts::Vector{DateTime}, pnl::Vector{Float64}; n_opens=length(pnl), n_closes=length(pnl)) =
    PnLSeries(ts, pnl, NaN, n_opens, n_closes, 0)

@testset "core metrics: empty series" begin
    s = _cr_series(DateTime[], Float64[])
    @test total_pnl(s) == 0.0
    @test n_round_trips(s) == 0
    @test isnan(hit_rate(s))
end

@testset "core metrics: all winners" begin
    ts2 = DateTime(2024, 1, 15, 16, 30)
    s = _cr_series([ts2], [1.0])                 # bought at 5.0, sold at 6.0
    @test total_pnl(s) ≈ 1.0
    @test n_round_trips(s) == 1
    @test hit_rate(s) == 1.0
end

@testset "core metrics: all losers" begin
    ts2 = DateTime(2024, 1, 15, 16, 30)
    s = _cr_series([ts2], [-1.0])                # bought at 6.0, sold at 5.0
    @test total_pnl(s) ≈ -1.0
    @test n_round_trips(s) == 1
    @test hit_rate(s) == 0.0
end

@testset "core metrics: mixed -- hit rate counts strictly positive" begin
    ts3 = DateTime(2024, 1, 15, 16, 30)
    ts4 = DateTime(2024, 1, 15, 16, 35)
    # Round trip 1: +1.0 (winner). Round trip 2: -0.5 (loser).
    s = _cr_series([ts3, ts4], [1.0, -0.5])
    @test total_pnl(s) ≈ 0.5
    @test n_round_trips(s) == 2
    @test hit_rate(s) == 0.5
end

@testset "core metrics: zero-PnL trade does not count as a win" begin
    ts2 = DateTime(2024, 1, 15, 16, 30)
    s = _cr_series([ts2], [0.0])                 # exact breakeven
    @test total_pnl(s) == 0.0
    @test n_round_trips(s) == 1
    @test hit_rate(s) == 0.0
end
