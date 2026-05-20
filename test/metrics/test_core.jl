# Tests for the always-on core metrics: total_pnl, n_round_trips, hit_rate.

const _CR_UND = Underlying("SPY")
const _CR_EXPIRY = DateTime(2024, 2, 16, 21, 0)

function _cr_pos(strike, otype, direction, qty, entry_price, ts)
    trd = Trade(_CR_UND, strike, _CR_EXPIRY, otype; direction=direction, quantity=qty)
    Position(trd, Float64(entry_price), 480.0, missing, missing, ts)
end

@testset "core metrics: empty series" begin
    s = pnl_series(Position[], 500.0)
    @test total_pnl(s) == 0.0
    @test n_round_trips(s) == 0
    @test isnan(hit_rate(s))
end

@testset "core metrics: all winners" begin
    ts1 = DateTime(2024, 1, 15, 15, 30)
    ts2 = DateTime(2024, 1, 15, 16, 30)
    open  = _cr_pos(480.0, Call, +1, 1.0, 5.0, ts1)
    close = _cr_pos(480.0, Call, -1, 1.0, 6.0, ts2)
    s = pnl_series([open, close], 500.0)
    @test total_pnl(s) ≈ 1.0
    @test n_round_trips(s) == 1
    @test hit_rate(s) == 1.0
end

@testset "core metrics: all losers" begin
    ts1 = DateTime(2024, 1, 15, 15, 30)
    ts2 = DateTime(2024, 1, 15, 16, 30)
    open  = _cr_pos(480.0, Call, +1, 1.0, 6.0, ts1)
    close = _cr_pos(480.0, Call, -1, 1.0, 5.0, ts2)
    s = pnl_series([open, close], 500.0)
    @test total_pnl(s) ≈ -1.0
    @test n_round_trips(s) == 1
    @test hit_rate(s) == 0.0
end

@testset "core metrics: mixed -- hit rate counts strictly positive" begin
    ts1 = DateTime(2024, 1, 15, 15, 30)
    ts2 = DateTime(2024, 1, 15, 15, 35)
    ts3 = DateTime(2024, 1, 15, 16, 30)
    ts4 = DateTime(2024, 1, 15, 16, 35)
    # Round trip 1: +1.0 (winner). Round trip 2: -0.5 (loser).
    open1  = _cr_pos(480.0, Call, +1, 1.0, 5.0, ts1)
    close1 = _cr_pos(480.0, Call, -1, 1.0, 6.0, ts3)
    open2  = _cr_pos(470.0, Put, +1, 1.0, 4.0, ts2)
    close2 = _cr_pos(470.0, Put, -1, 1.0, 3.5, ts4)
    s = pnl_series([open1, open2, close1, close2], 500.0)
    @test total_pnl(s) ≈ 0.5
    @test n_round_trips(s) == 2
    @test hit_rate(s) == 0.5
end

@testset "core metrics: zero-PnL trade does not count as a win" begin
    ts1 = DateTime(2024, 1, 15, 15, 30)
    ts2 = DateTime(2024, 1, 15, 16, 30)
    open  = _cr_pos(480.0, Call, +1, 1.0, 5.0, ts1)
    close = _cr_pos(480.0, Call, -1, 1.0, 5.0, ts2)   # exact breakeven
    s = pnl_series([open, close], 500.0)
    @test total_pnl(s) == 0.0
    @test n_round_trips(s) == 1
    @test hit_rate(s) == 0.0
end
