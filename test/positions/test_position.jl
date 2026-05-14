const _POS_UND = Underlying("SPY")
const _POS_EXP = DateTime(2024, 3, 15, 21, 0)
const _POS_TS  = DateTime(2024, 1, 15, 15, 30)

function _mk_quote(; underlying=_POS_UND, strike=100.0, expiry=_POS_EXP,
                     option_type=Call, bid=1.20, ask=1.25,
                     ts=_POS_TS)
    OptionQuote(
        "X", underlying, expiry, strike, option_type,
        bid, ask, missing, missing, missing, missing, ts,
    )
end

@testset "open_position: long fills at ask" begin
    t = Trade(_POS_UND, 100.0, _POS_EXP, Call)
    q = _mk_quote(bid=1.20, ask=1.25)
    pos = open_position(t, q, 99.5)

    @test pos.trade === t
    @test pos.entry_price == 1.25
    @test pos.entry_spot  == 99.5
    @test pos.entry_bid   == 1.20
    @test pos.entry_ask   == 1.25
    @test pos.entry_timestamp == _POS_TS
end

@testset "open_position: short fills at bid" begin
    t = Trade(_POS_UND, 100.0, _POS_EXP, Call; direction=-1)
    q = _mk_quote(bid=1.20, ask=1.25)
    pos = open_position(t, q, 99.5)
    @test pos.entry_price == 1.20
end

@testset "open_position: missing fill side throws" begin
    t_long  = Trade(_POS_UND, 100.0, _POS_EXP, Call)
    t_short = Trade(_POS_UND, 100.0, _POS_EXP, Call; direction=-1)

    @test_throws ArgumentError open_position(t_long,  _mk_quote(ask=missing), 99.5)
    @test_throws ArgumentError open_position(t_short, _mk_quote(bid=missing), 99.5)

    # Long is fine even when bid is missing (we only need ask).
    pos = open_position(t_long, _mk_quote(bid=missing), 99.5)
    @test pos.entry_price == 1.25
    @test ismissing(pos.entry_bid)
end

@testset "open_position: contract mismatch throws" begin
    t = Trade(_POS_UND, 100.0, _POS_EXP, Call)
    @test_throws ArgumentError open_position(t, _mk_quote(underlying=Underlying("QQQ")), 99.5)
    @test_throws ArgumentError open_position(t, _mk_quote(strike=105.0), 99.5)
    @test_throws ArgumentError open_position(t, _mk_quote(expiry=DateTime(2024, 4, 19, 21, 0)), 99.5)
    @test_throws ArgumentError open_position(t, _mk_quote(option_type=Put), 99.5)
end

@testset "entry_cost: sign reflects direction" begin
    long  = open_position(Trade(_POS_UND, 100.0, _POS_EXP, Call),
                          _mk_quote(bid=1.20, ask=1.25), 99.5)
    short = open_position(Trade(_POS_UND, 100.0, _POS_EXP, Call; direction=-1),
                          _mk_quote(bid=1.20, ask=1.25), 99.5)

    @test entry_cost(long)  ==  1.25   # paid premium
    @test entry_cost(short) == -1.20   # received premium

    # Scales with quantity.
    pos2 = open_position(Trade(_POS_UND, 100.0, _POS_EXP, Call; quantity=3.0),
                         _mk_quote(bid=1.20, ask=1.25), 99.5)
    @test entry_cost(pos2) == 3.75
end

@testset "pnl: payoff minus entry_cost" begin
    # Long call: paid 1.25, expires at spot=110 (intrinsic 10) -> PnL 8.75
    long = open_position(Trade(_POS_UND, 100.0, _POS_EXP, Call),
                         _mk_quote(bid=1.20, ask=1.25), 99.5)
    @test pnl(long, 110.0) ==  10.0 - 1.25
    @test pnl(long,  95.0) == -1.25

    # Short put: received 1.20, expires at spot=95 (put ITM -5 for the short) -> PnL -5 - (-1.20) = -3.80
    short_put = open_position(Trade(_POS_UND, 100.0, _POS_EXP, Put; direction=-1),
                              _mk_quote(option_type=Put, bid=1.20, ask=1.25), 99.5)
    @test pnl(short_put,  95.0) ≈ -5.0 + 1.20  atol=1e-12
    @test pnl(short_put, 105.0) ==  1.20      # OTM expiry, keep the premium
end

@testset "pnl: vector sums across legs" begin
    # Buy + sell same contract: payoffs cancel, PnL = -(ask - bid) per share.
    q = _mk_quote(bid=1.20, ask=1.25)
    long  = open_position(Trade(_POS_UND, 100.0, _POS_EXP, Call),                   q, 99.5)
    short = open_position(Trade(_POS_UND, 100.0, _POS_EXP, Call; direction=-1),     q, 99.5)

    @test pnl([long, short], 100.0) ≈ -(1.25 - 1.20) atol=1e-12
    @test pnl([long, short], 150.0) ≈ -(1.25 - 1.20) atol=1e-12

    # Empty vector returns 0.
    @test pnl(Position[], 100.0) == 0.0
end
