# Tests for run_backtest and resolve_quote.
#
# Quotes here carry real bid/ask so open_position can actually fill.

const _EN_UND = Underlying("SPY")

function _en_fixture()
    ts1 = DateTime(2024, 1, 15, 15, 30)
    ts2 = DateTime(2024, 1, 15, 15, 31)
    ts3 = DateTime(2024, 1, 15, 15, 32)
    spot = 480.0; r = 0.04; q = 0.015
    expiry = DateTime(2024, 2, 16, 21, 0)
    T = time_to_expiry(expiry, ts1)
    mk_quote(ts, strike, otype, bid, ask) = OptionQuote(
        "X", _EN_UND, expiry, strike, otype,
        bid, ask, (bid + ask) / 2, missing, missing, missing, ts,
    )
    mk_chain(ts) = [
        mk_quote(ts, 480.0, Call, 5.00, 5.10),
        mk_quote(ts, 480.0, Put,  4.80, 4.90),
    ]
    chains = Dict(ts1 => mk_chain(ts1), ts2 => mk_chain(ts2), ts3 => mk_chain(ts3))
    spots  = Dict(ts1 => spot,           ts2 => spot,           ts3 => spot)
    inner  = InMemoryDataSource(_EN_UND; chains=chains, spots=spots)
    mds    = ModelDataSource(inner; rate=FlatCurve(r), div=FlatCurve(q))
    (mds=mds, ts1=ts1, ts2=ts2, ts3=ts3, expiry=expiry, spot=spot)
end

# A test strategy that opens one long call at ts1 and nothing else.
struct _OpenOnceAt <: Strategy
    when::DateTime
    trade::Trade
end

function VolSurfaceAnalysis.decide(s::_OpenOnceAt, t::DateTime,
                                   ::TimeCutModelDataSource,
                                   ::AbstractVector{Position})::Vector{Trade}
    return t == s.when ? Trade[s.trade] : Trade[]
end

# A test strategy that opens at ts1 and closes (counter-trade) at ts2.
struct _OpenThenClose <: Strategy
    open_at::DateTime
    close_at::DateTime
    contract::Trade        # the open trade; close is its mirror
end

function VolSurfaceAnalysis.decide(s::_OpenThenClose, t::DateTime,
                                   ::TimeCutModelDataSource,
                                   ::AbstractVector{Position})::Vector{Trade}
    if t == s.open_at
        return Trade[s.contract]
    elseif t == s.close_at
        c = s.contract
        return Trade[Trade(c.underlying, c.strike, c.expiry, c.option_type;
                           direction=-c.direction, quantity=c.quantity)]
    else
        return Trade[]
    end
end

@testset "run_backtest: NoOpStrategy yields empty ledger" begin
    f = _en_fixture()
    positions = run_backtest(NoOpStrategy(), f.mds, f.ts1, f.ts3)
    @test isempty(positions)
end

@testset "run_backtest: single fill at scheduled tick" begin
    f = _en_fixture()
    trd = Trade(_EN_UND, 480.0, f.expiry, Call)
    positions = run_backtest(_OpenOnceAt(f.ts2, trd), f.mds, f.ts1, f.ts3)
    @test length(positions) == 1
    pos = positions[1]
    @test pos.trade === trd
    @test pos.entry_timestamp == f.ts2
    @test pos.entry_price == 5.10           # long crosses ask
    @test pos.entry_spot  == f.spot
end

@testset "run_backtest: counter-trade close lands in ledger" begin
    f = _en_fixture()
    open_trade = Trade(_EN_UND, 480.0, f.expiry, Call)
    s = _OpenThenClose(f.ts1, f.ts3, open_trade)
    positions = run_backtest(s, f.mds, f.ts1, f.ts3)
    @test length(positions) == 2
    @test positions[1].entry_timestamp == f.ts1
    @test positions[1].trade.direction == 1
    @test positions[2].entry_timestamp == f.ts3
    @test positions[2].trade.direction == -1
    # The contract net is flat -- a long and a short of the same call.
    @test positions[1].trade.underlying  == positions[2].trade.underlying
    @test positions[1].trade.strike      == positions[2].trade.strike
    @test positions[1].trade.expiry      == positions[2].trade.expiry
    @test positions[1].trade.option_type == positions[2].trade.option_type
end

@testset "resolve_quote: strike not in chain errors" begin
    f = _en_fixture()
    cut = TimeCutModelDataSource(f.mds, f.ts1)
    bogus = Trade(_EN_UND, 999.0, f.expiry, Call)
    @test_throws ErrorException resolve_quote(cut, bogus, f.ts1)
end

@testset "resolve_quote: returns matching contract" begin
    f = _en_fixture()
    cut = TimeCutModelDataSource(f.mds, f.ts1)
    put_trade = Trade(_EN_UND, 480.0, f.expiry, Put)
    q = resolve_quote(cut, put_trade, f.ts1)
    @test q.strike == 480.0
    @test q.option_type == Put
    @test q.bid == 4.80
    @test q.ask == 4.90
end
