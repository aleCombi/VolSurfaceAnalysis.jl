# Tests for run_backtest and resolve_quote, on an in-memory MarketData.
#
# Quotes here carry real bid/ask so open_position can actually fill. The
# map serves OptionQuote directly from an InMemory fixture: no bars, no
# surface, since these policies never look at one.

const _EN_UND = Underlying("SPY")
const _EN_CLOCK = Clock{OptionQuote}(_EN_UND)

function _en_fixture()
    ts1 = DateTime(2024, 1, 15, 15, 30)
    ts2 = DateTime(2024, 1, 15, 15, 31)
    ts3 = DateTime(2024, 1, 15, 15, 32)
    spot = 480.0
    expiry = DateTime(2024, 2, 16, 21, 0)
    mk_quote(ts, strike, otype, bid, ask) = OptionQuote(
        "X", _EN_UND, expiry, strike, otype,
        bid, ask, (bid + ask) / 2, missing, missing, missing, ts,
    )
    quotes = OptionQuote[]
    spots  = SpotPrice[]
    for ts in (ts1, ts2, ts3)
        push!(quotes, mk_quote(ts, 480.0, Call, 5.00, 5.10))
        push!(quotes, mk_quote(ts, 480.0, Put,  4.80, 4.90))
        push!(spots, SpotPrice(_EN_UND, spot, ts))
    end
    data = MarketData(InMemory(quotes), InMemory(spots))
    (data=data, ts1=ts1, ts2=ts2, ts3=ts3, expiry=expiry, spot=spot)
end

# A test policy that opens one long call at ts1 and nothing else.
struct _OpenOnceAt <: Policy
    when::DateTime
    trade::Trade
end

function VolSurfaceAnalysis.decide(s::_OpenOnceAt, t::DateTime,
                                   ::TimeCut,
                                   ::AbstractVector{Position})::Vector{Trade}
    return t == s.when ? Trade[s.trade] : Trade[]
end

# A test policy that opens at ts1 and closes (counter-trade) at ts2.
struct _OpenThenClose <: Policy
    open_at::DateTime
    close_at::DateTime
    contract::Trade        # the open trade; close is its mirror
end

function VolSurfaceAnalysis.decide(s::_OpenThenClose, t::DateTime,
                                   ::TimeCut,
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

@testset "run_backtest(policy): NoOpPolicy yields empty ledger" begin
    f = _en_fixture()
    positions = run_backtest(NoOpPolicy(), f.data, f.ts1, f.ts3, _EN_CLOCK)
    @test isempty(positions)
end

@testset "run_backtest(policy): single fill at scheduled tick" begin
    f = _en_fixture()
    trd = Trade(_EN_UND, 480.0, f.expiry, Call)
    positions = run_backtest(_OpenOnceAt(f.ts2, trd), f.data, f.ts1, f.ts3, _EN_CLOCK)
    @test length(positions) == 1
    pos = positions[1]
    @test pos.trade === trd
    @test pos.entry_timestamp == f.ts2
    @test pos.entry_price == 5.10           # long crosses ask
    @test pos.entry_spot  == f.spot
end

@testset "run_backtest(policy): counter-trade close lands in ledger" begin
    f = _en_fixture()
    open_trade = Trade(_EN_UND, 480.0, f.expiry, Call)
    s = _OpenThenClose(f.ts1, f.ts3, open_trade)
    positions = run_backtest(s, f.data, f.ts1, f.ts3, _EN_CLOCK)
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

@testset "run_backtest(agent): StaticAgent matches bare-policy result" begin
    f = _en_fixture()
    trd = Trade(_EN_UND, 480.0, f.expiry, Call)
    p = _OpenOnceAt(f.ts2, trd)
    via_policy = run_backtest(p, f.data, f.ts1, f.ts3, _EN_CLOCK)
    via_agent  = run_backtest(StaticAgent(p), f.data, f.ts1, f.ts3, _EN_CLOCK)
    @test length(via_agent) == length(via_policy) == 1
    @test via_agent[1].trade === via_policy[1].trade
    @test via_agent[1].entry_timestamp == via_policy[1].entry_timestamp
    @test via_agent[1].entry_price == via_policy[1].entry_price
end

# An agent that swaps from NoOpPolicy to _OpenOnceAt at a chosen instant.
# Demonstrates the engine actually re-queries current_policy each tick.
struct _SwapAgent <: Agent
    swap_at::DateTime
    after::Policy
end

function VolSurfaceAnalysis.current_policy(a::_SwapAgent, t::DateTime,
                                           ::TimeCut,
                                           ::AbstractVector{Position})
    t < a.swap_at ? NoOpPolicy() : a.after
end

@testset "run_backtest(agent): swap-mid-run agent acts only after swap" begin
    f = _en_fixture()
    trd = Trade(_EN_UND, 480.0, f.expiry, Call)
    agent_fires = _SwapAgent(f.ts2, _OpenOnceAt(f.ts3, trd))
    agent_silent = _SwapAgent(f.ts3 + Second(1), _OpenOnceAt(f.ts3, trd))
    @test length(run_backtest(agent_fires,  f.data, f.ts1, f.ts3, _EN_CLOCK)) == 1
    @test isempty(run_backtest(agent_silent, f.data, f.ts1, f.ts3, _EN_CLOCK))
end

@testset "run_backtest: the clock defines the ticks" begin
    f = _en_fixture()
    trd = Trade(_EN_UND, 480.0, f.expiry, Call)
    # A clock on a selector nothing serves is a broken configuration, not an
    # empty grid: enumerating it throws rather than running zero ticks.
    @test_throws UnservedSelector run_backtest(_OpenOnceAt(f.ts2, trd), f.data,
                                               f.ts1, f.ts3,
                                               Clock{OptionQuote}(Underlying("QQQ")))
    # A clock on the spot grid ticks at the same instants here.
    @test length(run_backtest(_OpenOnceAt(f.ts2, trd), f.data, f.ts1, f.ts3,
                              Clock{SpotPrice}(_EN_UND))) == 1
end

@testset "run_backtest: missing spot at a fill errors" begin
    f = _en_fixture()
    trd = Trade(_EN_UND, 480.0, f.expiry, Call)
    # served, but no row at the fill instant: the loud "missing spot" error
    thin_spots = MarketData(entry(f.data, OptionQuote),
                            InMemory([SpotPrice(_EN_UND, f.spot, f.ts1)]))
    @test_throws ErrorException run_backtest(_OpenOnceAt(f.ts2, trd), thin_spots,
                                             f.ts1, f.ts3, _EN_CLOCK)
    # nothing serves SpotPrice for SPY at all: structural, so it is named
    no_spots = MarketData(entry(f.data, OptionQuote), InMemory(SpotPrice[]))
    @test_throws UnservedSelector run_backtest(_OpenOnceAt(f.ts2, trd), no_spots,
                                               f.ts1, f.ts3, _EN_CLOCK)
end

@testset "resolve_quote: strike not in chain errors" begin
    f = _en_fixture()
    cut = TimeCut(f.data, f.ts1)
    bogus = Trade(_EN_UND, 999.0, f.expiry, Call)
    @test_throws ErrorException resolve_quote(cut, bogus, f.ts1)
    @test_throws ErrorException resolve_quote(cut, Trade(_EN_UND, 480.0, f.expiry, Call), f.ts2)  # masked
end

@testset "resolve_quote: returns matching contract" begin
    f = _en_fixture()
    cut = TimeCut(f.data, f.ts1)
    put_trade = Trade(_EN_UND, 480.0, f.expiry, Put)
    q = resolve_quote(cut, put_trade, f.ts1)
    @test q.strike == 480.0
    @test q.option_type == Put
    @test q.bid == 4.80
    @test q.ask == 4.90
end
