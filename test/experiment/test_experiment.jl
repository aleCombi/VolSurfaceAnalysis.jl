# Tests for the Experiment orchestrator, on in-memory MarketData maps.
# Quotes in the fixture are whole cents (5.00/5.10 call, 4.80/4.90 put,
# spot 480), so cash literals are whole cents: one contract at 5.10 is
# 51000, and IBKR's commission on a lone contract is 65 cents raised to
# the USD 1.00 minimum.

const _EX_UND = Underlying("SPY")
const _EX_USD = Currency("USD")
const _EX_CLOCK = Clock{OptionQuote}(_EX_UND)

function _ex_map(quotes::Vector{OptionQuote}, spots::Vector{SpotPrice}; r=0.04, q=0.015)
    MarketData(InMemory(quotes), InMemory(spots),
               Constant(RateCurve(_EX_USD, FlatCurve(r))),
               Constant(DivCurve(_EX_UND, FlatCurve(q))),
               SurfaceFrom(currency=_EX_USD))
end

function _ex_fixture()
    ts1 = DateTime(2024, 1, 15, 15, 30)
    ts2 = DateTime(2024, 1, 15, 15, 31)
    ts3 = DateTime(2024, 1, 15, 15, 32)
    spot = 480.0
    expiry = DateTime(2024, 2, 16, 21, 0)
    mk_quote(ts, strike, otype, bid, ask) = OptionQuote(
        "X", _EX_UND, expiry, strike, otype,
        bid, ask, (bid + ask) / 2, missing, missing, missing, ts,
    )
    quotes = OptionQuote[]
    spots  = SpotPrice[]
    for ts in (ts1, ts2, ts3)
        push!(quotes, mk_quote(ts, 480.0, Call, 5.00, 5.10))
        push!(quotes, mk_quote(ts, 480.0, Put,  4.80, 4.90))
        push!(spots, SpotPrice(_EX_UND, spot, ts))
    end
    call = ContractKey(_EX_UND, 480.0, expiry, Call)
    put  = ContractKey(_EX_UND, 480.0, expiry, Put)
    (data=_ex_map(quotes, spots), ts1=ts1, ts2=ts2, ts3=ts3, expiry=expiry, spot=spot,
     call=call, put=put)
end

# Policy that emits one order at a chosen tick and does nothing else.
struct _ExOpenOnceAt <: Policy
    when::DateTime
    order::Order
end

VolSurfaceAnalysis.decide(s::_ExOpenOnceAt, t::DateTime, ::TimeCut, ::Book)::Vector{Order} =
    t == s.when ? Order[s.order] : Order[]

# An opening order for one long contract.
_ex_long(contract::ContractKey) = Order(:long, [Leg(contract, Long, 1, Open)])

@testset "Experiment: kwarg constructor round-trips fields" begin
    f = _ex_fixture()
    exp = Experiment(name="smoke", agent=StaticAgent(NoOpPolicy()),
                     data=f.data, clock=_EX_CLOCK, from=f.ts1, to=f.ts3,
                     outputs=OutputSpec(metrics=[:sharpe]))
    @test exp.name == "smoke"
    @test exp.agent isa StaticAgent
    @test exp.data === f.data
    @test exp.clock === _EX_CLOCK
    @test exp.from == f.ts1
    @test exp.to == f.ts3
    @test exp.outputs.metrics == [:sharpe]
end

@testset "Experiment: default outputs = all registered metrics" begin
    f = _ex_fixture()
    exp = Experiment(name="default", agent=StaticAgent(NoOpPolicy()),
                     data=f.data, clock=_EX_CLOCK, from=f.ts1, to=f.ts3)
    @test Set(exp.outputs.metrics) ==
          Set([:sharpe, :sortino, :max_drawdown, :volatility, :profit_factor])
    @test exp.outputs.artifacts == [:marked_curve]
end

@testset "run_experiment: NoOpPolicy -> empty result, provenance carried" begin
    f = _ex_fixture()
    exp = Experiment(name="noop", agent=StaticAgent(NoOpPolicy()),
                     data=f.data, clock=_EX_CLOCK, from=f.ts1, to=f.ts3)
    res = run_experiment(exp)
    @test res.experiment === exp
    @test res.ledger isa Ledger
    @test isempty(res.ledger)
    @test isempty(res.ledger.orders)
    @test isempty(trade_pnl(res.ledger))
    @test res.curve isa MarkedCurve
    @test res.metrics.total_pnl == 0.0
    @test res.metrics.n_round_trips == 0
    @test res.metrics.n_opens == 0 && res.metrics.n_closes == 0
    @test isnan(res.metrics.hit_rate)
end

@testset "run_experiment: an open lot at the window end stays open" begin
    # The leg's expiry (Feb 16) is past the window end (Jan 15). Nothing is
    # force-settled: the lot is open in the book at exp.to, the series is
    # empty, and the cash is the premium paid plus the commission.
    f = _ex_fixture()
    exp = Experiment(name="open-at-end",
                     agent=StaticAgent(_ExOpenOnceAt(f.ts2, _ex_long(f.call))),
                     data=f.data, clock=_EX_CLOCK, from=f.ts1, to=f.ts3)
    res = run_experiment(exp)
    @test length(res.ledger.orders) == 1
    @test [typeof(e) for e in res.ledger.events] == [Fill, Fee]
    book = book_effective(res.ledger, exp.to)
    @test length(open_lots(book)) == 1
    @test only(open_lots(book)).contract == f.call
    @test book.cash == -51000 - 100                     # 5.10 paid, 65 cents raised to USD 1.00
    @test isempty(trade_pnl(res.ledger))
    @test res.metrics.total_pnl == 0.0
    @test res.metrics.n_opens == 1 && res.metrics.n_closes == 0
    # The window is two minutes inside one session, so no whole session lies
    # in it: the grid is empty, and that is temporal absence, not a failure.
    @test n_marked(res.curve) == 0 && n_unmarked(res.curve) == 0
end

@testset "run_experiment: an expiry inside the window is booked" begin
    # A map with a real spot at the leg's expiry, which is the window end.
    ts1 = DateTime(2024, 1, 15, 15, 30)
    ts2 = DateTime(2024, 1, 15, 15, 31)
    ts3 = DateTime(2024, 1, 15, 15, 32)
    expiry = ts3                                       # leg expires at window end
    spot   = 480.0
    mk_q(ts, K) = OptionQuote("X", _EX_UND, expiry, K, Call,
                              5.00, 5.10, 5.05, missing, missing, missing, ts)
    data = _ex_map([mk_q(ts1, 480.0), mk_q(ts2, 480.0), mk_q(ts3, 480.0)],
                   [SpotPrice(_EX_UND, spot, ts) for ts in (ts1, ts2, ts3)])
    call = ContractKey(_EX_UND, 480.0, expiry, Call)
    exp = Experiment(name="held-to-expiry",
                     agent=StaticAgent(_ExOpenOnceAt(ts2, _ex_long(call))),
                     data=data, clock=_EX_CLOCK, from=ts1, to=ts3)
    res = run_experiment(exp)
    @test length(res.ledger.orders) == 1
    @test count(e -> e isa Fill, res.ledger.events) == 1
    @test any(e isa Expiry for e in res.ledger.events)
end

@testset "run_experiment: a QQQ leg under a SPY clock fills against QQQ" begin
    # The clock says when to step, not whose price: the fill and its
    # observation are QQQ's own quote and spot at the tick.
    ts1 = DateTime(2024, 1, 15, 15, 30)
    ts3 = DateTime(2024, 1, 15, 15, 32)
    qqq = Underlying("QQQ")
    far = DateTime(2024, 2, 16, 21, 0)                 # past the window end
    spy_q(ts) = OptionQuote("SPY", _EX_UND, far, 480.0, Call,
                            5.00, 5.10, 5.05, missing, missing, missing, ts)
    qqq_q = OptionQuote("QQQ", qqq, far, 400.0, Call,
                        1.00, 1.10, 1.05, missing, missing, missing, ts1)
    data = _ex_map([spy_q(ts1), spy_q(ts3), qqq_q],
                   [SpotPrice(_EX_UND, 480.0, ts1), SpotPrice(_EX_UND, 480.0, ts3),
                    SpotPrice(qqq, 400.0, ts1)])
    qqq_call = ContractKey(qqq, 400.0, far, Call)
    exp = Experiment(name="foreign-leg",
                     agent=StaticAgent(_ExOpenOnceAt(ts1, _ex_long(qqq_call))),
                     data=data, clock=_EX_CLOCK, from=ts1, to=ts3)
    res = run_experiment(exp)
    fill = only(e for e in res.ledger.events if e isa Fill)
    @test fill.price == 1.10                            # QQQ's ask, not SPY's
    @test fill.contract == qqq_call
    obs = only(only(res.ledger.orders).observations)
    @test obs.spot == 400.0 && obs.spot_at == ts1        # QQQ's spot at the tick
    @test obs.bid == 1.00 && obs.ask == 1.10
    @test length(open_lots(book_effective(res.ledger, exp.to))) == 1
    @test isempty(trade_pnl(res.ledger))
end

@testset "run_experiment: requested optional metric appears in result" begin
    f = _ex_fixture()
    exp = Experiment(name="with-sharpe",
                     agent=StaticAgent(NoOpPolicy()),
                     data=f.data, clock=_EX_CLOCK, from=f.ts1, to=f.ts3,
                     outputs=OutputSpec(metrics=[:sharpe, :max_drawdown]))
    res = run_experiment(exp)
    @test haskey(res.metrics, :sharpe)
    @test haskey(res.metrics, :max_drawdown)
    @test isnan(res.metrics.sharpe)               # empty series
    @test res.metrics.max_drawdown == 0.0
end

@testset "run_experiment: unknown metric symbol errors" begin
    f = _ex_fixture()
    exp = Experiment(name="bogus", agent=StaticAgent(NoOpPolicy()),
                     data=f.data, clock=_EX_CLOCK, from=f.ts1, to=f.ts3,
                     outputs=OutputSpec(metrics=[:nonsense_metric]))
    @test_throws ErrorException run_experiment(exp)
end

@testset "run_experiment: no clock tick in the window errors" begin
    f = _ex_fixture()
    early = DateTime(2024, 1, 1, 0, 0)
    later = DateTime(2024, 1, 2, 0, 0)
    exp = Experiment(name="empty-window",
                     agent=StaticAgent(NoOpPolicy()),
                     data=f.data, clock=_EX_CLOCK, from=early, to=later)
    @test_throws ErrorException run_experiment(exp)
    # ticks exist before `from` but none inside the window
    after = Experiment(name="after-data", agent=StaticAgent(NoOpPolicy()),
                       data=f.data, clock=_EX_CLOCK, from=f.ts3 + Hour(1), to=f.ts3 + Hour(2))
    @test_throws ErrorException run_experiment(after)
    # a clock whose selector is not an Underlying: an experiment ticks on an underlying's grid
    exp3 = Experiment(name="ccy-clock", agent=StaticAgent(NoOpPolicy()),
                      data=f.data, clock=Clock{RateCurve}(_EX_USD), from=f.ts1, to=f.ts3)
    @test_throws ErrorException run_experiment(exp3)
    err = try run_experiment(exp3); nothing catch e; e end
    @test occursin("underlying's grid", err.msg)
end

@testset "run_experiment: provenance allows rerun via result.experiment" begin
    f = _ex_fixture()
    exp = Experiment(name="rerun",
                     agent=StaticAgent(_ExOpenOnceAt(f.ts2, _ex_long(f.call))),
                     data=f.data, clock=_EX_CLOCK, from=f.ts1, to=f.ts3)
    res1 = run_experiment(exp)
    res2 = run_experiment(res1.experiment)
    @test res1.metrics.total_pnl == res2.metrics.total_pnl
    @test length(res1.ledger) == length(res2.ledger) == 2
    @test length(res1.ledger.orders) == length(res2.ledger.orders) == 1
    @test book_as_known(res1.ledger, 2) == book_as_known(res2.ledger, 2)
end

# ---- DailyShortStrangle e2e ------------------------------------------------

# Multi-strike, two-expiry fixture priced from flat 20% BS so the surface
# inverts cleanly and `invert_delta` has a wide observed bracket. The
# quotes are not on the tick; the venue rounds the fills onto it.
function _strangle_ex_fixture()
    entry_ts = DateTime(2024, 6, 3, 15, 45)             # the entry tick
    pre_ts   = DateTime(2024, 6, 3, 15, 44)             # one minute before
    end_ts   = DateTime(2024, 6, 3, 15, 46)             # window end
    spot, r, q, sigma = 480.0, 0.045, 0.013, 0.20
    e_target = DateTime(2024, 6, 4, 20, 0)              # first expiry on/after entry+1d
    e_far    = DateTime(2024, 6, 7, 20, 0)
    strikes = 440.0:5.0:520.0

    function mk_q(ts, K, expiry, otype)
        T = time_to_expiry(expiry, ts)
        mark = bs_price(spot, K, T, sigma, otype; r=r, q=q)
        spread = max(0.02, 0.01 * mark)
        OptionQuote("X", _EX_UND, expiry, K, otype,
                    mark - spread / 2, mark + spread / 2, mark,
                    missing, missing, missing, ts)
    end

    quotes = OptionQuote[]
    spots  = SpotPrice[]
    for ts in (pre_ts, entry_ts, end_ts)
        for K in strikes, e in (e_target, e_far)
            push!(quotes, mk_q(ts, K, e, K >= spot ? Call : Put))
        end
        push!(spots, SpotPrice(_EX_UND, spot, ts))
    end
    (data=_ex_map(quotes, spots; r, q), pre_ts=pre_ts, entry_ts=entry_ts, end_ts=end_ts,
     spot=spot, e_target=e_target)
end

@testset "run_experiment: DailyShortStrangle books one two-leg order at the entry tick" begin
    f = _strangle_ex_fixture()
    policy = DailyShortStrangle(; underlying=_EX_UND,
                                entry_time=Time(15, 45),
                                expiry_interval=Day(1),
                                put_delta=0.20, call_delta=0.20,
                                quantity=1)
    exp = Experiment(name="strangle-e2e",
                     agent=StaticAgent(policy),
                     data=f.data, clock=_EX_CLOCK, from=f.pre_ts, to=f.end_ts)
    res = run_experiment(exp)
    L = res.ledger

    # The gate fires exactly once over [pre_ts, end_ts]: one order, two legs.
    @test length(L.orders) == 1
    rec = only(L.orders)
    @test rec.order.label == :daily_short_strangle
    @test length(rec.order.legs) == 2 && length(rec.observations) == 2
    @test rec.decided_at == f.entry_ts && rec.known_to == 0
    fills = [e for e in L.events if e isa Fill]
    @test length(fills) == 2
    @test all(x.side == Short && x.intent == Open && x.group == 1 for x in fills)
    @test all(effective_at(x) == f.entry_ts for x in fills)      # one structure, one instant
    @test all(x.contract.expiry == f.e_target for x in fills)
    @test all(x.fill_rule == :cross_spread for x in fills)
    # the fills are on the tick, at or below the bid the venue saw
    for (x, obs) in zip(fills, rec.observations)
        @test x.price * 100 ≈ round(x.price * 100)
        @test x.price <= obs.bid
        @test x.price == fill_price(:cross_spread, obs.bid, obs.ask, Short, 1)
        @test obs.spot == f.spot && obs.spot_at == f.entry_ts && obs.quote_at == f.entry_ts
    end
    # both premiums are in IBKR's 65-cent tier, so the order costs 130, above the minimum
    @test all(x.price >= 0.10 for x in fills)
    fees = [e for e in L.events if e isa Fee]
    @test [x.amount for x in fees] == [-65, -65]
    @test [x.source_id for x in fees] == [event_id(x) for x in fills]

    # nothing closes in this slice: no trades, two opens, one open group
    @test isempty(trade_pnl(res.ledger))
    @test res.metrics.n_opens == 2
    @test res.metrics.n_closes == 0
    @test open_groups(book_effective(L, exp.to)) == [1]
    @test length(lots(book_effective(L, exp.to), 1)) == 2
end
