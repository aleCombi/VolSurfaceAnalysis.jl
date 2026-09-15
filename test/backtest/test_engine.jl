# Tests for run_backtest, fill_legs, resolve_quote and check_join, on an
# in-memory MarketData whose quotes are whole cents (5.00/5.10 call,
# 4.80/4.90 put, spot 480), so fills land at the quoted side and the cash
# literals are whole cents: one contract at 5.10 is 51000. Commissions
# are IBKR's: one contract at any of these premiums is 65 cents, raised
# to the USD 1.00 minimum when it is the only leg. Needs
# test/ledger/fixtures.jl for _lg_seen and the _LG_ contracts.

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
    call = ContractKey(_EN_UND, 480.0, expiry, Call)
    put  = ContractKey(_EN_UND, 480.0, expiry, Put)
    (data=data, ts1=ts1, ts2=ts2, ts3=ts3, expiry=expiry, spot=spot, call=call, put=put)
end

# A test policy that emits one order at one instant and nothing else.
struct _OpenOnceAt <: Policy
    when::DateTime
    order::Order
end

VolSurfaceAnalysis.decide(s::_OpenOnceAt, t::DateTime, ::TimeCut, ::Book)::Vector{Order} =
    t == s.when ? Order[s.order] : Order[]

# A test policy that opens a long call at one instant and closes it, with a
# Close leg naming the only open group, at another.
struct _OpenThenClose <: Policy
    open_at::DateTime
    close_at::DateTime
    contract::ContractKey
end

function VolSurfaceAnalysis.decide(s::_OpenThenClose, t::DateTime, ::TimeCut, book::Book)::Vector{Order}
    if t == s.open_at
        return Order[Order(:open, [Leg(s.contract, Long, 1, Open)])]
    elseif t == s.close_at
        return Order[Order(:close, [Leg(s.contract, Short, 1, Close)]; group=only(open_groups(book)))]
    end
    return Order[]
end

# A policy that records a deep copy of the book it is handed on every tick,
# and otherwise behaves like _OpenThenClose.
mutable struct _Recording <: Policy
    inner::_OpenThenClose
    seen::Vector{Pair{DateTime,Book}}
end
_Recording(inner) = _Recording(inner, Pair{DateTime,Book}[])

function VolSurfaceAnalysis.decide(s::_Recording, t::DateTime, cut::TimeCut, book::Book)::Vector{Order}
    push!(s.seen, t => deepcopy(book))
    return decide(s.inner, t, cut, book)
end

@testset "run_backtest(policy): NoOpPolicy yields an empty ledger with no orders" begin
    f = _en_fixture()
    L = run_backtest(NoOpPolicy(), f.data, f.ts1, f.ts3, _EN_CLOCK).ledger
    @test L isa Ledger
    @test isempty(L)
    @test isempty(L.orders)
    @test (L.next_id, L.next_group, L.next_order_id, L.next_leg_id) == (1, 1, 1, 1)
end

@testset "fill_legs: prices every leg through the venue and writes nothing" begin
    f = _en_fixture()
    cut = TimeCut(f.data, f.ts1)
    order = Order(:straddle, [Leg(f.call, Long, 1, Open), Leg(f.put, Long, 1, Open)])
    legs = fill_legs(cut, order, f.ts1; fill_rule=:cross_spread, cost_model=:ibkr_pro_us_options, tick_cents=1)
    @test legs.prices == [5.10, 4.90]              # a buy crosses to the ask
    @test legs.fees == [-65, -65]                  # two legs at 65 cents each, above the minimum
    @test legs.fill_rule == :cross_spread
    @test length(legs.observations) == 2
    o1, o2 = legs.observations
    @test o1.quote_at == o1.spot_at == f.ts1
    @test o1.bid == 5.00 && o1.ask == 5.10 && o1.spot == 480.0
    @test o2.bid == 4.80 && o2.ask == 4.90 && o2.spot == 480.0 && o2.quote_at == f.ts1
    @test keys(legs) == (:prices, :fees, :observations, :fill_rule)
    # a free venue
    free = fill_legs(cut, order, f.ts1; fill_rule=:cross_spread, cost_model=:none, tick_cents=1)
    @test free.fees == [0, 0] && free.prices == legs.prices
    # a short leg crosses to the bid
    short = fill_legs(cut, Order(:sell, [Leg(f.put, Short, 2, Open)]), f.ts1;
                      fill_rule=:cross_spread, cost_model=:ibkr_pro_us_options, tick_cents=1)
    @test short.prices == [4.80] && short.fees == [-130]
end

@testset "run_backtest(policy): a single fill books one order, one fill and one fee" begin
    f = _en_fixture()
    p = _OpenOnceAt(f.ts2, Order(:call, [Leg(f.call, Long, 1, Open)]))
    L = run_backtest(p, f.data, f.ts1, f.ts3, _EN_CLOCK).ledger
    @test length(L) == 2
    @test length(L.orders) == 1
    rec = only(L.orders)
    @test rec.order_id == 1 && rec.first_leg_id == 1 && rec.group == 1 && rec.known_to == 0
    @test rec.decided_at == f.ts2
    @test length(rec.observations) == 1
    @test rec.observations[1].quote_at == f.ts2 && rec.observations[1].spot == 480.0
    fill, fee = L.events
    @test fill isa Fill && fill.side == Long && fill.intent == Open && fill.price == 5.10
    @test fill.fill_rule == :cross_spread && fill.execution_id == 1 && fill.order_leg_id == 1
    @test effective_at(fill) == recorded_at(fill) == f.ts2
    @test fee isa Fee && fee.source_id == event_id(fill) && fee.amount == -100   # 65 raised to USD 1.00
    book = book_as_known(L, last_sequence(L))
    @test book.cash == -51000 - 100
    @test open_groups(book) == [1]
    @test check_join(L) === nothing
    # every order leg has exactly one fill; execution ids are unique
    fills = [e for e in L.events if e isa Fill]
    @test sort([x.order_leg_id for x in fills]) == 1:length(fills)
    @test allunique(x.execution_id for x in fills)
end

@testset "run_backtest(policy): open then close, the Close leg names the group" begin
    f = _en_fixture()
    L = run_backtest(_OpenThenClose(f.ts1, f.ts3, f.call), f.data, f.ts1, f.ts3, _EN_CLOCK).ledger
    @test [typeof(e) for e in L.events] == [Fill, Fee, Fill, Match, Fee]
    @test length(L.orders) == 2
    r1, r2 = L.orders
    @test r1.group == 1 && r1.first_leg_id == 1 && r1.known_to == 0 && r1.decided_at == f.ts1
    @test r2.group == 1 && r2.first_leg_id == 2 && r2.known_to == 2 && r2.decided_at == f.ts3
    @test r2.order.group == 1 && r2.order.legs[1].intent == Close
    o, fee1, c, m, fee2 = L.events
    @test o.price == 5.10 && c.price == 5.00                 # buy at the ask, sell at the bid
    @test fee1.amount == -100 && fee2.amount == -100
    @test m.open_fill_id == event_id(o) && m.close_fill_id == event_id(c) && m.quantity == 1
    book = book_as_known(L, last_sequence(L))
    @test book.cash == -51000 - 100 + 50000 - 100        # -1200
    @test isempty(open_lots(book))
    trips = round_trips(L)
    @test length(trips) == 1
    @test trips[1].pnl == -1200                          # (50000 - 51000) - 100 - 100
    @test trips[1].pnl == book.cash
    @test trade_pnl(L) ≈ [-12.00]
    @test n_opens(L) == 1 && n_closes(L) == 1
    @test check_join(L) === nothing
    @test book == book_effective(L, DateTime(2030, 1, 1))
end

@testset "run_backtest: the book handed to decide is the known book" begin
    f = _en_fixture()
    p = _Recording(_OpenThenClose(f.ts1, f.ts3, f.call))
    L = run_backtest(p, f.data, f.ts1, f.ts3, _EN_CLOCK).ledger
    @test length(p.seen) == 3                            # one per tick
    @test [t for (t, _) in p.seen] == [f.ts1, f.ts2, f.ts3]
    for rec in L.orders
        seen = only(b for (t, b) in p.seen if t == rec.decided_at)
        @test seen == book_as_known(L, rec.known_to)
    end
    @test p.seen[1].second == Book()                     # nothing known at the first tick
    @test p.seen[2].second.cash == -51100                # the open and its fee
    @test p.seen[2].second == p.seen[3].second           # no tick between changes anything
    final = book_as_known(L, last_sequence(L))
    @test final == book_effective(L, DateTime(2030, 1, 1))
    @test final.cash == -1200
end

# An agent that swaps from NoOpPolicy to another policy at a chosen instant.
struct _SwapAgent <: Agent
    swap_at::DateTime
    after::Policy
end

VolSurfaceAnalysis.current_policy(a::_SwapAgent, t::DateTime, ::TimeCut, ::Book) =
    t < a.swap_at ? NoOpPolicy() : a.after

@testset "run_backtest(agent): StaticAgent matches the bare-policy result" begin
    f = _en_fixture()
    p = _OpenOnceAt(f.ts2, Order(:call, [Leg(f.call, Long, 1, Open)]))
    via_policy = run_backtest(p, f.data, f.ts1, f.ts3, _EN_CLOCK).ledger
    via_agent  = run_backtest(StaticAgent(p), f.data, f.ts1, f.ts3, _EN_CLOCK).ledger
    @test length(via_agent) == length(via_policy) == 2
    @test length(via_agent.orders) == length(via_policy.orders) == 1
    @test via_agent.events[1].price == via_policy.events[1].price == 5.10
    @test book_as_known(via_agent, 2) == book_as_known(via_policy, 2)
end

@testset "run_backtest(agent): a swap-mid-run agent acts only after the swap" begin
    f = _en_fixture()
    order = Order(:call, [Leg(f.call, Long, 1, Open)])
    agent_fires  = _SwapAgent(f.ts2, _OpenOnceAt(f.ts3, order))
    agent_silent = _SwapAgent(f.ts3 + Second(1), _OpenOnceAt(f.ts3, order))
    fired = run_backtest(agent_fires, f.data, f.ts1, f.ts3, _EN_CLOCK).ledger
    @test length(fired.orders) == 1 && count(e -> e isa Fill, fired.events) == 1
    silent = run_backtest(agent_silent, f.data, f.ts1, f.ts3, _EN_CLOCK).ledger
    @test isempty(silent) && isempty(silent.orders)
end

@testset "run_backtest: the clock defines the ticks" begin
    f = _en_fixture()
    p = _OpenOnceAt(f.ts2, Order(:call, [Leg(f.call, Long, 1, Open)]))
    # A clock on a selector nothing serves is a broken configuration, not an
    # empty grid: enumerating it throws rather than running zero ticks.
    @test_throws UnservedSelector run_backtest(p, f.data, f.ts1, f.ts3,
                                               Clock{OptionQuote}(Underlying("QQQ"))).ledger
    # A clock on the spot grid ticks at the same instants here.
    L = run_backtest(p, f.data, f.ts1, f.ts3, Clock{SpotPrice}(_EN_UND)).ledger
    @test length(L.orders) == 1 && count(e -> e isa Fill, L.events) == 1
    # the venue's values are keywords: a free venue books no fees
    free = run_backtest(p, f.data, f.ts1, f.ts3, _EN_CLOCK; cost_model=:none).ledger
    @test [typeof(e) for e in free.events] == [Fill]
    @test book_as_known(free, 1).cash == -51000
end

@testset "run_backtest: a leg that cannot honestly be priced is a named failure" begin
    f = _en_fixture()
    p = _OpenOnceAt(f.ts2, Order(:call, [Leg(f.call, Long, 1, Open)]))
    # served, but no spot at the fill instant
    thin_spots = MarketData(entry(f.data, OptionQuote), InMemory([SpotPrice(_EN_UND, f.spot, f.ts1)]))
    @test_throws UnpriceableLeg run_backtest(p, thin_spots, f.ts1, f.ts3, _EN_CLOCK).ledger
    err = try run_backtest(p, thin_spots, f.ts1, f.ts3, _EN_CLOCK).ledger; nothing catch e; e end
    @test err isa UnpriceableLeg && err.reason == :no_spot && err.contract == f.call && err.t == f.ts2
    @test occursin("UnpriceableLeg", sprint(showerror, err)) && occursin("no_spot", sprint(showerror, err))
    # nothing serves SpotPrice for SPY at all: structural, so it is named by the data layer
    no_spots = MarketData(entry(f.data, OptionQuote), InMemory(SpotPrice[]))
    @test_throws UnservedSelector run_backtest(p, no_spots, f.ts1, f.ts3, _EN_CLOCK).ledger
    # a strike not in the chain, and a masked instant, are :no_quote
    cut = TimeCut(f.data, f.ts1)
    bogus = ContractKey(_EN_UND, 999.0, f.expiry, Call)
    err = try resolve_quote(cut, bogus, f.ts1); nothing catch e; e end
    @test err isa UnpriceableLeg && err.reason == :no_quote && err.contract == bogus
    err = try resolve_quote(cut, f.call, f.ts2); nothing catch e; e end   # masked by the cut
    @test err isa UnpriceableLeg && err.reason == :no_quote && err.t == f.ts2
    @test_throws UnpriceableLeg run_backtest(_OpenOnceAt(f.ts2, Order(:bogus, [Leg(bogus, Long, 1, Open)])),
                                             f.data, f.ts1, f.ts3, _EN_CLOCK).ledger
    # a missing ask on a Long leg
    one_sided = OptionQuote("X", _EN_UND, f.expiry, 480.0, Call, 5.00, missing, missing,
                            missing, missing, missing, f.ts1)
    data = MarketData(InMemory([one_sided]), entry(f.data, SpotPrice))
    order = Order(:call, [Leg(f.call, Long, 1, Open)])
    err = try fill_legs(TimeCut(data, f.ts1), order, f.ts1; fill_rule=:cross_spread,
                        cost_model=:none, tick_cents=1); nothing catch e; e end
    @test err isa UnpriceableLeg && err.reason == :no_executable_side
    # the bid is there, so a Short leg prices
    short = fill_legs(TimeCut(data, f.ts1), Order(:sell, [Leg(f.call, Short, 1, Open)]), f.ts1;
                      fill_rule=:cross_spread, cost_model=:none, tick_cents=1)
    @test short.prices == [5.00] && ismissing(short.observations[1].ask)
end

@testset "resolve_quote: returns the matching contract" begin
    f = _en_fixture()
    cut = TimeCut(f.data, f.ts1)
    q = resolve_quote(cut, f.put, f.ts1)
    @test q.strike == 480.0
    @test q.option_type == Put
    @test q.bid == 4.80
    @test q.ask == 4.90
    @test q.timestamp == f.ts1
end

@testset "run_backtest: atomicity through the engine" begin
    f = _en_fixture()
    bogus = ContractKey(_EN_UND, 999.0, f.expiry, Call)
    # the second leg has no quote: nothing is written, every counter is at 1
    p = _OpenOnceAt(f.ts2, Order(:half, [Leg(f.call, Long, 1, Open), Leg(bogus, Long, 1, Open)]))
    err = try run_backtest(p, f.data, f.ts1, f.ts3, _EN_CLOCK).ledger; nothing catch e; e end
    @test err isa UnpriceableLeg && err.contract == bogus
    # the failure is before anything is written, so run a variant that records the ledger:
    # an agent whose policy fails on the second tick, after a first order landed
    struct_check = _OpenOnceAt(f.ts2, Order(:half, [Leg(f.call, Long, 1, Open), Leg(f.put, Long, 1, Close)]))
    err = try run_backtest(struct_check, f.data, f.ts1, f.ts3, _EN_CLOCK).ledger; nothing catch e; e end
    @test err isa NothingToClose
    # the same two orders through record_order! directly, on the ledger the engine would hold
    L = Ledger(); book = L.book
    cut = TimeCut(f.data, f.ts2)
    @test_throws UnpriceableLeg fill_legs(cut, p.order, f.ts2; fill_rule=:cross_spread,
                                          cost_model=:ibkr_pro_us_options, tick_cents=1)
    legs = fill_legs(cut, struct_check.order, f.ts2; fill_rule=:cross_spread,
                     cost_model=:ibkr_pro_us_options, tick_cents=1)
    @test_throws NothingToClose record_order!(L, struct_check.order; legs...,
                                              effective_at=f.ts2, recorded_at=f.ts2)
    @test isempty(L) && isempty(L.orders) && book == Book()
    @test (L.next_id, L.next_sequence, L.next_group, L.next_execution, L.next_order_id, L.next_leg_id) ==
          (1, 1, 1, 1, 1, 1)
end

# ---- check_join on hand-built ledgers ---------------------------------

# A ledger with one recorded order and its fills, built the way the engine
# builds it, then a variant with one field changed on the record or a fill.
function _en_hand_ledger(; group=1, decided_at=_LG_T_OPEN, obs=nothing)
    L = Ledger(); book = L.book
    order = Order(:strangle, [Leg(_LG_PUT470, Short, 1, Open), Leg(_LG_CALL490, Short, 1, Open)])
    record_order!(L, order; prices=[0.85, 1.10],
                  observations=[_lg_seen(0.85), _lg_seen(1.10)],
                  effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN, fill_rule=:cross_spread)
    return L
end

# Replace the one order record of `L` with a copy that has `field` changed.
function _en_with_record(L::Ledger, field::Symbol, value)
    r = only(L.orders)
    vals = Dict(f => getfield(r, f) for f in fieldnames(OrderRecord))
    vals[field] = value
    L.orders[1] = OrderRecord((vals[f] for f in fieldnames(OrderRecord))...)
    return L
end

# Replace fill `i` of `L` with a copy that has `field` changed (bypassing
# the write path: this is what a corrupt store would hand back).
function _en_with_fill(L::Ledger, i::Int, field::Symbol, value)
    f = L.events[i]::Fill
    vals = Dict(n => getfield(f, n) for n in fieldnames(Fill))
    vals[field] = value
    L.events[i] = Fill(vals[:header], vals[:group], vals[:order_leg_id], vals[:execution_id],
                       vals[:contract], vals[:side], vals[:intent], vals[:quantity],
                       vals[:price], vals[:fill_rule])
    return L
end

_en_join_error(L; kw...) = try check_join(L; kw...); nothing catch e; e end
_en_record_join_error(L, r=only(L.orders); kw...) = try check_join(L, r; kw...); nothing catch e; e end

@testset "check_join(L, rec): checks the appended record and only its fills" begin
    for make_case in (_lg_case_strangle_order, _lg_case_strangle_closed)
        L, _ = make_case()
        @test all(check_join(L, r) === nothing for r in L.orders)
    end
    f = _en_fixture()
    engine_ledger = run_backtest(_OpenThenClose(f.ts1, f.ts3, f.call),
                                 f.data, f.ts1, f.ts3, _EN_CLOCK).ledger
    @test all(check_join(engine_ledger, r) === nothing for r in engine_ledger.orders)
    @test occursin("per-record", string(@doc run_backtest))

    # Two orders from one decision boundary: each per-record scan selects by
    # its leg-id interval even though both have known_to == 0.
    L = Ledger(); book = L.book
    for (label, contract, price) in ((:put, _LG_PUT470, 0.85),
                                     (:call, _LG_CALL490, 1.10))
        record_order!(L, Order(label, [Leg(contract, Short, 1, Open)]);
                      prices=[price], observations=[_lg_seen(price)], known_to=0,
                      effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN,
                      fill_rule=:cross_spread)
    end
    r1, r2 = L.orders
    L.orders[1] = OrderRecord(r1.order_id, r1.first_leg_id, r1.group, r1.decided_at,
        r1.known_to, r1.order, [_lg_seen(0.86)])
    err = _en_record_join_error(L, L.orders[1])
    @test err isa JoinViolation && err.field == :price && err.id == 1
    @test occursin("JoinViolation", sprint(showerror, err))
    @test check_join(L, r2) === nothing
end

@testset "check_join(L, rec): names every per-record disagreement and is read-only" begin
    cases = [
        :contract => (_en_with_fill(_en_hand_ledger(), 1, :contract, _LG_PUT465B), :contract),
        :side => (_en_with_fill(_en_hand_ledger(), 1, :side, Long), :side),
        :intent => (_en_with_fill(_en_hand_ledger(), 1, :intent, Close), :intent),
        :group => (_en_with_fill(_en_hand_ledger(), 1, :group, 2), :group),
        :quantity => (_en_with_fill(_en_hand_ledger(), 1, :quantity, 2), :quantity),
        :quote_at => (_en_with_record(_en_hand_ledger(), :decided_at,
                                      _LG_T_OPEN - Minute(1)), :quote_at),
        :spot_at => (_en_with_record(_en_hand_ledger(), :observations,
            [LegObservation(_LG_T_OPEN, 0.85, 0.85, 480.0, _LG_T_OPEN + Minute(1)),
             _lg_seen(1.10)]), :spot_at),
        :fill_rule => (_en_with_fill(_en_hand_ledger(), 1, :fill_rule, :mid), :fill_rule),
        :bid => (_en_with_record(_en_hand_ledger(), :observations,
            [LegObservation(_LG_T_OPEN, missing, 0.85, 480.0, _LG_T_OPEN),
             _lg_seen(1.10)]), :bid),
        :price => (_en_with_record(_en_hand_ledger(), :observations,
                                   [_lg_seen(0.86), _lg_seen(1.10)]), :price),
    ]
    # A Long fill exercises the other executable side (:ask).
    L = Ledger(); book = L.book
    record_order!(L, Order(:buy, [Leg(_LG_PUT470, Long, 1, Open)]);
                  prices=[0.85], observations=[_lg_seen(0.85)],
                  effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN,
                  fill_rule=:cross_spread)
    push!(cases, :ask => (_en_with_record(L, :observations,
        [LegObservation(_LG_T_OPEN, 0.85, missing, 480.0, _LG_T_OPEN)]), :ask))
    push!(cases, :observations => (_en_with_record(_en_hand_ledger(), :observations,
                                                   LegObservation[]), :observations))

    for (name, (bad, field)) in cases
        before_events = copy(bad.events)
        before_orders = copy(bad.orders)
        before_counters = (bad.next_id, bad.next_sequence, bad.next_group,
                           bad.next_execution, bad.next_order_id, bad.next_leg_id)
        before_book = book_as_known(bad, last_sequence(bad))
        err = _en_record_join_error(bad)
        @test err isa JoinViolation && err.field == field
        @test occursin("JoinViolation", sprint(showerror, err))
        @test length(bad.events) == length(before_events) &&
              all(a === b for (a, b) in zip(bad.events, before_events))
        @test length(bad.orders) == length(before_orders) &&
              all(a === b for (a, b) in zip(bad.orders, before_orders))
        @test (bad.next_id, bad.next_sequence, bad.next_group, bad.next_execution,
               bad.next_order_id, bad.next_leg_id) == before_counters
        @test book_as_known(bad, last_sequence(bad)) == before_book
        @test name == field
    end

    broker = Ledger()
    rec = record_order!(broker,
        Order(:buy, [Leg(_LG_PUT470, Long, 1, Open)]); prices=[0.85],
        observations=[LegObservation(_LG_T_OPEN, 0.80, missing, 480.0, _LG_T_OPEN)],
        effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN, fill_rule=:broker_execution)
    @test check_join(broker, rec) === nothing
end

@testset "check_join: passes on ledgers the writer built, and names every disagreement" begin
    f = _en_fixture()
    @test check_join(run_backtest(_OpenThenClose(f.ts1, f.ts3, f.call), f.data, f.ts1, f.ts3, _EN_CLOCK).ledger) === nothing
    @test check_join(_en_hand_ledger()) === nothing
    @test check_join(Ledger()) === nothing
    L, _ = _lg_case_strangle_closed()
    @test check_join(L) === nothing
    # a fill whose leg record is missing
    L = _en_hand_ledger(); empty!(L.orders)
    err = _en_join_error(L)
    @test err isa DanglingReference && err.field == :order_leg_id && err.id == 1
    # a leg whose contract, side, intent or group differs from its fill
    for (field, value) in ((:contract, _LG_PUT465B), (:side, Long), (:intent, Close), (:group, 2))
        L = _en_with_fill(_en_hand_ledger(), 1, field, value)
        err = _en_join_error(L)
        @test err isa JoinViolation && err.field == field && err.id == 1
        @test occursin("JoinViolation", sprint(showerror, err))
    end
    # two fills of 1 on a leg of 1: the second fill over-fills the leg
    L = _en_hand_ledger()
    extra = Fill(EventHeader(L.next_id, _LG_T_OPEN, _LG_T_OPEN, L.next_sequence), 1, 1, 9,
                 _LG_PUT470, Short, Open, 1, 0.85, :cross_spread)
    push!(L.events, extra); L.index[event_id(extra)] = length(L.events)
    err = _en_join_error(L)
    @test err isa JoinViolation && err.field == :quantity && err.id == event_id(extra)
    # a :cross_spread fill whose observation is after the decision
    L = _en_with_record(_en_hand_ledger(), :decided_at, _LG_T_OPEN - Minute(1))
    err = _en_join_error(L)
    @test err isa JoinViolation && err.field == :quote_at
    late_spot = LegObservation(_LG_T_OPEN, 0.85, 0.85, 480.0, _LG_T_OPEN + Minute(1))
    L = _en_with_record(_en_hand_ledger(), :observations, [late_spot, _lg_seen(1.10)])
    err = _en_join_error(L)
    @test err isa JoinViolation && err.field == :spot_at
    # the required side missing: a Short leg needs the bid
    no_bid = LegObservation(_LG_T_OPEN, missing, 0.85, 480.0, _LG_T_OPEN)
    L = _en_with_record(_en_hand_ledger(), :observations, [no_bid, _lg_seen(1.10)])
    err = _en_join_error(L)
    @test err isa JoinViolation && err.field == :bid && err.id == 1
    # a price the rule does not produce: 0.85 against a bid of 0.86 on a Short leg
    L = _en_with_record(_en_hand_ledger(), :observations, [_lg_seen(0.86), _lg_seen(1.10)])
    err = _en_join_error(L)
    @test err isa JoinViolation && err.field == :price && err.id == 1
    # a Long leg filled at 0.85 against an ask of 0.86
    L = Ledger(); book = L.book
    record_order!(L, Order(:buy, [Leg(_LG_PUT470, Long, 1, Open)]); prices=[0.85],
                  observations=[_lg_seen(0.85)], effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN,
                  fill_rule=:cross_spread)
    L = _en_with_record(L, :observations, [LegObservation(_LG_T_OPEN, 0.84, 0.86, 480.0, _LG_T_OPEN)])
    err = _en_join_error(L)
    @test err isa JoinViolation && err.field == :price
    @test occursin("0.86", sprint(showerror, err))
    # the tick is part of the rule: a sale at 0.85 against a bid of 0.89 is what a
    # 5-cent tick produces (floor(89 / 5) * 5 = 85), not what a 1-cent tick does
    L = _en_with_record(_en_hand_ledger(), :observations, [_lg_seen(0.89), _lg_seen(1.10)])
    @test check_join(L; tick_cents=5) === nothing
    err = _en_join_error(L; tick_cents=1)
    @test err isa JoinViolation && err.field == :price
    # and a bid off the tick that floors to the fill passes on the default tick
    L = _en_with_record(_en_hand_ledger(), :observations,
                        [LegObservation(_LG_T_OPEN, 0.857, 0.857, 480.0, _LG_T_OPEN), _lg_seen(1.10)])
    @test check_join(L) === nothing
    # the default is TICK_CENTS, not a repeated literal: a Long leg filled
    # at 0.86 against an ask of 0.857 is ceil(85.7) on the penny tick
    long = Ledger()
    record_order!(long, Order(:buy, [Leg(_LG_PUT470, Long, 1, Open)]); prices=[0.86],
                  observations=[LegObservation(_LG_T_OPEN, 0.837, 0.857, 480.0, _LG_T_OPEN)],
                  effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN, fill_rule=:cross_spread)
    @test check_join(long) === nothing
    @test check_join(long; tick_cents=TICK_CENTS) === nothing
    # an unknown fill rule is a violation on :fill_rule
    L = _en_with_fill(_en_hand_ledger(), 1, :fill_rule, :mid)
    err = _en_join_error(L)
    @test err isa JoinViolation && err.field == :fill_rule && err.id == 1
    # order records whose ids or first_leg_ids are not contiguous
    L = _en_with_record(_en_hand_ledger(), :order_id, 2)
    err = _en_join_error(L)
    @test err isa JoinViolation && err.field == :order_id
    L = _en_with_record(_en_hand_ledger(), :first_leg_id, 2)
    err = _en_join_error(L)
    @test err isa JoinViolation && err.field == :first_leg_id
    L, _ = _lg_case_strangle_closed()
    r2 = L.orders[2]
    L.orders[2] = OrderRecord(r2.order_id, 4, r2.group, r2.decided_at, r2.known_to, r2.order, r2.observations)
    err = _en_join_error(L)
    @test err isa JoinViolation && err.field == :first_leg_id && err.id == 2
    # an observation count that differs from the leg count
    L = _en_with_record(_en_hand_ledger(), :observations, [_lg_seen(0.85)])
    err = _en_join_error(L)
    @test err isa JoinViolation && err.field == :observations
    # a :broker_execution fill with a missing ask passes: the observation is not consulted
    L = Ledger(); book = L.book
    record_order!(L, Order(:buy, [Leg(_LG_PUT470, Long, 1, Open)]); prices=[0.85],
                  observations=[LegObservation(_LG_T_OPEN, 0.80, missing, 480.0, _LG_T_OPEN)],
                  effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN, fill_rule=:broker_execution)
    @test check_join(L) === nothing
    # the same observation under :cross_spread is a violation on the missing side
    L = _en_with_fill(L, 1, :fill_rule, :cross_spread)
    err = _en_join_error(L)
    @test err isa JoinViolation && err.field == :ask
    # the failures print their names
    @test occursin("JoinViolation", sprint(showerror, JoinViolation(:price, 3, "why")))
    @test occursin("UnpriceableLeg", sprint(showerror, UnpriceableLeg(_LG_PUT470, _LG_T_OPEN, :no_quote)))
    for err in (JoinViolation(:price, 3, "why"), UnpriceableLeg(_LG_PUT470, _LG_T_OPEN, :no_quote))
        @test err isa Exception
    end
end

# ---------- a scheduled decision fills off the minute that just ended ----------
# Adjacent minutes with different prices, served through the real parquet
# readers so the stamps come from the row mapping rather than from the
# fixture. At a decision at 15:31 the fill must use the 15:30-15:31 bar --
# the one that has finished -- and not the 15:31-15:32 bar, which is the
# minute of up-to-one-minute lookahead the bar-open stamp handed out.

mktempdir() do root
    opts = joinpath(root, "options_1min")
    spots = joinpath(root, "spots_1min")
    tick = DateTime(2024, 1, 15, 15, 31)                # the decision instant
    just_ended = _md_row(tick)                           # 15:30, the completed minute
    not_yet = tick                                      # 15:31, still running
    tkr = "O:SPY240216C00480000"
    _md_write_options_parquet(
        joinpath(opts, "date=2024-01-15", "symbol=SPY", "data.parquet"),
        [(ticker=tkr, close=5.00, volume=1.0, open=5.00, high=5.00, low=5.00,
          timestamp=_md_row(just_ended)),                # visible 15:30
         (ticker=tkr, close=5.00, volume=1.0, open=5.00, high=5.00, low=5.00,
          timestamp=just_ended),                         # visible 15:31 -- the fill
         (ticker=tkr, close=9.00, volume=1.0, open=9.00, high=9.00, low=9.00,
          timestamp=not_yet)])                           # visible 15:32 -- lookahead
    _md_write_spot_parquet(joinpath(spots, "date=2024-01-15", "symbol=SPY", "data.parquet"),
                           _md_row.([DateTime(2024, 1, 15, 15, 30), tick,
                                     DateTime(2024, 1, 15, 15, 32)]),
                           [480.0, 480.0, 495.0])

    @testset "run_backtest: a fill reads the minute that has ended, not the one running" begin
        data = MarketData(ParquetOptionBars(opts), QuotesFromBars(SpreadFromOHLCV(1.0)),
                          ParquetSpots(spots))
        contract = ContractKey(_EN_UND, 480.0, DateTime(2024, 2, 16, 21, 0), Call)
        policy = _OpenOnceAt(tick, Order(:buy, [Leg(contract, Long, 1, Open)]))
        L = with_data(data) do d
            run_backtest(policy, d, tick, tick, _EN_CLOCK).ledger
        end
        f = only(e for e in L.events if e isa Fill)
        @test f.price == 5.00                  # the 15:30-15:31 close, λ = 1 so bid = ask
        @test effective_at(f) == tick
        # what the decision saw, recorded in the order journal
        o = only(L.orders)
        @test o.decided_at == tick
        obs = only(o.observations)
        @test obs.quote_at == tick && obs.spot_at == tick
        @test obs.bid == 5.00 && obs.ask == 5.00 && obs.spot == 480.0
    end
end
