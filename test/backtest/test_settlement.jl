# Tests for the lifecycle step: `settlement_price`, `settlements`, and the
# tick loop booking `Expiry` through the ledger's own writer. The dates are
# real NYSE ones, so the calendar check is exercised the way the ten-year
# run exercises it: 2024-12-24 closed early at 13:00 ET, 2025-01-09 was an
# unscheduled closure (Jimmy Carter's national day of mourning), 2024-01-18
# was an ordinary open Thursday. Cash literals are worked out by hand in
# whole USD cents: one contract at a per-share price p moves p * 100 * 100.

const _ST_SPY   = Underlying("SPY")
const _ST_CLOCK = Clock{OptionQuote}(_ST_SPY)

_st_et(d::Date, h::Int, m::Int) = et_to_utc(d, Time(h, m))
_st_spot(d::Date, h::Int, m::Int, price) = SpotPrice(_ST_SPY, price, _st_et(d, h, m))
_st_call(d::Date, strike) = ContractKey(_ST_SPY, strike, _st_et(d, 16, 0), Call)

# A regular session: prints at 09:30, 10:00 and the 16:00 close.
_st_session(d::Date, close_price) =
    [_st_spot(d, 9, 30, close_price - 5.0), _st_spot(d, 10, 0, close_price - 2.0),
     _st_spot(d, 16, 0, close_price)]

# One quote for `contract`, stamped 10:00 ET of `d`; the clock ticks on these.
_st_quote(d::Date, contract::ContractKey, bid, ask) =
    OptionQuote("X", _ST_SPY, contract.expiry, contract.strike, contract.option_type,
                bid, ask, (bid + ask) / 2, missing, missing, missing, _st_et(d, 10, 0))

const _ST_D16 = Date(2024, 1, 16)       # Tue
const _ST_D17 = Date(2024, 1, 17)
const _ST_D18 = Date(2024, 1, 18)
const _ST_D19 = Date(2024, 1, 19)       # Fri: contract A expires 16:00 ET
const _ST_D22 = Date(2024, 1, 22)       # the Monday after
const _ST_D26 = Date(2024, 1, 26)       # contract B's expiry, past every window

# Five sessions of SPY prints and quotes on two calls: A expires on the
# Friday and settles against that session's 480 close, B stays open.
function _st_fixture()
    a = _st_call(_ST_D19, 470.0)
    b = _st_call(_ST_D26, 480.0)
    days = (_ST_D16, _ST_D17, _ST_D18, _ST_D19, _ST_D22)
    spots = vcat(_st_session(_ST_D16, 471.0), _st_session(_ST_D17, 472.0),
                 _st_session(_ST_D18, 473.0), _st_session(_ST_D19, 480.0),
                 _st_session(_ST_D22, 485.0))
    quotes = OptionQuote[]
    for d in days
        push!(quotes, _st_quote(d, a, 5.00, 5.10))
        push!(quotes, _st_quote(d, b, 3.00, 3.10))
    end
    (data = MarketData(InMemory(quotes), InMemory(spots)), a = a, b = b,
     ticks = [_st_et(d, 10, 0) for d in days])
end

# A policy that opens one long call per named instant, and nothing else.
struct _ST_OpenAt <: Policy
    plan::Vector{Pair{DateTime,ContractKey}}
end

VolSurfaceAnalysis.decide(p::_ST_OpenAt, t::DateTime, ::TimeCut, ::Book)::Vector{Order} =
    Order[Order(:leg, [Leg(c, Long, 1, Open)]) for (when, c) in p.plan if when == t]

# A policy that records the book it is handed on every tick, and otherwise
# opens what its inner policy opens.
mutable struct _ST_Recording <: Policy
    inner::_ST_OpenAt
    seen::Vector{Pair{DateTime,Book}}
end
_ST_Recording(inner) = _ST_Recording(inner, Pair{DateTime,Book}[])

function VolSurfaceAnalysis.decide(p::_ST_Recording, t::DateTime, cut::TimeCut, book::Book)
    push!(p.seen, t => deepcopy(book))
    return decide(p.inner, t, cut, book)
end

# ---- the rule --------------------------------------------------------

@testset "settlement_price: the session close is the last regular-hours print" begin
    f = _st_fixture()
    cut = TimeCut(f.data, _st_et(_ST_D22, 10, 0))
    @test settlement_price(:session_close, cut, f.a, _st_et(_ST_D22, 10, 0)) == 480.0
    # Reading nothing but the cut: a cut before the close sees the earlier print.
    early = TimeCut(f.data, _st_et(_ST_D19, 10, 0))
    @test settlement_price(:session_close, early, f.a, _st_et(_ST_D19, 10, 0)) == 478.0
end

@testset "settlement_price: an unknown rule errors naming the known ones" begin
    f = _st_fixture()
    cut = TimeCut(f.data, _st_et(_ST_D22, 10, 0))
    @test_throws r"unknown settlement rule :nope" settlement_price(
        :nope, cut, f.a, _st_et(_ST_D22, 10, 0))
end

@testset "settlement_price: an early close settles at the 13:00 ET print" begin
    # 2024-12-24 is a scheduled 13:00 ET close: the last print of the
    # window IS the official close, so no early-close table is needed.
    d, prev = Date(2024, 12, 24), Date(2024, 12, 23)
    spots = vcat(_st_session(prev, 600.0),
                 [_st_spot(d, 9, 30, 604.0), _st_spot(d, 12, 0, 605.0),
                  _st_spot(d, 13, 0, 606.0)])
    data = MarketData(InMemory(spots))
    c = _st_call(d, 600.0)
    t = _st_et(d, 16, 0)
    @test settlement_price(:session_close, TimeCut(data, t), c, t) == 606.0
end

@testset "settlement_price: an unscheduled closure settles at the previous close" begin
    # 2025-01-09, Jimmy Carter's national day of mourning: no prints, and
    # the calendar calls it closed, so the walk steps back one session.
    d, prev = Date(2025, 1, 9), Date(2025, 1, 8)
    spots = vcat(_st_session(Date(2025, 1, 7), 590.0), _st_session(prev, 595.0))
    data = MarketData(InMemory(spots))
    c = _st_call(d, 580.0)
    t = _st_et(d, 16, 0)
    @test settlement_price(:session_close, TimeCut(data, t), c, t) == 595.0
end

@testset "settlement_price: the calendar carries the ad-hoc closures itself" begin
    # The const set is empty because BusinessDays 0.9.25 already lists both
    # days of mourning and the Hurricane Sandy closure. If a calendar
    # upgrade ever drops one, this is what goes red first.
    @test isempty(VolSurfaceAnalysis._ADHOC_CLOSURES)
    for d in (Date(2018, 12, 5), Date(2025, 1, 9), Date(2012, 10, 29), Date(2012, 10, 30))
        @test VolSurfaceAnalysis._closed_on(d)
    end
    @test !VolSurfaceAnalysis._closed_on(Date(2024, 1, 18))   # an ordinary Thursday
end

# ---- the gap: a printless weekday the calendar calls open --------------

# Contract A expires Thu 2024-01-18, on which SPY has quotes (so the clock
# still ticks) but no spot at all. The calendar calls that Thursday open,
# so it is a data gap and never a closure.
function _st_gap_fixture()
    a = _st_call(_ST_D18, 470.0)
    days = (_ST_D16, _ST_D17, _ST_D18, _ST_D19, _ST_D22)
    spots = vcat(_st_session(_ST_D16, 471.0), _st_session(_ST_D17, 472.0),
                 _st_session(_ST_D19, 480.0), _st_session(_ST_D22, 485.0))
    quotes = OptionQuote[_st_quote(d, a, 5.00, 5.10) for d in days]
    (data = MarketData(InMemory(quotes), InMemory(spots)), a = a,
     ticks = [_st_et(d, 10, 0) for d in days])
end

# A one-lot book holding `contract`, opened at `at` outside the engine.
function _st_one_lot_book(contract::ContractKey, at::DateTime)
    L = Ledger()
    g = mint_group!(L)
    record_fill!(L, Leg(contract, Long, 1, Open), g; price = 5.10,
                 effective_at = at, recorded_at = at, order_leg_id = 1,
                 fill_rule = :cross_spread)
    return L.book
end

@testset "settlement_price: a printless open weekday is :unexpected_gap" begin
    f = _st_gap_fixture()
    t = _st_et(_ST_D19, 10, 0)
    e = try
        settlement_price(:session_close, TimeCut(f.data, t), f.a, t)
    catch err
        err
    end
    @test e isa UnpriceableLeg
    @test e.contract == f.a && e.t == t && e.reason == :unexpected_gap
end

@testset "settlements: an unpriceable lot is warned about and left unsettled" begin
    f = _st_gap_fixture()
    t = _st_et(_ST_D19, 10, 0)
    book = _st_one_lot_book(f.a, _st_et(_ST_D16, 10, 0))
    out = @test_logs (:warn, "lot left open: no honest settlement price") settlements(
        TimeCut(f.data, t), book, _st_et(_ST_D18, 10, 0), t; settlement_rule = :session_close)
    @test isempty(out.settled)
    @test length(out.unsettled) == 1
    @test only(out.unsettled).reason == :unexpected_gap
    @test only(out.unsettled).contract == f.a
end

@testset "run_backtest: an unsettleable lot stays open, warned about once" begin
    f = _st_gap_fixture()
    p = _ST_OpenAt([f.ticks[1] => f.a])
    L = @test_logs (:warn, "lot left open: no honest settlement price") run_backtest(
        p, f.data, f.ticks[1], _st_et(_ST_D22, 20, 0), _ST_CLOCK)
    # Three ticks and a window-end pass follow the expiry; the warning fired
    # once, because the interval examines a lot exactly once, ever (D5).
    @test !any(e isa Expiry for e in L.events)
    @test [l.contract for l in open_lots(L.book)] == [f.a]
    @test L.book == book_effective(L, _st_et(_ST_D22, 20, 0))
end

# ---- the loop --------------------------------------------------------

@testset "run_backtest: an expiry is booked at the tick after its instant" begin
    f = _st_fixture()
    to = _st_et(_ST_D22, 10, 0)
    p = _ST_OpenAt([f.ticks[1] => f.a])
    L = run_backtest(p, f.data, f.ticks[1], to, _ST_CLOCK)
    @test [typeof(e) for e in L.events] == [Fill, Fee, Expiry]
    x = only(e for e in L.events if e isa Expiry)
    @test x.contract == f.a && x.quantity == 1 && x.outcome == CashSettled
    @test x.settlement_price == 480.0
    @test effective_at(x) == f.a.expiry          # D1: the obligation settles when it expires
    @test recorded_at(x) == to                   # ... and is learned at the tick that booked it
    # 480 - 470 = 10.00 intrinsic, paid 5.10, one lone contract raised to
    # the USD 1.00 minimum: 100000 - 51000 - 100.
    @test L.book.cash == 100000 - 51000 - 100
    @test isempty(open_lots(L.book))
    trips = round_trips(L)
    @test length(trips) == 1
    @test trips[1].kind == :expired && trips[1].pnl == 48900
    @test trips[1].closed_at == f.a.expiry
end

@testset "run_backtest: the window end settles what the last tick did not" begin
    f = _st_fixture()
    # The last policy tick is Friday 10:00 ET; the contract expires at 16:00
    # ET that day, inside the window but after every tick.
    to = _st_et(_ST_D19, 20, 0)
    p = _ST_OpenAt([f.ticks[1] => f.a])
    L = run_backtest(p, f.data, f.ticks[1], to, _ST_CLOCK)
    x = only(e for e in L.events if e isa Expiry)
    @test effective_at(x) == f.a.expiry
    @test recorded_at(x) == to
    @test x.settlement_price == 480.0
    @test isempty(open_lots(L.book))
end

@testset "run_backtest: lifecycle precedes the decision" begin
    f = _st_fixture()
    to = _st_et(_ST_D22, 10, 0)
    p = _ST_Recording(_ST_OpenAt([f.ticks[1] => f.a, f.ticks[5] => f.b]))
    L = run_backtest(p, f.data, f.ticks[1], to, _ST_CLOCK)
    seen = Dict(p.seen)
    @test length(p.seen) == 5
    # The expiry is effective Friday 16:00 ET and booked at the Monday tick;
    # the policy deciding at that tick already sees the lot gone.
    @test [l.contract for l in open_lots(seen[f.ticks[4]])] == [f.a]
    @test isempty(open_lots(seen[f.ticks[5]]))
    x = only(e for e in L.events if e isa Expiry)
    rec = only(r for r in L.orders if r.decided_at == f.ticks[5])
    @test rec.known_to >= sequence(x)            # the decision saw the expiry
    @test seen[f.ticks[5]] == book_as_known(L, rec.known_to)
end

@testset "run_backtest: both replays agree at the end and differ in between" begin
    f = _st_fixture()
    to = _st_et(_ST_D22, 10, 0)
    p = _ST_OpenAt([f.ticks[1] => f.a, f.ticks[5] => f.b])
    L = run_backtest(p, f.data, f.ticks[1], to, _ST_CLOCK)
    @test book_as_known(L, last_sequence(L)) == book_effective(L, to)
    # Saturday: the expiry is already true, and not yet known.
    x = only(e for e in L.events if e isa Expiry)
    saturday = _st_et(Date(2024, 1, 20), 12, 0)
    @test isempty(open_lots(book_effective(L, saturday)))
    @test [l.contract for l in open_lots(book_as_known(L, sequence(x) - 1))] == [f.a]
    @test book_effective(L, saturday) != book_as_known(L, sequence(x) - 1)
end

# A policy that opens two contracts at one instant, closes one at another,
# and leaves the rest to expire.
struct _ST_OpenThenHalfClose <: Policy
    open_at::DateTime
    close_at::DateTime
    contract::ContractKey
end

function VolSurfaceAnalysis.decide(p::_ST_OpenThenHalfClose, t::DateTime, ::TimeCut, book::Book)
    t == p.open_at &&
        return Order[Order(:open, [Leg(p.contract, Long, 2, Open)])]
    t == p.close_at &&
        return Order[Order(:half, [Leg(p.contract, Short, 1, Close)]; group = only(open_groups(book)))]
    return Order[]
end

@testset "run_backtest: a half-closed lot expires whole, once" begin
    f = _st_fixture()
    to = _st_et(_ST_D22, 10, 0)
    L = run_backtest(_ST_OpenThenHalfClose(f.ticks[1], f.ticks[2], f.a), f.data,
                     f.ticks[1], to, _ST_CLOCK)
    @test [typeof(e) for e in L.events] == [Fill, Fee, Fill, Match, Fee, Expiry]
    x = only(e for e in L.events if e isa Expiry)
    @test x.quantity == 1                         # the remainder, whole, and only once
    @test isempty(open_lots(L.book))
    # bought 2 at 5.10 (-102000, fee 130), sold 1 at 5.00 (+50000, fee 100),
    # the last contract settles at 10.00 intrinsic (+100000).
    @test L.book.cash == -102000 - 130 + 50000 - 100 + 100000
    trips = round_trips(L)
    @test [t.kind for t in trips] == [:closed, :expired]
    @test [t.quantity for t in trips] == [1, 1]
end

@testset "settlements: mutates nothing and is examined once" begin
    f = _st_fixture()
    to = _st_et(_ST_D22, 10, 0)
    L = run_backtest(_ST_OpenAt([f.ticks[1] => f.b]), f.data, f.ticks[1], to, _ST_CLOCK)
    book = L.book
    before = deepcopy(book)
    cut = TimeCut(f.data, to)
    # B expires after the window: nothing is due, and nothing changes.
    one = settlements(cut, book, f.ticks[1], to; settlement_rule = :session_close)
    two = settlements(cut, book, f.ticks[1], to; settlement_rule = :session_close)
    @test isempty(one.settled) && isempty(one.unsettled)
    @test one == two
    @test book == before
    # A lot whose expiry is at or before the interval's lower bound is not
    # returned at all: the bound is open below (D5).
    g = _st_fixture()
    held = _st_one_lot_book(g.a, _st_et(_ST_D16, 10, 0))
    due = settlements(TimeCut(g.data, to), held, _st_et(_ST_D18, 10, 0), to;
                      settlement_rule = :session_close)
    @test [lot.contract for (lot, _) in due.settled] == [g.a]
    @test only(due.settled)[2] == 480.0
    @test isempty(settlements(TimeCut(g.data, to), held, g.a.expiry, to;
                              settlement_rule = :session_close).settled)
    @test held == _st_one_lot_book(g.a, _st_et(_ST_D16, 10, 0))
end
