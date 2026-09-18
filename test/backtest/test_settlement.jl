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

# One quote for `contract`, stamped at an instant; the clock ticks on these.
_st_quote_at(t::DateTime, contract::ContractKey, bid, ask) =
    OptionQuote("X", _ST_SPY, contract.expiry, contract.strike, contract.option_type,
                bid, ask, (bid + ask) / 2, missing, missing, missing, t)

# The usual one: 10:00 ET of `d`.
_st_quote(d::Date, contract::ContractKey, bid, ask) =
    _st_quote_at(_st_et(d, 10, 0), contract, bid, ask)

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

# A one-lot ledger holding `contract`, opened at `at` outside the engine.
function _st_one_lot_ledger(contract::ContractKey, at::DateTime)
    L = Ledger()
    g = mint_group!(L)
    record_fill!(L, Leg(contract, Long, 1, Open), g; price = 5.10,
                 effective_at = at, recorded_at = at, order_leg_id = 1,
                 fill_rule = :cross_spread)
    return L
end

_st_one_lot_book(contract::ContractKey, at::DateTime) =
    _st_one_lot_ledger(contract, at).book

# The lifecycle step and the ledger's own writer, by hand the way the tick
# loop runs them: the lot of `L` falling due in `(prev, t]` is settled and
# booked, and the booked `Expiry` comes back. Used where the point is the
# bitemporal stamping (D1) rather than the whole engine.
function _st_book_due(L::Ledger, data, prev::DateTime, t::DateTime)
    due = settlements(TimeCut(data, t), L.book, prev, t)
    (lot, price) = only(due.settled)
    return record_expiry!(L, lot; settlement_price = price,
                          effective_at = lot.contract.expiry, recorded_at = t)
end

# ---- the rule --------------------------------------------------------

@testset "settlement_price: the session close is the last regular-hours print" begin
    f = _st_fixture()
    cut = TimeCut(f.data, _st_et(_ST_D22, 10, 0))
    @test settlement_price(:session_close, cut, f.a, _st_et(_ST_D22, 10, 0)) == 480.0
end

@testset "settlement_price: a cut short of the expiry is :no_session_close" begin
    f = _st_fixture()
    # The rule's domain. A cut at Friday 10:00 ET can see the 478 print but
    # not the 480 close of the session that settles contract A, so "what did
    # this settle at" has no honest answer yet: design rule 7 names it rather
    # than blessing a provisional morning quote as a settlement price.
    stale = TimeCut(f.data, _st_et(_ST_D19, 10, 0))
    for (name, t) in (("a stale cut", _st_et(_ST_D19, 20, 0)),      # t past the expiry
                      ("a premature request", _st_et(_ST_D19, 10, 0)))
        @testset "$name" begin
            e = try settlement_price(:session_close, stale, f.a, t) catch err; err end
            @test e isa UnpriceableLeg
            @test e.reason == :no_session_close
            @test e.contract == f.a && e.t == t
        end
    end
    # The bound is inclusive: a cut exactly at the expiry sees the close.
    @test settlement_price(:session_close, TimeCut(f.data, f.a.expiry), f.a,
                           f.a.expiry) == 480.0
    # And it is the cut, not `t`, that the domain is about: a cut that reaches
    # the expiry answers whatever reporting instant is passed.
    @test settlement_price(:session_close, TimeCut(f.data, f.a.expiry), f.a,
                           _st_et(_ST_D22, 10, 0)) == 480.0
end

@testset "settlement_price: an unknown rule errors naming the known ones" begin
    f = _st_fixture()
    cut = TimeCut(f.data, _st_et(_ST_D22, 10, 0))
    @test_throws r"unknown settlement rule :nope" settlement_price(
        :nope, cut, f.a, _st_et(_ST_D22, 10, 0))
end

@testset "settlement_price: an early close settles at the 13:00 ET print" begin
    # 2024-12-24 is a scheduled 13:00 ET close. No early-close table is
    # needed because no print here falls between the close and 16:00 ET,
    # so the last print of the window is the session's last print. The
    # testset below feeds the same rule data that breaks that assumption,
    # and shows what it costs.
    d, prev = Date(2024, 12, 24), Date(2024, 12, 23)
    spots = vcat(_st_session(prev, 600.0),
                 [_st_spot(d, 9, 30, 604.0), _st_spot(d, 12, 0, 605.0),
                  _st_spot(d, 13, 0, 606.0)])
    data = MarketData(InMemory(spots))
    c = _st_call(d, 600.0)
    t = _st_et(d, 16, 0)
    @test settlement_price(:session_close, TimeCut(data, t), c, t) == 606.0
    # D1: the 13:00 print stands in for the close, but the obligation still
    # ceased to exist at the listed 16:00 ET expiry, and is learned later.
    booked = _st_et(d, 17, 0)
    L = _st_one_lot_ledger(c, _st_et(prev, 10, 0))
    x = _st_book_due(L, data, _st_et(d, 10, 0), booked)
    @test x.settlement_price == 606.0
    @test effective_at(x) == c.expiry && recorded_at(x) == booked
end

@testset "settlement_price: an extended-hours print defeats the early close" begin
    # THIS TEST DOES NOT BLESS THE NUMBER IT ASSERTS. It pins what a
    # *violated assumption* produces, which is the whole reason the
    # exposure is written down (the `backtest` module doc and
    # `settlement_price`'s docstring; the `SpotPrice` docstring says the
    # record carries no session). The rule assumes no extended-hours print
    # inside the window. Nothing enforces it -- the parquet spot
    # reader selects every row its partitions hold -- and nothing in a
    # `SpotPrice` records which session a print came from, so the rule
    # cannot detect the violation and no narrower window rescues it: on a
    # 13:00 ET early close a 15:59 extended-hours print is inside the
    # 09:30-16:00 window and is regular-hours-shaped.
    #
    # If a later change makes this assert 606.0 instead, that is the
    # defect being fixed and not a regression; delete the broken-assumption
    # half rather than preserving 610.0.
    d, prev = Date(2024, 12, 24), Date(2024, 12, 23)
    session = [_st_spot(d, 9, 30, 604.0), _st_spot(d, 13, 0, 606.0)]
    c = _st_call(d, 600.0)
    t = _st_et(d, 16, 0)
    # Assumption holds: the 13:00 close settles the contract.
    kept = MarketData(InMemory(vcat(_st_session(prev, 600.0), session)))
    @test settlement_price(:session_close, TimeCut(kept, t), c, t) == 606.0
    # Assumption broken: one post-close print wins instead. Four dollars a
    # share of intrinsic, 40_000 cents per contract, with no warning --
    # which is why the exposure is written down.
    broken = MarketData(InMemory(vcat(_st_session(prev, 600.0), session,
                                      [_st_spot(d, 15, 59, 610.0)])))
    @test settlement_price(:session_close, TimeCut(broken, t), c, t) == 610.0
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
    # D1 made checkable: only *which* print stands in for the close moved
    # back a session; `effective_at` is still the listed expiry instant.
    booked = _st_et(d, 17, 0)
    L = _st_one_lot_ledger(c, _st_et(prev, 10, 0))
    x = _st_book_due(L, data, _st_et(d, 10, 0), booked)
    @test x.settlement_price == 595.0
    @test effective_at(x) == c.expiry && recorded_at(x) == booked
end

@testset "settlement_price: the deepest walk the calendar actually produces" begin
    # Hurricane Sandy: the NYSE was shut Monday 2012-10-29 and Tuesday the
    # 30th, and the weekend precedes them, so a Tuesday expiry settles four
    # calendar days back at Friday the 26th. Four is the longest run of
    # closed days this calendar produces, which is why the ten-day bound has
    # room; shortening it below this would turn a real closure into the
    # `:no_session` that the bound exists to make impossible here.
    d, friday = Date(2012, 10, 30), Date(2012, 10, 26)
    data = MarketData(InMemory(vcat(_st_session(Date(2012, 10, 25), 140.0),
                                    _st_session(friday, 141.0))))
    c = _st_call(d, 135.0)
    t = _st_et(d, 16, 0)
    @test settlement_price(:session_close, TimeCut(data, t), c, t) == 141.0
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
        TimeCut(f.data, t), book, _st_et(_ST_D18, 10, 0), t)
    @test isempty(out.settled)
    @test length(out.unsettled) == 1
    @test only(out.unsettled)[2].reason == :unexpected_gap
    @test only(out.unsettled)[1].contract == f.a
    @test only(out.unsettled)[2].contract == f.a
end

@testset "run_backtest: an unsettleable lot stays open, warned about once" begin
    f = _st_gap_fixture()
    p = _ST_OpenAt([f.ticks[1] => f.a])
    out = @test_logs (:warn, "lot left open: no honest settlement price") run_backtest(
        p, f.data, f.ticks[1], _st_et(_ST_D22, 20, 0), _ST_CLOCK)
    L = out.ledger
    # Three ticks and a window-end pass follow the expiry; the warning fired
    # once, because the interval examines a lot exactly once, ever (D5).
    @test !any(e isa Expiry for e in L.events)
    @test [l.contract for l in open_lots(L.book)] == [f.a]
    @test L.book == book_effective(L, _st_et(_ST_D22, 20, 0))
    # The engine is the *producer* of what persistence later keeps. A warning
    # dies with the run; this is the fact that leaves it, stamped at the tick
    # whose lifecycle pass asked -- the Friday, not the Thursday expiry.
    r = only(out.failures)
    @test r.stage === :settlement && r.reason === :unexpected_gap
    @test r.at == f.ticks[4]
    @test occursin("470.0C", r.subject) && occursin("lot@", r.subject)
end

@testset "run_backtest: the window-end pass retains its own unsettled lot" begin
    # The other lifecycle pass. The expiry falls after the last policy tick,
    # so nothing but the endpoint pass ever examines this lot -- and that
    # pass's `.unsettled` was the half that used to be warned about and then
    # dropped on the floor.
    f = _st_gap_fixture()
    to = _st_et(_ST_D18, 20, 0)                  # past the 16:00 ET expiry
    p = _ST_OpenAt([f.ticks[1] => f.a])
    out = @test_logs (:warn, "lot left open: no honest settlement price") run_backtest(
        p, f.data, f.ticks[1], to, _ST_CLOCK)
    @test !any(e isa Expiry for e in out.ledger.events)
    @test [l.contract for l in open_lots(out.ledger.book)] == [f.a]
    r = only(out.failures)
    @test r.stage === :settlement && r.reason === :unexpected_gap
    # Stamped at the window end, which is not any tick of the run: that is
    # what says which pass asked the question.
    @test r.at == to
    @test !any(t -> t == r.at, f.ticks)
    @test occursin("470.0C", r.subject) && occursin("lot@", r.subject)
end

@testset "run_backtest: an unsettleable lot is still in the book the policy sees" begin
    f = _st_gap_fixture()
    # The other half of D4, from where a policy stands. Contract A expired on
    # the Thursday and its settlement price cannot be resolved, so the lot
    # stays open -- and stays in the book handed to every later decision. The
    # policy is not told the leg is gone; `Lot` carries the contract, so
    # `lot.contract.expiry <= t` is what it reads to tell an unsettleable
    # expired lot from a live one -- inclusive, because the interval that
    # examines a lot is `(prev, t]` and so a tick *at* an expiry is already
    # a tick after the failed settlement. The strict `<` the docs used to
    # give a policy calls the lot at that tick live.
    p = _ST_Recording(_ST_OpenAt([f.ticks[1] => f.a]))
    L = @test_logs (:warn, "lot left open: no honest settlement price") run_backtest(
        p, f.data, f.ticks[1], _st_et(_ST_D22, 20, 0), _ST_CLOCK).ledger
    seen = Dict(p.seen)
    @test length(p.seen) == 5
    for t in (f.ticks[4], f.ticks[5])          # the two ticks after the expiry
        lots = open_lots(seen[t])
        @test [l.contract for l in lots] == [f.a]
        @test all(l.contract.expiry <= t for l in lots)
    end
    @test !any(e isa Expiry for e in L.events)
end

# The gap fixture with one extra quote stamped at contract A's own expiry,
# so the clock ticks exactly there: Tue, Wed and Thu 10:00, Thu 16:00 ET
# (the expiry itself), then Fri and Mon 10:00 ET.
function _st_gap_expiry_tick_fixture()
    a = _st_call(_ST_D18, 470.0)
    days = (_ST_D16, _ST_D17, _ST_D18, _ST_D19, _ST_D22)
    spots = vcat(_st_session(_ST_D16, 471.0), _st_session(_ST_D17, 472.0),
                 _st_session(_ST_D19, 480.0), _st_session(_ST_D22, 485.0))
    quotes = vcat(OptionQuote[_st_quote(d, a, 5.00, 5.10) for d in days],
                  [_st_quote_at(a.expiry, a, 5.00, 5.10)])
    (data = MarketData(InMemory(quotes), InMemory(spots)), a = a,
     ticks = [_st_et(_ST_D16, 10, 0), _st_et(_ST_D17, 10, 0), _st_et(_ST_D18, 10, 0),
              a.expiry, _st_et(_ST_D19, 10, 0), _st_et(_ST_D22, 10, 0)])
end

@testset "run_backtest: at the expiry tick itself the unsettled lot is visible" begin
    f = _st_gap_expiry_tick_fixture()
    # The tick the strict predicate got wrong. Lifecycle's interval is
    # inclusive at `t`, so the failed settlement happens *at* this tick and
    # the warned-about lot is in the book this very decision is handed.
    p = _ST_Recording(_ST_OpenAt([f.ticks[1] => f.a]))
    L = @test_logs (:warn, "lot left open: no honest settlement price") run_backtest(
        p, f.data, f.ticks[1], _st_et(_ST_D22, 20, 0), _ST_CLOCK).ledger
    seen = Dict(p.seen)
    @test length(p.seen) == 6
    at_expiry = seen[f.a.expiry]
    lots = open_lots(at_expiry)
    @test [l.contract for l in lots] == [f.a]
    @test all(l.contract.expiry <= f.a.expiry for l in lots)   # the documented predicate
    @test !all(l.contract.expiry < f.a.expiry for l in lots)   # ... and why it is not `<`
    # One warning for the whole run, not one per tick (D5), and no `Expiry`:
    # the lot could not be priced, so nothing was booked for it.
    @test !any(e isa Expiry for e in L.events)
    @test [l.contract for l in open_lots(L.book)] == [f.a]
end

# ---- the loop --------------------------------------------------------

@testset "run_backtest: an expiry is booked at the tick after its instant" begin
    f = _st_fixture()
    to = _st_et(_ST_D22, 10, 0)
    p = _ST_OpenAt([f.ticks[1] => f.a])
    L = run_backtest(p, f.data, f.ticks[1], to, _ST_CLOCK).ledger
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
    L = run_backtest(p, f.data, f.ticks[1], to, _ST_CLOCK).ledger
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
    L = run_backtest(p, f.data, f.ticks[1], to, _ST_CLOCK).ledger
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
    L = run_backtest(p, f.data, f.ticks[1], to, _ST_CLOCK).ledger
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
                     f.ticks[1], to, _ST_CLOCK).ledger
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
    L = run_backtest(_ST_OpenAt([f.ticks[1] => f.b]), f.data, f.ticks[1], to, _ST_CLOCK).ledger
    book = L.book
    before = deepcopy(book)
    cut = TimeCut(f.data, to)
    # B expires after the window: nothing is due, and nothing changes.
    one = settlements(cut, book, f.ticks[1], to)
    two = settlements(cut, book, f.ticks[1], to)
    @test isempty(one.settled) && isempty(one.unsettled)
    @test one == two
    @test book == before
    # A lot whose expiry is at or before the interval's lower bound is not
    # returned at all: the bound is open below (D5).
    g = _st_fixture()
    held = _st_one_lot_book(g.a, _st_et(_ST_D16, 10, 0))
    held_before = deepcopy(held)
    due = settlements(TimeCut(g.data, to), held, _st_et(_ST_D18, 10, 0), to)
    @test [lot.contract for (lot, _) in due.settled] == [g.a]
    @test only(due.settled)[2] == 480.0
    # Repeated on a book that *does* have a lot falling due: the answer is
    # the same one and the book is untouched, which is what "computes, never
    # records" means on the path that actually returns something.
    again = settlements(TimeCut(g.data, to), held, _st_et(_ST_D18, 10, 0), to)
    @test again == due
    @test held == held_before
    @test isempty(due.unsettled) && isempty(again.unsettled)
    # The same, on the path that fails: an unsettleable due lot is still
    # unsettleable, and warned about again, when the same interval is asked
    # twice. Once-only is the loop's interval, not memory inside the call.
    h = _st_gap_fixture()
    gap_book = _st_one_lot_book(h.a, _st_et(_ST_D16, 10, 0))
    gap_before = deepcopy(gap_book)
    gap_args = (TimeCut(h.data, to), gap_book, _st_et(_ST_D17, 10, 0), to)
    first_gap = @test_logs (:warn, "lot left open: no honest settlement price") settlements(
        gap_args...)
    second_gap = @test_logs (:warn, "lot left open: no honest settlement price") settlements(
        gap_args...)
    @test [e.reason for (_, e) in first_gap.unsettled] == [:unexpected_gap]
    @test [e.reason for (_, e) in second_gap.unsettled] ==
          [e.reason for (_, e) in first_gap.unsettled]
    @test isempty(first_gap.settled) && isempty(second_gap.settled)
    @test gap_book == gap_before
    @test isempty(settlements(TimeCut(g.data, to), held, g.a.expiry, to).settled)
    @test held == _st_one_lot_book(g.a, _st_et(_ST_D16, 10, 0))
end

# ---- opening at the expiry instant -----------------------------------

# A chain that quotes contract A at A's own expiry instant, so the clock
# ticks there and a policy can try to open the contract as it expires.
# Four ticks: Thu 10:00, Fri 10:00, Fri 16:00 ET (the expiry) and Mon 10:00.
function _st_expiry_tick_fixture()
    a = _st_call(_ST_D19, 470.0)
    spots = vcat(_st_session(_ST_D18, 473.0), _st_session(_ST_D19, 480.0),
                 _st_session(_ST_D22, 485.0))
    quotes = [_st_quote(_ST_D18, a, 5.00, 5.10), _st_quote(_ST_D19, a, 5.00, 5.10),
              _st_quote_at(a.expiry, a, 5.00, 5.10), _st_quote(_ST_D22, a, 5.00, 5.10)]
    (data = MarketData(InMemory(quotes), InMemory(spots)), a = a,
     ticks = [_st_et(_ST_D18, 10, 0), _st_et(_ST_D19, 10, 0), a.expiry,
              _st_et(_ST_D22, 10, 0)])
end

@testset "run_backtest: opening a contract at its own expiry instant is refused" begin
    f = _st_expiry_tick_fixture()
    monday = _st_et(_ST_D22, 20, 0)
    open_at_expiry = _ST_OpenAt([f.a.expiry => f.a])
    # The lifecycle interval (prev, t] has already passed over this expiry by
    # the time the tick fills, so such a lot would never be examined again --
    # not at a later tick, not at the window end. The engine refuses instead:
    # trading has stopped at the expiry instant, so the leg cannot be priced.
    cases = [("interior tick", f.ticks[1], monday),          # third of four ticks
             ("first tick",    f.a.expiry, monday),          # t == from
             ("last tick",     f.ticks[1], f.a.expiry)]      # t == to
    for (name, from, to) in cases
        @testset "$name" begin
            err = try run_backtest(open_at_expiry, f.data, from, to, _ST_CLOCK).ledger; nothing catch e; e end
            @test err isa UnpriceableLeg
            @test err.reason == :expired_contract
            @test err.contract == f.a
            @test err.t == f.a.expiry
            @test occursin("expired_contract", sprint(showerror, err))
        end
    end
    # The refusal is `fill_legs`', so it is raised before anything is written.
    order = Order(:leg, [Leg(f.a, Long, 1, Open)])
    @test_throws UnpriceableLeg fill_legs(TimeCut(f.data, f.a.expiry), order, f.a.expiry;
                                          fill_rule = :cross_spread,
                                          cost_model = :ibkr_pro_us_options, tick_cents = 1)
    # The ledger's own rule is unchanged and still permits a fill effective at
    # the expiry instant; the engine is the stricter layer (design rule 3).
    @test [l.contract for l in open_lots(_st_one_lot_book(f.a, f.a.expiry))] == [f.a]
    # One tick earlier the same order fills and the lot settles normally:
    # bought at 5.10 (-51000, fee 100), settled at 10.00 intrinsic (+100000).
    L = run_backtest(_ST_OpenAt([f.ticks[2] => f.a]), f.data, f.ticks[1], monday, _ST_CLOCK).ledger
    @test [typeof(e) for e in L.events] == [Fill, Fee, Expiry]
    @test L.book.cash == 100000 - 51000 - 100
    @test isempty(open_lots(L.book))
end

# ---- an intraday expiry ----------------------------------------------

# Contract C expires at NOON ET on the Friday, not at the 16:00 ET close.
# The Friday session prints 475 at 09:30, 478 at 10:00 and 480 at 16:00, so
# the last print of the whole session is one that did not exist until four
# hours after the contract stopped existing. Ticks: Thu 10:00, Fri 10:00,
# Mon 10:00 ET.
function _st_intraday_fixture()
    c = ContractKey(_ST_SPY, 470.0, _st_et(_ST_D19, 12, 0), Call)
    spots = vcat(_st_session(_ST_D18, 473.0), _st_session(_ST_D19, 480.0),
                 _st_session(_ST_D22, 485.0))
    quotes = OptionQuote[_st_quote(d, c, 5.00, 5.10) for d in (_ST_D18, _ST_D19, _ST_D22)]
    (data = MarketData(InMemory(quotes), InMemory(spots)), c = c,
     ticks = [_st_et(d, 10, 0) for d in (_ST_D18, _ST_D19, _ST_D22)])
end

@testset "settlement_price: an intraday expiry stops at the expiry instant" begin
    f = _st_intraday_fixture()
    monday = _st_et(_ST_D22, 10, 0)
    cut = TimeCut(f.data, monday)
    # The 16:00 print is there and the cut may read it -- this is not a
    # no-lookahead question. The window is what stops at noon.
    @test only(at(cut, SpotPrice, _ST_SPY, _st_et(_ST_D19, 16, 0))).price == 480.0
    @test settlement_price(:session_close, cut, f.c, monday) == 478.0
end

@testset "run_backtest: an intraday expiry booked later settles pre-expiry" begin
    f = _st_intraday_fixture()
    monday = _st_et(_ST_D22, 10, 0)
    L = run_backtest(_ST_OpenAt([f.ticks[2] => f.c]), f.data, f.ticks[1], monday, _ST_CLOCK).ledger
    @test [typeof(e) for e in L.events] == [Fill, Fee, Expiry]
    x = only(e for e in L.events if e isa Expiry)
    @test x.settlement_price == 478.0            # the 10:00 print, not the 16:00 one
    @test x.outcome == CashSettled
    @test effective_at(x) == f.c.expiry          # noon, the instant the price is from
    @test recorded_at(x) == monday               # learned at the tick that booked it
    # The event effective at noon carries a price that existed at noon, so
    # the effective-time replay is coherent at an instant between the two.
    saturday = _st_et(Date(2024, 1, 20), 12, 0)
    @test isempty(open_lots(book_effective(L, saturday)))
    @test book_as_known(L, last_sequence(L)) == book_effective(L, monday)
    # 478 - 470 = 8.00 intrinsic, paid 5.10, one lone contract raised to the
    # USD 1.00 minimum: 80000 - 51000 - 100.
    @test L.book.cash == 80000 - 51000 - 100
    @test [t.pnl for t in round_trips(L)] == [28900]
end

# ---- an expiry before its own session opens ---------------------------

@testset "settlement_price: an expiry before the session opens is :pre_open_expiry" begin
    # The expiry bound runs the window backwards here: 09:30 ET is after a
    # 09:00 ET expiry, so `[09:30, 09:00]` is empty however complete the data
    # is, and the failure names the contract instead of blaming observations
    # that could not exist. The contract is SPY, a PM-settled one: this is
    # `:session_close`'s own domain edge and not a stand-in for AM
    # settlement, which is now its own named failure below.
    @test contract_spec(_ST_SPY).settlement === PMSettled
    spots = vcat(_st_session(_ST_D18, 473.0), _st_session(_ST_D19, 480.0),
                 _st_session(_ST_D22, 485.0))
    data = MarketData(InMemory(spots))
    monday = _st_et(_ST_D22, 10, 0)
    cut = TimeCut(data, monday)
    # The Friday is a complete session in this fixture: three prints, and
    # the cut may read every one of them.
    @test [p.price for p in between(cut, SpotPrice, _ST_SPY,
                                    _st_et(_ST_D19, 9, 30), _st_et(_ST_D19, 16, 0))] ==
          [475.0, 478.0, 480.0]

    c = ContractKey(_ST_SPY, 470.0, _st_et(_ST_D19, 9, 0), Call)
    e = try settlement_price(:session_close, cut, c, monday) catch err; err end
    @test e isa UnpriceableLeg
    @test e.contract == c && e.t == monday && e.reason == :pre_open_expiry

    # Not the data gap it used to be called: the gap reason is for a date
    # that printed nothing, and this one printed all day.
    @test e.reason != :unexpected_gap
    # And `settlements` treats it like any other named failure (D4): one
    # warning, and the lot stays open rather than settling at a guess.
    book = _st_one_lot_book(c, _st_et(_ST_D18, 10, 0))
    out = @test_logs (:warn, "lot left open: no honest settlement price") settlements(
        cut, book, _st_et(_ST_D18, 10, 0), monday)
    @test isempty(out.settled)
    @test only(out.unsettled)[2].reason == :pre_open_expiry
end

@testset "settlement_price: an expiry at exactly 09:30 ET is a one-instant window" begin
    # The bound is `window_start > window_end`, not `>=`: at the opening
    # instant the window is a single point, and the opening print is in it.
    spots = vcat(_st_session(_ST_D18, 473.0), _st_session(_ST_D19, 480.0))
    data = MarketData(InMemory(spots))
    monday = _st_et(_ST_D22, 10, 0)
    c = ContractKey(_ST_SPY, 470.0, _st_et(_ST_D19, 9, 30), Call)
    @test settlement_price(:session_close, TimeCut(data, monday), c, monday) == 475.0
end

@testset "settlement_price: a pre-open expiry on a closed date still walks back" begin
    # Sunday 2024-03-10, the US spring-forward. The listed date's window is
    # degenerate here too, but the calendar calls the date closed, so that
    # says nothing about the contract: the Sunday is simply not a session.
    # The walk back crosses the DST change -- Sunday 09:00 ET is EDT, the
    # Friday close is EST -- and settles at Friday's 480.
    sunday, friday = Date(2024, 3, 10), Date(2024, 3, 8)
    c = ContractKey(_ST_SPY, 470.0, et_to_utc(sunday, Time(9, 0)), Call)
    data = MarketData(InMemory(vcat(_st_session(Date(2024, 3, 7), 478.0),
                                    _st_session(friday, 480.0))))
    t = et_to_utc(sunday, Time(12, 0))
    @test settlement_price(:session_close, TimeCut(data, t), c, t) == 480.0
    # The two instants really are an hour apart in offset, so this is the
    # DST crossing and not an accident of the fixture.
    @test et_to_utc(sunday, Time(9, 0)) - DateTime(sunday) == Hour(13)
    @test et_to_utc(friday, Time(9, 0)) - DateTime(friday) == Hour(14)
end

# ---- a settlement style no rule serves --------------------------------

# An AM-settled underlying, which `_CONTRACT_TABLE` does not list: the
# table holds facts about underlyings this project trades, so the only way
# to reach the AM branch is to add one for the duration of a test and take
# it out again. Restoring it is the `finally`'s job -- the table is module
# state shared with every later testset.
const _ST_AM = Underlying("AMX")

function _st_with_am_entry(f)
    VolSurfaceAnalysis._CONTRACT_TABLE[ticker(_ST_AM)] =
        ContractSpec(100, European, AMSettled, Cash)
    try
        f()
    finally
        delete!(VolSurfaceAnalysis._CONTRACT_TABLE, ticker(_ST_AM))
    end
end

@testset "settlements: an AM-settled lot throws UnsupportedSettlement" begin
    _st_with_am_entry() do
        # Not caught and warned like an unpriceable lot: that names a lot
        # whose price is unavailable at this instant, and design rule 7
        # leaves it open and says so; this names a contract class no rule
        # settles, so every later tick would answer the same and finishing
        # the run would report a never-valued position as merely still open.
        c = ContractKey(_ST_AM, 470.0, _st_et(_ST_D19, 16, 0), Call)
        L = _st_one_lot_ledger(c, _st_et(_ST_D16, 10, 0))
        snap, before = _lg_snapshot(L), deepcopy(L.book)
        data = MarketData(InMemory(vcat(_st_session(_ST_D18, 473.0),
                                        _st_session(_ST_D19, 480.0),
                                        _st_session(_ST_D22, 485.0))))
        monday = _st_et(_ST_D22, 10, 0)
        e = try settlements(TimeCut(data, monday), L.book,
                            _st_et(_ST_D18, 10, 0), monday); nothing catch err; err end
        @test e isa UnsupportedSettlement
        @test e.underlying == _ST_AM && e.style == AMSettled
        @test occursin("UnsupportedSettlement", sprint(showerror, e))
        @test occursin("AMX", sprint(showerror, e)) && occursin("AMSettled", sprint(showerror, e))
        # Nothing was settled, nothing was written, and the lot is still open.
        @test _lg_snapshot(L) == snap && L.book == before
        @test [l.contract for l in open_lots(L.book)] == [c]
    end
    # The fixture entry is gone again, so no later testset sees it.
    @test_throws UnknownContract contract_spec(_ST_AM)
end

# ---- `between` yields an iterable, not a container ---------------------

# What a lazy `between` is allowed to return: `IteratorSize` is
# `SizeUnknown()`, and there is deliberately no `lastindex` or `getindex`,
# so anything that indexes -- `last` included -- throws instead of quietly
# working on a vector. `once` makes it single-pass as well: a restart
# errors, which a generic reader streaming one partition at a time is
# entitled to do.
mutable struct _ST_LazyPrints
    rows::Vector{SpotPrice}
    once::Bool
    started::Bool
end

Base.IteratorSize(::Type{_ST_LazyPrints}) = Base.SizeUnknown()
Base.eltype(::Type{_ST_LazyPrints}) = SpotPrice

function Base.iterate(it::_ST_LazyPrints, i::Int = 1)
    if i == 1
        it.once && it.started && error("_ST_LazyPrints: traversed twice")
        it.started = true
    end
    i > length(it.rows) ? nothing : (it.rows[i], i + 1)
end

# The minimum a spot provider needs for `settlement_price`: a kind, a
# `serves` that answers (or `TimeCut` refuses the read), and `between`.
# Settlement is the only consumer of `between` outside `market_data`, so
# calling the rule directly is the level this pins.
struct _ST_LazySpots
    rows::Vector{SpotPrice}
    once::Bool
end

VolSurfaceAnalysis.kind(::_ST_LazySpots) = SpotPrice
VolSurfaceAnalysis.serves(::_ST_LazySpots, ::Any, ::Type{SpotPrice}, sel) = true

VolSurfaceAnalysis.between(p::_ST_LazySpots, ::Any, ::Type{SpotPrice}, sel,
                           from::DateTime, to::DateTime) =
    _ST_LazyPrints(SpotPrice[r for r in p.rows
                             if selector(r) == sel && from <= r.timestamp <= to],
                   p.once, false)

_st_lazy_data(rows; once::Bool = false) =
    MarketData(_ST_LazySpots(sort(collect(SpotPrice, rows); by = r -> r.timestamp), once))

@testset "settlement_price: the reference window is consumed as an iterable" begin
    a = _st_call(_ST_D19, 470.0)
    monday = _st_et(_ST_D22, 10, 0)
    session = vcat(_st_session(_ST_D18, 473.0), _st_session(_ST_D19, 480.0))

    @testset "a non-indexable iterator" begin
        # No `lastindex`, so `last(prints)` would be a `MethodError`.
        data = _st_lazy_data(session)
        @test settlement_price(:session_close, TimeCut(data, monday), a, monday) == 480.0
    end

    @testset "a single-pass iterator" begin
        # One traversal only: probing for emptiness and then reading again
        # is two, and a streaming reader need not survive the restart.
        data = _st_lazy_data(session; once = true)
        @test settlement_price(:session_close, TimeCut(data, monday), a, monday) == 480.0
    end

    @testset "one record in the window" begin
        # The shortest window that still has a close: the accumulator has
        # to keep the only record it ever sees.
        data = _st_lazy_data([_st_spot(_ST_D19, 16, 0, 481.0)]; once = true)
        @test settlement_price(:session_close, TimeCut(data, monday), a, monday) == 481.0
    end

    @testset "an empty window still walks back" begin
        # 2025-01-09 was closed, so nothing printed and the walk steps back
        # a session -- the fall-through the "saw no record" branch protects.
        d = Date(2025, 1, 9)
        c = _st_call(d, 580.0)
        t = _st_et(d, 16, 0)
        data = _st_lazy_data(vcat(_st_session(Date(2025, 1, 7), 590.0),
                                  _st_session(Date(2025, 1, 8), 595.0)); once = true)
        @test settlement_price(:session_close, TimeCut(data, t), c, t) == 595.0
    end

    @testset "an empty window on an open weekday is still :unexpected_gap" begin
        # The same branch, the other way out: the Thursday printed nothing
        # and the calendar calls it open, so it is a data gap (design rule 7).
        gap = _st_call(_ST_D18, 470.0)
        t = _st_et(_ST_D19, 10, 0)
        data = _st_lazy_data(_st_session(_ST_D17, 472.0); once = true)
        e = try settlement_price(:session_close, TimeCut(data, t), gap, t) catch err; err end
        @test e isa UnpriceableLeg
        @test e.contract == gap && e.t == t && e.reason == :unexpected_gap
    end
end

# ---------- the reference window against a real parquet spot tree ----------
# The rule's text does not change under bar-end visibility; its input does.
# A vendor row stamped 16:00 ET is the 16:00-16:01 minute, after the close,
# and is visible at 16:01 -- outside the 09:30-16:00 window. The row stamped
# 15:59 is the last completed regular-session minute and is visible at
# exactly 16:00, inside it. The distinct closes below are what tells the two
# apart, and the fixture goes through the parquet reader because an
# in-memory fixture is already stamped and could not catch the mapping.

mktempdir() do root
    spots = joinpath(root, "spots_1min")
    d = _ST_D19                                   # Fri 2024-01-19, an ordinary session
    _md_write_spot_parquet(
        joinpath(spots, "date=2024-01-19", "symbol=SPY", "data.parquet"),
        [_st_et(d, 9, 29), _st_et(d, 12, 0), _st_et(d, 15, 58), _st_et(d, 15, 59),
         _st_et(d, 16, 0)],
        [470.0, 475.0, 479.0, 480.0, 499.9])

    @testset "settlement: the window closes on the completed 15:59-16:00 bar" begin
        with_data(MarketData(ParquetSpots(spots))) do data
            close_utc = _st_et(d, 16, 0)
            # what the reader serves: the 15:59 row at the close, the 16:00
            # row a minute later
            @test only_or_missing(at(data, SpotPrice, _ST_SPY, close_utc)).price == 480.0
            @test only_or_missing(at(data, SpotPrice, _ST_SPY, close_utc + Minute(1))).price == 499.9
            @test only_or_missing(at(data, SpotPrice, _ST_SPY, _st_et(d, 15, 59))).price == 479.0

            c = _st_call(d, 470.0)                # expires 16:00 ET on d
            @test c.expiry == close_utc
            cut = TimeCut(data, close_utc + Hour(2))
            @test settlement_price(:session_close, cut, c, close_utc) == 480.0
            # the 16:00 row is not merely outranked, it is outside the window
            @test settlement_price(:session_close, TimeCut(data, close_utc), c, close_utc) == 480.0
        end
    end
end

# The early close, shaped like the production tree rather than a convenient
# one. 2024-12-24 closes at 13:00 ET. The tree holds a raw 13:00 row -- the
# 13:00-13:01 minute -- and nothing after it until the after-hours burst, so
# the row that wins the 09:30-16:00 window is visible at 13:01, the first
# minute *after* the official close.
#
# That is the model's stated departure, unchanged by bar-end stamping and not
# introduced by it: under bar-open the same row sat on the 13:00 boundary and
# won there instead. It is what an official per-series close would remove
# (see the backlog). The fixture pins the measured case, so a future change
# that silently picks the 12:59 bar instead has to argue with a test.
mktempdir() do root
    spots = joinpath(root, "spots_1min")
    d, prev = Date(2024, 12, 24), Date(2024, 12, 23)
    for (day, rows, prices) in (
            (prev, [_st_et(prev, 9, 29), _st_et(prev, 15, 59)], [595.0, 600.0]),
            (d, [_st_et(d, 9, 29), _st_et(d, 12, 0), _st_et(d, 12, 59),
                 _st_et(d, 13, 0), _st_et(d, 16, 30)],
                [604.0, 605.0, 606.0, 607.0, 611.0]))
        _md_write_spot_parquet(
            joinpath(spots, "date=" * Dates.format(day, "yyyy-mm-dd"), "symbol=SPY",
                     "data.parquet"), rows, prices)
    end

    @testset "settlement: an early close settles at the bar visible at 13:01 ET" begin
        with_data(MarketData(ParquetSpots(spots))) do data
            c = _st_call(d, 600.0)
            t = _st_et(d, 16, 0)
            # The 12:59-13:00 minute is visible at 13:00 and is *not* the last.
            @test only_or_missing(at(data, SpotPrice, _ST_SPY, _st_et(d, 13, 0))).price == 606.0
            # The 13:00-13:01 minute is visible at 13:01, still inside the window,
            # and is the one that settles: the minute after the official close.
            @test only_or_missing(at(data, SpotPrice, _ST_SPY, _st_et(d, 13, 1))).price == 607.0
            # Nothing else until the after-hours print, visible at 16:31, outside.
            @test isempty(collect(between(data, SpotPrice, _ST_SPY,
                                          _st_et(d, 13, 1) + Millisecond(1), t)))
            @test only_or_missing(at(data, SpotPrice, _ST_SPY, _st_et(d, 16, 31))).price == 611.0
            @test settlement_price(:session_close, TimeCut(data, t), c, t) == 607.0
        end
    end
end

# ---- the session grid --------------------------------------------------

@testset "session_closes: one instant per session, the last print in the window" begin
    # Mon 2024-01-15 is a holiday; Tue-Fri are ordinary sessions.
    days = [Date(2024, 1, d) for d in 15:19]
    spots = vcat((_st_session(d, 480.0 + i) for (i, d) in enumerate(days))...)
    data = MarketData(InMemory(spots))
    g = session_closes(data, _ST_SPY, DateTime(2024, 1, 15), DateTime(2024, 1, 19, 23, 59))
    @test isempty(g.gaps)
    @test g.closes == [_st_et(Date(2024, 1, d), 16, 0) for d in 16:19]
    @test issorted(g.closes)
end

@testset "session_closes: an early close is the 13:00 ET print, no table needed" begin
    # 2024-12-24 closed at 13:00 ET; the last print in the window is that one.
    d = Date(2024, 12, 24)
    spots = [_st_spot(d, 9, 30, 600.0), _st_spot(d, 13, 0, 604.0)]
    data = MarketData(InMemory(spots))
    g = session_closes(data, _ST_SPY, DateTime(2024, 12, 24), DateTime(2024, 12, 24, 23, 59))
    @test g.closes == [_st_et(d, 13, 0)]
    @test isempty(g.gaps)
end

@testset "session_closes: a date the calendar calls closed is not a session" begin
    # 2025-01-09, the Jimmy Carter national day of mourning: `USNYSE` carries
    # it, so it contributes nothing even though the spot tree prints.
    d = Date(2025, 1, 9)
    data = MarketData(InMemory(_st_session(d, 590.0)))
    g = session_closes(data, _ST_SPY, DateTime(2025, 1, 9), DateTime(2025, 1, 9, 23, 59))
    @test isempty(g.closes) && isempty(g.gaps)
end

@testset "session_closes: a printless open date is a named gap, never silence" begin
    d = Date(2024, 1, 17)                                # ordinary open Wednesday
    data = MarketData(InMemory(vcat(_st_session(Date(2024, 1, 16), 480.0),
                                    _st_session(Date(2024, 1, 18), 482.0))))
    g = session_closes(data, _ST_SPY, DateTime(2024, 1, 16), DateTime(2024, 1, 18, 23, 59))
    @test g.closes == [_st_et(Date(2024, 1, 16), 16, 0), _st_et(Date(2024, 1, 18), 16, 0)]
    @test g.gaps == [_st_et(d, 16, 0)]                   # the nominal close it could not place
    println("  session gap reported at: ", only(g.gaps))
end

@testset "session_closes: a session the window only partly covers is not counted" begin
    d = Date(2024, 1, 16)
    data = MarketData(InMemory(_st_session(d, 480.0)))
    # Starts after 09:30 ET: the run never saw the session end to end, so it
    # is outside the window rather than a short one.
    late = session_closes(data, _ST_SPY, _st_et(d, 10, 0), DateTime(2024, 1, 16, 23, 59))
    @test isempty(late.closes) && isempty(late.gaps)
    # Ends before 16:00 ET: the same.
    early = session_closes(data, _ST_SPY, DateTime(2024, 1, 16), _st_et(d, 15, 0))
    @test isempty(early.closes) && isempty(early.gaps)
    # The whole window: one session.
    whole = session_closes(data, _ST_SPY, DateTime(2024, 1, 16), DateTime(2024, 1, 16, 23, 59))
    @test whole.closes == [_st_et(d, 16, 0)]
end

# The production tree really holds a disagreeing pair at an overnight instant
# (2026-02-07T00:12 UTC, 690.21 vs 690.22), and the session-window
# exposure is bounded inside the session windows and nowhere else.
# So this fixture carries an actual conflict where the tree does: two rows,
# one instant, two prices, between one close and the next open.
#
# It goes through parquet rather than `InMemory`, because `InMemory` collapses
# snapshots in its constructor and would refuse the pair before a reader ever
# saw it -- on disk is the only place a conflict can wait to be read.
#
# This is a regression, not an illustration. An implementation that range read
# across the gap would raise `ConflictingRecords` and build no grid at all;
# reading one window at a time never meets the pair. A fixture without the
# conflict passes either way, which is what made the earlier version of this
# test vacuous.
mktempdir() do root
    spots = joinpath(root, "spots_1min")
    d1, d2 = Date(2024, 1, 16), Date(2024, 1, 17)
    # Vendor time: a row stamped one minute before the instant it stands for.
    _md_write_spot_parquet(
        joinpath(spots, "date=2024-01-16", "symbol=SPY", "data.parquet"),
        [_st_et(d1, 9, 29), _st_et(d1, 9, 59), _st_et(d1, 15, 59),
         _st_et(d1, 19, 11), _st_et(d1, 19, 11)],      # the conflicting pair
        [475.0, 478.0, 480.0, 690.21, 690.22])
    _md_write_spot_parquet(
        joinpath(spots, "date=2024-01-17", "symbol=SPY", "data.parquet"),
        [_st_et(d2, 9, 29), _st_et(d2, 9, 59), _st_et(d2, 15, 59)],
        [477.0, 480.0, 482.0])

    @testset "session_closes: reads only the session windows, never between them" begin
        with_data(MarketData(ParquetSpots(spots))) do data
            overnight = _st_et(d1, 19, 12)     # after the close, before the open
            # The pair is genuinely poisonous: anything reading that instant throws.
            @test_throws ConflictingRecords only_or_missing(
                at(data, SpotPrice, _ST_SPY, overnight))
            @test_throws ConflictingRecords collect(
                between(data, SpotPrice, _ST_SPY, _st_et(d1, 16, 0), _st_et(d2, 9, 30)))

            # The grid spans both sessions across it and is untroubled.
            g = session_closes(data, _ST_SPY, DateTime(2024, 1, 16),
                               DateTime(2024, 1, 17, 23, 59))
            @test g.closes == [_st_et(d1, 16, 0), _st_et(d2, 16, 0)]
            @test isempty(g.gaps)
            println("  grid stepped over a conflicting overnight pair at: ", overnight)
        end
    end
end

@testset "session_closes: the grid agrees with what :session_close settles at" begin
    # The same rule, read two ways: the grid's close for a date is the price
    # `settlement_price` uses for a contract expiring on it.
    d = Date(2024, 1, 18)
    data = MarketData(InMemory(_st_session(d, 486.0)))
    contract = _st_call(d, 480.0)
    cut = TimeCut(data, _st_et(d, 16, 0))
    g = session_closes(data, _ST_SPY, DateTime(2024, 1, 18), DateTime(2024, 1, 18, 23, 59))
    @test only(g.closes) == _st_et(d, 16, 0)
    @test settlement_price(:session_close, cut, contract, _st_et(d, 16, 0)) == 486.0
end
