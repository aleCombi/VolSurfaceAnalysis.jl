# The vocabulary: enums, keys, legs, orders, events, the ledger container.

@testset "types: side_sign and the enums" begin
    @test side_sign(Long) == 1
    @test side_sign(Short) == -1
    @test instances(Intent) == (Open, Close)
    @test instances(ExpiryOutcome) == (Worthless, CashSettled)
end

@testset "types: ContractKey has content hash and equality" begin
    a = ContractKey(Underlying("spy"), 470.0, _LG_EXPIRY_A, Put)
    b = ContractKey(Underlying("SPY"), 470,   _LG_EXPIRY_A, Put)
    @test a == b
    @test hash(a) == hash(b)
    @test a != ContractKey(_LG_SPY, 470.0, _LG_EXPIRY_A, Call)
    @test a != ContractKey(_LG_SPY, 471.0, _LG_EXPIRY_A, Put)
    @test a != ContractKey(_LG_SPY, 470.0, _LG_EXPIRY_B, Put)
    d = Dict((1, a) => 1)
    @test haskey(d, (1, b))
    @test !haskey(d, (2, b))
end

@testset "types: Leg checks its quantity" begin
    leg = Leg(_LG_PUT470, Short, 2, Open)
    @test leg.quantity == 2
    @test leg.intent == Open
    @test leg.side == Short
    @test_throws NonPositiveQuantity Leg(_LG_PUT470, Short, 0, Open)
    @test_throws NonPositiveQuantity Leg(_LG_PUT470, Short, -1, Close)
end

@testset "types: Order carries label, legs, group and operation" begin
    legs = [Leg(_LG_PUT470, Short, 1, Open), Leg(_LG_CALL490, Short, 1, Open)]
    o = Order(:strangle, legs)
    @test o.label == :strangle
    @test length(o.legs) == 2
    @test o.group === nothing
    @test o.operation === nothing
    c = Order(:unwind, [Leg(_LG_PUT470, Long, 1, Close)]; group=3, operation=7)
    @test c.group == 3
    @test c.operation == 7
end

@testset "types: event constructors reject non-positive quantities and invalid prices" begin
    h = EventHeader(1, _LG_T_OPEN, _LG_T_OPEN, 1)
    @test_throws NonPositiveQuantity Fill(h, 1, 1, 1, _LG_PUT470, Short, Open, 0, 0.85, :cross_spread)
    @test_throws NonPositiveQuantity Match(h, 1, 1, 2, 0)
    @test_throws NonPositiveQuantity Expiry(h, 1, 1, _LG_PUT470, Short, 0, 468.0, CashSettled)
    # a fill price is finite and positive
    @test_throws InvalidPrice Fill(h, 1, 1, 1, _LG_PUT470, Short, Open, 1, 0.0, :cross_spread)
    @test_throws InvalidPrice Fill(h, 1, 1, 1, _LG_PUT470, Short, Open, 1, -0.5, :cross_spread)
    @test_throws InvalidPrice Fill(h, 1, 1, 1, _LG_PUT470, Short, Open, 1, Inf, :cross_spread)
    @test_throws InvalidPrice Fill(h, 1, 1, 1, _LG_PUT470, Short, Open, 1, NaN, :cross_spread)
    # a settlement price is finite and non-negative: zero is a price, a negative one is not
    @test Expiry(h, 1, 1, _LG_PUT470, Short, 1, 0.0, CashSettled).settlement_price === 0.0
    @test_throws InvalidPrice Expiry(h, 1, 1, _LG_PUT470, Short, 1, -1.0, CashSettled)
    @test_throws InvalidPrice Expiry(h, 1, 1, _LG_PUT470, Short, 1, -Inf, CashSettled)
    @test_throws InvalidPrice Expiry(h, 1, 1, _LG_PUT470, Short, 1, NaN, CashSettled)
    err = try Fill(h, 1, 1, 1, _LG_PUT470, Short, Open, 1, -0.5, :cross_spread); nothing catch e; e end
    @test err isa InvalidPrice && err.value == -0.5
    @test occursin("InvalidPrice", sprint(showerror, err))
    err = try Expiry(h, 1, 1, _LG_PUT470, Short, 1, NaN, Worthless); nothing catch e; e end
    @test err isa InvalidPrice && isnan(err.value)
end

@testset "types: a fill's join ids are positive" begin
    # order_leg_id and execution_id name an order leg and an execution
    # report; nothing carries a non-positive id, so such a fill can never join
    h = EventHeader(1, _LG_T_OPEN, _LG_T_OPEN, 1)
    @test_throws DanglingReference Fill(h, 1, 0, 1, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread)
    @test_throws DanglingReference Fill(h, 1, 1, 0, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread)
    @test_throws DanglingReference Fill(h, 1, -3, 1, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread)
    err = try Fill(h, 1, 1, -3, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread); nothing catch e; e end
    @test err isa DanglingReference && err.field == :execution_id && err.id == -3
    err = try Fill(h, 1, 0, 1, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread); nothing catch e; e end
    @test err isa DanglingReference && err.field == :order_leg_id && err.id == 0
    @test Fill(h, 1, 1, 1, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread).order_leg_id == 1
end

@testset "types: a fill effective after its contract's expiry cannot be built" begin
    # the fill review's construction-time check: the value never exists; the
    # write path checks it again for an event that bypasses the constructor
    late = _LG_EXPIRY_A + Second(1)
    @test_throws FillAfterExpiry Fill(EventHeader(1, late, late, 1), 1, 1, 1, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread)
    @test_throws FillAfterExpiry Fill(EventHeader(1, _LG_T_NEXT, _LG_T_NEXT, 1), 1, 1, 1, _LG_CALL490, Long, Close, 1, 0.40, :cross_spread)
    err = try Fill(EventHeader(9, late, late, 9), 1, 1, 1, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread); nothing catch e; e end
    @test err isa FillAfterExpiry && err.id == 9 && err.effective_at == late && err.expiry == _LG_EXPIRY_A
    @test occursin("FillAfterExpiry", sprint(showerror, err))
    # at the expiry instant it can be built (a fill at expiry stays allowed),
    # and a later-dated contract at the same instant is fine
    @test Fill(EventHeader(1, _LG_EXPIRY_A, _LG_EXPIRY_A, 1), 1, 1, 1, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread).quantity == 1
    @test Fill(EventHeader(1, late, late, 1), 1, 1, 1, _LG_PUT465B, Short, Open, 1, 1.50, :cross_spread).contract == _LG_PUT465B
end

@testset "types: the container is a vector over the closed union" begin
    @test LedgerEvent === Union{Fill,Match,Expiry,Fee}
    @test eltype(Ledger().events) === LedgerEvent
    @test !(Lot <: LedgerEvent) && !(Leg <: LedgerEvent) && !(Order <: LedgerEvent)
    @test !(OrderRecord <: LedgerEvent) && !(LegObservation <: LedgerEvent)
    @test eltype(Ledger().orders) === OrderRecord
end

@testset "types: the module knows no quotes, spots or time cut" begin
    # the module doc's boundary: identity vocabulary from data only, so a
    # ledger is built, checked and replayed from its own events alone
    dir = joinpath(dirname(pathof(VolSurfaceAnalysis)), "ledger")
    files = filter(f -> endswith(f, ".jl"), readdir(dir))
    @test !isempty(files)                        # the scan must not be vacuous
    for f in files
        src = read(joinpath(dir, f), String)
        for name in ("OptionQuote", "SpotPrice", "TimeCut", "MarketData", "market_data")
            @test !occursin(name, src)
        end
    end
end

@testset "types: header accessors on every kind; group on lifecycle events only" begin
    h   = EventHeader(7, _LG_T_OPEN, _LG_T_CLOSE, 3)
    f   = Fill(h, 2, 5, 9, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread)
    m   = Match(h, 2, 1, 6, 1)
    x   = Expiry(h, 2, 1, _LG_PUT470, Short, 1, 468.0, CashSettled)
    fee = Fee(h, 6, -130)                       # 1.30 USD in cents
    for e in (f, m, x, fee)
        @test e isa LedgerEvent
        @test header(e) === h
        @test event_id(e) == 7
        @test effective_at(e) == _LG_T_OPEN
        @test recorded_at(e) == _LG_T_CLOSE
        @test sequence(e) == 3
    end
    @test group(f) == 2
    @test group(m) == 2
    @test group(x) == 2
    @test group(fee) === nothing
    @test f.price === 0.85
    @test f.execution_id == 9
    @test f.order_leg_id == 5
    @test x.settlement_price === 468.0
end

@testset "types: a fresh Ledger is empty with every counter at one" begin
    L = Ledger()
    @test length(L) == 0
    @test isempty(L)
    @test isempty(L.orders)
    @test L.book == Book()
    @test L.next_id == 1
    @test L.next_sequence == 1
    @test L.next_group == 1
    @test L.next_execution == 1
    @test L.next_order_id == 1
    @test L.next_leg_id == 1
    # an id the ledger never minted is a named failure, not a bare KeyError
    @test_throws DanglingReference VolSurfaceAnalysis.event(L, 1)
    err = try VolSurfaceAnalysis.event(L, 1); nothing catch e; e end
    @test err isa DanglingReference && err.field == :event_id && err.id == 1
    @test occursin("DanglingReference", sprint(showerror, err))
    @test sprint(show, L) == "Ledger(0 events, 0 orders, 0 open lots)"
end

@testset "types: Ledger(events) cannot reconstruct order ids or unused groups" begin
    source, _ = _lg_case_strangle_order()
    @test source.next_order_id == 2
    @test mint_group!(source) == 2                 # minted, but no event uses it
    rebuilt = Ledger(copy(source.events))
    @test rebuilt.next_order_id == 1
    @test rebuilt.next_group == 2                  # event maximum + 1, not source's 3
end

@testset "types: the order journal records construct and are immutable" begin
    obs = LegObservation(_LG_T_OPEN, 0.84, 0.86, 480.0, _LG_T_OPEN)
    @test obs.quote_at == _LG_T_OPEN && obs.bid == 0.84 && obs.ask == 0.86
    @test obs.spot == 480.0 && obs.spot_at == _LG_T_OPEN
    @test !ismutable(obs)
    # either side of the quote may be missing, as the source allows
    half = LegObservation(_LG_T_OPEN, missing, 0.86, 480.0, _LG_T_OPEN)
    @test ismissing(half.bid) && half.ask == 0.86
    order = Order(:strangle, [Leg(_LG_PUT470, Short, 1, Open), Leg(_LG_CALL490, Short, 1, Open)])
    rec = OrderRecord(1, 1, 1, _LG_T_OPEN, 0, order, [obs, obs])
    @test rec.order_id == 1 && rec.first_leg_id == 1 && rec.group == 1
    @test rec.decided_at == _LG_T_OPEN && rec.known_to == 0
    @test rec.order === order
    @test length(rec.observations) == 2
    @test !ismutable(rec)
    @test_throws ErrorException rec.order_id = 2
end

@testset "types: last_sequence is the boundary of everything known so far" begin
    @test last_sequence(Ledger()) == 0
    L, _ = _lg_case_round_trip()                 # three events, sequences 1 to 3
    @test last_sequence(L) == 3 == sequence(L.events[end])
    @test book_as_known(L, last_sequence(L)) == book_effective(L, _LG_FAR)
end

@testset "types: order_leg finds the record and the leg index for every minted leg id" begin
    L, _ = _lg_case_strangle_closed()            # two orders of two legs: leg ids 1 to 4
    @test length(L.orders) == 2
    r1, r2 = L.orders
    @test order_leg(L, 1) == (r1, 1)
    @test order_leg(L, 2) == (r1, 2)
    @test order_leg(L, 3) == (r2, 1)
    @test order_leg(L, 4) == (r2, 2)
    for id in 1:4
        r, k = order_leg(L, id)
        @test r.first_leg_id + k - 1 == id
        @test r.order.legs[k].contract == (isodd(id) ? _LG_PUT470 : _LG_CALL490)
    end
    # an unminted leg id is a named failure
    @test_throws DanglingReference order_leg(L, 9)
    err = try order_leg(L, 9); nothing catch e; e end
    @test err isa DanglingReference && err.field == :order_leg_id && err.id == 9
    @test occursin("DanglingReference", sprint(showerror, err))
    @test_throws DanglingReference order_leg(L, 0)
    @test_throws DanglingReference order_leg(Ledger(), 1)
end
