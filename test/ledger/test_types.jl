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
    @test L.next_id == 1
    @test L.next_sequence == 1
    @test L.next_group == 1
    @test L.next_execution == 1
    # an id the ledger never minted is a named failure, not a bare KeyError
    @test_throws DanglingReference VolSurfaceAnalysis.event(L, 1)
    err = try VolSurfaceAnalysis.event(L, 1); nothing catch e; e end
    @test err isa DanglingReference && err.field == :event_id && err.id == 1
    @test occursin("DanglingReference", sprint(showerror, err))
    @test sprint(show, L) == "Ledger(0 events)"
end
