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

@testset "types: event constructors reject non-positive quantities and prices" begin
    h = EventHeader(1, _LG_T_OPEN, _LG_T_OPEN, 1)
    @test_throws NonPositiveQuantity Fill(h, 1, 1, 1, _LG_PUT470, Short, Open, 0, 0.85, :cross_spread)
    @test_throws ArgumentError Fill(h, 1, 1, 1, _LG_PUT470, Short, Open, 1, 0.0, :cross_spread)
    @test_throws ArgumentError Fill(h, 1, 1, 1, _LG_PUT470, Short, Open, 1, -0.5, :cross_spread)
    @test_throws NonPositiveQuantity Match(h, 1, 1, 2, 0)
    @test_throws NonPositiveQuantity Expiry(h, 1, 1, _LG_PUT470, Short, 0, 468.0, CashSettled)
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
    @test_throws KeyError VolSurfaceAnalysis.event(L, 1)
    @test sprint(show, L) == "Ledger(0 events)"
end
