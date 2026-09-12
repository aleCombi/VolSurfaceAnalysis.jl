# Round trips and the reconciliation with book cash. Cases 1, 2, 5, 6.
# PnL literals are whole USD cents (0.85 per share is 8500 per contract).

@testset "round_trips: case 1, one trip of +4500" begin
    L, _ = _lg_case_round_trip()
    trips = round_trips(L)
    @test length(trips) == 1
    r = trips[1]
    @test r.group == 1 && r.contract == _LG_PUT470 && r.side == Short && r.quantity == 1
    @test r.open_id == 1 && r.close_id == 2
    @test r.opened_at == _LG_T_OPEN && r.closed_at == _LG_T_CLOSE
    @test r.kind == :closed
    @test r.pnl == 4500                          # 8500 - 4000
    @test r.pnl isa Int
end

@testset "round_trips: case 2, per-trip pnl by lot" begin
    L, book = _lg_case_split()
    trips = round_trips(L)
    @test [r.pnl for r in trips] == [9000, 5000]   # (8500 - 4000) * 2, (9000 - 4000) * 1
    @test [r.quantity for r in trips] == [2, 1]
    @test [r.open_id for r in trips] == [1, 2]
    @test all(r.close_id == 3 for r in trips)
    @test all(r.closed_at == _LG_T_CLOSE for r in trips)
    @test [r.opened_at for r in trips] == [_LG_T_OPEN, _LG_T_OPEN2]
    @test sum(r.pnl for r in trips) == book.cash
end

@testset "round_trips: case 3, only the closed group has a trip" begin
    L, _ = _lg_case_two_groups()
    trips = round_trips(L)
    @test length(trips) == 1
    @test trips[1].group == 2 && trips[1].open_id == 2 && trips[1].pnl == 5000   # 12000 - 7000
end

@testset "round_trips: case 4, an expired lot is a trip of kind :expired" begin
    L, _ = _lg_case_mixed_expiries()
    trips = round_trips(L)
    @test length(trips) == 1
    r = trips[1]
    @test r.kind == :expired && r.open_id == 1 && r.close_id == 3
    @test r.closed_at == _LG_EXPIRY_A
    @test r.pnl == -11500                       # (8500 - 20000) * 1 for a short
end

@testset "round_trips: case 5, fees across a partial close" begin
    L, book = _lg_case_fees()
    trips = round_trips(L)
    @test length(trips) == 2
    # a fee of -130 on the close of 3, shared by cumulative rounding over the
    # matches of 2 then 1: round(-130 * 2/3) = -87, then -130 - (-87) = -43
    @test trips[1].pnl == 8913                  # 9000 - 87
    @test trips[2].pnl == 4957                  # 5000 - 43
    @test sum(r.pnl for r in trips) - 14000 == -130
    @test sum(r.pnl for r in trips) == book.cash
    @test book.cash == 13870                    # 14000 - 130
end

@testset "round_trips: a fee on the opening fill is shared by its consumers" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 3, 0.85, g; leg_id=1)                          # fill 1
    record_fee!(L, book, 1, -90; effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN)             # fee 2 on the open (0.90 USD)
    _lg_fill!(L, book, _LG_PUT470, Long, Close, 1, 0.40, g; at=_LG_T_CLOSE, leg_id=2)          # fill 3, match 4
    _lg_fill!(L, book, _LG_PUT470, Long, Close, 2, 0.30, g; at=_LG_T_CLOSE + Hour(1), leg_id=3) # fill 5, match 6
    trips = round_trips(L)
    # cumulative rounding over the consumers of fill 1 (quantity 3):
    # round(-90 * 1/3) = -30, then round(-90 * 3/3) - (-30) = -60
    @test [r.pnl for r in trips] == [4470, 10940]      # (8500 - 4000) - 30, (8500 - 3000) * 2 - 60
    @test [r.close_id for r in trips] == [3, 5]
    @test sum(r.pnl for r in trips) == book.cash
    @test book.cash == 15410                           # 25500 - 90 - 4000 - 6000
end

@testset "round_trips: fee shares are whole cents by cumulative rounding" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 3, 0.85, g; leg_id=1)                    # fill 1
    record_fee!(L, book, 1, -100; effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN)       # fee 2 (1.00 USD)
    for k in 1:3
        _lg_fill!(L, book, _LG_PUT470, Long, Close, 1, 0.40, g; at=_LG_T_CLOSE + Hour(k), leg_id=k + 1)
    end
    trips = round_trips(L)
    # cumulative: round(-100 * 1/3) = -33, round(-100 * 2/3) = -67, round(-100 * 3/3) = -100
    # shares:     -33,                    -67 - (-33) = -34,      -100 - (-67) = -33
    @test [r.pnl for r in trips] == [4467, 4466, 4467]   # 4500 - 33, 4500 - 34, 4500 - 33
    @test sum(r.pnl for r in trips) == 3 * 4500 - 100
    @test sum(r.pnl for r in trips) == book.cash
end

@testset "round_trips: case 6, an open lot has no trip and its cash stays in the book" begin
    L, book = _lg_case_open_at_end()
    trips = round_trips(L)
    @test length(trips) == 1
    @test trips[1].contract == _LG_CALL490 && trips[1].pnl == 5000   # 11000 - 6000
    @test book.cash == 13500
    @test book.cash - sum(r.pnl for r in trips) == 8500    # the open put's premium
    @test open_lots(book) == [Lot(1, _LG_PUT470, Short, 1, 1, 0.85)]
end

@testset "round_trips: reconciliation is exact, trips sum to cash once nothing is open" begin
    for (name, build) in _LG_CASES
        L, book = build()
        if isempty(open_lots(book))
            @test sum(r.pnl for r in round_trips(L)) == book.cash
        end
        for lot in open_lots(book)
            record_expiry!(L, book, lot; settlement_price=480.0,
                           effective_at=lot.contract.expiry, recorded_at=_LG_FAR)
        end
        @test isempty(open_lots(book))
        @test sum(r.pnl for r in round_trips(L)) == book.cash
        @test book == book_effective(L, _LG_FAR)
    end
end

# The three parts of cash while lots are still open: the trips' pnl, the
# opening cash still tied up in open lots (-side * contract_cents(unit_price)
# * remaining per lot), and the fee not yet allocated to any trip (per fill
# with a fee, F - round(F * consumed // Q)). Returns the triple.
function _lg_reconciliation(L, book)
    cents = VolSurfaceAnalysis.contract_cents
    trips = round_trips(L)
    tied_up = sum((-side_sign(l.side) * cents(l.unit_price, _LG_SPEC) * l.remaining
                   for l in open_lots(book)); init=0)
    consumed = Dict{Int,Int}()                    # per fill, quantity consumed by trips
    for r in trips
        consumed[r.open_id] = get(consumed, r.open_id, 0) + r.quantity
        r.kind == :closed && (consumed[r.close_id] = get(consumed, r.close_id, 0) + r.quantity)
    end
    fees = Dict{Int,Int}()                        # total fee per source fill
    for e in L.events
        e isa Fee && (fees[e.source_id] = get(fees, e.source_id, 0) + e.amount)
    end
    unallocated = 0
    for (fid, F) in fees
        Q = VolSurfaceAnalysis.event(L, fid).quantity
        unallocated += F - round(Int, F * get(consumed, fid, 0) // Q)
    end
    return (sum((r.pnl for r in trips); init=0), tied_up, unallocated)
end

@testset "round_trips: partial reconciliation, trips plus open lots plus unallocated fees equal cash" begin
    expected = Dict(
        "full round trip"             => (4500, 0, 0),         # nothing open, no fee
        "close split across lots"     => (14000, 0, 0),
        "two groups on one contract"  => (5000, 11000, 0),     # group 1's short call: +1.10 * 10000 * 1
        "mixed expiries in one group" => (-11500, 15000, 0),   # the short put 465: +1.50 * 10000 * 1
        "fees across a partial close" => (13870, 0, 0),        # the close's fee is fully allocated
        "open at window end"          => (5000, 8500, 0),      # the short put 470: +0.85 * 10000 * 1
    )
    for (name, build) in _LG_CASES
        L, book = build()
        parts = _lg_reconciliation(L, book)
        @test parts == expected[name]
        @test sum(parts) == book.cash
    end
    # a fee on an opening fill consumed only in part: 3 short at 0.85 (25500),
    # fee -90, one closed at 0.40 (-4000): cash 21410; the trip is 4500 - 30 =
    # 4470, two lots remain (17000), and -90 - round(-90 * 1/3) = -60 is unallocated
    L, book = Ledger(), Book()
    g = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 3, 0.85, g; leg_id=1)
    record_fee!(L, book, 1, -90; effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN)
    _lg_fill!(L, book, _LG_PUT470, Long, Close, 1, 0.40, g; at=_LG_T_CLOSE, leg_id=2)
    @test book.cash == 21410                                   # 25500 - 90 - 4000
    @test _lg_reconciliation(L, book) == (4470, 17000, -60)
    @test sum(_lg_reconciliation(L, book)) == book.cash
    # one more closed at 0.30 (-3000): its share is round(-90 * 2/3) - (-30) =
    # -30, so the trip is 5500 - 30 = 5470; one lot remains (8500); -30 unallocated
    _lg_fill!(L, book, _LG_PUT470, Long, Close, 1, 0.30, g; at=_LG_T_CLOSE + Hour(1), leg_id=3)
    @test book.cash == 18410                                   # 21410 - 3000
    @test _lg_reconciliation(L, book) == (4470 + 5470, 8500, -30)
    @test sum(_lg_reconciliation(L, book)) == book.cash
    # the last one at 0.30: the remainder lands on it, nothing is open, exact
    _lg_fill!(L, book, _LG_PUT470, Long, Close, 1, 0.30, g; at=_LG_T_CLOSE + Hour(2), leg_id=4)
    @test _lg_reconciliation(L, book) == (4470 + 5470 + 5470, 0, 0)   # 5500 - 30 again: -90 - (-60)
    @test sum(r.pnl for r in round_trips(L)) == book.cash == 15410  # 25500 - 90 - 4000 - 3000 - 3000
end

@testset "round_trips: a pinned spec overrides the table" begin
    L, _ = _lg_case_round_trip()
    @test round_trips(L, ContractSpec(1, American, PMSettled, Physical))[1].pnl == 45   # multiplier 1: 85 - 40 cents
    @test round_trips(L, _LG_SPEC) == round_trips(L)
end

@testset "round_trips: rows are in sequence order; an empty ledger has none" begin
    L, _ = _lg_case_split()
    @test [r.open_id for r in round_trips(L)] == [1, 2]
    @test isempty(round_trips(Ledger()))
end
