# Round trips and the reconciliation with book cash. Cases 1, 2, 5, 6.

@testset "round_trips: case 1, one trip of +45" begin
    L, _ = _lg_case_round_trip()
    trips = round_trips(L)
    @test length(trips) == 1
    r = trips[1]
    @test r.group == 1 && r.contract == _LG_PUT470 && r.side == Short && r.quantity == 1
    @test r.open_id == 1 && r.close_id == 2
    @test r.opened_at == _LG_T_OPEN && r.closed_at == _LG_T_CLOSE
    @test r.kind == :closed
    @test r.pnl ≈ 45.0
end

@testset "round_trips: case 2, per-trip pnl by lot" begin
    L, book = _lg_case_split()
    trips = round_trips(L)
    @test [r.pnl for r in trips] ≈ [90.0, 50.0]
    @test [r.quantity for r in trips] == [2, 1]
    @test [r.open_id for r in trips] == [1, 2]
    @test all(r.close_id == 3 for r in trips)
    @test all(r.closed_at == _LG_T_CLOSE for r in trips)
    @test [r.opened_at for r in trips] == [_LG_T_OPEN, _LG_T_OPEN2]
    @test sum(r.pnl for r in trips) ≈ book.cash
end

@testset "round_trips: case 3, only the closed group has a trip" begin
    L, _ = _lg_case_two_groups()
    trips = round_trips(L)
    @test length(trips) == 1
    @test trips[1].group == 2 && trips[1].open_id == 2 && trips[1].pnl ≈ 50.0
end

@testset "round_trips: case 4, an expired lot is a trip of kind :expired" begin
    L, _ = _lg_case_mixed_expiries()
    trips = round_trips(L)
    @test length(trips) == 1
    r = trips[1]
    @test r.kind == :expired && r.open_id == 1 && r.close_id == 3
    @test r.closed_at == _LG_EXPIRY_A
    @test r.pnl ≈ -115.0                        # (0.85 - 2.00) * 100 for a short
end

@testset "round_trips: case 5, fees across a partial close" begin
    L, book = _lg_case_fees()
    trips = round_trips(L)
    @test length(trips) == 2
    @test trips[1].pnl ≈ 89.1333 atol=1e-4      # 90 + (-1.30 * 2/3)
    @test trips[2].pnl ≈ 49.5667 atol=1e-4      # 50 + (-1.30 * 1/3)
    @test sum(r.pnl for r in trips) - 140.0 ≈ -1.30
    @test sum(r.pnl for r in trips) ≈ book.cash
    @test book.cash ≈ 138.70
end

@testset "round_trips: a fee on the opening fill is shared by its consumers" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 3, 0.85, g; leg_id=1)                          # fill 1
    record_fee!(L, book, 1, -0.90; effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN)           # fee 2 on the open
    _lg_fill!(L, book, _LG_PUT470, Long, Close, 1, 0.40, g; at=_LG_T_CLOSE, leg_id=2)          # fill 3, match 4
    _lg_fill!(L, book, _LG_PUT470, Long, Close, 2, 0.30, g; at=_LG_T_CLOSE + Hour(1), leg_id=3) # fill 5, match 6
    trips = round_trips(L)
    @test [r.pnl for r in trips] ≈ [44.70, 109.40]     # 45 - 0.90/3, 110 - 0.90 * 2/3
    @test [r.close_id for r in trips] == [3, 5]
    @test sum(r.pnl for r in trips) ≈ book.cash
    @test book.cash ≈ 154.10
end

@testset "round_trips: case 6, an open lot has no trip and its cash stays in the book" begin
    L, book = _lg_case_open_at_end()
    trips = round_trips(L)
    @test length(trips) == 1
    @test trips[1].contract == _LG_CALL490 && trips[1].pnl ≈ 50.0
    @test book.cash ≈ 135.0
    @test book.cash - sum(r.pnl for r in trips) ≈ 85.0    # the open put's premium
    @test open_lots(book) == [Lot(1, _LG_PUT470, Short, 1, 1, 0.85)]
end

@testset "round_trips: reconciliation, trips sum to cash once nothing is open" begin
    for (name, build) in _LG_CASES
        L, book = build()
        if isempty(open_lots(book))
            @test sum(r.pnl for r in round_trips(L)) ≈ book.cash
        end
        for lot in open_lots(book)
            record_expiry!(L, book, lot; settlement_price=480.0,
                           effective_at=lot.contract.expiry, recorded_at=_LG_FAR)
        end
        @test isempty(open_lots(book))
        @test sum(r.pnl for r in round_trips(L)) ≈ book.cash
        @test book == book_effective(L, _LG_FAR)
    end
end

@testset "round_trips: a pinned spec overrides the table" begin
    L, _ = _lg_case_round_trip()
    @test round_trips(L, ContractSpec(1.0, American, PMSettled, Physical))[1].pnl ≈ 0.45
    @test round_trips(L, _LG_SPEC) == round_trips(L)
end

@testset "round_trips: rows are in sequence order; an empty ledger has none" begin
    L, _ = _lg_case_split()
    @test [r.open_id for r in round_trips(L)] == [1, 2]
    @test isempty(round_trips(Ledger()))
end
