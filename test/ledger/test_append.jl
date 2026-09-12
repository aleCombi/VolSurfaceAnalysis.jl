# The writers and the validated write path. Cases 1, 2, 3, 6 and 9.

@testset "append: mint_group! counts up" begin
    L = Ledger()
    @test mint_group!(L) == 1
    @test mint_group!(L) == 2
    @test L.next_group == 3
    @test length(L) == 0
end

@testset "append: case 1, full round trip" begin
    L, book = _lg_case_round_trip()
    @test length(L) == 3
    f1, f2, m = L.events
    @test f1 isa Fill && f1.intent == Open && f1.side == Short && f1.quantity == 1 && f1.price == 0.85
    @test f2 isa Fill && f2.intent == Close && f2.side == Long && f2.price == 0.40
    @test m isa Match && m.open_fill_id == event_id(f1) && m.close_fill_id == event_id(f2) && m.quantity == 1
    @test group(f1) == 1 && group(f2) == 1 && group(m) == 1
    @test [event_id(e) for e in L.events] == [1, 2, 3]
    @test [sequence(e) for e in L.events] == [1, 2, 3]
    @test f1.execution_id == 1 && f2.execution_id == 2
    @test f1.order_leg_id == 1 && f2.order_leg_id == 2
    @test f1.fill_rule == :cross_spread
    @test effective_at(m) == _LG_T_CLOSE && recorded_at(m) == _LG_T_CLOSE
    @test cash(f1) ≈ 85.0
    @test cash(f2) ≈ -40.0
    @test book.cash ≈ 45.0
    @test isempty(open_lots(book))
    @test isempty(open_groups(book))
    @test (L.next_id, L.next_sequence, L.next_execution, L.next_group) == (4, 4, 3, 2)
    @test VolSurfaceAnalysis.event(L, 3) === m
end

@testset "append: record_fill! returns the batch it appended" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    b1 = _lg_fill!(L, book, _LG_PUT470, Short, Open, 2, 0.85, g)
    @test length(b1) == 1 && b1[1] isa Fill
    b2 = _lg_fill!(L, book, _LG_PUT470, Long, Close, 2, 0.40, g; at=_LG_T_CLOSE, leg_id=2)
    @test length(b2) == 2 && b2[1] isa Fill && b2[2] isa Match
    @test L.events == vcat(b1, b2)
end

@testset "append: case 2, close split across lots" begin
    L, book = _lg_case_split()
    @test length(L) == 5
    matches = [e for e in L.events if e isa Match]
    @test [(m.open_fill_id, m.quantity) for m in matches] == [(1, 2), (2, 1)]   # FIFO
    @test all(m.close_fill_id == 3 for m in matches)
    @test [sequence(m) for m in matches] == [4, 5]
    @test book.cash ≈ 140.0
    @test isempty(open_lots(book))
end

@testset "append: a partial close leaves the oldest lot's remainder" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470, Short, Open,  2, 0.85, g; at=_LG_T_OPEN,  leg_id=1)
    _lg_fill!(L, book, _LG_PUT470, Short, Open,  1, 0.90, g; at=_LG_T_OPEN2, leg_id=2)
    batch = _lg_fill!(L, book, _LG_PUT470, Long, Close, 1, 0.40, g; at=_LG_T_CLOSE, leg_id=3)
    @test length(batch) == 2
    @test open_lots(book) == [Lot(1, _LG_PUT470, Short, 1, 1, 0.85), Lot(1, _LG_PUT470, Short, 2, 1, 0.90)]
    @test book.cash ≈ 220.0
end

@testset "append: case 3, two groups on one contract" begin
    L, book = _lg_case_two_groups()
    @test book.cash ≈ 160.0
    @test open_groups(book) == [1]
    @test lots(book, 1) == [Lot(1, _LG_CALL490, Short, 1, 1, 1.10)]
    @test isempty(lots(book, 2))
    m = only(e for e in L.events if e isa Match)
    @test m.group == 2 && m.open_fill_id == 2

    snap = _lg_snapshot(L)
    @test_throws ExceedsOpen    _lg_fill!(L, book, _LG_CALL490, Long, Close, 2, 0.70, 1; at=_LG_T_CLOSE, leg_id=4)
    @test_throws NothingToClose _lg_fill!(L, book, _LG_CALL490, Long, Close, 1, 0.70, 2; at=_LG_T_CLOSE, leg_id=4)
    @test_throws NothingToClose _lg_fill!(L, book, _LG_PUT470,  Long, Close, 1, 0.70, 1; at=_LG_T_CLOSE, leg_id=4)
    # a close matches the opposite side only: a same-side "close" has nothing to close
    @test_throws NothingToClose _lg_fill!(L, book, _LG_CALL490, Short, Close, 1, 0.70, 1; at=_LG_T_CLOSE, leg_id=4)
    @test _lg_snapshot(L) == snap
    @test book.cash ≈ 160.0
    err = try _lg_fill!(L, book, _LG_CALL490, Long, Close, 2, 0.70, 1; at=_LG_T_CLOSE, leg_id=4); nothing catch e; e end
    @test err isa ExceedsOpen && err.group == 1 && err.requested == 2 && err.available == 1
end

@testset "append: case 6, a lot left open at the window end" begin
    L, book = _lg_case_open_at_end()
    @test book.cash ≈ 135.0
    @test open_lots(book) == [Lot(1, _LG_PUT470, Short, 1, 1, 0.85)]
    @test open_groups(book) == [1]
end

@testset "append: record_expiry! settles the whole remaining lot" begin
    L, book = _lg_case_mixed_expiries()
    x = L.events[end]
    @test x isa Expiry
    @test x.open_fill_id == 1 && x.contract == _LG_PUT470 && x.side == Short && x.group == 1
    @test x.settlement_price == 468.0 && x.outcome == CashSettled
    @test effective_at(x) == _LG_EXPIRY_A && recorded_at(x) == _LG_T_NEXT
    @test cash(x) ≈ -200.0
    lot = only(open_lots(book))
    w = record_expiry!(L, book, lot; settlement_price=470.0, effective_at=_LG_EXPIRY_B, recorded_at=_LG_EXPIRY_B)
    @test w.outcome == Worthless
    @test w.quantity == 1
    @test cash(w) == 0.0
    @test isempty(open_lots(book))
    @test book.cash ≈ 35.0
    # the lot is gone: expiring it again over-consumes a lot with nothing left
    snap = _lg_snapshot(L)
    @test_throws ExceedsOpen record_expiry!(L, book, lot; settlement_price=470.0, effective_at=_LG_EXPIRY_B, recorded_at=_LG_EXPIRY_B)
    @test _lg_snapshot(L) == snap
end

@testset "append: record_fee! ties a cost to its fill" begin
    L, book = _lg_case_fees()
    fee = L.events[end]
    @test fee isa Fee
    @test fee.source_id == 3 && fee.amount == -1.30
    @test book.cash ≈ 138.70
    snap = _lg_snapshot(L)
    @test_throws DanglingReference record_fee!(L, book, 99, -1.0; effective_at=_LG_T_CLOSE, recorded_at=_LG_T_CLOSE)
    @test_throws DanglingReference record_fee!(L, book, 4, -1.0; effective_at=_LG_T_CLOSE, recorded_at=_LG_T_CLOSE)  # 4 is a Match
    @test _lg_snapshot(L) == snap
    @test book.cash ≈ 138.70
end

@testset "append: FillAfterExpiry" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    @test_throws FillAfterExpiry _lg_fill!(L, book, _LG_PUT470, Short, Open, 1, 0.85, g; at=_LG_EXPIRY_A + Second(1))
    @test _lg_snapshot(L) == (0, 1, 1, 2, 1)
    @test book == Book()
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 1, 0.85, g; at=_LG_EXPIRY_A)     # at the instant is allowed
    @test length(L) == 1
    @test_throws FillAfterExpiry _lg_fill!(L, book, _LG_PUT470, Long, Close, 1, 0.40, g; at=_LG_EXPIRY_A + Minute(1), leg_id=2)
    @test length(L) == 1
end

@testset "append: SequenceGap on either counter" begin
    L, book = _lg_case_round_trip()
    snap = _lg_snapshot(L)
    mk(id, seq) = Fill(EventHeader(id, _LG_T_CLOSE, _LG_T_CLOSE, seq), 1, 9, 9, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread)
    @test_throws SequenceGap commit!(L, book, LedgerEvent[mk(L.next_id, L.next_sequence + 1)], _LG_SPEC)
    @test_throws SequenceGap commit!(L, book, LedgerEvent[mk(L.next_id + 1, L.next_sequence)], _LG_SPEC)
    @test_throws SequenceGap commit!(L, book, LedgerEvent[mk(L.next_id - 1, L.next_sequence - 1)], _LG_SPEC)   # a replay of an old id
    err = try commit!(L, book, LedgerEvent[mk(L.next_id + 1, L.next_sequence)], _LG_SPEC); nothing catch e; e end
    @test err.counter == :id && err.expected == 4 && err.got == 5
    err = try commit!(L, book, LedgerEvent[mk(L.next_id, L.next_sequence + 2)], _LG_SPEC); nothing catch e; e end
    @test err.counter == :sequence && err.expected == 4 && err.got == 6
    # a batch whose second event skips
    @test_throws SequenceGap commit!(L, book, LedgerEvent[mk(L.next_id, L.next_sequence), mk(L.next_id + 2, L.next_sequence + 2)], _LG_SPEC)
    @test _lg_snapshot(L) == snap
    @test book.cash ≈ 45.0
end

@testset "append: DanglingReference" begin
    L, book = _lg_case_round_trip()          # events 1 (open), 2 (close), 3 (match); nothing open
    snap = _lg_snapshot(L)
    # an expiry naming a fill that does not exist
    @test_throws DanglingReference commit!(L, book, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), 1, 42, _LG_PUT470, Short, 1, 468.0, CashSettled)], _LG_SPEC)
    # an expiry naming the closing fill as its opening fill
    @test_throws DanglingReference commit!(L, book, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), 1, 2, _LG_PUT470, Long, 1, 468.0, CashSettled)], _LG_SPEC)
    # a match naming the match as its opening fill
    c = Fill(_lg_hdr(L, 0), 1, 9, 9, _LG_PUT470, Long, Close, 1, 0.40, :cross_spread)
    @test_throws DanglingReference commit!(L, book, LedgerEvent[c, Match(_lg_hdr(L, 1), 1, 3, event_id(c), 1)], _LG_SPEC)
    # a fee naming a match
    @test_throws DanglingReference commit!(L, book, LedgerEvent[Fee(_lg_hdr(L, 0), 3, -1.0)], _LG_SPEC)
    # a match on its own whose closing fill does not exist
    @test_throws DanglingReference commit!(L, book, LedgerEvent[Match(_lg_hdr(L, 0), 1, 1, 77, 1)], _LG_SPEC)
    @test _lg_snapshot(L) == snap
    err = try commit!(L, book, LedgerEvent[Fee(_lg_hdr(L, 0), 3, -1.0)], _LG_SPEC); nothing catch e; e end
    @test err isa DanglingReference && err.field == :source_id && err.id == 3
end

@testset "append: MatchMismatch" begin
    L, book = Ledger(), Book()
    g1 = mint_group!(L)
    g2 = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 2, 0.85, g1; leg_id=1)      # fill 1: a short lot of 2 in group 1
    _lg_fill!(L, book, _LG_PUT470, Long,  Open, 1, 0.50, g1; leg_id=2)      # fill 2: a long lot in group 1
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 1, 0.90, g2; leg_id=3)      # fill 3: a short lot in group 2
    snap = _lg_snapshot(L)
    mkclose(q, g=g1) = Fill(_lg_hdr(L, 0), g, 9, 9, _LG_PUT470, Long, Close, q, 0.40, :cross_spread)
    # the matches do not exhaust the close
    c = mkclose(2)
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[c, Match(_lg_hdr(L, 1), g1, 1, event_id(c), 1)], _LG_SPEC)
    # no matches at all
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[mkclose(1)], _LG_SPEC)
    # the matches over-fill the close
    c = mkclose(1)
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[c, Match(_lg_hdr(L, 1), g1, 1, event_id(c), 1), Match(_lg_hdr(L, 2), g1, 1, event_id(c), 1)], _LG_SPEC)
    # a match pairing a lot of another group
    c = mkclose(1)
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[c, Match(_lg_hdr(L, 1), g1, 3, event_id(c), 1)], _LG_SPEC)
    # a match whose own group differs from its closing fill's
    c = mkclose(1)
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[c, Match(_lg_hdr(L, 1), g2, 1, event_id(c), 1)], _LG_SPEC)
    # a match pairing a same-side lot
    c = mkclose(1)
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[c, Match(_lg_hdr(L, 1), g1, 2, event_id(c), 1)], _LG_SPEC)
    # a match on its own, naming a fill already committed
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[Match(_lg_hdr(L, 0), g1, 1, 1, 1)], _LG_SPEC)
    # an expiry whose copied side, contract or group differ from its opening fill's
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), g1, 1, _LG_PUT470,  Long,  1, 468.0, CashSettled)], _LG_SPEC)
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), g1, 1, _LG_CALL490, Short, 1, 468.0, CashSettled)], _LG_SPEC)
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), g2, 1, _LG_PUT470,  Short, 1, 468.0, CashSettled)], _LG_SPEC)
    @test _lg_snapshot(L) == snap
    @test length(open_lots(book)) == 3
    @test book.cash ≈ 210.0                     # 170 - 50 + 90
    err = try commit!(L, book, LedgerEvent[mkclose(1)], _LG_SPEC); nothing catch e; e end
    @test err isa MatchMismatch && err.id == L.next_id
end

@testset "append: ExceedsOpen on a hand-built over-consumption" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 2, 0.85, g; leg_id=1)   # fill 1
    snap = _lg_snapshot(L)
    c = Fill(_lg_hdr(L, 0), g, 9, 9, _LG_PUT470, Long, Close, 3, 0.40, :cross_spread)
    @test_throws ExceedsOpen commit!(L, book, LedgerEvent[c, Match(_lg_hdr(L, 1), g, 1, event_id(c), 3)], _LG_SPEC)
    # two matches in one batch that together over-consume the lot
    c = Fill(_lg_hdr(L, 0), g, 9, 9, _LG_PUT470, Long, Close, 3, 0.40, :cross_spread)
    @test_throws ExceedsOpen commit!(L, book, LedgerEvent[c, Match(_lg_hdr(L, 1), g, 1, event_id(c), 2), Match(_lg_hdr(L, 2), g, 1, event_id(c), 1)], _LG_SPEC)
    # an expiry for more than the lot holds
    @test_throws ExceedsOpen commit!(L, book, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), g, 1, _LG_PUT470, Short, 3, 468.0, CashSettled)], _LG_SPEC)
    @test _lg_snapshot(L) == snap
    @test open_lots(book) == [Lot(g, _LG_PUT470, Short, 1, 2, 0.85)]
end

@testset "append: one batch may open and close together; an empty batch is a no-op" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    o = Fill(_lg_hdr(L, 0, _LG_T_OPEN),  g, 1, 1, _LG_PUT470, Short, Open,  1, 0.85, :cross_spread)
    c = Fill(_lg_hdr(L, 1, _LG_T_CLOSE), g, 2, 2, _LG_PUT470, Long,  Close, 1, 0.40, :cross_spread)
    m = Match(_lg_hdr(L, 2, _LG_T_CLOSE), g, event_id(o), event_id(c), 1)
    commit!(L, book, LedgerEvent[o, c, m], _LG_SPEC)
    @test length(L) == 3
    @test book.cash ≈ 45.0
    @test isempty(open_lots(book))
    @test L.next_execution == 3
    commit!(L, book, LedgerEvent[], _LG_SPEC)
    @test _lg_snapshot(L) == (3, 4, 4, 2, 3)
    # a hand-built fill with a stale execution id does not move the counter backwards
    late = Fill(_lg_hdr(L, 0, _LG_T_CLOSE), g, 3, 1, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread)
    commit!(L, book, LedgerEvent[late], _LG_SPEC)
    @test L.next_execution == 3
end

@testset "append: NonPositiveQuantity is a named failure" begin
    @test_throws NonPositiveQuantity Leg(_LG_PUT470, Short, 0, Open)
    err = try Leg(_LG_PUT470, Short, -2, Open); nothing catch e; e end
    @test err isa NonPositiveQuantity && err.quantity == -2
    L, book = Ledger(), Book()
    @test_throws NonPositiveQuantity Expiry(_lg_hdr(L, 0), 1, 1, _LG_PUT470, Short, 0, 468.0, Worthless)
    @test_throws NonPositiveQuantity record_expiry!(L, book, Lot(1, _LG_PUT470, Short, 1, 0, 0.85);
                                                    settlement_price=470.0, effective_at=_LG_EXPIRY_A, recorded_at=_LG_EXPIRY_A)
    @test _lg_snapshot(L) == (0, 1, 1, 1, 1)
end

@testset "append: every error is an Exception that prints its name" begin
    for err in (NothingToClose(1, _LG_PUT470), ExceedsOpen(1, _LG_PUT470, 3, 2),
                FillAfterExpiry(1, _LG_T_OPEN, _LG_EXPIRY_A), DanglingReference(:open_fill_id, 9),
                MatchMismatch(4, "why"), SequenceGap(:sequence, 4, 6), NonPositiveQuantity(0))
        @test err isa Exception
        @test occursin(string(nameof(typeof(err))), sprint(showerror, err))
    end
end
