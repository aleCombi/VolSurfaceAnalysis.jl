# The writers and the validated write path. Cases 1, 2, 3, 6 and 9.
# Cash literals are whole USD cents (0.85 per share is 8500 per contract).

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
    @test cash(f1) == 8500                       # +0.85 * 100 * 100
    @test cash(f2) == -4000                      # -0.40 * 100 * 100
    @test book.cash == 4500                      # 8500 - 4000
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
    @test book.cash == 14000                     # 17000 + 9000 - 12000
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
    @test book.cash == 22000                     # 17000 + 9000 - 4000
end

@testset "append: case 3, two groups on one contract" begin
    L, book = _lg_case_two_groups()
    @test book.cash == 16000                     # 11000 + 12000 - 7000
    @test open_groups(book) == [1]
    @test lots(book, 1) == [Lot(1, _LG_CALL490, Short, 1, 1, 1.10)]
    @test isempty(lots(book, 2))
    m = only(e for e in L.events if e isa Match)
    @test m.group == 2 && m.open_fill_id == 2

    snap, before = _lg_snapshot(L), deepcopy(book)
    @test_throws ExceedsOpen    _lg_fill!(L, book, _LG_CALL490, Long, Close, 2, 0.70, 1; at=_LG_T_CLOSE, leg_id=4)
    @test_throws NothingToClose _lg_fill!(L, book, _LG_CALL490, Long, Close, 1, 0.70, 2; at=_LG_T_CLOSE, leg_id=4)
    @test_throws NothingToClose _lg_fill!(L, book, _LG_PUT470,  Long, Close, 1, 0.70, 1; at=_LG_T_CLOSE, leg_id=4)
    # a close matches the opposite side only: a same-side "close" has nothing to close
    @test_throws NothingToClose _lg_fill!(L, book, _LG_CALL490, Short, Close, 1, 0.70, 1; at=_LG_T_CLOSE, leg_id=4)
    @test _lg_snapshot(L) == snap
    @test book == before
    @test book.cash == 16000
    err = try _lg_fill!(L, book, _LG_CALL490, Long, Close, 2, 0.70, 1; at=_LG_T_CLOSE, leg_id=4); nothing catch e; e end
    @test err isa ExceedsOpen && err.group == 1 && err.requested == 2 && err.available == 1
    @test _lg_snapshot(L) == snap && book == before
end

@testset "append: case 6, a lot left open at the window end" begin
    L, book = _lg_case_open_at_end()
    @test book.cash == 13500                     # 8500 + 11000 - 6000
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
    @test cash(x) == -20000                      # -(2.00 * 100 * 100)
    lot = only(open_lots(book))
    w = record_expiry!(L, book, lot; settlement_price=470.0, effective_at=_LG_EXPIRY_B, recorded_at=_LG_EXPIRY_B)
    @test w.outcome == Worthless
    @test w.quantity == 1
    @test cash(w) == 0
    @test isempty(open_lots(book))
    @test book.cash == 3500                      # 8500 + 15000 - 20000
    # the lot is gone: expiring it again over-consumes a lot with nothing left
    snap, before = _lg_snapshot(L), deepcopy(book)
    @test_throws ExceedsOpen record_expiry!(L, book, lot; settlement_price=470.0, effective_at=_LG_EXPIRY_B, recorded_at=_LG_EXPIRY_B)
    @test _lg_snapshot(L) == snap
    @test book == before
end

@testset "append: record_fee! ties a cost to its fill" begin
    L, book = _lg_case_fees()
    fee = L.events[end]
    @test fee isa Fee
    @test fee.source_id == 3 && fee.amount == -130
    @test book.cash == 13870                     # 14000 - 130
    snap, before = _lg_snapshot(L), deepcopy(book)
    @test_throws DanglingReference record_fee!(L, book, 99, -100; effective_at=_LG_T_CLOSE, recorded_at=_LG_T_CLOSE)
    @test_throws DanglingReference record_fee!(L, book, 4, -100; effective_at=_LG_T_CLOSE, recorded_at=_LG_T_CLOSE)  # 4 is a Match
    @test _lg_snapshot(L) == snap
    @test book == before
    @test book.cash == 13870
end

@testset "append: FillAfterExpiry" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    @test_throws FillAfterExpiry _lg_fill!(L, book, _LG_PUT470, Short, Open, 1, 0.85, g; at=_LG_EXPIRY_A + Second(1))
    @test _lg_snapshot(L) == (0, 1, 1, 2, 1)
    @test book == Book()
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 1, 0.85, g; at=_LG_EXPIRY_A)     # at the instant is allowed
    @test length(L) == 1
    snap, before = _lg_snapshot(L), deepcopy(book)
    # the writer's fill is refused by the Fill constructor itself, before commit!
    @test_throws FillAfterExpiry _lg_fill!(L, book, _LG_PUT470, Long, Close, 1, 0.40, g; at=_LG_EXPIRY_A + Minute(1), leg_id=2)
    @test _lg_snapshot(L) == snap
    @test book == before
    @test open_lots(book) == [Lot(g, _LG_PUT470, Short, 1, 1, 0.85)]
    err = try _lg_fill!(L, book, _LG_PUT470, Long, Close, 1, 0.40, g; at=_LG_EXPIRY_A + Minute(1), leg_id=2); nothing catch e; e end
    @test err isa FillAfterExpiry && err.id == L.next_id && err.effective_at == _LG_EXPIRY_A + Minute(1) && err.expiry == _LG_EXPIRY_A
    @test occursin("FillAfterExpiry", sprint(showerror, err))
    @test _lg_snapshot(L) == snap && book == before
end

@testset "append: SequenceGap on either counter" begin
    L, book = _lg_case_round_trip()
    snap, before = _lg_snapshot(L), deepcopy(book)
    mk(id, seq) = Fill(EventHeader(id, _LG_T_CLOSE, _LG_T_CLOSE, seq), 1, 9, 9, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread)
    @test_throws SequenceGap commit!(L, book, LedgerEvent[mk(L.next_id, L.next_sequence + 1)])
    @test_throws SequenceGap commit!(L, book, LedgerEvent[mk(L.next_id + 1, L.next_sequence)])
    @test_throws SequenceGap commit!(L, book, LedgerEvent[mk(L.next_id - 1, L.next_sequence - 1)])   # a replay of an old id
    err = try commit!(L, book, LedgerEvent[mk(L.next_id + 1, L.next_sequence)]); nothing catch e; e end
    @test err.counter == :id && err.expected == 4 && err.got == 5
    err = try commit!(L, book, LedgerEvent[mk(L.next_id, L.next_sequence + 2)]); nothing catch e; e end
    @test err.counter == :sequence && err.expected == 4 && err.got == 6
    # a batch whose second event skips
    @test_throws SequenceGap commit!(L, book, LedgerEvent[mk(L.next_id, L.next_sequence), mk(L.next_id + 2, L.next_sequence + 2)])
    @test _lg_snapshot(L) == snap
    @test book == before
    @test book.cash == 4500
    @test occursin("SequenceGap", sprint(showerror, err))
end

@testset "append: DanglingReference" begin
    L, book = _lg_case_round_trip()          # events 1 (open), 2 (close), 3 (match); nothing open
    snap, before = _lg_snapshot(L), deepcopy(book)
    # an expiry naming a fill that does not exist
    @test_throws DanglingReference commit!(L, book, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), 1, 42, _LG_PUT470, Short, 1, 468.0, CashSettled)])
    # an expiry naming the closing fill as its opening fill
    @test_throws DanglingReference commit!(L, book, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), 1, 2, _LG_PUT470, Long, 1, 468.0, CashSettled)])
    # an expiry naming the match as its opening fill
    @test_throws DanglingReference commit!(L, book, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), 1, 3, _LG_PUT470, Short, 1, 468.0, CashSettled)])
    # a match naming the match as its opening fill
    c = Fill(_lg_hdr(L, 0), 1, 9, 9, _LG_PUT470, Long, Close, 1, 0.40, :cross_spread)
    @test_throws DanglingReference commit!(L, book, LedgerEvent[c, Match(_lg_hdr(L, 1), 1, 3, event_id(c), 1)])
    # a match naming an opening fill that comes later in the same batch: a
    # reference points backward in sequence, within a batch too
    o = Fill(_lg_hdr(L, 2), 1, 8, 8, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread)
    @test_throws DanglingReference commit!(L, book, LedgerEvent[c, Match(_lg_hdr(L, 1), 1, event_id(o), event_id(c), 1), o])
    # a fee naming a match
    @test_throws DanglingReference commit!(L, book, LedgerEvent[Fee(_lg_hdr(L, 0), 3, -100)])
    # a match on its own whose closing fill does not exist
    @test_throws DanglingReference commit!(L, book, LedgerEvent[Match(_lg_hdr(L, 0), 1, 1, 77, 1)])
    @test _lg_snapshot(L) == snap
    @test book == before
    err = try commit!(L, book, LedgerEvent[Fee(_lg_hdr(L, 0), 3, -100)]); nothing catch e; e end
    @test err isa DanglingReference && err.field == :source_id && err.id == 3
    @test occursin("DanglingReference", sprint(showerror, err))
    @test _lg_snapshot(L) == snap && book == before
    # a fee naming an expiry
    L, book = _lg_case_mixed_expiries()      # events 1, 2 (opens), 3 (expiry), recorded at T_NEXT
    snap, before = _lg_snapshot(L), deepcopy(book)
    @test_throws DanglingReference commit!(L, book, LedgerEvent[Fee(_lg_hdr(L, 0, _LG_T_NEXT), 3, -100)])
    @test_throws DanglingReference record_fee!(L, book, 3, -100; effective_at=_LG_T_NEXT, recorded_at=_LG_T_NEXT)
    @test _lg_snapshot(L) == snap
    @test book == before
end

@testset "append: MatchMismatch" begin
    L, book = Ledger(), Book()
    g1 = mint_group!(L)
    g2 = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 2, 0.85, g1; leg_id=1)      # fill 1: a short lot of 2 in group 1
    _lg_fill!(L, book, _LG_PUT470, Long,  Open, 1, 0.50, g1; leg_id=2)      # fill 2: a long lot in group 1
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 1, 0.90, g2; leg_id=3)      # fill 3: a short lot in group 2
    snap, before = _lg_snapshot(L), deepcopy(book)
    mkclose(q, g=g1) = Fill(_lg_hdr(L, 0), g, 9, 9, _LG_PUT470, Long, Close, q, 0.40, :cross_spread)
    # the matches do not exhaust the close
    c = mkclose(2)
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[c, Match(_lg_hdr(L, 1), g1, 1, event_id(c), 1)])
    # no matches at all
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[mkclose(1)])
    # the matches over-fill the close
    c = mkclose(1)
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[c, Match(_lg_hdr(L, 1), g1, 1, event_id(c), 1), Match(_lg_hdr(L, 2), g1, 1, event_id(c), 1)])
    # a match pairing a lot of another group
    c = mkclose(1)
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[c, Match(_lg_hdr(L, 1), g1, 3, event_id(c), 1)])
    # a match whose own group differs from its closing fill's
    c = mkclose(1)
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[c, Match(_lg_hdr(L, 1), g2, 1, event_id(c), 1)])
    # a match pairing a same-side lot
    c = mkclose(1)
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[c, Match(_lg_hdr(L, 1), g1, 2, event_id(c), 1)])
    # a match on its own, naming a fill already committed
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[Match(_lg_hdr(L, 0), g1, 1, 1, 1)])
    # an expiry whose copied side, contract or group differ from its opening fill's
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), g1, 1, _LG_PUT470,  Long,  1, 468.0, CashSettled)])
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), g1, 1, _LG_CALL490, Short, 1, 468.0, CashSettled)])
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), g2, 1, _LG_PUT470,  Short, 1, 468.0, CashSettled)])
    @test _lg_snapshot(L) == snap
    @test book == before
    @test length(open_lots(book)) == 3
    @test book.cash == 21000                    # 17000 - 5000 + 9000
    err = try commit!(L, book, LedgerEvent[mkclose(1)]); nothing catch e; e end
    @test err isa MatchMismatch && err.id == L.next_id
    @test occursin("MatchMismatch", sprint(showerror, err))
    @test _lg_snapshot(L) == snap && book == before
end

@testset "append: references point backward in effective time" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 1, 0.85, g; at=_LG_T_OPEN2, leg_id=1)   # fill 1 at T_OPEN2
    snap, before = _lg_snapshot(L), deepcopy(book)
    # a fee effective before its source fill
    @test_throws MatchMismatch record_fee!(L, book, 1, -65; effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN2)
    # a match at an instant other than its closing fill's
    c = Fill(_lg_hdr(L, 0, _LG_T_CLOSE), g, 2, 2, _LG_PUT470, Long, Close, 1, 0.40, :cross_spread)
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[c, Match(_lg_hdr(L, 1, _LG_T_CLOSE + Minute(1)), g, 1, event_id(c), 1)])
    @test _lg_snapshot(L) == snap
    @test book == before
    # at the source's own instant a fee is fine
    record_fee!(L, book, 1, -65; effective_at=_LG_T_OPEN2, recorded_at=_LG_T_OPEN2)
    @test book.cash == 8435                                          # 8500 - 65
end

@testset "append: ExceedsOpen on a hand-built over-consumption" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 2, 0.85, g; leg_id=1)   # fill 1
    snap, before = _lg_snapshot(L), deepcopy(book)
    c = Fill(_lg_hdr(L, 0), g, 9, 9, _LG_PUT470, Long, Close, 3, 0.40, :cross_spread)
    @test_throws ExceedsOpen commit!(L, book, LedgerEvent[c, Match(_lg_hdr(L, 1), g, 1, event_id(c), 3)])
    # two matches in one batch that together over-consume the lot
    c = Fill(_lg_hdr(L, 0), g, 9, 9, _LG_PUT470, Long, Close, 3, 0.40, :cross_spread)
    @test_throws ExceedsOpen commit!(L, book, LedgerEvent[c, Match(_lg_hdr(L, 1), g, 1, event_id(c), 2), Match(_lg_hdr(L, 2), g, 1, event_id(c), 1)])
    # an expiry for more than the lot holds
    @test_throws ExceedsOpen commit!(L, book, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), g, 1, _LG_PUT470, Short, 3, 468.0, CashSettled)])
    @test _lg_snapshot(L) == snap
    @test book == before
    @test open_lots(book) == [Lot(g, _LG_PUT470, Short, 1, 2, 0.85)]
    @test book.cash == 17000                    # 2 * 8500
    err = try commit!(L, book, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), g, 1, _LG_PUT470, Short, 3, 468.0, CashSettled)]); nothing catch e; e end
    @test err isa ExceedsOpen && err.group == g && err.contract == _LG_PUT470 && err.requested == 3 && err.available == 2
    @test occursin("ExceedsOpen", sprint(showerror, err))
    @test _lg_snapshot(L) == snap && book == before
end

@testset "append: one batch may open and close together; an empty batch is a no-op" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    o = Fill(_lg_hdr(L, 0, _LG_T_OPEN),  g, 1, 1, _LG_PUT470, Short, Open,  1, 0.85, :cross_spread)
    c = Fill(_lg_hdr(L, 1, _LG_T_CLOSE), g, 2, 2, _LG_PUT470, Long,  Close, 1, 0.40, :cross_spread)
    m = Match(_lg_hdr(L, 2, _LG_T_CLOSE), g, event_id(o), event_id(c), 1)
    commit!(L, book, LedgerEvent[o, c, m])
    @test length(L) == 3
    @test book.cash == 4500                      # 8500 - 4000
    @test isempty(open_lots(book))
    @test L.next_execution == 3
    commit!(L, book, LedgerEvent[])
    @test _lg_snapshot(L) == (3, 4, 4, 2, 3)
    # a hand-built fill with a stale execution id does not move the counter backwards
    late = Fill(_lg_hdr(L, 0, _LG_T_CLOSE), g, 3, 1, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread)
    commit!(L, book, LedgerEvent[late])
    @test L.next_execution == 3
end

@testset "append: NonIntegralCash refuses a batch whose cash is not whole cents" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    snap, before = _lg_snapshot(L), deepcopy(book)
    # 0.123456 per share on SPY is 0.123456 * 100 * 100 = 1234.56 cents per contract
    f = Fill(_lg_hdr(L, 0, _LG_T_OPEN), g, 1, 1, _LG_PUT470, Short, Open, 1, 0.123456, :cross_spread)
    @test_throws NonIntegralCash commit!(L, book, LedgerEvent[f])
    @test_throws NonIntegralCash _lg_fill!(L, book, _LG_PUT470, Short, Open, 1, 0.123456, g)
    @test _lg_snapshot(L) == snap
    @test book == before
    err = try commit!(L, book, LedgerEvent[f]); nothing catch e; e end
    @test err isa NonIntegralCash && err.value ≈ 1234.56
    @test _lg_snapshot(L) == snap && book == before
    # the batch is refused whole: a valid fill ahead of the bad one does not land
    ok  = Fill(_lg_hdr(L, 0, _LG_T_OPEN), g, 1, 1, _LG_PUT470,  Short, Open, 1, 0.85,     :cross_spread)
    bad = Fill(_lg_hdr(L, 1, _LG_T_OPEN), g, 2, 2, _LG_CALL490, Short, Open, 1, 0.123456, :cross_spread)
    @test_throws NonIntegralCash commit!(L, book, LedgerEvent[ok, bad])
    @test _lg_snapshot(L) == snap
    @test book == before
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
    @test book == Book()
    @test occursin("NonPositiveQuantity", sprint(showerror, err))
end

@testset "append: every error is an Exception that prints its name" begin
    for err in (NothingToClose(1, _LG_PUT470), ExceedsOpen(1, _LG_PUT470, 3, 2),
                FillAfterExpiry(1, _LG_T_OPEN, _LG_EXPIRY_A), DanglingReference(:open_fill_id, 9),
                MatchMismatch(4, "why"), SequenceGap(:sequence, 4, 6), NonPositiveQuantity(0),
                NonIntegralCash(1234.56), InvalidPrice(-0.5),
                RecordedOutOfOrder(7, _LG_T_OPEN, _LG_T_OPEN2), UnknownContract(Underlying("SPX")))
        @test err isa Exception
        @test occursin(string(nameof(typeof(err))), sprint(showerror, err))
    end
end

@testset "append: NothingToClose" begin
    L, book = Ledger(), Book()
    g1 = mint_group!(L); g2 = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 1, 0.85, g1; leg_id=1)
    snap, before = _lg_snapshot(L), deepcopy(book)
    # a group with nothing in it
    @test_throws NothingToClose _lg_fill!(L, book, _LG_PUT470, Long, Close, 1, 0.40, g2; at=_LG_T_CLOSE, leg_id=2)
    # a contract the group does not hold
    @test_throws NothingToClose _lg_fill!(L, book, _LG_CALL490, Long, Close, 1, 0.40, g1; at=_LG_T_CLOSE, leg_id=2)
    # only same-side lots on that contract: a short "close" against a short lot
    @test_throws NothingToClose _lg_fill!(L, book, _LG_PUT470, Short, Close, 1, 0.40, g1; at=_LG_T_CLOSE, leg_id=2)
    @test _lg_snapshot(L) == snap
    @test book == before
    err = try _lg_fill!(L, book, _LG_PUT470, Short, Close, 1, 0.40, g1; at=_LG_T_CLOSE, leg_id=2); nothing catch e; e end
    @test err isa NothingToClose && err.group == g1 && err.contract == _LG_PUT470
    @test occursin("NothingToClose", sprint(showerror, err))
    @test _lg_snapshot(L) == snap && book == before
end

@testset "append: UnknownContract is refused at commit before anything lands" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    snap, before = _lg_snapshot(L), deepcopy(book)
    spx = ContractKey(Underlying("SPX"), 4700.0, _LG_EXPIRY_A, Put)
    f = Fill(_lg_hdr(L, 0, _LG_T_OPEN), g, 1, 1, spx, Short, Open, 1, 10.0, :cross_spread)
    @test_throws UnknownContract commit!(L, book, LedgerEvent[f])
    @test_throws UnknownContract _lg_fill!(L, book, spx, Short, Open, 1, 10.0, g)
    # a listed fill ahead of it in the batch does not land either
    ok  = Fill(_lg_hdr(L, 0, _LG_T_OPEN), g, 1, 1, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread)
    bad = Fill(_lg_hdr(L, 1, _LG_T_OPEN), g, 2, 2, spx,        Short, Open, 1, 10.0, :cross_spread)
    @test_throws UnknownContract commit!(L, book, LedgerEvent[ok, bad])
    @test _lg_snapshot(L) == snap
    @test book == before
    err = try commit!(L, book, LedgerEvent[f]); nothing catch e; e end
    @test err isa UnknownContract && err.underlying == Underlying("SPX")
    @test _lg_snapshot(L) == snap && book == before
end

@testset "append: InvalidPrice is thrown at construction, before the write path" begin
    L, book = _lg_case_open_at_end()             # lot 1: short put 470, open
    snap, before = _lg_snapshot(L), deepcopy(book)
    @test_throws InvalidPrice _lg_fill!(L, book, _LG_PUT470, Long, Close, 1, 0.0,  1; at=_LG_T_CLOSE, leg_id=4)
    @test_throws InvalidPrice _lg_fill!(L, book, _LG_PUT470, Long, Close, 1, -0.4, 1; at=_LG_T_CLOSE, leg_id=4)
    @test_throws InvalidPrice _lg_fill!(L, book, _LG_PUT470, Long, Close, 1, Inf,  1; at=_LG_T_CLOSE, leg_id=4)
    lot = only(open_lots(book))
    @test_throws InvalidPrice record_expiry!(L, book, lot; settlement_price=-1.0, effective_at=_LG_EXPIRY_A, recorded_at=_LG_T_NEXT)
    @test_throws InvalidPrice record_expiry!(L, book, lot; settlement_price=NaN,  effective_at=_LG_EXPIRY_A, recorded_at=_LG_T_NEXT)
    @test _lg_snapshot(L) == snap
    @test book == before
end

@testset "append: an expiry's outcome must agree with its intrinsic value" begin
    L, book = _lg_case_open_at_end()             # lot 1: short put 470, open; last recorded at T_CLOSE
    snap, before = _lg_snapshot(L), deepcopy(book)
    mkx(price, outcome) = Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), 1, 1, _LG_PUT470, Short, 1, price, outcome)
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[mkx(468.0, Worthless)])     # intrinsic 2.00: CashSettled
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[mkx(475.0, CashSettled)])   # intrinsic 0: Worthless
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[mkx(470.0, CashSettled)])   # at the strike intrinsic is exactly 0
    @test _lg_snapshot(L) == snap
    @test book == before
    err = try commit!(L, book, LedgerEvent[mkx(468.0, Worthless)]); nothing catch e; e end
    @test err isa MatchMismatch && err.id == L.next_id && occursin("outcome", err.reason)
    @test _lg_snapshot(L) == snap && book == before
    # the agreeing outcome lands and the lot is gone
    commit!(L, book, LedgerEvent[mkx(468.0, CashSettled)])
    @test isempty(open_lots(book))
    @test book.cash == 13500 - 20000                                 # -(2.00 * 100 * 100) for the short
end

@testset "append: RecordedOutOfOrder, a fact is not recorded before it is true" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    snap, before = _lg_snapshot(L), deepcopy(book)
    # a fill effective at T_OPEN2, recorded an hour earlier at T_OPEN
    @test_throws RecordedOutOfOrder record_fill!(L, book, Leg(_LG_PUT470, Short, 1, Open), g; price=0.85,
                                                  effective_at=_LG_T_OPEN2, recorded_at=_LG_T_OPEN,
                                                  order_leg_id=1, fill_rule=:cross_spread)
    f = Fill(EventHeader(L.next_id, _LG_T_OPEN2, _LG_T_OPEN, L.next_sequence), g, 1, 1, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread)
    @test_throws RecordedOutOfOrder commit!(L, book, LedgerEvent[f])
    @test _lg_snapshot(L) == snap
    @test book == before
    err = try commit!(L, book, LedgerEvent[f]); nothing catch e; e end
    @test err isa RecordedOutOfOrder && err.id == L.next_id && err.recorded_at == _LG_T_OPEN && err.bound == _LG_T_OPEN2
    @test occursin("RecordedOutOfOrder", sprint(showerror, err))
    @test _lg_snapshot(L) == snap && book == before
    # the same for an expiry and a fee: book a lot, then try each recorded before it is true
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 1, 0.85, g; at=_LG_T_OPEN)
    snap, before = _lg_snapshot(L), deepcopy(book)
    lot = only(open_lots(book))
    @test_throws RecordedOutOfOrder record_expiry!(L, book, lot; settlement_price=468.0, effective_at=_LG_EXPIRY_A, recorded_at=_LG_T_CLOSE)
    @test_throws RecordedOutOfOrder record_fee!(L, book, 1, -65; effective_at=_LG_T_OPEN2, recorded_at=_LG_T_OPEN)
    @test _lg_snapshot(L) == snap
    @test book == before
    # recorded at the effective instant, or after it, is fine
    record_fee!(L, book, 1, -65; effective_at=_LG_T_OPEN2, recorded_at=_LG_T_OPEN2)
    record_expiry!(L, book, lot; settlement_price=468.0, effective_at=_LG_EXPIRY_A, recorded_at=_LG_T_NEXT)
    @test length(L) == 3
    @test book.cash == 8500 - 65 - 20000
end

@testset "append: RecordedOutOfOrder, recorded time is nondecreasing along sequence" begin
    # across the batch boundary: case 1's last event is recorded at T_CLOSE; a
    # fee on the opening fill, effective and recorded at T_OPEN2 (fine for the
    # fee on its own), is recorded before the journal's last entry
    L, book = _lg_case_round_trip()
    snap, before = _lg_snapshot(L), deepcopy(book)
    @test_throws RecordedOutOfOrder record_fee!(L, book, 1, -65; effective_at=_LG_T_OPEN2, recorded_at=_LG_T_OPEN2)
    err = try record_fee!(L, book, 1, -65; effective_at=_LG_T_OPEN2, recorded_at=_LG_T_OPEN2); nothing catch e; e end
    @test err isa RecordedOutOfOrder && err.id == L.next_id && err.recorded_at == _LG_T_OPEN2 && err.bound == _LG_T_CLOSE
    @test _lg_snapshot(L) == snap
    @test book == before
    # the same fee recorded at T_CLOSE, equal to the last recorded time, lands: equal instants are normal
    record_fee!(L, book, 1, -65; effective_at=_LG_T_OPEN2, recorded_at=_LG_T_CLOSE)
    @test book.cash == 4435                                          # 4500 - 65
    # within a batch: an open booked late (effective T_OPEN, recorded T_CLOSE)
    # followed by an open effective and recorded at T_OPEN2 steps recorded
    # time back, so the batch is refused whole
    L, book = Ledger(), Book()
    g = mint_group!(L)
    snap, before = _lg_snapshot(L), deepcopy(book)
    late  = Fill(EventHeader(L.next_id,     _LG_T_OPEN,  _LG_T_CLOSE, L.next_sequence),     g, 1, 1, _LG_PUT470,  Short, Open, 1, 0.85, :cross_spread)
    early = Fill(EventHeader(L.next_id + 1, _LG_T_OPEN2, _LG_T_OPEN2, L.next_sequence + 1), g, 2, 2, _LG_CALL490, Short, Open, 1, 1.10, :cross_spread)
    @test_throws RecordedOutOfOrder commit!(L, book, LedgerEvent[late, early])
    err = try commit!(L, book, LedgerEvent[late, early]); nothing catch e; e end
    @test err isa RecordedOutOfOrder && err.id == event_id(early) && err.recorded_at == _LG_T_OPEN2 && err.bound == _LG_T_CLOSE
    @test _lg_snapshot(L) == snap
    @test book == before
    # the other way round the batch is fine: effective time need not be monotone
    early = Fill(EventHeader(L.next_id,     _LG_T_OPEN2, _LG_T_OPEN2, L.next_sequence),     g, 2, 2, _LG_CALL490, Short, Open, 1, 1.10, :cross_spread)
    late  = Fill(EventHeader(L.next_id + 1, _LG_T_OPEN,  _LG_T_CLOSE, L.next_sequence + 1), g, 1, 1, _LG_PUT470,  Short, Open, 1, 0.85, :cross_spread)
    commit!(L, book, LedgerEvent[early, late])
    @test [effective_at(e) for e in L.events] == [_LG_T_OPEN2, _LG_T_OPEN]
    @test [recorded_at(e) for e in L.events] == [_LG_T_OPEN2, _LG_T_CLOSE]
    @test book.cash == 11000 + 8500
end

@testset "append: FIFO across lots opened in the same batch" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    snap, before = _lg_snapshot(L), deepcopy(book)
    a = Fill(_lg_hdr(L, 0, _LG_T_OPEN),  g, 1, 1, _LG_PUT470, Short, Open,  2, 0.85, :cross_spread)   # the older lot, of 2
    b = Fill(_lg_hdr(L, 1, _LG_T_OPEN2), g, 2, 2, _LG_PUT470, Short, Open,  1, 0.90, :cross_spread)   # the newer lot, of 1
    c = Fill(_lg_hdr(L, 2, _LG_T_CLOSE), g, 3, 3, _LG_PUT470, Long,  Close, 3, 0.40, :cross_spread)
    # the newer lot first: rejected
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[a, b, c,
        Match(_lg_hdr(L, 3), g, event_id(b), event_id(c), 1), Match(_lg_hdr(L, 4), g, event_id(a), event_id(c), 2)])
    # the older lot only in part, then the newer while the older still has one left: rejected
    c2 = Fill(_lg_hdr(L, 2, _LG_T_CLOSE), g, 3, 3, _LG_PUT470, Long, Close, 2, 0.40, :cross_spread)
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[a, b, c2,
        Match(_lg_hdr(L, 3), g, event_id(a), event_id(c2), 1), Match(_lg_hdr(L, 4), g, event_id(b), event_id(c2), 1)])
    @test _lg_snapshot(L) == snap
    @test book == before
    # the older lot exhausted first, then the newer: accepted, both gone
    commit!(L, book, LedgerEvent[a, b, c,
        Match(_lg_hdr(L, 3), g, event_id(a), event_id(c), 2), Match(_lg_hdr(L, 4), g, event_id(b), event_id(c), 1)])
    @test isempty(open_lots(book))
    @test book.cash == 14000                                         # 17000 + 9000 - 12000
    @test [r.pnl for r in round_trips(L)] == [9000, 5000]            # (8500 - 4000) * 2, (9000 - 4000) * 1
    # a lot already in the book is older than any lot the batch opens
    L, book = Ledger(), Book()
    g = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 1, 0.85, g; at=_LG_T_OPEN, leg_id=1)    # fill 1, in the book
    snap, before = _lg_snapshot(L), deepcopy(book)
    b = Fill(_lg_hdr(L, 0, _LG_T_OPEN2), g, 2, 2, _LG_PUT470, Short, Open,  1, 0.90, :cross_spread)
    c = Fill(_lg_hdr(L, 1, _LG_T_CLOSE), g, 3, 3, _LG_PUT470, Long,  Close, 2, 0.40, :cross_spread)
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[b, c,
        Match(_lg_hdr(L, 2), g, event_id(b), event_id(c), 1), Match(_lg_hdr(L, 3), g, 1, event_id(c), 1)])
    @test _lg_snapshot(L) == snap
    @test book == before
    commit!(L, book, LedgerEvent[b, c,
        Match(_lg_hdr(L, 2), g, 1, event_id(c), 1), Match(_lg_hdr(L, 3), g, event_id(b), event_id(c), 1)])
    @test isempty(open_lots(book))
    @test book.cash == 8500 + 9000 - 8000
    @test [(r.open_id, r.pnl) for r in round_trips(L)] == [(1, 4500), (2, 5000)]
end

@testset "append: id and sequence are separate counters, never used for each other" begin
    # a ledger whose ids start at 100 while sequence starts at 1 (direct struct
    # construction; nothing in the writers makes the two diverge yet)
    L, book = Ledger(LedgerEvent[], 100, 1, 1, 1, Dict{Int,Int}()), Book()
    g = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470, Short, Open,  2, 0.85, g; at=_LG_T_OPEN,  leg_id=1)   # id 100, seq 1
    _lg_fill!(L, book, _LG_PUT470, Long,  Close, 1, 0.40, g; at=_LG_T_CLOSE, leg_id=2)   # ids 101, 102; seq 2, 3
    record_fee!(L, book, 101, -65; effective_at=_LG_T_CLOSE, recorded_at=_LG_T_CLOSE)    # id 103, seq 4
    lot = only(open_lots(book))
    record_expiry!(L, book, lot; settlement_price=468.0, effective_at=_LG_EXPIRY_A, recorded_at=_LG_T_NEXT)   # id 104, seq 5
    @test [event_id(e) for e in L.events] == [100, 101, 102, 103, 104]
    @test [sequence(e) for e in L.events] == [1, 2, 3, 4, 5]
    @test (L.next_id, L.next_sequence) == (105, 6)
    # lookups are by id; a sequence number is not an id
    @test VolSurfaceAnalysis.event(L, 100) isa Fill && VolSurfaceAnalysis.event(L, 100).intent == Open
    @test VolSurfaceAnalysis.event(L, 102) isa Match
    @test VolSurfaceAnalysis.event(L, 104) isa Expiry
    @test_throws DanglingReference VolSurfaceAnalysis.event(L, 1)
    err = try VolSurfaceAnalysis.event(L, 5); nothing catch e; e end
    @test err isa DanglingReference && err.field == :event_id && err.id == 5
    @test occursin("DanglingReference", sprint(showerror, err))
    # references are by id
    m, fee, x = L.events[3], L.events[4], L.events[5]
    @test m.open_fill_id == 100 && m.close_fill_id == 101
    @test fee.source_id == 101
    @test x.open_fill_id == 100 && lot.open_fill_id == 100
    @test [(r.open_id, r.close_id) for r in round_trips(L)] == [(100, 101), (100, 104)]
    # the boundary of what was known is a sequence
    @test open_lots(book_as_known(L, 1)) == [Lot(g, _LG_PUT470, Short, 100, 2, 0.85)] && book_as_known(L, 1).cash == 17000
    @test book_as_known(L, 3).cash == 13000 && only(open_lots(book_as_known(L, 3))).remaining == 1   # 17000 - 4000
    @test book_as_known(L, 5) == book
    @test book_as_known(L, 100) == book                # a boundary past the end cuts nothing
    @test book_effective(L, _LG_FAR) == book
    _lg_check_book(book)
    # cash 17000 - 4000 - 65 - 20000; trips (8500 - 4000) - 65 and (8500 - 20000)
    @test book.cash == -7065
    @test [r.pnl for r in round_trips(L)] == [4435, -11500]
    @test sum(r.pnl for r in round_trips(L)) == book.cash
    # a hand-built event that uses its id as its sequence, or its sequence as its id, is a SequenceGap
    snap, before = _lg_snapshot(L), deepcopy(book)
    f = Fill(EventHeader(L.next_id, _LG_T_NEXT, _LG_T_NEXT, L.next_id), g, 3, 3, _LG_PUT465B, Short, Open, 1, 1.50, :cross_spread)
    @test_throws SequenceGap commit!(L, book, LedgerEvent[f])
    f = Fill(EventHeader(L.next_sequence, _LG_T_NEXT, _LG_T_NEXT, L.next_sequence), g, 3, 3, _LG_PUT465B, Short, Open, 1, 1.50, :cross_spread)
    @test_throws SequenceGap commit!(L, book, LedgerEvent[f])
    @test _lg_snapshot(L) == snap
    @test book == before
end

# The rejections pinned by the slice 1 review (findings 6.3, 7.1, 7.2 and
# 7.3), moved here from test_review_findings.jl, and the structure-atomicity
# promise (finding 6.4) that waits for slice 2.

@testset "ledger promise: validated batches enforce FIFO" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 1, 0.85, g;
              at=_LG_T_OPEN, leg_id=1)
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 1, 0.90, g;
              at=_LG_T_OPEN2, leg_id=2)
    snap, before = _lg_snapshot(L), deepcopy(book)
    close = Fill(_lg_hdr(L, 0), g, 3, L.next_execution, _LG_PUT470,
                 Long, Close, 1, 0.40, :cross_spread)
    match = Match(_lg_hdr(L, 1), g, 2, event_id(close), 1)   # names the newer lot
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[close, match])
    @test _lg_snapshot(L) == snap
    @test book == before
end

@testset "ledger promise: consumption is not effective before its open" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 1, 0.85, g;
              at=_LG_T_CLOSE, leg_id=1)
    snap, before = _lg_snapshot(L), deepcopy(book)
    @test_throws MatchMismatch record_fill!(
        L, book, Leg(_LG_PUT470, Long, 1, Close), g; price=0.40,
        effective_at=_LG_T_OPEN, recorded_at=_LG_T_CLOSE + Minute(1),
        order_leg_id=2, fill_rule=:cross_spread)
    @test _lg_snapshot(L) == snap
    @test book == before
end

@testset "ledger promise: expiry consumes the whole remaining lot" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 2, 0.85, g;
              at=_LG_T_OPEN, leg_id=1)
    snap, before = _lg_snapshot(L), deepcopy(book)
    expiry = Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), g, 1, _LG_PUT470,
                    Short, 1, 468.0, CashSettled)
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[expiry])
    @test _lg_snapshot(L) == snap
    @test book == before
end

@testset "ledger promise: expiry is not effective before contract expiry" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 1, 0.85, g;
              at=_LG_T_OPEN, leg_id=1)
    snap, before = _lg_snapshot(L), deepcopy(book)
    expiry = Expiry(_lg_hdr(L, 0, _LG_T_OPEN2), g, 1, _LG_PUT470,
                    Short, 1, 468.0, CashSettled)
    @test_throws MatchMismatch commit!(L, book, LedgerEvent[expiry])
    @test _lg_snapshot(L) == snap
    @test book == before
end

@testset "ledger promise: a structure lands whole or not at all" begin
    # Known broken until slice 2 adds record_order!; flip @test_broken to @test then.
    L, book = Ledger(), Book()
    before_group = L.next_group
    order = Order(:invalid_structure, [
        Leg(_LG_PUT470, Short, 1, Open),
        Leg(_LG_CALL490, Long, 1, Close),
    ])
    @test_broken begin
        threw_right = try
            record_order!(L, book, order; prices=[0.85, 0.40], effective_at=_LG_T_OPEN,
                          recorded_at=_LG_T_OPEN, order_leg_ids=[1, 2], fill_rule=:cross_spread)
            false                                   # it must throw
        catch e
            e isa NothingToClose
        end
        threw_right && length(L) == 0 && book == Book() && L.next_group == before_group
    end
end
