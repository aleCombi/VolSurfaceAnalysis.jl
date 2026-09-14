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
    L = Ledger(); book = L.book
    g = mint_group!(L)
    b1 = _lg_fill!(L, _LG_PUT470, Short, Open, 2, 0.85, g)
    @test length(b1) == 1 && b1[1] isa Fill
    b2 = _lg_fill!(L, _LG_PUT470, Long, Close, 2, 0.40, g; at=_LG_T_CLOSE, leg_id=2)
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
    L = Ledger(); book = L.book
    g = mint_group!(L)
    _lg_fill!(L, _LG_PUT470, Short, Open,  2, 0.85, g; at=_LG_T_OPEN,  leg_id=1)
    _lg_fill!(L, _LG_PUT470, Short, Open,  1, 0.90, g; at=_LG_T_OPEN2, leg_id=2)
    batch = _lg_fill!(L, _LG_PUT470, Long, Close, 1, 0.40, g; at=_LG_T_CLOSE, leg_id=3)
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
    @test_throws ExceedsOpen    _lg_fill!(L, _LG_CALL490, Long, Close, 2, 0.70, 1; at=_LG_T_CLOSE, leg_id=4)
    @test_throws NothingToClose _lg_fill!(L, _LG_CALL490, Long, Close, 1, 0.70, 2; at=_LG_T_CLOSE, leg_id=4)
    @test_throws NothingToClose _lg_fill!(L, _LG_PUT470,  Long, Close, 1, 0.70, 1; at=_LG_T_CLOSE, leg_id=4)
    # a close matches the opposite side only: a same-side "close" has nothing to close
    @test_throws NothingToClose _lg_fill!(L, _LG_CALL490, Short, Close, 1, 0.70, 1; at=_LG_T_CLOSE, leg_id=4)
    @test _lg_snapshot(L) == snap
    @test book == before
    @test book.cash == 16000
    err = try _lg_fill!(L, _LG_CALL490, Long, Close, 2, 0.70, 1; at=_LG_T_CLOSE, leg_id=4); nothing catch e; e end
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
    w = record_expiry!(L, lot; settlement_price=470.0, effective_at=_LG_EXPIRY_B, recorded_at=_LG_EXPIRY_B)
    @test w.outcome == Worthless
    @test w.quantity == 1
    @test cash(w) == 0
    @test isempty(open_lots(book))
    @test book.cash == 3500                      # 8500 + 15000 - 20000
    # the lot is gone: expiring it again over-consumes a lot with nothing left
    snap, before = _lg_snapshot(L), deepcopy(book)
    @test_throws ExceedsOpen record_expiry!(L, lot; settlement_price=470.0, effective_at=_LG_EXPIRY_B, recorded_at=_LG_EXPIRY_B)
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
    @test_throws DanglingReference record_fee!(L, 99, -100; effective_at=_LG_T_CLOSE, recorded_at=_LG_T_CLOSE)
    @test_throws DanglingReference record_fee!(L, 4, -100; effective_at=_LG_T_CLOSE, recorded_at=_LG_T_CLOSE)  # 4 is a Match
    @test _lg_snapshot(L) == snap
    @test book == before
    @test book.cash == 13870
end

@testset "append: FillAfterExpiry" begin
    L = Ledger(); book = L.book
    g = mint_group!(L)
    @test_throws FillAfterExpiry _lg_fill!(L, _LG_PUT470, Short, Open, 1, 0.85, g; at=_LG_EXPIRY_A + Second(1))
    @test _lg_snapshot(L) == (0, 1, 1, 2, 1, 1, 1, 0)
    @test book == Book()
    _lg_fill!(L, _LG_PUT470, Short, Open, 1, 0.85, g; at=_LG_EXPIRY_A)     # at the instant is allowed
    @test length(L) == 1
    snap, before = _lg_snapshot(L), deepcopy(book)
    # the writer's fill is refused by the Fill constructor itself, before commit!
    @test_throws FillAfterExpiry _lg_fill!(L, _LG_PUT470, Long, Close, 1, 0.40, g; at=_LG_EXPIRY_A + Minute(1), leg_id=2)
    @test _lg_snapshot(L) == snap
    @test book == before
    @test open_lots(book) == [Lot(g, _LG_PUT470, Short, 1, 1, 0.85)]
    err = try _lg_fill!(L, _LG_PUT470, Long, Close, 1, 0.40, g; at=_LG_EXPIRY_A + Minute(1), leg_id=2); nothing catch e; e end
    @test err isa FillAfterExpiry && err.id == L.next_id && err.effective_at == _LG_EXPIRY_A + Minute(1) && err.expiry == _LG_EXPIRY_A
    @test occursin("FillAfterExpiry", sprint(showerror, err))
    @test _lg_snapshot(L) == snap && book == before
end

@testset "append: SequenceGap on either counter" begin
    L, book = _lg_case_round_trip()
    snap, before = _lg_snapshot(L), deepcopy(book)
    mk(id, seq) = Fill(EventHeader(id, _LG_T_CLOSE, _LG_T_CLOSE, seq), 1, 9, 9, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread)
    @test_throws SequenceGap commit!(L, LedgerEvent[mk(L.next_id, L.next_sequence + 1)])
    @test_throws SequenceGap commit!(L, LedgerEvent[mk(L.next_id + 1, L.next_sequence)])
    @test_throws SequenceGap commit!(L, LedgerEvent[mk(L.next_id - 1, L.next_sequence - 1)])   # a replay of an old id
    err = try commit!(L, LedgerEvent[mk(L.next_id + 1, L.next_sequence)]); nothing catch e; e end
    @test err.counter == :id && err.expected == 4 && err.got == 5
    err = try commit!(L, LedgerEvent[mk(L.next_id, L.next_sequence + 2)]); nothing catch e; e end
    @test err.counter == :sequence && err.expected == 4 && err.got == 6
    # a batch whose second event skips
    @test_throws SequenceGap commit!(L, LedgerEvent[mk(L.next_id, L.next_sequence), mk(L.next_id + 2, L.next_sequence + 2)])
    @test _lg_snapshot(L) == snap
    @test book == before
    @test book.cash == 4500
    @test occursin("SequenceGap", sprint(showerror, err))
end

@testset "append: DanglingReference" begin
    L, book = _lg_case_round_trip()          # events 1 (open), 2 (close), 3 (match); nothing open
    snap, before = _lg_snapshot(L), deepcopy(book)
    # an expiry naming a fill that does not exist
    @test_throws DanglingReference commit!(L, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), 1, 42, _LG_PUT470, Short, 1, 468.0, CashSettled)])
    # an expiry naming the closing fill as its opening fill
    @test_throws DanglingReference commit!(L, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), 1, 2, _LG_PUT470, Long, 1, 468.0, CashSettled)])
    # an expiry naming the match as its opening fill
    @test_throws DanglingReference commit!(L, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), 1, 3, _LG_PUT470, Short, 1, 468.0, CashSettled)])
    # a match naming the match as its opening fill
    c = Fill(_lg_hdr(L, 0), 1, 9, 9, _LG_PUT470, Long, Close, 1, 0.40, :cross_spread)
    @test_throws DanglingReference commit!(L, LedgerEvent[c, Match(_lg_hdr(L, 1), 1, 3, event_id(c), 1)])
    # a match naming an opening fill that comes later in the same batch: a
    # reference points backward in sequence, within a batch too
    o = Fill(_lg_hdr(L, 2), 1, 8, 8, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread)
    @test_throws DanglingReference commit!(L, LedgerEvent[c, Match(_lg_hdr(L, 1), 1, event_id(o), event_id(c), 1), o])
    # a fee naming a match
    @test_throws DanglingReference commit!(L, LedgerEvent[Fee(_lg_hdr(L, 0), 3, -100)])
    # a match on its own whose closing fill does not exist
    @test_throws DanglingReference commit!(L, LedgerEvent[Match(_lg_hdr(L, 0), 1, 1, 77, 1)])
    @test _lg_snapshot(L) == snap
    @test book == before
    err = try commit!(L, LedgerEvent[Fee(_lg_hdr(L, 0), 3, -100)]); nothing catch e; e end
    @test err isa DanglingReference && err.field == :source_id && err.id == 3
    @test occursin("DanglingReference", sprint(showerror, err))
    @test _lg_snapshot(L) == snap && book == before
    # a fee naming an expiry
    L, book = _lg_case_mixed_expiries()      # events 1, 2 (opens), 3 (expiry), recorded at T_NEXT
    snap, before = _lg_snapshot(L), deepcopy(book)
    @test_throws DanglingReference commit!(L, LedgerEvent[Fee(_lg_hdr(L, 0, _LG_T_NEXT), 3, -100)])
    @test_throws DanglingReference record_fee!(L, 3, -100; effective_at=_LG_T_NEXT, recorded_at=_LG_T_NEXT)
    @test _lg_snapshot(L) == snap
    @test book == before
end

@testset "append: MatchMismatch" begin
    L = Ledger(); book = L.book
    g1 = mint_group!(L)
    g2 = mint_group!(L)
    _lg_fill!(L, _LG_PUT470, Short, Open, 2, 0.85, g1; leg_id=1)      # fill 1: a short lot of 2 in group 1
    _lg_fill!(L, _LG_PUT470, Long,  Open, 1, 0.50, g1; leg_id=2)      # fill 2: a long lot in group 1
    _lg_fill!(L, _LG_PUT470, Short, Open, 1, 0.90, g2; leg_id=3)      # fill 3: a short lot in group 2
    snap, before = _lg_snapshot(L), deepcopy(book)
    mkclose(q, g=g1) = Fill(_lg_hdr(L, 0), g, 9, 9, _LG_PUT470, Long, Close, q, 0.40, :cross_spread)
    # the matches do not exhaust the close
    c = mkclose(2)
    @test_throws MatchMismatch commit!(L, LedgerEvent[c, Match(_lg_hdr(L, 1), g1, 1, event_id(c), 1)])
    # no matches at all
    @test_throws MatchMismatch commit!(L, LedgerEvent[mkclose(1)])
    # the matches over-fill the close
    c = mkclose(1)
    @test_throws MatchMismatch commit!(L, LedgerEvent[c, Match(_lg_hdr(L, 1), g1, 1, event_id(c), 1), Match(_lg_hdr(L, 2), g1, 1, event_id(c), 1)])
    # a match pairing a lot of another group
    c = mkclose(1)
    @test_throws MatchMismatch commit!(L, LedgerEvent[c, Match(_lg_hdr(L, 1), g1, 3, event_id(c), 1)])
    # a match whose own group differs from its closing fill's
    c = mkclose(1)
    @test_throws MatchMismatch commit!(L, LedgerEvent[c, Match(_lg_hdr(L, 1), g2, 1, event_id(c), 1)])
    # a match pairing a same-side lot
    c = mkclose(1)
    @test_throws MatchMismatch commit!(L, LedgerEvent[c, Match(_lg_hdr(L, 1), g1, 2, event_id(c), 1)])
    # a match on its own, naming a fill already committed
    @test_throws MatchMismatch commit!(L, LedgerEvent[Match(_lg_hdr(L, 0), g1, 1, 1, 1)])
    # an expiry whose copied side, contract or group differ from its opening fill's
    @test_throws MatchMismatch commit!(L, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), g1, 1, _LG_PUT470,  Long,  1, 468.0, CashSettled)])
    @test_throws MatchMismatch commit!(L, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), g1, 1, _LG_CALL490, Short, 1, 468.0, CashSettled)])
    @test_throws MatchMismatch commit!(L, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), g2, 1, _LG_PUT470,  Short, 1, 468.0, CashSettled)])
    @test _lg_snapshot(L) == snap
    @test book == before
    @test length(open_lots(book)) == 3
    @test book.cash == 21000                    # 17000 - 5000 + 9000
    err = try commit!(L, LedgerEvent[mkclose(1)]); nothing catch e; e end
    @test err isa MatchMismatch && err.id == L.next_id
    @test occursin("MatchMismatch", sprint(showerror, err))
    @test _lg_snapshot(L) == snap && book == before
end

@testset "append: references point backward in effective time" begin
    L = Ledger(); book = L.book
    g = mint_group!(L)
    _lg_fill!(L, _LG_PUT470, Short, Open, 1, 0.85, g; at=_LG_T_OPEN2, leg_id=1)   # fill 1 at T_OPEN2
    snap, before = _lg_snapshot(L), deepcopy(book)
    # a fee effective before its source fill
    @test_throws MatchMismatch record_fee!(L, 1, -65; effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN2)
    # a match at an instant other than its closing fill's
    c = Fill(_lg_hdr(L, 0, _LG_T_CLOSE), g, 2, 2, _LG_PUT470, Long, Close, 1, 0.40, :cross_spread)
    @test_throws MatchMismatch commit!(L, LedgerEvent[c, Match(_lg_hdr(L, 1, _LG_T_CLOSE + Minute(1)), g, 1, event_id(c), 1)])
    @test _lg_snapshot(L) == snap
    @test book == before
    # at the source's own instant a fee is fine
    record_fee!(L, 1, -65; effective_at=_LG_T_OPEN2, recorded_at=_LG_T_OPEN2)
    @test book.cash == 8435                                          # 8500 - 65
end

@testset "append: ExceedsOpen on a hand-built over-consumption" begin
    L = Ledger(); book = L.book
    g = mint_group!(L)
    _lg_fill!(L, _LG_PUT470, Short, Open, 2, 0.85, g; leg_id=1)   # fill 1
    snap, before = _lg_snapshot(L), deepcopy(book)
    c = Fill(_lg_hdr(L, 0), g, 9, 9, _LG_PUT470, Long, Close, 3, 0.40, :cross_spread)
    @test_throws ExceedsOpen commit!(L, LedgerEvent[c, Match(_lg_hdr(L, 1), g, 1, event_id(c), 3)])
    # two matches in one batch that together over-consume the lot
    c = Fill(_lg_hdr(L, 0), g, 9, 9, _LG_PUT470, Long, Close, 3, 0.40, :cross_spread)
    @test_throws ExceedsOpen commit!(L, LedgerEvent[c, Match(_lg_hdr(L, 1), g, 1, event_id(c), 2), Match(_lg_hdr(L, 2), g, 1, event_id(c), 1)])
    # an expiry for more than the lot holds
    @test_throws ExceedsOpen commit!(L, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), g, 1, _LG_PUT470, Short, 3, 468.0, CashSettled)])
    @test _lg_snapshot(L) == snap
    @test book == before
    @test open_lots(book) == [Lot(g, _LG_PUT470, Short, 1, 2, 0.85)]
    @test book.cash == 17000                    # 2 * 8500
    err = try commit!(L, LedgerEvent[Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), g, 1, _LG_PUT470, Short, 3, 468.0, CashSettled)]); nothing catch e; e end
    @test err isa ExceedsOpen && err.group == g && err.contract == _LG_PUT470 && err.requested == 3 && err.available == 2
    @test occursin("ExceedsOpen", sprint(showerror, err))
    @test _lg_snapshot(L) == snap && book == before
end

@testset "append: one batch may open and close together; an empty batch is a no-op" begin
    L = Ledger(); book = L.book
    g = mint_group!(L)
    o = Fill(_lg_hdr(L, 0, _LG_T_OPEN),  g, 1, 1, _LG_PUT470, Short, Open,  1, 0.85, :cross_spread)
    c = Fill(_lg_hdr(L, 1, _LG_T_CLOSE), g, 2, 2, _LG_PUT470, Long,  Close, 1, 0.40, :cross_spread)
    m = Match(_lg_hdr(L, 2, _LG_T_CLOSE), g, event_id(o), event_id(c), 1)
    commit!(L, LedgerEvent[o, c, m])
    @test length(L) == 3
    @test book.cash == 4500                      # 8500 - 4000
    @test isempty(open_lots(book))
    @test L.next_execution == 3
    commit!(L, LedgerEvent[])
    @test _lg_snapshot(L) == (3, 4, 4, 2, 3, 1, 1, 0)
    # a hand-built fill whose execution id is already held is refused: an
    # execution report is one fact (it used to land and leave the counter alone)
    snap, before = _lg_snapshot(L), deepcopy(book)
    late = Fill(_lg_hdr(L, 0, _LG_T_CLOSE), g, 3, 1, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread)
    @test_throws DuplicateExecution commit!(L, LedgerEvent[late])
    @test _lg_snapshot(L) == snap
    @test book == before
    @test L.next_execution == 3
end

@testset "append: DuplicateExecution" begin
    L, book = _lg_case_round_trip()              # fills with execution ids 1 and 2; next is 3
    snap, before = _lg_snapshot(L), deepcopy(book)
    # an id held by a fill in the ledger
    held = Fill(_lg_hdr(L, 0, _LG_T_CLOSE), 1, 3, 2, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread)
    @test_throws DuplicateExecution commit!(L, LedgerEvent[held])
    @test_throws DuplicateExecution commit!(L, LedgerEvent[
        Fill(_lg_hdr(L, 0, _LG_T_CLOSE), 1, 3, 1, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread)])
    # two fills sharing an execution id inside one batch are refused whole
    a = Fill(_lg_hdr(L, 0, _LG_T_CLOSE), 1, 3, 7, _LG_PUT470,  Short, Open, 1, 0.85, :cross_spread)
    b = Fill(_lg_hdr(L, 1, _LG_T_CLOSE), 1, 4, 7, _LG_CALL490, Short, Open, 1, 1.10, :cross_spread)
    @test_throws DuplicateExecution commit!(L, LedgerEvent[a, b])
    @test _lg_snapshot(L) == snap
    @test book == before
    @test book.cash == 4500
    err = try commit!(L, LedgerEvent[a, b]); nothing catch e; e end
    @test err isa DuplicateExecution && err.id == 7
    @test occursin("DuplicateExecution", sprint(showerror, err))
    @test _lg_snapshot(L) == snap && book == before
    # an id above the counter lands and moves it one past
    commit!(L, LedgerEvent[a])
    @test L.next_execution == 8
    @test length(L) == 4
    # and the writer's own ids continue from there
    _lg_fill!(L, _LG_CALL490, Short, Open, 1, 1.10, 1; at=_LG_T_CLOSE, leg_id=4)
    @test L.events[end].execution_id == 8
    @test L.next_execution == 9
end

@testset "append: NonIntegralCash refuses a batch whose cash is not whole cents" begin
    L = Ledger(); book = L.book
    g = mint_group!(L)
    snap, before = _lg_snapshot(L), deepcopy(book)
    # 0.123456 per share on SPY is 0.123456 * 100 * 100 = 1234.56 cents per contract
    f = Fill(_lg_hdr(L, 0, _LG_T_OPEN), g, 1, 1, _LG_PUT470, Short, Open, 1, 0.123456, :cross_spread)
    @test_throws NonIntegralCash commit!(L, LedgerEvent[f])
    @test_throws NonIntegralCash _lg_fill!(L, _LG_PUT470, Short, Open, 1, 0.123456, g)
    @test _lg_snapshot(L) == snap
    @test book == before
    err = try commit!(L, LedgerEvent[f]); nothing catch e; e end
    @test err isa NonIntegralCash && err.value ≈ 1234.56
    @test _lg_snapshot(L) == snap && book == before
    # the batch is refused whole: a valid fill ahead of the bad one does not land
    ok  = Fill(_lg_hdr(L, 0, _LG_T_OPEN), g, 1, 1, _LG_PUT470,  Short, Open, 1, 0.85,     :cross_spread)
    bad = Fill(_lg_hdr(L, 1, _LG_T_OPEN), g, 2, 2, _LG_CALL490, Short, Open, 1, 0.123456, :cross_spread)
    @test_throws NonIntegralCash commit!(L, LedgerEvent[ok, bad])
    @test _lg_snapshot(L) == snap
    @test book == before
end

@testset "append: NonPositiveQuantity is a named failure" begin
    @test_throws NonPositiveQuantity Leg(_LG_PUT470, Short, 0, Open)
    err = try Leg(_LG_PUT470, Short, -2, Open); nothing catch e; e end
    @test err isa NonPositiveQuantity && err.quantity == -2
    L = Ledger(); book = L.book
    @test_throws NonPositiveQuantity Expiry(_lg_hdr(L, 0), 1, 1, _LG_PUT470, Short, 0, 468.0, Worthless)
    @test_throws NonPositiveQuantity record_expiry!(L, Lot(1, _LG_PUT470, Short, 1, 0, 0.85);
                                                    settlement_price=470.0, effective_at=_LG_EXPIRY_A, recorded_at=_LG_EXPIRY_A)
    @test _lg_snapshot(L) == (0, 1, 1, 1, 1, 1, 1, 0)
    @test book == Book()
    @test occursin("NonPositiveQuantity", sprint(showerror, err))
end

@testset "append: every error is an Exception that prints its name" begin
    for err in (NothingToClose(1, _LG_PUT470), ExceedsOpen(1, _LG_PUT470, 3, 2),
                FillAfterExpiry(1, _LG_T_OPEN, _LG_EXPIRY_A), DanglingReference(:open_fill_id, 9),
                MatchMismatch(4, "why"), SequenceGap(:sequence, 4, 6), NonPositiveQuantity(0),
                NonIntegralCash(1234.56), InvalidPrice(-0.5), DuplicateExecution(7),
                RecordedOutOfOrder(7, _LG_T_OPEN, _LG_T_OPEN2), UnknownContract(Underlying("SPX")))
        @test err isa Exception
        @test occursin(string(nameof(typeof(err))), sprint(showerror, err))
    end
end

@testset "append: NothingToClose" begin
    L = Ledger(); book = L.book
    g1 = mint_group!(L); g2 = mint_group!(L)
    _lg_fill!(L, _LG_PUT470, Short, Open, 1, 0.85, g1; leg_id=1)
    snap, before = _lg_snapshot(L), deepcopy(book)
    # a group with nothing in it
    @test_throws NothingToClose _lg_fill!(L, _LG_PUT470, Long, Close, 1, 0.40, g2; at=_LG_T_CLOSE, leg_id=2)
    # a contract the group does not hold
    @test_throws NothingToClose _lg_fill!(L, _LG_CALL490, Long, Close, 1, 0.40, g1; at=_LG_T_CLOSE, leg_id=2)
    # only same-side lots on that contract: a short "close" against a short lot
    @test_throws NothingToClose _lg_fill!(L, _LG_PUT470, Short, Close, 1, 0.40, g1; at=_LG_T_CLOSE, leg_id=2)
    @test _lg_snapshot(L) == snap
    @test book == before
    err = try _lg_fill!(L, _LG_PUT470, Short, Close, 1, 0.40, g1; at=_LG_T_CLOSE, leg_id=2); nothing catch e; e end
    @test err isa NothingToClose && err.group == g1 && err.contract == _LG_PUT470
    @test occursin("NothingToClose", sprint(showerror, err))
    @test _lg_snapshot(L) == snap && book == before
end

@testset "append: UnknownContract is refused at commit before anything lands" begin
    L = Ledger(); book = L.book
    g = mint_group!(L)
    snap, before = _lg_snapshot(L), deepcopy(book)
    spx = ContractKey(Underlying("SPX"), 4700.0, _LG_EXPIRY_A, Put)
    f = Fill(_lg_hdr(L, 0, _LG_T_OPEN), g, 1, 1, spx, Short, Open, 1, 10.0, :cross_spread)
    @test_throws UnknownContract commit!(L, LedgerEvent[f])
    @test_throws UnknownContract _lg_fill!(L, spx, Short, Open, 1, 10.0, g)
    # a listed fill ahead of it in the batch does not land either
    ok  = Fill(_lg_hdr(L, 0, _LG_T_OPEN), g, 1, 1, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread)
    bad = Fill(_lg_hdr(L, 1, _LG_T_OPEN), g, 2, 2, spx,        Short, Open, 1, 10.0, :cross_spread)
    @test_throws UnknownContract commit!(L, LedgerEvent[ok, bad])
    @test _lg_snapshot(L) == snap
    @test book == before
    err = try commit!(L, LedgerEvent[f]); nothing catch e; e end
    @test err isa UnknownContract && err.underlying == Underlying("SPX")
    @test _lg_snapshot(L) == snap && book == before
end

@testset "append: InvalidPrice is thrown at construction, before the write path" begin
    L, book = _lg_case_open_at_end()             # lot 1: short put 470, open
    snap, before = _lg_snapshot(L), deepcopy(book)
    @test_throws InvalidPrice _lg_fill!(L, _LG_PUT470, Long, Close, 1, 0.0,  1; at=_LG_T_CLOSE, leg_id=4)
    @test_throws InvalidPrice _lg_fill!(L, _LG_PUT470, Long, Close, 1, -0.4, 1; at=_LG_T_CLOSE, leg_id=4)
    @test_throws InvalidPrice _lg_fill!(L, _LG_PUT470, Long, Close, 1, Inf,  1; at=_LG_T_CLOSE, leg_id=4)
    lot = only(open_lots(book))
    @test_throws InvalidPrice record_expiry!(L, lot; settlement_price=-1.0, effective_at=_LG_EXPIRY_A, recorded_at=_LG_T_NEXT)
    @test_throws InvalidPrice record_expiry!(L, lot; settlement_price=NaN,  effective_at=_LG_EXPIRY_A, recorded_at=_LG_T_NEXT)
    @test _lg_snapshot(L) == snap
    @test book == before
end

@testset "append: an expiry's outcome must agree with its intrinsic value" begin
    L, book = _lg_case_open_at_end()             # lot 1: short put 470, open; last recorded at T_CLOSE
    snap, before = _lg_snapshot(L), deepcopy(book)
    mkx(price, outcome) = Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), 1, 1, _LG_PUT470, Short, 1, price, outcome)
    @test_throws MatchMismatch commit!(L, LedgerEvent[mkx(468.0, Worthless)])     # intrinsic 2.00: CashSettled
    @test_throws MatchMismatch commit!(L, LedgerEvent[mkx(475.0, CashSettled)])   # intrinsic 0: Worthless
    @test_throws MatchMismatch commit!(L, LedgerEvent[mkx(470.0, CashSettled)])   # at the strike intrinsic is exactly 0
    @test _lg_snapshot(L) == snap
    @test book == before
    err = try commit!(L, LedgerEvent[mkx(468.0, Worthless)]); nothing catch e; e end
    @test err isa MatchMismatch && err.id == L.next_id && occursin("outcome", err.reason)
    @test _lg_snapshot(L) == snap && book == before
    # the agreeing outcome lands and the lot is gone
    commit!(L, LedgerEvent[mkx(468.0, CashSettled)])
    @test isempty(open_lots(book))
    @test book.cash == 13500 - 20000                                 # -(2.00 * 100 * 100) for the short
end

@testset "append: RecordedOutOfOrder, a fact is not recorded before it is true" begin
    L = Ledger(); book = L.book
    g = mint_group!(L)
    snap, before = _lg_snapshot(L), deepcopy(book)
    # a fill effective at T_OPEN2, recorded an hour earlier at T_OPEN
    @test_throws RecordedOutOfOrder record_fill!(L, Leg(_LG_PUT470, Short, 1, Open), g; price=0.85,
                                                  effective_at=_LG_T_OPEN2, recorded_at=_LG_T_OPEN,
                                                  order_leg_id=1, fill_rule=:cross_spread)
    f = Fill(EventHeader(L.next_id, _LG_T_OPEN2, _LG_T_OPEN, L.next_sequence), g, 1, 1, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread)
    @test_throws RecordedOutOfOrder commit!(L, LedgerEvent[f])
    @test _lg_snapshot(L) == snap
    @test book == before
    err = try commit!(L, LedgerEvent[f]); nothing catch e; e end
    @test err isa RecordedOutOfOrder && err.id == L.next_id && err.recorded_at == _LG_T_OPEN && err.bound == _LG_T_OPEN2
    @test occursin("RecordedOutOfOrder", sprint(showerror, err))
    @test _lg_snapshot(L) == snap && book == before
    # the same for an expiry and a fee: book a lot, then try each recorded before it is true
    _lg_fill!(L, _LG_PUT470, Short, Open, 1, 0.85, g; at=_LG_T_OPEN)
    snap, before = _lg_snapshot(L), deepcopy(book)
    lot = only(open_lots(book))
    @test_throws RecordedOutOfOrder record_expiry!(L, lot; settlement_price=468.0, effective_at=_LG_EXPIRY_A, recorded_at=_LG_T_CLOSE)
    @test_throws RecordedOutOfOrder record_fee!(L, 1, -65; effective_at=_LG_T_OPEN2, recorded_at=_LG_T_OPEN)
    @test _lg_snapshot(L) == snap
    @test book == before
    # recorded at the effective instant, or after it, is fine
    record_fee!(L, 1, -65; effective_at=_LG_T_OPEN2, recorded_at=_LG_T_OPEN2)
    record_expiry!(L, lot; settlement_price=468.0, effective_at=_LG_EXPIRY_A, recorded_at=_LG_T_NEXT)
    @test length(L) == 3
    @test book.cash == 8500 - 65 - 20000
end

@testset "append: RecordedOutOfOrder, recorded time is nondecreasing along sequence" begin
    # across the batch boundary: case 1's last event is recorded at T_CLOSE; a
    # fee on the opening fill, effective and recorded at T_OPEN2 (fine for the
    # fee on its own), is recorded before the journal's last entry
    L, book = _lg_case_round_trip()
    snap, before = _lg_snapshot(L), deepcopy(book)
    @test_throws RecordedOutOfOrder record_fee!(L, 1, -65; effective_at=_LG_T_OPEN2, recorded_at=_LG_T_OPEN2)
    err = try record_fee!(L, 1, -65; effective_at=_LG_T_OPEN2, recorded_at=_LG_T_OPEN2); nothing catch e; e end
    @test err isa RecordedOutOfOrder && err.id == L.next_id && err.recorded_at == _LG_T_OPEN2 && err.bound == _LG_T_CLOSE
    @test _lg_snapshot(L) == snap
    @test book == before
    # the same fee recorded at T_CLOSE, equal to the last recorded time, lands: equal instants are normal
    record_fee!(L, 1, -65; effective_at=_LG_T_OPEN2, recorded_at=_LG_T_CLOSE)
    @test book.cash == 4435                                          # 4500 - 65
    # within a batch: an open booked late (effective T_OPEN, recorded T_CLOSE)
    # followed by an open effective and recorded at T_OPEN2 steps recorded
    # time back, so the batch is refused whole
    L = Ledger(); book = L.book
    g = mint_group!(L)
    snap, before = _lg_snapshot(L), deepcopy(book)
    late  = Fill(EventHeader(L.next_id,     _LG_T_OPEN,  _LG_T_CLOSE, L.next_sequence),     g, 1, 1, _LG_PUT470,  Short, Open, 1, 0.85, :cross_spread)
    early = Fill(EventHeader(L.next_id + 1, _LG_T_OPEN2, _LG_T_OPEN2, L.next_sequence + 1), g, 2, 2, _LG_CALL490, Short, Open, 1, 1.10, :cross_spread)
    @test_throws RecordedOutOfOrder commit!(L, LedgerEvent[late, early])
    err = try commit!(L, LedgerEvent[late, early]); nothing catch e; e end
    @test err isa RecordedOutOfOrder && err.id == event_id(early) && err.recorded_at == _LG_T_OPEN2 && err.bound == _LG_T_CLOSE
    @test _lg_snapshot(L) == snap
    @test book == before
    # the other way round the batch is fine: effective time need not be monotone
    early = Fill(EventHeader(L.next_id,     _LG_T_OPEN2, _LG_T_OPEN2, L.next_sequence),     g, 2, 2, _LG_CALL490, Short, Open, 1, 1.10, :cross_spread)
    late  = Fill(EventHeader(L.next_id + 1, _LG_T_OPEN,  _LG_T_CLOSE, L.next_sequence + 1), g, 1, 1, _LG_PUT470,  Short, Open, 1, 0.85, :cross_spread)
    commit!(L, LedgerEvent[early, late])
    @test [effective_at(e) for e in L.events] == [_LG_T_OPEN2, _LG_T_OPEN]
    @test [recorded_at(e) for e in L.events] == [_LG_T_OPEN2, _LG_T_CLOSE]
    @test book.cash == 11000 + 8500
end

@testset "append: FIFO across lots opened in the same batch" begin
    L = Ledger(); book = L.book
    g = mint_group!(L)
    snap, before = _lg_snapshot(L), deepcopy(book)
    a = Fill(_lg_hdr(L, 0, _LG_T_OPEN),  g, 1, 1, _LG_PUT470, Short, Open,  2, 0.85, :cross_spread)   # the older lot, of 2
    b = Fill(_lg_hdr(L, 1, _LG_T_OPEN2), g, 2, 2, _LG_PUT470, Short, Open,  1, 0.90, :cross_spread)   # the newer lot, of 1
    c = Fill(_lg_hdr(L, 2, _LG_T_CLOSE), g, 3, 3, _LG_PUT470, Long,  Close, 3, 0.40, :cross_spread)
    # the newer lot first: rejected
    @test_throws MatchMismatch commit!(L, LedgerEvent[a, b, c,
        Match(_lg_hdr(L, 3), g, event_id(b), event_id(c), 1), Match(_lg_hdr(L, 4), g, event_id(a), event_id(c), 2)])
    # the older lot only in part, then the newer while the older still has one left: rejected
    c2 = Fill(_lg_hdr(L, 2, _LG_T_CLOSE), g, 3, 3, _LG_PUT470, Long, Close, 2, 0.40, :cross_spread)
    @test_throws MatchMismatch commit!(L, LedgerEvent[a, b, c2,
        Match(_lg_hdr(L, 3), g, event_id(a), event_id(c2), 1), Match(_lg_hdr(L, 4), g, event_id(b), event_id(c2), 1)])
    @test _lg_snapshot(L) == snap
    @test book == before
    # the older lot exhausted first, then the newer: accepted, both gone
    commit!(L, LedgerEvent[a, b, c,
        Match(_lg_hdr(L, 3), g, event_id(a), event_id(c), 2), Match(_lg_hdr(L, 4), g, event_id(b), event_id(c), 1)])
    @test isempty(open_lots(book))
    @test book.cash == 14000                                         # 17000 + 9000 - 12000
    @test [r.pnl for r in round_trips(L)] == [9000, 5000]            # (8500 - 4000) * 2, (9000 - 4000) * 1
    # a lot already in the book is older than any lot the batch opens
    L = Ledger(); book = L.book
    g = mint_group!(L)
    _lg_fill!(L, _LG_PUT470, Short, Open, 1, 0.85, g; at=_LG_T_OPEN, leg_id=1)    # fill 1, in the book
    snap, before = _lg_snapshot(L), deepcopy(book)
    b = Fill(_lg_hdr(L, 0, _LG_T_OPEN2), g, 2, 2, _LG_PUT470, Short, Open,  1, 0.90, :cross_spread)
    c = Fill(_lg_hdr(L, 1, _LG_T_CLOSE), g, 3, 3, _LG_PUT470, Long,  Close, 2, 0.40, :cross_spread)
    @test_throws MatchMismatch commit!(L, LedgerEvent[b, c,
        Match(_lg_hdr(L, 2), g, event_id(b), event_id(c), 1), Match(_lg_hdr(L, 3), g, 1, event_id(c), 1)])
    @test _lg_snapshot(L) == snap
    @test book == before
    commit!(L, LedgerEvent[b, c,
        Match(_lg_hdr(L, 2), g, 1, event_id(c), 1), Match(_lg_hdr(L, 3), g, event_id(b), event_id(c), 1)])
    @test isempty(open_lots(book))
    @test book.cash == 8500 + 9000 - 8000
    @test [(r.open_id, r.pnl) for r in round_trips(L)] == [(1, 4500), (2, 5000)]
end

@testset "append: id and sequence are separate counters, never used for each other" begin
    # a ledger whose ids start at 100 while sequence starts at 1 (direct struct
    # construction; nothing in the writers makes the two diverge yet)
    L = Ledger(LedgerEvent[], OrderRecord[], Book(), 100, 1, 1, 1, 1, 1, Dict{Int,Int}())
    book = L.book
    g = mint_group!(L)
    _lg_fill!(L, _LG_PUT470, Short, Open,  2, 0.85, g; at=_LG_T_OPEN,  leg_id=1)   # id 100, seq 1
    _lg_fill!(L, _LG_PUT470, Long,  Close, 1, 0.40, g; at=_LG_T_CLOSE, leg_id=2)   # ids 101, 102; seq 2, 3
    record_fee!(L, 101, -65; effective_at=_LG_T_CLOSE, recorded_at=_LG_T_CLOSE)    # id 103, seq 4
    lot = only(open_lots(book))
    record_expiry!(L, lot; settlement_price=468.0, effective_at=_LG_EXPIRY_A, recorded_at=_LG_T_NEXT)   # id 104, seq 5
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
    @test_throws SequenceGap commit!(L, LedgerEvent[f])
    f = Fill(EventHeader(L.next_sequence, _LG_T_NEXT, _LG_T_NEXT, L.next_sequence), g, 3, 3, _LG_PUT465B, Short, Open, 1, 1.50, :cross_spread)
    @test_throws SequenceGap commit!(L, LedgerEvent[f])
    @test _lg_snapshot(L) == snap
    @test book == before
end

# The rejections pinned by the slice 1 review (findings 6.3, 7.1, 7.2 and
# 7.3), moved here from test_review_findings.jl, and the structure-atomicity
# promise (finding 6.4) that waits for slice 2.

@testset "ledger promise: validated batches enforce FIFO" begin
    L = Ledger(); book = L.book
    g = mint_group!(L)
    _lg_fill!(L, _LG_PUT470, Short, Open, 1, 0.85, g;
              at=_LG_T_OPEN, leg_id=1)
    _lg_fill!(L, _LG_PUT470, Short, Open, 1, 0.90, g;
              at=_LG_T_OPEN2, leg_id=2)
    snap, before = _lg_snapshot(L), deepcopy(book)
    close = Fill(_lg_hdr(L, 0), g, 3, L.next_execution, _LG_PUT470,
                 Long, Close, 1, 0.40, :cross_spread)
    match = Match(_lg_hdr(L, 1), g, 2, event_id(close), 1)   # names the newer lot
    @test_throws MatchMismatch commit!(L, LedgerEvent[close, match])
    @test _lg_snapshot(L) == snap
    @test book == before
end

@testset "ledger promise: consumption is not effective before its open" begin
    L = Ledger(); book = L.book
    g = mint_group!(L)
    _lg_fill!(L, _LG_PUT470, Short, Open, 1, 0.85, g;
              at=_LG_T_CLOSE, leg_id=1)
    snap, before = _lg_snapshot(L), deepcopy(book)
    @test_throws MatchMismatch record_fill!(
        L, Leg(_LG_PUT470, Long, 1, Close), g; price=0.40,
        effective_at=_LG_T_OPEN, recorded_at=_LG_T_CLOSE + Minute(1),
        order_leg_id=2, fill_rule=:cross_spread)
    @test _lg_snapshot(L) == snap
    @test book == before
end

@testset "ledger promise: expiry consumes the whole remaining lot" begin
    L = Ledger(); book = L.book
    g = mint_group!(L)
    _lg_fill!(L, _LG_PUT470, Short, Open, 2, 0.85, g;
              at=_LG_T_OPEN, leg_id=1)
    snap, before = _lg_snapshot(L), deepcopy(book)
    expiry = Expiry(_lg_hdr(L, 0, _LG_EXPIRY_A), g, 1, _LG_PUT470,
                    Short, 1, 468.0, CashSettled)
    @test_throws MatchMismatch commit!(L, LedgerEvent[expiry])
    @test _lg_snapshot(L) == snap
    @test book == before
end

@testset "ledger promise: expiry is not effective before contract expiry" begin
    L = Ledger(); book = L.book
    g = mint_group!(L)
    _lg_fill!(L, _LG_PUT470, Short, Open, 1, 0.85, g;
              at=_LG_T_OPEN, leg_id=1)
    snap, before = _lg_snapshot(L), deepcopy(book)
    expiry = Expiry(_lg_hdr(L, 0, _LG_T_OPEN2), g, 1, _LG_PUT470,
                    Short, 1, 468.0, CashSettled)
    @test_throws MatchMismatch commit!(L, LedgerEvent[expiry])
    @test _lg_snapshot(L) == snap
    @test book == before
end

@testset "ledger promise: a structure lands whole or not at all" begin
    # Was @test_broken until slice 2 added record_order!; a two-leg order whose
    # second leg has nothing to close leaves the ledger, the book and the group
    # counter untouched.
    L = Ledger(); book = L.book
    before_group = L.next_group
    order = Order(:invalid_structure, [
        Leg(_LG_PUT470, Short, 1, Open),
        Leg(_LG_CALL490, Long, 1, Close),
    ])
    @test begin
        threw_right = try
            record_order!(L, order; prices=[0.85, 0.40],
                          observations=[_lg_seen(0.85), _lg_seen(0.40)],
                          effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN, fill_rule=:cross_spread)
            false                                   # it must throw
        catch e
            e isa NothingToClose
        end
        threw_right && length(L) == 0 && book == Book() && L.next_group == before_group
    end
    @test isempty(L.orders)
    @test _lg_snapshot(L) == (0, 1, 1, 1, 1, 1, 1, 0)
end

# ---- record_order!, the structure-level writer (slice 2) ----------------

@testset "record_order!: a short strangle lands as one batch, the group minted inside" begin
    L, book = _lg_case_strangle_order()
    rec = only(L.orders)
    @test rec isa OrderRecord
    @test rec.order_id == 1 && rec.first_leg_id == 1 && rec.group == 1 && rec.known_to == 0
    @test rec.decided_at == _LG_T_OPEN
    @test rec.order.label == :strangle && rec.order.group === nothing
    @test length(rec.observations) == 2
    @test rec.observations[1].bid == 0.85 && rec.observations[2].ask == 1.10
    @test length(L) == 4
    f1, f2, fee1, fee2 = L.events
    @test f1 isa Fill && f2 isa Fill && fee1 isa Fee && fee2 isa Fee
    @test [event_id(e) for e in L.events] == [1, 2, 3, 4]
    @test [sequence(e) for e in L.events] == [1, 2, 3, 4]
    @test (f1.execution_id, f2.execution_id) == (1, 2)
    @test (f1.order_leg_id, f2.order_leg_id) == (1, 2)
    @test (fee1.source_id, fee2.source_id) == (1, 2)
    @test (fee1.amount, fee2.amount) == (-65, -65)
    @test f1.contract == _LG_PUT470 && f1.side == Short && f1.intent == Open && f1.price == 0.85
    @test f2.contract == _LG_CALL490 && f2.side == Short && f2.intent == Open && f2.price == 1.10
    @test group(f1) == group(f2) == 1
    @test all(effective_at(e) == _LG_T_OPEN && recorded_at(e) == _LG_T_OPEN for e in L.events)
    @test book.cash == 19370                     # 8500 + 11000 - 130
    @test L.next_group == 2
    @test L.next_leg_id == 3
    @test L.next_order_id == 2
    @test (L.next_id, L.next_sequence, L.next_execution) == (5, 5, 3)
    @test open_groups(book) == [1]
    @test length(lots(book, 1)) == 2
    @test book == book_as_known(L, 4) == book_effective(L, _LG_FAR)
    @test last_sequence(L) == 4
    @test check_join(L) === nothing
    _lg_check_book(book)
end

@testset "record_order!: a close order names its group and matches FIFO per leg" begin
    L, book = _lg_case_strangle_closed()
    @test length(L.orders) == 2
    rec = L.orders[2]
    @test rec.order_id == 2 && rec.first_leg_id == 3 && rec.group == 1 && rec.known_to == 4
    @test rec.decided_at == _LG_T_CLOSE
    @test rec.order.group == 1
    @test length(L) == 10
    kinds = [typeof(e) for e in L.events[5:10]]
    @test kinds == [Fill, Match, Fill, Match, Fee, Fee]
    c1, m1, c2, m2, fee1, fee2 = L.events[5:10]
    @test c1.order_leg_id == 3 && c2.order_leg_id == 4
    @test c1.execution_id == 3 && c2.execution_id == 4
    @test m1.open_fill_id == 1 && m1.close_fill_id == event_id(c1) && m1.quantity == 1
    @test m2.open_fill_id == 2 && m2.close_fill_id == event_id(c2) && m2.quantity == 1
    @test fee1.source_id == event_id(c1) && fee2.source_id == event_id(c2)
    @test book.cash == 9240                      # 19370 - 4000 - 6000 - 130
    @test isempty(open_lots(book))
    @test isempty(open_groups(book))
    trips = round_trips(L)
    @test [r.pnl for r in trips] == [4370, 4870]  # (8500 - 4000) - 65 - 65, (11000 - 6000) - 65 - 65
    @test sum(r.pnl for r in trips) == book.cash
    @test trade_pnl(L) ≈ [92.40]                 # one structure trade at the close instant
    @test n_opens(L) == 2 && n_closes(L) == 2
    @test book == book_as_known(L, 10) == book_effective(L, _LG_FAR)
    @test check_join(L) === nothing
    @test L.next_group == 2                      # a named group mints nothing
    @test (L.next_order_id, L.next_leg_id) == (3, 5)
end

@testset "record_order!: legs are planned against the book after the earlier legs" begin
    L = Ledger(); book = L.book
    order = Order(:open_and_trim, [Leg(_LG_PUT470, Short, 2, Open), Leg(_LG_PUT470, Long, 1, Close)])
    rec = record_order!(L, order; prices=[0.85, 0.40],
                        observations=[_lg_seen(0.85), _lg_seen(0.40)],
                        effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN, fill_rule=:cross_spread)
    @test length(L) == 3
    o, c, m = L.events
    @test o isa Fill && o.intent == Open && o.quantity == 2
    @test c isa Fill && c.intent == Close && c.quantity == 1
    @test m isa Match && m.open_fill_id == event_id(o) && m.close_fill_id == event_id(c) && m.quantity == 1
    @test open_lots(book) == [Lot(1, _LG_PUT470, Short, 1, 1, 0.85)]
    @test book.cash == 13000                     # 17000 - 4000
    @test rec.group == 1 && length(rec.observations) == 2
    @test check_join(L) === nothing
    @test [r.pnl for r in round_trips(L)] == [4500]
end

@testset "record_order!: zero fees book no Fee" begin
    L = Ledger(); book = L.book
    order = Order(:strangle, [Leg(_LG_PUT470, Short, 1, Open), Leg(_LG_CALL490, Short, 1, Open)])
    record_order!(L, order; prices=[0.85, 1.10],
                  observations=[_lg_seen(0.85), _lg_seen(1.10)],
                  effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN, fill_rule=:cross_spread)
    @test length(L) == 2
    @test all(e isa Fill for e in L.events)
    @test book.cash == 19500                     # 8500 + 11000
    # an explicit zero beside a cost: one Fee, on the costed leg only
    L = Ledger(); book = L.book
    record_order!(L, order; prices=[0.85, 1.10], fees=[0, -65],
                  observations=[_lg_seen(0.85), _lg_seen(1.10)],
                  effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN, fill_rule=:cross_spread)
    @test length(L) == 3
    @test L.events[3] isa Fee && L.events[3].source_id == 2 && L.events[3].amount == -65
    @test book.cash == 19435
end

@testset "record_order!: a structure lands whole or not at all, whichever leg fails" begin
    spx = ContractKey(Underlying("SPX"), 4700.0, _LG_EXPIRY_A, Put)
    early = ContractKey(_LG_SPY, 470.0, _LG_T_OPEN - Day(1), Put)     # expired before effective_at
    cases = [
        # (name, second leg, its price, the failure)
        ("ExceedsOpen",     Leg(_LG_PUT470, Long, 2, Close),  0.40,     ExceedsOpen),
        ("NonIntegralCash", Leg(_LG_CALL490, Short, 1, Open), 0.123456, NonIntegralCash),
        ("UnknownContract", Leg(spx, Short, 1, Open),          10.0,     UnknownContract),
        ("FillAfterExpiry", Leg(early, Short, 1, Open),        0.85,     FillAfterExpiry),
        ("InvalidPrice",    Leg(_LG_CALL490, Short, 1, Open), 0.0,      InvalidPrice),
        ("NothingToClose",  Leg(_LG_CALL490, Short, 1, Close), 0.60,    NothingToClose),   # same side as the lot
    ]
    for (name, leg2, price2, failure) in cases
        L, book = _lg_case_strangle_order()                          # group 1: short put, short call
        snap, before = _lg_snapshot(L), deepcopy(book)
        order = Order(Symbol(name), [Leg(_LG_PUT465B, Short, 1, Open), leg2]; group=1)
        kw = (prices=[1.50, price2], fees=[-65, -65],
              observations=[_lg_seen(1.50; at=_LG_T_CLOSE), _lg_seen(price2; at=_LG_T_CLOSE)],
              effective_at=_LG_T_CLOSE, recorded_at=_LG_T_CLOSE, fill_rule=:cross_spread)
        @test_throws failure record_order!(L, order; kw...)
        err = try record_order!(L, order; kw...); nothing catch e; e end
        @test err isa failure
        @test occursin(string(nameof(failure)), sprint(showerror, err))
        @test _lg_snapshot(L) == snap
        @test book == before
        @test length(L.orders) == 1 && L.next_group == 2 && L.next_order_id == 2 && L.next_leg_id == 3
        @test length(L) == 4 && book.cash == 19370
    end
    # the first leg of a fresh order may also be the one that fails: nothing is minted
    L, book = _lg_case_strangle_order()
    snap, before = _lg_snapshot(L), deepcopy(book)
    bad_first = Order(:bad_first, [Leg(_LG_PUT470, Long, 5, Close), Leg(_LG_CALL490, Long, 1, Close)]; group=1)
    @test_throws ExceedsOpen record_order!(L, bad_first; prices=[0.40, 0.60],
                                           observations=[_lg_seen(0.40), _lg_seen(0.60)],
                                           effective_at=_LG_T_CLOSE, recorded_at=_LG_T_CLOSE, fill_rule=:cross_spread)
    err = try record_order!(L, bad_first; prices=[0.40, 0.60],
                            observations=[_lg_seen(0.40), _lg_seen(0.60)],
                            effective_at=_LG_T_CLOSE, recorded_at=_LG_T_CLOSE, fill_rule=:cross_spread); nothing catch e; e end
    @test err isa ExceedsOpen && err.group == 1 && err.requested == 5 && err.available == 1
    @test _lg_snapshot(L) == snap && book == before
    # an opening order that fails mints no group
    L = Ledger(); book = L.book
    @test_throws InvalidPrice record_order!(L, Order(:bad, [Leg(_LG_PUT470, Short, 1, Open)]);
                                            prices=[-1.0], observations=[_lg_seen(0.85)],
                                            effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN, fill_rule=:cross_spread)
    @test _lg_snapshot(L) == (0, 1, 1, 1, 1, 1, 1, 0)
    @test book == Book()
end

@testset "record_order!: shape is an ArgumentError, the call is malformed" begin
    L, book = _lg_case_strangle_order()
    snap, before = _lg_snapshot(L), deepcopy(book)
    order = Order(:strangle, [Leg(_LG_PUT470, Short, 1, Open), Leg(_LG_CALL490, Short, 1, Open)])
    ok = (observations=[_lg_seen(0.85), _lg_seen(1.10)],
          effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN, fill_rule=:cross_spread)
    @test_throws ArgumentError record_order!(L, order; prices=[0.85], ok...)
    @test_throws ArgumentError record_order!(L, order; prices=[0.85, 1.10, 0.5], ok...)
    @test_throws ArgumentError record_order!(L, order; prices=[0.85, 1.10], fees=[-65], ok...)
    @test_throws ArgumentError record_order!(L, order; prices=[0.85, 1.10],
                                             observations=[_lg_seen(0.85)],
                                             effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN, fill_rule=:cross_spread)
    @test_throws ArgumentError record_order!(L, Order(:empty, Leg[]); prices=Float64[],
                                             observations=LegObservation[],
                                             effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN, fill_rule=:cross_spread)
    @test _lg_snapshot(L) == snap
    @test book == before
end

@testset "record_order!: a named group must be minted; an Open leg may join one" begin
    L, book = _lg_case_strangle_order()          # next_group == 2
    snap, before = _lg_snapshot(L), deepcopy(book)
    stray = Order(:stray, [Leg(_LG_PUT465B, Short, 1, Open)]; group=7)
    kw = (prices=[1.50], observations=[_lg_seen(1.50)],
          effective_at=_LG_T_OPEN, recorded_at=_LG_T_OPEN, fill_rule=:cross_spread)
    @test_throws DanglingReference record_order!(L, stray; kw...)
    err = try record_order!(L, stray; kw...); nothing catch e; e end
    @test err isa DanglingReference && err.field == :group && err.id == 7
    @test occursin("DanglingReference", sprint(showerror, err))
    @test_throws DanglingReference record_order!(L, Order(:zero, stray.legs; group=0); kw...)
    @test_throws DanglingReference record_order!(L, Order(:next, stray.legs; group=2); kw...)   # not yet minted
    @test _lg_snapshot(L) == snap
    @test book == before
    # an Open leg into group 1 adds a third lot to that structure
    rec = record_order!(L, Order(:add_leg, stray.legs; group=1); kw...)
    @test rec.group == 1 && rec.order_id == 2 && rec.first_leg_id == 3
    @test length(lots(book, 1)) == 3
    @test open_groups(book) == [1]
    @test L.next_group == 2
    @test book.cash == 19370 + 15000
    @test check_join(L) === nothing
end

@testset "record_order!: known_to defaults to the last sequence at the call" begin
    L, book = _lg_case_strangle_order()          # four events
    order = Order(:add, [Leg(_LG_PUT465B, Short, 1, Open)]; group=1)
    kw = (prices=[1.50], observations=[_lg_seen(1.50; at=_LG_T_OPEN2)],
          effective_at=_LG_T_OPEN2, recorded_at=_LG_T_OPEN2, fill_rule=:cross_spread)
    rec = record_order!(L, order; kw...)
    @test rec.known_to == 4
    # the engine passes what the tick saw before its first order: the second
    # order of a tick does not appear to have seen the first order's fills
    rec2 = record_order!(L, order; known_to=4, kw...)
    @test rec2.known_to == 4 && rec2.order_id == 3
    @test book_as_known(L, rec2.known_to) != book
    @test check_join(L) === nothing
end
