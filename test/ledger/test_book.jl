# The book: apply! per kind, the accessors, both replays. Cases 4, 7, 8.

@testset "book: a fresh book is empty" begin
    b = Book()
    @test b.cash == 0.0
    @test isempty(open_lots(b))
    @test isempty(open_groups(b))
    @test isempty(lots(b, 1))
    @test b == Book()
end

@testset "book: apply! per kind" begin
    b = Book()
    hdr(i, t=_LG_T_OPEN) = EventHeader(i, t, t, i)
    f_open = Fill(hdr(1), 1, 1, 1, _LG_PUT470, Short, Open, 3, 0.85, :cross_spread)
    @test apply!(b, f_open, _LG_SPEC) === b
    @test b.cash ≈ 255.0
    @test open_lots(b) == [Lot(1, _LG_PUT470, Short, 1, 3, 0.85)]
    @test open_groups(b) == [1]

    f_close = Fill(hdr(2, _LG_T_CLOSE), 1, 2, 2, _LG_PUT470, Long, Close, 1, 0.40, :cross_spread)
    apply!(b, f_close, _LG_SPEC)
    @test b.cash ≈ 215.0
    @test open_lots(b) == [Lot(1, _LG_PUT470, Short, 1, 3, 0.85)]   # a close fill touches no lot

    apply!(b, Match(hdr(3, _LG_T_CLOSE), 1, 1, 2, 1), _LG_SPEC)
    @test b.cash ≈ 215.0                                             # a match moves no cash
    @test open_lots(b) == [Lot(1, _LG_PUT470, Short, 1, 2, 0.85)]

    apply!(b, Fee(hdr(4, _LG_T_CLOSE), 2, -1.30), _LG_SPEC)
    @test b.cash ≈ 213.70

    apply!(b, Expiry(hdr(5, _LG_EXPIRY_A), 1, 1, _LG_PUT470, Short, 2, 468.0, CashSettled), _LG_SPEC)
    @test b.cash ≈ -186.30                       # 213.70 - 2 * 2 * 100
    @test isempty(open_lots(b))
    @test isempty(open_groups(b))
    @test isempty(b.lots)                        # the emptied key is dropped
end

@testset "book: the one-argument apply! resolves the spec from the contract" begin
    b = Book()
    f = Fill(EventHeader(1, _LG_T_OPEN, _LG_T_OPEN, 1), 1, 1, 1, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread)
    @test apply!(b, f) === b
    @test b.cash ≈ 85.0
    apply!(b, Fee(EventHeader(2, _LG_T_OPEN, _LG_T_OPEN, 2), 1, -0.65))
    @test b.cash ≈ 84.35
end

@testset "book: consuming a lot the book does not hold is a named failure" begin
    b = Book()
    @test_throws DanglingReference apply!(b, Match(EventHeader(1, _LG_T_CLOSE, _LG_T_CLOSE, 1), 1, 99, 98, 1), _LG_SPEC)
    apply!(b, Fill(EventHeader(2, _LG_T_OPEN, _LG_T_OPEN, 2), 1, 1, 1, _LG_PUT470, Short, Open, 1, 0.85, :cross_spread), _LG_SPEC)
    @test_throws ExceedsOpen apply!(b, Match(EventHeader(3, _LG_T_CLOSE, _LG_T_CLOSE, 3), 1, 2, 98, 2), _LG_SPEC)
    @test_throws DanglingReference apply!(b, Match(EventHeader(3, _LG_T_CLOSE, _LG_T_CLOSE, 3), 2, 2, 98, 1), _LG_SPEC)  # wrong group
    @test open_lots(b) == [Lot(1, _LG_PUT470, Short, 2, 1, 0.85)]
end

@testset "book: open_lots, lots and open_groups, ordered by opening fill" begin
    L, book = _lg_case_two_groups()
    @test open_groups(book) == [1]
    @test open_lots(book) == [Lot(1, _LG_CALL490, Short, 1, 1, 1.10)]
    @test lots(book, 1) == [Lot(1, _LG_CALL490, Short, 1, 1, 1.10)]
    @test isempty(lots(book, 2))

    L, book = _lg_case_open_at_end()
    @test open_lots(book) == [Lot(1, _LG_PUT470, Short, 1, 1, 0.85)]

    # several groups and contracts: ascending groups, lots by opening fill
    L, book = Ledger(), Book()
    g1 = mint_group!(L); g2 = mint_group!(L); g3 = mint_group!(L)
    _lg_fill!(L, book, _LG_CALL490, Short, Open, 1, 1.10, g3; leg_id=1)
    _lg_fill!(L, book, _LG_PUT470,  Short, Open, 1, 0.85, g1; leg_id=2)
    _lg_fill!(L, book, _LG_PUT465B, Short, Open, 2, 1.50, g1; leg_id=3)
    @test open_groups(book) == [1, 3]
    @test [l.open_fill_id for l in open_lots(book)] == [1, 2, 3]
    @test [l.open_fill_id for l in lots(book, g1)] == [2, 3]
    @test isempty(lots(book, g2))
end

@testset "book: case 4, mixed expiries in one group" begin
    L, book = _lg_case_mixed_expiries()
    @test book.cash ≈ 35.0
    @test open_lots(book) == [Lot(1, _LG_PUT465B, Short, 2, 1, 1.50)]
    x = L.events[end]
    @test x isa Expiry
    @test x.outcome == CashSettled
    @test x.quantity == 1

    before = book_effective(L, _LG_EXPIRY_A - Second(1))
    @test before.cash ≈ 235.0
    @test open_lots(before) == [Lot(1, _LG_PUT470, Short, 1, 1, 0.85),
                                Lot(1, _LG_PUT465B, Short, 2, 1, 1.50)]

    at = book_effective(L, _LG_EXPIRY_A)
    @test at.cash ≈ 35.0
    @test open_lots(at) == [Lot(1, _LG_PUT465B, Short, 2, 1, 1.50)]
    @test at == book
end

@testset "book: case 8, known versus true" begin
    L, book = _lg_case_mixed_expiries()
    x = L.events[end]
    @test recorded_at(x) > effective_at(x)
    known = book_as_known(L, sequence(x) - 1)       # the sequence just before it
    @test length(open_lots(known)) == 2
    @test known.cash ≈ 235.0
    @test book_as_known(L, sequence(x)) == book
    true_at = book_effective(L, effective_at(x))    # its effective instant
    @test length(open_lots(true_at)) == 1
    @test true_at.cash ≈ 35.0
    @test book_effective(L, recorded_at(x)) == book
    @test book_effective(L, _LG_T_OPEN) == book_as_known(L, 1)
    @test book_as_known(L, 0) == Book()
end

@testset "book: lifecycle sorts before fills at an equal effective time" begin
    x = Expiry(EventHeader(9, _LG_EXPIRY_A, _LG_T_NEXT, 9), 1, 1, _LG_PUT470, Short, 1, 468.0, CashSettled)
    f = Fill(EventHeader(8, _LG_EXPIRY_A, _LG_EXPIRY_A, 8), 1, 1, 1, _LG_CALL490, Short, Open, 1, 1.10, :cross_spread)
    @test VolSurfaceAnalysis._priority(x) < VolSurfaceAnalysis._priority(f)
    @test VolSurfaceAnalysis._priority(Fee(EventHeader(1, _LG_T_OPEN, _LG_T_OPEN, 1), 1, -1.0)) ==
          VolSurfaceAnalysis._priority(f)
end

@testset "book: case 7, incremental equals replay in every case" begin
    for (name, build) in _LG_CASES
        L, book = build()
        @test book == book_effective(L, _LG_FAR)
        @test book == book_as_known(L, sequence(L.events[end]))
        stepped = Book()
        for e in L.events
            apply!(stepped, e)
        end
        @test stepped == book
        @test book.cash ≈ sum(cash(e) for e in L.events)
    end
end
