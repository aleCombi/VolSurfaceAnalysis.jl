# These tests pin the ledger slice 1 review findings and are expected to stay
# red until their corresponding fixes land:
#   public writes replay consistently                         -> finding 6.1
#   equal-time accepted writes replay without throwing        -> finding 6.2
#   validated batches enforce FIFO                            -> finding 6.3
#   orders are committed atomically                           -> finding 6.4
#   consumption cannot be effective before its open           -> finding 7.1
#   expiry consumes the whole remaining lot                   -> finding 7.2
#   expiry cannot be effective before contract expiry         -> finding 7.3
#   incremental and effective books are exactly equal         -> finding 7.4
# The fix round of 2026-09-12 turned seven of them green. The atomic-order
# test is @test_broken until slice 2 adds record_order!. Cash is whole cents
# inside the ledger, so the exact-equality test holds by construction.

# Keep hand-built batch tests valid after commit!'s pinned-spec overload is
# removed as proposed by finding 6.1.
function _lg_commit_review!(L, book, batch)
    if applicable(commit!, L, book, batch, _LG_SPEC)
        return commit!(L, book, batch, _LG_SPEC)
    end
    return commit!(L, book, batch)
end

@testset "ledger promise: every public write replays to its incremental book" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    fill = Fill(_lg_hdr(L, 0, _LG_T_OPEN), g, 1, 1, _LG_PUT470,
                Short, Open, 1, 0.85, :cross_spread)
    spec10 = ContractSpec(10.0, American, PMSettled, Physical)
    ok = try
        commit!(L, book, LedgerEvent[fill], spec10)
        book == book_effective(L, _LG_FAR) && book == book_as_known(L, 1)
    catch e
        e isa MethodError
    end
    @test ok # today: incremental 8.5, both replays 85.0
end

@testset "ledger promise: accepted equal-time lifecycle events replay safely" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 1, 0.85, g;
              at=_LG_EXPIRY_A, leg_id=1)
    lot = only(open_lots(book))
    record_expiry!(L, book, lot; settlement_price=468.0,
                   effective_at=_LG_EXPIRY_A, recorded_at=_LG_T_NEXT)
    @test book_effective(L, _LG_EXPIRY_A) == book
    @test book.cash == -11500 # +0.85*100 - (470-468)*100 = 85 - 200
end

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
    match = Match(_lg_hdr(L, 1), g, 2, event_id(close), 1)
    @test_throws MatchMismatch _lg_commit_review!(L, book, LedgerEvent[close, match])
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
    @test_throws MatchMismatch _lg_commit_review!(L, book, LedgerEvent[expiry])
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
    @test_throws MatchMismatch _lg_commit_review!(L, book, LedgerEvent[expiry])
    @test _lg_snapshot(L) == snap
    @test book == before
end

@testset "ledger promise: incremental book exactly equals effective replay" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    later_put = _lg_put(470.0; expiry=_LG_T_NEXT + Day(4))
    _lg_fill!(L, book, _LG_PUT470, Short, Open, 1, 0.07, g;
              at=_LG_T_OPEN, leg_id=1)
    _lg_fill!(L, book, later_put, Short, Open, 1, 0.07, g;
              at=_LG_T_NEXT, leg_id=2)
    lot = only(l for l in open_lots(book) if l.contract == _LG_PUT470)
    record_expiry!(L, book, lot; settlement_price=469.83,
                   effective_at=_LG_EXPIRY_A,
                   recorded_at=_LG_T_NEXT + Minute(1))
    @test book == book_effective(L, _LG_FAR)
end
