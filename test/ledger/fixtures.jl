# Shared fixtures for the ledger suites. Constants are prefixed `_LG_` and
# helpers `_lg_` because runtests.jl includes every suite into one module.
# Every scenario is a hand-built ledger; the expected numbers in the
# suites are literals worked out by hand from the inputs listed here, in
# whole USD cents: one contract at a per-share price p moves p * 100 * 100
# cents, so 0.85 per share is 8500 cents.

const _LG_SPY  = Underlying("SPY")
const _LG_SPEC = ContractSpec(100, American, PMSettled, Physical)

const _LG_EXPIRY_A = DateTime(2024, 1, 19, 21, 0)   # Fri 2024-01-19, 16:00 ET
const _LG_EXPIRY_B = DateTime(2024, 1, 26, 21, 0)   # Fri 2024-01-26, 16:00 ET
const _LG_T_OPEN   = DateTime(2024, 1, 16, 14, 35)
const _LG_T_OPEN2  = DateTime(2024, 1, 16, 15, 35)
const _LG_T_CLOSE  = DateTime(2024, 1, 18, 20, 0)
const _LG_T_NEXT   = DateTime(2024, 1, 22, 14, 35)  # first tick after expiry A
const _LG_FAR      = DateTime(2030, 1, 1)

_lg_put(strike; expiry=_LG_EXPIRY_A)  = ContractKey(_LG_SPY, strike, expiry, Put)
_lg_call(strike; expiry=_LG_EXPIRY_A) = ContractKey(_LG_SPY, strike, expiry, Call)

const _LG_PUT470  = _lg_put(470.0)
const _LG_CALL490 = _lg_call(490.0)
const _LG_PUT465B = _lg_put(465.0; expiry=_LG_EXPIRY_B)

# Book one leg, recorded when it is effective.
function _lg_fill!(L, book, contract, side, intent, qty, price, group;
                   at=_LG_T_OPEN, leg_id=1, rule=:cross_spread)
    record_fill!(L, book, Leg(contract, side, qty, intent), group;
                 price=price, effective_at=at, recorded_at=at,
                 order_leg_id=leg_id, fill_rule=rule)
end

# Case 1. Short 1 put at 0.85, buy to close at 0.40.
#   cash = +8500 - 4000 = 4500; one match; one round trip of +4500; book empty.
function _lg_case_round_trip()
    L, book = Ledger(), Book()
    g = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470, Short, Open,  1, 0.85, g; at=_LG_T_OPEN,  leg_id=1)
    _lg_fill!(L, book, _LG_PUT470, Long,  Close, 1, 0.40, g; at=_LG_T_CLOSE, leg_id=2)
    return (L, book)
end

# Case 2. Lots of 2 at 0.85 and 1 at 0.90 on one contract in one group;
# close 3 at 0.40.
#   cash = 17000 + 9000 - 12000 = 14000; matches of 2 and 1;
#   trips (8500 - 4000) * 2 = +9000 and (9000 - 4000) * 1 = +5000.
function _lg_case_split()
    L, book = Ledger(), Book()
    g = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470, Short, Open,  2, 0.85, g; at=_LG_T_OPEN,  leg_id=1)
    _lg_fill!(L, book, _LG_PUT470, Short, Open,  1, 0.90, g; at=_LG_T_OPEN2, leg_id=2)
    _lg_fill!(L, book, _LG_PUT470, Long,  Close, 3, 0.40, g; at=_LG_T_CLOSE, leg_id=3)
    return (L, book)
end

# Case 3. Groups 1 and 2 each short 1 of the same call (at 1.10 and
# 1.20); close group 2 only, at 0.70.
#   cash = 11000 + 12000 - 7000 = 16000; group 1 untouched; one trip of +5000.
function _lg_case_two_groups()
    L, book = Ledger(), Book()
    g1 = mint_group!(L)
    g2 = mint_group!(L)
    _lg_fill!(L, book, _LG_CALL490, Short, Open,  1, 1.10, g1; at=_LG_T_OPEN,  leg_id=1)
    _lg_fill!(L, book, _LG_CALL490, Short, Open,  1, 1.20, g2; at=_LG_T_OPEN2, leg_id=2)
    _lg_fill!(L, book, _LG_CALL490, Long,  Close, 1, 0.70, g2; at=_LG_T_CLOSE, leg_id=3)
    return (L, book)
end

# Case 4. One group, two lots with different expiries: short 1 put 470
# (expiry A) at 0.85 and short 1 put 465 (expiry B) at 1.50. Expire the
# first at settlement 468 (intrinsic 2.00 per share, 20000 per contract),
# effective at expiry A and recorded at the next tick.
#   cash = 8500 + 15000 - 20000 = 3500; the put-465 lot stays open.
function _lg_case_mixed_expiries()
    L, book = Ledger(), Book()
    g = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470,  Short, Open, 1, 0.85, g; at=_LG_T_OPEN,  leg_id=1)
    _lg_fill!(L, book, _LG_PUT465B, Short, Open, 1, 1.50, g; at=_LG_T_OPEN2, leg_id=2)
    lot = only(l for l in lots(book, g) if l.contract == _LG_PUT470)
    record_expiry!(L, book, lot; settlement_price=468.0,
                   effective_at=_LG_EXPIRY_A, recorded_at=_LG_T_NEXT)
    return (L, book)
end

# Case 5. Case 2 plus a fee of -130 (1.30 USD) on the closing fill.
#   shares by cumulative rounding over the matches of 2 then 1:
#   round(-130 * 2/3) = -87, then -130 - (-87) = -43;
#   trips 9000 - 87 = 8913 and 5000 - 43 = 4957; cash = 14000 - 130 = 13870.
function _lg_case_fees()
    L, book = _lg_case_split()
    close_id = event_id(only(e for e in L.events if e isa Fill && e.intent == Close))
    record_fee!(L, book, close_id, -130; effective_at=_LG_T_CLOSE, recorded_at=_LG_T_CLOSE)
    return (L, book)
end

# Case 6. One group: short 1 put 470 at 0.85, never closed, and short 1
# call 490 at 1.10, closed at 0.60.
#   cash = 8500 + 11000 - 6000 = 13500; one trip of +5000; the put lot stays open.
function _lg_case_open_at_end()
    L, book = Ledger(), Book()
    g = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470,  Short, Open,  1, 0.85, g; at=_LG_T_OPEN,  leg_id=1)
    _lg_fill!(L, book, _LG_CALL490, Short, Open,  1, 1.10, g; at=_LG_T_OPEN,  leg_id=2)
    _lg_fill!(L, book, _LG_CALL490, Long,  Close, 1, 0.60, g; at=_LG_T_CLOSE, leg_id=3)
    return (L, book)
end

const _LG_CASES = [
    "full round trip"             => _lg_case_round_trip,
    "close split across lots"     => _lg_case_split,
    "two groups on one contract"  => _lg_case_two_groups,
    "mixed expiries in one group" => _lg_case_mixed_expiries,
    "fees across a partial close" => _lg_case_fees,
    "open at window end"          => _lg_case_open_at_end,
]

# The k-th header (from 0) of a hand-built batch, minted the way the
# writers mint theirs; and the ledger state a failed batch must leave.
_lg_hdr(L, k, t=_LG_T_CLOSE) = EventHeader(L.next_id + k, t, t, L.next_sequence + k)
_lg_snapshot(L) = (length(L), L.next_id, L.next_sequence, L.next_group, L.next_execution)

# The structural promises of a book, checked after any fold: a lot with
# nothing remaining is gone (and so is an emptied key), every lot sits
# under the key of its own group and contract, and the lots of one key
# are in opening order (opening fill ids are minted in sequence, so
# ascending id is FIFO order). Records into the enclosing testset.
function _lg_check_book(book::Book)
    for ((g, c), v) in book.lots
        @test !isempty(v)
        for l in v
            @test l.remaining > 0
            @test l.group == g && l.contract == c
        end
        ids = [l.open_fill_id for l in v]
        @test issorted(ids) && allunique(ids)
    end
    return nothing
end
