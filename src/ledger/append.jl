# The writers the engine calls. Each builds its events, validates the
# whole batch against the ledger and the book, appends it as one unit and
# applies it to the book. Nothing is appended if any check fails.
#
# Names avoid `Base.fill!` and `Base.append!`.

# ---- named failures --------------------------------------------------

"""
    NothingToClose

A `Close` leg on a `(group, contract)` with no open lot of the opposite
side.
"""
struct NothingToClose <: Exception
    group::Int
    contract::ContractKey
end

"""
    ExceedsOpen

Consumption of more than is open: a `Close` leg for more than its
`(group, contract)` holds on the opposite side, or a `Match` / `Expiry`
for more than its lot's remaining (zero once the lot is gone).
"""
struct ExceedsOpen <: Exception
    group::Int
    contract::ContractKey
    requested::Int
    available::Int
end

"""
    FillAfterExpiry

A fill whose effective time is after its contract's expiry. Thrown by
the `Fill` constructor, so no such value exists, and checked again on
append so that an event which bypasses the constructor (one loaded from
storage) is refused too.
"""
struct FillAfterExpiry <: Exception
    id::Int
    effective_at::DateTime
    expiry::DateTime
end

"""
    DanglingReference

A reference (`field`, holding `id`) that points to nothing: not to an
earlier event of the right kind (an `Open` fill for `open_fill_id`, a
fill for `close_fill_id` or `source_id`); an id the ledger never minted,
asked of [`event`](@ref) (`:event_id`); or, at construction, a `Fill`
whose `order_leg_id` or `execution_id` is not positive, since no order
leg or execution report carries such an id. Also thrown by a fold that
cannot find the lot an event consumes.
"""
struct DanglingReference <: Exception
    field::Symbol
    id::Int
end

"""
    MatchMismatch

A batch whose shape the invariants forbid, for one of these reasons: the
matches following a `Close` fill do not reference it and exhaust it
exactly; a match pairs a lot of the wrong group, contract or side, or
skips an older open lot (FIFO); the contract, side or group copied onto
an `Expiry` differ from its opening fill's, its `outcome` disagrees with
the intrinsic value at its settlement price, the expiry settles less
than the lot's remaining, or it is effective before its contract's
expiry; or an event is effective before an event it references (a match
or an expiry before its opening fill, a match at an instant other than
its closing fill's, a fee before its source fill). `id` is the offending
event and `reason` says which.
"""
struct MatchMismatch <: Exception
    id::Int
    reason::String
end

"""
    InvalidPrice

A `Fill` whose `price` is not finite and positive, or an `Expiry` whose
`settlement_price` is not finite and non-negative; `value` is the price
supplied. Thrown by the constructors, so no such value exists.
"""
struct InvalidPrice <: Exception
    value::Float64
end

"""
    RecordedOutOfOrder

An event recorded before it could have been: its `recorded_at` is
before its own effective time (a fact cannot be recorded before it is
true) or before the recorded time of the event preceding it in sequence
(the journal learns things in order, across batches and within one).
`bound` is the time `recorded_at` fell short of, whichever of the two.
"""
struct RecordedOutOfOrder <: Exception
    id::Int
    recorded_at::DateTime
    bound::DateTime
end

"""
    SequenceGap

A batch whose `sequence` (or `id`) does not continue the ledger's
`counter`: `expected` was the next value, `got` was supplied.
"""
struct SequenceGap <: Exception
    counter::Symbol
    expected::Int
    got::Int
end

"""
    NonPositiveQuantity

A `Leg`, `Fill`, `Match` or `Expiry` whose quantity is not a positive
integer. Thrown by the constructors, so no such value exists.
"""
struct NonPositiveQuantity <: Exception
    quantity::Int
end

"""
    NonIntegralCash

A price (or an intrinsic value) whose cash per contract,
`price * multiplier * 100`, is not a whole number of cents; `value` is
the offending product. Thrown by [`contract_cents`](@ref). `commit!`
computes every event's cash before anything lands, so no such event
reaches the ledger.
"""
struct NonIntegralCash <: Exception
    value::Float64
end

Base.showerror(io::IO, e::NothingToClose) =
    print(io, "NothingToClose: group ", e.group, " holds nothing to close on ", e.contract)
Base.showerror(io::IO, e::ExceedsOpen) =
    print(io, "ExceedsOpen: group ", e.group, " asked to consume ", e.requested,
          " of ", e.available, " open on ", e.contract)
Base.showerror(io::IO, e::FillAfterExpiry) =
    print(io, "FillAfterExpiry: fill ", e.id, " effective at ", e.effective_at,
          " is after its contract's expiry ", e.expiry)
Base.showerror(io::IO, e::DanglingReference) =
    print(io, "DanglingReference: ", e.field, " = ", e.id, " points to nothing ",
          "(no earlier event of the right kind, no minted id, or no positive join id)")
Base.showerror(io::IO, e::MatchMismatch) =
    print(io, "MatchMismatch: event ", e.id, ": ", e.reason)
Base.showerror(io::IO, e::SequenceGap) =
    print(io, "SequenceGap: expected ", e.counter, " ", e.expected, ", got ", e.got)
Base.showerror(io::IO, e::NonPositiveQuantity) =
    print(io, "NonPositiveQuantity: quantity must be a positive integer, got ", e.quantity)
Base.showerror(io::IO, e::NonIntegralCash) =
    print(io, "NonIntegralCash: ", e.value, " cents per contract is not a whole number of cents")
Base.showerror(io::IO, e::InvalidPrice) =
    print(io, "InvalidPrice: ", e.value, " is not a finite price ",
          "(positive for a fill, non-negative for a settlement)")
Base.showerror(io::IO, e::RecordedOutOfOrder) =
    print(io, "RecordedOutOfOrder: event ", e.id, " is recorded at ", e.recorded_at,
          ", before ", e.bound, " (its effective time, or the recorded time of the event before it)")

# ---- writers ---------------------------------------------------------

"""
    mint_group!(L::Ledger) -> Int

A fresh group id for an opening order.
"""
function mint_group!(L::Ledger)::Int
    g = L.next_group
    L.next_group += 1
    return g
end

# The header of the k-th event (from 0) of a batch about to be committed.
_header(L::Ledger, k::Int, effective_at::DateTime, recorded_at::DateTime)::EventHeader =
    EventHeader(L.next_id + k, effective_at, recorded_at, L.next_sequence + k)

"""
    record_fill!(L, book, leg::Leg, group::Int;
                 price, effective_at, recorded_at, order_leg_id, fill_rule)
        -> Vector{LedgerEvent}

Book one executed leg into `group` at `price` per share. An `Open` leg
appends one `Fill`; a `Close` leg appends one `Fill` followed by its
`Match`es, FIFO within `(group, contract)` among the open lots of the
opposite side. Throws [`NothingToClose`](@ref) when no such lot exists
and [`ExceedsOpen`](@ref) when they hold less than the leg's quantity;
the batch then goes through [`commit!`](@ref). Returns the events
appended.
"""
function record_fill!(L::Ledger, book::Book, leg::Leg, group::Int;
                      price::Real, effective_at::DateTime, recorded_at::DateTime,
                      order_leg_id::Int, fill_rule::Symbol)::Vector{LedgerEvent}
    fill = Fill(_header(L, 0, effective_at, recorded_at), group, order_leg_id,
                L.next_execution, leg.contract, leg.side, leg.intent, leg.quantity,
                price, fill_rule)
    batch = LedgerEvent[fill]
    if leg.intent == Close
        candidates = Lot[l for l in _lots_at(book, group, leg.contract) if l.side != leg.side]
        isempty(candidates) && throw(NothingToClose(group, leg.contract))
        available = sum(l.remaining for l in candidates)
        available >= leg.quantity ||
            throw(ExceedsOpen(group, leg.contract, leg.quantity, available))
        left = leg.quantity
        for lot in candidates
            left == 0 && break
            q = min(left, lot.remaining)
            push!(batch, Match(_header(L, length(batch), effective_at, recorded_at),
                               group, lot.open_fill_id, event_id(fill), q))
            left -= q
        end
    end
    commit!(L, book, batch)
    return batch
end

# The outcome an expiry carries: `Worthless` when intrinsic is zero,
# `CashSettled` otherwise. The writer derives it here and `_validate`
# checks a hand-built expiry against the same rule.
_outcome(contract::ContractKey, settlement_price::Real)::ExpiryOutcome =
    intrinsic(contract, settlement_price) == 0.0 ? Worthless : CashSettled

"""
    record_expiry!(L, book, lot::Lot; settlement_price, effective_at, recorded_at) -> Expiry

Settle the whole remaining quantity of `lot` at `settlement_price`. The
outcome is `Worthless` when intrinsic is zero and `CashSettled`
otherwise. `effective_at` is the settlement instant, at or after the
contract's expiry; `recorded_at` the tick that booked it.
"""
function record_expiry!(L::Ledger, book::Book, lot::Lot;
                        settlement_price::Real, effective_at::DateTime,
                        recorded_at::DateTime)::Expiry
    outcome = _outcome(lot.contract, settlement_price)
    e = Expiry(_header(L, 0, effective_at, recorded_at), lot.group, lot.open_fill_id,
               lot.contract, lot.side, lot.remaining, settlement_price, outcome)
    commit!(L, book, LedgerEvent[e])
    return e
end

"""
    record_fee!(L, book, source_id::Int, amount::Integer; effective_at, recorded_at) -> Fee

Book a signed `amount` of whole USD cents (a cost is negative) caused by
the fill `source_id`. Throws [`DanglingReference`](@ref) when `source_id`
is not a fill in `L`.
"""
function record_fee!(L::Ledger, book::Book, source_id::Int, amount::Integer;
                     effective_at::DateTime, recorded_at::DateTime)::Fee
    e = Fee(_header(L, 0, effective_at, recorded_at), source_id, amount)
    commit!(L, book, LedgerEvent[e])
    return e
end

"""
    commit!(L, book, batch::AbstractVector{<:LedgerEvent}) -> nothing

The single validated write path. Checks `batch` against `L` and `book`,
then appends it as one unit and applies it to `book`; every event's
contract facts come from the table, so the incremental book and the
replays fold the same numbers. The checks, each with its named failure:

- `id` and `sequence` continue the ledger's counters ([`SequenceGap`](@ref));
- every event is recorded at or after its effective time, and recorded
  times are nondecreasing along sequence, across the batch boundary and
  within the batch ([`RecordedOutOfOrder`](@ref));
- every event's cash is whole cents ([`NonIntegralCash`](@ref); an
  unlisted underlying is [`UnknownContract`](@ref));
- a fill is effective at or before its contract's expiry ([`FillAfterExpiry`](@ref),
  already refused by the `Fill` constructor);
- every reference points to an earlier event of the right kind ([`DanglingReference`](@ref));
- every reference points backward in effective time: a match's opening
  fill is effective at or before its closing fill and the match at the
  closing fill's instant, an expiry's opening fill at or before the
  expiry, a fee's source fill at or before the fee ([`MatchMismatch`](@ref));
- the matches immediately following a `Close` fill reference it, pair
  lots of its group and contract on the opposite side, each consume the
  oldest still-eligible lot (FIFO), and sum to its quantity; an
  `Expiry`'s copied contract, side and group equal its opening fill's,
  its outcome agrees with the intrinsic value at its settlement price,
  it settles the whole remaining lot, and it is effective at or after
  the contract's expiry ([`MatchMismatch`](@ref));
- consumption never exceeds a lot's remaining ([`ExceedsOpen`](@ref)).

Nothing is appended if any check fails.
"""
function commit!(L::Ledger, book::Book, batch::AbstractVector{<:LedgerEvent})::Nothing
    isempty(batch) && return nothing
    _validate(L, book, batch)
    for e in batch
        push!(L.events, e)
        L.index[event_id(e)] = length(L.events)
        L.next_id += 1
        L.next_sequence += 1
        if e isa Fill
            L.next_execution = max(L.next_execution, e.execution_id + 1)
        end
        apply!(book, e)
    end
    return nothing
end

# ---- validation ------------------------------------------------------

# An event by id: the batch validated so far (`seen`) first, then the
# ledger; `nothing` when neither has it.
function _lookup(L::Ledger, seen, id::Int)
    seen !== nothing && haskey(seen, id) && return seen[id]
    haskey(L.index, id) && return event(L, id)
    return nothing
end

function _fill_ref(L::Ledger, seen, id::Int, field::Symbol)::Fill
    e = _lookup(L, seen, id)
    e isa Fill || throw(DanglingReference(field, id))
    return e
end

function _opening_fill(L::Ledger, seen, id::Int, field::Symbol)::Fill
    f = _fill_ref(L, seen, id, field)
    f.intent == Open || throw(DanglingReference(field, id))
    return f
end

# What this batch may still consume of the lot opened by `o`: the whole
# fill when this batch opened it, else the book's remaining, minus what
# earlier events of the batch already consumed.
function _available(book::Book, opened::Vector{Fill}, consumed::Dict{Int,Int}, o::Fill)::Int
    oid = event_id(o)
    held = any(f -> event_id(f) == oid, opened) ? o.quantity :
           _remaining(book, o.group, o.contract, oid)
    return held - get(consumed, oid, 0)
end

# The lot a close of `c` must consume next, FIFO: the book's lots for its
# (group, contract) in their order, then the fills this batch opened, in
# batch order; opposite side, with something left after what the batch
# already consumed. `nothing` when no lot is eligible.
function _first_eligible(book::Book, opened::Vector{Fill}, consumed::Dict{Int,Int},
                         c::Fill)::Union{Nothing,Int}
    for l in _lots_at(book, c.group, c.contract)
        l.side != c.side || continue
        l.remaining - get(consumed, l.open_fill_id, 0) > 0 && return l.open_fill_id
    end
    for f in opened
        (f.group == c.group && f.contract == c.contract && f.side != c.side) || continue
        f.quantity - get(consumed, event_id(f), 0) > 0 && return event_id(f)
    end
    return nothing
end

function _check_match(L::Ledger, book::Book, seen, opened, consumed, m::Match, c::Fill)::Nothing
    m.group == c.group ||
        throw(MatchMismatch(event_id(m), "group differs from closing fill $(event_id(c))"))
    o = _opening_fill(L, seen, m.open_fill_id, :open_fill_id)
    (o.group == c.group && o.contract == c.contract && o.side != c.side) ||
        throw(MatchMismatch(event_id(m), "opening fill $(event_id(o)) is not an " *
                            "opposite-side lot of the closing fill's group and contract"))
    effective_at(o) <= effective_at(c) ||
        throw(MatchMismatch(event_id(m), "opening fill $(event_id(o)) is effective after " *
                            "closing fill $(event_id(c))"))
    effective_at(m) == effective_at(c) ||
        throw(MatchMismatch(event_id(m), "match is not effective at the instant of " *
                            "closing fill $(event_id(c))"))
    available = _available(book, opened, consumed, o)
    m.quantity <= available ||
        throw(ExceedsOpen(o.group, o.contract, m.quantity, available))
    _first_eligible(book, opened, consumed, c) == event_id(o) ||
        throw(MatchMismatch(event_id(m), "match skips an older open lot"))
    consumed[event_id(o)] = get(consumed, event_id(o), 0) + m.quantity
    return nothing
end

function _check_expiry(L::Ledger, book::Book, seen, opened, consumed, e::Expiry)::Nothing
    o = _opening_fill(L, seen, e.open_fill_id, :open_fill_id)
    (e.contract == o.contract && e.side == o.side && e.group == o.group) ||
        throw(MatchMismatch(event_id(e),
            "copied contract, side or group differ from opening fill $(event_id(o))"))
    e.outcome == _outcome(e.contract, e.settlement_price) ||
        throw(MatchMismatch(event_id(e),
            "outcome $(e.outcome) disagrees with an intrinsic value of " *
            "$(intrinsic(e.contract, e.settlement_price)) at settlement $(e.settlement_price)"))
    effective_at(o) <= effective_at(e) ||
        throw(MatchMismatch(event_id(e),
            "expiry is effective before its opening fill $(event_id(o))"))
    effective_at(e) >= e.contract.expiry ||
        throw(MatchMismatch(event_id(e), "expiry effective before the contract's expiry"))
    available = _available(book, opened, consumed, o)
    e.quantity <= available ||
        throw(ExceedsOpen(o.group, o.contract, e.quantity, available))
    e.quantity == available ||
        throw(MatchMismatch(event_id(e),
            "expiry of $(e.quantity) leaves $(available - e.quantity) open"))
    consumed[event_id(o)] = get(consumed, event_id(o), 0) + e.quantity
    return nothing
end

function _validate(L::Ledger, book::Book, batch::AbstractVector{<:LedgerEvent})::Nothing
    last_recorded = isempty(L) ? typemin(DateTime) : recorded_at(L.events[end])
    for (k, e) in enumerate(batch)
        want_id  = L.next_id + k - 1
        want_seq = L.next_sequence + k - 1
        event_id(e) == want_id || throw(SequenceGap(:id, want_id, event_id(e)))
        sequence(e) == want_seq || throw(SequenceGap(:sequence, want_seq, sequence(e)))
        # A fact cannot be recorded before it is true, and the journal
        # learns things in sequence order: recorded time never steps back.
        recorded_at(e) >= effective_at(e) ||
            throw(RecordedOutOfOrder(event_id(e), recorded_at(e), effective_at(e)))
        recorded_at(e) >= last_recorded ||
            throw(RecordedOutOfOrder(event_id(e), recorded_at(e), last_recorded))
        last_recorded = recorded_at(e)
    end
    # Every event's cash must resolve to whole cents before anything lands
    # (NonIntegralCash; UnknownContract for an unlisted underlying).
    foreach(cash, batch)
    seen     = Dict{Int,LedgerEvent}()   # batch events validated so far
    opened   = Fill[]                    # Open fills of this batch, in batch order
    consumed = Dict{Int,Int}()           # consumption by this batch, per opening fill
    n = length(batch)
    k = 1
    while k <= n
        e = batch[k]
        if e isa Fill
            effective_at(e) <= e.contract.expiry ||
                throw(FillAfterExpiry(event_id(e), effective_at(e), e.contract.expiry))
            seen[event_id(e)] = e
            k += 1
            if e.intent == Open
                push!(opened, e)
            else
                total = 0
                while k <= n && batch[k] isa Match && batch[k].close_fill_id == event_id(e)
                    m = batch[k]
                    _check_match(L, book, seen, opened, consumed, m, e)
                    seen[event_id(m)] = m
                    total += m.quantity
                    k += 1
                end
                total == e.quantity || throw(MatchMismatch(event_id(e),
                    "matches consume $total of a close of $(e.quantity)"))
            end
        elseif e isa Match
            _fill_ref(L, seen, e.close_fill_id, :close_fill_id)
            throw(MatchMismatch(event_id(e), "match does not immediately follow its closing fill"))
        elseif e isa Expiry
            _check_expiry(L, book, seen, opened, consumed, e)
            seen[event_id(e)] = e
            k += 1
        else                                       # Fee
            source = _fill_ref(L, seen, e.source_id, :source_id)
            effective_at(source) <= effective_at(e) ||
                throw(MatchMismatch(event_id(e),
                    "fee is effective before its source fill $(event_id(source))"))
            seen[event_id(e)] = e
            k += 1
        end
    end
    return nothing
end
