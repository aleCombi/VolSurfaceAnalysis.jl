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

A fill whose effective time is after its contract's expiry.
"""
struct FillAfterExpiry <: Exception
    id::Int
    effective_at::DateTime
    expiry::DateTime
end

"""
    DanglingReference

A reference (`field`, holding `id`) that does not point to an earlier
event of the right kind: an `Open` fill for `open_fill_id`, a fill for
`close_fill_id` or `source_id`. Also thrown by a fold that cannot find
the lot an event consumes.
"""
struct DanglingReference <: Exception
    field::Symbol
    id::Int
end

"""
    MatchMismatch

The matches following a `Close` fill do not reference it and exhaust it
exactly, a match pairs a lot of the wrong group, contract or side, or the
contract, side or group copied onto an `Expiry` differ from its opening
fill's. `id` is the offending event.
"""
struct MatchMismatch <: Exception
    id::Int
    reason::String
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

Base.showerror(io::IO, e::NothingToClose) =
    print(io, "NothingToClose: group ", e.group, " holds nothing to close on ", e.contract)
Base.showerror(io::IO, e::ExceedsOpen) =
    print(io, "ExceedsOpen: group ", e.group, " asked to consume ", e.requested,
          " of ", e.available, " open on ", e.contract)
Base.showerror(io::IO, e::FillAfterExpiry) =
    print(io, "FillAfterExpiry: fill ", e.id, " effective at ", e.effective_at,
          " is after its contract's expiry ", e.expiry)
Base.showerror(io::IO, e::DanglingReference) =
    print(io, "DanglingReference: ", e.field, " = ", e.id,
          " is not an earlier event of the right kind")
Base.showerror(io::IO, e::MatchMismatch) =
    print(io, "MatchMismatch: event ", e.id, ": ", e.reason)
Base.showerror(io::IO, e::SequenceGap) =
    print(io, "SequenceGap: expected ", e.counter, " ", e.expected, ", got ", e.got)
Base.showerror(io::IO, e::NonPositiveQuantity) =
    print(io, "NonPositiveQuantity: quantity must be a positive integer, got ", e.quantity)

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
    spec = contract_spec(leg.contract.underlying)
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
    commit!(L, book, batch, spec)
    return batch
end

"""
    record_expiry!(L, book, lot::Lot; settlement_price, effective_at, recorded_at) -> Expiry

Settle the whole remaining quantity of `lot` at `settlement_price`. The
outcome is `Worthless` when intrinsic is zero and `CashSettled`
otherwise. `effective_at` is the settlement instant, `recorded_at` the
tick that booked it.
"""
function record_expiry!(L::Ledger, book::Book, lot::Lot;
                        settlement_price::Real, effective_at::DateTime,
                        recorded_at::DateTime)::Expiry
    spec = contract_spec(lot.contract.underlying)
    outcome = intrinsic(lot.contract, settlement_price) == 0.0 ? Worthless : CashSettled
    e = Expiry(_header(L, 0, effective_at, recorded_at), lot.group, lot.open_fill_id,
               lot.contract, lot.side, lot.remaining, settlement_price, outcome)
    commit!(L, book, LedgerEvent[e], spec)
    return e
end

"""
    record_fee!(L, book, source_id::Int, amount; effective_at, recorded_at) -> Fee

Book a signed cash `amount` (a cost is negative) caused by the fill
`source_id`. Throws [`DanglingReference`](@ref) when `source_id` is not
a fill in `L`.
"""
function record_fee!(L::Ledger, book::Book, source_id::Int, amount::Real;
                     effective_at::DateTime, recorded_at::DateTime)::Fee
    source = _fill_ref(L, nothing, source_id, :source_id)
    e = Fee(_header(L, 0, effective_at, recorded_at), source_id, amount)
    commit!(L, book, LedgerEvent[e], contract_spec(source.contract.underlying))
    return e
end

"""
    commit!(L, book, batch::AbstractVector{<:LedgerEvent}, spec::ContractSpec) -> nothing

The single validated write path. Checks `batch` against `L` and `book`,
then appends it as one unit and applies it to `book`. The checks, each
with its named failure:

- `id` and `sequence` continue the ledger's counters ([`SequenceGap`](@ref));
- a fill is effective at or before its contract's expiry ([`FillAfterExpiry`](@ref));
- every reference points to an earlier event of the right kind ([`DanglingReference`](@ref));
- the matches immediately following a `Close` fill reference it, pair
  lots of its group and contract on the opposite side, and sum to its
  quantity; an `Expiry`'s copied contract, side and group equal its
  opening fill's ([`MatchMismatch`](@ref));
- consumption never exceeds a lot's remaining ([`ExceedsOpen`](@ref)).

`spec` is the contract spec every event of the batch is booked under.
Nothing is appended if any check fails.
"""
function commit!(L::Ledger, book::Book, batch::AbstractVector{<:LedgerEvent},
                 spec::ContractSpec)::Nothing
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
        apply!(book, e, spec)
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

# Consumption of one opening fill's lot by this batch, against what the
# book holds plus what the batch itself opened, minus what it consumed.
function _consume_check!(book::Book, opened::Dict{Int,Int}, consumed::Dict{Int,Int},
                         o::Fill, quantity::Int)::Nothing
    oid = event_id(o)
    held = _remaining(book, o.group, o.contract, oid) + get(opened, oid, 0)
    available = held - get(consumed, oid, 0)
    quantity <= available || throw(ExceedsOpen(o.group, o.contract, quantity, available))
    consumed[oid] = get(consumed, oid, 0) + quantity
    return nothing
end

function _check_match(L::Ledger, book::Book, seen, opened, consumed, m::Match, c::Fill)::Nothing
    m.group == c.group ||
        throw(MatchMismatch(event_id(m), "group differs from closing fill $(event_id(c))"))
    o = _opening_fill(L, seen, m.open_fill_id, :open_fill_id)
    (o.group == c.group && o.contract == c.contract && o.side != c.side) ||
        throw(MatchMismatch(event_id(m), "opening fill $(event_id(o)) is not an " *
                            "opposite-side lot of the closing fill's group and contract"))
    _consume_check!(book, opened, consumed, o, m.quantity)
    return nothing
end

function _validate(L::Ledger, book::Book, batch::AbstractVector{<:LedgerEvent})::Nothing
    for (k, e) in enumerate(batch)
        want_id  = L.next_id + k - 1
        want_seq = L.next_sequence + k - 1
        event_id(e) == want_id || throw(SequenceGap(:id, want_id, event_id(e)))
        sequence(e) == want_seq || throw(SequenceGap(:sequence, want_seq, sequence(e)))
    end
    seen     = Dict{Int,LedgerEvent}()   # batch events validated so far
    opened   = Dict{Int,Int}()           # lots opened by this batch
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
                opened[event_id(e)] = e.quantity
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
            o = _opening_fill(L, seen, e.open_fill_id, :open_fill_id)
            (e.contract == o.contract && e.side == o.side && e.group == o.group) ||
                throw(MatchMismatch(event_id(e),
                    "copied contract, side or group differ from opening fill $(event_id(o))"))
            _consume_check!(book, opened, consumed, o, e.quantity)
            seen[event_id(e)] = e
            k += 1
        else                                       # Fee
            _fill_ref(L, seen, e.source_id, :source_id)
            seen[event_id(e)] = e
            k += 1
        end
    end
    return nothing
end
