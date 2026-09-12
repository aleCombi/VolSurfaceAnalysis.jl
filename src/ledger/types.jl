# Ledger vocabulary: the order a policy emits, the four event kinds the
# engine books, their shared header, and the append-only container.
#
# Identity vocabulary (`Underlying`, `OptionType`, `Call`, `Put`) comes
# from `data/quotes.jl`. Nothing here refers to quotes, spots or the time
# cut: a ledger is built and replayed from its own events, which is what
# keeps the module testable on hand-built ledgers.
#
# `NonPositiveQuantity`, `InvalidPrice`, `DanglingReference` and
# `FillAfterExpiry`, thrown by the constructors and by `event` below, are
# defined with the other named failures in `append.jl`.

@enum Side Long Short
@enum Intent Open Close
@enum ExpiryOutcome Worthless CashSettled

"""
    side_sign(s::Side) -> Int

`+1` for `Long`, `-1` for `Short`.
"""
side_sign(s::Side)::Int = s == Long ? 1 : -1

"""
    ContractKey

Identity of one listed option contract: `underlying`, `strike`, `expiry`
and `option_type`. `Trade` minus direction and quantity.
"""
struct ContractKey
    underlying::Underlying
    strike::Float64
    expiry::DateTime
    option_type::OptionType
end

# Content hash and equality, for the reason `Underlying` has them: the
# book keys lots on (group, contract), and a dictionary keyed on the
# default objectid-based hash iterates in a build-dependent order.
Base.hash(c::ContractKey, h::UInt) =
    hash(c.option_type, hash(c.expiry, hash(c.strike, hash(c.underlying, hash(:ContractKey, h)))))
Base.:(==)(a::ContractKey, b::ContractKey) =
    a.underlying == b.underlying && a.strike == b.strike &&
    a.expiry == b.expiry && a.option_type == b.option_type

"""
    Leg

One leg of an [`Order`](@ref): a contract, a side, a positive integer
number of contracts, and a declared `intent` (`Open` or `Close`). The
constructor throws [`NonPositiveQuantity`](@ref) otherwise.
"""
struct Leg
    contract::ContractKey
    side::Side
    quantity::Int
    intent::Intent

    function Leg(contract::ContractKey, side::Side, quantity::Integer, intent::Intent)
        quantity > 0 || throw(NonPositiveQuantity(Int(quantity)))
        new(contract, side, Int(quantity), intent)
    end
end

"""
    Order

One structure-level instruction out of `decide`: a `label`, its `legs`,
the `group` it opens into or closes (`nothing` on an opening order: the
ledger mints one), and an optional `operation` id linking the two orders
of a roll.
"""
struct Order
    label::Symbol
    legs::Vector{Leg}
    group::Union{Nothing,Int}
    operation::Union{Nothing,Int}
end

# `Base.Order` is a module Base does not export, so this shadows nothing;
# still, never `using Base.Order` inside the package.
Order(label::Symbol, legs::AbstractVector{Leg};
      group::Union{Nothing,Int}=nothing, operation::Union{Nothing,Int}=nothing) =
    Order(label, collect(legs), group, operation)

"""
    EventHeader

What every event carries, by composition: a stable `id` (a reference,
never an index), the bitemporal pair `effective_at` (when the fact is
true) and `recorded_at` (when the ledger learned it), and `sequence`,
the replay order.
"""
struct EventHeader
    id::Int
    effective_at::DateTime
    recorded_at::DateTime
    sequence::Int
end

"""
    Fill

One execution, effective at or before its contract's expiry.
`order_leg_id` and `execution_id` join it to the order journal and to
the execution report, so both are positive; `contract`, `side`,
`intent`, a positive `quantity` of contracts, a finite positive `price`
per share, and the `fill_rule` that produced the price
(`:cross_spread`, `:broker_execution`, ...). Carries nothing about the
market it was filled against. The constructor throws
[`NonPositiveQuantity`](@ref), [`InvalidPrice`](@ref),
[`DanglingReference`](@ref) (a non-positive join id) or
[`FillAfterExpiry`](@ref) otherwise, so no such value exists.
"""
struct Fill
    header::EventHeader
    group::Int
    order_leg_id::Int
    execution_id::Int
    contract::ContractKey
    side::Side
    intent::Intent
    quantity::Int
    price::Float64
    fill_rule::Symbol

    function Fill(header::EventHeader, group::Integer, order_leg_id::Integer,
                  execution_id::Integer, contract::ContractKey, side::Side,
                  intent::Intent, quantity::Integer, price::Real, fill_rule::Symbol)
        quantity > 0 || throw(NonPositiveQuantity(Int(quantity)))
        (isfinite(price) && price > 0) || throw(InvalidPrice(Float64(price)))
        order_leg_id > 0 || throw(DanglingReference(:order_leg_id, Int(order_leg_id)))
        execution_id > 0 || throw(DanglingReference(:execution_id, Int(execution_id)))
        header.effective_at <= contract.expiry ||
            throw(FillAfterExpiry(header.id, header.effective_at, contract.expiry))
        new(header, group, order_leg_id, execution_id, contract, side, intent,
            Int(quantity), Float64(price), fill_rule)
    end
end

"""
    Match

One lot allocation after a closing fill: `close_fill_id` consumed
`quantity` of the lot opened by `open_fill_id`, inside `group`. Moves no
cash. Exists because one close can split across several lots and
because a change to the matching rule must not rewrite old results.
"""
struct Match
    header::EventHeader
    group::Int
    open_fill_id::Int
    close_fill_id::Int
    quantity::Int

    function Match(header::EventHeader, group::Integer, open_fill_id::Integer,
                   close_fill_id::Integer, quantity::Integer)
        quantity > 0 || throw(NonPositiveQuantity(Int(quantity)))
        new(header, group, open_fill_id, close_fill_id, Int(quantity))
    end
end

"""
    Expiry

One remaining lot reaching settlement: `quantity` of the lot opened by
`open_fill_id` settles at `settlement_price` (finite and non-negative,
else [`InvalidPrice`](@ref) at construction) with `outcome`. `contract`
and `side` are copied from the opening fill and checked against it on
append, so the event's cash is local to the event; `outcome` is derived
from the intrinsic value and checked on append too.
"""
struct Expiry
    header::EventHeader
    group::Int
    open_fill_id::Int
    contract::ContractKey
    side::Side
    quantity::Int
    settlement_price::Float64
    outcome::ExpiryOutcome

    function Expiry(header::EventHeader, group::Integer, open_fill_id::Integer,
                    contract::ContractKey, side::Side, quantity::Integer,
                    settlement_price::Real, outcome::ExpiryOutcome)
        quantity > 0 || throw(NonPositiveQuantity(Int(quantity)))
        (isfinite(settlement_price) && settlement_price >= 0) ||
            throw(InvalidPrice(Float64(settlement_price)))
        new(header, group, open_fill_id, contract, side, Int(quantity),
            Float64(settlement_price), outcome)
    end
end

"""
    Fee

A cost tied to its cause: `source_id` is the fill that caused it and
`amount` is signed cash in whole USD cents (a cost is negative).
"""
struct Fee
    header::EventHeader
    source_id::Int
    amount::Int
end

"""
    LedgerEvent

The closed union of event kinds. Closedness lets serialisation be
exhaustive; the container is a vector over it.
"""
const LedgerEvent = Union{Fill,Match,Expiry,Fee}

"""
    Ledger

The append-only journal of economic facts. `events` is in sequence
order; the counters are private to the writers in `append.jl`. `id` and
`sequence` are separate counters that coincide in a fresh ledger and are
never used for each other: events are looked up by id through
[`event`](@ref), not by index.
"""
mutable struct Ledger
    events::Vector{LedgerEvent}
    next_id::Int
    next_sequence::Int
    next_group::Int
    next_execution::Int
    index::Dict{Int,Int}   # event id -> position in `events`
end

Ledger() = Ledger(LedgerEvent[], 1, 1, 1, 1, Dict{Int,Int}())

Base.length(L::Ledger) = length(L.events)
Base.isempty(L::Ledger) = isempty(L.events)
Base.show(io::IO, L::Ledger) = print(io, "Ledger(", length(L), " events)")

"""
    event(L::Ledger, id::Int) -> LedgerEvent

The event with `id`. Throws [`DanglingReference`](@ref) (`:event_id`)
for an id the ledger never minted; a sequence number is not an id.
"""
function event(L::Ledger, id::Int)::LedgerEvent
    i = get(L.index, id, nothing)
    i === nothing && throw(DanglingReference(:event_id, id))
    return L.events[i]
end

# Accessors on the shared header. `group` is the one field a kind lacks:
# it sits on lifecycle events only, so a `Fee` answers `nothing`.
header(e::LedgerEvent)::EventHeader = e.header
event_id(e::LedgerEvent)::Int = e.header.id
effective_at(e::LedgerEvent)::DateTime = e.header.effective_at
recorded_at(e::LedgerEvent)::DateTime = e.header.recorded_at
sequence(e::LedgerEvent)::Int = e.header.sequence
group(e::Union{Fill,Match,Expiry})::Int = e.group
group(::Fee) = nothing
