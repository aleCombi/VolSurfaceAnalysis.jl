# The book is a view by replay, never stored: lots per (group, contract),
# FIFO within, plus cash. `apply!` is the fold step; the two replays
# differ only in which events they fold and in what order.
#
# `ExceedsOpen` and `DanglingReference`, thrown when a fold consumes a lot
# the book does not hold, are defined with the other named failures in
# `append.jl`; `commit!` checks the same conditions before anything is
# applied.

"""
    Lot

An open lot: `remaining` contracts of the fill `open_fill_id`, opened at
`unit_price` per share on `contract` and `side`, inside `group`.
"""
struct Lot
    group::Int
    contract::ContractKey
    side::Side
    open_fill_id::Int
    remaining::Int
    unit_price::Float64
end

"""
    Book

Lots per `(group, contract)`, FIFO within each vector, plus `cash` in
whole USD cents. Built by folding events with [`apply!`](@ref). Two
books are equal when their open lots and their cash are; cash is an
integer, so the comparison is exact.
"""
mutable struct Book
    lots::Dict{Tuple{Int,ContractKey},Vector{Lot}}
    cash::Int
end

Book() = Book(Dict{Tuple{Int,ContractKey},Vector{Lot}}(), 0)

Base.:(==)(a::Book, b::Book) = a.cash == b.cash && open_lots(a) == open_lots(b)

_by_open(v::Vector{Lot}) = sort!(v; by = l -> l.open_fill_id)

"""
    open_lots(book) -> Vector{Lot}

Every open lot, ordered by opening fill id.
"""
open_lots(book::Book)::Vector{Lot} =
    _by_open(Lot[l for v in values(book.lots) for l in v])

"""
    lots(book, group::Int) -> Vector{Lot}

The open lots of one group, ordered by opening fill id.
"""
lots(book::Book, group::Int)::Vector{Lot} =
    _by_open(Lot[l for (k, v) in book.lots if k[1] == group for l in v])

"""
    open_groups(book) -> Vector{Int}

Groups with at least one open lot, ascending.
"""
open_groups(book::Book)::Vector{Int} =
    sort!(unique!(Int[k[1] for (k, v) in book.lots if !isempty(v)]))

# The lots of one (group, contract) in FIFO order; empty when none.
_lots_at(book::Book, group::Int, contract::ContractKey)::Vector{Lot} =
    get(book.lots, (group, contract), Lot[])

# Remaining quantity of the lot opened by `open_fill_id`; zero once gone.
function _remaining(book::Book, group::Int, contract::ContractKey, open_fill_id::Int)::Int
    for l in _lots_at(book, group, contract)
        l.open_fill_id == open_fill_id && return l.remaining
    end
    return 0
end

# Consume `quantity` of the lot opened by `open_fill_id` in `group`,
# dropping the lot at zero and the (group, contract) key when it empties.
function _consume!(book::Book, group::Int, open_fill_id::Int, quantity::Int)::Book
    for (key, v) in book.lots
        key[1] == group || continue
        i = findfirst(l -> l.open_fill_id == open_fill_id, v)
        i === nothing && continue
        lot = v[i]
        quantity <= lot.remaining ||
            throw(ExceedsOpen(group, lot.contract, quantity, lot.remaining))
        left = lot.remaining - quantity
        if left == 0
            deleteat!(v, i)
            isempty(v) && delete!(book.lots, key)
        else
            v[i] = Lot(lot.group, lot.contract, lot.side, lot.open_fill_id, left, lot.unit_price)
        end
        return book
    end
    throw(DanglingReference(:open_fill_id, open_fill_id))
end

"""
    apply!(book, e::LedgerEvent, spec::ContractSpec) -> Book
    apply!(book, e::LedgerEvent) -> Book

Fold one event into `book`. An `Open` fill adds a lot and credits its
cash; a `Close` fill only credits its cash (the matches that follow
consume the lots); a `Match` reduces the named lot's remaining and drops
it at zero; an `Expiry` does the same and credits its cash; a `Fee`
credits its amount. The one-argument form resolves `spec` from the
event's contract.
"""
function apply!(book::Book, e::Fill, spec::ContractSpec)::Book
    if e.intent == Open
        push!(get!(() -> Lot[], book.lots, (e.group, e.contract)),
              Lot(e.group, e.contract, e.side, event_id(e), e.quantity, e.price))
    end
    book.cash += cash(e, spec)
    return book
end

function apply!(book::Book, e::Match, ::Any)::Book
    _consume!(book, e.group, e.open_fill_id, e.quantity)
    return book                                  # a match moves no cash
end

function apply!(book::Book, e::Expiry, spec::ContractSpec)::Book
    _consume!(book, e.group, e.open_fill_id, e.quantity)
    book.cash += cash(e, spec)
    return book
end

function apply!(book::Book, e::Fee, ::Any)::Book
    book.cash += e.amount
    return book
end

apply!(book::Book, e::Union{Fill,Expiry})::Book =
    apply!(book, e, contract_spec(e.contract.underlying))
apply!(book::Book, e::Union{Match,Fee})::Book = apply!(book, e, nothing)

"""
    book_as_known(L::Ledger, boundary::Int) -> Book

What the ledger knew at sequence `boundary`: the fold over events with
`sequence <= boundary`, in sequence order. This is the view a decision
could have seen. Recorded time is not a safe boundary for it, because
fills appended at the same tick after the decision share its recorded
time.
"""
function book_as_known(L::Ledger, boundary::Int)::Book
    book = Book()
    for e in L.events
        sequence(e) <= boundary && apply!(book, e)
    end
    return book
end

"""
    book_effective(L::Ledger, t::DateTime) -> Book

What was true at `t`: the fold over events with `effective_at <= t`,
ordered by `(effective_at, sequence)`, so events at an equal instant
fold in journal order. Every reference points backward in effective
time (checked on append), so the fold never meets a lot it has not yet
opened. Differs from [`book_as_known`](@ref) only by lifecycle booked at
the tick after its instant.
"""
function book_effective(L::Ledger, t::DateTime)::Book
    due = LedgerEvent[e for e in L.events if effective_at(e) <= t]
    sort!(due; by = e -> (effective_at(e), sequence(e)))
    book = Book()
    for e in due
        apply!(book, e)
    end
    return book
end
