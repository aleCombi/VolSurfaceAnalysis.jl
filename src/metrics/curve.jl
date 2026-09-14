# MarkedCurve: portfolio profit through time, sampled at session closes.
# The path metrics read it; the trade metrics read per-trade dollars
# instead (`trades.jl`). Built by `marked_curve` in `marks.jl`, which is
# the only place market data enters this module.
#
# Two pairs of parallel vectors and nothing else: the sessions that were
# marked, and the sessions that could not be. There is no hierarchy, no
# per-point wrapper and no sentinel -- a session is in exactly one of the
# two pairs, so an unanswerable mark cannot be mistaken for a value.

"""
    cents_to_usd(cents::Integer) -> Float64

The one boundary where the ledger's whole USD cents become floating-point
dollars. The inverse direction is [`contract_cents`](@ref), the ledger's
one rounding point; this is its counterpart on the way out, and every
dollar figure this module reports passes through it.

Marks do not: a quote mid or a surface price is floating point at source
and is never whole cents (the mid of 1.05/1.06 is 1.055), so the
unrealised term of a marked profit is float arithmetic from the start.
The realised term is integer cents until it crosses here.
"""
cents_to_usd(cents::Integer)::Float64 = cents / 100

"""
    MarkedCurve

Marked portfolio profit at each session close of an evaluation window,
in USD, measured from zero. The record of profit *through time*, as round
trips are the record of realised trade outcomes.

At every point,

> marked profit = realised profit + unallocated fees + unrealised profit

which is the ledger's cash at that instant plus the value of the open
book marked to market (see [`marked_curve`](@ref) for why those are the
same number). It is zero-based, not an account value: the codebase has no
deposited capital and none is invented here.

# Fields
- `timestamps::Vector{DateTime}` -- the session closes that were marked, ascending.
- `profit::Vector{Float64}` -- marked profit in USD at each, parallel to `timestamps`.
- `unmarked_at::Vector{DateTime}` -- session closes that could **not** be marked, ascending.
- `unmarked_reason::Vector{Symbol}` -- why, parallel to `unmarked_at`.

An unmarkable session carries no profit value anywhere: it is absent from
`timestamps` / `profit` and present, named, in the second pair (design
rule 7). [`session_changes`](@ref) then refuses to span it, so a broken
point costs two observations rather than inventing one.
"""
struct MarkedCurve
    timestamps      :: Vector{DateTime}
    profit          :: Vector{Float64}
    unmarked_at     :: Vector{DateTime}
    unmarked_reason :: Vector{Symbol}

    function MarkedCurve(timestamps::AbstractVector{DateTime},
                         profit::AbstractVector{<:Real},
                         unmarked_at::AbstractVector{DateTime},
                         unmarked_reason::AbstractVector{Symbol})
        length(timestamps) == length(profit) || throw(ArgumentError(
            "MarkedCurve: $(length(timestamps)) timestamps for $(length(profit)) profit values"))
        length(unmarked_at) == length(unmarked_reason) || throw(ArgumentError(
            "MarkedCurve: $(length(unmarked_at)) unmarked instants for " *
            "$(length(unmarked_reason)) reasons"))
        issorted(timestamps) ||
            throw(ArgumentError("MarkedCurve: timestamps must be ascending"))
        issorted(unmarked_at) ||
            throw(ArgumentError("MarkedCurve: unmarked_at must be ascending"))
        new(collect(DateTime, timestamps), collect(Float64, profit),
            collect(DateTime, unmarked_at), collect(Symbol, unmarked_reason))
    end
end

"""
    n_marked(curve::MarkedCurve) -> Int

Session closes that carry a marked profit.
"""
n_marked(c::MarkedCurve)::Int = length(c.timestamps)

"""
    n_unmarked(curve::MarkedCurve) -> Int

Session closes on the grid that could not be marked. Zero is the honest
answer "every session was marked", never "nobody looked": a curve built
without asking the question does not exist, because building it is what
asks it.
"""
n_unmarked(c::MarkedCurve)::Int = length(c.unmarked_at)

"""
    session_changes(curve::MarkedCurve) -> Vector{Float64}

The session-to-session changes in marked profit, in USD: one observation
per pair of **adjacent** marked sessions. A pair straddling an unmarked
session yields no observation, because the change across it spans two
periods and the path metrics scale by the number of periods.

Empty for a curve with fewer than two marked points; a curve whose every
other session is unmarked is empty too, and that is the honest answer
rather than a series of double-length steps.
"""
function session_changes(c::MarkedCurve)::Vector{Float64}
    out = Float64[]
    for i in 2:length(c.timestamps)
        a, b = c.timestamps[i - 1], c.timestamps[i]
        # The first unmarked instant at or after `a`; an unmarked session is
        # never also a marked one, so it is strictly after `a`.
        k = searchsortedfirst(c.unmarked_at, a)
        k <= length(c.unmarked_at) && c.unmarked_at[k] < b && continue
        push!(out, c.profit[i] - c.profit[i - 1])
    end
    return out
end
