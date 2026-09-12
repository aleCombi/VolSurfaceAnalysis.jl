# One row per consumed lot, from `Match` and `Expiry`, in sequence order.
# Data-free: everything a row needs is in the ledger.

"""
    RoundTrip

One consumed lot: `quantity` of the lot `open_id` closed by `close_id`
(the closing `Fill` when `kind == :closed`, the `Expiry` when
`kind == :expired`); `side` is the opening fill's; `opened_at` and
`closed_at` are effective times; `pnl` is USD and includes the pro rata
share of any fee on the opening or the closing fill.
"""
struct RoundTrip
    group::Int
    contract::ContractKey
    side::Side
    quantity::Int
    open_id::Int
    close_id::Int
    opened_at::DateTime
    closed_at::DateTime
    kind::Symbol
    pnl::Float64
end

"""
    round_trips(L::Ledger) -> Vector{RoundTrip}
    round_trips(L::Ledger, spec::ContractSpec) -> Vector{RoundTrip}

One row per `Match` and per `Expiry`, in sequence order. PnL of a closed
trip is `side_sign(side) * (close.price - open.price) * quantity *
multiplier`; of an expired trip `side_sign(side) * (intrinsic -
open.price) * quantity * multiplier`; each plus, for every `Fee` whose
source is the opening or the closing fill, `amount * quantity / that
fill's quantity`. When no lot is left open the rows sum to the replayed
book's cash. The one-argument form resolves the multiplier per contract
from the table; the two-argument form pins one spec for every row.
"""
round_trips(L::Ledger)::Vector{RoundTrip} =
    _round_trips(L, c -> contract_spec(c.underlying))
round_trips(L::Ledger, spec::ContractSpec)::Vector{RoundTrip} =
    _round_trips(L, _ -> spec)

function _round_trips(L::Ledger, spec_for)::Vector{RoundTrip}
    fees = Dict{Int,Float64}()                    # total fee per source fill
    for e in L.events
        e isa Fee && (fees[e.source_id] = get(fees, e.source_id, 0.0) + e.amount)
    end
    share(f::Fill, q::Int) = get(fees, event_id(f), 0.0) * q / f.quantity

    out = RoundTrip[]
    for e in L.events
        if e isa Match
            o = event(L, e.open_fill_id)::Fill
            c = event(L, e.close_fill_id)::Fill
            mult = spec_for(o.contract).multiplier
            pnl = side_sign(o.side) * (c.price - o.price) * e.quantity * mult +
                  share(o, e.quantity) + share(c, e.quantity)
            push!(out, RoundTrip(o.group, o.contract, o.side, e.quantity, event_id(o),
                                 event_id(c), effective_at(o), effective_at(c), :closed, pnl))
        elseif e isa Expiry
            o = event(L, e.open_fill_id)::Fill
            mult = spec_for(o.contract).multiplier
            pnl = side_sign(o.side) * (intrinsic(o.contract, e.settlement_price) - o.price) *
                  e.quantity * mult + share(o, e.quantity)
            push!(out, RoundTrip(o.group, o.contract, o.side, e.quantity, event_id(o),
                                 event_id(e), effective_at(o), effective_at(e), :expired, pnl))
        end
    end
    return out
end
