# One row per consumed lot, from `Match` and `Expiry`, in sequence order.
# Data-free: everything a row needs is in the ledger.

"""
    RoundTrip

One consumed lot: `quantity` of the lot `open_id` closed by `close_id`
(the closing `Fill` when `kind == :closed`, the `Expiry` when
`kind == :expired`); `side` is the opening fill's; `opened_at` and
`closed_at` are effective times; `pnl` is whole USD cents and includes
the share of any fee on the opening or the closing fill.
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
    pnl::Int
end

"""
    round_trips(L::Ledger) -> Vector{RoundTrip}
    round_trips(L::Ledger, spec::ContractSpec) -> Vector{RoundTrip}

One row per `Match` and per `Expiry`, in sequence order. The price part
of a closed trip is `side_sign(side) * (contract_cents(close.price) -
contract_cents(open.price)) * quantity`; of an expired trip the same
with `contract_cents(intrinsic)` in place of the closing price. Both are
integer by construction.

Fees are shared in whole cents by cumulative rounding: for a fill with
total fee `F` and quantity `Q`, the trips that consume it are walked in
sequence order keeping the cumulative consumed quantity `c`, and a
trip's share is `round(F * c / Q)` minus the same value at the previous
trip. The shares over the trips that fully consume a fill sum to `F`
exactly; a fill left partly open leaves the remainder unallocated. When
no lot is left open the rows therefore sum exactly to the replayed
book's cash. The one-argument form resolves the multiplier per contract
from the table; the two-argument form pins one spec for every row.
"""
round_trips(L::Ledger)::Vector{RoundTrip} =
    _round_trips(L, c -> contract_spec(c.underlying))
round_trips(L::Ledger, spec::ContractSpec)::Vector{RoundTrip} =
    _round_trips(L, _ -> spec)

function _round_trips(L::Ledger, spec_for)::Vector{RoundTrip}
    fees = Dict{Int,Int}()                        # total fee per source fill, cents
    for e in L.events
        e isa Fee && (fees[e.source_id] = get(fees, e.source_id, 0) + e.amount)
    end
    progress = Dict{Int,Int}()                    # quantity of each fill consumed so far
    # The share of fill `f`'s fee that falls to the trip consuming `q` of
    # it next: cumulative rounding, so the shares of a fully consumed fill
    # sum to the fee exactly. The rational keeps the quotient exact;
    # round(Int, x) is to nearest, ties to even.
    function share!(f::Fill, q::Int)::Int
        fid = event_id(f)
        before = get(progress, fid, 0)
        after = before + q
        progress[fid] = after
        total = get(fees, fid, 0)
        total == 0 && return 0
        return round(Int, total * after // f.quantity) - round(Int, total * before // f.quantity)
    end

    out = RoundTrip[]
    for e in L.events
        if e isa Match
            o = event(L, e.open_fill_id)::Fill
            c = event(L, e.close_fill_id)::Fill
            spec = spec_for(o.contract)
            pnl = side_sign(o.side) *
                  (contract_cents(c.price, spec) - contract_cents(o.price, spec)) * e.quantity +
                  share!(o, e.quantity) + share!(c, e.quantity)
            push!(out, RoundTrip(o.group, o.contract, o.side, e.quantity, event_id(o),
                                 event_id(c), effective_at(o), effective_at(c), :closed, pnl))
        elseif e isa Expiry
            o = event(L, e.open_fill_id)::Fill
            spec = spec_for(o.contract)
            settled = contract_cents(intrinsic(o.contract, e.settlement_price), spec)
            pnl = side_sign(o.side) * (settled - contract_cents(o.price, spec)) * e.quantity +
                  share!(o, e.quantity)
            push!(out, RoundTrip(o.group, o.contract, o.side, e.quantity, event_id(o),
                                 event_id(e), effective_at(o), effective_at(e), :expired, pnl))
        end
    end
    return out
end
