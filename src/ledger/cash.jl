# Cash is a method on an event, never a stored field, so inputs and cash
# cannot disagree. The rules, fixed here (proposal, section 2):
#
#   Fill     -side_sign(side) * price * quantity * multiplier
#   Match    0
#   Expiry    side_sign(side) * intrinsic(contract, settlement_price) * quantity * multiplier
#   Fee      amount
#
# The two-argument form takes the contract spec explicitly so a test can
# pin the multiplier without the table; the one-argument form resolves it
# from the event's contract. A `Match` and a `Fee` carry no contract and
# need no spec.

"""
    intrinsic(c::ContractKey, spot::Real) -> Float64

Intrinsic value per share of `c` at `spot`: `max(spot - strike, 0)` for
a call, `max(strike - spot, 0)` for a put.
"""
function intrinsic(c::ContractKey, spot::Real)::Float64
    s = Float64(spot)
    return c.option_type == Call ? max(s - c.strike, 0.0) : max(c.strike - s, 0.0)
end

"""
    cash(e::LedgerEvent, spec::ContractSpec) -> Float64
    cash(e::LedgerEvent) -> Float64

Signed cash the event moves, in USD, by the rules at the top of
`cash.jl`. The one-argument form resolves `spec` from the event's
contract through [`contract_spec`](@ref).
"""
cash(e::Fill, spec::ContractSpec)::Float64 =
    -side_sign(e.side) * e.price * e.quantity * spec.multiplier
cash(::Match, ::Any)::Float64 = 0.0
cash(e::Expiry, spec::ContractSpec)::Float64 =
    side_sign(e.side) * intrinsic(e.contract, e.settlement_price) * e.quantity * spec.multiplier
cash(e::Fee, ::Any)::Float64 = e.amount

cash(e::Union{Fill,Expiry})::Float64 = cash(e, contract_spec(e.contract.underlying))
cash(::Match)::Float64 = 0.0
cash(e::Fee)::Float64 = e.amount
