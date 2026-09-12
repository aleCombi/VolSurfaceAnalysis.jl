# Cash is a method on an event, never a stored field, so inputs and cash
# cannot disagree. Inside the ledger cash is an integer number of USD
# cents. There is one rounding point, `contract_cents`, the cash per
# contract for a per-share price; everything after it is integer
# arithmetic, so the incremental book, both replays and the round trips
# agree exactly rather than to a tolerance. The rules (proposal,
# section 2):
#
#   Fill     -side_sign(side) * contract_cents(price) * quantity
#   Match    0
#   Expiry    side_sign(side) * contract_cents(intrinsic(contract, settlement_price)) * quantity
#   Fee      amount
#
# The two-argument forms take the contract spec explicitly so a test can
# pin the multiplier without the table; the one-argument forms resolve it
# from the event's contract. A `Match` and a `Fee` carry no contract and
# need no spec.
#
# `NonIntegralCash`, thrown by `contract_cents`, is defined with the other
# named failures in `append.jl`.

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
    contract_cents(price::Real, spec::ContractSpec) -> Int

The cash one contract moves at a per-share `price`, in whole USD cents:
`price * multiplier * 100`, rounded to the nearest integer. The product
must already be whole cents to within `1e-6`; a price that is not is
refused with [`NonIntegralCash`](@ref) rather than rounded, so no real
amount is ever rounded away; a non-finite or out-of-range product is
refused the same way. This is the ledger's one rounding point.
"""
function contract_cents(price::Real, spec::ContractSpec)::Int
    value = Float64(price) * spec.multiplier * 100
    # A non-finite or out-of-range product is not whole cents either; refuse
    # it here so the named failure, not an InexactError, reaches the caller.
    (isfinite(value) && abs(value) < 9.2e18) || throw(NonIntegralCash(value))
    # round(Int, x) is Julia's default rounding: to nearest, ties to even.
    # The tolerance below only absorbs floating-point noise on a product
    # that is whole cents, so a tie never reaches it.
    cents = round(Int, value)
    abs(value - cents) <= 1e-6 || throw(NonIntegralCash(value))
    return cents
end

"""
    cash(e::LedgerEvent, spec::ContractSpec) -> Int
    cash(e::LedgerEvent) -> Int

Signed cash the event moves, in whole USD cents, by the rules at the top
of `cash.jl`. The one-argument form resolves `spec` from the event's
contract through [`contract_spec`](@ref). Throws [`NonIntegralCash`](@ref)
when the price (or the intrinsic value) is not whole cents per contract.
"""
cash(e::Fill, spec::ContractSpec)::Int =
    -side_sign(e.side) * contract_cents(e.price, spec) * e.quantity
cash(::Match, ::Any)::Int = 0
cash(e::Expiry, spec::ContractSpec)::Int =
    side_sign(e.side) * contract_cents(intrinsic(e.contract, e.settlement_price), spec) * e.quantity
cash(e::Fee, ::Any)::Int = e.amount

cash(e::Union{Fill,Expiry})::Int = cash(e, contract_spec(e.contract.underlying))
cash(::Match)::Int = 0
cash(e::Fee)::Int = e.amount
