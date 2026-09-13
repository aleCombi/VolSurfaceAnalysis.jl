# Contract facts per underlying: a table in code keyed by ticker.
#
# These are facts, not choices. The multiplier changes cash, and a wrong
# entry is wrong rather than a variant of the experiment. Nothing observes
# them and they have no visibility time, so they are neither config nor a
# market-data kind: they live in code the way exchange calendars do. The
# engine projects the *resolved* values into run identity (slice 4), so a
# correction here is a new run id, never a silent change to old results.
# An underlying missing from the table is a loud error; there is no
# default contract.
#
# Source: OCC product specifications for listed equity and ETF options
# (theocc.com, "Product Specifications"): 100 shares per contract,
# American exercise, PM settlement against the underlying's close,
# physical delivery of the shares.

@enum ExerciseStyle American European
@enum SettlementStyle AMSettled PMSettled
@enum Delivery Physical Cash

"""
    ContractSpec

Facts about one underlying's listed option contracts.

# Fields
- `multiplier::Int` -- shares per contract; cash per contract is the
  price per share times this (in whole cents through `contract_cents`).
- `exercise::ExerciseStyle` -- `American` or `European`.
- `settlement::SettlementStyle` -- `AMSettled` or `PMSettled`.
- `delivery::Delivery` -- `Physical` or `Cash`.
"""
struct ContractSpec
    multiplier::Int
    exercise::ExerciseStyle
    settlement::SettlementStyle
    delivery::Delivery
end

"""
    UnknownContract

Thrown by [`contract_spec`](@ref) for an underlying the table does not
list. Carries the `underlying` asked for.
"""
struct UnknownContract <: Exception
    underlying::Underlying
end

Base.showerror(io::IO, e::UnknownContract) =
    print(io, "UnknownContract: no contract spec for ", e.underlying)

const _CONTRACT_TABLE = Dict{String,ContractSpec}(
    "SPY" => ContractSpec(100, American, PMSettled, Physical),
    "QQQ" => ContractSpec(100, American, PMSettled, Physical),
    "IWM" => ContractSpec(100, American, PMSettled, Physical),
)

"""
    contract_spec(u::Underlying) -> ContractSpec

The contract facts for `u`. Throws [`UnknownContract`](@ref) when `u` is
not in the table.
"""
function contract_spec(u::Underlying)::ContractSpec
    spec = get(_CONTRACT_TABLE, ticker(u), nothing)
    spec === nothing && throw(UnknownContract(u))
    return spec
end
