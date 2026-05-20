# Policy abstraction.
#
# A policy is a stateless decision function over (current timestamp,
# time-cut data view, current ledger). It emits trade *deltas* -- new
# orders to fill -- not a replacement portfolio. Closes are expressed as
# counter-trades (opposite direction, same contract); the engine appends
# every fill to the ledger and "net open" is a computed view.
#
# Higher-level evolution (refit cadence, parameter learning, swapping
# one policy for another over time) is the [`Agent`](@ref) layer's job;
# a Policy itself is frozen between the moments an Agent hands it out.

"""
    Policy

Abstract supertype for backtest policies. Concrete policies hold their
immutable configuration (schedules, parameters, fitted models) and
implement [`decide`](@ref). A Policy is the unit a [`Agent`](@ref)
hands to the engine at each tick; it is expected to be static for the
duration of that tick.
"""
abstract type Policy end

"""
    decide(policy::Policy, t::DateTime, data::TimeCutModelDataSource,
           positions::AbstractVector{Position}) -> Vector{Trade}

Return the trades the policy wants to fire at time `t`. An empty vector
means "no action this tick." Closes are emitted as counter-trades.

`data` is a [`TimeCutModelDataSource`](@ref) cut to `t`; the type signature
makes the supported data interface no-lookahead by construction. Accessors on
`data` return absent values for timestamps strictly after `t`.

`positions` is the full ledger of fills so far (open *and* offsetting
closes). Policies that want only currently-open net positions can
derive that view by netting `direction * quantity` per contract.
"""
function decide(::Policy, ::DateTime, ::TimeCutModelDataSource, ::AbstractVector{Position})::Vector{Trade}
    error("decide not implemented for this Policy")
end

"""
    NoOpPolicy()

Trivial policy that never trades. Useful as a smoke test for the engine
and as a base case in tests.
"""
struct NoOpPolicy <: Policy end

decide(::NoOpPolicy, ::DateTime, ::TimeCutModelDataSource, ::AbstractVector{Position}) = Trade[]
