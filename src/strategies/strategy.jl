# Strategy abstraction.
#
# A strategy is a stateless decision function over (current timestamp,
# time-cut data view, current ledger). It emits trade *deltas* -- new
# orders to fill -- not a replacement portfolio. Closes are expressed as
# counter-trades (opposite direction, same contract); the engine appends
# every fill to the ledger and "net open" is a computed view.

"""
    Strategy

Abstract supertype for backtest strategies. Concrete strategies hold
their immutable configuration (schedules, parameters, fitted models) and
implement [`decide`](@ref).
"""
abstract type Strategy end

"""
    decide(strategy::Strategy, t::DateTime, data::TimeCutModelDataSource,
           positions::AbstractVector{Position}) -> Vector{Trade}

Return the trades the strategy wants to fire at time `t`. An empty vector
means "no action this tick." Closes are emitted as counter-trades.

`data` is a [`TimeCutModelDataSource`](@ref) cut to `t`; the type signature
makes the supported data interface no-lookahead by construction. Accessors on
`data` return absent values for timestamps strictly after `t`.

`positions` is the full ledger of fills so far (open *and* offsetting
closes). Strategies that want only currently-open net positions can
derive that view by netting `direction * quantity` per contract.
"""
function decide(::Strategy, ::DateTime, ::TimeCutModelDataSource, ::AbstractVector{Position})::Vector{Trade}
    error("decide not implemented for this Strategy")
end

"""
    NoOpStrategy()

Trivial strategy that never trades. Useful as a smoke test for the engine
and as a base case in tests.
"""
struct NoOpStrategy <: Strategy end

decide(::NoOpStrategy, ::DateTime, ::TimeCutModelDataSource, ::AbstractVector{Position}) = Trade[]
