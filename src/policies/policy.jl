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

"""
    tick_times(policy::Policy, source::ModelDataSource,
               from::DateTime, to::DateTime) -> Union{Nothing, Vector{DateTime}}

Optional override letting a sparse policy tell the engine "I only need to
be called at these specific timestamps in `[from, to]`." Default returns
`nothing`, in which case the engine falls back to walking every
`available_timestamps(source, from, to)` and the policy gates inside
`decide`. Concrete policies whose `decide` is a hard no-op on most ticks
(e.g. once-a-day-at-19:30 strategies on minute data) can implement this
to skip the engine churn entirely.

Implementations are not required to filter against `available_timestamps`
themselves -- the engine treats the returned vector as candidates and
tolerates timestamps where no chain exists (`decide` will see
`get_surface(...) === nothing` and return `Trade[]`).

**Contract** (the engine trusts the return verbatim -- no sort, dedupe, or
range filter is applied at `run_backtest`):

- All returned timestamps must lie within `[from, to]`.
- Sorted ascending.
- Unique. Duplicates would cause double-firing on that tick, which the
  ledger does not deduplicate.

For agent-level overrides that union per-policy schedules, normalize
(sort + unique) inside the agent's `tick_times` implementation rather
than relying on the engine.
"""
tick_times(::Policy, ::ModelDataSource, ::DateTime, ::DateTime) = nothing
