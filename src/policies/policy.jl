# Policy abstraction.
#
# A policy is a stateless decision function over (current timestamp,
# time-cut data view, the book). It emits orders -- structure-level
# instructions with declared intent -- not a replacement portfolio. A
# close is a `Close` leg naming the group it closes; the engine books
# every order into the ledger and the book is the fold the policy reads.
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
    decide(policy::Policy, t::DateTime, data::TimeCut, book::Book) -> Vector{Order}

Return the orders the policy wants to fire at time `t`. An empty vector
means "no action this tick." Each `Order` declares its intent per leg:
an opening order leaves `group` as `nothing` and the ledger mints one; a
close is a `Close` leg in an order naming the group it closes
(`open_groups(book)`, `lots(book, g)`), never a counter-trade. The engine
books every order whole or not at all.

`data` is a [`TimeCut`](@ref) of the market data at `t`; the type
signature makes the supported data interface no-lookahead by
construction. Every shape on `data` is empty for timestamps strictly
after `t`, including reads made by derived providers on the policy's
behalf. Policies name kinds and selectors (`at(data, OptionQuote,
underlying, t)`), never storage.

`book` is the engine's own fold of the ledger as known at this tick
(open lots per group and contract, plus cash); it equals
`book_as_known(L, known_to)` for the order records this tick produces. A
policy reads it and must not mutate it.
"""
function decide(::Policy, ::DateTime, ::TimeCut, ::Book)::Vector{Order}
    error("decide not implemented for this Policy")
end

"""
    NoOpPolicy()

Trivial policy that never trades. Useful as a smoke test for the engine
and as a base case in tests.
"""
struct NoOpPolicy <: Policy end

decide(::NoOpPolicy, ::DateTime, ::TimeCut, ::Book) = Order[]

"""
    declared_underlyings(policy::Policy) -> Tuple of Underlying

The underlyings a policy fixes in its own configuration, known without
running it. Empty when it declares none, which means it cannot be checked
at load -- a policy that chooses its underlying per tick is the case the
default covers.

`load_experiment` uses it to enforce the real invariant of this codebase:
one experiment, one underlying. The clock selector answers *when* to step,
not *whose price*, and fills resolve per leg; asserting the two agree is
what makes that safe by construction rather than by assumption.
"""
declared_underlyings(::Policy) = ()

"""
    tick_times(policy::Policy, data::MarketData,
               from::DateTime, to::DateTime) -> Union{Nothing, Vector{DateTime}}

Optional override letting a sparse policy tell the engine "I only need to
be called at these specific timestamps in `[from, to]`." Default returns
`nothing`, in which case the engine walks the experiment's declared
clock and the policy gates inside `decide`. Concrete policies whose
`decide` is a hard no-op on most ticks (e.g. once-a-day-at-19:30
strategies on minute data) can implement this to skip the engine churn
entirely.

Implementations are not required to filter against the data's
timestamps -- the engine treats the returned vector as candidates and
tolerates timestamps where no chain exists (`decide` sees an empty
result and returns `Order[]`). The experiment's window end is still the
last *clock* tick, never a candidate emitted here.

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
tick_times(::Policy, ::MarketData, ::DateTime, ::DateTime) = nothing
