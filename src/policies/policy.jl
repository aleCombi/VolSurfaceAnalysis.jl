# Policy, `decide`, `NoOpPolicy`; the optional `declared_underlyings` and `tick_times`.

"""
    Policy

Abstract supertype for backtest policies. Concrete policies hold their
immutable configuration (schedules, parameters, fitted models) and
implement [`decide`](@ref). An [`Agent`](@ref) hands one to the engine
per tick, and it does not change during that tick.
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

`data` is a [`TimeCut`](@ref) at `t`: no read on it, direct or through a
derived provider, sees past `t`.

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
"""
declared_underlyings(::Policy) = ()

"""
    tick_times(policy::Policy, data::MarketData,
               from::DateTime, to::DateTime) -> Union{Nothing, Vector{DateTime}}

The candidate timestamps at which the engine calls `decide`, or `nothing`
(the default) to walk the experiment's declared clock. The engine trusts
the return verbatim: every timestamp in `[from, to]`, sorted, unique
(a duplicate fires `decide` twice on one tick). Candidates need not
exist in the data.
"""
tick_times(::Policy, ::MarketData, ::DateTime, ::DateTime) = nothing
