# Agent, `current_policy`, `StaticAgent`; agent-level `declared_underlyings` and `tick_times`.

"""
    Agent

Abstract supertype for backtest agents. Concrete agents hold whatever
state they need (current policy, refit schedule, training-window
buffers, fitted-model registry) and implement [`current_policy`](@ref).
"""
abstract type Agent end

"""
    current_policy(agent::Agent, t::DateTime, data::TimeCut, book::Book) -> Policy

Return the [`Policy`](@ref) the agent wants the engine to use at time
`t`. Called once per tick by [`run_backtest`](@ref) before `decide`.

The returned Policy must be valid for at least this tick. An agent
that refits periodically returns the same Policy on every tick between
refits, and a fresh one on the tick where the refit fires.

`data` and `book` are there for a refit to read; a schedule-free agent
ignores them.
"""
function current_policy(::Agent, ::DateTime, ::TimeCut, ::Book)::Policy
    error("current_policy not implemented for this Agent")
end

"""
    StaticAgent(policy::Policy)

The trivial Agent: holds one [`Policy`](@ref) and returns it for every
tick, forever. Bridges the "fixed policy, no learning" case into the
Agent-driven engine so all backtests share one driver path.
"""
struct StaticAgent{P<:Policy} <: Agent
    policy::P
end

current_policy(a::StaticAgent, ::DateTime, ::TimeCut, ::Book) = a.policy

"""
    declared_underlyings(agent::Agent) -> Tuple of Underlying

Agent-level view of [`declared_underlyings(::Policy)`](@ref). Default
empty; `StaticAgent` delegates to its policy.
"""
declared_underlyings(::Agent) = ()

declared_underlyings(a::StaticAgent) = declared_underlyings(a.policy)

"""
    tick_times(agent::Agent, data::MarketData,
               from::DateTime, to::DateTime) -> Union{Nothing, Vector{DateTime}}

Agent-level override of the engine's tick cadence, with the contract of
[`tick_times(::Policy, ...)`](@ref). Default `nothing`; `StaticAgent`
delegates to its policy.
"""
tick_times(::Agent, ::MarketData, ::DateTime, ::DateTime) = nothing

tick_times(a::StaticAgent, data::MarketData, from::DateTime, to::DateTime) =
    tick_times(a.policy, data, from, to)
