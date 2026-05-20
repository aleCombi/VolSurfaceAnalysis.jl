# Agent abstraction.
#
# An Agent is the higher-level object that owns how a [`Policy`](@ref)
# evolves across backtest (or live) time. The engine queries the Agent
# at every tick for the Policy that should make the decision at that
# moment; between ticks the Agent is free to refit, swap, schedule, or
# otherwise update what it returns next.
#
# This is the Sutton-&-Barto split: Policy = the decide function;
# Agent = the thing that carries the Policy and the machinery that
# changes it over time.

"""
    Agent

Abstract supertype for backtest agents. Concrete agents hold whatever
state they need (current policy, refit schedule, training-window
buffers, fitted-model registry) and implement [`current_policy`](@ref).
"""
abstract type Agent end

"""
    current_policy(agent::Agent, t::DateTime, data::TimeCutModelDataSource,
                   positions::AbstractVector{Position}) -> Policy

Return the [`Policy`](@ref) the agent wants the engine to use at time
`t`. Called once per tick by [`run_backtest`](@ref) before `decide`.

The returned Policy must be valid for at least this tick. An agent
that refits periodically returns the same Policy on every tick between
refits, and a fresh one on the tick where the refit fires.

`data` and `positions` are passed in case the refit logic needs to
inspect the current data view or ledger (e.g. "refit only on the first
tick of a new month, using the lookback window in `data`"). Stateless,
schedule-free agents simply ignore them.
"""
function current_policy(::Agent, ::DateTime, ::TimeCutModelDataSource, ::AbstractVector{Position})::Policy
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

current_policy(a::StaticAgent, ::DateTime, ::TimeCutModelDataSource, ::AbstractVector{Position}) = a.policy
