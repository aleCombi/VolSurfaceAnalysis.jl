# `agents` module

An agent owns how a policy changes over time. Per tick the engine asks
`current_policy` and then calls `decide` on what it gets, the same
`(t, cut, book)` visible to both; the returned policy must be valid for
at least that tick. Between ticks the agent may refit, swap or advance
a schedule; the policy it hands out is frozen
([`policies`](policies.md)).

## The query

Everything an agent schedules it gates inside `current_policy`, the way
a policy gates inside `decide`: there is no refit-schedule protocol,
and the engine knows nothing of an agent's internals.
`declared_underlyings` and `tick_times` exist at this layer so the
loader and the engine ask one object; `StaticAgent` delegates both to
its one policy.

## Boundaries

**Owns** `Agent`, `current_policy`, `StaticAgent`, the two delegations.
**Does not own** the decision ([`policies`](policies.md)); the tick
loop ([`backtest`](backtest.md)); P&L (downstream).

## Decisions

| Decision | Why |
|---|---|
| **Per-tick query, not policy-change events** | One loop shape, mirroring the per-tick `decide`; a refit-on-schedule agent is a calendar check inside `current_policy`. |
| **`current_policy` sees `(t, cut, book)`** | A refit needs `t`; a lookback reads through the cut, history before the window visible and nothing after `t`; sizing reads the book. |
| **An agent is not a policy** | One evolves over time and is mutable, the other decides for one tick and is frozen. An agent that never changes is a `StaticAgent`, not a policy worn as an agent. |
| **The engine accepts both** | `run_backtest(policy)` wraps `StaticAgent(policy)`, so every backtest shares one driver path and scoring a single candidate needs no agent. |

## Conventions consulted

- **Policy and agent as two types.** Sutton & Barto, *Reinforcement
  Learning*: the policy is the decision function, the agent carries it
  and the machinery that changes it. Adopted as stated rather than
  overloading one type with both; a policy built on a fitted model is
  still a frozen policy, and the fitting lives on its agent.
