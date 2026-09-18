# `agents` module

An agent owns how a policy changes over time. Per tick the engine asks
`current_policy` and then calls `decide` on what it gets, the same
`(t, cut, book)` visible to both; the returned policy must be valid for
at least that tick. Between ticks the agent may refit, swap or advance
a schedule; the policy it hands out is frozen
([`policies`](policies.md)).

## What an agent does per tick

The engine asks the agent for a policy on every tick. An agent that
refits on a schedule checks the date inside `current_policy` and
returns a new policy when the date says so, and the same one otherwise.
`declared_underlyings` and `tick_times` exist at the agent level so the
loader and the engine have one object to ask; `StaticAgent` passes both
through to its policy.

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
