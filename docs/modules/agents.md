# `agents` module

Agent abstraction: the higher-level object that owns how a
[`Policy`](policies.md) evolves across backtest (or live) time. The
backtest engine queries the Agent at every tick for the Policy that
should make the decision at that moment; between ticks the Agent is
free to refit parameters, swap policies, advance a schedule, or
otherwise update what it returns next.

This is the Sutton-&-Barto split: **Policy** = the `decide` function;
**Agent** = the thing that carries the Policy and the machinery that
changes it over time. A policy that depends on a periodically-refit
ridge model is still a frozen Policy; the *fitting cadence and the
fit itself* live on the Agent.


## The abstraction


One method, four arguments, one Policy returned; plus the optional
`tick_times` override (default `nothing`; `StaticAgent` delegates to
its policy's), where a multi-policy agent unions its policies'
schedules. Concrete agents
subtype `Agent` and implement `current_policy`. The returned Policy
must be valid for at least the current tick. `book` is the engine's own
fold of the ledger as known at this tick (open lots per group and
contract, plus cash); like the policy, an agent reads it and must not
mutate it.

`declared_underlyings` mirrors the [policy-level trait](policies.md) at
this layer: default empty, `StaticAgent` delegates to its one policy, and
an agent that swaps policies over time reports their union or nothing
when it cannot say ahead of time. `load_experiment` reads it to check the
clock and the strategy name one underlying.

### `StaticAgent`


The trivial agent: holds one Policy and returns it forever. Bridges
the fixed-policy case into the Agent-driven engine so every backtest
shares one driver path, and lets `run_backtest(policy, ...)` be a
one-line wrapper around the Agent overload.

## Key decisions

| Decision | Why |
|---|---|
| **Per-tick query, not per-event callback** | The engine calls `current_policy` on every tick rather than asking the Agent to push policy-change events. This keeps the engine loop one-shape (mirrors the per-tick `decide` call) and means a "refit on schedule" Agent is a trivial calendar check inside `current_policy`. Cost on minute-data over a year for a no-op `current_policy`: dwarfed by data IO. |
| **`current_policy` sees `(t, cut, book)`** | Same arguments as `decide`. A refit-on-month-boundary Agent needs `t`; an Agent that retrains on a lookback window reads it through `cut` (history before `from` is visible, anything after `t` is not, derived data included); an Agent that adapts position sizing to current exposure reads the `book`. |
| **Agent is not itself a Policy** | The two have different responsibilities (evolve over time vs. decide for one tick) and different invariants (mutable cadence/state vs. frozen for the tick). Conflating them collapses the split that motivates the abstraction in the first place. An Agent that *never* changes its Policy is a `StaticAgent`, not a Policy worn as an Agent. |
| **Engine accepts both `Agent` and `Policy`** | `run_backtest(policy, ...)` is a one-line wrapper around `run_backtest(StaticAgent(policy), ...)`. The bare-policy form is the natural primitive for training/evaluation code that wants to score a single candidate Policy over a window without constructing an Agent. |
| **No refit-schedule protocol** | The engine does not have a separate `refit_times(agent, source)` hook. Anything an Agent wants to schedule it gates inside `current_policy`, the same way policies gate inside `decide`. One uniform query model, no engine-side knowledge of how an Agent is structured internally. |

## Responsibility boundaries

**Owns:** the `Agent` abstract type, the `current_policy` contract,
the agent-level `declared_underlyings` delegation, the `StaticAgent`
base case.

**Does NOT own:**

- The decide function. That is the [`policies`](policies.md) module.
- The tick loop. That is the [backtest engine](backtest.md); the
  engine drives both `current_policy` and `decide`.
- Reporting / PnL aggregation. Downstream of the engine, just like
  for policies.



