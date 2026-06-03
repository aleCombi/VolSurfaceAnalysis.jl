# `backtest` module

The driver that turns an [`Agent`](agents.md) (which hands out a
[`Policy`](policies.md) per tick) plus a [`ModelDataSource`](model_data.md)
into a ledger of filled [`Position`s](positions.md). Two concerns:

- A composition wrapper that enforces no-lookahead at the type level.
- A single tick loop that asks the agent for the current policy, asks
  that policy what to do, and fills its returned trades.

Reporting / PnL aggregation is intentionally not here -- the engine
returns the bare ledger and downstream code (later) computes
metrics off it.

## Data flow

```mermaid
flowchart LR
    MDS[ModelDataSource]
    Agent[Agent]
    Engine([run_backtest])
    MDS --> Engine
    Agent --> Engine

    subgraph Loop["per tick t"]
        direction LR
        Cut[TimeCutModelDataSource]
        CP([current_policy])
        D([decide])
        RQ([resolve_quote])
        OP([open_position])
        Cut --> CP
        CP -->|Policy| D
        Cut --> D
        D -->|orders| RQ
        RQ -->|OptionQuote| OP
        OP -->|Position| Ledger[(positions)]
    end

    Engine --> Loop
    Ledger --> Out[Vector Position]
```

## `TimeCutModelDataSource`

```julia
struct TimeCutModelDataSource
    inner::ModelDataSource
    cutoff::DateTime
end
```

Composition wrapper around a `ModelDataSource`. Every accessor
forwards to `inner` after a `ts <= cutoff` check; queries past the
cutoff return the natural absent-value (`nothing` for chains and
surfaces, `missing` for spots). Rate and div curves pass through
unfiltered -- they are math objects, not historical observations,
and evaluating them at any `ts` (past or future) is a legitimate
forward query rather than lookahead.

Not a Julia subtype of `ModelDataSource` (concrete structs are
final). Policies and agents declare the cut wrapper as their data
parameter explicitly: that gives the supported accessor interface a
no-lookahead boundary rather than relying on each call site to
remember the rule.

Surface and chain caches live on the inner `ModelDataSource`; cuts
do not invalidate them (surfaces at `ts <= cutoff` are immutable
historical facts).

## The engine

```julia
run_backtest(agent::Agent, source::ModelDataSource,
             from::DateTime, to::DateTime) -> Vector{Position}

run_backtest(policy::Policy, source::ModelDataSource,
             from::DateTime, to::DateTime) -> Vector{Position}
```

The loop:

```julia
positions = Position[]
for t in available_timestamps(source, from, to)
    cut    = TimeCutModelDataSource(source, t)
    policy = current_policy(agent, t, cut, positions)
    orders = decide(policy, t, cut, positions)
    for trd in orders
        qte      = resolve_quote(cut, trd, t)
        spot_val = get_spot(cut, t)
        push!(positions, open_position(trd, qte, Float64(spot_val)))
    end
end
return positions
```

That is the whole engine. The bare-`Policy` overload is a one-line
wrapper that delegates to `run_backtest(StaticAgent(policy), ...)`,
so a single driver path handles fixed-policy backtests and
agent-driven (refitting / learning) backtests alike. Policies (or
agents) that need a sub-range filter the schedule themselves.

### `resolve_quote`

```julia
resolve_quote(cut::TimeCutModelDataSource, trade::Trade, t::DateTime)
    -> OptionQuote
```

Looks up the `OptionQuote` in `get_chain(cut, t)` whose contract
matches `trade` exactly on `(underlying, strike, expiry, option_type)`.
Errors on absent chain or strike-not-found -- both indicate the
policy emitted a trade for a contract it should not have known
about.

Goes through `get_chain` rather than `get_surface` because surfaces
retain only inverted IVs; the raw bid/ask the fill needs lives on
the chain quote, not the slice.

## Key decisions

| Decision | Why |
|---|---|
| **Engine driven by `Agent`, not `Policy`** | The agent layer is where policy-evolution lives (refits, swaps, learning). Making the engine ask `current_policy` per tick means a fixed-policy backtest, a monthly-refit backtest, and an online-learning backtest all use the same loop. The bare-`Policy` overload exists only for ergonomics (and to make training/evaluation code that scores a single Policy concise). |
| **`run_backtest(policy, ...)` wraps `StaticAgent`** | One primitive, one wrapper. Keeps two entry points but a single tick-loop implementation; avoids two copies of the fill path drifting apart. |
| **Composition over inheritance for the cut wrapper** | Julia concrete structs are final, so `TimeCutModelDataSource <: ModelDataSource` is not an option. Composition (`inner::ModelDataSource` + `cutoff::DateTime`) plus parallel accessor methods is the idiomatic alternative. Code that needs the cut states it in the signature; helpers that take a raw `ModelDataSource` are not callable with a cut, which is correct -- those helpers do not respect the cutoff. |
| **No-lookahead at the type level** | Both `current_policy` and `decide` take `TimeCutModelDataSource`, not `ModelDataSource`. Through exported accessors, neither an agent's refit logic nor a policy's decision logic can accidentally reach future observations. The legacy codebase enforced this with a runtime wrapper passed in via an argument; the rebuild lifts it into the function signature. |
| **`resolve_quote` reads chains, not surfaces** | The rebuild's `RawSurface` stores only inverted IVs; raw bid/ask lives on the `OptionQuote`s in the chain. Going through the chain for fills keeps the spread-respecting semantics of the legacy codebase without forcing a price-from-IV path on every tick. (A future BS-priced-quote fill mode would dispatch off a separate trait on the data source.) |
| **Bare ledger return, no `BacktestResult`** | Reporting needs are not nailed down yet, and the legacy `BacktestResult.pnl` parallel vector breaks once close-as-counter-trade lands (a close `Position` does not have its own contribution -- it nets another fill). Returning `Vector{Position}` lets the reporting layer pick its own shape (per-tick cash flows, per-contract netting, ...) without committing now. |
| **Walk every available timestamp** | The legacy codebase let strategies provide an `entry_schedule(strategy)` that drove the engine loop. The rebuild walks every available `ts` and lets the policy gate inside `decide` (and the agent gate inside `current_policy`). Cost on minute-data: ~7ms per backtest-year from no-op calls -- negligible. Benefit: one engine loop shape, no special case for event-driven or monitoring policies, and the no-lookahead boundary always covers the current `t`. |
| **No `clear_cache!` between ticks** | Surface and chain caches stay warm across the whole loop. Long backtests that need to bound memory can call `clear_cache!(source)` themselves; the engine does not invent a policy. |

## Responsibility boundaries

**Owns:** `TimeCutModelDataSource`, the tick loop, the
`Trade -> OptionQuote -> Position` filling chain, the bare-`Policy`
convenience overload.

**Does NOT own:**

- Policy logic. That is the [`policies`](policies.md) module.
- Policy-evolution logic (refit cadence, learning, swaps). That is
  the [`agents`](agents.md) module.
- Data acquisition. That is the [`data`](data.md) and
  [`model_data`](model_data.md) modules.
- Reporting / PnL aggregation. Today the caller computes whatever
  it needs from the returned `Vector{Position}`.
- Concurrency. Single-threaded.

## Failure modes

| Condition | Behavior |
|---|---|
| `current_policy` returns a Policy whose `decide` returns `Trade[]` | normal; engine continues |
| `decide` emits trade for a contract not in the chain at `t` | `resolve_quote` errors |
| Spot missing at a tick where `decide` emits an order | `run_backtest` errors |
| Policy reads `get_surface(cut, t')` for `t' > t` | accessor returns `nothing` |
| Agent or policy never emits any trade | engine returns empty `Vector{Position}` |

## Future work

- **Result wrapper for reporting.** A `BacktestResult` carrying the
  ledger plus precomputed views (open-positions snapshots, per-tick
  cash flows) once the reporting module exists.
- **Sparse tick override.** Optional
  `tick_times(agent_or_policy, source) -> iterable` for sparse logic
  that precomputes its schedule and wants to skip the per-tick
  `current_policy` / `decide` call entirely.
- **BS-priced fills.** When a chain lacks bid/ask but a surface
  exists, an alternative `resolve_quote` mode would synthesize a
  quote from `price(surface, ...)` plus a configurable spread.
  Dispatch off a `QuoteConvention` trait on the data source.
- **Multi-asset backtest.** Today `get_spot(cut, t)` returns the
  single underlying's spot. Multi-symbol policies would need
  per-`Underlying` spot lookups during fills.

## Layout

```
src/backtest/
    time_cut.jl     # TimeCutModelDataSource
    engine.jl       # resolve_quote + run_backtest (Agent and Policy)

test/backtest/
    test_time_cut.jl
    test_engine.jl
```

All files are `include`d into the top-level `VolSurfaceAnalysis`
module; no submodule wrappers.
