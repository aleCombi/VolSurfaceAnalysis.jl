# `experiment` module

One-shot orchestrator: wires `(Agent, ModelDataSource, time window,
requested metrics)` into a single rerunnable record. Owns one
struct, one result wrapper, one entry point. Train / val / test
splits, refit cadence, and learning live inside the
[`Agent`](agents.md); the `Experiment` only sees the evaluation
window.

## Data flow

```mermaid
flowchart LR
    Exp[Experiment] --> RB([run_backtest])
    RB -->|positions| PS([pnl_series])
    Exp -->|spot at to| PS
    PS -->|PnLSeries| CM([compute_metrics])
    Exp -->|requested| CM
    CM -->|NamedTuple| ER[ExperimentResult]
    PS -->|series| ER
    RB -->|positions| ER
    Exp -->|provenance| ER
```

Per call to `run_experiment`: tick the engine, resolve a settlement
spot at the last available timestamp `<= exp.to`, aggregate the
ledger into a `PnLSeries`, compute always-on plus requested optional
metrics, and pack everything (including the originating `Experiment`)
into one `ExperimentResult`.

## The abstraction

```julia
struct Experiment
    name    :: String
    agent   :: Agent
    source  :: ModelDataSource
    from    :: DateTime
    to      :: DateTime
    metrics :: Vector{Symbol}
end

Experiment(; name, agent, source, from, to, metrics=Symbol[])

struct ExperimentResult
    experiment :: Experiment
    positions  :: Vector{Position}
    pnl_series :: PnLSeries
    metrics    :: NamedTuple
end

run_experiment(exp::Experiment) -> ExperimentResult
```

`Experiment.metrics` lists the *optional* metrics the run wants;
always-on core metrics (`:total_pnl`, `:n_round_trips`, `:n_opens`,
`:n_closes`, `:hit_rate`) always appear in `result.metrics` and are
not listed in `metrics`.

### Rerun

```julia
res  = run_experiment(exp)
res2 = run_experiment(res.experiment)   # same inputs, same result
```

The full `Experiment` rides in `ExperimentResult.experiment`, so a
single function call reproduces the run without needing the caller
to remember any other state.

### Train / val / test splits live on the Agent

`Experiment.(from, to)` is the *evaluation* window. Anything
sub-windowed (train on `[from, t_split]` and evaluate on
`[t_split, to]`, walk-forward refits inside the window, lookback
buffers warmed on data *before* `from`) is the `Agent`'s concern.
The Agent receives a `TimeCutModelDataSource` per tick whose `inner`
is the full `ModelDataSource` -- it can read history before `from`
freely, and only data strictly after the current tick is blocked.

## Key decisions

| Decision | Why |
|---|---|
| **`run_experiment`, not `run`** | `Base.run` is exported and dispatches on `Cmd`; shadowing it for a domain verb is exactly the convention warning every Julia style guide gives. `run_experiment` also reads as a peer of `run_backtest`. |
| **Result carries the full `Experiment`, not just `name`** | Rerun is the primary use case for provenance. `run_experiment(result.experiment)` is the obvious primitive; a bare `name` would force a sidecar registry to look up the rest. The cost is one cheap struct reference. |
| **Settlement spot = `get_spot(source, last_ts_in_window)`** | The metrics layer needs a number, not a policy. Using the last available timestamp `<= exp.to` keeps the contract single-decision and avoids per-leg expiry lookups in this slice. Documented as "mark-to-spot at window end" so `total_pnl` semantics are unambiguous; per-leg-expiry settlement is future work. |
| **Always-on metrics not in `Experiment.metrics`** | They are computed unconditionally and cost nothing extra. Listing them in `metrics` would force every experiment to repeat a boilerplate list and would imply they were opt-in, which they are not. |
| **`metrics::Vector{Symbol}`, not `Vector{Function}`** | Symbols survive serialization to disk, read cleanly in config dumps, and let `compute_metrics` carry the per-symbol default kwargs in one place ([`compute_metrics`](metrics.md)). Function references would skip the table at the cost of looking less like a config artifact. |
| **No per-metric kwargs on `Experiment` yet** | The optional kwargs path lives on `compute_metrics` (`kwargs::Dict{Symbol,NamedTuple}`); callers who want non-default Sharpe can compute it themselves on `result.pnl_series`. Promotes the override to `Experiment` only once a workflow needs the kwargs to survive into provenance. |
| **No persistence in this slice** | Writing results to disk, building a knowledge base of completed experiments, and indexing by name / config hash are the reporting layer's job. `Experiment` and `ExperimentResult` are pure value types today; serialization is a downstream decision. |
| **`run_experiment` errors loudly on missing data** | Empty time window or a missing settlement spot both indicate the experiment is mis-specified or the data source has gaps the caller did not expect. Silent zeros would invent a "result" that doesn't exist. |

## Responsibility boundaries

**Owns:** the `Experiment` struct, the `ExperimentResult` wrapper,
the `run_experiment` entry point, and the policy choice for
resolving the settlement spot at the window end.

**Does NOT own:**

- Tick loop and fill semantics. That is the [backtest
  engine](backtest.md).
- Policy logic. That is the [`policies`](policies.md) module.
- Policy evolution / refit cadence / learning. That is the
  [`agents`](agents.md) module.
- Metric implementations and their dispatch table. That is the
  [`metrics`](metrics.md) module.
- Persistence, plotting, knowledge-base writes. Downstream
  layers, not landed yet.

## Failure modes

| Condition | Behavior |
|---|---|
| Empty time window (`available_timestamps(source, from, to)` empty) | `run_experiment` errors with the window and experiment name. |
| Settlement spot missing at the last in-window timestamp | `run_experiment` errors with the timestamp and experiment name. |
| `exp.metrics` contains an unknown symbol | `compute_metrics` errors with the offending symbol and the known list. |
| Agent / Policy never trades | `result.positions` and `result.pnl_series.pnl` are empty; always-on metrics are `0.0` / `0` / `NaN` per their empty-series conventions. |
| Spot present but no fills happened | Settlement spot is recorded on `pnl_series` even when unused -- harmless and keeps the field non-optional. |

## Future work

- Per-leg expiry settlement (Q1 in the design discussion): mark
  each still-open residual at `get_spot(source, leg.expiry)`
  instead of one global spot. Strictly more correct; one decision
  per leg instead of one per run.
- Persistence: serializable projection of `Experiment` (name +
  agent descriptor + source descriptor + window + metrics) plus a
  knowledge-base writer that indexes completed `ExperimentResult`s.
- Parallel sweeps: an `experiments::Vector{Experiment}` runner that
  parallelizes across runs (the engine is single-threaded; the
  parallelism layer is here).
- Config-file loading: TOML / YAML descriptors that resolve to
  `Experiment` values, so a rerun can come from a file rather than
  a Julia expression.
- Live-trading sibling: same `Experiment` shape with `run_backtest`
  swapped for a live loop driver.

## Layout

```
src/experiment/
    experiment.jl     # Experiment + ExperimentResult + run_experiment

test/experiment/
    test_experiment.jl
```

All files are `include`d into the top-level `VolSurfaceAnalysis`
module; no submodule wrappers.
