# Status

This is the `rebuilt` branch -- a deliberate clean slate. The full codebase
lives on `master` and is the reference we mine from. Everything here is being
built up one small, deliberate piece at a time.

Rebuild order:

1. **Data** -- done.
2. **Modelling** (vol surface) -- done. `Curve`s, `surfaces` module,
   and `ModelDataSource` composition are in place.
3. **Positions** -- done. `Trade` / `Position` records and the pure
   `payoff` / `open_position` / `entry_cost` / `realized_pnl` primitives.
4. **Policy + Agent + backtesting** -- minimal slice landed.
   `Policy` abstract type with stateless `decide(p, t, cut, positions)
   -> Vector{Trade}`; `Agent` abstract type with `current_policy(a, t,
   cut, positions) -> Policy` (the layer that owns refit / learning /
   policy evolution); `StaticAgent` wraps a fixed Policy.
   `TimeCutModelDataSource` gives no-lookahead a supported-interface
   guarantee; `run_backtest(agent, ...)` drives the tick loop and
   `run_backtest(policy, ...)` is a `StaticAgent` wrapper for
   training / evaluation. Returns a bare `Vector{Position}` ledger.
   Reporting, result wrappers, and concrete policy / agent types
   (iron condor, strangle, walk-forward refit, ...) are next.
5. **Metric computation** -- done. `PnLSeries` round-trip
   intermediate (`src/metrics/pnl_series.jl`) plus always-on core
   metrics (`total_pnl`, `n_round_trips`, `hit_rate`) and a
   symbol-addressable optional set (`sharpe`, `sortino`,
   `max_drawdown`, `volatility`, `profit_factor`) dispatched by
   `compute_metrics`. Per-metric default kwargs live on the dispatch
   table; experiment-level overrides are deferred until a workflow
   needs them.
6. **Experiment orchestration** -- minimal end-to-end runnable.
   `Experiment` wires `(Agent, ModelDataSource, [from, to], requested
   metrics)` into a single rerunnable record; `run_experiment(exp)`
   returns an `ExperimentResult` with positions, the PnL intermediate
   (marked to spot at the window end), and the computed metrics.
   TOML configs resolve to `Experiment` values via `load_experiment`
   (stdlib `TOML` + per-sum-type builder registries), and
   `scripts/run_experiment.jl <config.toml>` prints the result via
   `Base.show(::IO, ::MIME"text/plain", ::ExperimentResult)`.
   Settlement uses the last available timestamp in the window.
   Per-leg-expiry settlement, persistence (sidecar dump of config +
   metrics), and parallel sweeps are future work.

Visualization is added incrementally alongside each stage, not as a phase
of its own.
