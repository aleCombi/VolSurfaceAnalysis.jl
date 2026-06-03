# Status

This is the `rebuilt` branch -- a deliberate clean slate. The full codebase
lives on `master` and is the reference we mine from. Everything here is being
built up one small, deliberate piece at a time.

Rebuild order:

1. **Data** -- done. Includes the `OptionBar` + `QuoteSynthesizer`
   adapter that closes the Polygon OHLCV → bid/ask gap; data sources
   declare their synthesizer (e.g. `SpreadFromOHLCV(0.7)`) at construction.
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
6. **Experiment orchestration** -- end-to-end runnable.
   `Experiment` wires `(Agent, ModelDataSource, [from, to], OutputSpec)`
   into a single rerunnable record; `run_experiment(exp)` returns an
   `ExperimentResult` with positions, the PnL intermediate (per-leg
   settled), and the computed metrics. Outputs are declared in config:
   an `[outputs]` table (`metrics`, per-metric params, `artifacts`)
   resolves to an `OutputSpec`, defaulting to all registered metrics and
   the default artifact set when omitted. TOML configs resolve via
   `load_experiment` (stdlib `TOML` + per-sum-type builder registries);
   `scripts/run_experiment.jl <config.toml> [--save] [--out-dir <dir>]`
   prints the result, and optionally persists it / renders artifacts.
   Parallel sweeps are future work.
7. **Persistence + identity** -- `RunStore` writes runs to a
   Hive-partitioned parquet tree at `<root>/runs/run_id=<full_hash>/`
   (config.toml verbatim, manifest / metrics / positions / pnl_series
   parquet, and an `artifacts/` subdir). Identity is canonical and
   layered: `full_hash(experiment)` is the run id; `core_hash` (source +
   agent + window) is shared by output variations of one backtest. Both
   come from a `to_dict` projection (`experiment/identity.jl`) over the
   *resolved* experiment, so whitespace / key order / `name` / cache
   knobs don't fork ids. Every run records code provenance
   (`commit_sha` / `dirty` from `code_provenance`). `save_run` writes,
   `load_run` reads back into an `ExperimentResult` (source validates
   lazily, so loading works off-machine). Cross-run queries are DuckDB
   SQL against the parquet glob. Compute reuse (skip the backtest on a
   `core_hash` hit) and a curation gate are the next slices.

Visualization is added incrementally alongside each stage, not as a phase
of its own.

First concrete trading policy landed alongside step 4:
`DailyShortStrangle` (target |Δ| per leg via `invert_delta`, snap to
chain quotes of the required option type, single entry time per day,
fixed quantity, expiry by interval). TOML builder + smoke config under
`configs/`; `scripts/delta_map_demo.jl` visualizes the strike↔|Δ| map
for sanity checks against real SPY surfaces.

Step 5 / 6 then gained per-leg expiry settlement: `pnl_series` takes a
caller-supplied `settle(expiry) -> Union{Float64, Missing}` closure
instead of a single scalar; held-to-expiry legs are stamped at their
own `trade.expiry` using `get_spot(source, expiry)`; legs whose expiry
is past the experiment window mark at the window-end spot (case 1);
legs whose expiry-time spot is unavailable inside the window count in
`PnLSeries.n_unmarked` and are excluded from realized PnL (case 2 --
no silent fallback). `scripts/run_experiment.jl --out-dir <dir>` renders
the equity-curve artifact from any config (via `scripts/lib/artifacts.jl`
+ `viz/pnl.jl`).

## In flight

- **Surface-based theoretical settle for case 2.** When `get_spot` at
  the leg's exact expiry is `missing` (Polygon minute bars are sparse
  at the 16:00 ET close minute), today's policy returns `missing` and
  the lot is unmarked. The fix is to compute the leg's theoretical
  mark from the surface at (or just before) the expiry. Lands in
  `experiment._build_settle`; transparent to the metrics contract.
- **Second concrete policy** -- on deck once case-2 settlement is
  honest. Candidate: a daily iron condor (same scheduled-gate /
  `invert_delta` shape, four legs instead of two). Once the duplication
  is visible, decide whether to extract a `Structure` abstraction
  (`policies.md` Future work) or keep policies as 4-leg inline
  `decide` bodies.
