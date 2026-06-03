# Status

Where the project is now and what it's working on. This doc is deliberately
short and current -- the long-term shape lives in [vision.md](vision.md), the
rules we build under in [design.md](design.md), and deferred per-module work
in each [docs/modules/](modules) doc's "Future work". Status only pulls up the
few things actually in motion; when a focus is done its capability moves to
"Where we are" and the focus is replaced. No standing backlog lives here.

## Where we are

The end-to-end research path runs: a config-defined `Experiment` loads a
`ModelDataSource`, an `Agent` hands a `Policy` to the no-lookahead engine per
tick, fills accumulate into a ledger, and the ledger is aggregated and scored.
Concretely, all of the following exist and are tested:

- **Data** -- lazy, day-cached `ParquetDataSource` over Polygon OHLCV, with the
  bid/ask synthesis policy (`SpreadFromOHLCV`) declared per source so the fill
  assumption is visible in every run.
- **Surfaces** -- `build_surface` (mark-price IV inversion, OTM-side picking),
  self-contained BS pricing / greeks, and `invert_delta`.
- **Positions** -- immutable `Trade` / `Position` plus the pure
  `payoff` / `entry_cost` / `realized_pnl` primitives.
- **Engine** -- `Policy` / `Agent` split (RL convention), no-lookahead enforced
  at the type level by `TimeCutModelDataSource`, one concrete policy
  (`DailyShortStrangle`).
- **Metrics** -- round-trip `PnLSeries` (FIFO, per-leg settle closure, honest
  `n_unmarked`) feeding always-on core metrics plus a symbol-dispatched
  optional set.
- **Experiment + identity** -- `run_experiment`, TOML config loading, and
  content-addressed identity (`core_hash` / `full_hash`) over the resolved
  experiment.
- **Persistence** -- Hive-partitioned parquet knowledge base with code
  provenance; `save_run` / `load_run` round-trip; cross-run queries are DuckDB
  SQL against the parquet glob.

What this does **not** yet give us: a number we can trust as a measure of edge.
That is the current focus.

## Current focus: trustworthy, capital-aware results

The framework is well ahead of the evidence. Before adding breadth (more
policies, refitting agents, richer surfaces), the path that already runs has to
produce numbers that are *correct* and *comparable*. Three steps, in order --
each gates the next:

1. **Honest expiry settlement (case 2).** When `get_spot` at a leg's exact
   expiry minute is `missing`, the leg currently lands in `n_unmarked` and is
   dropped from PnL -- which silently discards a non-random slice of a
   held-to-expiry strategy's P&L (the flagship `DailyShortStrangle` is exactly
   this shape). Compute the leg's theoretical mark from the surface at (or just
   before) expiry. Lands in `experiment._build_settle`; transparent to the
   metrics contract.
2. **A capital base, so metrics are returns, not raw dollars.** Today
   `sharpe` / `sortino` / `volatility` annualize the std of *dollar* round-trip
   PnL -- they are per-trade dollar t-stats, not return ratios, and are not
   comparable across strategies of different size. Introduce a margin / notional
   base (the legacy `margin_by_key` / ROI-on-margin is the reference) and make
   the return-based metrics compute against it.
3. **One real backtest under CI.** Add a data-gated end-to-end test that runs a
   backtest over real parquet and asserts its metrics, auto-skipping where the
   data is absent (so CI and data-less machines skip cleanly). This is the first
   slice of the reproducibility harness: rerun saved runs and assert their
   `metrics` / `pnl_series` still match, guarding run identity against code
   drift.

Done when a single config produces a capital-normalized, settlement-complete
result that a CI test reproduces.

## Next up

- `invert_delta`: detect a monotonicity violation and return `nothing` rather
  than the last bisection midpoint -- the one place the code fabricates an
  answer instead of failing loudly.
- Write-to-temp-then-rename for atomic `save_run`.
- A second concrete policy (daily iron condor) once settlement is honest --
  then decide whether the leg duplication warrants a `Structure` abstraction.
