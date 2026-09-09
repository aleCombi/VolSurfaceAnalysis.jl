# Status

This is `master` -- the active clean-line codebase. The prior full codebase
lives on the `legacy` branch and remains the reference we mine from. Work here
advances toward the long-term shape in [vision.md](vision.md), one small,
deliberate piece at a time.

Progress toward vision:

1. **Data** -- done, redesigned 2026-09-07 around *kinds*
   (`docs/modules/market_data.md`): record types keyed by type with
   `timestamp` as visibility time and a selector per kind; four shapes
   (`at`, `between`, `asof`, `timestamps`) over a `MarketData` map of
   one provider per kind; per-storage specs (`ParquetOptionBars`,
   `ParquetSpots`, `Constant`, `InMemory`, `BySelector`) opened into
   run-scoped readers by `open_data` / `close_data!` / `with_data`;
   derived providers (`QuotesFromBars` with its `QuoteSynthesizer`,
   `SurfaceFrom`) that read through the map they are called from, so a
   `TimeCut` is a structural no-lookahead through derived data. The
   `data` module keeps the canonical records and the Polygon row
   mapping.
2. **Modelling** (vol surface) -- done. `Curve` types, `RateCurve` /
   `DivCurve` kinds, the `surfaces` module, and the `SurfaceFrom`
   provider with a bounded, cut-independent surface cache.
3. **Positions** -- done. `Trade` / `Position` records and the pure
   `payoff` / `open_position` / `entry_cost` / `realized_pnl` primitives.
4. **Policy + Agent + backtesting** -- minimal slice landed.
   `Policy` abstract type with stateless `decide(p, t, cut, positions)
   -> Vector{Trade}`; `Agent` abstract type with `current_policy(a, t,
   cut, positions) -> Policy` (the layer that owns refit / learning /
   policy evolution); `StaticAgent` wraps a fixed Policy.
   `TimeCut` gives no-lookahead a supported-interface guarantee;
   `run_backtest(agent, data, from, to, clock)` drives the tick loop on
   the experiment's declared `Clock` and
   `run_backtest(policy, ...)` is a `StaticAgent` wrapper for
   training / evaluation. Returns a bare `Vector{Position}` ledger.
   Reporting, result wrappers, and concrete policy / agent types
   (iron condor, strangle, walk-forward refit, ...) are next.
5. **Metric computation** -- done. `PnLSeries`
   (`src/metrics/pnl_series.jl`) is the canonical per-round-trip PnL
   series: it FIFO-matches position fills, emits one PnL sample per
   closed round trip or honestly settled residual lot, and counts
   unmarked residuals. Always-on core metrics (`total_pnl`,
   `n_round_trips`, `n_opens`, `n_closes`, `hit_rate`) are computed
   for every result. Optional metrics (`sharpe`, `sortino`,
   `max_drawdown`, `volatility`, `profit_factor`) are selected by
   symbol through `compute_metrics`; the `_METRIC_TABLE` in
   `src/metrics/dispatch.jl` maps each symbol to its function and
   default kwargs. Per-experiment overrides flow through
   `OutputSpec.metric_params`.
6. **Experiment orchestration** -- end-to-end runnable.
   `Experiment` wires `(Agent, MarketData specs, Clock, [from, to],
   OutputSpec)` into a single rerunnable record; `run_experiment(exp)`
   opens the data for the run and returns an
   `ExperimentResult` with positions, the `PnLSeries` (per-leg
   settled), and the computed metrics. Outputs are declared in config:
   an `[outputs]` table (`metrics`, per-metric params, `artifacts`)
   resolves to an `OutputSpec`, defaulting to all registered metrics and
   the default artifact set when omitted. TOML configs (`[data.<kind>]`
   tables plus a `clock`) resolve via `load_experiment` (stdlib `TOML`
   + per-sum-type builder registries);
   `scripts/run_experiment.jl <config.toml> [--save] [--out-dir <dir>]`
   prints the result, and optionally persists it / renders artifacts.
   Parallel sweeps are future work.
7. **Persistence + identity** -- `RunStore` writes runs to a
   Hive-partitioned parquet tree at `<root>/runs/run_id=<full_hash>/`
   (config.toml verbatim, manifest / metrics / positions / pnl_series
   parquet, and an `artifacts/` subdir). Identity is canonical and
   layered: `full_hash(experiment)` is the run id; `core_hash` (data +
   clock + agent + window) is shared by output variations of one
   backtest. Both
   come from a `to_dict` projection (`experiment/identity.jl`) over the
   *resolved* experiment, so whitespace / key order / `name` / cache
   knobs don't fork ids. Every run records code provenance
   (`commit_sha` / `dirty` from `code_provenance`). `save_run` writes,
   `load_run` reads back into an `ExperimentResult` (specs are pure
   values, so loading works off-machine; a manifest `schema_version`
   guards the one-time id break of the data-kinds migration). Cross-run
   queries are DuckDB
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
caller-supplied `settle(trade) -> Union{Float64, Missing}` closure
instead of a single scalar. Each residual lot settles at the spot of
**its own trade's underlying** at `min(trade.expiry, window_end)`, the
same selector the engine priced its fill against, and is stamped at the
leg's own expiry; a lot whose spot is missing there counts in
`PnLSeries.n_unmarked` and is excluded from realized PnL (no silent
fallback). `window_end_spot` is provenance only. The clock selector says
*when* to step, not whose price, so `load_experiment` asserts a declared
policy underlying matches it -- one experiment, one underlying.
`scripts/run_experiment.jl --out-dir <dir>` renders the equity-curve
artifact from any config (via `scripts/lib/artifacts.jl` + `viz/pnl.jl`).

The review of the data-kinds branch (PR #9) found six correctness
defects that were one stance: an unanswerable question reported as an
ordinary empty result, so every consumer downstream correctly concluded
it had nothing to do. That stance is now design rule 7 and the
`market_data` protocol enforces it: `serves` answers the structural
question and the map-level shapes check it; two spot rows at one instant
collapse if identical and throw `ConflictingRecords` if they disagree,
on every spot shape including `asof`; the surface `asof` walks back under
a `lookback_ticks` bound and throws `DerivationExhausted` past it; a
`Constant` honours its visibility stamp in every shape; settlement
follows each lot's own trade, with `load_experiment` asserting a declared
policy underlying matches the clock selector. The partition convention is
time-ordered with a one-day spill allowance, and SQL range bounds keep
millisecond precision. One regression testset per finding lives in
`test/regressions/test_review_findings.jl` and is part of the gate.
**Gate on the DevBox (2 cores, 3.7 GB, Julia 1.12.7): 1206 passed, 0
failed.** What the review deferred is in the backlog below.

## In flight

- **Ledger rebuild (proposal).** Review of the fill-vector ledger found
  the lifecycle split over three layers (engine fills, `pnl_series`
  matches, `run_experiment` settles) with no shared record, plus a
  FIFO float-residue defect, per-share units labelled USD, and per-leg
  sampling that inflates the annualised ratios. The plan -- typed event
  ledger booked inside the run, `Order` with intent and structure id out
  of `decide`, a `Book` view in, lifecycle rules in core identity,
  metrics over `round_trips` and an equity curve -- is in
  [docs/proposals/ledger.md](proposals/ledger.md). Decisions in its
  section 4 are open; no code yet.

## Backlog

Backlog items are concrete parked work: visible enough to preserve the
intended direction, but not currently in flight.

- **Leaning out the architectural docs.** Pass over `docs/modules/*`
  (and the top-level docs) to bring them in line with design rule 6 --
  invariants and boundaries kept, drift-prone implementation detail
  (magic numbers, internal data structures, incidental library names,
  API walkthroughs) dropped. Motivation: resuming the library after a
  few-week pause, the docs should be the trustworthy entry point to read
  back in from. `data.md` is the first pass / template; the other module
  docs follow. `market_data.md` (new) follows the template from the
  start. Parked after PR #9 (2026-09-08); no slice in progress.
- **Settlement rule (replaces "surface-based theoretical settle").**
  A contract's `expiry` is stamped at parse time as the listed date at
  16:00 ET; settlement reads that instant as the last price the market
  put on the contract, which it is not. On the ten-year strangle run
  (`5700d3f242f8132e`) 1691 of 1700 expiry instants have a spot; the
  nine misses are calendar, not sparse data: six early-close sessions
  (the official close was 13:00 ET), two unscheduled closures
  (2018-12-05, 2025-01-09, where the OCC settled against the previous
  session's close), and the final pair past the end of the data. Real
  mechanics for SPY: exercise by exception, intrinsic against the
  official close of the last session on or before the listed expiry
  date. The component: a settlement rule owned by the experiment,
  answering (1) the settlement session (last session on or before the
  listed date), (2) its close instant and reference price (the last
  regular-session spot print stands in for the auction), (3) the payoff
  (intrinsic), and separately (4) the mark for a leg still open past the
  window end (the contract's own quote mark at the window end, surface
  price as fallback; today it is intrinsic at the window-end spot, and
  the sample is stamped at the expiry rather than the mark's instant, so
  the equity curve runs past `exp.to`). Sessions derived from the spot
  tree (a date is a session if the underlying printed in regular hours;
  its close is the last print at or before 16:00 ET) rather than a static
  calendar, with an override hook. In core identity, so one more id
  break. Not a surface problem at all. Parked 2026-09-08: 18 of 4480 legs
  plus the final pair, all with a known correct answer; revisit with the
  first policy that holds past a session close by design.
- **Second concrete policy** -- on deck once settlement is
  honest. Candidate: a daily iron condor (same scheduled-gate /
  `invert_delta` shape, four legs instead of two). Once the duplication
  is visible, decide whether to extract a `Structure` abstraction
  (`policies.md` Future work) or keep policies as 4-leg inline
  `decide` bodies. Parked 2026-09-08 behind the settle item.
- **Reproducibility harness for stored runs.** Opt-in, data-gated tests
  that rerun each saved run (`load_run` -> `run_experiment`) and assert its
  `metrics` / `pnl_series` still match, auto-skipping where the source data
  is absent (so CI / data-less machines skip cleanly); plus a
  `scripts/revalidate_runs.jl` utility that refreshes a run's `commit_sha` /
  `dirty` when a rerun reproduces it, and *flags* divergences rather than
  overwriting. Run identity is config-derived (one result per `run_id`), so
  this is what guards that invariant against code drift. Not started.
- **Dataset fingerprint in identity.** The parquet specs carry their
  root in a reserved `dataset` slot of the identity projection; a real
  logical dataset id and version (so the same tree at two paths, or a
  re-collected tree at one path, hash right) is its own design note.
  Declined in data-kinds v3.
- **Capability-restricted views.** A structural raw/model boundary (a
  policy view that cannot address `OptionBar`) was declined in
  data-kinds v3 in favour of a doc rule; revisit if a policy ever
  couples to vendor bars.
- **Bar-end timestamp convention as a spec option.** Polygon minute
  bars keep their bar-open stamp as the visibility time, a documented
  one-minute allowance. A `stamp = :bar_end` option on
  `ParquetOptionBars`, in identity, would make the choice explicit per
  experiment.
- **Path metrics over simultaneous samples.** `pnl_series` orders
  samples at one timestamp by pnl (losses first) so `max_drawdown` is
  deterministic; aggregating simultaneous samples for path metrics is
  the fuller answer.
- **Quote synthesis cost (PR #9 finding B). Closed 2026-09-08, measured,
  not worth a cache.** `at(::QuotesFromBars, ...)` re-synthesizes the
  chain on every call and one firing tick performs `n + 2` passes. On the
  DevBox against the SPY tree (2024-01-16) a one-minute chain holds
  300-500 bars, one synthesis pass costs 0.01-0.05 ms, and the parquet
  read under it costs 5-45 ms cold: the repeated work is two to three
  orders of magnitude below the read it sits on, so the 20% whole-run
  threshold is unreachable. No chain cache, no engine change; the
  per-order chain fetch in `run_backtest` stays as it is. Reopen only
  with a policy on the full minute grid and a whole-run measurement that
  says otherwise.
- **Reader and SQL duplication in `market_data/parquet.jl`.**
  `ParquetBarsReader` and `ParquetSpotsReader` repeat open, close,
  partition listing, the backward walk and the grid, differing only in
  how a timestamp is read from a partition; unifying them would have
  closed the spot `asof` gap by construction. The SQL timestamp formatter
  duplicates one in the store module and the path quoter one in the
  polygon module; extraction needs a home across module boundaries.
