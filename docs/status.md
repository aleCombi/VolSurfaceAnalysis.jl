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
3. **Ledger** -- the journal of economic facts
   (`docs/modules/ledger.md`): `Order` with declared intent per leg,
   `Fill` / `Match` / `Expiry` / `Fee` events with a bitemporal header,
   the order journal (`OrderRecord`, `LegObservation`) recording what
   each decision saw, `Book` by replay (as known by sequence, as true by
   effective time), `round_trips`, cash in whole USD cents, and one
   validated write path with named failures; `record_order!` books a
   structure whole or not at all with the group minted inside the
   transaction. Replaced the `positions` module (slice 2 of the ledger
   rebuild).
4. **Policy + Agent + backtesting** -- on the ledger. `Policy` with
   stateless `decide(p, t, cut, book::Book) -> Vector{Order}`; `Agent`
   with `current_policy(a, t, cut, book) -> Policy` (the layer that owns
   refit / learning / policy evolution); `StaticAgent` wraps a fixed
   Policy. `TimeCut` gives no-lookahead a supported-interface guarantee;
   `run_backtest(agent, data, from, to, clock; fill_rule, cost_model,
   tick_cents)` drives the tick loop on the experiment's declared
   `Clock`, prices every leg of every order through the simulated venue
   (`docs/modules/backtest.md`: `:cross_spread` on the tick, IBKR Pro's
   US options commissions as `Fee` events) before anything is written,
   and returns a `Ledger`; per-record `check_join` validates the
   fill-to-order join at each append. Lifecycle (expiries in the tick
   loop) is slice 3.
5. **Metric computation** -- on the ledger. `PnLSeries` is built by
   `pnl_series(::Ledger)` from the ledger's round trips, one sample per
   structure closed at one instant, in USD. Always-on core metrics
   (`total_pnl`, `n_round_trips`, `n_opens`, `n_closes`, `hit_rate`)
   are computed for every result. Optional metrics (`sharpe`, `sortino`,
   `max_drawdown`, `volatility`, `profit_factor`) are selected by
   symbol through `compute_metrics`; the `_METRIC_TABLE` in
   `src/metrics/dispatch.jl` maps each symbol to its function and
   default kwargs. Per-experiment overrides flow through
   `OutputSpec.metric_params`. `window_end_spot` and `n_unmarked` are
   placeholders until slice 5 brings the structure series and the
   equity curve.
6. **Experiment orchestration** -- end-to-end runnable.
   `Experiment` wires `(Agent, MarketData specs, Clock, [from, to],
   OutputSpec)` into a single rerunnable record; `run_experiment(exp)`
   opens the data for the run and returns an `ExperimentResult` with
   the ledger, the `PnLSeries` and the computed metrics; open lots at
   the window end stay open, nothing is force-settled. Outputs are
   declared in config: an `[outputs]` table (`metrics`, per-metric
   params, `artifacts`) resolves to an `OutputSpec`, defaulting to all
   registered metrics and the default artifact set when omitted. TOML
   configs (`[data.<kind>]` tables plus a `clock`) resolve via
   `load_experiment` (stdlib `TOML` + per-sum-type builder registries);
   `scripts/run_experiment.jl <config.toml> [--save] [--out-dir <dir>]`
   prints the result, and optionally persists it / renders artifacts.
   Parallel sweeps are future work.
7. **Persistence + identity** -- `RunStore` writes runs to a
   Hive-partitioned parquet tree at `<root>/runs/run_id=<full_hash>/`
   (config.toml verbatim, manifest / metrics / events / orders /
   order_legs / pnl_series parquet, and an `artifacts/` subdir).
   Identity is canonical and layered: `full_hash(experiment)` is the
   run id; `core_hash` (data + clock + agent + window) is shared by
   output variations of one backtest. Both come from a `to_dict`
   projection (`experiment/identity.jl`) over the *resolved* experiment,
   so whitespace / key order / `name` / cache knobs don't fork ids.
   Every run records code provenance (`commit_sha` / `dirty` from
   `code_provenance`). `save_run` writes, `load_run` rebuilds the ledger
   through `commit!` and `check_join` and reads back into an
   `ExperimentResult` (specs are pure values, so loading works
   off-machine; the manifest `schema_version`, now 3, refuses runs
   written under the positions schema). Cross-run queries are DuckDB
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

Step 5 / 6 had gained per-leg expiry settlement through a caller-supplied
`settle(trade)` closure in `pnl_series`, marking each residual lot at its
own underlying's spot at `min(expiry, window_end)`. The ledger rebuild
superseded it: settlement is a lifecycle event booked in the tick loop
(slice 3), open lots at the window end stay open until the equity curve
marks them (slice 5), and the closure, the window-end spot lookup and
the fill-vector builder are gone with slice 2. What survives: the clock
selector says *when* to step, not whose price, and `load_experiment`
asserts a declared policy underlying matches it -- one experiment, one
underlying. `scripts/run_experiment.jl --out-dir <dir>` renders the
equity-curve artifact from any config (via `scripts/lib/artifacts.jl` +
`viz/pnl.jl`).

The review of the data-kinds branch (PR #9) found six correctness
defects that were one stance: an unanswerable question reported as an
ordinary empty result, so every consumer downstream correctly concluded
it had nothing to do. That stance is now design rule 7 and the
`market_data` protocol enforces it: `serves` answers the structural
question and the map-level shapes check it; two spot rows at one instant
collapse if identical and throw `ConflictingRecords` if they disagree,
on every spot shape including `asof`; the surface `asof` walks back under
a `lookback_ticks` bound and throws `DerivationExhausted` past it; a
`Constant` honours its visibility stamp in every shape; every leg is
priced against its own underlying, with `load_experiment` asserting a
declared policy underlying matches the clock selector. The partition convention is
time-ordered with a one-day spill allowance, and SQL range bounds keep
millisecond precision. One regression testset per finding lives in
`test/regressions/test_review_findings.jl` and is part of the gate.
**Gate on the DevBox (2 cores, 3.7 GB, Julia 1.12.7): 1206 passed, 0
failed.** What the review deferred is in the backlog below.

## In flight

- **Ledger rebuild (proposal).** The fill-vector ledger splits the
  lifecycle over three layers (engine fills, `pnl_series` matches,
  `run_experiment` settles) with no shared record, plus a FIFO
  float-residue defect, per-share units labelled USD, and per-leg
  sampling that inflates the annualised ratios. The plan is an event
  journal booked inside the run: `Order` with intent out of `decide`, a
  `Book` view in, `Fill` / `Match` / `Expiry` / `Fee` events with a
  bitemporal header, an order journal outside the ledger holding the
  quotes decisions saw, contract facts / simulated venue / named
  simplifications as three identity-bearing config values, ratios on a
  daily equity curve. Revised after four reviews; see
  [docs/proposals/ledger.md](proposals/ledger.md), whose section 3
  decisions are being settled slice by slice. **Slice 1 landed
  2026-09-11**: the pure
  `ledger` module ([docs/modules/ledger.md](modules/ledger.md)) -- the
  order and event vocabulary, the contract table, the cash rules, the
  validated write path, the book with both replays, `round_trips`, and
  the `pnl_series(::Ledger)` adapter so today's metrics read a ledger
  unchanged -- with its tests on hand-built ledgers. The engine still
  runs on `positions` until slice 2. Gate after slice 1: 1626
  passed, 0 failed. **Review 2026-09-12**
  ([ledger-slice1-review.md](proposals/ledger-slice1-review.md)): not
  mergeable. The write path accepts batches the invariants forbid (a
  caller-supplied spec at `commit!` that the replays ignore, consumption
  effective before its open, a partial or early expiry, a non-FIFO
  match) and there is no structure-level atomic writer; the driver's
  review added that book equality is exact float while the two replays
  add in different orders. Every finding is pinned as a failing testset
  in `test/ledger/test_review_findings.jl`, so the gate went red on
  purpose: 1630 passed, 15 failed, 1 errored, all in that file. **Fix
  round landed 2026-09-12**
  ([ledger-slice1-fix.md](proposals/ledger-slice1-fix.md)): `commit!`
  resolves contract facts from the table itself, with no caller-supplied
  spec; every reference must point backward in effective time and the
  effective replay folds equal instants in sequence order; FIFO is
  checked on append; an expiry settles the whole remaining lot at or
  after the contract's expiry; cash is whole USD cents everywhere inside
  the ledger, with `contract_cents` as the one rounding point
  (`NonIntegralCash` refuses a price that is not whole cents per
  contract) and fee shares by cumulative rounding, so book equality and
  the trips-to-cash reconciliation are exact. Gate after the fix round:
  1673 passed, 0 failed, 1 broken. **Hardening round landed 2026-09-12**
  ([ledger-slice1-hardening.md](proposals/ledger-slice1-hardening.md);
  the inventory is
  [ledger-slice1-coverage.md](proposals/ledger-slice1-coverage.md)):
  every invariant and named failure the module documents is mapped to
  its enforcing code and its test; `test_review_findings.jl` is
  dissolved into the suites beside the behaviour they check; documented
  promises the code did not enforce now are (an expiry's outcome agrees
  with its intrinsic value; recorded time is at or after effective time
  and nondecreasing along sequence, `RecordedOutOfOrder`; prices are
  finite, `InvalidPrice`; an unminted id at `event` and a non-positive
  join id are `DanglingReference`). The rule additions are listed in the
  coverage document for veto. Codex's review of the round
  ([ledger-slice1-hardening-review.md](proposals/ledger-slice1-hardening-review.md))
  found the fill review's construction-time post-expiry check missing:
  `FillAfterExpiry` is now thrown by the `Fill` constructor too, and
  every rejection in every failure testset checks the ledger snapshot
  and the book. Gate after hardening: 2306 passed, 0 failed, 1 broken,
  the structure-atomicity testset waiting for slice 2. **Slice 2 landed
  2026-09-12** ([ledger-slice2.md](proposals/ledger-slice2.md)): the
  engine computes, the ledger records. `decide` takes the `Book` and
  returns `Order`s; the venue (`src/backtest/execution.jl`) prices every
  leg through `:cross_spread` on the class's tick and IBKR Pro's US
  options commissions as `Fee` events; `record_order!` books a
  structure whole or not at all with the group minted inside the
  transaction; the order journal (`OrderRecord`, `LegObservation`) lives
  in the `Ledger` and `check_join` validates the fill-to-order join at
  each engine append, before persistence write and on load;
  `DuplicateExecution` refuses a repeated
  execution id; `positions` is retired and persistence writes
  `events` / `orders` / `order_legs` under schema version 3, so stored
  runs written under version 2 (the ten-year strangle
  `5700d3f242f8132e`) rerun from their configs. The structure-atomicity
  testset is `@test`. **Gate after the slice 2 fix round: 2893 passed,
  0 failed, 0 errored,
  2 broken.** The two Broken wait for slice 3: an expiry inside the
  window is booked as an `Expiry` (`test/experiment/test_experiment.jl`)
  and the PR #9 regression "settlement uses trade underlying" settles
  its in-window leg by an `Expiry` against the lot's own underlying
  (`test/regressions/test_review_findings.jl`). Next: slice 3, lifecycle
  in the tick loop; then the one auditable strangle run.

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
