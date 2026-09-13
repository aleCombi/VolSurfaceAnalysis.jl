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
   settlement_rule, tick_cents)` drives the tick loop on the
   experiment's declared `Clock`, prices every leg of every order
   through the simulated venue (`docs/modules/backtest.md`:
   `:cross_spread` on the tick, IBKR Pro's US options commissions as
   `Fee` events) before anything is written, and returns a `Ledger`;
   per-record `check_join` validates the fill-to-order join at each
   append. Lifecycle runs first at every tick and once more at the
   evaluation endpoint: `settlements` says which lots fell due and at
   what price, the ledger's own `record_expiry!` books each one, and the
   settlement rule (`:session_close`) reads sessions off the spot tree,
   consulting the NYSE calendar only to contradict it.
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
superseded it: settlement is a lifecycle event booked in the tick loop,
open lots at the window end stay open until the equity curve marks them
(slice 5), and the closure, the window-end spot lookup and the
fill-vector builder are gone with slice 2. What survives: the clock
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
  (`test/regressions/test_review_findings.jl`).
  **Slice 3 landed 2026-09-13**
  ([ledger-lifecycle.md](proposals/ledger-lifecycle.md)): expiries are
  booked in the tick loop, and the parked "Settlement rule" backlog item
  lands with them. `settlements` (`src/backtest/settlement.jl`) is
  `fill_legs`' twin -- a function of the cut and the book returning the
  lots falling due in `(prev, t]` paired with the price each settles at
  -- and the loop calls the ledger's own `record_expiry!`, one call per
  lot, so the engine still defines nothing that mutates. The rule is a
  symbol through a table in the `_FILL_RULES` style: under
  `:session_close` a date is a session when the underlying printed
  in a window running from 09:30 ET to the earlier of 16:00 ET and its
  own expiry, and its close is the last of those prints, which settles
  the early closes with no early-close table;
  BusinessDays.jl's `USNYSE` is consulted only to contradict the tree, so
  a printless date the calendar calls open is
  `UnpriceableLeg(:unexpected_gap)`, warned about once and left open
  (design rule 7). An expiry is
  effective at the contract's expiry and recorded at the tick that
  booked it, which is the sole source of the two replays disagreeing at
  an intermediate instant. The two Broken flipped. The settlement rule
  is not in config or identity yet (that is the next slice), so this
  round changes results under unchanged run ids, and nothing on disk
  tells the two apart: `load_run` checks the schema number, not which
  code produced the run, and schema 3 was already being written before
  lifecycle existed, so a stored pre-lifecycle run loads clean under the
  same run id as a run of the same config made now. Separating them is
  the identity work of the next slice. What the deleted backlog entry
  parked and this round deliberately does not land: the mark for a leg
  still open past the window end (the contract's own quote mark there,
  surface price as fallback) belongs to the equity curve, slice 5.
  **Gate after slice 3 and both rounds of review fixes: 3362 passed, 0
  failed, 0 broken.** The ten-year strangle
  (`configs/strangle_spy_16d_1dte.local.toml`, 2016-03-28 to 2026-03-27,
  1-DTE SPY, one contract per leg) now closes
  itself: 2240 orders and 4480 opening fills produce 4478 `Expiry`
  events over 1699 expiry instants, 4478 expired round trips and no
  warnings at all. The old backlog entry's nine misses resolve as eight
  in-window instants -- six early closes (2017-11-24, 2018-07-03,
  2019-07-03, 2022-11-25, 2024-12-24, 2025-07-03, each settling at its
  13:00 ET print) and two unscheduled closures (2018-12-05, 2025-01-09,
  each settling at the previous session's close) -- plus a ninth instant,
  the final pair, whose 2026-03-30 expiry is *past* the window end and so
  is never examined: those two lots stay open, as decision 8 says they
  should. **Review fix, finding 1:** lifecycle runs before the fill, so
  a lot opened at the very instant its contract expires escaped the
  interval that would have settled it -- and every later one, including
  the window-end pass -- and stayed open with no `Expiry` and no
  warning. The venue is now stricter than the ledger about expiry:
  `fill_legs` refuses a leg whose contract expires at or before the tick
  (`UnpriceableLeg(:expired_contract)`), which is what the interval's
  completeness rests on. `src/ledger/` is untouched; it still accepts a
  fill effective at the expiry instant. **Review fix, finding 2:** the
  reference window ended at the candidate date's 16:00 ET whatever the
  contract's own expiry, so an intraday expiry settled at a print from
  after it expired -- allowed by the tick's cut, but incoherent in the
  effective-time replay, where the `Expiry`'s own instant carried a
  price that did not exist then. The window now ends at the earlier of
  16:00 ET and the expiry; under the 16:00 ET convention the parser
  stamps, nothing moves, which is why the ten-year run was right. And
  `settlement_price` now has a stated domain: a cut that cannot see the
  settlement session's close is
  `UnpriceableLeg(:no_session_close)` rather than a provisional morning
  print blessed as a settlement (design rule 7). The tick loop cannot
  reach it; a direct caller of the exported lifecycle step can.
  **Review fix, finding 3:** the reference window was read with `isempty`
  and then `last` -- indexing, and two traversals -- while `between`
  promises only an iterable. A provider that streams its range, the shape
  `ParquetBarsReader` already returns for `OptionBar`, would have thrown
  a `MethodError` on `lastindex` there; the defect was latent only
  because both spot readers happen to return vectors. Settlement is the
  single consumer of `between` outside `market_data`, and it now consumes
  each window once, keeping the last record. **Review fixes, findings
  4-6, are prose:** the tick order and the policy doc promised that a
  policy never sees an expired leg, which D4's gap path contradicts --
  an unsettleable lot stays open and stays in the book handed to every
  later decision, and `lot.contract.expiry <= t` is what a policy reads
  to tell it apart. A recording policy on the gap fixture now pins that.
  The schema-3 reassurance above was the same kind of overclaim and is
  corrected in place. And D1's equality is this engine's choice, not the
  ledger's rule: `_check_expiry` permits settlement at or after the
  contract's expiry, which is what makes an expiry booked at a later
  tick legal.
  **Second review round, two more.** The expiry bound of the reference
  window could invert it: a contract expiring before 09:30 ET on its
  listed date produced the window `[09:30, expiry]`, which no print can
  satisfy, and the fall-through blamed the data for a gap complete data
  could not close. That contract is the AM-settled one -- SPX-style
  options settle against the *opening* print -- and `SettlementStyle`
  has carried `AMSettled` with no user since the ledger landed, every
  underlying in the contract table being `PMSettled`. So `:session_close`
  now names it, `UnpriceableLeg(:pre_open_expiry)`, and the rule that
  would serve it, `:session_open`, is recorded as future work beside
  `_SETTLEMENT_RULES` rather than approximated by walking back to a
  session the contract never settled against. The check is the listed
  date's alone -- a walked-back date closes before the expiry, so its
  window is always the whole session -- and only when the calendar calls
  that date open; a pre-open expiry on a closed date still walks back,
  across a DST change if need be. Nothing on real data moves: the
  parser stamps every expiry at 16:00 ET. And the predicate the prose
  handed a policy for spotting an unsettleable expired lot was
  `expiry < t` where the interval is `prev < expiry <= t`, so at a tick
  exactly at an expiry it called the warned-about lot live; it is `<=`
  here, in both module docs, and in the test that had encoded the same
  error.
  Next: the settlement rule and the venue into config and identity.

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
- **Second concrete policy** -- unblocked now that settlement is
  honest. Candidate: a daily iron condor (same scheduled-gate /
  `invert_delta` shape, four legs instead of two). Once the duplication
  is visible, decide whether to extract a `Structure` abstraction
  (`policies.md` Future work) or keep policies as 4-leg inline
  `decide` bodies. Parked 2026-09-08 behind the settle item, which
  landed with slice 3 of the ledger rebuild.
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
