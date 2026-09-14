# Status

This is `master` -- the active clean-line codebase. The prior full codebase
is not a branch: it is this repository's own history. `6d016c8` ("Wipe to
clean slate for rebuild", 2026-05-03) emptied the tree, so its parent
`bc31405` is the last commit carrying the old 37-file `src/`, and that is
the reference we mine from -- `git show bc31405:src/<file>`. Work here
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
   mapping, in which a record read off a minute bar is visible at **bar
   end** (`bar_visible_at`): the vendor row's open stamp plus the bar
   interval, so a decision at `t` reads the completed `[t - 1min, t)`
   minute rather than one still running. That is the convention, fixed
   in code and not a setting; the readers apply it where rows become
   records and every shape above them speaks visibility time.
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
   `run_backtest(agent, data, from, to, clock; fill_rule, cost_model)`
   drives the tick loop on the
   experiment's declared `Clock`, prices every leg of every order
   through the simulated venue (`docs/modules/backtest.md`:
   `:cross_spread` on the tick, IBKR Pro's US options commissions as
   `Fee` events) before anything is written, and returns a `Ledger`;
   per-record `check_join` validates the fill-to-order join at each
   append. Lifecycle runs first at every tick and once more at the
   evaluation endpoint: `settlements` says which lots fell due and at
   what price, the ledger's own `record_expiry!` books each one, and the
   settlement rule reads sessions off the spot tree, consulting the NYSE
   calendar only to contradict it; which rule a lot gets is the contract
   fact `contract_spec(u).settlement`, and `:session_close` (PM) is the
   one style served.
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
   off-machine; the manifest `schema_version`, now 4, refuses runs
   written under the positions schema and under the pre-identity one).
   Every value that changes a result is either in `core_hash` -- the data
   specs, clock, agent, window, the venue's `fill_rule` / `cost_model`,
   the contract facts resolved for the experiment's underlying, and the
   parquet specs' bar-stamp convention, projected as the constant
   `"bar_end"` because it decides which minute every decision reads --
   or a stated constant in code. Cross-run queries are DuckDB
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
  (`ledger-slice1-review.md`, retired): not
  mergeable. The write path accepts batches the invariants forbid (a
  caller-supplied spec at `commit!` that the replays ignore, consumption
  effective before its open, a partial or early expiry, a non-FIFO
  match) and there is no structure-level atomic writer; the driver's
  review added that book equality is exact float while the two replays
  add in different orders. Every finding is pinned as a failing testset
  in `test/ledger/test_review_findings.jl`, so the gate went red on
  purpose: 1630 passed, 15 failed, 1 errored, all in that file. **Fix
  round landed 2026-09-12**
  (`ledger-slice1-fix.md`, retired): `commit!`
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
  (`ledger-slice1-hardening.md`, retired;
  the inventory is
  `ledger-slice1-coverage.md`, retired):
  every invariant and named failure the module documents is mapped to
  its enforcing code and its test; `test_review_findings.jl` is
  dissolved into the suites beside the behaviour they check; documented
  promises the code did not enforce now are (an expiry's outcome agrees
  with its intrinsic value; recorded time is at or after effective time
  and nondecreasing along sequence, `RecordedOutOfOrder`; prices are
  finite, `InvalidPrice`; an unminted id at `event` and a non-positive
  join id are `DanglingReference`). The rule additions are listed in the
  coverage document for veto. Codex's review of the round
  (`ledger-slice1-hardening-review.md`, retired)
  found the fill review's construction-time post-expiry check missing:
  `FillAfterExpiry` is now thrown by the `Fill` constructor too, and
  every rejection in every failure testset checks the ledger snapshot
  and the book. Gate after hardening: 2306 passed, 0 failed, 1 broken,
  the structure-atomicity testset waiting for slice 2. **Slice 2 landed
  2026-09-12** (`ledger-slice2.md`, retired): the
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
  (`ledger-lifecycle.md`, retired): expiries are
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
  the early closes with no early-close table *given* the regular-session
  `SpotPrice` input contract (stated in `market_data.md` after the PR #13
  review; an extended-hours print inside the window would settle an early
  close instead, undetectably);
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
  **Gate after slice 3 and the review-fix rounds: 3364 passed, 0
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
  should. Two rounds of review closed eight findings; the shape they left is
  the shape above, plus four boundaries worth keeping here. A third round, on
  the PR #13 review, closed one finding with documentation and a test
  rather than logic: the early close is right because the spot input is
  regular-session only, so that requirement is now stated as the
  `SpotPrice` contract and pinned by a test that feeds the rule a
  violating extended-hours print. **The venue
  is stricter than the ledger about expiry**: `fill_legs` refuses a leg
  whose contract expires at or before the tick, because trading has
  stopped, and that refusal is what the interval's completeness rests
  on -- without it a lot opened at its own expiry instant escapes every
  later interval, the window-end pass included. `src/ledger/` is
  untouched and still accepts a fill effective at that instant. **The
  reference window ends at the earlier of the session close and the
  contract's expiry**, so an intraday expiry cannot settle at a print
  from after it expired; under the 16:00 ET convention the parser
  stamps, nothing moves. **The rule has a stated domain**: a cut that
  cannot see the settlement session's close is `:no_session_close`, and
  an expiry before its own session opens is `:pre_open_expiry` -- the
  AM-settled shape, which `:session_open` would serve and which is
  recorded as future work rather than approximated. **A policy is not
  promised that expired legs are gone**: an unsettleable lot stays open
  and stays in the book handed to every later decision, and
  `lot.contract.expiry <= t` is what tells a policy it holds one,
  inclusive because the settlement interval is.
  **PR 3 landed 2026-09-14**
  ([docs/proposals/ledger-identity.md](proposals/ledger-identity.md)):
  config and identity, the run-id break. Four values changed results and
  appeared in no run id. Two of them are choices and are now config:
  `fill_rule` and `cost_model` are `Experiment` fields, read from an
  optional `[venue]` table whose two keys both default to today's values,
  and they enter `core_hash` -- together with the `ContractSpec` resolved
  for the experiment's one underlying, which is what puts the contract
  table's facts in the hash without forking every id on an entry the run
  never touches. The other two are not choices at all. The tick is
  `const TICK_CENTS = 1` (every underlying the table lists trades in
  penny increments at every premium), still a defaulted parameter of
  `fill_price` / `fill_legs` / `check_join` so the join check can
  recompute a fill from an observation and say which tick it checked
  against. Settlement style is a contract fact: `settlements` routes per
  lot off `contract_spec(u).settlement`, and `run_backtest` has no
  `settlement_rule` or `tick_cents` keyword left. The settlement *price
  source* stays hardcoded, deliberately: `:session_close`'s replacement
  is an official-close feed, which arrives as a market-data kind with a
  provider spec -- already inside identity -- not as a venue symbol.
  `AMSettled` is named at both ends: `load_experiment` refuses such a
  config when it reads it, and `settlements` throws
  `UnsupportedSettlement`, which is deliberately *not* caught and warned
  like `UnpriceableLeg` -- that names one lot unpriceable at this
  instant, this names a contract class nothing here can settle, so the
  run stops. `:session_open` stays unwritten: there is no AM-settled
  underlying to test it against, and an untested settlement rule is worse
  than an absent one. `RUN_SCHEMA_VERSION` is 4, so every stored run id
  moved; the version is also what finally separates the schema-3 tree
  (runs written before lifecycle booked expiries) from runs of the same
  config made now. `market_data.md`'s `SpotPrice` paragraph now states
  what the tree actually provides rather than what the rule needs:
  Polygon's minute aggregates deliberately update on extended-hours
  trades (SPY on 2024-12-24 holds bars from 04:00 to 16:59 ET), so the
  six early closes settle at their 13:00 ET prints on a *measured*
  property of the data -- zero bars in (13:00, 16:00] ET on every one of
  them -- rather than on a guarantee, and the official-close kind is what
  would make that structural. Results are unchanged by construction: only
  ids move. **Gate: 3425 passed, 0 failed, 0 errored, 0 broken.** The
  ten-year strangle rerun (same 4478 `Expiry` events over 1699 instants,
  same 2240 orders, same metrics, new run id) is the regression that
  closes the round.
  **Bar-end stamps landed 2026-09-14**
  ([docs/proposals/bar-stamp.md](proposals/bar-stamp.md)): the clock
  correction, and a deliberate break in comparability. A vendor minute
  bar is stamped at its open, but its close, high and low -- and the
  bid/ask `SpreadFromOHLCV` builds from them -- are knowable only when
  the minute ends, so a decision at `t` was reading the `[t, t+1min)`
  bar. That is up to one minute of lookahead on every bar-based fill and
  every settlement price, and `TimeCut` could not catch it: the machinery
  is sound, but the record admitted through it claimed to be knowable
  before it was, so the guarantee failed below the cut. A record read off
  a minute bar is now visible at bar end (`bar_visible_at`, one
  `BAR_INTERVAL` = one minute for both production trees). The shift lives
  in one place, the parquet readers' boundary between rows and records:
  every timestamp leaving DuckDB is moved forward, every SQL bound is
  moved back, so records, the cached per-partition timestamp lists, the
  spot blocks and all four shapes speak visibility time and no call site
  above knows the vendor clock exists. Synthesis preserves the instant
  and adds nothing. It is **the** convention, fixed in code: no `stamp`
  option, no compatibility mode, because one of the two settings would
  enable lookahead -- and the backlog entry that parked it as a spec
  option is deleted as decided rather than left parked. The partition
  convention survives: a `D 23:59` row is now visible on `D + 1` without
  leaving partition `D`, and the existing `Date(ts) - 1` / `Date(ts)`
  candidate pair still finds it, with no next-day partition required; the
  one-day spill bound is restated so that raw `[D 00:00, D+1 02:00)`
  and visible `[D 00:01, D+1 02:01)` are told apart. Settlement needed no
  rule change, only an honest input: the vendor row stamped 16:00 ET is
  the 16:00-16:01 minute and now becomes visible at 16:01, outside the
  09:30-16:00 window, so the 15:59-16:00 bar -- visible at exactly 16:00
  -- wins it. On 2024-01-16 that moves SPY's settlement print from 475.02
  to 474.95. The six early closes do not move: their winning record is
  the 13:00-13:01 bar, on the boundary before and one minute inside now,
  re-measured rather than assumed. The correction also unblocks the curve
  round's blocking case with no marking workaround: an exact quote lookup
  at the session close now finds the last completed option bar (501 SPY
  quotes at 21:00 UTC on 2024-01-16, where the old clock found none).
  **The identity break is the point, not a side effect.** The parquet
  specs project the bar-stamp convention as the constant `"bar_end"`, so
  `core_hash` and every run id move; it is not a user option and not in
  `OutputSpec`. The ten-year strangle's id goes from
  `838ba0b70857c331` / `b47d70c2da9b4dd5` to `2bde5de695f9c90a` /
  `f555f4bdbaf8d1d8`. `RUN_SCHEMA_VERSION` stays 4: the file layout does
  not change and the id break is itself what separates the two
  populations. **Stored runs made under bar-open visibility do not
  reproduce under this code, and their ledgers cannot be reused as
  results of the corrected backtest.** That cost is accepted. **Gate:
  3549 passed, 0 failed, 0 errored, 0 broken.** The new ten-year
  baseline, which supersedes the pre-correction numbers everywhere they
  appear (the curve round's brief included): 13204 events (4402 fills,
  4400 expiries, 4402 fees) over 2201 orders and 2200 round trips,
  cash USD 29942.23, two lots still open at the window end; `total_pnl`
  29694.53, `hit_rate` 0.7805, `sharpe` 1.2686, `sortino` 1.4208,
  `max_drawdown` 6399.89, `volatility` 2681.17, `profit_factor` 1.3028.
  Against the old 13438 / 2240 / USD 32008.66 (`total_pnl` 31785.96,
  `sharpe` 1.3481, `profit_factor` 1.3288) the minute of lookahead was
  worth about USD 2066 of cash over ten years, roughly 6.5%, and 0.06 of
  Sharpe. Thirty-nine fewer orders is not a grid change -- the entry
  instant has a chain on 2492 of the 3652 days either way, differing on
  one -- but the contents of the minute the policy now reads.
  Next: PR 4, outputs -- `pnl_series`, the metrics, the equity curve and
  the structure series, and then the marked-curve round against this
  baseline.

## Backlog

Backlog items are concrete parked work: visible enough to preserve the
intended direction, but not currently in flight.

- **Official closing prices instead of the session-close print.** The
  `:session_close` rule reads the underlying's last regular-session print
  of the settlement session, and that stand-in is the model's one stated
  departure in settlement: the official closing auction is not in minute
  bars, so the 16:00 (or 13:00) print takes its place. A source of
  *official* closes would remove the departure outright, and with it the
  early-close exposure the production spot tree currently avoids by
  measurement rather than by contract -- an official close is stamped by its session
  rather than inferred from a window, so an extended-hours print could
  not be mistaken for one. Candidate source: Polygon's daily aggregates,
  which `massive/polygon` may already deliver; whether they carry the
  official close, what kind or provider shape they take, and what that
  costs in identity, is the design note. Not investigated. Parked
  2026-09-13 from the PR #13 review finding.
- **Named columns in the parquet writes.** Every write site in
  `src/persistence/store.jl` is `INSERT INTO _writebuf VALUES (...)`,
  positional, and `events.parquet` is one wide union table, so each event
  type pads the columns it lacks with positional `"NULL"` strings. A
  miscount writes a value into the wrong column and nothing in the type
  system catches it; what defends it today is `load_run` rebuilding
  through one `commit!` and re-running `check_join`, which turns a
  misalignment into a load-time failure rather than a plausible wrong
  number. Naming the columns removes the padding entirely -- DuckDB nulls
  what an insert does not name -- and the class of bug with it. The file
  written is byte-identical, so there is no schema version change and no
  migration. *Considered and rejected:* a table per event type plus a
  spine by id. Cleaner modelling, but `save_run` has no transaction
  across files, so a crash between writes would leave fills without their
  matches and `load_run` would rebuild a wrong ledger rather than fail to
  find one; and every read of the journal becomes a four-way `UNION ALL`
  re-sorted by sequence, in `load_run` and in cross-run SQL alike. The
  sparse columns themselves cost almost nothing -- parquet stores NULLs
  cheaply. Parked 2026-09-13 from the PR #13 code read; the natural home
  is whichever slice is already inside `store.jl`.
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
