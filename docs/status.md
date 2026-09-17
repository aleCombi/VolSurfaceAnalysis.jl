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
   (`docs/modules/data.md`): record types keyed by type with
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
2. **Modelling** (vol surface) -- done, relaid out 2026-09-17 under
   design rule 9. `Curve` types, Black-Scholes, the surface types and
   `build_surface` are the `pricing` module
   (`docs/modules/pricing.md`); the `RateCurve` / `DivCurve` records and
   the `VolatilitySurface` kind contract are in `data/kinds`; the
   `SurfaceFrom` provider, with its bounded cut-independent cache, is in
   `data/providers`. The `surfaces` folder is gone: curve and surface are
   now cut the same way, by stage rather than one by stage and one by
   object.
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
   `Fee` events) before anything is written, and returns the named pair
   `(ledger, failures)` -- the journal, and the `RunFailure`s the run
   retained for lots no honest settlement price could close, from both
   lifecycle passes; per-record `check_join` validates the fill-to-order
   join at each append. Lifecycle runs first at every tick and once more at the
   evaluation endpoint: `settlements` says which lots fell due and at
   what price, the ledger's own `record_expiry!` books each one, and the
   settlement rule reads sessions off the spot tree, consulting the NYSE
   calendar only to contradict it; which rule a lot gets is the contract
   fact `contract_spec(u).settlement`, and `:session_close` (PM) is the
   one style served.
5. **Metric computation** -- two honest inputs, because there are two
   questions (`docs/modules/metrics.md`). Trade questions read
   `trade_pnl(::Ledger)`, a plain vector of per-trade dollars, one entry
   per structure closed at one instant. Path questions read a
   `MarkedCurve`: marked portfolio profit at the close of every trading
   session in the window, built by `marked_curve` from the ledger's cash
   plus the open book marked at its own quote mid (surface price as
   fallback). Always-on core metrics (`total_pnl`, `n_round_trips`,
   `n_opens`, `n_closes`, `hit_rate`) are computed for every result;
   optional metrics (`sharpe`, `sortino`, `max_drawdown`, `volatility`,
   `profit_factor`) are selected by symbol through `compute_metrics`, and
   the `_METRIC_TABLE` in `src/metrics/dispatch.jl` maps each symbol to
   its function and its default kwargs. **Every metric takes both
   inputs** and reads whichever is its sample unit, so the table records
   no per-metric input; the consequence is that the dispatcher cannot
   tell a curve-reading metric from a trade-reading one, and with no
   curve it omits the optional set wholesale rather than part of it.
   Per-experiment overrides flow through `OutputSpec.metric_params`.
   Capital is fixed at 1 and is not an argument: at a zero risk-free rate
   it cancels from every ratio.
6. **Experiment orchestration** -- end-to-end runnable.
   `Experiment` wires `(Agent, MarketData specs, Clock, [from, to],
   OutputSpec)` into a single rerunnable record; `run_experiment(exp)`
   opens the data for the run and returns an `ExperimentResult` with
   the ledger, the marked curve and the computed metrics; marking runs
   there because it is not a pure function of the ledger. Open lots at
   the window end stay open, nothing is force-settled, and the curve is
   what values them at each session close. Outputs are
   declared in config: an `[outputs]` table (`metrics`, per-metric
   params, `artifacts`) resolves to an `OutputSpec`, defaulting to all
   registered metrics and the default artifact set when omitted. TOML
   configs (`[data.<kind>]` tables plus a `clock`) resolve via
   `load_experiment` (stdlib `TOML` + per-sum-type builder registries);
   `scripts/run_experiment.jl <config.toml> [--save] [--out-dir <dir>]`
   prints the result, and optionally persists it / renders artifacts.
   Parallel sweeps are future work.
7. **Persistence + identity** -- `RunStore` writes runs to a
   Hive-partitioned parquet tree at `<root>/runs/run_id=<full_hash>/`.
   **A run folder has two jobs.** It keeps the inputs needed to run the
   experiment again -- `config.toml` and `Manifest.toml`, both verbatim,
   plus the code provenance (`commit_sha` / `dirty` from
   `code_provenance`) on the manifest row -- and the outputs needed to
   verify that a rerun produced the same answer: `events` / `orders` /
   `order_legs`, `metrics`, `curve` and `failures` parquet, with an
   `artifacts/` subdir beside them.
   Identity is canonical and layered: `full_hash(experiment)` is the
   run id; `core_hash` (data + clock + agent + window) is shared by
   output variations of one backtest. Both come from a `to_dict`
   projection (`experiment/identity.jl`) over the *resolved* experiment,
   so whitespace / key order / `name` / cache knobs -- and metric
   parameters spelled at their defaults -- don't fork ids.
   **`load_run` reads the record and opens no market data**: the ledger
   is rebuilt through `commit!` and `check_join`, and the curve, the
   failures and the metrics come back as the run reported them. The
   reason is evidence -- recomputing on load destroys the witness a
   reproduction check needs, since two fresh computations can agree
   perfectly and both differ from the recorded run. `reproduce(store,
   run_id)` is the other operation: it reruns against live data and
   reports success, divergence by output/row/field, or inability, and it
   never writes over the witness, and it names both code provenances and
   both dependency environments, since a divergence attributes to code or
   dependencies by elimination. The manifest is the index over the record
   and carries **one count per output table**, all checked on load: a
   truncated table is not an empty one. The manifest `schema_version`, now
   7, refuses every earlier schema; none of them holds a curve, a failure
   table, or the counts that protect them, to migrate from.
   Every value that changes a result is either in `core_hash` -- the data
   specs, clock, agent, window, the venue's `fill_rule` / `cost_model`,
   the contract facts resolved for the experiment's underlying, and the
   parquet specs' bar-stamp convention, projected as the constant
   `"bar_end"` because it decides which minute every decision reads --
   or a stated constant in code. Dataset versioning is **dropped, not
   deferred**: the `dataset` slot holds a root path and the Massive trees
   are trusted as stable, so a divergence attributes to code or
   dependencies by elimination. Cross-run queries are DuckDB
   SQL against the parquet glob. Compute reuse (skip the backtest on a
   `core_hash` hit) and a curation gate are the next slices.

Visualization is added incrementally alongside each stage, not as a phase
of its own.

`src/`'s top level is a list of stages, not of financial objects (design
rule 9, adopted 2026-09-17). Every market object is therefore split the
same way -- math in `pricing`, record and kind contract in `data/kinds`,
provider in `data/providers` -- and the seam that keeps it honest is that
nothing in `pricing` reaches the protocol, a provider, a cut or an
experiment. The remaining asymmetry between curve and surface is a
modelling one, not a layout one: a curve is a payload inside a record, a
surface *is* its kind, and giving the surface the same split is deferred
(`docs/modules/pricing.md`, future work).

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
open lots at the window end stay open and the marked curve values them at
every session close, and the closure, the window-end spot lookup and the
fill-vector builder are gone with slice 2. What survives: the clock
selector says *when* to step, not whose price, and `load_experiment`
asserts a declared policy underlying matches it -- one experiment, one
underlying. `scripts/run_experiment.jl --out-dir <dir>` renders the
marked-curve artifact from any config (via `scripts/lib/artifacts.jl` +
`viz/pnl.jl`).

The review of the data-kinds branch (PR #9) found six correctness
defects that were one stance: an unanswerable question reported as an
ordinary empty result, so every consumer downstream correctly concluded
it had nothing to do. That stance is now design rule 7 and the
`data` protocol enforces it: `serves` answers the structural
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

Nothing. The ledger rebuild finished on 2026-09-15; the next work is the
backlog below, where **Second concrete policy** and **Official closing
prices instead of the session-close print** are the two items that most
directly advance [vision.md](vision.md).

## The ledger rebuild, finished 2026-09-15

The record of the rebuild, slice by slice, kept because the measured
numbers and the decisions behind them are what a later reader needs and
the working notes that carried them have been retired. Every design
decision named here now lives in the module docs.

- **Ledger rebuild.** The fill-vector ledger split the
  lifecycle over three layers (engine fills, `pnl_series` matches,
  `run_experiment` settles) with no shared record, plus a FIFO
  float-residue defect, per-share units labelled USD, and per-leg
  sampling that inflated the annualised ratios. What replaced it: an event
  journal booked inside the run -- `Order` with intent out of `decide`, a
  `Book` view in, `Fill` / `Match` / `Expiry` / `Fee` events with a
  bitemporal header, an order journal holding the quotes decisions saw,
  contract facts and the simulated venue inside identity, and ratios on a
  curve sampled at session closes rather than at trades. The plan was
  revised after four reviews and settled slice by slice; twelve
  result-changing decisions came out of it, and every one of them is now
  in [ledger.md](modules/ledger.md), [backtest.md](modules/backtest.md),
  [metrics.md](modules/metrics.md), [experiment.md](modules/experiment.md)
  or [persistence.md](modules/persistence.md). **Slice 1 landed
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
  `SpotPrice` input contract (stated in `data.md` after the PR #13
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
  surface price as fallback) belongs to the marked curve, which is PR 4
  below.
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
  **PR 3 landed 2026-09-14**: config and identity, the run-id break. Four values changed results and
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
  config made now. `data.md`'s `SpotPrice` paragraph now states
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
  **Bar-end stamps landed 2026-09-14**: the clock correction, and a
  deliberate break in comparability. A vendor minute
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
  **PR 4 (first half) landed 2026-09-14**: the marked curve and the
  metrics that read it. Every metric read a
  series of closed trades, so Sharpe, Sortino and volatility annualised
  by the square root of 252 while their observations were *trades*: a
  strategy closing about 252 structures a year looked plausible by
  accident and a weekly one was overstated by roughly a factor of two,
  and holding the same trades overnight or for a month gave the same
  ratio. `max_drawdown` had the same defect in another form -- a curve of
  closed trades is flat while a position is open, so a book could move
  deeply against itself and recover with no drawdown at all. The four
  path metrics now read a `MarkedCurve` sampled at session closes and the
  trade metrics keep per-trade dollars. Every metric takes both inputs and
  reads whichever is its sample unit, so `_METRIC_TABLE` stays `(fn,
  defaults)` and adding a metric is one row and one function; the price is
  that with no curve the whole optional set is omitted rather than part of
  it. The curve is the ledger's cash at
  the instant plus the open book marked to market, which is the
  `realised + unallocated fees + unrealised` identity with the cost basis
  cancelled; cash alone would book a short strangle's opening premium as
  profit. The grid is `session_closes` in `backtest/settlement.jl`, the
  `:session_close` rule enumerated rather than applied to one contract,
  so the grid a ratio is annualised over and the price a contract settles
  at cannot drift apart; a session counts only when its whole reference
  window is inside the evaluation bounds. **The three decisions the brief
  left open**, taken here: an unmarkable session leaves the curve, is
  named and counted in `unmarked_at` / `unmarked_reason`, and
  `session_changes` refuses to span it (so a break costs two observations
  and never invents a carried-forward value or a `NaN`);
  `compute_metrics(ledger, curve, requested)` takes the ledger as the
  authority for the trade side and derives `trade_pnl` once inside, with
  `n_opens(L)` / `n_closes(L)` as plain ledger functions; and the curve is
  `MarkedCurve`, two pairs of parallel vectors, with `cents_to_usd` as the
  named counterpart of `contract_cents` on the way out of whole cents.
  `PnLSeries`, `pnl_series`, `equity_curve` and `window_end_spot` are
  gone; the default artifact `:equity_curve` became `:marked_curve`, which
  is what moves `full_hash` while `core_hash` -- a projection marks cannot
  reach -- stays put, so an existing ledger remains reusable.
  `RUN_SCHEMA_VERSION` is 5 and `pnl_series.parquet` is no longer written.
  **Gate: 3719 passed, 0 failed, 0 errored, 0 broken.** Measured against
  the bar-end baseline, not the pre-correction one: the ten-year strangle
  rerun is unchanged where it must be -- 13,204 events, 2,201 orders,
  USD 29,942.23 cash, 2,200 trades, `total_pnl` 29,694.53, `hit_rate`
  0.7805, `profit_factor` 1.3028 -- and its `core_hash` is the same
  `2bde5de695f9c90a` it was before the round while `full_hash` moved from
  `f555f4bdbaf8d1d8` to `6990a511c201c1aa`. Its curve marks **2,515 of
  2,516 sessions**. The one it cannot mark is 2018-10-25T20:00:00,
  `:no_mark`: no two-sided quote for an open lot at that close and no
  surface stamped there either, on the single day whose entry-instant
  chain visibility the bar-end correction also moved. So the named-failure
  path does fire on real data, once in ten years, and the break costs the
  two observations either side of it -- 2,513 session changes from 2,515
  marked points. The ratios moved as expected, trade-sampled to
  session-sampled: sharpe 1.2686 to 1.0389, sortino 1.4208 to 1.1898,
  volatility 2681.17 to 2852.16, max_drawdown 6399.89 to 6431.04. The
  last is the telling one -- the old figure was the deepest trough of
  *closed* trades, and the new one sees the book while it is open.
  One thing the round found in the data: the SPY spot tree holds two
  disagreeing rows at 2026-02-07T00:12:00 (690.21 vs 690.22), an
  extended-hours instant. `session_closes` reads one session window at a
  time, exactly the windows `:session_close` reads, so it never sees it --
  a range read across the gaps between sessions does, and aborts with
  `ConflictingRecords`. The regular-session input contract is claimed
  inside those windows and nowhere else, and the grid now keeps to them.
  **The finishing round landed 2026-09-15**, three commits in one branch:
  the metric-parameter identity fix, the persistence split, then a docs
  sweep that retired every proposal. It
  superseded what PR 4's second half had planned: `round_trips`, `marks`
  and `equity` export tables and the manifest completeness flag are
  dropped, with reasons, and the load path stops recomputing instead of
  keeping to exports only.
  **Commit 1 landed 2026-09-15**: metric-parameter identity.
  `to_dict(::OutputSpec)` serialised `metric_params` as spelled, so a
  config naming a parameter at its `_METRIC_TABLE` default forked
  `full_hash` from one that omitted it, though both run the identical
  metric -- the one place the "omitted-vs-explicit defaults do not move an
  id" invariant was untrue. The projection now emits, per *requested*
  metric, the parameters that metric will actually run under: the table
  defaults with the experiment's override merged over them, which is how
  `[venue]` and `lookback_ticks` already project. An override for a metric
  the experiment does not compute reaches no result and is not projected
  at all; an unknown requested metric still hashes, because naming it is
  `compute_metrics`' failure at run time and not identity's. Every
  `full_hash` moves and no `core_hash` does. Nothing live is stored, so
  there is nothing to migrate and `RUN_SCHEMA_VERSION` stays 5. **Gate:
  3735 passed, 0 failed, 0 errored, 0 broken.** The ten-year strangle is
  unchanged where it must be -- 13,204 events, 2,201 orders, USD 29,942.23
  cash, `sharpe` 1.0389, 2,515 of 2,516 sessions marked -- and keeps
  `core_hash` `2bde5de695f9c90a` while its `full_hash` moves from
  `6990a511c201c1aa` to `f402707b152aab0c`.
  **Commit 2 landed 2026-09-15**: the persistence split. A run folder now
  has two jobs -- keep the inputs needed to run the experiment again
  (`config.toml` and `Manifest.toml`, both verbatim, plus the code
  provenance) and the outputs needed to verify the answer is the same
  (`curve.parquet` and `failures.parquet` join the ledger tables and the
  metrics). `load_run` **reads that record and opens no market data**,
  which reverses the recompute-on-load decision of 2026-09-14: the ground
  then was that a loaded result must agree with today's inputs, and the
  ground now is that a reproduction check needs a witness it can disagree
  with -- two fresh computations can agree perfectly and both differ from
  the recorded run, and recomputing on load destroyed the only thing that
  would have said so. `reproduce(store, run_id)` is the other operation:
  it rebuilds the experiment from the stored config, reruns it against
  live data, and reports success, divergence by output / row / field with
  both values, or *inability* -- missing data is inability, never a
  successful comparison of empty outputs, and a config that no longer
  hashes to its own folder is a named identity mismatch carrying the
  regenerated projection. It never writes over the witness. Integers,
  instants, identifiers and reasons compare exactly; finite floats use an
  absolute `1e-9`, with NaN matching NaN and same-signed infinities
  settled before any subtraction.
  The engine keeps what it could not answer: `settlements(...).unsettled`
  now leaves `run_backtest` as `RunFailure`s from **both** lifecycle
  passes, the window-end one included, and marking keeps every failed
  subject instead of stopping at the first -- a broken session is still
  one curve entry and as many failure records as it had failed lots.
  Neither is a ledger event, because nothing happened, so no replay could
  recover them. `run_backtest` and `marked_curve` return named pairs
  (`(ledger, failures)`, `(curve, failures)`), the shape `settlements` and
  `fill_legs` already use. Two backlog items close here: the named-column
  parquet writes land (positional `VALUES` lists and the events table's
  per-kind NULL padding are gone; the bytes written are identical, so no
  schema change of its own), and **Dataset fingerprint in identity is
  deleted as decided rather than parked again** -- the Massive trees are
  trusted as stable, the `dataset` slot's root path is the accepted
  contract, and a divergence therefore attributes to code or dependencies
  by elimination. `RUN_SCHEMA_VERSION` became 6 at this commit and refuses
  every earlier schema, including 5: none of them holds a curve or a
  failure table to migrate from. Deliberately not added, with reasons in
  `persistence.md`: `round_trips.parquet` (no cross-run consumer yet) and
  the manifest completeness flag (it would not assert what it appears to,
  since the curve samples whole session closes and the window endpoint
  need not be one). **Gate: 3911 passed, 0 failed, 0 errored, 0 broken.**
  The ten-year strangle is unchanged in every figure -- 13,204 events,
  2,201 orders, USD 29,942.23 cash, `total_pnl` 29,694.53, `sharpe`
  1.0389, `sortino` 1.1898, `max_drawdown` 6,431.04, `volatility`
  2,852.16, `profit_factor` 1.3028, 2,515 of 2,516 sessions marked -- and
  neither hash moves. What is new is the account: that one unmarked
  session, 2018-10-25T20:00:00, retains **two** `:no_mark` failures, one
  per open lot of the strangle. The old builder broke at the first lot and
  reported one reason; both legs were unpriceable all along.
  **The round's own witness.** The ten-year strangle was saved under
  schema 6 as `run_id=f402707b152aab0c` at `commit_sha`
  `c856696` with a clean tree, and `reproduce` on it reported
  `:reproduced` with **zero divergences** -- 13,204 events, 2,201 order
  records, 4,402 order legs with their observations, 2,516 curve points, 2
  failures and 10 metrics, every one compared field by field against a
  live rerun. That is the first time a stored run in this repository has
  been checked against its own rerun rather than merely reloaded. (The
  review commit below moved the schema to 7, so that saved folder was
  refused by version and re-saved from the same config; the run id, every
  figure and the zero-divergence result are unchanged.)
  **Commit 3 landed 2026-09-15**: the docs sweep, no code. Every file in
  `docs/proposals/` is deleted, this round's brief included, after moving
  what survived into the module docs: the venue's unmodelled parts and why
  `:cross_spread` is conservative, the assignment / exercise deferral and
  its trigger, the Sharpe-flatters-short-premium caveat and the
  capital-base survey, the Sharpe (1994) citation, and the rule that a
  constant enters the hash only when it separates two populations of
  stored runs. Two things were deliberately *not* absorbed. Four
  cross-module house rules the retired orchestration note carried are
  parked in the backlog as a proposal rather than written into
  `design.md`, because design rule 3 says a rule change is surfaced and
  not absorbed. And two source comments still point at the retired notes
  by name -- `src/ledger/cash.jl` ("the rules (proposal, section 2)",
  which is now [ledger.md](modules/ledger.md)'s cash rules) and
  `test/regressions/test_review_findings.jl` ("under proposal decision 8",
  which is now "an open lot at the window end is valued by the marked
  curve, never force-settled"). The sweep commit touches no code, so they
  are the one loose end it leaves.
  **The review commit landed 2026-09-15**, closing a review of the finished
  branch: four merge blockers, two status defects and one test suggestion.
  Three blockers were places the load path accepted a record it could not
  vouch for. *Leg identity*: `order_leg_id` and `leg_idx` were read as sort
  keys and never checked, and a leg whose `order_id` named no order was
  dropped in silence, so a rewritten id, indices shifted without changing
  their order, or an orphan row all still loaded as an untouched ledger.
  Both columns are now checked against the order that claims them, and
  every input row is accounted for. *Membership*: nothing protected the
  metric or failure tables, so an empty `metrics.parquet` read as a run
  that reported no metrics, and deleting the settlement failures -- or one
  of the two 2018-10-25 mark failures while keeping the other -- passed
  every structural check there was. `n_metrics` and `n_failures` join the
  manifest, giving every output table membership evidence, and
  `RUN_SCHEMA_VERSION` moves to **7**: a schema-6 manifest never wrote the
  two counts down, so there is nothing to migrate and its runs are refused
  by version like every earlier one. *Agreement*: curve and failures were
  compared on instants only, so a curve reporting `:no_mark` loaded against
  failures that all said `:unexpected_gap`; each curve reason must now
  occur among the mark failures of its own session. The fourth blocker was
  the reproduction report, which carried only Git provenance although
  attribution is by elimination: it now names both dependency documents by
  digest and every package whose version moved between them -- the
  settlement rule reads the calendar `BusinessDays` ships, so two identical
  commits over two environments are not the same run. Differing
  dependencies are provenance, not divergence, and never move the status.
  Each blocker has a tamper test that names the refusal; the producer side
  of the failure tables is asserted directly now too, at both lifecycle
  passes (the tick pass and the window-end pass, told apart by the instant
  the question was stamped with) and for two unpriceable lots in one
  session. The two status defects: `run_backtest` returns `(ledger,
  failures)` and not a `Ledger`, and `_METRIC_TABLE` records no per-metric
  input because every metric takes both -- both descriptions corrected
  above. The four parked house rules were **binding** in the retired
  orchestration note, not "proposed but not adopted", and the backlog entry
  had also dropped *named failures, never bare errors*; it now states their
  prior standing as it was and lists five rules.
  **Gate: 3978 passed, 0 failed, 0 errored, 0 broken** (2m04s), up from
  3,911. The witness was re-exercised end to end: the schema-6 folder was
  refused by name, re-saved from the same config at schema 7 -- same
  `run_id=f402707b152aab0c`, 13,204 events, 2,201 orders, USD 29,942.23
  cash, `sharpe` 1.0389, 2,515 of 2,516 sessions marked, 2 retained
  failures -- and `reproduce` on it reported `:reproduced` with **zero
  divergences** in 91 s, both environments identical. On copies in `/tmp`,
  deleting one of the two mark failures, contradicting the curve's reason,
  rewriting an `order_leg_id`, adding an orphan leg row and emptying
  `metrics.parquet` were each refused by name, and swapping the recorded
  `Manifest.toml` reproduced with zero divergences while reporting
  `BusinessDays 0.0.1 -> 0.9.25`.
  **The second-pass review commit landed 2026-09-15**, closing the four
  findings a re-review of that commit raised. The blocker: *failure
  membership was interchangeable across stages*. A single `n_failures`
  policed the table's size and nothing inside it, and the curve/failure
  agreement ignores every non-mark row, so changing one of two mark
  failures at an unmarked instant into a settlement failure -- or deleting
  a settlement failure and duplicating a mark failure -- kept the count,
  kept the agreement, and moved what the run is recorded to have asked.
  The manifest now keeps **one count per stage** (`n_mark_failures`,
  `n_settlement_failures`), `stage` is a closed vocabulary refused on both
  the write and the load path, and the agreement check is explicitly the
  mark rows' guard while the settlement count guards the rest.
  `RUN_SCHEMA_VERSION` moves to **8**: a schema-7 manifest wrote only the
  total, so it cannot say what its stages held and would be silently
  exempt from the new check -- it is refused by version like every earlier
  one, with no migration. Three non-blocking findings went with it. The
  reproduction report said "neither names a version" whenever two
  `Manifest.toml` digests differed with the version maps agreeing, which
  is false when both documents name many versions and only a comment, the
  project hash or a dependency path moved; it now says the documents
  differ with no recorded version changes. The version map was keyed by
  package **name**, so two distinct packages sharing a name (which Pkg
  supports) collapsed and a version move in the overwritten one vanished
  from the report; it is keyed by UUID now, with the name kept for display
  and the uuid printed when the name does not identify the package alone.
  And `persistence.md` contradicted the code in three places -- one count
  per output table (there is no order-leg count and the curve has two), no
  earlier schema holding a curve or failure table (6 and 7 both do, they
  just lack the evidence a later reader checks), and persistence not
  interpreting the dependency document (it reads it for provenance) --
  all three corrected.
  **Gate: 4005 passed, 0 failed, 0 errored, 0 broken** (2m12s), up from
  3,978. The witness was re-exercised again: the schema-7 folder was
  refused by name, re-saved from the same config at schema 8 -- same
  `run_id=f402707b152aab0c`, 13,204 events, 2,201 orders, USD 29,942.23
  cash, `sharpe` 1.0389, 2,515 of 2,516 sessions marked, 2 retained mark
  failures -- and `reproduce` reported `:reproduced` with **zero
  divergences** in 96 s, both environments identical. On copies in `/tmp`,
  retagging one of the two mark failures as a settlement failure was
  refused (`n_mark_failures says 2 ... hold 1`), and against a consistent
  baseline carrying one settlement failure, deleting it while duplicating
  a mark failure -- the row count unchanged at three -- was refused too,
  as was deleting it on its own (`n_settlement_failures says 1 ... hold
  0`) and retagging a row to a stage no pass emits.

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
  not be mistaken for one. Candidate source: Polygon's daily aggregates.
  The vendor asymmetry is the reason to look there first -- minute
  aggregates deliberately relax the SIP sale-condition rules so
  extended-hours trades update them, while daily bars follow the
  end-of-day guidelines and do not, which is the opposite of the property
  that makes the current rule fragile. The collector needs a
  `us_stocks_sip/day_aggs_v1` pipeline before a kind can read one; then
  the kind and its provider spec follow, already inside identity as a
  `[data.*]` entry, and `:session_close` reads it instead of walking the
  minute tree. That changes results and ids change with them. Parked
  2026-09-13 from the PR #13 review finding; the vendor detail recorded
  2026-09-15 from the retired identity note.
- **Leaning out the architectural docs.** Pass over `docs/modules/*`
  (and the top-level docs) to bring them in line with design rule 6 --
  invariants and boundaries kept, drift-prone implementation detail
  (magic numbers, internal data structures, incidental library names,
  API walkthroughs) dropped. Motivation: resuming the library after a
  few-week pause, the docs should be the trustworthy entry point to read
  back in from. `data.md` is the first pass / template; the other module
  docs follow. `data.md` (new) follows the template from the
  start. Parked after PR #9 (2026-09-08); no slice in progress.
- **Second concrete policy** -- unblocked now that settlement is
  honest. Candidate: a daily iron condor (same scheduled-gate /
  `invert_delta` shape, four legs instead of two). Once the duplication
  is visible, decide whether to extract a `Structure` abstraction
  (`policies.md` Future work) or keep policies as 4-leg inline
  `decide` bodies. Parked 2026-09-08 behind the settle item, which
  landed with slice 3 of the ledger rebuild.
- **Reproducibility harness for stored runs.** The comparison itself
  landed as `reproduce(store, run_id)`; what remains is the harness around
  it. Opt-in, data-gated integration tests that reproduce every stored
  schema-8 run and skip cleanly where the source data is absent (a
  data-less machine must *skip*, while an invoked reproduction on one
  reports inability rather than success); `scripts/revalidate_runs.jl`, a
  utility that refreshes a run's `commit_sha` / `dirty` after a successful
  reproduction -- never as a side effect of divergence, and never
  replacing the stored outputs or the dependency document; and extending
  `compare_runs.jl` to the curve and the failures rather than the manifest
  and the metrics alone.
- **Conflicting extended-hours spot rows.** The SPY tree holds two
  disagreeing rows at 2026-02-07T00:12:00 (690.21 vs 690.22). Nothing in
  the codebase reads outside the regular-session window, so nothing sees
  it today; whether it is one bad delivery or a class of them, and whether
  the collection step should reject it at write time, is uninvestigated.
  Found 2026-09-14 while building the session grid.
- **Five cross-module house rules, binding in the retired note, not yet
  in `design.md`.** The orchestration note recorded them under its
  *Decisions taken (binding)* headings, and they governed the whole ledger
  rebuild; what is unsettled is only whether they become standing design
  rules for every future round, since design rule 3 says a rule change is
  proposed and not absorbed. They are parked here, with their prior
  standing stated as it was: (1) tests live beside the source they test,
  one file per source file in the mirrored folder; (2) **named failures,
  never bare errors** -- every refusal has its own name, and every failure
  test checks that it fires, that it leaves the refused state exactly as
  it was, and that it prints its own name; (3) describe the boundary,
  never claim impossibility -- Julia has no private fields, so "cannot
  happen" and "by construction" claim an absolute the code cannot deliver,
  and three reviews in a row caught this codebase doing it; (4) before
  proposing a struct, ask whether a symbol, a function or an existing type
  does the job -- a type hierarchy is what a table graduates to when each
  entry needs its own behaviour, not where it starts; (5) working notes
  are retired by the change that lands their work, not carried forward for
  a later sweep. The module docs already *illustrate* all five, and the
  code follows them. Promoting them into [design.md](design.md) is a
  one-line decision each; surfaced 2026-09-15, restated 2026-09-15 after
  a review found the parking note had downgraded them to "proposed but
  not adopted" and dropped the named-failures requirement outright.
- **Capability-restricted views.** A structural raw/model boundary (a
  policy view that cannot address `OptionBar`) was declined in
  data-kinds v3 in favour of a doc rule; revisit if a policy ever
  couples to vendor bars.

- **Path metrics on a finer grid than one session.** The marked curve
  samples session closes, which is what annualising by sessions means;
  an intraday grid would see moves that open and close inside a session,
  and would need its own annualisation constant and its own answer to
  what an unmarkable point costs. Closed for the trade-ordering half:
  path metrics no longer read simultaneous trade samples at all.
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
- **Reader and SQL duplication in `data/providers/parquet.jl`.**
  `ParquetBarsReader` and `ParquetSpotsReader` repeat open, close,
  partition listing, the backward walk and the grid, differing only in
  how a timestamp is read from a partition; unifying them would have
  closed the spot `asof` gap by construction. The SQL timestamp formatter
  duplicates one in the store module and the path quoter one in the
  polygon module; extraction needs a home across module boundaries.
- **Implied-forward calibration, per snapshot.** A slice holds one IV per
  strike inverted against `S * exp((r-q)*T)`, so the residual put-call IV
  gap at short tenors is carried rather than calibrated away. The parked
  work is a per-snapshot forward from put-call parity (the legacy
  `recalibrate_iv` is the reference), a slice-level forward field, and
  pricing off that forward instead of the spot-implied one. Parked
  because the one-IV-per-strike slice is honest without it and nothing
  in the repo yet measures the gap it would close -- the first consumer
  should be a measurement, not a policy.
- **A payload/record split for the vol surface.** Deferred deliberately
  when `pricing` landed (2026-09-17). A curve is a payload inside a
  stamped, selected record; a surface *is* its kind, so its math object
  doubles as its record. Splitting it the same way is what would let one
  instant carry two surface conventions, which today's *one provider per
  kind* rule makes two runs instead. Parked, not rejected: nothing needs
  two conventions at one instant yet, and the asymmetry costs nothing
  while that holds.
