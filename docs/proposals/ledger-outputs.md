# PR 4: outputs, and the end of the rebuild

The last pull request of the ledger rebuild
([ledger-orchestration.md](ledger-orchestration.md)). It is the
proposal's slices 5, 6 and 7 together: the equity curve and the ratios
that move onto it, the derived persistence tables, and the docs sweep
that deletes the proposals. After it there is no ledger rebuild, only
the library.

## Read first

1. `docs/design.md`, all seven rules.
2. `docs/proposals/ledger.md`, section 1 (the defects this rebuild set
   out to fix) and slices 5 and 6 of section 4. Three of the section 1
   defects are still live; they are the subject of this round.
3. `docs/modules/metrics.md` and `src/metrics/` in full -- it is five
   short files.
4. `docs/modules/persistence.md`, the derived-tables entry.
5. `docs/status.md`, entries 5 and 7.

## The problem

Three defects from the proposal's original list survive into today's
code, all of them in the metrics layer, all of them because the sample
unit is a closed round trip rather than a point in time.

**Ratios annualise against the wrong thing.** `sharpe`, `sortino` and
`volatility` treat each round trip as one observation and scale by
`sqrt(periods_per_year)` with a default of 252 (`src/metrics/optional.jl`,
and the sampling-convention comment at its head says so plainly). That
is only right for a strategy that closes 252 structures a year. The
ten-year strangle closes 2239 over ten years, so the number reported
today is roughly right by coincidence of a daily strategy; a weekly one
would be silently wrong by a factor of two, and nothing would say so.

**Equity is realised-only.** `max_drawdown` runs over
`cumsum(series.pnl)` -- one step per closed structure. Between opening a
strangle and its expiry the curve is flat however far the position moves
against the book, so drawdown cannot see open exposure at all.

**The placeholders are still placeholders.** `PnLSeries.window_end_spot`
is `NaN` and `n_unmarked` is `0`, filled by the ledger adapter because
"the ledger neither marks nor skips an open lot"
(`src/metrics/pnl_series.jl`). Lots still open at the window end
contribute nothing and are not counted. The ten-year strangle ends with
two such lots.

A fourth thing is missing rather than wrong: **the engine discards its
own failures.** `settlements` returns `(settled, unsettled)` and warns
per unsettled lot, and `run_backtest` uses `.settled` only -- `unsettled`
appears nowhere in `src/backtest/engine.jl`. A `failures` table has
nothing to read until that changes.

## What this round lands

1. The equity curve: open positions marked on a grid, so the curve moves
   between closes.
2. `sharpe`, `sortino`, `volatility` and `max_drawdown` move onto it.
   `hit_rate`, `n_round_trips`, `n_opens`, `n_closes` and
   `profit_factor` stay on the round trips, where they are already
   right.
3. `window_end_spot` and `n_unmarked` resolve -- a mark, or a named
   failure that is counted and listed.
4. The derived persistence tables: `round_trips`, `marks`, `equity`,
   `failures`, and a completeness flag in the manifest.
5. The metric-parameter identity gap deferred from the PR #14 review:
   explicit metric params equal to their defaults fork `full_hash`.
6. The docs sweep: `ledger.md`, `ledger-orchestration.md`,
   `ledger-identity.md` and this file all deleted, their surviving
   content in the module docs.

## Decisions to take before implementing

These are the round's real content. None is settled.

**1. What grid does the curve mark on?** The experiment's `Clock` is
one-minute `option_quote`; marking every tick over ten years is millions
of chain reads, and `sqrt(252)` wants days. A daily grid is what the
ratios mean. Candidates: derive it the way `:session_close` already
derives a session close from the spot tree (one machine, two callers);
a `mark_grid` value in `[outputs]`; or the settlement rule's own
session walk reused wholesale. Prefer reusing the session machinery --
it is written, tested, and already honest about early closes and gaps --
but that is a judgment to confirm, not an instruction.

**2. What is a mark?** `status.md` records the intent: the contract's
own quote mark at that instant, the surface price as fallback. That
needs market data, so the curve cannot be a pure function of the ledger
the way `pnl_series` is: it needs the cut. Decide where it is computed
(inside `run_experiment`, where the data is open) and what its type is.

**3. The curve is an output, not a backtest.** This is the important
one. Marking changes no event, so `core_hash` must not move and a stored
ledger stays reusable -- exactly the reuse the `core_hash` / `full_hash`
split was built for. That means the mark grid belongs in `OutputSpec`
and `full_hash`, and this round breaks run ids **without** invalidating
backtests. Confirm that reading before building anything on it.

**4. `compute_metrics` needs two inputs.** `_METRIC_TABLE` maps a symbol
to a function of a `PnLSeries`. After the move, some metrics take the
curve and some take the series. Either the table carries which input a
metric wants, or both are passed and each function takes what it needs.
The first keeps the functions honest; the second keeps the table
simpler. Choose and say why.

**5. A mark that cannot be produced is a named failure.** Design rule 7:
no NaN standing in for an unanswerable question. It is counted
(`n_unmarked`), named (the `failures` table), and reported once -- the
shape `settlements` already uses for an unpriceable lot. Decide whether
an unmarkable point breaks the curve, carries the previous mark, or
truncates it, and say which in the doc.

**6. The engine must keep its failures.** For a `failures` table to
exist, `run_backtest` has to stop dropping `settlements(...).unsettled`.
That changes what the engine returns, or adds somewhere for them to go.
Smallest honest change wins; a second return value is not obviously
worse than a field.

**7. Is this one pull request or two?** It is three proposal slices. The
equity curve (1-3, 5) and the persistence tables (4) are separable, and
`master` stays usable after either. The four-PR plan says one; the plan
also says to cut where the diff is one thing. If the equity curve alone
runs past about six hundred lines, split it and say so rather than
shipping a reviewer two subjects at once.

**8. What the completeness flag actually asserts.** A boolean in the
manifest is worthless unless its meaning is exact. Candidate: every lot
the run opened was either closed, settled, or marked at the window end,
and no mark failed. Write the sentence before writing the column.

## Out of scope

- `compare_runs.jl` -- named in the proposal's slice 6, but it is a
  cross-run tool over tables that do not exist yet. It follows.
- Compute reuse (skipping a backtest on a `core_hash` hit). This round
  makes it *possible* by keeping `core_hash` still; it does not build it.
- The official-close data kind and the collector work behind it
  (`options-collector`, `us_stocks_sip/day_aggs_v1`). Independent of
  this round and tracked in `market_data.md` and the backlog.
- The named-column parquet writes (backlog). This round is inside
  `store.jl` and the temptation will be real; it is still its own thing,
  and it touches every write site rather than the four being added.
- A second policy.

## Tests

The house rules stand: tests beside the source, one file per source
file; named failures, never bare errors; every failure test checks it
fires, leaves state untouched, and prints its name.

Specific to this round:

- **Ratios on a hand-built curve.** A curve with known daily returns and
  a hand-computed Sharpe. This is the first time these functions can be
  checked against an arithmetic answer rather than against themselves --
  take it.
- **Drawdown sees open exposure.** A position opened, marked down, and
  closed flat has a non-zero max drawdown. Under today's code it is
  exactly zero, which is the defect in one line.
- **A mark that cannot be produced** is counted, named, listed in
  `failures`, and does not silently become a number.
- **`core_hash` does not move** when the mark grid changes; `full_hash`
  does. The whole reuse story rests on this pair.
- **Omitted and explicit metric params hash the same** (the deferred
  finding).
- **Round trip through the store**: the four new tables write and read
  back, and the completeness flag says what it claims.
- **Regression**: the ten-year strangle's *ledger* is unchanged --
  13438 events, 2240 orders, cash USD 32008.66. Its ratios will change,
  and that is the point of the round; record the before and after rather
  than asserting equality.

## How to run here

Gate (2 cores, 3.7 GB; check `free -m` first, and if under about 1.3 GB
available exit the `julia` REPL before running, then relaunch it):

```
ws run "JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=. -e 'using Pkg; Pkg.test()'"
```

Do not wait on pane text and do not `pgrep` for a pattern your own
waiting command contains -- both false-match. Wait on the julia process,
then `ws capture shell 60`.

The ten-year strangle:

```
julia --project=. scripts/run_experiment.jl configs/strangle_spy_16d_1dte.local.toml
```

Not `--save` unless the REPL is down; that needs about 1.7 GB.

## Done means

- The reported Sharpe is annualised against time, not against trade
  count, and a hand-computed test says so.
- Drawdown moves while a position is open.
- No metric reports a placeholder: every open lot is marked or named.
- `core_hash` is unchanged by anything in this round; `full_hash` moves.
- The four derived tables round-trip, and the completeness flag asserts
  a sentence written down before the column existed.
- `docs/proposals/` is empty and the module docs carry what survived.
- Gate green.
