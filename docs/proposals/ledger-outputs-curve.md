# Ledger outputs: the marked curve and metrics

This is the first half of the outputs round described in
[ledger-outputs.md](ledger-outputs.md). It fixes the time-based metrics
and leaves derived exports and failure retention to the second half.
It also takes the load path and schema change needed to keep loaded
results honest. The central change is small in concept: round trips
remain the record of realised trade outcomes, while a new session-close
curve becomes the record of portfolio profit through time.

## The defect

Today every metric reads `PnLSeries`, whose samples are structures at
the instants they close. That is the right unit for questions about
trades: total realised profit, hit rate, profit factor and the counts of
round trips, opens and closes. It is the wrong unit for questions about
time.

Sharpe, Sortino and volatility nevertheless treat each closed structure
as one period and multiply by the square root of 252. The scale is
therefore trade count, not trading days. A strategy closing about 252
structures a year can look plausible by accident; a weekly strategy is
silently overstated by roughly a factor of two. Holding the same trades
overnight or for a month produces the same ratio, and structures closed
at one instant can still become separate observations. Dollars divided
by dollars conceal the mistake because the result has no units.

`max_drawdown` has the same root defect in another form. It walks the
cumulative profit of closed structures, so the curve is flat while a
position is open. A position may move deeply against the book and
recover before closing without producing any drawdown at all.

The repair is not to reinterpret a trade sample as a day. The metrics
that describe a path need a path sampled in equal periods of time. The
trade metrics continue to use round trips.

## The accounting boundary

At any instant, marked profit is realised profit plus fees on fills
that have not yet been fully allocated to a closed round trip, plus the
unrealised profit of the open book:

> marked profit = realised profit + unallocated fees + unrealised profit

This identity bounds the round. The ledger alone supplies the first two
terms. Only unrealised profit needs market data, and it is zero whenever
the book is flat. Building the curve therefore does not require a new
account model or any change to ledger events; it requires valuing only
the inventory that remains open at each point on the grid.

Ledger cash is not a substitute. Selling a strangle credits premium at
open, but that receipt is offset by the liability represented by the
short options. A curve made by sampling cash would report the opening
premium as profit and materially overstate a short book.

## Capital and returns

The codebase models profit from zero and has no deposited account or
net asset value. In particular, a short option position receives cash
rather than paying a purchase price, so there is no natural capital
denominator already present in the run.

For this round capital is fixed at 1, and that is exact rather than an
approximation. With a constant capital base and a zero risk-free rate,
dividing every period's dollar profit by the same capital scales both
the mean and its standard deviation equally. The scale cancels from
Sharpe, so Sharpe on dollar changes is the same number as Sharpe on
returns for capital 1, 100,000 or any other positive constant. The same
constant-scale reasoning applies to the other dispersion calculation.

Capital becomes meaningful only when the risk-free hurdle is non-zero,
the base compounds, or results are printed as percentages. Real funds
normally use NAV; option backtests often declare a notional base; Cboe
option-writing indices use fully collateralised notional. Broker margin
is not a universal base and would make the result broker-dependent.
None belongs in this round. A separate economic caveat also remains:
standard deviation treats frequent small gains and rare large losses
symmetrically, so Sharpe can flatter short-premium strategies even when
it is computed correctly.

## Decisions

1. **The grid is the close of each trading session and is derived, not
   configured.** Reuse the session machinery already used by
   `:session_close`, including its treatment of early closes,
   unscheduled closures and gaps in the spot tree. Daily observations
   are what annualisation by trading sessions means. There is no
   `mark_grid` setting.
2. **An open contract is marked at its own quote mid, with the surface
   price as fallback and a named failure after that.** Last trade is
   unsuitable because it may be stale for hours at a quiet strike.
   Marking runs inside `run_experiment`, while the cut is open. It is
   intentionally not a pure function of the ledger. The expected cost
   is modest: the strangle holds at most two lots, requiring roughly two
   lookups per session, or about five thousand across ten years.
3. **Anything derivable is recomputed when a run is loaded.** Derived
   tables are still written to parquet so cross-run SQL can query them
   without replaying ledgers, but they are exports, never inputs to an
   `ExperimentResult`. Today `load_run` rebuilds the ledger through
   `commit!` and then reads `pnl_series.parquet`; after this round it
   derives the round trips and marked curve from the ledger and market
   data it just loaded instead. A loaded result is therefore always
   truthful to its authoritative inputs. Loading degrades by piece when
   market data is absent: the ledger and trade metrics still load, but
   the marked curve does not. Unlike round trips, it is not a function
   of the ledger, so its recomputation needs the market data open. Accept
   that boundary for now rather than reading a stored curve as truth.
   This makes the backlog item **Dataset fingerprint in identity**
   load-bearing. Previously stale or re-collected data could make a
   rerun differ under one id; now a plain load can differ while appearing
   to read recorded history. This round names that risk and does not
   solve it.
4. **This remains an output-side change.** Marks cannot alter an event,
   so `core_hash` must not move and an existing ledger remains reusable.
   The round changes `full_hash` and therefore run ids. Any new identity
   projection belongs to `OutputSpec`, not the backtest core.
5. **Capital is fixed at 1 and is not an input.** At zero risk-free rate
   it cannot change a reported ratio, and a powerless configuration
   value would only create a contract to maintain. When capital first
   matters, it is added with non-zero-rate support on the output side.
   It would enter `core_hash` only if a future policy sized positions
   from equity; no current policy does.
6. **The risk-free rate comes from market data, not an experiment
   field.** `RateCurve` already has a currency selector, a `rate_curve`
   data specification, a constant provider and identity through the
   data specs. It stays zero in this round. Turning it on must choose the
   short-end tenor for the per-session hurdle and explicitly retain the
   economically intentional coupling to the curve used by
   `SurfaceFrom`.
7. **Delete today's `equity_curve`; do not preserve or rename the
   realised-only helper.** Once a marked curve exists, the old curve is
   the wrong input for every path metric and chart that uses it. Keeping
   two near-identical names makes accidental misuse likely. The module
   documentation should still use the standard realised/unrealised
   vocabulary. `PnLSeries` goes too: the schema is moving in this round,
   so preserving the obsolete wrapper only defers another schema break.
   Its `window_end_spot` placeholder has no successor and is removed;
   `n_unmarked` belongs with the marked curve.
8. **Move only the path metrics.** Sharpe, Sortino, volatility and
   maximum drawdown use the marked curve. Hit rate, number of round
   trips, opens, closes and profit factor remain on round trips, where
   their sample unit is already correct. Preserve that structure-level
   grouping as a function returning plain per-trade dollar vectors, not
   a struct with four scalars attached. `n_opens` and `n_closes` remain
   real outputs and need an explicit home. `total_pnl` remains the
   ledger's realised total rather than quietly changing meaning.

## Open design before implementation

Three choices remain deliberately unsettled. They must be resolved in
the implementation brief or review before code fixes their answers.

- Decide what one unmarkable point does to the curve: break the curve,
  carry the previous mark, or truncate it. Whichever policy wins changes
  the ratios. Design rule 7 already fixes the invariant around that
  choice: the failure is named and counted and never represented by
  `NaN` or silently turned into an ordinary value.
- Decide how `compute_metrics` receives its two honest inputs. Its
  dispatch table may record whether a metric consumes the marked curve
  or per-trade dollar vectors, or the dispatcher may pass both and let
  each function select its input. As part of that choice, give the real
  `n_opens` and `n_closes` counts a home outside the deleted wrapper.
- Name the curve and its plain concrete type. Use parallel vectors, not
  a hierarchy. Also name the conversion boundary where whole ledger
  cents meet floating-point marks, as explicitly as `contract_cents`
  names the existing rounding boundary.

## Scope

This brief covers construction of the session-close marked-profit
curve, migration of the four path metrics and their dispatch, removal
of the realised-only curve and the `PnLSeries` type, preservation of
structure-level round-trip grouping and the open/close counts, the
necessary output identity change, the load path, and the schema-version
bump. Run ids moving at this stage is explicitly acceptable. The marked
curve may read the experiment's already-open market-data cut; on load it
must reopen that data and recompute rather than hydrate a stored curve.
It must not change the ledger or the backtest that produced it.

The second half of the outputs round owns the derived persistence
exports, the `failures` table, the manifest completeness flag, and
retaining `settlements(...).unsettled` from the engine. It does not build
separate marks and equity tables: the recomputed marked curve is the one
path result and any parquet form is an export only. Also out of scope
are the metric-parameter identity gap from the PR #14 review, compute
reuse, the load-bearing dataset-fingerprint backlog item, a non-zero
risk-free rate, capital or NAV reporting, and a second policy.

## Tests

- Check Sharpe against a hand-computed arithmetic answer from known
  daily marked-profit changes.
- Pin the capital argument: scaling the same curve from capital 1 to
  capital 100,000 produces exactly the same zero-rate Sharpe.
- Open a position, mark it down, then close it flat; maximum drawdown
  must be non-zero although today's implementation reports zero.
- Give two ledgers identical trade profits but different holding
  periods; their annualised path ratios must differ because the session
  grid, not trade count, sets the observations and scaling.
- Exercise the chosen unmarkable-point policy and assert that the point
  is counted and named, never represented silently as a number.
- Assert that the round leaves `core_hash` unchanged and changes
  `full_hash`.
- Save and load a run with data available; assert that the marked curve
  and all metrics are recomputed, not read from derived parquet.
- Load without market data; assert that the ledger and trade metrics are
  available while the marked curve and its path metrics degrade by the
  declared boundary rather than making the whole load fail.
- Pin the schema-version bump and the absence of `PnLSeries`, including
  a home for `n_opens` and `n_closes` and structure-grouped dollar inputs
  for hit rate and profit factor.
- Rerun the ten-year strangle and verify that its ledger remains 13,438
  events, 2,240 orders and USD 32,008.66 cash. Record ratio values before
  and after rather than asserting equality; changing them is expected.

Tests remain beside the source they exercise, one test file per source
file. New failure paths use named errors, and their tests check the
error name and that failed construction does not masquerade as a
complete curve.

## Done means

The round is complete when Sharpe, Sortino and volatility observe
session-to-session marked-profit changes and annualise by sessions
rather than closed trades; maximum drawdown moves while a position is
open; and trade metrics still read structure-grouped per-trade dollars.
Every open lot at every required point is either honestly marked or
accounted for by the chosen named-failure policy, with no placeholder
field or `NaN` standing for an unanswerable mark.

Capital remains fixed at 1 with its cancellation pinned by a test. The
ledger and `core_hash` are unchanged, while the output identity and
`full_hash` reflect the new result and the schema version has moved.
`PnLSeries`, `window_end_spot` and the old realised-only curve are gone;
the round-trip grouping and open/close counts survive without them.
`load_run` rebuilds derived results, exposes the ledger and trade metrics
without data, and exposes the marked curve only when its market data is
available. No derived parquet table is read into the result. The
ten-year regression preserves its events, orders and cash, the focused
and full test gates pass, and `docs/modules/metrics.md` describes the
delivered boundaries.
