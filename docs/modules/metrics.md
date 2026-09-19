# `metrics` module

Metrics are ordinary functions over two inputs: a marked curve, for
questions about time, and a vector of dollars per closed trade, for
questions about trades. A metric's answer is only as honest as its
sample unit: Sharpe, Sortino and volatility observe periods of time
and annualise by sessions per year, drawdown observes the path in
dollars, and total profit, hit rate and profit factor observe trades.
Only the curve needs market data, the spot tree for its session grid
and quotes or surfaces to mark an open book; everything else is a
function of the ledger.

## The marked curve

Profit is measured from zero. The codebase has no deposited account and
no net asset value, and none is invented here. Capital is fixed at 1,
and at a zero risk-free rate a constant base cancels from every ratio,
so the number is exact; a NAV denominator is parked in
[status](../status.md).

Marked profit at an instant is the ledger's cash plus the open book
marked to market, because cash already carries the cost basis of every
open lot with the opposite sign. Cash alone is therefore not the curve:
selling a strangle credits premium at open, and a curve sampling cash
would report that receipt as profit.

The grid is the close of each session in the window, from
`session_closes` in [`backtest`](backtest.md), the same rule settlement
uses, so the grid a ratio is annualised over and the price a contract
settles at cannot drift apart. A session counts only when its whole
window lies inside the bounds; a clipped session is one this run did
not see end to end, and that absence is temporal.

A lot is marked at its own quote mid, then at the surface price, then
it is a named failure, `:no_mark`. The mid rather than the last trade,
because a trade may be hours stale at a quiet strike while a two-sided
quote is a price someone stands behind at `t`; a one-sided, non-finite
or negative quote falls through to the surface as an absent one does.
The surface must be stamped at `t` exactly: a surface built earlier
values the contract at its own instant, and reusing it across sessions
would carry a price forward. Marking reads through a cut at each point,
so no-lookahead holds here as in the tick loop.

**One lot without an honest price makes the whole session
unanswerable.** The point leaves the curve, is named and counted, and
`session_changes` refuses to span it: a break costs two observations,
since neither the step into the broken session nor the step out of it
is a period this run observed. The builder still examines the rest of
the session's lots and retains a `RunFailure` for each it could not
price, so the curve carries one unmarked entry per broken session and
the failures name every lot, and the two agree instant for instant.
`max_drawdown` walks the marked levels across a break, so over a broken
curve it is a lower bound and `n_unmarked` says how much it could not
see. Breaking is right for a backtest, which has no obligation to
publish a number; the trigger for rethinking it is volume, and the
answer then is official per-series marks rather than invented values.

Whole cents become dollars at one named crossing, `cents_to_usd`, the
counterpart of the ledger's `contract_cents`. Marks are floating point
at source and never cross it.

## Dispatch

The always-on core, total profit, round trips, opens, closes and hit
rate, is computed on every call. Optional metrics are requested by
symbol, and each symbol carries its own default keyword arguments, so
the symbol is the whole contract. Every metric takes both inputs and
reads whichever is its sample unit, so when there is no curve the whole
optional set is omitted, `profit_factor` included though it needs no
curve. An absent key means not computed; `NaN` means computed and
undefined, as Sharpe is on one session change. No runner produces an
absent curve today, `marked_curve` always builds one and `load_run`
computes no metrics, so that arm serves a direct caller.

`trade_pnl` gives one dollar figure per closed structure by default, or
per leg on request. The pairing is the ledger's own, `Match` and
`Expiry` under a named rule, never inferred here, and the entries are
ordered by closing instant then profit, losses first, so a run's trade
vector is reconstructible.

## Decisions

| Decision | Why |
|---|---|
| **Two inputs, not one series** | Before this, every metric read a series of closed trades, so Sharpe annualised by 252 while its observations were trades, and a curve of closed trades is flat while a position is open, so drawdown saw nothing. A Sharpe ratio is a statistic on a series with a stated period (Sharpe 1994). |
| **Ordinary functions, no `Metric` type** | A registry type would add ceremony without polymorphism anyone needs; the symbol table gives selection by name. |
| **Every metric takes both inputs** | Adding a metric is one table row and one function. The price is an ignored argument per body and the wholesale omission when there is no curve. |
| **Capital fixed at 1, not an argument** | A keyword that cannot change a result is only a contract to maintain; the tests pin the cancellation by scaling a curve. |
| **The grid is derived, not configured** | Daily observations are what annualising by sessions means; a configurable grid lets a run pick a cadence its scaling constant does not match. |
| **Break the curve rather than carry or truncate** | Carrying the previous mark invents a value; truncating discards honest points after the first gap. Breaking loses exactly the unanswerable observations, and the count says how many. |
| **`total_pnl` stays realised** | Changing what a reported number means while keeping its name breaks every comparison with a stored run. |
| **`sortino` sits beside `sharpe`** | Standard deviation treats frequent small gains and rare large losses alike, so Sharpe flatters a short-premium book; Sortino is not blind in the same way. |
| **Default kwargs in the dispatch entry** | One source of truth per symbol; two call sites cannot drift. |

## Conventions consulted

- **`MarkedCurve` as a plain struct of vectors, no supertype, no type
  parameters.** Julia style guide ("don't use unnecessary static
  parameters"); Tables.jl (interface objects need not subtype);
  BlueStyle (concrete field types).
- **`cents_to_usd` as one named conversion.** `Dates.value` and
  FixedPointDecimals.jl: the ecosystem crosses from an integer
  representation through a named accessor, never an implicit promotion.
- **A symbol table of `(fn, defaults)` named tuples.** MLJ.jl's registry
  entries are named tuples selected by name; Optim.jl's singleton
  instances are what a table graduates to when each entry needs its own
  dispatch, which five reductions do not.
- **`Union{MarkedCurve,Nothing}` for an absent curve.** Julia manual
  FAQ: `nothing` is the absence of a value,
  `missing` an unobserved observation. An empty curve would conflate
  not computed with computed and empty.
