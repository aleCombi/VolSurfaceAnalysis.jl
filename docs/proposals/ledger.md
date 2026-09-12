# Ledger rebuild -- proposal

Status: proposal, 2026-09-11, revised after four reviews (the diagnosis
checked against the code; the event list, the fill's contents, and the
cash, return and time-boundary rules reviewed separately). Nothing is
implemented. The plan replaces the fill-vector ledger with an event
journal booked inside the run, shaped so the live-trading loop of
[vision.md](../vision.md) hands the same book to a policy.

## 1. Problem

The ledger is a fill log. `run_backtest` returns a bare `Vector{Position}`;
`pnl_series` FIFO-matches it after the fact and guesses intent from
direction; `run_experiment` settles through a closure that is neither
configured nor in identity. Verified consequences:

- **Expiry is not an event.** A policy sees expired legs as live. Nothing
  implements the net-open view the policy doc asks policies to derive.
- **The sample unit is a leg.** A strangle is two samples per day, so
  `hit_rate` is per leg and `sharpe` annualises with 252 over about 446
  samples a year on the stored ten-year run.
- **Equity is realised-only.** Drawdown cannot see open exposure, and
  residuals stamped at expiry run the curve past the window.
- **No lineage.** The persisted series is `(idx, timestamp, pnl)`;
  `n_unmarked` counts skipped legs but does not name them.
- **Float quantities leave FIFO residue.** Open 0.3, close 0.1 three
  times: a `5e-17` lot survives and is settled as a fourth sample
  (reproduced).
- **Smaller defects.** Units are per share while docs say contracts and
  USD; `sharpe` subtracts a rate from a dollar PnL; matching ignores
  expiry; `n_opens` counts side flips; `window_end_spot` computes nothing
  yet gates the clock type and errors when missing.
- **The live split is blocked.** `decide` takes a fill vector; a broker
  hands back positions.

## 2. Shape

```
decide ──► Order ──► engine ──► events ──► Ledger ──► Book (view for decide)
                        ▲                     │
             lifecycle model ─────────────────┤
                                              ▼
                        round_trips ──► PnLSeries   ──► hit rate, counts
                        marks       ──► EquityCurve ──► sharpe, drawdown, vol
```

Four principles:

1. **The ledger is an append-only journal of economic facts.** The engine
   is the only writer; nothing is edited. Marks, diagnostics and order
   status live outside it.
2. **The book is a view by replay, never stored.** A backtest policy and
   a live loop receive the same type.
3. **Intent is declared, cash is derived, lineage is recorded.** Orders
   say open or close; cash follows from price, quantity and multiplier;
   every close names the lot it consumed.
4. **Contract facts are a table in code; the simulated venue and the
   stated simplifications are two config values. All three project into
   core identity.**

### Orders

An order is one structure-level instruction: a label, a group, and legs.
Each leg is a contract, a side, an integer number of contracts, and an
intent (`Open` or `Close`). The engine mints groups for opening orders;
a closing order names the group it closes. A close with nothing to close
is an error at fill time. A roll is a close order plus an open order in
separate groups, linked by an operation id.

`decide(policy, t, cut, book::Book) -> Vector{Order}`

### Events

| Kind | What it books | Essential fields |
|---|---|---|
| `Fill` | one execution | order leg id, execution id, contract, side, intent, quantity, price, group, fill-rule id |
| `Match` | one lot allocation after a closing fill | open fill id, close fill id, quantity |
| `Expiry` | one remaining lot reaching settlement | open fill id, quantity, settlement instant and price, outcome |
| `Fee` | a cost tied to its cause | source event id, amount |

A fill is an execution fact and carries nothing about the market it
was filled against; the quote and spot the decision saw belong to the
order journal below. Cash is a method on the event, never a stored
field, so inputs and cash cannot disagree. `Match` exists because one
close can split across several lots and because a change to the
matching rule must not rewrite old results. `Expiry` is per lot, not
per contract or group, so lineage survives mixed expiries inside one
structure. `Expiry` also carries the side and contract of the lot it
closes, copied from the opening fill and checked against it on append,
so every event's cash is local to the event.

**Cash rules**, fixed in slice 1 and tested there. Cash of a fill is
minus side times price times quantity times the contract multiplier.
Cash of an expiry is side times intrinsic at the settlement price times
quantity times multiplier. A match moves no cash. The matches of one
closing fill exhaust its quantity exactly, and consumption of an opening
lot never exceeds its quantity. A fee references the fill that caused
it; a round trip carries the share of that fee proportional to the
quantity it consumed, so per-trip PnL sums to portfolio cash.

Every event shares a header by composition: id, effective time, recorded
time, sequence. Group sits only on lifecycle events. The container is a
vector over a closed union of the concrete kinds: closedness lets
serialisation be exhaustive; at thousands of events performance is a
wash either way.

**Not events.** Orders and the market state they were decided against
(the order journal). Marks (a mark series keyed by time, contract and
source). Valuation failures (a diagnostics table naming the lot and the
reason, per design rule 7; the run is explicitly incomplete). Rolls
(intent). Assignment and exercise are deferred under the lifecycle
model's named no-early-assignment assumption; they are research
extensions with the same header, due once a dividend source exists,
since assignment risk is real near expiry and ex-dividend dates. Cash
movements arrive with the broker adapter.

### Order journal

Observations sit outside the journal of economic facts, whether they
come before a fill or after it. The order journal records, per order,
its legs and, per leg, the quote and spot the engine resolved it
against and the fill rule applied. A fill references its order leg by
id. A broker execution report carries no quote, so this is the only
shape that is complete for both loops without a sentinel or a
fabricated record: a live fill has an order record with no observation
row and a `BrokerExecution` fill rule.

Three records for research: order, order leg, observation per leg.
The order record also carries the ledger sequence at the moment of the
decision, which is the boundary of what that decision could have seen.
Status and split observation tables arrive with the live adapter.

The join is validated, not assumed. At engine append, at persistence
write and at load: every fill's order leg exists; fill contract, side,
intent and group equal the leg's; fill quantity per leg never exceeds
the ordered quantity; execution id is unique per source, so a retried
live fill is idempotent; a research fill's price equals the fill rule
applied to the observed quote. A load that finds a dangling fill
fails; it does not drop the join.

### Book and replay

The book holds lots per group and contract, FIFO within, plus cash.
Two replays: what was *known* (by sequence, up to the boundary an order
record names; what that decision could see) and what was *true* at `t`
(by effective time). Recorded time alone is not a safe boundary for the
first: fills appended at the same tick after the decision share its
recorded time. The two replays differ only by lifecycle booked at the
tick after its instant.

### Tick order

1. **Lifecycle.** Expiries due at or before `t` are booked with
   effective time at the settlement instant and recorded time `t`, so
   a policy sees expired legs gone.
2. **Decide** on the book.
3. **Fill.** Every leg of the order is validated first: quote
   resolvable, executable side present, spot present, something to
   close. Only then is anything written: the journal, then `Fill`, its
   `Match`es and any `Fee`, as one batch. A leg that fails validation
   is an error before the batch, so no partial structure reaches the
   ledger. That a structure fills whole at its resolved quotes is the
   venue's behaviour, not a simplification: a guaranteed combo order at
   Interactive Brokers fills in whole units and never as a lone leg.
   The execution model names it `GuaranteedCombo`; partial fills in
   whole units are a later model.
4. **Window end.** Lifecycle once more, at the evaluation endpoint
   `exp.to`, not at the last policy tick, which may be earlier. Lots
   still open stay open and are marked; that is the unrealised line.

### Contract, venue, simplifications

The engine needs three things beyond the data and the policy to turn a
decision into cash. They are different in kind and are kept apart.

- **`ContractSpec`, per underlying: facts.** Multiplier, exercise style,
  settlement style, delivery. SPY options are 100 per point, American,
  PM-settled, physically delivered. From the OCC spec; the same in
  research and live; wrong is wrong, not a variant. Not config and not
  a data kind (nothing observes it and it has no visibility time): a
  small table in code keyed by underlying, the way exchange calendars
  live in code. An unknown underlying is a loud error. The engine
  projects the *resolved* values into identity, so a correction to the
  table is a new run id, never a silent change to old results.
- **`ExecutionModel`: the simulated venue, shaped like Interactive
  Brokers.** Three parts, each a choice. *Structure:* a combo order
  fills in whole units or not at all (`GuaranteedCombo`), as IBKR
  guarantees for all-option combos on one underlying; partial fills in
  whole units are a later model, and a lone leg never happens. *Price:*
  how resolved quotes become leg prices. `CrossSpread` (buy at the ask,
  sell at the bid, per leg) is the current behaviour and a conservative
  one; IBKR fills a combo at one net price on the exchange's complex
  order book, often inside the legs' spreads, and allocates leg prices
  from it, so a net-price rule with a stated fraction of the combined
  spread is a later model. *Cost:* a per-contract commission booked as
  a `Fee` on each fill, with a per-order minimum, defaulting to IBKR's
  published fixed-rate schedule for US options (the numbers are cited
  from their page when the model lands); zero stays selectable. Margin
  checks and order rejections are not modelled: there is no capital
  base. The live loop replaces the whole model with the broker, whose
  fills carry `BrokerExecution` in place of a rule and whose
  commissions arrive as their own reports.
- **`LifecycleModel`: stated departures from the facts.** Where the data
  or the scope cannot support the real mechanic, the simplification is
  named: SPY cash-settles at intrinsic instead of delivering shares; the
  *session-close print* (the underlying's last regular-session print on
  or before the close) stands in for the official close; early
  assignment never happens. Sessions are derived from the spot tree and
  validated against an exchange calendar (BusinessDays.jl's NYSE
  calendar is the ecosystem's standard): a weekday with no prints that
  the calendar does not list as closed is a named valuation failure,
  never evidence that the exchange was closed, and the lot stays open.
  This is the settlement backlog item, unchanged in substance. The live
  loop replaces it with what the clearinghouse actually does.

All three project into `core_hash`, each for its own reason: a wrong
multiplier changes cash, the venue is an experimental choice, and two
runs under different simplifications are different experiments.

### Derived tables

- `round_trips(ledger)`: one row per consumed lot, from `Match` and
  `Expiry`. Data-free.
- `pnl_series(ledger; unit = :structure)`: per structure by default.
  Hit rate, profit factor and counts read it.
- `equity_curve(ledger, marks)`: cash plus marks on the session grid
  of the window, every session included. Needs a mark source, so it is
  not data-free; the marks it used are persisted with timestamp and
  source, so a reloaded number can be explained and recomputed.
- **The series the ratios read** is the session-to-session difference
  of cash equity, with flat sessions contributing zero. Sharpe, Sortino
  and volatility are computed on that series and annualised by
  sessions per year; drawdown is peak-to-trough on equity levels, in
  cash. There is no capital base, so there are no returns and no
  risk-free term to subtract: that is the convention, not a
  simplification. Return-based metrics wait for a capital convention.

### Invariants

Sequence is replay order; ids are stable and never the index; effective
time need not be monotone; every reference points backward; quantities
are positive integers; a close matches only within its group and
contract; the matches of a close exhaust it exactly and never
over-consume a lot; no fill after expiry; replayed cash equals the sum
of event cash, and the incrementally updated book equals the full
replay; every fill joins to exactly one order leg; an expiry's copied
side and contract equal its opening fill's.

## 3. Decisions

Each changes stored results; settle them together with the id break the
settlement item already budgeted.

1. Quantity is an integer number of contracts, price per share, cash
   in whole USD cents, multiplier from the contract spec.
2. `Match` is an event; matching is within the named group; matches
   exhaust the closing fill; fees are shared by quantity in whole cents,
   by cumulative rounding, so the shares sum to the fee exactly.
3. Bitemporal header: effective time, recorded time, sequence. A
   decision's view is cut by sequence, not by recorded time.
4. Closed union container.
5. Marks, failures and order context stay outside the journal; a fill
   carries only the execution and joins to its order leg by id.
6. Sample unit is the structure. Ratios read session-to-session
   differences of cash equity, flat sessions as zero, no capital base,
   no risk-free term. Drawdown reads equity levels.
7. `ContractSpec` is a table in code keyed by underlying, its resolved
   values projected into identity; `ExecutionModel` and `LifecycleModel`
   are config values in core identity.
8. Open lots at the window end are marked at the evaluation endpoint,
   not force-settled; `window_end_spot` goes.
9. SPY settles at intrinsic against the session-close print, standing
   in for the official close. Sessions from the spot tree, validated
   against an exchange calendar; a gap is a named failure.
10. A structure fills whole or not at all, as a guaranteed combo does
    at IBKR; a leg that cannot be priced is an error before anything is
    written. Commissions are per-contract `Fee` events on each fill.
11. `Expiry` copies side and contract from its opening fill, checked on
    append, so every event's cash is local.
12. Assignment and exercise deferred under the named assumption, as
    research extensions; cash movements with the broker adapter.

## 4. Slices

Each lands green with its module doc.

1. `ledger` module: order and event types, `Book`, both replays,
   `round_trips`, the cash rules with the contract spec as a ledger
   parameter, and a `pnl_series(ledger)` adapter so metrics stay green
   until slice 5. Pure; tests on hand-built ledgers with hand-computed
   answers: a full round trip, a close split across lots, two groups on
   one contract, mixed expiries in one group, fees across a partial
   close, a lot left open at the window end; incremental book equals
   full replay in every case. Lands beside `positions`, which stays
   until the engine switches.
2. Engine: `Order` in, `Book` out, `Match` and `Fee` booked; the
   order journal written and its join invariants tested;
   `DailyShortStrangle` emits one order per day; `positions` retired
   and the `pnl_series(ledger)` adapter takes over.
3. Lifecycle: `LifecycleModel` with the session calendar, `Expiry` in
   the tick loop, window-end lifecycle at the evaluation endpoint. Then
   one auditable strangle run before anything widens.
4. `ExecutionModel` and `LifecycleModel` in config and identity; the
   resolved `ContractSpec` projected into identity; schema version
   bump.
5. Metrics: structure-level series; equity curve from chain-mid marks;
   ratios move onto it.
6. Persistence: `events`, `orders` (orders, legs, observations),
   `round_trips`, `marks`, `equity`, `failures` tables, and a
   completeness flag in the manifest; `load_run` rebuilds the ledger
   and validates the fill-to-order join; `compare_runs.jl` follows.
7. Docs per slice; this proposal deleted once landed.

## 5. Conventions consulted

Moves to the module doc when code lands (design rule 5).

| Convention | Source | Consequence |
|---|---|---|
| Append-only journal; corrections are new entries | Double-entry practice | events immutable, never edited |
| Orders declare intent; a close with nothing to close is rejected | US broker tickets; OCC position reporting | `Leg.intent`; engine errors |
| Positions are the net of fills; lot pairing is recorded under a named rule | IBKR statements (positions per contract, FIFO default lot method); IRS FIFO default | `Book` by replay; `Match` event |
| A multi-leg option order fills in whole units, one execution report per leg, commissions per contract | Interactive Brokers: combo orders (guaranteed vs non-guaranteed), TWS API execution reports, US options commission schedule | `GuaranteedCombo`; one `Fill` per leg; `Fee` per fill |
| An execution report carries ids, quantity, price and time, not the quote the client saw | Broker execution reports | `Fill` is execution only; the order journal holds the quote |
| Prices per share, cash per contract times 100; style, settlement and delivery are listed per product | OCC contract spec | `ContractSpec` table in code, per underlying |
| Exercise by exception at expiry; PM settlement against the official close | OCC Rule 805; Cboe procedures | `Expiry` outcome; `LifecycleModel` names where it departs |
| Realised and unrealised are separate lines; drawdown on equity | Fund reporting | `round_trips` vs `equity_curve` |
| Sharpe is a statistic on a return series with a stated period; without a capital base the analogue is the cash-PnL difference series | Sharpe (1994); fund reporting | session-difference series, sessions per year |
| Exchange holidays come from a calendar, not from gaps in the data | NYSE holiday schedule; BusinessDays.jl | sessions validated; gaps are named failures |
| Composition plus accessor methods, not inherited fields | Julia manual, Interfaces | shared event header |
| Avoid abstract-element containers; small closed unions are the idiom | Julia manual, Performance Tips | closed union vector |
