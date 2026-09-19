# `backtest` module

The engine runs an agent over the ticks of a clock, with a market-data
map open, and produces a [`ledger`](ledger.md). The engine computes and
the ledger records: the engine turns a decision into per-leg prices, fees and
observations and makes one call, and every id, group, order record and
event is minted inside that call. The engine holds no state beyond the
ledger, which owns the book it folds.

## What the engine does per tick

1. **Settle** the lots that fell due since the previous tick, one
   `record_expiry!` per lot, so the policy cannot see them. A lot with
   no honest settlement price stays open and visible.
2. **Ask** the agent for its policy, and the policy for its orders, on
   the ledger's own book.
3. **Price** every leg of each order before anything is written: the
   contract must still trade, then its quote, its fill price, its
   underlying's spot, and last the commission of the whole order. A leg
   that cannot be priced is a named failure, so no partial structure
   reaches the ledger.
4. **Book** each order as one transaction, recording the ledger
   sequence its decision saw, captured after this tick's expiries, so
   the second order of a tick did not see the first's fills.
5. **Check** that the fills just written agree with the order record on
   leg, contract, side, intent, group and quantity, and that the price
   is the fill rule applied to an observation taken before the
   decision. The same check runs when a run is saved and loaded.

After the last tick, settlement runs once more at the window end. Lots
still open after it stay open, and every lot a settlement pass could
not price leaves the run as a `RunFailure`, since no event was written
for it.

## The venue

Shaped like Interactive Brokers, as a fill rule and a cost model, both
`Experiment` fields in `core_hash` because they are choices, and a tick
that is a constant because every underlying the contract table lists
trades in one-cent increments at every premium. Settlement style is a
contract fact, read per lot from `contract_spec`. A combo order fills
whole or not at all, as a guaranteed combo does at IBKR.

`:cross_spread` takes the ask on a buy and the bid on a sale, rounded
onto the tick against the trader. It is conservative on purpose: IBKR
fills an all-option combo at one net price, often inside the legs' own
spreads, so crossing every leg pays more than the real venue would.
Margin is not modelled, because there is no capital base for a margin
rule to bind against.

One observation per leg is recorded, the quote and the spot the fill
saw, so the join between a fill and its order is checked rather than
assumed. `:broker_execution` names a price the broker reported; its
observation is kept and never consulted.

## Settlement

A contract settles at intrinsic against the session-close print of its
underlying: the last print inside the settlement session's window,
standing in for the official closing auction, which is not in the data.
That is the one stated departure from the facts here; the payoff itself
is real, intrinsic under exercise by exception. Early assignment and
physical delivery are not modelled: SPY cash-settles here.

**Sessions come from the spot tree; the calendar is a check.** A date is
a session when the underlying printed in the 09:30-16:00 ET window on
it, and its close is the last of those prints, so an early close needs
no table. A printless date the calendar calls open is a named failure,
`:unexpected_gap`, never evidence that the exchange was closed; the walk
back from the listed expiry date is bounded, and exhausting it is
`:no_session`.

**The exposure.** The last print in the window of a 13:00 ET close is
the 13:00 one only where no extended-hours print falls inside the
window. The production tree does serve extended hours (SPY prints from
04:00 to 16:59 ET) and is measured, not promised, to print nothing
between 13:00 and 16:00 ET on any of the six early-close sessions of a
ten-year run, so all six settle at their 13:00 prints. Nothing in a
`SpotPrice` records its session, so nothing here can catch a print that
did. [`data`](data.md) promises nothing about sessions; the
official-close data kind that would make the rule structural is a
backlog item in [status](../status.md).

**The window's last print is a completed minute.** A vendor row stamped
16:00 ET is the 16:00-16:01 minute, after the close; under bar-end
visibility it becomes visible at 16:01, outside the window, and the
15:59-16:00 bar, visible at exactly 16:00, is the last print inside it.

**The window ends at the earlier of 16:00 ET and the contract's own
expiry.** For the 16:00 ET convention the ticker parser stamps that is
the session close; for an intraday expiry it is the expiry, so a
contract never settles at a print from after it stopped existing, which
would show a lot settled at a future number in the effective-time
replay. A contract expiring before 09:30 ET on a date the calendar calls
open has an empty window by construction: that is the AM-settled case,
named `:pre_open_expiry` rather than blamed on the data. AM settlement
itself is parked in [status](../status.md).

**The settlement instant is the contract's expiry.** When the reference
print comes from an earlier session, only which print stands in for the
close moves; the obligation ceased when the contract expired. So an
expiry is effective at the expiry and recorded at the tick that booked
it, which is the one source of the two replays disagreeing.

**A lot is examined for settlement exactly once.** `settlements` takes
the lots falling due in `(prev, t]`. An unsettleable lot stays open, and
its answer is fixed by its expiry, so a threshold of `expiry <= t` would
re-derive the same failure at every later tick. The interval misses
nothing because the venue refuses to fill a leg at or after its
contract's expiry, so every lot opened through the engine is in the book
strictly before its expiry. A caller writing through `record_order!`
directly forfeits that.

**An unsettleable lot stays open, loudly.** `settlement_price` throws a
named failure; `settlements` is the one place that catches it, warns
once with the contract, its expiry and the reason, and returns the lot
paired with the failure. A bad day must not kill a ten-year run, and it
must never pass silently. `settlement_price` also refuses a cut that
does not reach the expiry, `:no_session_close`, so a direct caller gets
a name rather than a provisional intraday print; the tick loop never
reaches it.

**An unserved settlement style stops the run.** `UnsupportedSettlement`
names a contract class nothing here can settle, which every later tick
would answer the same way, so it is not caught like an unpriceable lot.
`load_experiment` throws the same type when it reads such a config.

**The session grid is the same rule, enumerated.** `session_closes`
answers when each session in a window closed, one instant per session,
and the [`metrics`](metrics.md) module samples its marked curve on it:
one rule, so the grid a ratio is annualised over and the price a
contract settles at cannot drift apart. A session counts only when its
whole window lies inside the bounds; a clipped one is temporal absence.
It reads one session window at a time, never the gaps between them,
because the exposure above is bounded inside the windows and the
production tree holds a disagreeing pair at an overnight instant. Each
window is consumed once as a stream, so a provider that streams one
partition at a time serves the rule as well as a vector does.

## Decisions

| Decision | Why |
|---|---|
| **The engine computes, the ledger records, for fills and expiries alike** | The engine keeps no parallel journal, no expiry queue, no cached calendar and no `try`/`catch` in the loop; a live loop replaces `fill_legs` with the broker's reports without touching the writer. |
| **Venue as two symbols and a tick, no `VenueSpec`** | Two plain symbols are what config and identity carry; a struct would name the same things twice. |
| **Fill prices on the tick, rounded against the trader** | The ledger refuses cash that is not whole cents, synthesized quotes are off the tick, and exchanges only trade on it. The observation keeps the raw quote; the fill carries the tick price. |
| **Sessions from the tree, calendar as the check** | A calendar as the source would have to carry every half-day and every ad-hoc closure correctly forever; as the check it only answers whether a printless date was closed, and a wrong answer is loud. |
| **One `record_expiry!` per lot, never a batched commit** | Two lots expiring at one instant are independent facts; batching would claim an atomicity that does not exist, and one unpriceable lot would reject the others. |
| **Settlement as a symbol through a table** | The same shape as the fill rules and cost models, for the same reason. |
| **The venue is stricter than the ledger about expiry** | The ledger accepts a fill effective at the expiry instant; trading has stopped by then. It is also what makes the settlement interval complete. |
| **The warning lives in `settlements`** | A caller could forget to report, and the window-end pass is such a caller. |
| **The loop returns `(ledger, failures)`, not a new type** | A named pair is the shape `settlements`, `session_closes` and `fill_legs` already return. |
| **`known_to` captured once per tick** | Sequence, not recorded time, bounds what a decision saw. |
| **Driven by `Agent`** | One loop serves a fixed policy and a learning agent alike; the bare-policy overload is a convenience. |
| **A declared clock** | The tick grid is part of the experiment; two experiments on the same data with different clocks are different experiments. |
| **`resolve_quote` reads quotes** | A surface retains only inverted IVs; the raw bid and ask a fill needs live on the chain quote. |

## Conventions consulted

| Convention | Source | Consequence |
|---|---|---|
| A multi-leg option order fills in whole units or not at all | Interactive Brokers, [guaranteed combination orders](https://www.ibkrguides.com/kb/guaranteed-non-guaranteed-combo-orders.htm) | every leg priced first, then one `record_order!` |
| Commission per contract by premium tier, with a per-order minimum | Interactive Brokers, [US options commissions](https://www.interactivebrokers.com/en/pricing/commissions-options.php), IBKR Pro fixed, fetched 2026-09-12; the page's worked examples are the test literals | `:ibkr_pro_us_options` |
| Options on SPY, QQQ and IWM trade in one-cent increments at every premium | MIAX, [Options Penny Program](https://www.miaxglobal.com/markets/us-options/all-options-exchanges/penny-program) | `TICK_CENTS = 1`, a constant, since every listed underlying is one of the three |
| An execution report carries a unique execution id | FIX [ExecutionReport](https://www.onixs.biz/fix-dictionary/4.4/msgtype_8_8.html), `ExecID` | one execution id per fill; a duplicate is refused |
| Backend selection by symbol through a table with defaults | Optim.jl, MLJ.jl | the fill-rule, cost-model and settlement-rule tables |
| An expiring in-the-money option is exercised without an instruction | OCC, [exercise by exception](https://www.optionseducation.org/referencelibrary/faq/options-exercise), checked 2026-09-13 | an expiring lot settles at intrinsic with no closing order; the OCC threshold is not modelled |
| Closed days and 13:00 ET early closes are calendar facts | NYSE, [Holidays & Trading Hours](https://www.nyse.com/markets/hours-calendars), checked 2026-09-13 | the calendar is the check on a printless date |
| An expiring option stops trading at 16:00 ET and settles against the 16:00 ET close | Cboe, [extended trading hours FAQ](https://www.cboe.com/document/tech-spec/content/technical-specifications/equity-options-extended-trading-hours-faq/regular-trading-hours-vs.-globalcurb-trading-hours/), checked 2026-09-13 | a leg at or after its expiry is refused, and `parse_massive_ticker` stamps the expiry at 16:00 ET; the 16:15 ET Curb session is not modelled |
| An exchange calendar as a library | [BusinessDays.jl](https://github.com/JuliaFinance/BusinessDays.jl) `USNYSE`, v0.9.25, checked 2026-09-13 | `isbday` is the whole calendar check; the ad-hoc closure set beside it is empty |
