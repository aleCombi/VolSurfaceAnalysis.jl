# `backtest` module

The driver that turns an [`Agent`](agents.md) (which hands out a
[`Policy`](policies.md) per tick) plus a [`MarketData`](data.md)
map and a `Clock` into a [`Ledger`](ledger.md), and the simulated venue
that prices what the policy decides. One principle: **the engine
computes, the ledger records.** The engine turns a decision into
immutable inputs (per-leg prices, fees and observations) and makes one
call; every id, the group, the order record and the events are minted
inside that call. The record is constructed before the event commit;
afterward it is pushed beside the events and the counters advance. The
lifecycle step has the same shape: a function of the cut says which lots
fell due and at what price, and the loop calls the ledger's own
`record_expiry!`. The engine defines nothing that mutates, and holds no
state beyond the ledger, which owns the book it folds.


## The tick order

1. **Lifecycle.** The lots falling due in `(prev, t]` are settled
   before anything else, so a lot that settled is already gone from the
   book the policy is handed. A lot whose settlement price could not be
   resolved stays open, stays visible to every later decision, and has
   been warned about once; `lot.contract.expiry <= t` is what tells a
   policy it is holding one -- inclusive at `t`, because the interval
   that examines a lot is too, so the tick *at* an expiry already hands
   the policy the lot that failed to settle there. `settlements`
   computes what settles and at
   what price; the ledger's own `record_expiry!` books each one,
   separately -- two lots expiring at one instant are independent facts,
   and batching them would claim an atomicity that does not exist.
2. **Decide.** The cut at `t`, the current policy, and `L.book`: the
   ledger's owned fold, kept equal to `book_as_known(L, known_to)` by
   supported ledger calls for every order record the tick produces. A
   policy reads it and must not mutate it; direct mutation would be outside
   the supported ledger boundary.
3. **Fill.** For each order, `fill_legs` prices every leg before
   anything is written: the contract still trades at `t` (its expiry is
   strictly later), the quote through `resolve_quote`, the price through
   the fill rule, the spot of the leg's own underlying, then the
   commission of the whole order. A leg that cannot honestly be priced
   is a named failure here, so no partial structure reaches the ledger.
   Then `record_order!` books the structure as one transaction; every
   order of a tick records the ledger sequence before the tick's first
   order as what it could see (`known_to`), so the second order of a tick
   does not appear to have seen the first order's fills.
4. **Window end.** The lifecycle once more, at the evaluation endpoint
   `to`, which may be later than the last policy tick. Lots still open
   after it stay open; nothing is force-settled.

`known_to` is captured *after* the tick's expiries, so the sequence an
order records as what its decision saw already includes them. The
per-record `check_join` runs immediately after each append. The
whole-ledger form remains the persistence write/load audit.

## The venue

Shaped like Interactive Brokers, as three values: a price rule, a cost
model and the class's tick. The rule and the model are symbols dispatched
through two small tables in the `_METRIC_TABLE` style (`Fill.fill_rule`
is literally the table key) and they are *choices*, so they are
[`Experiment`](experiment.md) fields, part of `core_hash`, and keywords
here with the same defaults for a direct caller. The tick is not a
choice: `TICK_CENTS`, a constant, because every underlying the contract
table lists trades in penny increments at every premium. It stays a
parameter of `fill_price`, `fill_legs` and `check_join` so the join check
can recompute a fill from an observation and state the tick it checked
against. Settlement style is not a value here at all: it is a contract
fact, read per lot from `contract_spec`.

- **Structure.** A combo order fills in whole units or not at all, as
  a guaranteed combo does at IBKR; a lone leg never happens. This is
  not a value: it is `record_order!` plus the engine pricing every leg
  first.
- **Price** (`fill_price(rule, bid, ask, side, tick_cents = TICK_CENTS)`, raw values
  in so `check_join` can recompute it from an observation).
  `:cross_spread`: a buy takes the ask, a sale the bid, rounded onto the
  tick away from the trader (rule R5 below); a missing required side is
  `missing`. `:broker_execution` is not a rule of ours and is not in the
  table: it names a price the broker reported. The engine still records
  one observation per leg under it, permits missing quote sides, and does
  not consult that observation; an absent observation waits for the live adapter.
`:cross_spread` is **conservative rather than accurate**, and that is the
point of it. IBKR fills an all-option combo at one *net* price on the
exchange's complex order book, often inside the legs' own spreads, and
allocates leg prices from that; crossing every leg pays more than the real
venue would. A net-price rule taking a stated fraction of the combined
spread is the later model, as are partial fills in whole units. Margin
checks and order rejections are not modelled at all -- there is no capital
base for one to bind against, so a margin rule here would be a number
invented to constrain another invented number.

- **Cost** (`commission(model, prices, quantities)`, non-negative cents
  per leg, negated into `Fee` amounts). `:none`, and
  `:ibkr_pro_us_options`: IBKR Pro's fixed-rate US options schedule at
  the lowest monthly-volume tier with its per-order minimum, the order's
  total shared over the legs in whole cents by cumulative rounding so the
  shares sum exactly. Higher tiers and third-party fees are not modelled.
  A commission of zero books no `Fee`.

An unknown rule or model errors naming the known ones.

## Lifecycle and settlement

A `ContractKey`'s `expiry` is an instant, and the ticker parser stamps
the listed date at 16:00 ET; nothing constrains it to that. What settles
a contract is the *session-close print*: the underlying's last
regular-hours print of the settlement session, which the rule reads as
the last print of its reference window and which is the same thing only
where the spot tree holds regular-session prints alone, which the
production tree does not. That is a stated departure from the
facts -- the official closing auction is not in the data -- and it is
the only one here; the payoff itself is real, intrinsic under exercise
by exception.

**Which lot settles under which rule is a contract fact.**
`settlements` reads `contract_spec(underlying).settlement` per lot, so a
book holding two styles settles each correctly; a run-level symbol could
not, and agreed with the table today only because every underlying it
lists is `PMSettled`. A style no rule serves is
`UnsupportedSettlement`, which stops the run -- see the failures below.

**Sessions come from the spot tree; the calendar only contradicts it.** A
date is a session when the underlying printed in the reference window on
it, and its close is the last of those prints. Nothing else is
consulted, so an early close needs no early-close table -- but what makes
the answer right there is a property of the input, not of the bounds: the
last print in the window of a 13:00 ET close is the 13:00 one only where
the tree holds regular-session prints alone. Where it does not, the rule
breaks silently: a 15:59 extended-hours print on an early-close day is
inside the 09:30-16:00 window and becomes the settlement price, and
nothing here can tell it apart from a regular one -- a `SpotPrice` does
not record which session it came from, and no narrower window helps,
since 15:59 is regular-hours-shaped. The production tree does serve
extended hours (SPY prints from 04:00 to 16:59 ET) and is *measured* not
to print inside that window on the dates that matter: none of the six
early-close sessions of a ten-year SPY run has a bar between 13:00 and
16:00 ET, so all six settle at their 13:00 prints. The exposure is this
module's own: [`data`](data.md) promises nothing about sessions, and the
official-close data kind that would make the rule structural is a
backlog item in [status](../status.md). The
exchange calendar answers one question only, and it is a *check*: a
printless date the calendar calls open is a data gap, named and
reported, never evidence that the exchange was closed (design rule 7). Ad-hoc closures the calendar may lag behind have
a cited `const` seam beside it, empty today.

**The window's last print is a completed minute.** The rule's bounds are
unchanged by bar-end visibility ([`data`](data.md)); its input is what
became honest. A vendor row stamped 16:00 ET is the 16:00-16:01 minute,
after the close, and under a bar-open stamp it could win the window and
settle the contract. Under bar-end visibility it becomes visible at
16:01, outside the window, and the 15:59-16:00 bar -- visible at exactly
16:00 -- is the last print inside it. An early close does not move at all:
its winning record is the 13:00-13:01 bar, which was stamped 13:00 and
sat on the window boundary before and is visible at 13:01 and one minute
inside it now -- the same row, the same price, and the same exposure
to the tree holding nothing else in that window.

**The session grid is that same rule, enumerated.** `session_closes`
answers "when did each session in this window close", one instant per
session, and it is what the [`metrics`](metrics.md) module samples its
marked curve on -- one rule, so the grid a ratio is annualised over and
the price a contract settles at cannot drift apart. It returns the
printless open dates separately as `gaps` rather than skipping them, the
`:unexpected_gap` case in grid form, and it counts a session only when
its *whole* reference window lies inside the requested bounds: a window
the bounds clip is a session the caller did not see end to end, which is
temporal absence, not a failure.

It reads **one session window at a time**, exactly the windows
`:session_close` reads, and never the gaps between them. That is not an
implementation detail: the session-window exposure is bounded inside
those windows and nowhere else, and the production tree does hold a
disagreeing pair at an overnight instant, so a single range read across ten
years would abort on data the rule is not entitled to and does not need.

**The reference window never runs past the expiry instant.** It opens at
09:30 ET and closes at the earlier of 16:00 ET and the contract's own
expiry. For the 16:00 ET convention the parser stamps, and for every
walked-back session, that is the session close and nothing changes; for
an intraday expiry it is the expiry. Without the bound such a contract
settles at a print from after it stopped existing -- permitted by the
tick's cut, which is at or after the expiry, but incoherent in the
effective-time replay: an `Expiry` stamped at noon would carry a price
that did not exist until 16:00, so `book_effective` at 12:30 would show
a lot settled at a future number.

**The window is consumed as a stream.** `between` promises an iterable,
not a container, and settlement is its only consumer outside
`data`. Each window is therefore traversed exactly once, keeping
the last record it yields -- never indexed, and never probed for
emptiness and then read again. A provider that streams its range one
partition at a time, as the parquet bar reader already does, serves the
rule as well as an in-memory vector does.

**The rule has a domain, and says so.** `settlement_price` requires the
cut to reach the contract's expiry; a cut short of it is
`UnpriceableLeg(..., :no_session_close)`. A cut that cannot see the
settlement session's close cannot answer *what did this settle at* -- it
would return a provisional intraday print, or read a truncated window
and walk back past a session that had not finished printing -- and
design rule 7 gives that a name instead of a plausible-looking number.
The tick loop can never reach it, since lifecycle builds the cut at a
tick at or after the expiry; the check is for a direct caller driving
the exported lifecycle step from its own loop.

The other half of the domain is the contract rather than the cut.
`:session_close` is PM settlement, so an expiry earlier than 09:30 ET on
its own listed date has no session close behind it: the window the rule
would read is empty however complete the data is, and what such a
contract settles against is the *opening* print -- AM settlement, which
no rule here serves yet. That is `UnpriceableLeg(..., :pre_open_expiry)`,
naming the contract instead of reporting a data gap that adding data
could never close. It applies only when the calendar calls the listed
date open; when the listed date is not a session at all, the walk back
answers as it does for any closure. An expiry at exactly 09:30 ET is a
one-instant window, not a degenerate one, and settles at the opening
print. `SettlementStyle` distinguishes the two styles and every underlying in
the contract table is `PMSettled`, so `:pre_open_expiry` is this rule's
own domain edge rather than a stand-in for the AM case; the
`:session_open` rule that would serve AM settlement is unwritten on
purpose, since there is no AM-settled underlying to test it against.

**The settlement instant is always the contract's expiry.** When the
reference price comes from an earlier session -- an unscheduled closure
-- only *which print stands in for the official close* moves; the
obligation still ceased to exist when it expired. So `effective_at` is
`lot.contract.expiry` and `recorded_at` is the tick that booked it, which
is the general bitemporal case and the sole source of the two replays
disagreeing at an intermediate instant.

**A lot is examined for settlement exactly once, ever.** `settlements`
takes the interval `(prev, t]`, not every open lot with `expiry <= t`: a
lot that cannot be settled stays open by design, and its answer is fixed
by the contract's expiry rather than by `t`, so a threshold would
re-derive the same failure at every later tick. The interval misses
nothing because the venue refuses a leg at or after its contract's
expiry, so a lot opened through this engine is in the book strictly
before its own expiry and therefore inside the interval that examines
it. A caller that writes through `record_order!` directly, bypassing
the venue, gets no such guarantee. That refusal is the
engine's, not the ledger's: the ledger accepts a fill effective at the
expiry instant, and without the stricter venue rule a lot opened at the
expiry tick would fall past every later interval, including the
window-end pass, and stay open with no `Expiry` and no warning. The
interval is open below, so a contract expiring exactly at the window
start is never examined; that becomes live only when a ledger is seeded
with open lots.

**An unsettleable lot stays open, loudly.** `settlement_price` throws
the named failure like every other named failure here; `settlements` is
the one boundary that catches it, warns once with the contract, its
expiry and the reason, and returns the lot paired with the failure in
`unsettled`. A bad day must not kill a ten-year run, but it must never
pass silently either, and reporting at the boundary rather than at the
call site is what makes that structural. `settlements` mutates no state;
it is not side-effect-free, and the distinction is deliberate.

**And the run keeps it.** A warning does not survive the run, so
`run_backtest` carries every entry out as a `RunFailure` on its result --
from **both** lifecycle passes, the window-end one included. It cannot be
recovered any other way: no event was written, because nothing happened,
and inventing an `Expiry` for a settlement that did not occur would put a
fiction in the journal of facts. The lot rides along with the failure
because the account has to say *which* lot went unanswered, and two lots
of one contract falling due together are two questions.

**An unserved settlement style stops the run.** `UnsupportedSettlement`
is deliberately *not* caught and warned that way. `UnpriceableLeg` names
one lot whose price is unavailable at this instant, and design rule 7
says leave it open and say so; `UnsupportedSettlement` names a contract
class the codebase cannot settle at all, which every later tick would
answer identically. Finishing the run would report a position that was
never valued as though it were merely still open. It is a configuration
error, and `load_experiment` throws the same type when it reads such a
config, so a reader who has met one has met the other.

Early assignment and physical delivery are not modelled: SPY
cash-settles at intrinsic here instead of delivering shares. Marking a
lot still open past the window end belongs to the marked curve
([`metrics`](metrics.md)), not to the lifecycle.




## Key decisions

| Decision | Why |
|---|---|
| **The engine computes, the ledger records** | `fill_legs` is a pure function of the cut and the order; `record_order!` mints every id and the group inside one transaction and constructs the record before committing events. The engine keeps no parallel journal; a live loop replaces `fill_legs` with the broker's reports without touching the writer. |
| **Every leg priced before anything is written** | A structure fills whole or not at all, as a guaranteed combo does at IBKR. A leg that cannot be priced is an error before the batch, so no partial structure ever reaches the ledger. |
| **Venue as two symbol tables and a tick, no type hierarchy** | Two plain symbols are what config and identity carry, and the tick is a constant; `Fill.fill_rule` already stores the key. A hierarchy -- or a `VenueSpec` struct -- would name the same things twice. |
| **R5: fill prices on the tick, rounded away from the trader** | The ledger refuses cash that is not whole cents; synthesized and modelled quotes are not on the tick; exchanges only trade on it. Rounding against the trader keeps the rule as conservative as crossing the spread already is. The observation keeps the raw quote; the fill carries the tick price. |
| **Observations recorded per leg, fills carry none** | Research records what pricing saw. The journal retains an observation row with optional quote sides for broker executions but ignores it during validation; truly observation-less live records arrive with the adapter. The join is validated, never assumed. |
| **Sessions come from the spot tree; the calendar only contradicts it** | A date is a session when the underlying printed in the reference window, so an early close needs no table: where the tree holds regular-session prints alone, the last print in the window is the 13:00 one. A calendar as the source would have to carry every half-day and every ad-hoc closure correctly forever; as the check it only has to answer whether a printless date was closed, and design rule 7 makes a wrong answer loud. The cost is that the correctness of an early close is the data's to keep -- an extended-hours print inside the window would settle the contract instead, undetectably -- an exposure this module states as its own, since `data` promises nothing about sessions. |
| **The settlement instant is always the contract's expiry** | When the reference price comes from an earlier session, the departure from reality is *which print stands in for the official close*, never *when the obligation ceased to exist*. The engine always passes the contract's expiry; the ledger's `_check_expiry` permits settlement at or after it, which is what makes an expiry booked at a later tick legal. The equality is this engine's choice, not the ledger's rule. |
| **The lifecycle computes, the ledger records** | `settlements` is a function of the cut and the book, `fill_legs`' twin; the writer is the ledger's `record_expiry!`. The engine gains no expiry queue, no cached calendar and no `try`/`catch` in the loop -- `prev` is a loop variable, not state. |
| **Settlement as a symbol through a table, no type hierarchy** | The same shape as `_FILL_RULES` and `_COST_MODELS`, and the same reason: the value config and identity will carry is a symbol, so a struct would name one thing twice. |
| **One `record_expiry!` per lot, never one batched `commit!`** | A structure's legs are jointly atomic, which is what `record_order!` is for; two lots expiring at one instant are independent facts. Batching would claim an atomicity that does not exist, and one unpriceable lot would reject the others. |
| **The interval `(prev, t]`, not `expiry <= t`** | An unsettleable lot stays open and its answer is fixed by its expiry, so a threshold would re-examine it at every later tick: thousands of identical warnings and calendar walks on a ten-year run. The venue's refusal to fill at or after expiry is what makes the interval miss nothing. |
| **The venue is stricter than the ledger about expiry** | The ledger accepts a fill effective at the expiry instant; `fill_legs` does not, because trading has stopped by then. It is also what the lifecycle interval rests on: lifecycle runs before the fill, so a lot opened at its own expiry tick would escape every later interval and stay open silently. |
| **The reference window ends at the earlier of 16:00 ET and the expiry** | The cut permits reading later prints -- it sits at or after the expiry -- but a settlement price from after the contract expired makes the event's own effective instant carry a number that did not exist then, and `book_effective` is the ledger's claim about what was true when. The bound costs nothing under the 16:00 ET convention and is the whole answer for an intraday expiry. |
| **`settlement_price` rejects a cut short of the expiry** | The function's name promises a settlement price; a cut that cannot see the session's close has only a provisional print to offer, and design rule 7 says such a question gets a name. Documenting the domain instead would leave an exported function handing a direct caller a confident wrong answer. |
| **A pre-open expiry names the contract, not the data** | A close-settled rule handed a contract that expires before its session opens has an empty window by construction, and calling that a data gap blames observations that could never exist. It is the AM-settled contract, whose rule (`:session_open`, the opening print) is not written yet; walking back to the previous session instead would settle it against the wrong session entirely. |
| **The warning lives in `settlements`, not at the call site** | If reporting were the caller's job, the window-end pass could forget it, and a silent gap is exactly the failure design rule 7 exists to prevent. |
| **The loop returns `(ledger, failures)`, not a new type** | An unsettled lot is a runtime observation with no event behind it, so it cannot be re-derived from what the loop wrote; it has to leave with the result. A named pair is what `settlements`, `session_closes` and `fill_legs` already return -- a struct here would name one ledger and one vector twice. |
| **`known_to` captured once per tick** | Sequence, not recorded time, bounds what a decision saw; the second order of a tick did not see the first's fills. |
| **Engine driven by `Agent`, not `Policy`** | Refits, swaps and learning live in the agent layer; one loop serves a fixed policy and a learning agent alike. The bare-`Policy` overload is ergonomics. |
| **No-lookahead at the type level, through derived data** | `current_policy` and `decide` take `TimeCut`; every read a derived provider makes on the policy's behalf goes through the cut. |
| **A declared clock** | The tick grid is part of the experiment; two experiments on the same data with different clocks are different experiments. |
| **`resolve_quote` reads quotes, not surfaces** | A surface retains only inverted IVs; the raw bid/ask the fill needs lives on the chain quote. |


## Conventions consulted

| Convention | Source | Consequence |
|---|---|---|
| A multi-leg option order fills in whole units or not at all | Interactive Brokers, [Understanding Guaranteed vs. Non-guaranteed Combination Orders](https://www.ibkrguides.com/kb/guaranteed-non-guaranteed-combo-orders.htm): "a guaranteed multi-leg order is one in which executions are guaranteed to be delivered simultaneously for each leg and in proportion to the leg ratio" | `record_order!` plus every leg priced first; a lone leg never happens |
| Commission per contract by premium tier, with a per-order minimum | Interactive Brokers, [US options commissions](https://www.interactivebrokers.com/en/pricing/commissions-options.php), IBKR Pro fixed, monthly volume ≤ 10,000, fetched 2026-09-12: USD 0.25 below a 0.05 premium, 0.50 from 0.05 to below 0.10, 0.65 at 0.10 and above, minimum USD 1.00 per order; the page's worked examples are the test literals | `:ibkr_pro_us_options`; one `Fee` per fill |
| Options on SPY, QQQ and IWM quote and trade in one-cent increments at every premium | MIAX, [Options Penny Program, all options exchanges](https://www.miaxglobal.com/markets/us-options/all-options-exchanges/penny-program), describing the industry-wide Penny Interval Program: penny classes trade in $0.01 below $3.00 and $0.05 at or above, but options overlying QQQ, SPY and IWM "are quoted and traded in minimum increments of $0.01 for all series regardless of the price" | `const TICK_CENTS = 1`, a constant rather than config, since every underlying the contract table lists is one of those three; a non-penny tick per class is a later model |
| An execution report carries a broker-assigned execution id, unique per report | FIX [ExecutionReport (35=8)](https://www.onixs.biz/fix-dictionary/4.4/msgtype_8_8.html), `ExecID` (tag 17) | one execution id per fill, minted by the writer here, reported by the broker live; a duplicate is refused |
| Backend selection by symbol through a dispatch table with defaults | Optim.jl, MLJ.jl; this repo's `_METRIC_TABLE` | `_FILL_RULES`, `_COST_MODELS`, `_SETTLEMENT_RULES` |
| An expiring in-the-money listed option is exercised without an instruction | OCC / The Options Industry Council, [Options exercise FAQ](https://www.optionseducation.org/referencelibrary/faq/options-exercise), checked 2026-09-13: "'Exercise by exception' is an administrative procedure used by OCC to expedite the exercise of expiring options by clearing members. In this procedure, OCC exercises options that are in-the-money by specified threshold amounts unless the clearing member submits instructions not to exercise" | an expiring lot settles at intrinsic against the settlement session's close, with no closing order; the OCC threshold itself is not modelled |
| The exchange's closed days and its 13:00 ET early closes are calendar facts, not data | NYSE, [Holidays & Trading Hours](https://www.nyse.com/markets/hours-calendars), checked 2026-09-13: the annual holiday list, and "Each market will close early at 1:00 p.m. (1:15 p.m. for eligible options)" on the named half-days | the calendar is the *check* on a printless date, never the source of sessions; early closes need no table, since the session's last print in the window is the 13:00 one wherever the tree holds regular-session prints alone -- a provider serving extended-hours prints defeats that, an exposure this module states as its own |
| An expiring listed option stops trading at the 16:00 ET close, and settles against the underlying's 16:00 ET close | Cboe, [Equity Options Extended Trading Hours FAQ](https://www.cboe.com/document/tech-spec/content/technical-specifications/equity-options-extended-trading-hours-faq/regular-trading-hours-vs.-globalcurb-trading-hours/), checked 2026-09-13: "Expiring equity single stock options will trade until 4:00 p.m. ET as part of RTH and 4:15 p.m. ET in the Curb session on expiration day", and "OCC also bases in/out-of-the-money determination based on the 4:00 p.m. ET closing price of the underlying equity security" | `fill_legs` refuses a leg at or after its contract's expiry (`:expired_contract`), and the settlement price is the underlying's session-close print. **Stated departure:** the 16:15 ET Curb session is not modelled, so this venue stops fifteen minutes before the real one does; a contract's expiry is stamped at 16:00 ET (`parse_polygon_ticker`), which is the RTH close and the instant OCC prices against |
| An exchange calendar as a library, not a hand-rolled table | [BusinessDays.jl](https://github.com/JuliaFinance/BusinessDays.jl) `USNYSE`, checked 2026-09-13 at v0.9.25: it carries the weekends, the ten annual holidays, both national days of mourning (2018-12-05, 2025-01-09) and the 2012 Hurricane Sandy closure | `isbday(USNYSE(), d)` is the whole calendar check; the ad-hoc `const` set beside it is **empty**, kept as the seam for a future closure the library will not have on the day |

