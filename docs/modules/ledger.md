# `ledger` module: the journal of economic facts

Defines *what happened* to a portfolio: an append-only journal of typed
events, the book that is replayed from it, and the round trips derived
from it. The engine is the journal's only writer (it switches over in
slice 2 of the [ledger proposal](../proposals/ledger.md)); policies
read the book; metrics read the derived tables. The module knows only
the identity vocabulary of [`data`](data.md) (`Underlying`,
`OptionType`) and the contract facts it defines itself -- no quotes,
spots or time cut -- so a ledger is built, checked and replayed from
its own events alone. It lands beside `positions`, which the engine
keeps using until it switches.

## The kinds it defines

- `Order` -- one structure-level instruction out of `decide`: a label,
  its legs (`Leg`: a `ContractKey`, a side, an integer number of
  contracts, a declared `Open` / `Close` intent), the group it opens
  into or closes, and an operation id linking the two orders of a roll.
  Intent is declared, never inferred from direction.
- `Fill`, `Match`, `Expiry`, `Fee` -- the four event kinds, a closed
  union (`LedgerEvent`). A fill is one execution and carries nothing
  about the market it was filled against (that belongs to the order
  journal, outside this module); a match names the lot one closing fill
  consumed; an expiry settles one remaining lot; a fee names the fill
  that caused it. Every event carries an `EventHeader` by composition:
  a stable id (a reference, never an index), effective time (when the
  fact is true), recorded time (when the ledger learned it) and
  sequence (replay order). A group sits on `Fill`, `Match` and `Expiry`;
  a `Fee` has none.
- `Ledger` -- the container; `Book` and `Lot` -- the view by replay;
  `RoundTrip` -- one consumed lot.
- `ContractSpec` -- per-underlying facts: multiplier, exercise style,
  settlement style, delivery. A table in code keyed by ticker, from the
  OCC product specifications; an unlisted underlying is
  `UnknownContract`, never a default.

## Cash rules

Cash is a method on an event, never a stored field, so inputs and cash
cannot disagree. Quantities are integer contracts, prices are per
share, cash is USD through the contract multiplier:

| Event | Cash |
|---|---|
| `Fill` | `-side * price * quantity * multiplier`: a buy pays, a sale receives |
| `Match` | none |
| `Expiry` | `side * intrinsic(contract, settlement price) * quantity * multiplier` |
| `Fee` | its signed amount; a cost is negative |

`Expiry` copies side and contract from its opening fill, checked on
append, so every event's cash is local to the event. A round trip's
PnL is its opening and closing cash plus the pro rata share, by
quantity, of any fee on either fill; once nothing is left open, the
round trips sum to the book's cash.

## Book and the two replays

The book holds lots per `(group, contract)`, FIFO within, plus cash. It
is never stored: it is the fold of the events, and the incrementally
updated book always equals the full replay.

Two replays answer two questions. *What was known* cuts by sequence,
everything appended up to a boundary: the view a decision could have
seen. Recorded time is not a safe cut for it, because fills booked at
the same tick after the decision share its recorded time. *What was
true* at an instant cuts by effective time and folds in (effective
time, priority, sequence) order, expiries first at an equal instant.
The two differ only by expiries booked at the tick after their instant:
an expiry booked on Monday's first tick is already true at Friday's
settlement.

## Invariants

Every write goes through one validated path. A batch is checked whole
against the ledger and the book; nothing is appended if any check
fails, and each failure has a name:

- sequence and id continue the ledger's counters -- `SequenceGap`;
- a fill is effective at or before its contract's expiry --
  `FillAfterExpiry`;
- every reference points backward to an event of the right kind --
  `DanglingReference`;
- the matches of a closing fill immediately follow it, pair lots of its
  own group and contract on the opposite side, and exhaust it exactly;
  an expiry's copied fields equal its opening fill's -- `MatchMismatch`;
- consumption never exceeds a lot's remaining -- `ExceedsOpen`; a close
  with nothing to close -- `NothingToClose`;
- quantities are positive integers -- `NonPositiveQuantity`, thrown at
  construction so no such value exists.

A close matches only within its own group and contract, FIFO among the
open lots of the opposite side, so two groups on one contract never
touch each other's lots. Effective time need not be monotone in
sequence; ids and sequence are separate counters that coincide in a
fresh ledger and are never used for each other.

## Responsibility boundaries

**Owns:** the order and event vocabulary, the contract table, the cash
rules, the validated write path, the book and its replays, round trips,
and the `pnl_series(ledger)` adapter in `metrics` that lets today's
metrics read a ledger unchanged (`unit = :structure` samples per group
and closing instant, `unit = :leg` per round trip).

**Does not own:** deciding what to trade (policies); resolving quotes,
turning them into prices, and the order journal that records what a
decision saw (the engine and its execution model); when and at what
price a lot settles (the lifecycle model); marks, the equity curve and
valuation failures (outside the journal); persistence. Those arrive in
the later slices of the proposal.

## Key decisions

| Decision | Why |
|---|---|
| **Append-only journal; the book is a view** | Corrections are new entries, never edits, so a stored run can be replayed and audited; a backtest policy and a live loop receive the same `Book` type. |
| **`Match` is an event** | One close can split across lots, and a change to the matching rule must not rewrite old results; the pairing is recorded, not recomputed. |
| **`Expiry` is per lot and carries its side and contract** | Lineage survives mixed expiries inside one structure, and the event's cash needs no lookup. The copy is checked on append. |
| **Bitemporal header, decision view cut by sequence** | Effective and recorded time separate what was true from what was known; sequence, not recorded time, bounds what a decision could see. |
| **Closed union container** | Serialisation can be exhaustive; at thousands of events performance is a wash either way. |
| **Contract facts are a table in code** | They are facts with no visibility time and no observer; wrong is wrong, not a variant. Resolved values project into identity later, so a correction is a new run id. |
| **Integer quantities, per-share prices, USD cash** | Removes the FIFO float residue and the per-share-labelled-USD units of the fill-vector ledger. |
| **Validation of the whole batch at the write** | A structure is either booked whole or not at all; a leg that fails validation is an error before anything is written. |
| **`ContractKey` hashes by content** | The book keys lots on `(group, contract)`; the default `objectid` hash is build-dependent, as `Underlying` documents. |
| **The adapter keeps `PnLSeries` unchanged** | Metrics stay green while the engine switches; `window_end_spot` and `n_unmarked` are placeholders that leave with the slice-5 metrics. |

## Conventions consulted

| Convention | Source | Consequence |
|---|---|---|
| Append-only journal; corrections are new entries | Double-entry practice | events immutable, never edited |
| Orders declare intent; a close with nothing to close is rejected | US broker tickets; OCC position reporting | `Leg.intent`; `NothingToClose` |
| Positions are the net of fills; lot pairing is recorded under a named rule | Broker statements; IRS FIFO default | `Book` by replay; `Match` event, FIFO within group and contract |
| An execution report carries ids, quantity, price and time, not the quote the client saw | Broker execution reports | `Fill` is execution only; the order journal holds the quote |
| Prices per share, cash per contract times 100; style, settlement and delivery are listed per product | OCC contract specifications | `ContractSpec` table in code, per underlying |
| Exercise by exception at expiry; PM settlement against the official close | OCC Rule 805; Cboe procedures | `Expiry` outcome; the lifecycle model names where it departs |
| Realised and unrealised are separate lines | Fund reporting | `round_trips` now; the equity curve later |
| Composition plus accessor methods, not inherited fields | Julia manual, Interfaces | shared `EventHeader`; `header`, `event_id`, `effective_at`, `recorded_at`, `sequence`, `group` |
| Avoid abstract-element containers; small closed unions are the idiom | Julia manual, Performance Tips | `Vector{LedgerEvent}` over a closed union |
| Content-based `hash` and `==` for value types used as dictionary keys | Julia manual, `Base.hash` docstring; `data.md` on `Underlying` | `ContractKey` |

## Layout

```
src/ledger/
    contracts.jl    # ContractSpec, the table, UnknownContract
    types.jl        # Side, Intent, ContractKey, Leg, Order, EventHeader,
                    # Fill, Match, Expiry, Fee, LedgerEvent, Ledger, accessors
    cash.jl         # intrinsic, cash
    book.jl         # Lot, Book, apply!, book_as_known, book_effective
    append.jl       # named failures; mint_group!, record_*!, commit!
    round_trips.jl  # RoundTrip, round_trips
src/metrics/ledger_series.jl   # pnl_series(::Ledger)
test/ledger/                   # hand-built ledgers, hand-computed answers
```

Files are `include`d into the top-level `VolSurfaceAnalysis` module;
no submodule wrappers. Metrics depend on the ledger, never the reverse.
