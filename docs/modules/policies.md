# `policies` module

The engine walks the experiment's clock and calls `decide` on every
tick of it; `decide` returns the orders to send. Which orders depends
on the strategy, through dispatch on the policy type, and on three
inputs:

- the time,
- the market data, cut at that time,
- the current book.

A policy does not change between calls; when something has to change
over time, that is the [`agents`](agents.md) layer's job.

## Orders and the book

An order is composed of one or more legs. A leg names a contract, a
side, a number of contracts, and its intent, `Open` or `Close`; an
order with `Close` legs names the group it closes. A close is
therefore a first-class thing rather than a counter-trade, and closing
a group that is not open is refused, so a policy's mistake is a named
failure at fill time rather than a silent new lot. An order is booked
whole or not at all.

The book is the set of open lots, grouped by the order that opened
them, per contract, plus cash, as of this tick. Expired lots are
settled before `decide` runs, so an expired lot is gone from the book.
A lot whose settlement price could not be found stays open and
visible, and a policy recognises it by `lot.contract.expiry <= t`.

`decide` receives market data already cut at `t`, so nothing it reads,
directly or through a derived provider, can be later than `t`.

`decide` is stateless. The struct holds configuration; anything the
recurrence might want is either derivable from the three inputs plus
that configuration, or it belongs to an agent that hands out a fresh
policy when state advances.

## Scheduling

A scheduled policy checks the time inside `decide` and returns no
orders off-schedule. It may also hand the engine its own schedule
through `tick_times`; the engine then walks that schedule instead of
the clock, and the whole tick, settlement and fills included, runs at
those times only. The window end is a clock property, so a schedule
cannot move the final settlement.

`declared_underlyings` lets an experiment refuse, before it runs, a
policy whose declared underlying differs from the one its clock steps
on. A policy that declares none is not checked.

## Decisions

| Decision | Why |
|---|---|
| **Orders, not a target portfolio** | Unchanged positions need no restating, no-op is `Order[]`, and open-versus-close bookkeeping stays in the ledger. |
| **A `Close` leg names its group** | A counter-trade left the engine to guess intent from direction, and it guessed wrong at a side flip. A position effect is what a broker ticket says. |
| **The book, not the fill log** | The view by replay is what a live loop hands a policy too; no netting of a fill vector. |
| **Stateless `decide`** | No setup to test, deterministic replay, no question of mutate versus rebuild between ticks. |
| **`decide` receives data already cut at `t`** | The old code enforced no-lookahead with a view handed in at runtime, which a caller could skip; a cut the type carries cannot be skipped. |
| **`t` explicit** | A schedule asks "is this my entry time" without digging through timestamps. |
| **Check the time in `decide`, narrow the calls with `tick_times`** | The check keeps the policy correct on any clock; the schedule saves calls. |
