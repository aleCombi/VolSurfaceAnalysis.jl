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
side, a number of contracts, and its intent: `Open`, or `Close` with
the group it closes. A close is therefore a first-class thing rather
than a counter-trade, and closing a group that is not open is refused,
so a policy's mistake is a named failure at fill time rather than a
silent new lot. An order is booked whole or not at all.

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

A scheduled policy gates inside `decide`, which is correct on any
clock, and may narrow the engine's calls with `tick_times`. The window
end stays a clock property, so a schedule can never move the
settlement.

`declared_underlyings` exists so the loader can hold an experiment to
one underlying: the clock selector says *when* to step, fills price per
leg, and their agreement is what keeps that safe. A policy that chooses
its underlying per tick declares nothing and is not checked.

## Decisions

| Decision | Why |
|---|---|
| **Orders, not a target portfolio** | Unchanged positions need no restating, no-op is `Order[]`, and open-versus-close bookkeeping stays in the ledger. |
| **A `Close` leg names its group** | A counter-trade left the engine to guess intent from direction, and it guessed wrong at a side flip. A position effect is what a broker ticket says. |
| **The book, not the fill log** | The view by replay is what a live loop hands a policy too; no netting of a fill vector. |
| **Stateless `decide`** | No setup to test, deterministic replay, no question of mutate versus rebuild between ticks. |
| **`decide` receives data already cut at `t`** | The old code enforced no-lookahead with a view handed in at runtime, which a caller could skip; a cut the type carries cannot be skipped. |
| **`t` explicit** | A schedule asks "is this my entry time" without digging through timestamps, and the engine gets a trivial crosscheck against the cutoff. |
| **Gate in `decide`, narrow with `tick_times`** | The gate keeps the policy correct on any clock; the schedule only saves calls. |
