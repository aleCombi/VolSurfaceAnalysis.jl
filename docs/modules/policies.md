# `policies` module

A policy is a pure decision function. The engine calls `decide` once
per tick with `(t, cut, book)` and books what comes back; the policy
returns orders, not a portfolio, and is frozen for the tick it was
handed out on. Anything that changes between ticks -- a refit, a swap,
a schedule that advances -- is the [`agents`](agents.md) layer's.

## The decision

Intent is declared per leg. A close is a `Close` leg in an order naming
the group it closes, never a counter-trade, and it is refused when there
is nothing to close, so a policy's mistake is a named failure at fill
time rather than a silent new lot. The engine books an order whole or
not at all.

The book is what a policy sees, not the fill log: open lots per group
and contract, plus cash, as known at this tick. Lifecycle is booked
before the decision, so a settled lot is already gone; a lot whose
settlement could not be resolved stays open and visible, and
`lot.contract.expiry <= t` is how a policy tells it is holding one,
inclusive at `t` because the settlement interval is.

No-lookahead is a type, not a convention: `decide` takes a
[`TimeCut`](data.md), so every read, including those a derived
provider makes on the policy's behalf, is cut at `t`.

`decide` is stateless. The struct holds configuration; anything the
recurrence might want is either derivable from `(t, data, book)` plus
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

## `DailyShortStrangle`

Once a day at `entry_time`, a short OTM put and a short OTM call picked
by target |delta| through [`invert_delta`](pricing.md), as one order.
It only opens; lifecycle settles its lots.

- **Two legs, one order.** Filled whole or not at all, sampled as one
  structure.
- **One wing failing skips the entry.** A one-legged strangle is a
  different structure, and trading the surviving leg would corrupt the
  backtest silently.
- **Snap to the chain, not the slice.** `invert_delta` returns a
  continuous strike; a fill needs an exact match on strike *and* type,
  and a slice keeps one side per strike, so the strikes quoted for the
  leg's type are the only honest targets.

## Boundaries

**Owns** `Policy`, `decide`, `declared_underlyings`, `tick_times`,
`NoOpPolicy`, `DailyShortStrangle`.
**Does not own** the tick loop and fills ([`backtest`](backtest.md));
change over time ([`agents`](agents.md)); lot lifecycle -- there is no
`close!`, and expiries are booked by the engine; P&L (downstream).

## Decisions

| Decision | Why |
|---|---|
| **Orders, not a target portfolio** | Unchanged positions need no restating, no-op is `Order[]`, and open-versus-close bookkeeping stays in the ledger. |
| **A `Close` leg names its group** | A counter-trade left the engine to guess intent from direction, and it guessed wrong at a side flip. A position effect is what a broker ticket says. |
| **The book, not the fill log** | The view by replay is what a live loop hands a policy too; no netting of a fill vector. |
| **Stateless `decide`** | No setup to test, deterministic replay, no question of mutate versus rebuild between ticks. |
| **`TimeCut` in the signature** | The legacy code enforced no-lookahead with a view passed at runtime; the type makes it unbypassable. |
| **`t` explicit** | A schedule asks "is this my entry time" without digging through timestamps, and the engine gets a trivial crosscheck against the cutoff. |
| **Gate in `decide`, narrow with `tick_times`** | The gate keeps the policy correct on any clock; the schedule only saves calls. |
