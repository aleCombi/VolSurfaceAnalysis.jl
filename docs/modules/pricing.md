# `pricing` module

Valuation math: curves, vol surfaces, Black-Scholes, implied-vol
inversion, and the chain -> surface build.

Nothing here reaches the data protocol, a provider, or a cut (design
rule 9). `pricing` may name a record type (`build_surface` consumes a
chain) but never asks for one. A surface comes from `SurfaceFrom` in
[`data`](data.md), the only caller of `build_surface`. That a surface
doubles as its own record is `data`'s business; nothing here depends on
it.

## Curves

A `Curve` is a function of time. It flat-extrapolates outside its
knots, so a consumer has no off-the-end case. Curve math sits here and
not beside the records that stamp it because a curve is arithmetic
over values, with no timestamp, selector or provenance; that keeps
curve and surface cut the same way, math here, record and provider in
`data`.

## Surfaces

A surface is frozen at one instant. It carries the spot, rate, div and
timestamp it was built from, and **every query is answered from those
values**, never by re-reading market data. So two queries against one
surface cannot disagree, and none can reach past the cut it was built
under.

One IV per strike, not one per option type: a call IV and a put IV at
one strike would encode a put-call-parity inconsistency the build does
not calibrate against. IV is linear in log-moneyness between strikes
and flat outside them. There is no cross-expiry interpolation; a query
names a quoted expiry and an unquoted one is an error.

**Linear in log-moneyness is monotone in strike order, and
`invert_delta`'s bisection depends on it.** A smoother smile must
re-establish that or change the search; the failure otherwise is a
plausible wrong strike after the iteration bound, not an error.

`RawSurface` is the only representation. The abstract type is where a
second one would attach; `invert_delta` is the one query written
against it.

## Building a surface from a chain

The build is lossy, and rule 7 governs the loss. An expiry at or before
the chain's timestamp is dropped; so is a strike whose mark will not
invert. No surviving slice is temporal absence (the builder returns
`nothing`, `SurfaceFrom` answers empty); an empty input chain is a
programmer error and throws.

Per strike the OTM-side mark is inverted, because OTM marks are the
more reliable; the ITM side is the fallback, not dropping the strike.

One quote convention, the mark price `QuotesFromBars` produces. Nothing
dispatches on convention; a feed quoting pre-computed IVs or bid/ask
without a mark is not served (parked in [status.md](../status.md)).

## Boundaries

**Owns** representation, BS math, IV inversion, the build.
**Does not own** chain access or providers ([`data`](data.md)); the
kind contract of anything defined here (`data/kinds`); strategy,
portfolio pricing, P&L (downstream).

## Decisions

| Decision | Why |
|---|---|
| **BS with continuous div yield, not Black-76** | Equity options give `S`, `r`, `q` directly; the forward falls out and is never manipulated. |
| **Self-contained normal CDF** | No SpecialFunctions dependency. About 1.5e-7 absolute error in probability against an IV round-trip the tests pin to 1e-4 in vol. |
| **Bisection for IV and delta** | No derivative, no convergence surprises. Brent/Newton's speed does not register in backtest cost. |
| **A fixed day count, not a calendar** | The legacy convention. A calendar would change every IV and greek for no consumer that asked. |
| **`build_surface` is a free function** | It returns `nothing` when nothing survives; a constructor cannot decline. |
| **`invert_delta` brackets on observed strikes** | Beyond them IV is flat and uninformative. "No observed strike carries this delta" is a `nothing`, not a fabricated answer. |

## Conventions consulted

One entry per naming decision (design rule 5).

- **`pricing`, holding curves beside surfaces.** QuantLib keeps term
  structures (yield curves and vol surfaces together) and pricing
  engines in one library split by role, not instrument. `math` was
  rejected: it names the contents, not a stage of a run.
