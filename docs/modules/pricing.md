# `pricing` module

Valuation math: what an instrument is worth, and how a surface is built
out of an option chain. Curves, vol surfaces, Black-Scholes, implied-vol
inversion.

Nothing here knows the data protocol (design rule 9). A curve and a
surface are math objects, not records: they are built from values handed
to them and answer by arithmetic. `pricing` may *name* a record type --
`build_surface` consumes a chain -- but it never asks a map or a provider
for one and never sees a time cut. Where a surface *comes from* is
`SurfaceFrom`, a provider in [`data`](data.md), which is also the only
caller of `build_surface`.

## Curves

A `Curve` is a function of time; subtypes carry the representation.
Evaluation outside the knots flat-extrapolates, so a consumer has no
"off the end of the curve" case to handle.

The curve's math sits here and not beside the records that carry it,
because a `Curve` is no more a data concept than Black-Scholes is: it is
arithmetic over values, with no timestamp, selector or provenance. That
keeps curve and surface symmetric -- math in `pricing`, record in
`data/kinds`, provider in `data/providers` -- where before, one object
was cut by stage and the other by object. [`data`](data.md) owns the rule
for which of the two needs a record wrapped around it.

## Surfaces

A surface is frozen at one instant: it carries the spot, rate, div and
timestamp it was built from, and **every query is answered from those
values** -- prices, greeks and forwards come from the slice's cached
time-to-expiry and the surface's own spot, rate and div, never from
re-reading market data. That is what makes a surface an observation,
complete at one instant: two queries against one surface cannot
disagree, and no query can reach past the cut the surface was built
under.

Within a slice, strikes carry one IV each -- one, not one per option
type, because a call IV and a put IV at the same strike would encode a
put-call-parity inconsistency the build does not calibrate against.
Between strikes the IV is interpolated linearly in log-moneyness, and
outside the observed strikes it is flat; there is no cross-expiry
interpolation, because no consumer needs one, so a query names a quoted
expiry exactly and an unquoted one is an error rather than a smoothing
question.

`RawSurface`, which stores the slices directly with no parametric form,
is the only representation: the smallest honest one. The abstract type
is the seam a second representation would attach at, and `invert_delta`
is the one query already written against the seam rather than the
representation.

## Building a surface from a chain

The build is lossy by design, and rule 7 governs what it may say about
the loss. An expiry already past at the chain's timestamp is dropped, and
so is a strike whose mark will not invert; both are data, not errors.
*No* surviving slice is temporal absence -- the builder returns nothing
and `SurfaceFrom` turns that into an empty answer -- while an empty input
chain is a programmer error and throws. The two exits differ for that
reason alone.

Per strike, the IV is inverted from the OTM-side mark, because OTM marks
are the more reliable (tighter spreads, more liquid in the wings); when
the OTM side is missing the ITM side is used rather than the strike
dropped.

One quote convention is supported: the mark-price one that
`QuotesFromBars` produces. Nothing dispatches on convention, so a feed
that prices its chain another way -- pre-computed IVs, bid/ask with no
mark -- is not served by this build.

## Responsibility boundaries

**Owns:** curve and surface representation, BS pricing math, IV
inversion, the chain -> surface build.

**Does NOT own:**

- Raw chain access, storage, or any provider: that is [`data`](data.md).
- The kind contract (`selector`, `snapshot`) of anything defined here,
  which lives in `data/kinds` with every other kind's.
- Strategy logic, whole-portfolio pricing, P&L -- downstream.

## Key decisions

| Decision | Why |
|---|---|
| **BS with continuous div yield, not Black-76** | Natural for equity options where we have `S` (spot), `r` (rate), `q` (div yield). The forward `F = S * exp((r-q)*T)` falls out, but we never need to manipulate it separately. |
| **Self-contained normal CDF, no SpecialFunctions dep** | Keeps the dep set minimal during the rebuild. The polynomial approximation is accurate well past what IV inversion needs; swap for `SpecialFunctions.erf` if higher precision is ever required. |
| **Bisection for IV and delta inversion** | Robust, no derivative needed, no failure-to-converge surprises. Brent/Newton are faster but the speed difference does not register in backtest cost; bisection is the safer default. |
| **A fixed day count, not a calendar** | `time_to_expiry` divides by a constant year length, the legacy codebase's and the usual equity-options convention. A calendar would change every IV and greek for no consumer that asked. |
| **`build_surface` is a free function, not a `RawSurface` constructor** | Non-trivial work returning an abstract type, so a constructor would tie it to one concrete surface. Concrete surface types still keep plain outer constructors. |
| **`invert_delta` brackets on observed strikes, not on `spot * [lo, hi]`** | The slice flat-extrapolates IV outside its observed strikes, so a wider bracket would land inversions where the surface stops being informative. Capping the bracket at the observed range makes "no observed strike carries this delta" a `nothing` return rather than a fabricated answer. A representation with an explicit extrapolation policy can widen the bracket without changing the contract. |

## Conventions consulted

One entry per naming decision, with the source checked (design rule 5).

- **Named for the stage, and holding curves beside surfaces.** QuantLib
  keeps term structures (`ql/termstructures`, where yield curves and
  vol surfaces sit together) and pricing engines (`ql/pricingengines`)
  inside one library, split by role rather than by instrument -- so this
  module takes both the role vocabulary for its name and the arrangement
  that puts a curve and a surface in one folder. `math` was considered
  and rejected: it names a property of the contents rather than a stage
  of a run.
