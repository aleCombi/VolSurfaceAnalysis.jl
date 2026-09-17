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

A `Curve` is a function of time; subtypes carry the representation
(constant and piecewise-constant today). Evaluation outside the knots
flat-extrapolates by construction rather than by a branch, so a consumer
has no "off the end of the curve" case to handle.

A curve carries no timestamp and no selector of its own, which is why it
lives here rather than beside the records that stamp it, and why a
surface -- already one underlying at one instant -- is instead a kind in
its own right. [`data`](data.md) owns that rule.

## Surfaces

A surface is a sorted list of per-expiry slices plus the spot, rate, div
and timestamp they were built from. **Every query is answered from those
frozen values** -- prices, greeks and forwards come from the slice's
cached time-to-expiry and the surface's own spot, rate and div, never
from re-reading market data. That is what makes a surface an
observation, complete at one instant: two queries against one surface
cannot disagree, and no query can reach past the cut the surface was
built under.

Within a slice, strikes carry one IV each -- one, not one per option
type, because a call IV and a put IV at the same strike would encode a
put-call-parity inconsistency the build does not calibrate against.

The extension contract is deliberately narrow: a new representation
implements `expiries` and `get_slice` and exposes spot, rate and div, and
every price, greek, forward and delta inversion is derived from those
without a consumer change. `RawSurface`, which stores the slices
directly with no parametric form, is the only one today.

## Building a surface from a chain

The build is lossy by design, and rule 7 governs what it may say about
the loss. An expiry already past at the chain's timestamp is dropped, and
so is a strike whose mark will not invert; both are data, not errors.
*No* surviving slice is temporal absence -- the builder returns nothing
and `SurfaceFrom` turns that into an empty answer -- while an empty input
chain is a programmer error and throws. The two exits differ for that
reason alone.

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
| **Self-contained normal CDF, no SpecialFunctions dep** | Keeps the dep set minimal during the rebuild. A&S 7.1.26 (~1.5e-7 absolute error) is sufficient for IV inversion to 4 decimal places. Swap for `SpecialFunctions.erf` if higher precision is later required. |
| **Bisection for IV inversion** | Robust, no derivative needed, no failure-to-converge surprises. Brent/Newton are faster but the speed difference does not register in backtest cost; bisection is the safer default. |
| **OTM-side picking per strike** | OTM marks are more reliable (tighter spreads, more liquid in the wings). When OTM is missing we fall back to ITM rather than dropping the strike entirely. |
| **One IV per strike per expiry** | A slice models the strike dimension; carrying both call and put IVs would imply a put-call-parity inconsistency we are not yet calibrating against. |
| **`RawSurface` only, no parametric form** | Smallest honest representation, and the abstraction is the extension point: another representation implements `iv` from its own parameters and no consumer changes. |
| **Linear interp in log-moneyness within a slice** | Cheap, monotone in strike order, naturally handles uneven strike spacing. There is no cross-expiry interpolation: no consumer needs one. |
| **Strikes/expiries out of range flat-extrapolate / error respectively** | Strike interpolation has well-defined endpoints (IV at the wings); flat-extrap is the sensible default. Expiry queries are not interpolated in v1, so an out-of-range expiry is a bug, not a smoothing question -- throw. |
| **`time_to_expiry` uses 365.25-day year** | Matches the convention in the legacy codebase; standard in equity-options pricing. |
| **`build_surface` is a free function, not a `RawSurface` constructor** | Non-trivial work returning an abstract type, so a constructor would tie it to one concrete surface. Concrete surface types still keep plain outer constructors. |
| **`invert_delta` brackets on observed strikes, not on `spot * [lo, hi]`** | The slice already flat-extrapolates IV outside its observed strike range, so a wider bracket would land delta inversions in the extrapolation regime where the surface stops being informative. Capping the bracket at `[strikes[1], strikes[end]]` makes "no observed strike carries this delta" a `nothing` return rather than a fabricated answer. A representation with an explicit extrapolation policy can widen the bracket without changing the contract. |
| **Curve math sits here, not in `data/kinds` beside the records that carry it** | A `Curve` is no more a data concept than Black-Scholes is: it is arithmetic over values, with no timestamp, selector or provenance. Keeping it here makes the curve and the surface symmetric -- math in `pricing`, record in `data/kinds`, provider in `data/providers` -- where before, one object was cut by stage and the other by object (design rule 9). |

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
