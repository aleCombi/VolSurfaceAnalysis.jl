# `surfaces` module

Vol surface representation and construction. Owns the math objects
(`ExpirySlice`, `VolatilitySurface`, `RawSurface`), BS pricing helpers,
IV inversion, and the `build_surface` function that turns an option
chain into a queryable surface.

A surface is a math object: it knows its spot, rate, div, timestamp,
and a sorted list of per-expiry IV slices. It knows nothing about
where its data came from -- builders are the only place this module
meets the raw `data` layer.

## Types

### `ExpirySlice`

One expiry's worth of inverted IVs. Strikes sorted ascending, one IV
per strike. `tau` (years to expiry, 365.25-day year) is cached at
build time so query code does not recompute it.

### `VolatilitySurface` (abstract)

Query API:

- `expiries(s) :: Vector{DateTime}`
- `get_slice(s, expiry) :: Union{ExpirySlice, Nothing}`
- `iv(s, expiry, strike) :: Float64`
- `price(s, expiry, strike, option_type) :: Float64`
- `delta(s, expiry, strike, option_type) :: Float64`
- `gamma(s, expiry, strike) :: Float64`
- `vega(s, expiry, strike) :: Float64`
- `forward(s, expiry) :: Float64`

Concrete subtypes implement `expiries`, `get_slice`, and expose
`spot`, `rate`, `div` fields. The rest are derived.

### `RawSurface <: VolatilitySurface`

Stores the slices directly (no parametric form). Fields: `underlying`,
`timestamp`, `spot`, `rate`, `div`, `slices` (sorted by expiry).

## Builder

```julia
build_surface(chain::Vector{OptionQuote}, spot::Float64,
              rate::Float64, div::Float64) :: RawSurface
```

1. Groups quotes by expiry.
2. Drops expiries with `tau <= 0` (already expired at the chain's
   timestamp).
3. Per expiry, picks the OTM-side quote at each strike (call when
   `K >= spot`, put otherwise); falls back to the ITM side if the
   OTM side has no usable mark.
4. Inverts BS for sigma from the picked mark. Strikes with no mark,
   zero/negative mark, or out-of-bracket prices are dropped.
5. Builds an `ExpirySlice` per surviving expiry. Throws on an empty
   input chain (programmer error). Returns `nothing` when no slices
   survive (e.g. every expiry has passed); callers treat that as
   "no surface at this timestamp."

Today this is the only builder method and handles only the
mark-price convention (`ParquetDataSource`'s output). The seam for
multi-vendor support is a future `QuoteConvention` trait on the
chain source.

## BS helpers (`src/surfaces/bs.jl`)

Self-contained Black-Scholes with continuous dividend yield. Inputs
throughout: spot `S`, strike `K`, time `T` (years), vol `sigma`,
rate `r`, dividend yield `q`. Public functions:

- `bs_price`, `bs_delta`, `bs_gamma`, `bs_vega`
- `implied_vol(price, S, K, T, option_type; r, q, lo, hi, tol, maxiter)`
- `time_to_expiry(expiry, now)` -- years, 365.25-day basis

The normal CDF is implemented inline (Abramowitz & Stegun 7.1.26,
~1.5e-7 max absolute error). Sufficient for round-trip IV inversion
to 1e-4. No SpecialFunctions / Distributions dependency.

## Query semantics

- `iv(s, expiry, strike)` linearly interpolates in log-moneyness
  (`log(strike / spot)`) within the matching slice; out-of-range
  strikes flat-extrapolate at the endpoint IV.
- Cross-expiry interpolation is not supported. `get_slice`,
  `iv`, `price`, `delta`, `gamma`, `vega`, and `forward` all error
  loudly when the expiry is not in the surface.
- Greeks and prices are computed from BS using the slice's `tau`
  and the surface's `spot`, `rate`, `div`.

## Responsibility boundaries

**Owns:** surface types, BS pricing math, IV inversion, the chain ->
surface builder.

**Does NOT own:**

- Raw chain access (that is `data`).
- Composition with rate/div curves at a specific `ts` (that is
  `model_data` -- it evaluates curves and hands the scalars here).
- Strategy logic, position pricing as a whole portfolio, P&L --
  downstream.

## Key decisions

| Decision | Why |
|---|---|
| **BS with continuous div yield, not Black-76** | Natural for equity options where we have `S` (spot), `r` (rate), `q` (div yield). The forward `F = S * exp((r-q)*T)` falls out, but we never need to manipulate it separately. |
| **Self-contained normal CDF, no SpecialFunctions dep** | Keeps the dep set minimal during the rebuild. A&S 7.1.26 (~1.5e-7 absolute error) is sufficient for IV inversion to 4 decimal places. Swap for `SpecialFunctions.erf` if higher precision is later required. |
| **Bisection for IV inversion** | Robust, no derivative needed, no failure-to-converge surprises. Brent/Newton are faster but the speed difference does not register in backtest cost; bisection is the safer default. |
| **OTM-side picking per strike** | OTM marks are more reliable (tighter spreads, more liquid in the wings). When OTM is missing we fall back to ITM rather than dropping the strike entirely. |
| **One IV per strike per expiry** | A slice models the strike dimension; carrying both call and put IVs would imply a put-call-parity inconsistency we are not yet calibrating against. v2 with implied-forward calibration can hold richer per-strike data. |
| **`RawSurface` only; no SVI/SABR yet** | Smallest honest representation. New parametric surfaces slot in as new `<: VolatilitySurface` subtypes whose `iv` reads from parameters; no consumer change. |
| **Linear interp in log-moneyness within a slice** | Cheap, monotone in strike order, naturally handles uneven strike spacing. Cross-expiry interp deferred until a consumer needs it. |
| **Strikes/expiries out of range flat-extrapolate / error respectively** | Strike interpolation has well-defined endpoints (IV at the wings); flat-extrap is the sensible default. Expiry queries are not interpolated in v1, so an out-of-range expiry is a bug, not a smoothing question -- throw. |
| **`time_to_expiry` uses 365.25-day year** | Matches the convention on master; standard in equity-options pricing. |
| **`build_surface` is a free function, not a `RawSurface` constructor** | Non-trivial work, returns an abstract type, will dispatch on a future `QuoteConvention` trait on the chain source. Concrete surface types still keep plain outer constructors. |

## Future work

- **`QuoteConvention` trait.** Once a second vendor lands (Deribit
  pre-computed IV, bid/ask-only sources), `build_surface` becomes
  `build_surface(quote_convention(chain_source), chain, spot, r, q)`
  and dispatches. Today's single method is the `MarkPriceQuotes`
  case.
- **Implied-forward calibration.** Per-snapshot forward calibration
  from put-call parity (see master's `recalibrate_iv`). Drops the
  residual put-call IV gap at short tenors. Adds a slice-level
  forward and uses it in pricing instead of `S * exp((r-q)*T)`.
- **Parametric surfaces.** SVI / SABR as additional
  `<: VolatilitySurface` subtypes, with their own builders that
  fit parameters from a raw slice.
- **Cross-expiry interpolation.** Linear-in-variance over time
  between adjacent slices, when a consumer needs it.
- **Theta / rho.** Add when a consumer needs them.
- **Higher-precision normal CDF.** Promote to `SpecialFunctions.erf`
  if a use case demands sub-1e-7 accuracy.
