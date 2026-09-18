# `metrics` module

Pure functions over **two** honest inputs: a [`MarkedCurve`](#the-marked-curve)
for questions about time, and a plain vector of per-trade dollars for
questions about trades. No `Metric` abstract type, no registry trait --
metrics are ordinary functions, and [`compute_metrics`](@ref) dispatches a
small symbol table of optional ones on top of a fixed always-on core set.

## Two units, because there are two questions

A metric's answer is only as honest as its sample unit.

- **Trade questions** -- total realised profit, hit rate, profit factor,
  how many round trips, opens and closes -- have a trade as their unit.
  They read `trade_pnl(::Ledger)`, one dollar figure per closed structure.
- **Path questions** -- Sharpe, Sortino, volatility, maximum drawdown --
  have a *period of time* as their unit. They read the marked curve, whose
  points are session closes, and annualise by sessions per year.



## The marked curve


### The accounting identity

At any instant,

> marked profit = realised profit + unallocated fees + unrealised profit

The ledger alone supplies the first two terms; only the third needs market
data, and it is zero whenever the book is flat. The builder uses the
equivalent form that falls out of the cash rules:

> marked profit = ledger cash + the open book marked to market

because cash already carries the cost basis of every open lot with the
opposite sign. This is why **cash alone is not the curve**: selling a
strangle credits premium at open, and sampling cash would report that
receipt as profit and materially overstate a short book.

Profit is measured from zero. The codebase has no deposited account and no
net asset value, and none is invented here.

### The grid

The close of each trading session in the evaluation window, derived rather
than configured -- there is no `mark_grid` setting. It comes from
`session_closes` in the [backtest](backtest.md) module, the same session
machinery `:session_close` settles against: a calendar-open date is a
session when the underlying printed in the 09:30-16:00 ET window, and its
close is the last of those prints. Early closes and unscheduled closures
therefore need no table, under the same session-window exposure
[`backtest`](backtest.md) states -- and the grid reads exactly those
windows, one at a time, so it is exposed to the tree only where
settlement is.

A session counts only when its **whole** window lies inside the evaluation
bounds. A session the bounds clip is not a short session but one this run
did not see end to end, and every observation must span a whole one for
annualising by sessions to mean anything. That absence is temporal: the
session is outside the window, not unanswerable inside it.

### Marking one lot

`mark_price(cut, contract, t)` is the contract's own quote mid
`(bid + ask) / 2`, with the surface price as fallback, and
`UnpriceableLeg(:no_mark)` after that. The mid rather than the last trade:
a trade may be hours stale at a quiet strike, while a two-sided quote is a
price someone stands behind at `t`. A one-sided quote is not a mid and
falls through to the surface exactly as an absent one does.

Marking runs inside `run_experiment`, while the cut is open, and reads
through a `TimeCut` at each point, so no-lookahead is structural here as in
the tick loop. It is intentionally *not* a pure function of the ledger.

### What an unmarkable point does

**One lot without an honest price makes the whole session unanswerable, so
the point leaves the curve, is named and counted, and
`session_changes` refuses to span it.** No partial sum, no carried-forward
value, no `NaN`, and the curve is not truncated (design rule 7). Three
consequences worth stating plainly:

- The session appears in `unmarked_at` / `unmarked_reason` and nowhere
  else. `n_unmarked` is how many.
- **The builder still examines the rest of the session's lots**, and
  retains a `RunFailure` for each one it cannot price. The running total
  is discarded: carrying on is for the account of what went unanswered,
  never for a partial portfolio value. So a broken session is one curve
  entry and as many failure records as it had failed lots, and the two
  agree instant for instant. A printless session names its underlying
  instead of a lot. `marked_curve` returns both, and the run carries them
  (see [experiment](experiment.md)); only `UnpriceableLeg` is caught, so
  an unexpected error still propagates rather than becoming an empty
  output under a successful-looking run.
- A break costs **two** observations, not one: neither the step into the
  broken session nor the step out of it is a period this run observed. A
  step spanning it would cover two periods while being scaled as one.
- `max_drawdown` still walks the marked levels across a break, because
  profit is measured from zero rather than reset per segment. Its answer
  over a curve with unmarked sessions is a lower bound on the true depth,
  and `n_unmarked` says how much it could not see.

**When to revisit this.** Breaking is right for a backtest, which has no
obligation to publish a number; it is not what an accounting system does.
A fund descends a fair-value hierarchy (quoted price, then observable
model inputs, then judgment) precisely because it must report daily, and
listed options have a published end-of-day settlement price for every
series so the question rarely arises at all. The trigger for rethinking
is therefore volume, not principle: while `n_unmarked` stays near zero
the choice costs nothing, and if a run starts losing many series the
answer is to collect official per-series marks rather than to start
inventing values. Accepted on those terms 2026-09-14.

Reasons: `:no_mark` (neither a quote mid nor a surface price for some open
lot) and `:unexpected_gap` (a calendar-open date whose window holds no
print, stamped at that session's nominal 16:00 ET close -- a label on a
failure, never a value). `marked_curve` warns once with the count, the same
way `settlements` reports an unpriceable lot: the boundary that finds a gap
is the boundary that reports it.

### Capital

Capital is fixed at 1, and that is exact rather than an approximation. At a
zero risk-free rate a constant capital base scales both the mean and the
standard deviation of the period profits equally, so it cancels from every
ratio here: Sharpe on dollar changes is Sharpe on returns for capital 1,
100,000 or any other positive constant. It is therefore **not an argument**
-- a kwarg that cannot change a result would only be a contract to
maintain. `test_optional.jl` pins the cancellation by scaling a curve
instead.



## Trade-level input


One entry per `(group, closed_at)` by default -- a strangle closed at one
instant is one trade -- summed in whole cents and converted once;
`unit = :leg` gives one entry per round trip. The pairing is the ledger's
own (`Match` events under a named rule), never inferred here. Entries are
ordered by `(closed_at, pnl)`, losses first within an instant, so a run's
trade vector is deterministic and reconstructible.

## The cents boundary

`cents_to_usd(cents::Integer) -> Float64` is the one place the ledger's
whole USD cents become floating-point dollars, as explicitly as
`contract_cents` names the rounding boundary in the other direction. Every
dollar figure this module reports crosses there.

Marks do not cross it: a quote mid or a surface price is floating point at
source and is never whole cents (the mid of 1.05/1.06 is 1.055), so the
unrealised term of a marked profit is float arithmetic from the start. The
realised term stays integer until the boundary.




**The two inputs.** The ledger is the authority for the trade side:
`trade_pnl(L)` is derived once inside, and the two fill counts are read
straight off it. The curve is passed in because it is the one result that
is *not* a function of the ledger -- whoever has the market data open
builds it.

Each dispatch-table entry is `(fn=..., defaults=(...))`. The dispatcher
hands both inputs to every entry, so the table records no per-metric
input and adding a metric is one row and one function. The cost is an
ignored argument in each body, and one consequence worth knowing: the
dispatcher cannot tell a session metric from a trade metric, so when
there is no curve it omits the **whole** optional set rather than part
of it. `:profit_factor` needs no curve and is dropped with the rest.

`kwargs` is a per-metric override map. Each entry merges with the default
kwargs baked into the metric's dispatch-table entry, so a partial override
(`Dict(:sharpe => (periods_per_year=12,))`) keeps the rest of that metric's
defaults. Unknown symbols error loudly with the list of known names.

**`curve === nothing`** is that dependence made visible: the always-on core
is computed as always, and **every** optional metric is omitted from the
result rather than reported as `NaN`. An absent key says "not computed";
`NaN` would say "computed, and undefined", a different and false claim. The
omission is wholesale rather than selective because every metric takes both
inputs, so the table does not record which one each reads: `:profit_factor`
goes with the others though it needs no curve. It is a run-time path: a
run whose market data cannot be opened has no curve. `load_run` does not
take it, because loading reads the metrics the run reported and computes
none.

## Key decisions

| Decision | Why |
|---|---|
| **Two inputs, not one series** | A metric's sample unit is the whole of its honesty. Trades and sessions are different units answering different questions, and one intermediate serving both is what let Sharpe annualise trade counts as though they were days. |
| **Pure functions, no `Metric` trait** | Metrics are just functions. A registry / abstract `Metric` type would add ceremony without buying polymorphism we need; the symbol table gives "select-by-name" without a type hierarchy. |
| **Uniform arity: every metric takes both inputs** | Adding a metric is one table row and one function, with no second place to keep in sync. The price is an ignored argument per body and a wholesale rather than selective omission when there is no curve; the sample unit lives in the docstring and the tests instead of in a column. |
| **Marked profit from zero, capital fixed at 1** | There is no deposited account or NAV to divide by, and at a zero rate the base cancels from every ratio. Inventing a denominator would be a number with no source; a powerless config value would be a contract to maintain. |
| **Cash is not the curve** | Premium received on a short option is a receipt against a liability. A curve sampling cash would book the opening credit as profit. |
| **The grid is derived from sessions, not configured** | Daily observations are what annualisation by trading sessions means. A configurable grid would let a run pick a cadence its scaling constant does not match. |
| **A quote mid, surface as fallback, then a named failure** | A last trade may be stale for hours at a quiet strike. Two named sources and then an error is design rule 7 applied to a leg that cannot honestly be priced. |
| **An unmarkable point breaks the curve rather than carrying or truncating** | Carrying the previous mark invents a value the data does not have; truncating discards honest points after the first gap. Breaking loses exactly the observations that were unanswerable, and the count says how many. |
| **`trade_pnl` returns a plain `Vector{Float64}`** | Structure-level grouping is preserved as a *function*, not as a struct with scalars attached: nothing that reads it needs more than the numbers. |
| **`n_opens` / `n_closes` are ledger functions** | They are counts of the ledger's own events, so the ledger is their natural argument -- and it is a home that does not depend on a wrapper type existing. |
| **`total_pnl` stays realised** | Changing what a reported number means, while keeping its name, breaks every comparison with a previously stored run. |
| **Absent key, not `NaN`, for an uncomputable metric** | `NaN` is the answer to "computed, and undefined" (one trade, zero variance). "Could not compute at all" is a different fact and must not read the same. |
| **Default kwargs baked into the dispatch entry** | The symbol carries the contract; default-arg drift between two call sites is impossible because there is only one source of truth. |
| **Canonical trade order: `(closed_at, pnl)` ascending** | Trades closing at one instant have no natural order; losses-first is deterministic and reconstructible. |

## Conventions consulted

Per design rule 5, the sources checked before fixing this round's new
public API shapes (checked 2026-09-14).

| Decision | Source checked | What it says |
|---|---|---|
| A session-difference series annualised by sessions per year, rather than a return series | Sharpe (1994), *The Sharpe Ratio*; fund performance reporting | A Sharpe ratio is a statistic on a series with a stated period, not on a series of trades. Without a capital base there are no returns, so the analogue is the period-to-period cash difference, annualised by sessions. That is the convention rather than a simplification -- and it is why a *trade*-sampled ratio was wrong even before the annualisation constant was. |
| `MarkedCurve` as a plain `struct` of `Vector` fields, with no supertype and no type parameters | [Julia manual, Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/) | "Don't use unnecessary static parameters" -- a parameter not used in the body should not exist -- and "avoid elaborate container types". The fields here are always `Vector{DateTime}` / `Vector{Float64}` / `Vector{Symbol}`, so there is no variation to abstract over. |
| Same, on whether a supertype is needed to participate in ecosystem interfaces | [Tables.jl, implementing the interface](https://tables.juliadata.org/stable/implementing-the-interface/) | Interface objects "are not required to subtype, but only implement the required interface methods"; its abstract types are explicitly not for dispatch. TimeSeries.jl's parametric `TimeArray <: AbstractTimeSeries` is the counter-case, and it is a *generic container library* where element and array types genuinely vary. |
| Same, on field type concreteness | [BlueStyle, "Type annotation"](https://github.com/JuliaDiff/BlueStyle) | Use the concrete field type rather than an abstract one; optimise with parametric types later rather than designing for variation that has not appeared. |
| `cents_to_usd` as one explicitly named conversion at the integer/float boundary | [Julia `Dates`, `Dates.value`](https://docs.julialang.org/en/v1/stdlib/Dates/) and [FixedPointDecimals.jl](https://github.com/JuliaMath/FixedPointDecimals.jl) | The ecosystem precedent is a named accessor, never an implicit promotion: `Dates.value(Millisecond(10))` is the only way to the raw integer, and FixedPointDecimals keeps money on an integer representation precisely so "$0.30 is actually 30 cents", making the crossing explicit and checked. |
| Keeping `_METRIC_TABLE` entries as plain named tuples of `(fn, defaults)`, selected by symbol | [MLJ.jl model search](https://juliaai.github.io/MLJ.jl/stable/model_search/) | MLJ's registry entries are literally named tuples of metadata, selected by name, and they already carry exactly this kind of field: `input_scitype`, `target_scitype`, `fit_data_scitype` record *what data the algorithm consumes* alongside its constructor and defaults. Metadata named tuples selected by name are the established shape; this table stays at the minimal end of it, carrying only the callable and its defaults. |
| Same, on when to graduate away from a symbol table | [Optim.jl minimization](https://julianlsolvers.github.io/Optim.jl/stable/user/minimization/) and [StatsBase `pacf`](https://github.com/JuliaStats/StatsBase.jl/blob/master/src/signalcorr.jl) | Optim selects algorithms by singleton *instances* (`LBFGS()`), which is what a table graduates to once each algorithm needs its own dispatch-driven behaviour; StatsBase's `method::Symbol=:regression` is the low-ceremony end. Five parameterless reductions sit at the StatsBase end. |
| `Union{MarkedCurve,Nothing}` for "this derived result could not be computed" | [Julia manual FAQ, nothingness](https://docs.julialang.org/en/v1/manual/faq/) and [Missing Values](https://docs.julialang.org/en/v1/manual/missing/) | `nothing` is "the absence of a meaningful return value", and `Union{T,Nothing}` is named as the recommended type "when a value `x` of type `T` exists only sometimes". `missing` is reserved for the statistical sense -- "no value is available for a variable in an observation, but a valid value theoretically exists". A curve that could not be built is absence of a thing, not an unobserved observation; an empty curve would be worse than both, conflating "not computed" with "computed, came out empty". |




