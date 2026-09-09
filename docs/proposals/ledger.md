# Ledger rebuild -- proposal

Status: proposal, 2026-09-09. Nothing here is implemented. The plan
replaces the fill-vector ledger (`Vector{Position}` out of
`run_backtest`, FIFO-matched after the fact by `pnl_series`) with an
event ledger whose lifecycle is booked inside the run. It follows how
options are actually kept by brokers and clearinghouses, and it is meant
to be the same book the live-trading loop of [vision.md](../vision.md)
hands to a policy.

## 1. Where the ledger stands today

The ledger is a fill log. `run_backtest` appends one `Position` per fill
and returns the bare vector. Every lifecycle question is answered
somewhere else:

- The **engine** records the fill but not its intent (open or close) nor
  the structure it belongs to. Two strangle legs land as two unrelated
  rows.
- The **metrics** layer (`pnl_series`) owns the matching rule (FIFO,
  intent guessed from direction), the stamping rule (residuals at the
  leg's expiry) and the sample-order rule, while its doc says it does
  not own settlement.
- The **experiment** owns the settlement rule (`_build_settle`: spot at
  `min(expiry, window_end)`, exact instant) in code only. It is neither
  configured nor in identity, so a change to it silently changes every
  stored result.

Issues found in the review, most consequential first:

1. **Expiry is not an event.** A policy at tick `t` receives the full
   fill log with expired legs indistinguishable from live ones. The
   `policy.jl` doc tells policies to derive net-open themselves and
   nothing implements that view. `DailyShortStrangle` ignores the ledger,
   which is why this has not bitten yet.
2. **The sample unit is a leg, not a structure.** A daily strangle emits
   two samples per day at one timestamp. `hit_rate` is per leg;
   `sharpe` annualises with 252 while the series carries about 504
   samples a year (inflated by about 1.41). The losses-first tie-break is
   a workaround for the missing aggregation.
3. **Equity is realised-only.** The curve moves only at close fills and
   expiries; `max_drawdown` cannot see open exposure, which for a
   short-premium strategy is the drawdown that matters. A residual marked
   at the window end is still stamped at its expiry, so the curve runs
   past the window.
4. **The persisted series has no lineage.** `pnl_series.parquet` is
   `(idx, timestamp, pnl)`: no leg index, quantity, settlement spot, or
   closed-vs-settled flag. `n_unmarked` says how many legs were skipped,
   not which. Cross-run SQL cannot break PnL out by expiry or structure.
5. **`window_end_spot` is dead but load-bearing.** No computation reads
   it, yet `run_experiment` refuses a non-`Underlying` clock and dies when
   the window-end spot is missing, for a provenance field.
6. **The live-trading split is blocked by the interface.** `decide` takes
   a fill vector; a broker hands back net positions.

Concrete defects on top of the structure:

- **FIFO float residue creates phantom lots.** `Trade.quantity` accepts
  any positive real. Open 0.3, close 0.1 three times: the last match
  leaves `2.8e-17`, which the overshoot branch pushes as a flipped lot on
  the opposite side, later settled as a residual sample.
- **Units are per share; docs say USD.** `entry_cost` has no contract
  multiplier while the doc calls `quantity` "contracts".
- **`sharpe` / `sortino` subtract a rate from a dollar PnL**
  (`risk_free / periods_per_year`), invisible at the default of zero.
- **Two public answers to "what did this ledger make".** The vector
  `realized_pnl` overload double-counts on a ledger with counter-trade
  closes, as the `pnl_series` header itself notes; intrinsic math is
  duplicated between `payoff` and `_unit_payoff`.
- **Matching ignores expiry.** A fill stamped after its contract's expiry
  is matched as a close rather than rejected.
- **Counts measure side changes, not intent.** `n_opens` increments on
  any direction flip; `n_round_trips` includes unsettled residual chunks.

## 2. Conventions consulted

Bookkeeping conventions common to trading books and how they bear on
this design. Per design rule 5, this is the durable record; the module
doc inherits it when the code lands.

| Convention | Source | Consequence here |
|---|---|---|
| Append-only journal; corrections are new entries | Double-entry bookkeeping practice | `Ledger` is a `Vector` of immutable events; nothing is edited |
| Every order declares intent (buy/sell to open/close); a close with nothing to close is rejected | US broker order tickets; OCC position reporting | `Leg.intent`; the engine errors on an unmatched `Close` |
| A fill is not a position; positions are the net of fills per instrument | Broker position statements | `Book` is a view derived by replay, never stored |
| Lot matching has a named rule (FIFO default) and the pairing is recorded | IRS default for securities; broker realised-PnL reports | FIFO in the engine at fill time; `matched_id` on the closing event |
| Contract multiplier: prices per share, cash per contract times 100 | OCC standard equity option contract | `LedgerRules.multiplier`, applied once when an event's `cash` is computed |
| Expiry outcomes: OTM removed without cash; ITM auto-exercised above a small threshold (exercise by exception) | OCC Rule 805 (exercise by exception) | `Expiry` event with `outcome` in `{worthless, exercised}` |
| PM-settled contracts settle against the official close of the last session on or before the listed date; early closes and unscheduled closures move the session, not the rule | Cboe / OCC settlement procedures | `SettlementRule` owns session + reference price (the existing backlog item) |
| Physical delivery on SPY; cash at intrinsic on index products | OCC deliverable specs | Research simplification: cash-settle SPY at intrinsic, named in the rule |
| American style: early assignment is rational for deep-ITM short calls before ex-dividend and deep-ITM short puts with no extrinsic left | Standard options texts (Hull, ch. on American options); OCC random assignment | `ExerciseRule` with `Never` and `Rational(threshold)`; an `Assignment` event that books intrinsic plus the dividend owed |
| Realised and unrealised are separate lines; equity is cash plus marks; drawdown is measured on equity | Broker statements; fund reporting | `round_trips` (realised) and `equity_curve` (marks from a `MarkSource`) |
| Fees per contract | Broker / exchange / OCC / regulatory fee schedules | `CostModel`, a `Fee` event, zero by default |
| Abstract-typed containers box elements and dispatch dynamically; avoid in hot loops | Julia manual, Performance Tips ("Avoid containers with abstract type parameters") | Accepted here: a ledger holds thousands of events walked once per metric; the flat form lives at the storage boundary |

Not relevant for research and left out: taxes, T+1 cash timing,
corporate actions on SPY, account structures.

## 3. Proposed structure

```
decide ──► Order ──► engine fills ──► Event ──► Ledger ──► Book (view for decide)
                                                   │
                        lifecycle rules ───────────┤ (Expiry, Assignment, Fee)
                                                   ▼
                                     round_trips ──► PnLSeries ──► metrics
                                     equity      ──► EquityCurve ──► drawdown, vol
```

### 3.1 What a policy says

An order is one structure-level instruction. The policy names the legs
and the intent; the engine assigns identity.

```julia
@enum Intent Open Close

struct Leg
    trade  :: Trade        # contract, direction, quantity, unchanged
    intent :: Intent
end

struct Order
    label :: Symbol                  # :strangle, :condor, :roll, ...
    legs  :: Vector{Leg}
    group :: Union{Nothing,Int}      # nothing => the engine mints a fresh group
end

decide(policy, t, cut, book::Book) -> Vector{Order}
```

An opening strangle is one `Order` with two `Open` legs and
`group = nothing`. Closing it is one `Order` with two `Close` legs naming
the group found in the book. A `Close` with nothing to close is an error
at fill time.

### 3.2 What the ledger holds

Typed events under an abstract `Event`; each kind carries only its own
fields and dispatch does the work (`apply!(book, ::Fill)`,
`cash(::Expiry)`). Every event has an `id`, a `t`, and a `group`. Cash is
a consequence of events, not a separate state.

```julia
abstract type Event end

struct Fill <: Event
    id, t, group, trade, intent
    price, bid, ask, spot            # snapshot at fill
    matched_id                       # opening fill this closes against; 0 for an open
    cash                             # signed, times multiplier
end

struct Expiry <: Event
    id, t, group, trade, quantity
    session, settlement_price        # the official close used
    outcome                          # :worthless | :exercised
    cash
end

struct Assignment <: Event           # exercise rule fired on a short lot
    id, t, group, trade, quantity, price, dividend, cash
end

struct Fee <: Event
    id, t, group, cash
end

struct Unsettled <: Event            # design rule 7: named, not counted
    id, t, group, trade, quantity, reason
end

struct Ledger
    events     :: Vector{Event}
    multiplier :: Float64
    currency   :: Currency
end
```

The **book** is a view, never stored. It is what a policy and a live
broker loop both see.

```julia
struct Lot
    group, trade, open_fill_id, remaining, unit_cost
end

struct Book
    lots :: Dict{ContractKey, Vector{Lot}}   # FIFO order per contract
    cash :: Float64
end

book(ledger)        # replay from the start
book(ledger, t)     # replay up to t, for inspection
```

Container choice: `Vector{Event}` with `Event` abstract is the cleaner
API (one type per kind, no sentinel fields, new kinds are new methods).
The cost is one dynamic dispatch per event on a walk of a few thousand
events, well under the parquet read any metric sits on. If a profile ever
disagrees, a concrete parametric container behind the same
`AbstractVector{Event}` API is a local change.

### 3.3 Who writes which event

The engine is the only writer. Rules are values in the experiment
config and therefore in core identity.

```julia
struct LedgerRules
    multiplier :: Float64          # 100 for US equity options
    settlement :: SettlementRule   # session + reference price
    exercise   :: ExerciseRule     # Never | Rational(threshold)
    fills      :: FillRule         # CrossSpread | Mid
    costs      :: CostModel        # per-contract fee, zero by default
end
```

Per tick, in this order:

1. **Lifecycle first.** `due(rules, book, cut, t)` yields the `Expiry`
   and `Assignment` events whose instant is at or before `t`. They are
   appended and the book updated, so a policy sees expired legs gone
   before it decides.
2. **Decide.** `current_policy` and `decide` receive the `Book`, not the
   raw events.
3. **Fill.** Each leg resolves a quote through the cut, produces a `Fill`
   and any `Fee`, and updates the book. A `Close` leg FIFO-matches the
   book's lots and records `matched_id`.
4. **Window end.** Lifecycle runs once more for anything due at the last
   tick. Lots still open are not force-settled: they stay open and are
   marked, which is the unrealised line.

Settlement (session, reference price) is the backlog item "Settlement
rule"; it moves into `SettlementRule` unchanged in substance. Early
exercise is modelled as a rational counterparty: a short call the day
before an ex-dividend date when the dividend exceeds the call's extrinsic
value, a short put when extrinsic has gone to zero; booked at intrinsic
plus the dividend owed. The never-vs-rational gap is the model risk and
is expected to be negligible for 1-DTE structures.

### 3.4 What the metrics read

Two derived tables, both reconstructible from the ledger with no data
access.

```julia
struct RoundTrip
    group, label, trade, quantity
    open_id, close_id             # event ids: the pairing is auditable
    opened_at, closed_at
    kind                          # :closed | :expired | :assigned
    pnl
end
round_trips(ledger) -> Vector{RoundTrip}

struct PnLSeries
    unit :: Symbol                # :structure (default) | :leg
    timestamps, pnl, groups
    n_unsettled :: Int            # derived from Unsettled events
end
pnl_series(ledger; unit=:structure)

struct EquityCurve                # cash + marks on the clock grid or daily
    timestamps, realized, unrealized, equity
end
equity_curve(ledger, marks)       # marks from a MarkSource: chain mid, surface fallback
```

Metrics stay pure functions. Sharpe / Sortino read `PnLSeries` at
structure level with the sampling frequency derived from the series, not
a default. Drawdown and volatility read `EquityCurve`, so open exposure
counts. Hit rate is per structure. `window_end_spot` is dropped.

### 3.5 Inspection

- **Tables, not counts.** Persist `events`, `round_trips`, and `equity`
  as parquet (flat rows with a `kind` column at the storage boundary, one
  `to_row` per event type next to its `apply!`). Cross-run SQL can break
  PnL out by label, group, expiry, or outcome without rerunning.
- **Replay.** `book(ledger, t)` answers "what was open at `t`" for any
  `t`; replayed cash must equal the sum of event cash (reconciliation).
- **Names, not tallies.** An `Unsettled` event carries the leg and the
  reason; the count is derived.
- **One summary.** `summary(result)` prints per-label round trips, hit
  rate, total realised, unrealised at window end, fees, and any
  unsettled legs.
- **Live parity.** A broker adapter emits the same `Fill`, `Expiry`, and
  `Assignment` events, so the book a policy sees is the same type in
  both loops.

## 4. Decisions to take before coding

Each of these changes stored results, so they are settled once, together
with the id break the settlement backlog item already budgeted for.

1. **Units.** `quantity` in contracts, prices per share,
   `multiplier = 100`, cash in USD. (Recommended.)
2. **Sample unit.** Structure by default, leg on request.
3. **Rules in core identity.** `LedgerRules` hashes into `core_hash`.
4. **Window-end handling.** Open lots are marked, not force-settled;
   `window_end_spot` goes.
5. **SPY settlement.** Cash at intrinsic against the official close,
   named in `SettlementRule` as a research simplification.

## 5. Slices

Small, each landing green with its module doc, in this order:

1. `ledger` module: `Order` / `Leg` / `Event` types, `Book`, `apply!`,
   `round_trips`; `positions` folds into it (`Trade` stays; `Position`
   becomes `Fill`). Pure, no data access. Tests by hand-built ledgers.
2. Engine: `decide` returns `Vector{Order}` and receives a `Book`; fills
   record intent, group, and matching; `Close` with nothing to close
   errors. `DailyShortStrangle` emits one `Order` per day.
3. `lifecycle` module: `SettlementRule` (sessions from the spot tree,
   official close), `Expiry` booked in the tick loop. `ExerciseRule`
   starts at `Never`; `Rational` follows once a dividend source exists.
4. `LedgerRules` in config and identity; schema version bump; multiplier
   applied.
5. Metrics: `PnLSeries` over `round_trips` at structure level;
   `EquityCurve` from a chain-mid `MarkSource`; drawdown and volatility
   move onto it; the risk-free term is dropped or re-expressed per
   sample in cash.
6. Persistence: `events` / `round_trips` / `equity` tables replace
   `positions` / `pnl_series`; `load_run` rebuilds the ledger;
   `compare_runs.jl` follows.
7. Docs: `positions.md` / `backtest.md` / `metrics.md` /
   `experiment.md` / `persistence.md` updated per slice; this proposal
   deleted once landed, as `docs/proposals` was after the data-kinds
   rebuild.
