# Slice 2 brief: the engine switches to orders and the book

An implementation brief for the second slice of [ledger.md](ledger.md),
in the shape of [ledger-slice1.md](ledger-slice1.md) and derived from
the code as it stands after slice 1's hardening round (commit
`051b805`). The proposal is the design and is binding where this brief
is silent; the decisions of 2026-09-12 in
[ledger-orchestration.md](ledger-orchestration.md) are binding too.
This is the second draft: the first was reviewed with the human on
2026-09-12 and redrawn around one principle, **the engine computes,
the ledger records**. The engine turns a decision into immutable
inputs and makes one call; every id, the group, the order record and
the events are minted inside that call, so nothing is kept in sync and
nothing is rolled back. Records stay immutable; the only mutation is
`commit!`, as in slice 1.

What lands: `decide` takes a `Book` and returns orders; the venue
prices every leg of an order as a pure function; `record_order!` books
the structure as one transaction, whole or not at all, with the group
minted inside it and the order journal recorded beside the events;
commissions land as per-contract `Fee` events under IBKR's schedule;
`positions` is retired; persistence writes the ledger in place of
`positions.parquet`; the `@test_broken` at the end of
`test/ledger/test_append.jl` is rewritten to the writer's signature
and flipped to `@test`.

## Read first

1. [docs/design.md](../design.md), all seven rules.
2. [docs/proposals/ledger.md](ledger.md), sections 2 and 3 in full;
   "Orders", "Order journal", "Tick order", "Contract, venue,
   simplifications" and "Invariants" are the parts this slice lands.
3. [ledger-orchestration.md](ledger-orchestration.md), "Decisions taken
   on 2026-09-12".
4. [ledger-slice1-coverage.md](ledger-slice1-coverage.md), table 2: rows
   D1 to D6, D9 and D11 close in this slice; D4 and D7 partly (see
   "Deferred").
5. [ledger-fill-review.md](ledger-fill-review.md), "Enforced
   invariants": the cross-record checks are the contract of
   `check_join`.
6. Code as it stands: `src/ledger/types.jl`, `src/ledger/append.jl`
   (the writers, `_validate` and its helpers `_available` and
   `_first_eligible`), `src/backtest/engine.jl`, `src/policies/*.jl`,
   `src/agents/agent.jl`, `src/experiment/experiment.jl`, `show.jl`,
   `config.jl`, `src/persistence/store.jl`, `src/metrics/pnl_series.jl`,
   `src/metrics/ledger_series.jl`, `src/metrics/dispatch.jl` (the
   symbol-to-function table this slice copies the shape of);
   `test/ledger/fixtures.jl` and the last testset of
   `test/ledger/test_append.jl`; `test/backtest/test_engine.jl`,
   `test/experiment/test_experiment.jl`, `test/persistence/test_store.jl`.
7. IBKR's US options commission page, fetched on 2026-09-12 (the site
   refuses some fetchers; a copy is at
   `/tmp/claude-1000/-home-ale-dev-VolSurfaceAnalysis-jl/ad179082-6f51-49de-89d4-5b451f111fd6/scratchpad/ibkr_options.html`;
   the numbers are quoted under `execution.jl` below).

## The shape

The whole engine:

```julia
function run_backtest(agent::Agent, data::MarketData, from::DateTime, to::DateTime,
                      clock::Clock; fill_rule::Symbol = :cross_spread,
                      cost_model::Symbol = :ibkr_pro_us_options, tick_cents::Int = 1)::Ledger
    L, book = Ledger(), Book()
    ticks = tick_times(agent, data, from, to)
    if ticks === nothing
        ticks = timestamps(data, clock, from, to)      # keep the `if`: do not enumerate the grid when an override exists
    end
    for t in ticks
        # lifecycle: slice 3
        cut    = TimeCut(data, t)
        policy = current_policy(agent, t, cut, book)
        orders = decide(policy, t, cut, book)
        known_to = last_sequence(L)                    # what every order of this tick saw
        for order in orders
            record_order!(L, book, order; fill_legs(cut, order, t; fill_rule, cost_model, tick_cents)...,
                          effective_at = t, recorded_at = t, known_to)
        end
    end
    check_join(L; tick_cents)                          # holds by construction; the assertion is cheap
    return L
end
```

`fill_legs` is the venue: a pure function that resolves each leg's
quote and spot, applies the price rule and the commission schedule,
and returns the per-leg keywords `record_order!` takes, or throws a
named failure before anything is written. `record_order!` is the one
writer: it mints the group, the order and leg ids and the execution
ids, plans fills and matches, commits the events, and records the
order beside them. `book` is the fold `commit!` already maintains; it
is handed to the policy and returned nowhere, since it is derivable.
The engine holds no other state.

## Interpretations made in this brief

Decisions the proposal leaves open, taken here so the slice is
complete. Numbers 1, 2 and 8 were agreed with the human on 2026-09-12;
the rest can be vetoed in one sentence before the go.

1. **The order journal lives inside the `Ledger` container**, as
   `L.orders` beside `L.events`. The proposal's "outside the journal of
   economic facts" holds in substance: replays and cash fold `events`
   only, and the module still knows no quotes, spots or time cut (the
   records hold plain numbers and timestamps). The module doc's
   sentence that the journal belongs to the engine is redrawn: the
   engine resolves it, the ledger records it.
2. **No type hierarchy for the venue.** The price rule and the cost
   model are symbols dispatched through two small tables in the
   `_METRIC_TABLE` style; `Fill.fill_rule` is literally the table key
   it already stores; the tick is an integer. Slice 4 puts the three
   values into config and identity as plain values.
3. **Persistence lands with the slice.** Retiring `positions` removes
   `positions.parquet` and the type `load_run` rebuilt, so a result
   must be saved and reloaded as what it now is: `events.parquet`,
   `orders.parquet` and `order_legs.parquet` replace it, `load_run`
   rebuilds the ledger through one `commit!` and runs `check_join`,
   and `RUN_SCHEMA_VERSION` becomes 3. The proposal placed the `events`
   and `orders` tables in slice 6; that slice keeps `round_trips`,
   `marks`, `equity`, `failures`, the completeness flag and the rest of
   `compare_runs.jl`. Stored runs written under version 2 (the
   ten-year strangle run `5700d3f242f8132e`) are unreadable by
   `load_run` from this slice, one slice earlier than the handoff's
   "slice 4 breaks run ids"; that break is the identity hash and still
   comes in slice 4. No migration: the store holds one run and its
   config reruns.
4. **Fill prices are whole cents: the venue's tick.** The ledger
   refuses cash that is not whole cents (decision 1 of 2026-09-12), and
   neither the synthesized quotes (`SpreadFromOHLCV`:
   `low + λ(close − low)`) nor the fixtures' Black-Scholes quotes are on
   the tick. Exchanges only trade on the tick, so `:cross_spread` fills
   a buy at the ask rounded up and a sale at the bid rounded down to
   the class's tick, USD 0.01 for SPY, QQQ and IWM at every premium
   under the Penny Interval Program. The observation keeps the raw
   quote; the fill carries the tick price. Rule addition R5.
5. **A commission of zero books no `Fee`**; `:none` produces a ledger
   with no `Fee` events.
6. **`decide` receives the engine's own book.** It equals
   `book_as_known(L, known_to)` by the replay invariant (a test asserts
   it on every tick). A policy must not mutate it: a documented rule in
   `policies.md`, not a copy per tick.
7. **The venue's three values are keywords on `run_backtest`**;
   `run_experiment` uses the defaults. There is deliberately no keyword
   on `run_experiment`: it would change results without changing the
   run id. Slice 4 makes them `Experiment` fields.
8. **Fewer types.** No `OrderLegRecord` (the record embeds the `Order`
   the policy emitted; leg `k` has id `first_leg_id + k - 1`); no
   transient per-leg struct (`fill_legs` returns per-leg vectors, the
   shape codex proposed); no structure-rule marker (the behaviour is
   `record_order!` plus the engine pricing every leg first; slice 4
   names it when identity needs a name); one failure for a leg that
   cannot be priced, with the reason as a symbol, the way
   `MatchMismatch` bundles its reasons; a per-leg vector of the wrong
   length is an `ArgumentError`, Julia's own name for a malformed call.
9. **The cost model is IBKR Pro's US options schedule at the lowest
   monthly-volume tier** (≤ 10,000 contracts a month; a ten-year daily
   strangle trades about 5,000 in total). Third-party fees (exchange,
   ORF, OCC clearing, FINRA CAT, SEC) are not modelled. Both are named
   in the docstring.
10. **A duplicate execution id is refused** by `commit!`
    (`DuplicateExecution`), closing D4 for the simulated venue; the live
    adapter's idempotent retry is the adapter's job, later. The one
    slice 1 assertion that accepted a stale id flips.
11. **`Order.group` must name a minted group**; an unminted one is
    `DanglingReference(:group, g)`. An `Open` leg into an existing group
    adds a lot to that structure.
12. **`DailyShortStrangle.quantity` is an `Int`**; the config accepts
    `1` and `1.0` and refuses `1.5`; `to_dict` is unchanged and the
    canonical form already collapses `1` and `1.0`, so no run id moves.
13. **Two `@test_broken` placeholders wait for slice 3**, the pattern
    slice 1 used for this slice: an expiry inside the window is booked
    (`test_experiment.jl`), and the PR #9 regression "settlement uses
    trade underlying" becomes an `Expiry` against the lot's own
    underlying (`test_review_findings.jl`). That regression's second
    assertion, a window-end mark of an open lot as a PnL sample, is
    obsolete under proposal decision 8 and is dropped with a comment.
14. **`known_to` defaults to `last_sequence(L)`** at the call, right for
    one order per tick; the engine passes the value it captured before
    the loop, so the second order of a tick does not appear to have
    seen the first order's fills.

## Scope

Change, on the current branch, without committing:

- `src/ledger/types.jl`: `LegObservation`, `OrderRecord`, the
  `Ledger` fields and counters for them, `last_sequence`, `order_leg`.
- `src/ledger/append.jl`: `record_order!`, `DuplicateExecution` and its
  check in `_validate`. Nothing else in `src/ledger/` changes.
- `src/backtest/execution.jl` (new), `src/backtest/engine.jl`
  (rewritten), included in that order before `metrics`.
- `src/policies/policy.jl`, `src/policies/daily_short_strangle.jl`,
  `src/agents/agent.jl`: the new signatures.
- `src/experiment/experiment.jl`, `show.jl`, `config.jl`.
- `src/metrics/pnl_series.jl`: the positions-based builder goes;
  `PnLSeries` and `equity_curve` stay. `ledger_series.jl` unchanged.
- `src/persistence/store.jl`, `scripts/compare_runs.jl`.
- `src/VolSurfaceAnalysis.jl`: includes and exports.
- `src/positions/` deleted; `test/positions/` deleted;
  `docs/modules/positions.md` deleted.
- `src/data/synth.jl` line 94 and `docs/modules/market_data.md` lines
  34 to 36: the two mentions of `open_position` / `selector(::Trade)`
  reworded; nothing else in those files.
- Tests: `test/ledger/fixtures.jl`, `test_types.jl`, `test_append.jl`;
  `test/backtest/test_execution.jl` (new), `test_engine.jl`;
  `test/policies/test_policy.jl`; `test/agents/test_agent.jl`;
  `test/experiment/test_experiment.jl`, `test_config.jl`,
  `test_identity.jl`; `test/metrics/test_pnl_series.jl`, `test_core.jl`;
  `test/persistence/test_store.jl`; `test/market_data/test_parquet.jl`
  lines 57 and 58 only; `test/regressions/test_review_findings.jl` (the
  one policy and testset that use `Trade` / `Position`);
  `test/runtests.jl`.
- Docs: `docs/modules/ledger.md`, `backtest.md`, `policies.md`,
  `agents.md`, `experiment.md`, `metrics.md`, `persistence.md`,
  `docs/status.md`.

Touch nothing else. In particular: no lifecycle, no `Expiry` in the
tick loop, no window-end lifecycle, no marks, no venue values in
config or identity, no change to `identity.jl`, no change to
`test/ledger/test_book.jl`, `test_cash.jl`, `test_contracts.jl`,
`test_round_trips.jl` beyond what the fixture changes force, no new
dependency in `Project.toml`. Tests live beside the source they test,
one file per source file.

## Files and public surface

### `src/ledger/types.jl`

```julia
struct LegObservation                  # what one leg was priced against
    quote_at :: DateTime               # the quote's timestamp
    bid      :: Union{Float64,Missing}
    ask      :: Union{Float64,Missing}
    spot     :: Float64                # the leg's own underlying
    spot_at  :: DateTime
end

struct OrderRecord                     # one order as recorded
    order_id     :: Int
    first_leg_id :: Int                # leg k of `order` has id first_leg_id + k - 1
    group        :: Int                # the group minted or named for it
    decided_at   :: DateTime           # the tick
    known_to     :: Int                # book_as_known(L, known_to) is the book the policy was handed
    order        :: Order              # as the policy emitted it
    observations :: Vector{LegObservation}   # one per leg
end

mutable struct Ledger                  # gains, beside events, index and the four counters:
    orders        :: Vector{OrderRecord}
    next_order_id :: Int
    next_leg_id   :: Int
end

last_sequence(L::Ledger) -> Int                       # L.next_sequence - 1; 0 when empty
order_leg(L::Ledger, id::Int) -> (OrderRecord, Int)   # the record and the leg index k; DanglingReference(:order_leg_id, id) when none
```

The boundary test scans `src/ledger/*.jl` for the names of the
market-data types: comments here say "the quote" and "the spot", never
the type names.

### `src/ledger/append.jl`

```julia
record_order!(L, book, order::Order;
              prices::AbstractVector{<:Real},                  # per leg, per share
              observations::AbstractVector{LegObservation},    # per leg
              fees::AbstractVector{<:Integer} = zeros(Int, length(order.legs)),   # per leg, cents, a cost negative
              effective_at::DateTime, recorded_at::DateTime,
              known_to::Int = last_sequence(L),
              fill_rule::Symbol) -> OrderRecord
```

The structure-level writer, the one the engine calls. In order:

1. Shape: at least one leg, and every per-leg vector as long as the
   legs, else `ArgumentError`.
2. Group: `order.group === nothing` mints `L.next_group`; a named
   group must satisfy `1 <= g < L.next_group`, else
   `DanglingReference(:group, g)`.
3. Ids: `order_id = L.next_order_id`, `first_leg_id = L.next_leg_id`;
   leg `k` gets `order_leg_id = first_leg_id + k - 1` and
   `execution_id = L.next_execution + k - 1`; event ids and sequences
   continue the counters as `record_fill!`'s do.
4. Plan: for each leg in order, its `Fill`; for a `Close` leg its
   `Match`es, chosen with the validator's own helpers
   (`_first_eligible`, `_available`, over the `opened` fills and the
   `consumed` quantities of the batch so far), so a leg may close a lot
   the same order opened; `NothingToClose` when no lot is eligible,
   `ExceedsOpen` when the eligible lots run out. No scratch book.
5. Fees: one `Fee` per leg whose amount is non-zero, `source_id` the
   leg's fill, after all fills and matches.
6. `commit!(L, book, batch)`.
7. Then, and only then: push the `OrderRecord`, advance
   `next_order_id`, `next_leg_id` by the leg count, and `next_group`
   if it was minted. Nothing after `commit!` can fail, so a structure
   lands whole or not at all.

On any failure the ledger, its seven counters, its orders and the book
are unchanged. `record_fill!`, `record_expiry!`, `record_fee!` and
`commit!` keep their signatures; sharing the leg planner between
`record_fill!` and `record_order!` is welcome, not required.

New named failure: `DuplicateExecution(id::Int)`, `<: Exception`,
printing its name: a `Fill` whose `execution_id` is already held by a
fill in the ledger or earlier in the batch, checked in `_validate`.
`L.next_execution` still moves to `max(next, id + 1)`.

### `src/backtest/execution.jl`

The simulated venue, shaped like Interactive Brokers, as two
symbol-to-function tables in the `_METRIC_TABLE` style plus the tick:

```julia
const _FILL_RULES  = Dict{Symbol,Function}(:cross_spread => _cross_spread)
const _COST_MODELS = Dict{Symbol,Function}(:none => _no_commission,
                                            :ibkr_pro_us_options => _ibkr_pro_us_options)

fill_price(rule::Symbol, bid, ask, side::Side, tick_cents::Int) -> Union{Float64,Missing}
commission(model::Symbol, prices::AbstractVector{<:Real}, quantities::AbstractVector{<:Integer}) -> Vector{Int}
```

An unknown symbol errors listing the known ones, as `compute_metrics`
does. `:broker_execution` is not a rule of ours and is not in the
table; `check_join` recognises it.

`fill_price` takes raw values so `check_join` can recompute it from an
observation. `:cross_spread`: `Long` takes the ask, `Short` the bid; a
missing side returns `missing`. The result is on the tick, rounded
away from the trader, with the noise tolerance `contract_cents` uses:

```
cents = price * 100
Long : ceil(cents / tick - 1e-6) * tick / 100
Short: floor(cents / tick + 1e-6) * tick / 100
```

(`1.07 * 100` is `107.00000000000001` in binary; without the tolerance
a whole-cent ask would round up to `1.08`.) A price on the tick is
unchanged; every result passes `contract_cents` for a listed
underlying.

`commission` returns non-negative cents per leg; `fill_legs` negates
them into `Fee` amounts. `:none` returns zeros. `:ibkr_pro_us_options`
is IBKR Pro's schedule for US options at monthly volume ≤ 10,000
contracts, from
<https://www.interactivebrokers.com/en/pricing/commissions-options.php>
as fetched on 2026-09-12, quoted in the source with the URL and date:

| premium per share | per contract |
|---|---|
| < USD 0.05 | USD 0.25 |
| ≥ USD 0.05 and < USD 0.10 | USD 0.50 |
| ≥ USD 0.10 | USD 0.65 |

Minimum per order USD 1.00. The page's own examples are the test
literals: 1 contract at USD 2 premium = USD 1.00; 2 at USD 5 = USD
1.30; 3 at USD 0.075 = USD 1.50; 5 at USD 0.03 = USD 1.25. The order's
commission is the sum over legs of the rate at that leg's price times
its quantity, raised to the minimum when below it, then shared over
the legs in whole cents by cumulative rounding in proportion to each
leg's unraised amount (the rule `round_trips` uses for fee shares), so
the shares sum to the order's commission exactly. Higher volume tiers
and third-party fees are not modelled; the docstring says so.

### `src/backtest/engine.jl`

```julia
resolve_quote(cut::TimeCut, contract::ContractKey, t::DateTime) -> OptionQuote
fill_legs(cut::TimeCut, order::Order, t::DateTime; fill_rule::Symbol, cost_model::Symbol, tick_cents::Int)
    -> (prices, fees, observations, fill_rule)        # a NamedTuple: the per-leg keywords of record_order!
check_join(L::Ledger; tick_cents::Int = 1) -> Nothing
run_backtest(agent::Agent, data, from, to, clock; fill_rule, cost_model, tick_cents) -> Ledger
run_backtest(policy::Policy, data, from, to, clock; kw...) = run_backtest(StaticAgent(policy), ...)
```

`fill_legs` resolves, for every leg before anything else happens, the
quote (`resolve_quote`), the price through the rule, and the spot of
the leg's own underlying; fees are `-commission(cost_model, prices, quantities)`;
each observation is the quote's bid, ask and timestamp and the spot's
price and timestamp. It reads through the cut and writes nothing.

Named failures, `<: Exception`, printing their names:

- `UnpriceableLeg(contract::ContractKey, t::DateTime, reason::Symbol)`:
  design rule 7's "a leg that cannot honestly be priced". Reasons:
  `:no_quote` (an empty chain, or the contract absent from it; thrown
  by `resolve_quote`), `:no_executable_side` (the side the rule needs
  is `missing`), `:no_spot` (the underlying is served but has no spot
  at `t`). Nothing serving the selector stays `UnservedSelector`,
  thrown by `at`.
- `JoinViolation(field::Symbol, id::Int, reason::String)`: a fill and
  its order leg disagree (below).

`check_join` is the fill review's cross-record contract, run at the
end of `run_backtest` and by `load_run`. Order records: ids are
`1, 2, ...` in order, and each `first_leg_id` is the previous record's
plus its leg count (so leg ids are contiguous and never shared);
`length(observations) == length(order.legs)`. For every `Fill`:
`order_leg(L, id)` exists (`DanglingReference` otherwise, reused);
contract, side and intent equal the leg's and group equals the
record's; over the whole ledger the fills of that leg sum to at most
the leg's quantity; when `fill_rule != :broker_execution`, the leg's
observation has `quote_at` and `spot_at` at or before `decided_at`,
the required side present, and `price == fill_price(fill_rule, bid, ask, side, tick_cents)`
(an unknown rule is a violation on `:fill_rule`); when
`fill_rule == :broker_execution` the observation is not consulted.
Execution-id uniqueness is `commit!`'s `DuplicateExecution`; not
repeated here.

### `src/policies/policy.jl`, `src/agents/agent.jl`

```julia
decide(p::Policy, t::DateTime, data::TimeCut, book::Book) -> Vector{Order}
current_policy(a::Agent, t::DateTime, data::TimeCut, book::Book) -> Policy
```

`NoOpPolicy` returns `Order[]`. Docstrings: closes are `Close` legs
naming the group they close (`open_groups(book)`, `lots(book, g)`),
never counter-trades; the book is the engine's and must not be
mutated. `tick_times` and `declared_underlyings` unchanged.

### `src/policies/daily_short_strangle.jl`

`quantity::Int` (constructor takes `Integer`, positive; the keyword
form defaults to `1`). `decide` returns one
`Order(:daily_short_strangle, [Leg(put, Short, q, Open), Leg(call, Short, q, Open)])`
with `ContractKey(p.underlying, K, expiry, type)` legs, or `Order[]`
on the same gates as today.

### `src/experiment/experiment.jl`, `show.jl`, `config.jl`

```julia
struct ExperimentResult
    experiment :: Experiment
    ledger     :: Ledger
    pnl_series :: PnLSeries
    metrics    :: NamedTuple
end
```

`run_experiment`: `with_data`, `run_backtest` with the defaults,
`series = pnl_series(ledger)`, `compute_metrics`. `_build_settle`, the
window-end spot lookup and the "window-end spot missing" error go. The
"no clock tick in the window" error and the "clock selector must be an
`Underlying`" error stay as they are (the latter's message now says an
experiment ticks on an underlying's grid). Docstring: open lots at the
window end stay open and contribute nothing until the equity curve of
slice 5 marks them; nothing is force-settled.

`show`: the `positions` line becomes the event count with per-kind
counts, the order count, and the book at the last sequence (open lots,
open groups, cash in USD); the window-end spot line and the unmarked
count go.

`config.jl`: `quantity` defaults to `1`, accepts an integer or an
integral float, and errors in the loader's style
("policy(daily_short_strangle): quantity must be a whole number of
contracts, got 1.5") otherwise.

### `src/metrics/pnl_series.jl`

Keep `PnLSeries` and `equity_curve`; delete
`pnl_series(::AbstractVector{Position}; ...)` and its helpers. The
header comment says the series is built by `pnl_series(::Ledger)` in
`ledger_series.jl` and that `window_end_spot` and `n_unmarked` are
placeholders until slice 5.

### `src/persistence/store.jl`

`RUN_SCHEMA_VERSION = 3`. A run folder holds `config.toml`,
`manifest.parquet`, `metrics.parquet`, `events.parquet`,
`orders.parquet`, `order_legs.parquet`, `pnl_series.parquet` and
`artifacts/`. `group` is a SQL keyword: the column is `group_id`.
`_dt_sql` writes milliseconds (`"yyyy-mm-dd HH:MM:SS.sss"`) so a
loaded ledger equals the saved one exactly.

- `events.parquet`, one row per event in sequence order, `kind` plus
  nullable kind-specific columns: `run_id, sequence, id, kind
  ('fill'|'match'|'expiry'|'fee'), effective_at, recorded_at, group_id,
  order_leg_id, execution_id, underlying, strike, expiry, option_type
  ('C'|'P'), side ('long'|'short'), intent ('open'|'close'), quantity,
  price, fill_rule, open_fill_id, close_fill_id, settlement_price,
  outcome ('worthless'|'cash_settled'), source_id, amount`.
- `orders.parquet`, one row per order: `run_id, order_id, first_leg_id,
  label, group_id, operation (nullable), decided_at, known_to`.
- `order_legs.parquet`, one row per leg, the leg and its observation:
  `run_id, order_id, order_leg_id, leg_idx, underlying, strike, expiry,
  option_type, side, intent, quantity, quote_at, bid (nullable), ask
  (nullable), spot, spot_at`.
- `manifest.parquet`: `n_positions` becomes `n_events` and `n_orders`;
  `n_opens`, `n_closes`, `n_unmarked`, `window_end_spot` stay until
  slice 5.

`load_run` builds every event through its constructor in sequence
order and commits them to a fresh `Ledger()` as **one batch** through
`commit!` (the book is empty, so FIFO among batch-opened lots is
sequence order; a loaded ledger has passed every append-time check),
rebuilds the order records from the two tables, sets every counter one
past the largest id it saw (groups included), runs `check_join(L)`,
and returns the result. A load that fails a check throws that check's
named failure; it never drops the join.

`scripts/compare_runs.jl`: `positions` is replaced by `events` (key
`sequence`; exact on every column but `strike`, `price` and
`settlement_price`, within `TOL`), `orders` (key `order_id`) and
`order_legs` (key `order_leg_id`; doubles within `TOL`, NULL equals
NULL). The header comment says runs written under schema version 2
are no longer comparable.

### Exports

Add `LegObservation`, `OrderRecord`, `last_sequence`, `order_leg`,
`record_order!`, `DuplicateExecution`, `fill_price`, `commission`,
`fill_legs`, `check_join`, `UnpriceableLeg`, `JoinViolation`. Remove
`Trade`, `Position`, `payoff`, `open_position`, `entry_cost`,
`realized_pnl`. `resolve_quote` and `run_backtest` stay exported.

## Tests

Hand-computed literals, the arithmetic in a comment, cash in whole
cents (0.85 per share on SPY is 8500 per contract). Every failure test
checks that the failure fires, that the ledger snapshot and the book
are unchanged, and that the error prints its name. `_lg_snapshot`
gains `next_order_id`, `next_leg_id` and `length(L.orders)`; the
fixtures gain `_lg_seen(price; at = _LG_T_OPEN) = LegObservation(at, price, price, 480.0, at)`,
an observation whose bid and ask both equal the fill price, so
`check_join` passes on hand-built ledgers.

`test/ledger/test_types.jl`: the two records construct and are
immutable; `last_sequence` is 0 on a fresh ledger and the last sequence
after case 1; `order_leg` returns the record and the index for every
leg id of a two-order ledger and throws `DanglingReference(:order_leg_id, 9)`
for an unminted one; the boundary scan still passes.

`test/ledger/test_append.jl`:

1. **A short strangle lands as one batch, the group minted inside.**
   Fresh ledger; `Order(:strangle, [Leg(put470, Short, 1, Open), Leg(call490, Short, 1, Open)])`,
   prices `[0.85, 1.10]`, fees `[-65, -65]`: the record has `order_id 1`,
   `first_leg_id 1`, `group 1`, `known_to 0`, two observations; four
   events `Fill, Fill, Fee, Fee` with ids and sequences 1 to 4,
   execution ids 1 and 2, `order_leg_id`s 1 and 2, fee sources 1 and
   2; cash `8500 + 11000 - 130 = 19370`; `L.next_group == 2`,
   `L.next_leg_id == 3`; `open_groups(book) == [1]`;
   `book == book_as_known(L, 4) == book_effective(L, _LG_FAR)`;
   `check_join(L)` passes.
2. **A close order names its group and matches FIFO per leg.** On the
   ledger of 1, `Order(:close, [Leg(put470, Long, 1, Close), Leg(call490, Long, 1, Close)]; group=1)`
   at `[0.40, 0.60]`, fees `[-65, -65]`: six events
   `Fill, Match, Fill, Match, Fee, Fee`; the record has `order_id 2`,
   `first_leg_id 3`, `known_to 4`; cash
   `19370 - 4000 - 6000 - 130 = 9240`; book empty; round trips
   `[4370, 4870]` (put `(8500 - 4000) - 65 - 65`, call
   `(11000 - 6000) - 65 - 65`), summing to the cash;
   `pnl_series(L).pnl ≈ [92.40]`, one structure sample at the close
   instant.
3. **Legs are planned against the book after the earlier legs.** One
   order `[Leg(put470, Short, 2, Open), Leg(put470, Long, 1, Close)]`
   at `[0.85, 0.40]`: events `Fill(open 2), Fill(close 1), Match(1 → 2, q 1)`;
   one lot of 1 remains; cash `17000 - 4000 = 13000`.
4. **Zero fees book no `Fee`**: the order of 1 with the default `fees`
   is two events, cash 19500.
5. **A structure lands whole or not at all.** The existing
   `@test_broken` rewritten to the signature above (observations from
   `_lg_seen`) and flipped to `@test`; plus, each on the ledger of 1, a
   two-leg order whose *second* leg fails with `ExceedsOpen` (close 2
   of the put), `NonIntegralCash` (price 0.123456), `UnknownContract`
   (an SPX contract), `FillAfterExpiry` (a contract expiring before
   `effective_at`), `InvalidPrice` (price 0.0): nothing lands,
   `next_group`, `next_order_id`, `next_leg_id` and `L.orders`
   untouched.
6. **Shape**: prices, fees or observations of the wrong length, and an
   order with no legs, are `ArgumentError`; unchanged.
7. **A named group must be minted**: `group=7` on a ledger with
   `next_group == 2` is `DanglingReference(:group, 7)`; an `Open` leg
   into group 1 adds a third lot to it.
8. **`DuplicateExecution`**: in "one batch may open and close
   together", the stale-id assertion flips to a rejection; two fills
   sharing an execution id inside one batch are refused whole; an id
   above the counter lands and moves it; the name joins the "every
   error prints its name" loop.

`test/backtest/test_execution.jl` (new):

1. **`fill_price(:cross_spread, ...)`**: `Long` takes the ask, `Short`
   the bid; bid 1.02 / ask 1.065 give 1.02 short and 1.07 long; bid
   0.827261 gives 0.82, ask 0.847261 gives 0.85; whole-cent quotes are
   unchanged (1.07 stays 1.07 long, 1.02 stays 1.02 short);
   `tick_cents=5` on ask 1.06 gives 1.10 and on bid 1.06 gives 1.05; a
   missing required side returns `missing`, the other side missing is
   fine; every result passes `contract_cents` with `_LG_SPEC`; an
   unknown rule errors naming the known ones.
2. **`commission(:ibkr_pro_us_options, ...)`**, the page's examples:
   `([2.00], [1]) → [100]`; `([5.00], [2]) → [130]`; `([0.075], [3]) → [150]`;
   `([0.03], [5]) → [125]`; tier edges `([0.05], [3]) → [150]`,
   `([0.10], [2]) → [130]`, `([0.0499], [3]) → [100]` (75 raised to the
   minimum); a strangle `([0.85, 1.10], [1, 1]) → [65, 65]`; the
   minimum shared `([0.03, 0.50], [1, 1]) → [28, 72]` (25 + 65 = 90
   raised to 100; `round(100 * 25 / 90) = 28`, then `100 - 28 = 72`);
   `:none` gives zeros; an empty order gives an empty vector; an
   unknown model errors naming the known ones.

`test/backtest/test_engine.jl` (rewritten on the existing in-memory
fixture, whose quotes are whole cents: 5.00/5.10 call, 4.80/4.90 put,
spot 480):

1. `NoOpPolicy`: an empty ledger with no orders.
2. **`fill_legs` alone**: on the fixture cut at `ts1`, a two-leg order
   (long call, long put) gives prices `[5.10, 4.90]`, fees `[-65, -65]`,
   two observations with `quote_at == spot_at == ts1`, bid/ask as in
   the fixture and spot 480, and `fill_rule == :cross_spread`; with
   `cost_model=:none` fees are zeros.
3. **Single fill**: a policy opening one long call at `ts2`: one order
   record (`decided_at ts2`, `known_to 0`, group 1, one observation),
   one `Fill` (Long, Open, 5.10, `:cross_spread`, execution id 1) and
   one `Fee` of `-100` (1 contract at 5.10 is 65, raised to the USD 1.00
   minimum); cash `-51000 - 100 = -51100`; `check_join(L)` passes;
   every order leg has exactly one fill; execution ids are unique.
4. **Open then close**: open long at `ts1` (5.10, fee −100), close at
   `ts3` with a `Close` leg naming `only(open_groups(book))` (5.00 bid,
   fee −100): events `Fill, Fee, Fill, Match, Fee`; the second record
   has group 1, `first_leg_id 2`, `known_to 2`; cash
   `-51000 - 100 + 50000 - 100 = -1200`; one round trip of `-1200`
   (`(50000 - 51000) - 100 - 100`) equal to the cash;
   `pnl_series(L).pnl ≈ [-12.00]` at `ts3`; `n_opens == 1`,
   `n_closes == 1`.
5. **The book handed to `decide` is the known book**: a recording
   policy deep-copies the book it receives on each tick; each copy
   equals `book_as_known(L, rec.known_to)` for that tick's record, and
   the engine's final book equals `book_effective(L, far)`.
6. **Agent swap** and **the clock defines the ticks** as today, counted
   in orders and fills; `UnservedSelector` for the QQQ clock.
7. **Named failures**: spot served but absent at the fill tick is
   `UnpriceableLeg` with `:no_spot`; nothing serving spots is
   `UnservedSelector`; a strike not in the chain is `:no_quote`, and so
   is a masked instant; a missing ask on a `Long` leg is
   `:no_executable_side`; `resolve_quote(cut, contract, t)` returns the
   put with bid 4.80 and ask 4.90.
8. **Atomicity through the engine**: a two-leg order whose second leg
   has no quote throws with the ledger empty and every counter at 1; a
   two-leg order whose second leg closes a group with nothing to close
   throws `NothingToClose` the same way.
9. **`check_join`** on hand-built ledgers: passes on 3 and 4; fails
   with the right `field` on a fill whose leg record is missing
   (`DanglingReference`), a leg whose contract, side, intent or group
   differs, two fills of 1 on a leg of 1, a `:cross_spread` fill with
   `quote_at` after `decided_at`, with the required side missing, and
   with a price the rule does not produce (fill 0.85 against ask 0.86
   on a `Long` leg), an unknown `fill_rule`, and order records whose
   ids or `first_leg_id`s are not contiguous; a `:broker_execution`
   fill with a `missing` ask passes. `JoinViolation` and
   `UnpriceableLeg` print their names.

`test/policies/test_policy.jl`, `test/agents/test_agent.jl`: `Book()`
in place of `Position[]`; the strangle happy path returns one `Order`
labelled `:daily_short_strangle` with two `Short` `Open` legs of
quantity 1 (an `Int`), the same expiry, `group === nothing`, put strike
below spot below call strike, both resolvable with
`resolve_quote(cut, leg.contract, ts)`; `quantity=2` gives legs of 2;
the gate, the one-wing failure, the missing surface and the early cut
return `Order[]`.

`test/experiment/test_experiment.jl`:

1. NoOp: `isempty(res.ledger)`, no orders, empty series,
   `total_pnl == 0.0`, `n_round_trips == 0`, `hit_rate` NaN.
2. **An open lot at the window end stays open**: the long call of the
   existing fixture: one open lot in `book_effective(res.ledger, exp.to)`,
   empty series, `total_pnl == 0.0`, cash `-51100`.
3. **`@test_broken` for slice 3**: on the held-to-expiry fixture
   (expiry at the window end), `any(e isa Expiry for e in res.ledger.events)`;
   comment: flip when slice 3 books lifecycle in the tick loop.
4. **A QQQ leg under a SPY clock fills against QQQ**: the existing
   fixture; the fill is at 1.10 (QQQ's ask) and its observation's spot
   is 400 at `ts1`; the lot stays open; no error.
5. "window end is the last clock tick" and "expiry inside the window
   without a spot" are dropped (no window-end spot, no settlement in
   this slice); "no clock tick in the window errors" keeps the
   empty-window, after-data and currency-clock cases and drops the two
   spot cases (engine tests now).
6. **Strangle end to end**: one order record with two legs and two
   observations; two `Short` `Open` fills in group 1 sharing
   `effective_at == entry_ts`; assert both fill prices are ≥ 0.10, then
   both fees are `-65`; empty series (nothing closes in this slice),
   `n_opens == 2`, `n_closes == 0`, `open_groups == [1]`.
7. Rerun via `result.experiment` gives the same number of events.

`test/experiment/test_config.jl`: `quantity` 2.0 builds `2` (an
`Int`), the default is `1`, `1.5` errors naming "whole number"; the
parquet smoke asserts `isempty(res.ledger)`; the `show` test looks for
"events" and "orders" instead of "Window-end spot".
`test/experiment/test_identity.jl`: a config with `quantity = 1` and
one with `quantity = 1.0` have the same `full_hash`.

`test/metrics/test_pnl_series.jl`: slimmed to `PnLSeries` construction
and `equity_curve`; the canonical-order promise already lives in
`test_ledger_series.jl`. `test_core.jl`: build `PnLSeries` directly,
literals unchanged.

`test/persistence/test_store.jl`: the smoke result is the strangle of
`test_append.jl` cases 1 and 2 (ten events, two orders, cash 9240)
built through `record_order!` with `_lg_seen` observations;
`save_run` writes the seven files; `events.parquet` has ten rows,
`orders.parquet` two, `order_legs.parquet` four; the manifest carries
`n_events == 10`, `n_orders == 2`, `schema_version == 3`; `load_run`
returns a ledger whose events equal the saved ones (ids, sequences,
effective and recorded times, kinds, `cash(e)`), whose order records
equal the saved ones field by field, whose counters equal the saved
ledger's, and whose `book_as_known` at the last sequence equals the
saved book; a run whose `order_legs.parquet` is rewritten without one
leg (through DuckDB, as the `schema_version` test rewrites the
manifest) fails to load with a named failure from `check_join`; an
empty result round-trips with empty tables; the version test asserts
`RUN_SCHEMA_VERSION == 3` and keeps its two rewritten-manifest cases.

`test/market_data/test_parquet.jl` lines 57 and 58: the synthesized
call's prices through `fill_price(:cross_spread, c.bid, c.ask, side, 1)`
are 1.07 long (ask 1.065 rounded up) and 1.02 short.

`test/regressions/test_review_findings.jl`: the policy emits an order
per trade; the testset asserts both QQQ fills are at 1.00 (QQQ's ask)
with QQQ's spot 100 in their observations, and holds one `@test_broken`
for slice 3: the in-window leg is settled by an `Expiry` at 120 with a
round trip of `189900` cents (`(20.00 - 1.00) * 100 * 100`, less the
`100` commission on its opening fill, a lone contract raised to the
USD 1.00 minimum). The window-end mark assertion is dropped with a
comment naming proposal decision 8.

## Docs

- `docs/modules/ledger.md`: the order journal (the two records, ids
  and `known_to`) under "The kinds it defines"; `record_order!` in the
  invariants ("a structure lands whole or not at all; the group is
  minted inside the transaction"); `DuplicateExecution` among the named
  failures; "Responsibility boundaries" redrawn: the ledger records
  what a decision saw, the engine resolves it. "Conventions consulted"
  gains nothing.
- `docs/modules/backtest.md`: rewritten around orders, the venue as
  two dispatch tables and a tick, `fill_legs`, `check_join`; the tick
  order with the lifecycle slot named; the failure modes table with
  the named failures; a "Conventions consulted" section (design rule 5)
  citing IBKR's guaranteed combo orders, the commissions page (URL and
  fetch date), the Penny Interval Program for the tick, and FIX
  `ExecID` for execution ids. Lean: invariants and boundaries.
- `docs/modules/policies.md`, `agents.md`: the new signatures, closes
  as `Close` legs naming a group, the book must not be mutated.
- `docs/modules/experiment.md`: result fields, data flow, open lots at
  the window end, failure modes.
- `docs/modules/metrics.md`: the series is built from the ledger; the
  positions-based description goes; the two placeholder fields named.
- `docs/modules/persistence.md`: the file set and schemas above,
  version 3, the load path (one batch through `commit!`, then
  `check_join`).
- `docs/modules/positions.md` deleted; `market_data.md` loses its
  `selector(::Trade)` paragraph.
- `docs/status.md`: progress items 3 to 7 no longer describe
  `positions`; the in-flight entry says slice 2 landed, with the gate
  counts and the two Broken named.

## How to run here

One suite in a fresh process, about ten seconds once precompiled; the
ledger suites need `test/ledger/fixtures.jl`, the persistence suite
`using DuckDB`:

```
JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=. -e 'using VolSurfaceAnalysis, Test, Dates; include("test/ledger/fixtures.jl"); @testset "one" begin include("test/backtest/test_engine.jl") end' 2>&1 | tail -40
```

The gate is `Pkg.test()` in the shell window: `ws test`. Wait for the
julia process to exit rather than for echoed text (`ws wait`
false-matches the command), then `ws capture shell 60`. The box has
3.7 GB and two cores: check `free -m`; under about 1.3 GB available,
exit the REPL in the julia window with `ws repl "exit()"` first and say
so in your report, then relaunch it afterwards
(`julia --project=. -e 'using Revise' -i`, then `using VolSurfaceAnalysis`).
Always set `JULIA_NUM_PRECOMPILE_TASKS=1`.

## Done means

- The full gate is green: every surviving test passes, the new ones are
  in it, and there are exactly two Broken, both waiting for slice 3 and
  named above. Report the counts.
- The last testset of `test/ledger/test_append.jl` is `@test`, not
  `@test_broken`.
- `run_backtest` is the loop under "The shape", give or take names;
  the engine holds no state beyond the ledger and its book.
- `src/positions/`, `test/positions/` and `docs/modules/positions.md`
  are gone; nothing references `Trade`, `Position`, `open_position`,
  `payoff`, `entry_cost` or `realized_pnl`.
- The ledger module's boundary test still passes.
- Every module doc in scope matches the code; `docs/status.md` updated.
- No file outside the scope list changed; `git status` shows only
  scope files plus the two pre-existing untracked reviews. Nothing
  committed.
- A report: what landed (files, test count), the gate's last lines, the
  rule addition with its one-sentence reason, every new named failure,
  the deferred list below confirmed or corrected, and anything in this
  brief or the proposal you had to interpret or found wrong.

## Rule additions

- **R5. Fill prices are on the venue's tick, rounded away from the
  trader.** `:cross_spread` fills a buy at the ask rounded up and a sale
  at the bid rounded down to the class's tick, USD 0.01 for SPY, QQQ
  and IWM. The ledger refuses cash that is not whole cents, synthesized
  and modelled quotes are not on the tick, and exchanges only trade on
  it; rounding against the trader keeps the rule as conservative as
  crossing the spread already is. One function, one testset, one line
  in `backtest.md`.

Enforcement of what was already written, not additions: a duplicate
execution id refused (`DuplicateExecution`); an unminted group refused
(`DanglingReference(:group, g)`); the cross-record checks of
`check_join` (the fill review's list).

## Deferred, so nothing is lost

- Slice 3: lifecycle in the tick loop (`Expiry` with effective time at
  the settlement instant and recorded time at the tick), the window-end
  lifecycle at `exp.to`, the session calendar and the named valuation
  failure; the two `@test_broken` placeholders flip; the one auditable
  strangle run on the stored config follows.
- Slice 4: `fill_rule`, `cost_model`, `tick_cents` and the lifecycle
  model in config and identity, the structure rule named, the resolved
  `ContractSpec` projected into identity, the run-id break.
- Slice 5: the structure series and the equity curve from chain-mid
  marks; `window_end_spot` and `n_unmarked` leave `PnLSeries`, the
  manifest and `show`.
- Slice 6: `round_trips`, `marks`, `equity`, `failures` tables, the
  completeness flag, `compare_runs.jl` over them.
- Later models and adapters: partial fills in whole units, net-price
  allocation inside the combined spread, higher IBKR volume tiers and
  third-party fees, a non-penny tick per class, assignment and
  exercise, cash movements, the order status table, who mints the
  operation id of a roll, the live `:broker_execution` adapter with
  its idempotent retry and observation-less legs.
