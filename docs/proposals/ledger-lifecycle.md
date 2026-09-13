# Brief -- lifecycle: expiries in the tick loop, settled on the session calendar

The second half of PR 2 (`docs/proposals/ledger-orchestration.md`, "Pull
requests", item 2). The wiring landed in `9650e96`: the engine turns
decisions into orders and the ledger records them. Nothing yet settles a
contract that reaches its expiry, so `master` would get a strangle run
that opens and never closes.

This round books `Expiry` in the tick loop and gives it an honest
settlement price. It absorbs the parked **"Settlement rule"** backlog
item (`docs/status.md`, Backlog) in full: the user chose the calendar
rather than the minimal booking.

## Read first

1. `docs/design.md`, all seven rules. Rule 7 governs the failure cases
   here; rule 4 means `docs/status.md` moves in the same commit.
2. `docs/proposals/ledger.md`, "Tick order" (line 158) and
   "`LifecycleModel`: stated departures from the facts" (line 212).
3. `docs/status.md`, the Backlog entry "Settlement rule (replaces
   'surface-based theoretical settle')" -- it has the measured numbers
   this round must answer for.
4. `src/backtest/engine.jl` (the loop, `run_backtest`) and
   `src/backtest/execution.jl` (the venue as symbol tables -- the shape
   to copy).
5. `src/ledger/append.jl:374` (`record_expiry!`, already written) and
   `_check_expiry` at line 520 (the rules an expiry must satisfy).

Derive the work from that code, not from memory of earlier slices.

## What already exists -- do not rebuild it

- `record_expiry!(L, lot; settlement_price, effective_at, recorded_at)`
  settles a whole lot, derives `Worthless` / `CashSettled` from
  intrinsic, and commits. **This is the only writer this round needs.**
- `apply!(book, ::Expiry, spec)` folds it; `round_trips` already reads
  `Expiry`; `pnl_series` already reads `round_trips`.
- `_check_expiry` already enforces: copied contract / side / group match
  the opening fill, outcome agrees with intrinsic, effective at or after
  the opening fill, **effective at or after the contract's expiry**, and
  whole-lot (a partial expiry is refused).
- Persistence already writes and reads `Expiry` rows at schema 3.

## The five decisions this brief takes

**D1. The settlement instant is always the contract's expiry.** Even
when the reference price comes from an earlier session (an unscheduled
closure), `effective_at = lot.contract.expiry`. The obligation settles
when it expires; the departure from reality is *which print stands in
for the official close*, never *when the lot ceased to exist*. This is
also the only value `_check_expiry` accepts.

**D2. No type hierarchy, and no engine-side writer.** The settlement
rule is a symbol dispatched through a table, exactly like `_FILL_RULES`
and `_COST_MODELS`:
`const _SETTLEMENT_RULES = Dict{Symbol,Function}(:session_close => ...)`.
Before adding a struct, ask whether a symbol, a function or an existing
type does the job. And the engine computes, the ledger records: the
lifecycle step is a function returning what to settle and at what
price, and the loop calls the ledger's own `record_expiry!`, exactly as
`fill_legs` computes and the loop calls `record_order!`. The engine
defines nothing that mutates. Note the precise claim: `settlements`
mutates no state, but it is not side-effect-free -- it warns (D4).
Describe the boundary, never claim an absolute the code does not
deliver (`ledger-orchestration.md`, decisions of 2026-09-13).

**D3. Sessions come from the spot tree; the calendar only contradicts
it.** A date is a session when the underlying printed in regular hours
(09:30-16:00 ET) on it. BusinessDays.jl's NYSE calendar is not the
source of sessions -- it is the check that a printless weekday was
really closed. This is what makes the six early-close sessions
(official close 13:00 ET) settle correctly with no early-close table:
the session-close print is simply the last print at or before 16:00 ET,
which on those days is the 13:00 one.

**D4. An unsettleable lot stays open, loudly, and `settlements` is
what says so.** Under rule 7 a weekday with no prints that the calendar
does not list as closed is a *named valuation failure*, never evidence
that the exchange was closed. But a single bad day must not kill a
ten-year run, and the proposal says the lot stays open. `fill_legs`
throws because an unpriceable leg is the policy's bug and killing the
run is right; an unsettleable lot is a data gap the proposal expects.
So `settlement_price` throws the named failure like every other named
failure here, and `settlements` -- the one boundary -- catches it,
emits one `@warn` carrying contract, expiry and reason, and returns the
lot in `unsettled`.

The warning is inside `settlements`, not at the call site, for rule 7's
own reason: if reporting were the caller's job the window-end pass
could forget it, and a silent gap is the exact failure rule 7 exists to
prevent. Purity here means *mutates no state*, which still holds; the
two gap tests wrap in `@test_logs`. `unsettled` is still returned,
because the tests assert on it and PR 4's completeness flag will count
it (`ledger-orchestration.md`, "What remains", item 4).

**D5. A lot is examined for settlement exactly once.** `settlements`
takes an interval and returns the lots falling due in `(prev, t]`, not
every open lot with `expiry <= t`. An unsettleable lot stays open by
design, and its answer is fixed by the contract's expiry date rather
than by `t`, so a threshold would re-examine it at every later tick --
thousands of identical warnings and thousands of wasted calendar walks
on the ten-year run. The interval is safe because a fill after expiry
is already impossible (`FillAfterExpiry`), so every lot is in the book
at the first tick at or after its expiry. `prev` is a loop variable,
not engine state.

The interval is open below, so `prev = from` means a contract expiring
exactly at `from` is never examined. That cannot arise today, because
nothing is open before the first tick and a fill after expiry is
refused. It becomes live the day a ledger is seeded with open lots --
resuming a run, or a live loop starting against an existing book. Leave
the bound open and say so in the docstring; do not special-case the
first tick for a case that does not exist yet.

## Scope

### In

1. **`src/backtest/settlement.jl`** (new file; note `src/backtest/lifecycle.jl`
   would collide by name with `src/market_data/lifecycle.jl`).

   - **No new exception type.** `UnpriceableLeg(contract, t, reason)`
     (`engine.jl:30`) already has the fields; settling is pricing a leg
     at intrinsic against a reference print. Add three reasons to its
     docstring: `:unexpected_gap` (a weekday with no prints the
     calendar calls open), `:no_session` (the bounded walk back found
     none), `:no_print` (a session with no print at or before its
     close).
   - `settlement_price(rule::Symbol, cut, contract, t)` -> `Float64`,
     reads through the cut, mutates nothing, throws `UnpriceableLeg`. The `:session_close` rule:
     1. listed date = `Date` of `contract.expiry` converted to ET
        (expiries are stamped `et_to_utc(date, Time(16,0))` in
        `src/data/polygon.jl:21`; timestamps on disk are UTC).
     2. walk back from the listed date, at most **10 calendar days**
        (bound; exhausting it is `:no_session`, in the spirit of
        `DerivationExhausted`). For each candidate date D:
        `between(cut, SpotPrice, contract.underlying, 09:30 ET of D,
        16:00 ET of D)`. Non-empty -> D is the settlement session, and
        the reference price is the **last** record of that range.
        Empty and D is a weekday the calendar does not call a holiday
        -> `:unexpected_gap`. Empty otherwise -> keep walking.
     3. Every read goes through the tick's `TimeCut`, and every instant
        queried is at or before the expiry, which is at or before `t`.
        No-lookahead is structural; do not add a call-site convention.
   - `settlements(cut, book, prev, t; settlement_rule)` ->
     `(settled = Vector{Tuple{Lot,Float64}}, unsettled = Vector{UnpriceableLeg})`.
     Walks `open_lots(book)` (already ordered by opening fill id, so
     replay reproduces) for `prev < lot.contract.expiry <= t` (D5),
     calls `settlement_price` per lot, and partitions. The `try`/`catch`
     lives here and nowhere else, and so does the `@warn` per unsettled
     lot (D4). Mutates nothing. This is `fill_legs`' twin: the venue as
     a function returning what `record_expiry!` takes.
   - Ad-hoc closures the calendar may miss (2018-12-05, the national
     day of mourning; 2025-01-09) go in a small cited `const` set in
     this file, consulted beside `isholiday`. **Check first** whether
     BusinessDays.jl's `USNYSE` already carries them -- if it does, the
     set stays empty and the comment says so.

2. **`src/backtest/engine.jl`** -- the loop gets the proposal's four
   steps, in order. See "The loop after" below. The loop is a `foreach`
   over `settlements(...).settled` calling `record_expiry!` with
   `effective_at = lot.contract.expiry`, `recorded_at = t`, and a
   `prev` loop variable carrying the interval's lower bound (D5).
   **Do not add a writer to the engine.** The engine's mutation is two
   calls, `record_order!` and `record_expiry!`, both the ledger's.

   **Not one batched `commit!`**, though `commit!` takes a vector: a
   structure's legs are jointly atomic, which is what `record_order!`
   is for, but two lots expiring at the same instant are independent
   facts. Batching them would claim an atomicity that does not exist,
   and one unpriceable lot would reject the others.

3. **`run_backtest` gains `settlement_rule::Symbol = :session_close`**,
   beside `fill_rule` and `cost_model`. `run_experiment` passes no
   keyword, for the reason already written at
   `src/experiment/experiment.jl:113`: a value that changes results
   belongs in the run id, which PR 3 arranges.

4. **The window-end pass**: the same `foreach` once more after the
   loop, on a `TimeCut(data, to)` over `(prev, to]`. `to` is the
   evaluation endpoint and may be later than the last policy tick, so a
   contract expiring after the last tick settles here. Lots still open
   after it **stay open** (proposal decision 8) -- nothing is
   force-settled. The repetition is deliberate: the proposal names step
   4 as its own step, and reading it inline is the proof nothing hidden
   happens at the window end.

5. **`Project.toml`**: add `BusinessDays` with a compat bound.

6. **Exports**: `settlement_price`, `settlements` beside `fill_legs`
   in `src/VolSurfaceAnalysis.jl`; a custom loop needs step 1 as much
   as step 3.

7. **Docs** (rule 1, rule 4): `docs/modules/backtest.md` gains the
   lifecycle section and a conventions-table row with a *checked* source
   (OCC exercise by exception at expiry; the NYSE holiday schedule).
   Verify every URL returns 200 -- two citations in this file were 404
   last round. `docs/status.md`: step 4 loses "Lifecycle ... is slice
   3", the in-flight entry moves, and the Backlog entry "Settlement
   rule" is **deleted**, since this lands it.

### Out -- do not touch

- `src/ledger/` in any form. The ledger module shipped in #11 and this
  round only calls it. If a rule there looks wrong, surface it (rule 3)
  rather than editing it.
- Config and identity. `LifecycleModel` projecting into `core_hash` is
  PR 3. **State in the docs that this round changes results under
  unchanged run ids**; nothing on disk silently disagrees, because
  schema 3 already refuses runs written under schema 2.
- The equity curve, `window_end_spot`, `n_unmarked`, the structure
  series, the completeness flag -- all PR 4.
- Marking an open lot past the window end. Item (4) of the old backlog
  entry (the mark for a leg still open at the window end) is the equity
  curve's, not this round's; carry it into the PR 4 notes as it is
  deleted from the backlog.
- Early assignment, exercise, physical delivery. Named as departures in
  the doc, not modelled.

## The loop after

`run_backtest`'s body becomes the proposal's tick order, read top to
bottom. Steps 1 and 3 have the same shape -- a function of the cut, then
the ledger's own writer -- and the engine defines nothing that mutates.

```julia
function run_backtest(agent::Agent, data::MarketData, from::DateTime, to::DateTime,
                      clock::Clock; fill_rule::Symbol = :cross_spread,
                      cost_model::Symbol = :ibkr_pro_us_options,
                      settlement_rule::Symbol = :session_close,
                      tick_cents::Int = 1)::Ledger
    L = Ledger()
    # Sparse policies (once a day on minute data) override `tick_times` so
    # the engine never enumerates the clock's grid; keep the `if`.
    ticks = tick_times(agent, data, from, to)
    if ticks === nothing
        ticks = timestamps(data, clock, from, to)
    end
    prev = from                                    # lower bound of the lifecycle interval
    for t in ticks
        cut = TimeCut(data, t)
        # 1. Lifecycle, before the decision: a policy sees expired legs gone.
        #    `settlements` warns about what it could not price; a lot falling
        #    due in (prev, t] is examined exactly once, ever.
        foreach(settlements(cut, L.book, prev, t; settlement_rule).settled) do (lot, p)
            record_expiry!(L, lot; settlement_price = p,
                           effective_at = lot.contract.expiry, recorded_at = t)
        end
        prev = t
        # 2. Decide on the book the ledger owns.
        policy = current_policy(agent, t, cut, L.book)
        orders = decide(policy, t, cut, L.book)
        known_to = last_sequence(L)                # what every order of this tick saw
        # 3. Fill: every leg priced before anything is written.
        for order in orders
            rec = record_order!(L, order; fill_legs(cut, order, t; fill_rule, cost_model, tick_cents)...,
                                effective_at = t, recorded_at = t, known_to)
            check_join(L, rec; tick_cents)
        end
    end
    # 4. Window end: lifecycle once more at the evaluation endpoint, which
    #    may be later than the last policy tick. Lots still open stay open.
    foreach(settlements(TimeCut(data, to), L.book, prev, to; settlement_rule).settled) do (lot, p)
        record_expiry!(L, lot; settlement_price = p,
                       effective_at = lot.contract.expiry, recorded_at = to)
    end
    return L
end
```

Four things worth seeing in that shape:

- **The engine holds no state and owns no writer.** It reads `L.book`,
  which the ledger folds for itself, and writes through the ledger's two
  writers. No expiry queue, no calendar cached on the engine, no
  `try`/`catch` in the loop. `prev` is a loop variable, not state.
- **`known_to` is captured after the lifecycle step**, so the sequence
  an order records as "what the decision saw" already includes that
  tick's expiries. That is what makes `book_as_known` honest.
- **`recorded_at = t` with `effective_at = lot.contract.expiry`** is the
  general case the slice 1 hardening round's rules R1 (recorded at or
  after effective) and R2 (recorded nondecreasing) were written for, and
  it is the sole source of the two replays disagreeing at an
  intermediate instant.
- **Nothing binds the result.** With the warning inside `settlements`
  the call site needs no local, so there is no vaguely-named `due` or
  `s` standing between the reader and the two things that happen.

## Tests

`test/backtest/test_settlement.jl` (new, added to `runtests.jl`), plus
the two flips. Use hand-computed literals, never a value read back from
the code under test.

1. **Flip `test/experiment/test_experiment.jl:134`** from `@test_broken`
   to `@test`: an expiry inside the window is booked.
2. **Flip `test/regressions/test_review_findings.jl:81`** to `@test`:
   the in-window QQQ leg settles against **QQQ's** spot (120.0, not
   SPY's 90.0) and the round trip is **189900 cents** -- `(20.00 - 1.00)
   * 100 * 100 - 100`, the trailing 100 being the USD 1.00 minimum
   commission on the opening fill. Delete the "Flip to `@test` when
   slice 3 lands" comments.
3. **Lifecycle precedes the decision**: a policy whose `decide` asserts
   on the book it is handed sees the expired lot already gone, and
   `known_to` for that tick's order is at or after the expiry's
   sequence.
4. **Early close**: a session whose last print is 13:00 ET settles at
   that print, with `effective_at` still the 16:00 ET expiry instant.
5. **Unscheduled closure**: no prints on the listed expiry date, the
   calendar (or the cited const set) calls it closed -> settles at the
   **previous** session's close print, `effective_at` unchanged at the
   listed expiry instant. This is D1 made checkable.
6. **Unexpected gap**: a weekday with no prints that the calendar calls
   open -> `settlement_price` throws `UnpriceableLeg` with reason
   `:unexpected_gap`; `settlements` on a one-lot `Book` returns it in
   `unsettled` and nothing in `settled`, under `@test_logs` for the
   warning; after `run_backtest` the lot is still in `open_lots` and no
   `Expiry` exists for it.
7. **Window end after the last tick**: a contract expiring between the
   last policy tick and `exp.to` is settled, with `recorded_at == to`.
8. **Both replays agree**: after a run with expiries,
   `book_as_known(L, last_sequence(L)) == book_effective(L, to)`, and an
   expiry booked at a tick after its effective instant makes the two
   disagree at an intermediate `t` -- the proposal's "the two replays
   differ only by lifecycle booked at the tick after its instant"
   (line 157), pinned rather than assumed.
9. **Whole-lot only**: the engine never asks `record_expiry!` for part
   of a lot (a lot half-closed by a `Match` then expiring settles its
   remainder, once).
10. **`settlements` mutates nothing**: called twice on the same `Book`
    and cut it returns equal values and the book is unchanged.
11. **A lot is examined once (D5)**: an unsettleable lot in a run of
    many ticks after its expiry produces **exactly one** warning, and a
    lot whose expiry is at or before `prev` is not returned at all.
    This is the guard against thousands of identical warnings on the
    ten-year run.

## How to run here

Gate, from the orchestration doc's step 3 -- do not wait on pane text,
a grep for your own marker matches the echoed command line:

```
ws run "JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=. -e 'using Pkg; Pkg.test()'"
```

Check `free -m` first; under about 1.3 GB available, exit the REPL in
the `julia` window before running and relaunch it after. Adding
BusinessDays means a precompile -- keep `JULIA_NUM_PRECOMPILE_TASKS=1`
for `Pkg.instantiate()` too. The box is 2 cores, 3.7 GB, no sudo.

## Done means

1. Gate green with **0 broken** -- the two `@test_broken` were the only
   ones, and they flip here.
2. One auditable strangle run on the stored config,
   `configs/strangle_spy_16d_1dte.local.toml` (2016-03-28 to
   2026-03-27, 1-DTE SPY, one contract per leg). The old backlog entry
   measured the target: of 1700 expiry instants, 1691 had a spot at the
   16:00 instant and nine did not -- six early closes, two unscheduled
   closures, and a final pair past the end of the data. **After this
   round the first eight must settle and only the final pair stay open**,
   with no `:unexpected_gap` warnings. Report the actual counts; a
   number that disagrees with this brief is a finding, not a failure to
   hide.
3. `docs/modules/backtest.md` and `docs/status.md` current, the
   Backlog's "Settlement rule" entry deleted, every cited URL checked.
4. A proposed commit split, and stop. Commits are the human's call.
