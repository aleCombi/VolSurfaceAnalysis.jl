# Slice 2 fix brief: the join is checked where the proposal says

A fix round on slice 2 of [ledger.md](ledger.md), closing the four
findings of codex's review
([ledger-slice2-review.md](ledger-slice2-review.md)) and two leftovers
the implementing agent reported. The slice itself is in the working
tree, uncommitted, gate 2795 passed, 0 failed, 2 broken; this round
lands on top of it and is committed with it. The slice 2 brief
([ledger-slice2.md](ledger-slice2.md)) stays the contract for
everything this brief does not touch.

## Read first

1. [ledger-slice2-review.md](ledger-slice2-review.md), the four findings
   and "Decisions the brief left to the code".
2. [ledger-slice2.md](ledger-slice2.md), sections "The shape" and
   "Interpretations made in this brief" (items 1, 2 and 8), and the
   `engine.jl` and `append.jl` parts of "Files and public surface".
3. [docs/proposals/ledger.md](ledger.md), section "Order journal": "The
   join is validated, not assumed. At engine append, at persistence
   write and at load."
4. Code as it stands: `src/ledger/append.jl` (`record_order!`),
   `src/backtest/engine.jl` (`check_join`, `run_backtest`),
   `src/persistence/store.jl` (`save_run`, `_load_ledger`);
   `test/backtest/test_engine.jl` (the `check_join` testset and its
   `_en_with_record` / `_en_with_fill` helpers),
   `test/persistence/test_store.jl`, `test/ledger/fixtures.jl` (cases 7
   and 8).

## Scope

Change, on the current branch, without committing:

- `src/ledger/append.jl`: `record_order!` only (part C).
- `src/backtest/engine.jl`: `check_join` (part A), the loop (part A).
- `src/persistence/store.jl`: `save_run` (part B).
- `src/metrics/ledger_series.jl`: docstring wording only (part E).
- Tests: `test/backtest/test_engine.jl`, `test/persistence/test_store.jl`,
  `test/metrics/test_ledger_series.jl` (one testset name).
- Docs: `docs/modules/ledger.md`, `docs/modules/backtest.md`,
  `docs/modules/persistence.md` (parts C and D), `docs/status.md`
  (gate counts).

Touch nothing else. The two citation rows in `backtest.md`'s
"Conventions consulted" table were already replaced by the driver
(IBKR knowledge base, MIAX penny program); leave them. Do not commit.

## Part A: the engine checks the join at every append (finding 1, High)

`record_order!` records observations without interpreting them: the
price rule lives in the venue, which the ledger module cannot see. So
the check that a fill's price is the rule applied to its observation,
and that the observation was taken at or before the decision, is the
engine's, and it must run when the order is appended, not once at the
end of the run.

Add a per-record form of `check_join`:

```julia
check_join(L::Ledger, rec::OrderRecord; tick_cents::Int = 1) -> Nothing
```

It checks one record and its fills: `length(rec.observations) ==
length(rec.order.legs)` (`:observations`); then every `Fill` whose
`sequence` is greater than `rec.known_to` and whose `order_leg_id` lies
in `rec.first_leg_id : rec.first_leg_id + n - 1` (scan `L.events` from
the end backwards and stop at the first event with sequence at or below
`known_to`; every event of the order has a later sequence, so the scan
is bounded by the tick), with exactly the per-fill checks the
whole-ledger form makes: contract, side and intent equal the leg's
(`:contract`, `:side`, `:intent`); group equals the record's (`:group`);
the fill's quantity is at most the leg's (`:quantity`; the cumulative
check across many fills of one leg stays in the whole-ledger form);
under any rule but `:broker_execution`: `quote_at` and `spot_at` at or
before `decided_at`, a known rule, the required side present, and
`price == fill_price(rule, bid, ask, side, tick_cents)`. Share one
private per-fill checker between the two forms so the rules cannot
drift.

In `run_backtest`, call `check_join(L, rec; tick_cents)` on the record
`record_order!` returns, inside the loop, right after the call. Remove
the whole-ledger `check_join(L; tick_cents)` at the end of the run: it
is now redundant and, through `order_leg`'s linear scan, quadratic in
the number of orders. The whole-ledger form stays for `load_run`,
`save_run` and the tests. Update the loop's docstring and comments.

## Part B: `save_run` checks the join before it writes (finding 2, High)

`save_run` runs `check_join(result.ledger)` before `mkpath` and before
any file is written, so a ledger that fails the join writes no folder.
The tick is the default until slice 4 stores it. Docstring updated;
`persistence.md` says the load path *and the write path* validate.

## Part C: the record is built before `commit!` (finding 3, Medium)

In `record_order!`, construct the `OrderRecord` (including
`collect(LegObservation, observations)`) before `commit!`; after
`commit!` only `push!(L.orders, record)` and the three counter
increments remain. Reword the docstring and both module docs
(`ledger.md` "Invariants" and "Key decisions", `backtest.md` header
and "Key decisions"): after `commit!` nothing validates and nothing is
allocated beyond the growth of one vector, so short of the process
running out of memory, which would interrupt `commit!` itself just as
well, the record and the events land together. Do not claim "nothing
can fail".

## Part D: broker-execution wording (finding 4, Low)

`backtest.md`, the venue section and `check_join` prose: this slice
keeps one `LegObservation` per order leg under every fill rule; under
`:broker_execution` its quote sides may be `missing` and the observation
is not consulted. A record with no observation at all is the live
adapter's shape and is deferred with it. Say the same in `ledger.md`
where `LegObservation` is introduced, in one sentence.

## Part E: leftovers

- `src/metrics/ledger_series.jl`: the docstring still says samples are
  ordered "exactly as `pnl_series(positions)` orders them"; the
  positions builder is gone. Say the order is the series' own rule,
  `(timestamp, pnl)`, losses first within an instant.
- `test/metrics/test_ledger_series.jl`: the testset named
  "pnl_series(ledger): ordering matches pnl_series(positions)" is
  renamed to say the canonical `(timestamp, pnl)` order; assertions
  unchanged.

## Tests

Literals as before, whole cents, arithmetic in comments.

`test/backtest/test_engine.jl`:

1. **`check_join(L, rec)` passes on what the writer built**: each record
   of fixture cases 7 and 8, and each record of the engine's
   open-then-close run.
2. **It finds only its own fills**: two orders in one tick share
   `known_to == 0` (build them with `record_order!` passing
   `known_to = 0` for both, as the engine does); checking the second
   record does not touch the first's fills, and a violation planted on
   the first record's observations (ask 0.86 against the 0.85 fill)
   fails the first record's check and not the second's.
3. **Every violation the whole form names, through the per-record
   form**, reusing `_en_with_record` and `_en_with_fill`: `:observations`,
   `:contract`, `:side`, `:intent`, `:group`, `:quantity` (a fill of 2 on
   a leg of 1, built by hand), `:quote_at`, `:spot_at`, `:fill_rule`,
   `:bid` / `:ask`, `:price`; a `:broker_execution` fill with a `missing`
   ask passes; the failing check leaves the ledger and book untouched
   (it is read-only) and prints its name.
4. **The loop checks at the append**: `run_backtest` on the existing
   fixtures still passes the whole-ledger `check_join` afterwards, and
   the engine's docstring names the per-record check; there is no way
   to make the engine's own `fill_legs` disagree with its observations,
   so the per-record function carries the tests.

`test/persistence/test_store.jl`:

5. **`save_run` refuses a ledger that fails the join before writing**:
   on the smoke result, replace the first order record's observations
   with `[_lg_seen(0.86), _lg_seen(1.10)]` (a local copy of the record
   helper); `save_run` throws `JoinViolation` with field `:price`, and
   `run_dir(store, full_hash(exp))` does not exist afterwards; empty a
   record's observations and it throws with `:observations`, still
   writing nothing. The existing "lost a leg" load test stays as it is.

`test/ledger/test_append.jl`: no new testset is required; the existing
"a structure lands whole or not at all, whichever leg fails" covers
part C's failure side.

## How to run here

As in the slice 2 brief: one suite at a time in a fresh process with
`JULIA_NUM_PRECOMPILE_TASKS=1`; the gate is `Pkg.test()`, run as
`ws run "JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=. -e 'using Pkg; Pkg.test()'"`
(`ws test` sets no precompile variable); wait for the julia process to
exit, then `ws capture shell 60`. Check `free -m` first; under about
1.3 GB available exit the REPL in the julia window with
`ws repl "exit()"`, say so, and relaunch it afterwards
(`julia --project=. -e 'using Revise' -i`, then `using VolSurfaceAnalysis`).

## Done means

- The full gate is green with exactly two Broken (the slice 3
  placeholders); report the counts.
- `run_backtest` calls the per-record `check_join` inside the loop and
  nothing after it; `save_run` calls the whole-ledger form before
  writing; `record_order!` builds the record before `commit!`.
- Both module docs no longer claim that nothing after `commit!` can
  fail; the broker-execution sentence is as in part D.
- No file outside the scope list changed. Nothing committed.
- A report: what changed per part, the gate's last lines, anything
  found wrong in this brief or the review.
