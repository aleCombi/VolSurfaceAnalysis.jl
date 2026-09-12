# Slice 1 fix brief: close the review findings

An implementation brief for the fix round on slice 1 of
[ledger.md](ledger.md). The proposal is binding where this brief is
silent; this brief is the contract for one round of code. Done is
defined by a test file that already exists and is red.

## Read first

1. [docs/design.md](../design.md), rules 1, 4, 5, 6 and 7.
2. [docs/proposals/ledger.md](ledger.md), sections 2 and 3, and the
   "Contract, venue, simplifications" subsection as revised on
   2026-09-12 (`GuaranteedCombo`, per-contract commissions).
3. [ledger-slice1.md](ledger-slice1.md), what slice 1 was asked to
   build, and [ledger-slice1-review.md](ledger-slice1-review.md), what
   codex found. Finding numbers below (6.1 .. 7.4) follow the driver's
   walkthrough, which grouped codex's four (6.1 .. 6.4) and the driver's
   four (7.1 .. 7.4); the test file's header comment carries the same
   numbering.
4. `test/ledger/test_review_findings.jl`: eight testsets, red today,
   one per finding. Read its header comment. Seven of them turning green
   is the definition of done; the eighth is handled in section E below.
5. `src/ledger/*.jl` and `src/metrics/ledger_series.jl` as they are.
   The changes below name the functions they touch.

## Decisions already taken; do not reopen

- Cash is an integer number of USD cents everywhere inside the ledger.
- A fill exactly at its contract's expiry instant stays allowed (`<=`).
- The rejected batch shapes use the existing named errors that the
  review tests expect (`MatchMismatch`, `ExceedsOpen`), with a reason in
  the message. One new named error, `NonIntegralCash`, is added.
- `record_order!` is slice 2 work. Its testset becomes known-broken,
  not green.
- `test/metrics/test_ledger_series.jl` stays where it is.

## Scope

Edit only:

- `src/ledger/contracts.jl`, `types.jl`, `cash.jl`, `book.jl`,
  `append.jl`, `round_trips.jl`; `src/metrics/ledger_series.jl`.
- `src/VolSurfaceAnalysis.jl`: only to export `NonIntegralCash`.
- `test/ledger/*.jl`: literals to cents; the one test that asserts the
  removed tie-break deleted; the review tests only as section E allows;
  the new tests in section G added to the suite they belong to.
- `test/metrics/test_ledger_series.jl`: expected to need no change (the
  adapter still returns dollars); if it does, explain why in the report.
- `docs/modules/ledger.md`, `docs/status.md`, and in
  `docs/proposals/ledger.md` decision 1 of section 3 only.

Do not touch: the engine, policies, experiment, persistence, positions,
any other metrics file, `test/runtests.jl`. Do not add `record_order!`.
Nothing is committed; commits are the human's.

## The changes

### A. `commit!` loses its spec argument (6.1)

- The signature becomes `commit!(L, book, batch)`. It applies each
  event with the one-argument `apply!`, which resolves the contract
  spec from the table per event. Delete the four-argument method; no
  deprecated alias.
- `record_fill!`, `record_expiry!` and `record_fee!` stop resolving a
  spec.
- The two-argument `cash(e, spec)` and `apply!(book, e, spec)` stay, for
  tests that pin a multiplier without the table.
- Existing tests that call `commit!(L, book, batch, _LG_SPEC)` move to
  the three-argument form. They all use the table's SPY spec, so no
  number changes from this step alone. `_lg_commit_review!` in the
  review tests already handles both forms.

### B. References point backward in effective time; the effective replay folds in journal order at an equal instant (6.2, 7.1)

In `_validate`:

- For a `Match`, the opening fill's effective time must be at or before
  the closing fill's, and the match's own effective time must equal the
  closing fill's. Otherwise `MatchMismatch(id, reason)`.
- For an `Expiry`, the opening fill's effective time must be at or
  before the expiry's. Otherwise `MatchMismatch`.
- For a `Fee`, the source fill's effective time must be at or before
  the fee's. Otherwise `MatchMismatch`.

Extend `MatchMismatch`'s docstring: it also names an event whose
effective time precedes an event it references.

In `book.jl`: `book_effective` sorts by `(effective_at, sequence)`.
Delete `_priority`. With the checks above, nothing can reference an
event that folds later, so the fold cannot meet a missing lot. Delete
the testset "book: lifecycle sorts before fills at an equal effective
time" in `test/ledger/test_book.jl`; it asserts the removed rule, and
review testset 2 covers the behaviour that replaces it.

### C. FIFO checked at validation (6.3)

In `_check_match`: the opening fill a match names must be the first
eligible lot in FIFO order. Eligible means: same group and contract,
opposite side to the closing fill, and remaining quantity greater than
zero after subtracting what earlier matches in this batch already
consumed (`consumed`). Order: the book's lots for that `(group,
contract)` in their vector order, then fills opened earlier in this
batch (`opened`) in batch order. If the named lot is not the first
eligible one, `MatchMismatch(id, "match skips an older open lot")`.

### D. An expiry settles the whole remaining lot, at or after the contract's expiry (7.2, 7.3)

In `_validate`'s `Expiry` branch, computing `available` the way
`_consume_check!` does:

- `quantity > available` is `ExceedsOpen`, as today.
- `quantity < available` is `MatchMismatch(id, "expiry of q leaves r
  open")`.
- `effective_at(e) < e.contract.expiry` is `MatchMismatch(id, "expiry
  effective before the contract's expiry")`.

`record_expiry!` already passes `lot.remaining`, so the writer is
unaffected.

### E. Cash in whole cents (7.4 and the decision)

One rounding point, then integer arithmetic everywhere.

- `ContractSpec.multiplier::Int`; the table entries become `100`.
- New in `cash.jl`: `contract_cents(price::Real, spec)::Int`, the cash
  per contract for a per-share `price`: `round(Int, price *
  spec.multiplier * 100)`, refused with `NonIntegralCash(value)` when
  the exact product is more than `1e-6` from the nearest integer.
  `round(Int, x)` is Julia's default, ties to even; say so in a comment.
- `cash(e, spec)::Int`: `Fill` is `-side_sign(side) *
  contract_cents(price, spec) * quantity`; `Expiry` is `side_sign(side)
  * contract_cents(intrinsic(contract, settlement_price), spec) *
  quantity`; `Match` is `0`; `Fee` is its amount. One-argument forms
  unchanged in shape.
- `NonIntegralCash <: Exception`, defined with the other named failures
  in `append.jl`, with a `showerror`, exported. `_validate` calls
  `cash(e)` on every event of the batch so the failure fires before
  anything lands.
- `Fee.amount::Int` cents; `record_fee!(L, book, source_id,
  amount::Integer; ...)`.
- `Book.cash::Int`, starting at `0`. `apply!` adds integers.
- `Fill.price`, `Lot.unit_price` and `Expiry.settlement_price` stay
  `Float64` per share.
- `RoundTrip.pnl::Int` cents. Price part of a closed trip:
  `side_sign(o.side) * (contract_cents(c.price) - contract_cents(o.price))
  * quantity`; of an expired trip the same with
  `contract_cents(intrinsic)` in place of the closing price. Both are
  integer by construction, so trips reconcile to cash exactly.
- Fee shares in whole cents by cumulative rounding. For each source
  fill with total fee `F` cents and quantity `Q`, walk the trips that
  consume it in sequence order, keeping the cumulative consumed
  quantity `c_k`; the share of trip `k` is `round(Int, F * c_k / Q) -
  round(Int, F * c_{k-1} / Q)`. The shares over the trips that fully
  consume the fill sum to `F` exactly; a fill left partly open leaves
  the remainder unallocated, as today.
- `pnl_series(::Ledger)`: `pnl` becomes `trip.pnl / 100` as `Float64`
  dollars, so `PnLSeries` and every existing metric are untouched.
- Book equality is now exact integer equality; review testset 8 passes
  without edits.
- Tests: convert every cash literal to cents by hand and write the
  arithmetic in a comment. `≈` on cash becomes `==`. Re-derive fee
  shares with cumulative rounding. For fixture case 5 (a fee of -130 on
  a close of 3 that consumed lots of 2 and 1): shares are `round(-130
  * 2/3) = -87` and `-130 - (-87) = -43`; trips `9000 - 87 = 8913` and
  `5000 - 43 = 4957`; they sum to `13870`, the book's cash. Re-derive
  the opening-fee test in `test_round_trips.jl` the same way.
- In `test/ledger/test_review_findings.jl` two edits are allowed and no
  others: testset 2's `book.cash == -115.0` becomes `== -11500`; and
  testset 4 becomes known-broken, exactly:

```julia
@testset "ledger promise: a structure lands whole or not at all" begin
    # Known broken until slice 2 adds record_order!; flip @test_broken to @test then.
    L, book = Ledger(), Book()
    before_group = L.next_group
    order = Order(:invalid_structure, [
        Leg(_LG_PUT470, Short, 1, Open),
        Leg(_LG_CALL490, Long, 1, Close),
    ])
    @test_broken begin
        threw_right = try
            record_order!(L, book, order; prices=[0.85, 0.40], effective_at=_LG_T_OPEN,
                          recorded_at=_LG_T_OPEN, order_leg_ids=[1, 2], fill_rule=:cross_spread)
            false                                   # it must throw
        catch e
            e isa NothingToClose
        end
        threw_right && length(L) == 0 && book == Book() && L.next_group == before_group
    end
end
```

Today the undefined name is caught, the expression is `false`, and the
test records as Broken. When slice 2 lands the writer it records as an
unexpected pass, which is the signal to flip it.

### F. Docs

- `docs/modules/ledger.md`: the cash rules table in cents with the one
  rounding point named; the invariants list gains the effective-time
  ordering, whole-lot expiry at or after contract expiry, FIFO checked
  on append, and `NonIntegralCash`; the replay paragraph loses
  "expiries first at an equal instant" and says equal instants fold in
  sequence order; the equality invariant is exact by construction.
  Drop the Layout section (rule 6). Leave the conventions table alone;
  it was revised on 2026-09-12.
- `docs/status.md`: the in-flight entry says the fix round landed, the
  gate is green with one known-broken testset waiting for slice 2, and
  slice 2 is next.
- `docs/proposals/ledger.md`, section 3, decision 1: "cash in USD"
  becomes "cash in whole USD cents".

### G. New tests

Add to the suite of the code they exercise:

1. `NonIntegralCash`: a hand-built fill priced at `0.123456` on SPY
   (`1234.56` cents per contract) is refused at commit; ledger counters
   and book unchanged.
2. Cumulative rounding: a fee of `-100` on an opening fill of 3,
   consumed by three closes of 1: shares `-33, -34, -33` (cumulative
   `round(-33.33) = -33`, `round(-66.67) = -67`, `round(-100) =
   -100`), sum `-100`.
3. Reconciliation exact: for every fixture case with nothing left open,
   `sum(pnl) == book.cash` with `==`, not `≈`.

## How to run here

- Fast iteration, one suite at a time in a fresh process (about ten
  seconds once precompiled):

```
JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=. -e 'using VolSurfaceAnalysis, Test, Dates; include("test/ledger/fixtures.jl"); @testset "one" begin include("test/ledger/test_append.jl") end' 2>&1 | tail -40
```

- The `julia` tmux window holds a REPL with Revise; `ws repl <code>`
  sends to it and a disk edit is live on the next call. Fixtures define
  constants, so prefer the fresh-process command above for running
  suites.
- The gate is `Pkg.test()` in the shell window: `ws test`. Do not wait
  on echoed text (`ws wait` false-matches the command); wait for the
  julia process to exit, then `ws capture shell 40`. The box has 3.7 GB
  and two cores: check `free -m` first, and if less than about 1.2 GB
  is available, exit the REPL with `ws repl "exit()"` before the gate
  and say so in your report. Set `JULIA_NUM_PRECOMPILE_TASKS=1`.

## Done means

- The full gate is green: every existing test, the review tests, and
  the new ones, with exactly one Broken result, the atomicity testset.
  Report the pass count.
- `docs/modules/ledger.md`, `docs/status.md` and the proposal's decision
  1 updated as in F.
- No file outside the scope list changed. Nothing committed.
- A short report: what landed, the counts, every literal that changed
  and its arithmetic, anything in this brief or the proposal you had to
  interpret or found wrong.
