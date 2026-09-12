# Slice 1 hardening brief: every reviewed issue has a test, every promise is enforced

An implementation brief for a hardening round on the `ledger` module.
The goal is stated by the human: every issue any review has raised must
be verified by a test, tests must live where they belong, and every
invariant and named failure the module documents must be enforced by
code and covered by a test. Where the code does not yet enforce a
documented promise, enforce it.

## Read first

1. [docs/design.md](../design.md), all rules; rule 7 especially.
2. [docs/proposals/ledger.md](ledger.md), sections 2 and 3; the
   "Invariants" subsection is binding.
3. [docs/modules/ledger.md](../modules/ledger.md): its "Invariants"
   section is the list of what the module promises today, and its
   named failures are the vocabulary.
4. Every review so far, in this order:
   - [ledger-events-review.md](ledger-events-review.md), section
     "Ordering and identity invariants" (six numbered items) and the
     two-replay definitions.
   - [ledger-fill-review.md](ledger-fill-review.md), section "Enforced
     invariants": the construction-time checks apply now; the seven
     cross-record checks are slice 2 and are only to be listed as
     deferred.
   - [ledger-slice1-review.md](ledger-slice1-review.md), especially its
     "Invariant and append-validation coverage" table and its "Open
     interpretations" list.
   - [ledger-slice1-fix-review.md](ledger-slice1-fix-review.md).
   - The driver's walkthrough, sections 6 and 7:
     `/tmp/claude-1000/-home-ale-dev-VolSurfaceAnalysis-jl/37dddf7c-1c03-4ade-bc64-5f23a21d535c/scratchpad/ledger-slice1-walkthrough.md`.
   - [ledger-slice1-fix.md](ledger-slice1-fix.md), the round that closed
     most of them.
5. `src/ledger/*.jl`, `src/metrics/ledger_series.jl`, `test/ledger/*.jl`,
   `test/metrics/test_ledger_series.jl` as they stand.

## Scope

Edit only:

- `src/ledger/*.jl`: only to enforce a documented invariant, to add a
   named failure, or to replace an unnamed error with a named one.
- `src/VolSurfaceAnalysis.jl`: exports of new named failures only.
- `test/ledger/*.jl`, including deleting `test_review_findings.jl`.
- `test/runtests.jl`: only to remove the include of
  `ledger/test_review_findings.jl`.
- `docs/modules/ledger.md` (invariants and named failures),
  `docs/status.md`, and a new `docs/proposals/ledger-slice1-coverage.md`.

Do not touch: the engine, policies, experiment, persistence, positions,
any metrics source, `test/metrics/test_ledger_series.jl` (it sits beside
the source it tests, which is where it belongs). Do not add
`record_order!` or any slice 2 behaviour. Nothing is committed.

## Part 1: the inventory

Build one table of every issue, invariant and named failure that the
documents above state and that applies to the ledger module as it
stands. One row per statement. Columns: the statement; where it is
stated; the code that enforces it (file and function) or "not
enforced"; the test that covers it (file and testset) or "none"; the
action this round took. Items that belong to slice 2 (order-leg joins,
execution-id uniqueness and idempotency, fill-rule and observation
validation, the order journal, cross-journal atomicity, `record_order!`)
go in a second, shorter table headed "deferred to slice 2" so nothing is
lost. Write both to `docs/proposals/ledger-slice1-coverage.md`. This is
the map for parts 2 to 4 and the document the human reads first.

## Part 2: tests where they belong

The rule: one test file per source file, in the mirrored folder, and a
test lives beside the behaviour it checks. Dissolve
`test/ledger/test_review_findings.jl`:

- replay-equality tests (spec, equal-instant expiry, exact equality) go
  to `test_book.jl`;
- rejection tests (FIFO, close before open, partial expiry, early
  expiry) go to `test_append.jl`;
- the `@test_broken` structure-atomicity test goes to the end of
  `test_append.jl` with its comment intact.

Keep every assertion and every name that states a promise. Drop the
`_lg_commit_review!` helper and call the three-argument `commit!`
directly. Delete the file and its include line.

## Part 3: coverage

For every inventory row without a test, add one in the suite where it
belongs, with hand-computed literals and the arithmetic in a comment.
For every named failure, three things are tested: it fires on its
condition; after the failed batch `_lg_snapshot(L)` and the book are
unchanged; it prints its name. Gaps already known, to include:

- FIFO with two lots opened in the same batch and several matches
  consuming across both, in one batch (fix review).
- Id and sequence as separate counters, positively exercised: build a
  `Ledger` whose `next_id` starts at 100 and `next_sequence` at 1
  (direct struct construction), commit through the writers, and check
  `event(L, id)` by id, `book_as_known` by sequence, and references by
  id all work. Today nothing makes the two diverge, so the promise is
  untested.
- Every fixture case: `book_as_known(L, k)` equals folding the first
  `k` events, for every `k`, not only the last.
- Every fixture case, after every event: no lot with `remaining <= 0`;
  every lot sits under the key of its own group and contract; lots
  within a key are in opening order. Write one `_lg_check_book` helper
  and apply it everywhere.
- Partial reconciliation on every fixture, including those with lots
  left open: the trips' pnl, plus the opening cash still tied up in
  open lots (`-side_sign(side) * contract_cents(unit_price) *
  remaining` per lot), plus the fee remainder not yet allocated (per
  fill with a fee, `F - round(F * consumed // Q)`), equals `book.cash`
  exactly.
- `NothingToClose` when the group holds only same-side lots on that
  contract.
- `DanglingReference` for an expiry naming a `Match` id, and for a fee
  naming an `Expiry` id.
- `UnknownContract` through `commit!`: a hand-built fill on an unlisted
  underlying is refused before anything lands.
- `book_effective` at an instant before the first event is an empty
  book.

## Part 4: enforcement

Where the inventory shows a documented promise with no enforcing code,
add the check, in `_validate` or in a constructor (so that no such value
exists), with a named failure, and add the tests from part 3. Known
cases, all to be done:

- An `Expiry`'s `outcome` must agree with its intrinsic value:
  `Worthless` when intrinsic is zero, `CashSettled` otherwise. Copied
  and derived fields are checked on append; this one is not.
  `MatchMismatch` with a reason.
- Every event's `recorded_at` is at or after its `effective_at`: a fact
  cannot be recorded before it is true. And `recorded_at` is
  nondecreasing along sequence, across the batch boundary and within a
  batch. New named failure, suggested `RecordedOutOfOrder(id,
  recorded_at, bound)`. These two are rule additions (see below).
- Prices: `Fill.price` must be finite and positive;
  `Expiry.settlement_price` finite and non-negative. Today a
  non-positive fill price is an unnamed `ArgumentError` and the rest is
  unchecked until `NonIntegralCash`. Replace with one named failure,
  suggested `InvalidPrice(value)`, thrown by the constructors.
- `event(L, id)` for an id the ledger never minted is a bare `KeyError`;
  make it `DanglingReference(:event_id, id)`.

Rule additions. The `recorded_at` checks and any other check that is
not already written in the proposal or the module doc are new rules.
Implement them, since the human asked for correctness well enforced,
but keep each to one check, one test and one line in the module doc,
list them under a "Rule additions" heading in the coverage document
and in your report, and say in one sentence each why the rule is right,
so the human can veto any of them cheaply.

## How to run here

- One suite at a time in a fresh process, about ten seconds once
  precompiled:

```
JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=. -e 'using VolSurfaceAnalysis, Test, Dates; include("test/ledger/fixtures.jl"); @testset "one" begin include("test/ledger/test_append.jl") end' 2>&1 | tail -40
```

- The gate is `Pkg.test()` in the shell window: `ws test`. Wait for
  the julia process to exit rather than for echoed text, then
  `ws capture shell 40`. The box has 3.7 GB and two cores: check
  `free -m`; under about 1.3 GB available, exit the REPL in the julia
  window with `ws repl "exit()"` first and say so in your report, then
  relaunch it afterwards (`julia --project=. -e 'using Revise' -i`, then
  `using VolSurfaceAnalysis`). Always set `JULIA_NUM_PRECOMPILE_TASKS=1`.

## Done means

- The coverage document exists; every row has a test or a stated
  reason; rule additions are listed.
- `test_review_findings.jl` is gone and its tests live in the right
  suites; the include line is gone.
- The full gate is green with exactly one Broken, the structure
  atomicity test. Report the counts.
- `docs/modules/ledger.md` invariants and named failures match the
  code; `docs/status.md` updated.
- No file outside the scope list changed. Nothing committed.
- A report: what was unenforced and what you did about it; every rule
  addition with its one-sentence reason; every new named failure; the
  deferred-to-slice-2 list; anything in this brief or the docs you had
  to interpret or found wrong.
