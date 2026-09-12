# Slice 1 review: ledger

## Findings, ranked by severity

### High: a successfully committed ledger need not replay to its incremental book

`commit!` applies every event with the caller-supplied `spec` (`src/ledger/append.jl:221-234`), but both replay functions call one-argument `apply!`, which resolves the table spec (`src/ledger/book.jl:137-139,150-155,172-179`). Committing a SPY fill with a pinned multiplier of 10 therefore puts 8.50 in the incremental book and 85.00 in either replay. This violates the binding cash/replay invariant (`docs/proposals/ledger.md:229-231`) through a public API expressly supplied for pinned-spec tests (`docs/proposals/ledger-slice1.md:195,231`). Remove the `spec` argument from `commit!` and always resolve contract facts, or make replay receive the same resolver; add a non-100 multiplier reconciliation test.

### High: an append-valid ledger can fail `book_effective`

A fill exactly at expiry is explicitly accepted and tested (`test/ledger/test_append.jl:128-135`). A later-sequence expiry at that same instant is also append-valid, but effective replay sorts the expiry before the opening fill (`src/ledger/book.jl:158-179`), so `_consume!` throws `DanglingReference` because the lot does not exist yet (`src/ledger/book.jl:81-99`). The brief requires both “fill at or before expiry” and lifecycle-first equal-time replay (`docs/proposals/ledger-slice1.md:205-210,246-251`), making its own valid state space inconsistent. Resolve the contract now: either fills must be strictly before expiry, or equal-time priority needs a dependency-safe exception. Test the chosen boundary end to end, not only `_priority` (`test/ledger/test_book.jl:116-121`).

### Medium: `commit!` does not enforce FIFO

`record_fill!` emits matches in the book vector's FIFO order (`src/ledger/append.jl:149-162`), and the ordinary-path test confirms it (`test/ledger/test_append.jl:44-50`). But the single validated write path accepts a hand-built close whose matches consume a newer lot before an older one: `_check_match` checks group, contract, opposite side, and capacity, never FIFO (`src/ledger/append.jl:272-279`). That contradicts the documented invariant (`docs/modules/ledger.md:92-96`) and allows loaded/broker-built batches to encode a different accounting rule. Validate each match against the oldest remaining eligible lot and add a rejection test.

### Medium: slice 2 cannot obtain structure-level atomicity from the writer API as landed

`record_fill!` commits one leg immediately (`src/ledger/append.jl:141-165`). The proposal requires every leg of an order to be validated before the order journal and all ledger events are written as one all-or-none batch (`docs/proposals/ledger.md:157-164`). Slice 2 therefore needs a batch planner/builder plus one transaction spanning the order journal and ledger; calling this API once per leg would permit partial structures. Settle that API before wiring the engine.

### Low: the conventions record is not traceable enough for design rule 5

The section exists, but entries such as “Double-entry practice”, “US broker tickets”, “Broker statements”, and “Fund reporting” do not identify a publication or link that can be checked (`docs/modules/ledger.md:128-141`). Replace these with specific sources (and preferably links/titles); retain the Julia Manual and OCC rule/spec citations but make their exact pages identifiable. Architecturally the doc otherwise states boundaries and invariants well (`docs/modules/ledger.md:73-111`), avoids a usage walkthrough, and is mostly compliant with rule 6; the file-by-file layout is mildly drift-prone (`docs/modules/ledger.md:143-159`) and should be dropped.

### Low: the working tree contains out-of-scope changes

The adapter test is under `test/metrics/test_ledger_series.jl` and included there (`test/runtests.jl:35-36`), while the scope says all new slice tests are under `test/ledger/` (`docs/proposals/ledger-slice1.md:31-35,299-302`). Move it under `test/ledger/`. `git status --short` also reports a tracked edit to `docs/proposals/ledger.md` and untracked `docs/proposals/ledger-events-review.md`, `ledger-fill-review.md`, `ledger-orchestration.md`, and `ledger-slice1.md`, none in the slice scope. If those are pre-existing review inputs, exclude them from the slice; otherwise the done condition is not met. No engine, policy, experiment, persistence, or existing metric source was edited.

## Correctness results

- Cash formulas are exact (`src/ledger/cash.jl:33-42`): the literals +85-40=45, split-lot cash 170+90-120=140 and trip PnLs 90/50, expiry PnL 85-200=-115, and fee split 90-1.30(2/3)=89.1333 and 50-1.30(1/3)=49.5667 are all arithmetically correct (`test/ledger/fixtures.jl:32-50`; `test/ledger/test_round_trips.jl:15-24,34-52`). The opening-fee literals 45-0.30=44.70 and 110-0.60=109.40 are also correct (`test/ledger/test_round_trips.jl:55-66`).
- Opposite-side matching is correctly restricted to `(group, contract)` and ordinary writes are FIFO (`src/ledger/append.jl:149-162`); over-close and empty-close select the intended errors (`src/ledger/append.jl:151-154`). The validation hole above is the exception.
- `book_as_known` correctly cuts by sequence (`src/ledger/book.jl:150-155`); `book_effective` correctly sorts `(effective_at, lifecycle priority, sequence)` (`src/ledger/book.jl:158-179`) except for the equal-expiry contradiction above. The bitemporal test otherwise has the right expected cash, 235 before and 35 after a short put settles for -200 (`test/ledger/test_book.jl:80-113`).
- `round_trips` emits one row per `Match`/`Expiry` in journal sequence and allocates opening and closing fill fees pro rata (`src/ledger/round_trips.jl:44-70`). Its formulas and full-close reconciliation are correct under the table spec (`test/ledger/test_round_trips.jl:79-92`).
- The adapter implements structure `(group, closed_at)` aggregation, leg rows, counts, placeholders, and final `(timestamp,pnl)` ordering exactly (`src/metrics/ledger_series.jl:21-50`). Its literals -50/+45, profit factor 0.9, total -5, and combined legs -15+50=35 are correct (`test/metrics/test_ledger_series.jl:31-60`). Empty output is legitimate temporal/no-realisation output, not a hidden failure (`test/metrics/test_ledger_series.jl:76-81`), so design rule 7 is respected.

## Invariant and append-validation coverage

| Requirement | Code | Test | Assessment |
|---|---:|---:|---|
| Sequence is contiguous/replay order | `src/ledger/append.jl:283-289` | `test/ledger/test_append.jl:140-154` | both |
| IDs stable, distinct from index/sequence | `src/ledger/types.jl:209-230`; `src/ledger/append.jl:285-289` | `test/ledger/test_append.jl:19-31,140-153` | both, though divergence is not positively exercised |
| Effective time may be non-monotone | `src/ledger/book.jl:172-179` | `test/ledger/test_book.jl:100-113` | both |
| Every reference points backward and has the right kind | `src/ledger/append.jl:242-258,317-329` | `test/ledger/test_append.jl:157-173` | both |
| Quantities are positive integers | `src/ledger/types.jl:58-61,121-128,146-150,171-177` | `test/ledger/test_types.jl:23-50` | both |
| Close stays within group/contract/opposite side | `src/ledger/append.jl:272-279` | `test/ledger/test_append.jl:193-201` | both |
| Matches immediately follow and exhaust close | `src/ledger/append.jl:305-318` | `test/ledger/test_append.jl:185-203` | both |
| Consumption never exceeds remaining | `src/ledger/append.jl:262-269` | `test/ledger/test_append.jl:215-228` | both |
| No fill after expiry | `src/ledger/append.jl:297-299` | `test/ledger/test_append.jl:128-137` | both; boundary is inconsistent with replay |
| Replay cash=sum(event cash); incremental=full replay | `src/ledger/book.jl:112-139,150-179` | `test/ledger/test_book.jl:124-135` | tested only for table specs; false for public pinned-spec commit |
| Every fill joins exactly one order leg | only an unchecked integer field at `src/ledger/types.jl:109-128` | none | neither; intentionally deferred, but mandatory in slice 2 |
| Expiry copies opening group/contract/side | `src/ledger/append.jl:319-324` | `test/ledger/test_append.jl:204-208` | both |
| FIFO matching | writer only at `src/ledger/append.jl:149-162` | ordinary path at `test/ledger/test_append.jl:44-50` | not enforced at append |

All named brief errors do fire on their covered conditions: `NothingToClose`/`ExceedsOpen` (`src/ledger/append.jl:150-154,262-269`), `FillAfterExpiry` (298-299), `DanglingReference` (248-257), `MatchMismatch` (272-279,313-324), `SequenceGap` (283-289), `NonPositiveQuantity` (`src/ledger/types.jl:58-60,124-125,148-149,174-175`), and `UnknownContract` (`src/ledger/contracts.jl:65-68`). No listed error is missing for a stated append check; FIFO needs either `MatchMismatch` or a new named error. Validation precedes mutation (`src/ledger/append.jl:223-234`), and the failure snapshots cover ledger counters and book state (`test/ledger/test_append.jl:75-84,140-154,176-228`), so failed batches leave them untouched for the implemented checks. `mint_group!` is a separate successful counter mutation (`src/ledger/append.jl:118-122`), not rolled back when later order validation fails; slice 2 must decide whether group allocation belongs inside the all-or-none transaction.

## Scope and public surface

All brief-listed public names are present and exported (`src/VolSurfaceAnalysis.jl:72-87`); `pnl_series` was already exported (`src/VolSurfaceAnalysis.jl:92`). Includes are in the required dependency order (`src/VolSurfaceAnalysis.jl:24-37`). The new ledger/adapter files contain no `OptionQuote`, `SpotPrice`, `TimeCut`, or `market_data` reference, and there is no `using Base.Order`; the comment explicitly records the permitted shadowing (`src/ledger/types.jl:79-83`).

## Open interpretations and slice-2 decisions

- FIFO is interpreted as append/sequence order, because lots are pushed in replay order (`src/ledger/book.jl:112-116`) rather than sorted by effective time. I agree: sequence is the authoritative journal order, but the proposal should say so.
- Structure sampling groups by `(group, closed_at)` and therefore splits one group closed at different instants (`src/metrics/ledger_series.jl:25-35`). I agree; this is the brief's literal rule, but slice 2 must ensure all legs of an all-or-none structure share one effective instant.
- Fees are discovered across the complete ledger, including fees appended after their source (`src/ledger/round_trips.jl:44-49`). I agree for an eventual derived table; callers must not mistake it for an as-of derivation.
- Slice 2 must add order-leg join validation, ordered-quantity limits, execution-id uniqueness/idempotency, and fill-rule/observation validation; none can be expressed by this ledger alone (`src/ledger/types.jl:103-128`; proposal requirements at `docs/proposals/ledger.md:117-139`). Define ownership of execution IDs too: accepting stale duplicates is currently deliberate (`test/ledger/test_append.jl:244-247`) and would be wrong once broker retries exist.
- The proposal is underspecified on atomicity across two journals and on rollback/allocation of group, event, sequence, and execution IDs. A prevalidation failure must leave both journals and all counters untouched; specify a planned transaction object rather than coordinating public mutators ad hoc.

**Verdict: not mergeable, because valid public writes can violate replay equality or make effective replay throw, and append validation does not enforce the documented FIFO rule.**
