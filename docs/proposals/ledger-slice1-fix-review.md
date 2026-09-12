# Slice 1 fix-round review: ledger

## Findings, ranked by severity

### Medium: the documentation diff exceeds the brief's explicit scope

The brief permits changing only decision 1 in `docs/proposals/ledger.md`, but
the working tree also rewrites the tick-order/venue discussion, decisions 10,
and proposal conventions (`docs/proposals/ledger.md:157-166,185-202,274-276,321-322`).
Those edits consistently express the separately revised `GuaranteedCombo` and
per-contract-commission decisions, so their content is right, but landing them
in this fix round violates the scope contract and obscures provenance. Either
move them to their own revision commit or amend the brief/scope explicitly.

The brief also says to leave the module's conventions table untouched, but it
was substantially rewritten (`docs/modules/ledger.md:156-169`). These changes
improve the traceability weakness from the slice-1 review and are substantively
right under design rule 5, but they are still a deliberate deviation from F.

### Low: `NonIntegralCash` does not name every invalid cash product

`contract_cents` calls `round(Int, value)` before it can throw
`NonIntegralCash` (`src/ledger/cash.jl:49-54`). A hand-built fill or expiry can
carry `NaN`, `Inf`, or a finite product outside `Int` range because prices remain
unrestricted `Float64`; those values raise `InexactError` instead. This is a new
edge in the stated invariant that every non-whole-cent event is refused by the
named error (`docs/modules/ledger.md:95-97`). Check `isfinite(value)` and the
representable range before conversion, or narrow the documented guarantee and
validate prices elsewhere. The required `0.123456` case is correctly raised
before ledger counters or the book mutate (`src/ledger/append.jl:354-365`;
`test/ledger/test_append.jl:268-285`).

## Correctness assessment

- A landed exactly: the only public write is `commit!(L, book, batch)`, writers
  no longer resolve a spec, and application uses table-resolving `apply!(book,e)`
  (`src/ledger/append.jl:162-181,197-223,234-261`). The explicit-spec `cash` and
  `apply!` test surfaces remain. The old four-argument call at
  `test/ledger/test_review_findings.jl:31` intentionally proves its removal by
  accepting `MethodError`; it is not an extant API assertion.
- B is correct: matches check open <= close and match == close, expiries check
  open <= expiry, and fees check source <= fee
  (`src/ledger/append.jl:316-328,337-341,399-404`). `MatchMismatch` documents all
  cases (`src/ledger/append.jl:58-72`). Effective replay sorts
  `(effective_at,sequence)` and `_priority` plus its old test are gone
  (`src/ledger/book.jl:159-176`; `test/ledger/test_book.jl:105-124`). Dependencies
  are safe: validation requires references to earlier journal events, and the
  effective inequalities ensure those dependencies sort no later; equal-time
  dependencies retain sequence order.
- C is correct for committed book lots and opens earlier in the same batch.
  `_available` subtracts earlier batch consumption; `_first_eligible` scans book
  vector order then batch-open order and skips fully consumed candidates; the
  selected id is enforced before consumption advances
  (`src/ledger/append.jl:284-312,329-331`). The tests cover ordinary and rejected
  FIFO plus a same-batch open/close, though no single test combines two
  batch-open lots with multiple earlier matches.
- D is correct: expiry rejects over-consumption as `ExceedsOpen`, under-
  consumption as `MatchMismatch`, requires effective time at or after contract
  expiry, and consumes exactly the available remainder
  (`src/ledger/append.jl:333-353`). Equality at expiry remains allowed.
- E's intended finite-value design is correct. `ContractSpec.multiplier`,
  `Fee.amount`, `Book.cash`, and `RoundTrip.pnl` are `Int`
  (`src/ledger/contracts.jl:33-38`; `src/ledger/types.jl:186-190`;
  `src/ledger/book.jl:33-40`; `src/ledger/round_trips.jl:12-24`). The sole money
  conversion is `contract_cents`, with a `1e-6`-cent noise tolerance and an
  accurate ties-to-even note (`src/ledger/cash.jl:34-54`). All downstream ledger
  arithmetic is integer. Fee shares use exact rationals and cumulative
  ties-to-even rounding, so a fully consumed fill receives exactly its fee
  (`src/ledger/round_trips.jl:51-73`). `pnl_series` sums cents first and divides
  once at the metrics boundary (`src/metrics/ledger_series.jl:23-52`). Book
  equality is exact (`src/ledger/book.jl:40`).
- Every changed test literal is arithmetically correct. In particular: 0.85,
  0.90, 0.40, 1.10, 1.20, 0.70, 0.60 map to 8500, 9000, 4000, 11000, 12000,
  7000, 6000 cents; intrinsic 2.00 maps to 20000; cases 1-6 produce 4500,
  14000, 16000, 3500, 13870, 13500. Fee `-130` splits `-87/-43`, giving
  `8913/4957`; opening fee `-90` splits `-30/-60`, giving `4470/10940`; fee
  `-100` splits `-33/-34/-33`, giving `4467/4466/4467`
  (`test/ledger/test_round_trips.jl:13-119`). No wrong literal found.
- The slice-1 review's three merge blockers (spec divergence, unsafe effective
  replay, and unenforced FIFO) are closed. Structure atomicity remains openly
  deferred to slice 2 by the exact required `@test_broken` block
  (`test/ledger/test_review_findings.jl:67-85`). Order-leg joins, execution-id
  uniqueness/idempotency, fill-rule validation, and cross-journal transaction
  ownership likewise remain slice-2 work, not new regressions.

## Brief coverage

| Section | Status | Evidence / assessment |
|---|---|---|
| A | landed | Three-argument write only; table resolution on apply. |
| B | landed | All effective-time checks; equal instants use sequence. |
| C | landed | FIFO includes book and batch-open lots, net of prior consumption. |
| D | landed | Whole remainder only; expiry lower bound enforced. |
| E | partial | Required cents behavior landed; non-finite/range failures escape the named error. |
| F | partial | Required cash/invariant/replay/status updates and Layout removal landed; conventions and extra proposal sections changed contrary to scope. |
| G | landed | Non-integral batch, cumulative fee, and exact fixture reconciliation tests added. |

Scope otherwise matches: only listed source/tests/docs are modified, plus the
allowed untracked `ledger-events-review.md`, `ledger-fill-review.md`, and the
brief. No metrics test changed. I did not rerun the memory-heavy full gate; the
reported result is 1673 passed, 1 broken, 0 failed, 0 errored. Focused fresh-
process checks passed: append 148/148 and round trips 63/63.

**Verdict: not mergeable until the documentation scope deviations are separated
or explicitly authorized and `NonIntegralCash` consistently names invalid cash
products; the requested finite-value ledger behavior itself is correct.**
