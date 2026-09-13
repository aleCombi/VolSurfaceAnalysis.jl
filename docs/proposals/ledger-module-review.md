**Verdict: do not merge as it stands.** The implementation is sound on this
review, additive, and ready for the wiring PR, but the landing module document
still does not document much of the exported surface and links to a deliberately
deleted working note. That violates the explicit documentation requirement for
this PR and design rule 1; it is cheap to close before merge.

## Findings, ranked by severity

### Medium

1. **The module document is still incomplete for the public API.** The package
exports the enum types and values, accessors, lookups, views, and individual
writers at `src/VolSurfaceAnalysis.jl:72-89`, but the module document never names
`ExerciseStyle`/`American`/`European`, `SettlementStyle`/`AMSettled`/`PMSettled`,
`Delivery`/`Physical`, `Side`/`Long`/`Short`, `ExpiryOutcome`, `contract_spec`,
`last_sequence`, `side_sign`, `open_lots`, `open_groups`, `mint_group!`,
`record_fill!`, `record_expiry!`, or `record_fee!` (`docs/modules/ledger.md:13-49,
165-181`). It also does not state the useful signatures/return semantics for the
header accessors or `order_leg`, even though these are exported. This is the same
documentation class as the earlier PR-1 review, not a reopened code finding: the
ownership edits made the boundary language true but did not finish the requested
export inventory. Add a compact public-surface section; it need not duplicate the
docstrings.

### Low

2. **The durable module document links to a note this PR deletes.** The invariant
section sends readers to `ledger-slice1-coverage.md`
(`docs/modules/ledger.md:164-168`), but that file is absent at the head by design.
Either retain the durable inventory or remove/replace the dangling link while
keeping the invariant list self-contained.

3. **Two remaining descriptions preserve the pre-ownership wording.** The
binding proposal still says the book is “a view by replay, never stored”
(`docs/proposals/ledger.md:39-40`), whereas `Ledger` owns an incrementally stored
`book` (`src/ledger/types.jl:311-324`); the module document says fill market
context belongs to an order journal “outside this module”
(`docs/modules/ledger.md:20-24`), although that journal is ledger-owned. Both can
be made true by saying “not persisted; reproducible by replay” and “outside the
economic-event journal,” respectively. The proposal's rewritten status and slice
list otherwise match the current code and the next-PR boundary.

## Review results and documentation answer

The production diff is additive. Outside `src/ledger/`, it adds only module
includes/exports and `pnl_series(::Ledger)`; no existing positions, engine,
persistence, or metric implementation changes. The overload is disjoint from
`pnl_series(::AbstractVector{Position})`, so `master` behaviour is preserved.

On the module's own terms, the single supported write path validates before
mutation; the owned book closes the caller-supplied-book hole; both replays fold
standalone books; `Ledger(events)` validates and folds and accurately documents
the counters it cannot reconstruct. `apply!` and `event` are qualified internal
helpers rather than exports. `record_order!` plans later legs against earlier
legs, validates one batch, commits once, constructs the record before commit, and
mints its group only on success. FIFO, whole-cent cash, event-local contract
resolution, backward effective references, whole expiry, duplicate execution
protection, and every documented named failure are substantively tested. Failure
tests snapshot all six counters plus event/order lengths and copy the book; each
named exception's rendering contains its type name. I do not reopen the earlier
mutable-field objection: `apply!` is now internal and the document accurately
states the supported boundary at `docs/modules/ledger.md:84-93`.

The public surface needed next is present: `record_order!`, `L.book`, order/leg
ids and lookup, replay boundaries, and `Ledger(events)`. I found no signature or
semantic change the wiring PR must force in the ledger. The venue-dependent
fill/order join correctly remains outside this module.

The five removed `check_join(L) === nothing` lines covered only that future
cross-record join. Their surrounding tests still assert the ledger-owned facts:
record shapes and ids, fill fields, observations, counters, group minting, cash,
FIFO matches, round trips, known-to replay, and incremental/replay equality.
Nothing ledger-local was covered only by those five assertions.

I independently recomputed the hand literals: 4,500; 9,000/5,000; 16,000;
-11,500 and 3,500 cash in the mixed-expiry case; 8,913/4,957 with fee shares
-87/-43; 4,467/4,466/4,467 with -33/-34/-33; partial reconciliations 21,410,
18,410 and 15,410; order cash 19,370 then 9,240; order trips 4,370/4,870;
and adapter samples 140, -50/45, and 35 USD. They are correct. I found no
assertion that passes vacuously. Two metric tests create an obsolete standalone
`Book()` (`test/metrics/test_ledger_series.jl:33,51`), and the fixture helper's
`book` argument is now unused (`test/ledger/fixtures.jl:27-33`), but the actual
assertions inspect `L`/`L.book` and are substantive; this is harmless cleanup,
not a finding.

I ran one fresh-process focused suite with `JULIA_NUM_PRECOMPILE_TASKS=1`: all
ledger tests (`contracts`, `types`, `cash`, `book`, `append`, `round_trips`),
1,425 passed. I did not run the full gate or touch the existing REPL.
