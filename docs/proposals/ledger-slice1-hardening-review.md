# Slice 1 hardening review: ledger

## Findings, ranked by severity

### High: post-expiry fills are still accepted by the `Fill` constructor

The fill review makes effective time no later than contract expiry a
construction-time invariant (`docs/proposals/ledger-fill-review.md:73-79`), and the
hardening brief says those construction-time checks apply now
(`docs/proposals/ledger-slice1-hardening.md:20-23`). `Fill` validates quantity,
price, and join ids but not `header.effective_at <= contract.expiry`
(`src/ledger/types.jl:126-135`). The inventory marks this promise covered by the
append validator and explicitly interprets construction-time enforcement away
(`docs/proposals/ledger-slice1-coverage.md:69,230-232`); the cited test likewise
only rejects through `commit!`/`record_fill!` and therefore would stay green if
constructor enforcement remained absent (`test/ledger/test_append.jl:130-143`).
Move or duplicate the check in `Fill` construction, retain the named
`FillAfterExpiry`, and add a direct-constructor rejection. Until then part 4 and
the inventory are incomplete.

### Medium: the inventory overstates failure atomicity coverage

Rows 1 and 61 say every failure testset checks the complete ledger snapshot and
book (`docs/proposals/ledger-slice1-coverage.md:56,116`), but some cited tests
only check a snapshot or cash. For example, the second-expiry failure does not
compare the book (`test/ledger/test_append.jl:111-115`), and the fee-reference
failures check cash rather than full book equality (`test/ledger/test_append.jl:123-127`).
Other dedicated tests do establish atomicity for those failure types, so this is
not an implementation defect, but the inventory's universal statement is false.
Either add the missing `book == before` assertions beside each rejection or say
that atomicity is covered once per named failure/validation path.

### Low: two inventory tests do not pin the full stated promise

- Row 5 says an unknown `event` id is a *named* failure, but its condition tests
  assert only the type and fields (`test/ledger/test_types.jl:129-134` and
  `test/ledger/test_append.jl:529-531`). The generic constructed-error loop pins
  `DanglingReference` printing, not the actual `event` failure. Add
  `occursin("DanglingReference", sprint(showerror, err))` beside the lookup.
- Row 54's source-text scan hard-codes exactly six ledger files
  (`test/ledger/test_types.jl:96-105`). This is brittle repository-shape testing,
  not enforcement of the architectural boundary, and a seventh valid ledger
  source would fail vacuously. Keep the forbidden-name scan, but remove the file
  count; the closed-union and field tests already cover the substantive shape.

No other applicable statement from the four prior reviews or the current module
doc was missing. In particular, the same-batch FIFO gap, replay equality and
equal-time dependency ordering, finite prices, outcome/intrinsic agreement,
separate id/sequence counters, and slice-2 join/atomicity deferrals are mapped to
real enforcement and non-vacuous tests.

## Brief coverage

| Part | Status | Assessment |
|---|---|---|
| 1. Inventory | partial | Broad and useful, but row 14 misclassifies the required constructor check and rows 1/61 overclaim per-test atomicity. |
| 2. Tests where they belong | landed | `test_review_findings.jl` and its include are gone; all eight named testsets moved intact; the helper is gone; the three-argument `commit!` is direct; the atomicity `@test_broken` remains last with its comment. |
| 3. Coverage | landed | Prefix replay and book shape run after every event of every `_LG_CASES` fixture. Partial reconciliation covers every fixture plus staged open-lot/fee remainders; its signs, quantities, cumulative fee rounding, and literals are correct. |
| 4. Enforcement | partial | Outcome, recorded-time rules, price validity, and unknown-id naming landed, but post-expiry `Fill` construction is still unenforced. |

## Rule additions

| Rule | Opinion | Reason |
|---|---|---|
| R1: `recorded_at >= effective_at` | keep | It prevents the as-known journal from containing a fact before that fact is effective. |
| R2: recorded time nondecreasing in sequence | keep | It is already required by events-review invariant 2 and makes sequence a coherent knowledge order; equality remains allowed. |
| R3: settlement price finite and non-negative | keep | Negative/non-finite underlying settlement prints are invalid, and construction-time rejection preserves the meaning of `NonIntegralCash`. |
| R4: positive fill join ids | keep | Positive integers are the defensible concrete meaning of the fill review's “nonempty” ids and reject sentinel zeros. It is implemented as two field checks, so the coverage doc should not literally claim every rule addition is “one check”; it is one logical rule, one testset, and one module-doc bullet. |

The module invariants and named failures otherwise match code, the status entry is
updated, and scope is clean: only brief-listed files changed, plus the two
pre-existing untracked reviews and the new coverage document. I relied on the
reported full gate: **2281 passed, 0 failed, 0 errored, 1 broken**; the sole Broken
is the preserved structure-atomicity testset. I did not rerun `Pkg.test()`.

**Verdict: not mergeable, because the binding construction-time post-expiry
`Fill` invariant is neither enforced nor directly tested, while the inventory
incorrectly reports append-time validation as sufficient.**
