# Slice 1 coverage: every promise, its code and its test

The inventory the [hardening brief](ledger-slice1-hardening.md) asked
for, built 2026-09-12 against `src/ledger/*.jl` as it stands after this
round and revised after codex's review of it
([ledger-slice1-hardening-review.md](ledger-slice1-hardening-review.md)). One row per statement any of the documents makes about the
ledger module; the second table holds what belongs to later slices so
nothing is lost. Read the summary, then the rule additions (each can be
vetoed in one sentence), then the tables.

## Summary

- 66 statements apply to the module today (table 1); 16 more are
  deferred (table 2).
- 7 of the 66 had no enforcing code, or an unnamed one, at the start of
  the round: rows 5, 19, 21, 22, 23, 47, 48. All seven are enforced
  now, with two new named failures (`InvalidPrice`,
  `RecordedOutOfOrder`) and two broadened ones (`MatchMismatch` for an
  expiry's outcome, `DanglingReference` for an unminted id and a
  non-positive join id). One more, row 14, was enforced on append only
  where the fill review asks for construction time; codex's review of
  the round caught it, and the `Fill` constructor now refuses a
  post-expiry fill with `FillAfterExpiry`.
- 21 of the 66 had no test, or a partial one: rows 1 (book unchanged
  after a failure was asserted only sometimes), 3, 5, 7, 11, 19, 21,
  22, 23, 26, 32, 35, 36, 37, 40, 43, 46, 47, 48, 53, 54. Every row now
  names a test or states why none is needed.
- Four of the checks are rule additions in the brief's sense (not
  written in the proposal or the module doc before this round); they
  are listed under "Rule additions" below.
- `test/ledger/test_review_findings.jl` is gone; its eight testsets
  live in `test_book.jl` (three replay-equality promises) and
  `test_append.jl` (four rejections and the known-broken atomicity
  promise), names intact. See "Where the review-findings tests live".

## How to read the tables

Sources: **MD** = [docs/modules/ledger.md](../modules/ledger.md);
**P** = [ledger.md](ledger.md) (the proposal, section 2 unless a
decision number is given); **ER** = [ledger-events-review.md](ledger-events-review.md)
(numbers are its "Ordering and identity invariants"); **FR** =
[ledger-fill-review.md](ledger-fill-review.md) ("Enforced invariants",
construction-time list); **SR** = [ledger-slice1-review.md](ledger-slice1-review.md);
**FXR** = [ledger-slice1-fix-review.md](ledger-slice1-fix-review.md);
**FX** = [ledger-slice1-fix.md](ledger-slice1-fix.md); **WT** = the
driver's walkthrough, sections 6 and 7; **HB** = the hardening brief;
**S1** = [ledger-slice1.md](ledger-slice1.md); **D7** = design rule 7.

Code is under `src/ledger/` unless a path is given. Tests are under
`test/ledger/`; a testset's prefix names its file (`types:` in
`test_types.jl`, `contracts:`, `cash:`, `book:`, `append:`,
`round_trips:`; `pnl_series(ledger):` is `test/metrics/test_ledger_series.jl`;
the `ledger promise:` testsets say which file they are in). "none
needed" states its reason in the action column.

## Table 1: statements that apply to the module as it stands

| # | Statement | Stated in | Enforced by | Tested by | Action this round |
|---|---|---|---|---|---|
| 1 | Every write goes through one validated path; a batch is checked whole against ledger and book; nothing is appended if any check fails | MD Invariants; P Tick order; ER 4 | `append.jl` `commit!` (validates, then mutates), `_validate` | every rejection in every failure testset, the error-capturing attempts included, is followed by `_lg_snapshot(L) == snap` and `book == before` (a deep copy taken before the attempt); whole-batch refusals in "append: NonIntegralCash refuses a batch...", "append: UnknownContract is refused at commit before anything lands", "append: RecordedOutOfOrder, recorded time is nondecreasing along sequence", "append: FIFO across lots opened in the same batch" | `snap, before` and both checks added beside every rejection that lacked them (codex review, Medium); two new batch-refused-whole tests |
| 2 | `sequence` continues the ledger's counter: contiguous, unique, append-only, the replay order | MD; P; ER 2 | `_validate` (`SequenceGap(:sequence)`) | "append: SequenceGap on either counter"; "append: id and sequence are separate counters, never used for each other" | book-unchanged and printed-name asserted; id-as-sequence rejection added |
| 3 | `id` continues its own counter; stable, unique, never the vector index; ids and sequence are separate counters never used for each other | MD; P; ER 1; S1; SR ("divergence is not positively exercised") | `_validate` (`SequenceGap(:id)`); `types.jl` `Ledger.index`, `event` | "append: id and sequence are separate counters, never used for each other" (new: `next_id` 100, `next_sequence` 1; all three writers; lookups by id, replay by sequence, references and trips by id, sequence-as-id rejected) | test added |
| 4 | `event(L, id)` looks up by id, never by index | MD; S1 | `types.jl` `event` via `index` | "append: case 1, full round trip" (`event(L, 3) === m`); row 3's test | none needed beyond row 3 |
| 5 | `event(L, id)` for an id the ledger never minted is a named failure | HB Part 4 (was a bare `KeyError`) | `types.jl` `event` -> `DanglingReference(:event_id, id)` | "types: a fresh Ledger is empty with every counter at one"; row 3's test | **enforced this round** (named failure replaces `KeyError`) |
| 6 | Groups are minted by the ledger for opening orders, a separate counter | MD Kinds; P Orders | `append.jl` `mint_group!` | "append: mint_group! counts up" | none; rollback on a failed order is slice 2 (table 2, D9) |
| 7 | Every reference points backward in sequence to an event of the right kind: `open_fill_id` an `Open` fill, `close_fill_id` and `source_id` a fill | MD; P Invariants; ER 4 | `_lookup`, `_fill_ref`, `_opening_fill` in `_validate`, `_check_match`, `_check_expiry` | "append: DanglingReference"; "append: record_fee! ties a cost to its fill"; "book: consuming a lot the book does not hold is a named failure" (the fold's guard) | added: an expiry naming a `Match`, a fee naming an `Expiry`, a forward reference inside one batch, book unchanged, printed name |
| 8 | The matches of a closing fill immediately follow it | MD; ER 4, Match section | `_validate` `Close` branch and lone-`Match` branch | "append: MatchMismatch" (a match on its own) | `book == before` added |
| 9 | The matches of a close reference it and exhaust it exactly; otherwise rejected before anything is appended | MD; P Cash rules; ER Match | `_validate` (`total == e.quantity`) | "append: MatchMismatch" (under, none, over) | none needed |
| 10 | A match pairs a lot of the closing fill's own group and contract on the opposite side; the match's own group is the close's | MD; P; ER 5 | `_check_match` | "append: MatchMismatch" (other group, other match group, same side) | none needed |
| 11 | FIFO is checked on append: each match takes the oldest still-eligible lot, book lots first then lots opened earlier in the batch, net of the batch's own consumption | MD; SR 6.3; FX C; FXR ("no single test combines two batch-open lots") | `_first_eligible`, `_check_match` | "ledger promise: validated batches enforce FIFO" (`test_append.jl`, moved); "append: case 2, close split across lots" and "append: a partial close leaves the oldest lot's remainder" (the writer); "append: FIFO across lots opened in the same batch" (new: two batch lots, newer-first and partial-older rejected, canonical accepted; a book lot before a batch lot) | moved; same-batch test added |
| 12 | Consumption never exceeds a lot's remaining (match, expiry, the writer's close, the fold) | MD; P; ER 5 | `_check_match`, `_check_expiry` (`ExceedsOpen`); `record_fill!`; `book.jl` `_consume!` | "append: ExceedsOpen on a hand-built over-consumption"; "append: case 3, two groups on one contract"; "append: record_expiry! settles the whole remaining lot" (expire again); "book: consuming a lot the book does not hold is a named failure" | fields and printed name asserted |
| 13 | A close with nothing to close is an error at fill time; a close matches the opposite side only | MD; P Orders; ER Close | `record_fill!` (`NothingToClose`) | "append: case 3, two groups on one contract"; "append: NothingToClose" (new: empty group, contract not held, only same-side lots; fields; unchanged; name) | dedicated testset added. Interpretation: through `commit!` a hand-built close with no matches is `MatchMismatch` ("matches consume 0 of a close of q"); `NothingToClose` is the writer's failure |
| 14 | A fill is effective at or before its contract's expiry; equality allowed; no such value exists | MD; P; ER 5; FR (construction-time); FX decision | `types.jl` `Fill` constructor (`FillAfterExpiry`, so no such value exists); `_validate` checks it again for an event that bypasses the constructor (one loaded from storage, slice 6) | "types: a fill effective after its contract's expiry cannot be built" (direct constructor: rejected after expiry, allowed at the instant, fields, name); "append: FillAfterExpiry" (through the writer; snapshot and book unchanged; fields; name) | **enforced at construction this round** (codex review, High); the append-time check kept, unreachable through the constructor and so without a direct test |
| 15 | Every reference points backward in effective time: a match's opening fill at or before its closing fill, the match at the closing fill's instant, an expiry's opening fill at or before the expiry, a fee's source at or before the fee | MD; FX B; WT 7.1, 6.2 | `_check_match`, `_check_expiry`, `_validate` `Fee` branch (`MatchMismatch`) | "append: references point backward in effective time"; "ledger promise: consumption is not effective before its open" (`test_append.jl`, moved); "ledger promise: accepted equal-time lifecycle events replay safely" (`test_book.jl`, moved) | moved |
| 16 | An expiry's copied contract, side and group equal its opening fill's | MD; P decision 11; ER Expiry | `_check_expiry` | "append: MatchMismatch" (three variants); "append: record_expiry! settles the whole remaining lot" | none needed |
| 17 | An expiry settles the whole remaining lot, exactly one lot remainder | MD; ER Expiry; WT 7.2; FX D | `_check_expiry` (`quantity == available`) | "ledger promise: expiry consumes the whole remaining lot" (`test_append.jl`, moved); "append: record_expiry! settles the whole remaining lot" | moved |
| 18 | An expiry is effective at or after its contract's expiry | MD; WT 7.3; FX D | `_check_expiry` | "ledger promise: expiry is not effective before contract expiry" (`test_append.jl`, moved) | moved |
| 19 | An expiry's `outcome` is `Worthless` when intrinsic is zero and `CashSettled` otherwise | `record_expiry!` docstring; P Events (`outcome`); ER Expiry; HB Part 4 | was writer-only; now `append.jl` `_outcome` (one rule for writer and validator), checked in `_check_expiry` (`MatchMismatch` "outcome ... disagrees") | "append: an expiry's outcome must agree with its intrinsic value" (new); the writer in "append: record_expiry! settles the whole remaining lot" | **enforced this round** |
| 20 | Quantities are positive integers (`Leg`, `Fill`, `Match`, `Expiry`), thrown at construction | MD; P; ER 5; FR | constructors in `types.jl` (`NonPositiveQuantity`) | "types: Leg checks its quantity"; "types: event constructors reject non-positive quantities and invalid prices"; "append: NonPositiveQuantity is a named failure" | book unchanged and printed name asserted |
| 21 | `Fill.price` is finite and positive | `Fill` docstring; FR; HB Part 4 (a non-positive price was an unnamed `ArgumentError`; non-finite went unchecked until `NonIntegralCash`) | `Fill` constructor (`InvalidPrice`) | "types: event constructors reject non-positive quantities and invalid prices" (0, negative, `Inf`, `NaN`); "append: InvalidPrice is thrown at construction, before the write path" | **enforced this round** with the named failure |
| 22 | `Expiry.settlement_price` is finite and non-negative | HB Part 4 (rule addition R3) | `Expiry` constructor (`InvalidPrice`) | same two testsets (0 accepted; negative, `-Inf`, `NaN` refused; through `record_expiry!`) | **enforced this round**; rule addition |
| 23 | A fill's `order_leg_id` and `execution_id` are positive ("nonempty", read for integers) | FR construction-time list ("nonempty `order_leg_id` and `execution_id`") | `Fill` constructor (`DanglingReference(:order_leg_id / :execution_id)`) | "types: a fill's join ids are positive" | **enforced this round**; rule addition R4 |
| 24 | Side and intent are valid | FR | by type (`@enum Side`, `@enum Intent`) | "types: side_sign and the enums" | none needed (a Julia enum admits no other value) |
| 25 | Every event's cash is whole cents; a product that is not (or is non-finite or beyond `Int`) is refused as `NonIntegralCash`, never rounded; the batch is refused before anything lands | MD Cash rules; FX E, G1; FXR Low (closed at ed41825) | `cash.jl` `contract_cents`; `_validate` (`foreach(cash, batch)`) | "cash: contract_cents is the one rounding point"; "append: NonIntegralCash refuses a batch whose cash is not whole cents" | none needed |
| 26 | An unlisted underlying is `UnknownContract`, never a default; refused at commit before anything lands | MD; P; HB Part 3 | `contracts.jl` `contract_spec`; reached through `_validate`'s cash pass | "contracts: an unknown ticker throws UnknownContract"; "cash: an unlisted underlying needs an explicit spec"; "append: UnknownContract is refused at commit before anything lands" (new: hand-built fill, the writer, a listed fill ahead of it in the batch) | commit-path test added; printed name asserted |
| 27 | Cash is a method on an event, never a stored field | MD; P; ER Fill | by construction: no event struct has a cash field (`types.jl`) | `test_cash.jl` (every rule computed from fields) | none needed (structural) |
| 28 | Cash rules: `Fill` `-side * cents * q`; `Match` 0; `Expiry` `side * cents(intrinsic) * q`; `Fee` its amount | MD table; P Cash rules | `cash.jl` `cash` | "cash: a fill is minus side...", "cash: a match moves no cash", "cash: an expiry is side times...", "cash: a fee is its amount in cents", "cash: intrinsic per share" | none needed |
| 29 | One rounding point, `contract_cents`; integer arithmetic after it | MD; FX E | `cash.jl` | "cash: contract_cents is the one rounding point" (`isa Int`) | none needed |
| 30 | Contract facts are resolved from the table on every write, never a caller-supplied spec, so the incremental book equals both replays | MD Key decisions; SR 6.1; FX A | `commit!` applies with one-argument `apply!` | "ledger promise: every public write replays to its incremental book" (`test_book.jl`, moved; asserts no four-argument `commit!` method, then every writer on every fixture); "book: case 7, incremental equals replay in every case" | moved and restated with the three-argument `commit!` |
| 31 | A round trip's PnL is opening plus closing cash plus a fee share; shares by cumulative rounding in whole cents; over a fully consumed fill they sum to the fee exactly | MD; P decision 2; FX E, G2 | `round_trips.jl` `share!` | "round_trips: case 5, fees across a partial close"; "round_trips: a fee on the opening fill is shared by its consumers"; "round_trips: fee shares are whole cents by cumulative rounding" | none needed |
| 32 | A fill left partly open leaves the fee remainder unallocated; trips plus the cash tied up in open lots plus the unallocated fee equal `book.cash` exactly | MD; HB Part 3 | `round_trips.jl` | "round_trips: partial reconciliation, trips plus open lots plus unallocated fees equal cash" (new: every fixture with literals, plus a fee on a partly consumed open at three stages) | test added |
| 33 | Once nothing is left open, round trips sum to the book's cash exactly | MD; P; FX G3 | `round_trips.jl` | "round_trips: reconciliation is exact, trips sum to cash once nothing is open" | none needed |
| 34 | `pnl_series(ledger)` converts cents to USD at its boundary; nothing inside the ledger is float money | MD | `src/metrics/ledger_series.jl` | `pnl_series(ledger):` testsets | none (out of scope; the file sits beside its source) |
| 35 | The book holds lots per `(group, contract)`, FIFO within, plus cash; a lot with nothing remaining is dropped, and so is an emptied key | MD; `apply!` docstring | `book.jl` `apply!`, `_consume!` | `_lg_check_book` (fixtures) applied after every event of every fixture in "book: what was known at every boundary is the fold of the first k events", on every fixture in "book: case 7...", after every replay, in "ledger promise: every public write...", in row 3's test; "book: apply! per kind" | helper added and applied everywhere |
| 36 | Every lot sits under the key of its own group and contract | HB Part 3 (structural) | `book.jl` `apply!` keys on `(e.group, e.contract)` | `_lg_check_book`, as row 35 | test added |
| 37 | Lots within a key are in opening order | MD ("FIFO within"); HB Part 3; SR open interpretation (FIFO is sequence order) | `book.jl` `apply!` pushes in fold order | `_lg_check_book` (ascending, distinct opening fill ids), as row 35 | test added. Interpretation: opening order = ascending opening fill id = sequence order |
| 38 | The book is never stored; it is the fold of the events; the incremental book equals the full replay exactly | MD; P | `book.jl` | "book: case 7, incremental equals replay in every case"; "ledger promise: incremental book exactly equals effective replay" (`test_book.jl`, moved) | moved |
| 39 | Replayed cash equals the sum of event cash | P Invariants | `book.jl` `apply!` | "book: case 7..." | none needed |
| 40 | `book_as_known(L, k)` cuts by sequence and equals the fold of the first `k` events, for every `k` | MD; P; ER two replays; HB Part 3 | `book.jl` `book_as_known` | "book: case 8, known versus true"; "book: what was known at every boundary is the fold of the first k events" (new: every `k` of every fixture) | test added |
| 41 | Recorded time is not a safe cut for what was known: events booked at one tick share it | MD; P Book and replay | design of `book_as_known` | "book: what was known at every boundary..." (a close fill and its matches share a recorded instant yet differ in knowledge) | assertion added |
| 42 | `book_effective` cuts by effective time and folds `(effective_at, sequence)`; equal instants fold in journal order, dependency-safe | MD; FX B | `book.jl` `book_effective` | "book: case 4, mixed expiries in one group"; "book: case 8, known versus true"; "ledger promise: accepted equal-time lifecycle events replay safely" (moved) | moved |
| 43 | A replay at an instant before the first event is an empty book: empty means temporal absence | HB Part 3; D7 | `book_effective` | "book: a replay at an instant before the first event is an empty book" (new: every fixture and the empty ledger) | test added |
| 44 | The two replays differ only by lifecycle booked at the tick after its instant | MD; P; ER | design | "book: case 8..." (the expiry: absent as known, present as true); "book: what was known at every boundary..." (on the effective-monotone fixtures the two agree at every instant) | per-instant agreement added |
| 45 | Effective time need not be monotone in sequence | MD; P; ER 3 | `_validate` imposes no such check | "ledger promise: incremental book exactly equals effective replay" (asserts the journal is not effective-sorted); "append: RecordedOutOfOrder, recorded time is nondecreasing along sequence" (a batch in reverse effective order is accepted) | made explicit |
| 46 | The stored journal is never sorted in place | ER 3 | `book_effective` sorts a copy (`due`) | "ledger promise: incremental book exactly equals effective replay" (sequence order intact after a replay) | assertion added |
| 47 | `recorded_at` is nondecreasing along sequence, across the batch boundary and within a batch; equal timestamps are normal | ER 2; HB Part 4 (rule addition R2) | `_validate` (`RecordedOutOfOrder`, bound = previous recorded time) | "append: RecordedOutOfOrder, recorded time is nondecreasing along sequence" (across the boundary via `record_fee!`, within a batch via `commit!`, an equal instant accepted, the reverse effective order accepted) | **enforced this round**; rule addition |
| 48 | Every event's `recorded_at` is at or after its `effective_at`; effective may precede recorded (a late-booked expiry) but never the reverse | HB Part 4 (rule addition R1); ER 3 (the allowed direction); P Tick order | `_validate` (`RecordedOutOfOrder`, bound = effective time) | "append: RecordedOutOfOrder, a fact is not recorded before it is true" (fill, expiry, fee; equal accepted); the allowed direction in fixture case 4 and "book: case 8..." | **enforced this round**; rule addition |
| 49 | An expiry between ticks is booked with effective time at the settlement instant and recorded time at the next tick; a policy sees the lot gone, an audit before the tick does not back-date knowledge | ER; P Tick order | the writers accept it (row 48's rule allows it) | fixture case 4; "book: case 8, known versus true" | none needed |
| 50 | Nothing depends on `Dict` iteration order: accessors and the effective replay sort | ER 6 | `book.jl` `open_lots`, `lots`, `open_groups` (`_by_open`), `book_effective` | "book: open_lots, lots and open_groups, ordered by opening fill" | none needed; the tick-order half of ER 6 is table 2, D10 |
| 51 | `ContractKey` hashes and compares by content | MD Key decisions | `types.jl` `Base.hash`, `Base.:(==)` | "types: ContractKey has content hash and equality" | none needed |
| 52 | Group sits on `Fill`, `Match`, `Expiry`; a `Fee` has none and answers `nothing` | MD; ER Common fields | `types.jl` `group` | "types: header accessors on every kind; group on lifecycle events only" | none needed |
| 53 | The container is a vector over the closed union `LedgerEvent` | MD; P decision 4; ER Container | `types.jl` `LedgerEvent`, `Ledger.events` | "types: the container is a vector over the closed union" (new) | test added |
| 54 | The module knows no quotes, spots or time cut: identity vocabulary from `data` only | MD; S1 Scope; FR Tests | by absence in `src/ledger/*.jl` | "types: the module knows no quotes, spots or time cut" (new: reads the six source files) | test added |
| 55 | A fill carries nothing about the market it was filled against | MD; P Events; FR decision C | `Fill` fields | row 54's test (no `OptionQuote`/`SpotPrice`); "types: header accessors..." | none needed beyond row 54 |
| 56 | Intent is declared, never inferred from direction; a same-side "close" has nothing to close | MD; P Orders | `Leg.intent`; `record_fill!` candidates by opposite side | "append: case 3..." ; "append: NothingToClose" | dedicated case added |
| 57 | `Match` is an event, one per lot allocation; one close can split across lots | MD; P; ER Match | `record_fill!` | "append: case 2, close split across lots"; "round_trips: case 2, per-trip pnl by lot" | none needed |
| 58 | `Expiry` is per lot and carries its side and contract | MD; P; ER Expiry | `Expiry` fields; `record_expiry!` | "book: case 4, mixed expiries in one group"; "round_trips: case 4, an expired lot is a trip of kind :expired" | none needed |
| 59 | A fee names the fill that caused it; its amount is signed whole cents | MD; ER Fee | `Fee`; `_validate` `Fee` branch | "append: record_fee! ties a cost to its fill"; "cash: a fee is its amount in cents" | none needed |
| 60 | Two groups on one contract never touch each other's lots | MD | `_lots_at(book, group, contract)` | "append: case 3, two groups on one contract"; "append: MatchMismatch" (a lot of another group) | none needed |
| 61 | Every named failure is an `Exception` that prints its name; each fires on its condition; after a failed batch the ledger counters and the book are unchanged | S1 Tests 9; HB Part 3 | `showerror` methods in `append.jl`, `contracts.jl` | "append: every error is an Exception that prints its name" (all eleven); per-failure testsets listed in the "Named failures" table below, each of which follows every rejection with `_lg_snapshot(L) == snap` and `book == before`; the fold's own failures in "book: consuming a lot the book does not hold is a named failure" leave the book as it was | three names added to the loop; snapshot and book checks completed beside every rejection (codex review, Medium) |
| 62 | Stale or duplicate execution ids are accepted; the counter never moves backwards | "append: one batch may open and close together..." (deliberate, SR, WT 7.5) | `commit!` (`max`) | that testset | none; rejection is slice 2 (table 2, D4) |
| 63 | `n_opens` and `n_closes` count fills, as the old series did | MD adapter; WT 7.5 | `ledger_series.jl` | `pnl_series(ledger):` testsets | none (out of scope) |
| 64 | Structure sampling is per `(group, closed_at)` | MD; SR open interpretation | `ledger_series.jl` | `pnl_series(ledger):` testsets | none; legs sharing one instant is table 2, D11 |
| 65 | Fees are discovered across the complete ledger, a derived table, not an as-of view | SR open interpretation | `round_trips.jl` (first pass collects every `Fee`) | "round_trips: case 5..." (the fee is appended after the matches it is shared over) | none needed |
| 66 | An empty result is temporal absence only: an empty `round_trips`, an empty `pnl_series`, an empty book before the first event | D7; SR (adapter) | `round_trips`, `pnl_series`, `book_effective` | "round_trips: rows are in sequence order; an empty ledger has none"; "pnl_series(ledger): an empty ledger, and an unknown unit"; row 43 | row 43 added |

## Named failures: fires, leaves nothing behind, prints its name

| Failure | Fires on its condition | Ledger counters and book unchanged | Prints its name |
|---|---|---|---|
| `SequenceGap` | "append: SequenceGap on either counter"; row 3's test | same | same; the name loop |
| `RecordedOutOfOrder` (new) | the two "append: RecordedOutOfOrder..." testsets | same | same; the name loop |
| `NonIntegralCash` | "append: NonIntegralCash refuses a batch..."; "cash: contract_cents is the one rounding point" | "append: NonIntegralCash..." | "cash: contract_cents..."; the name loop |
| `UnknownContract` | "contracts: an unknown ticker throws UnknownContract"; "append: UnknownContract is refused at commit before anything lands" | the latter | "contracts: an unknown ticker..."; the name loop |
| `FillAfterExpiry` | "types: a fill effective after its contract's expiry cannot be built" (the constructor); "append: FillAfterExpiry" (the writer) | "append: FillAfterExpiry" | both; the name loop |
| `DanglingReference` | "append: DanglingReference"; "types: a fill's join ids are positive"; "types: a fresh Ledger..." and "append: id and sequence are separate counters..." (`event` on an unminted id); "book: consuming a lot the book does not hold..." | "append: DanglingReference"; "append: record_fee! ties a cost to its fill" | "append: DanglingReference"; both `event` lookups; the name loop |
| `MatchMismatch` | "append: MatchMismatch"; "append: references point backward in effective time"; "append: an expiry's outcome must agree..."; the four moved `ledger promise:` rejections; "append: FIFO across lots opened in the same batch" | all of them | "append: MatchMismatch"; the name loop |
| `ExceedsOpen` | "append: ExceedsOpen on a hand-built over-consumption"; "append: case 3..."; "append: record_expiry! settles..." | same | "append: ExceedsOpen..."; the name loop |
| `NothingToClose` | "append: NothingToClose"; "append: case 3..." | same | "append: NothingToClose"; the name loop |
| `NonPositiveQuantity` | "types: Leg checks its quantity"; "types: event constructors reject..."; "append: NonPositiveQuantity is a named failure" | "append: NonPositiveQuantity..." (through `record_expiry!`) | same; the name loop |
| `InvalidPrice` (new) | "types: event constructors reject non-positive quantities and invalid prices"; "append: InvalidPrice is thrown at construction, before the write path" | the latter (through `record_fill!` and `record_expiry!`) | the former; the name loop |

## Rule additions

Checks this round added that the proposal and the module doc did not
state before it. Each is one rule, one testset and one line in the
module doc (R4 is two field checks under one rule); each can be vetoed
by deleting those three things.

- **R1. `recorded_at >= effective_at` for every event** (`_validate`,
  `RecordedOutOfOrder`; row 48). A journal entry says the ledger learned
  a fact at `recorded_at`, and learning a fact before it is true is a
  contradiction that would let the as-known replay show a future.
- **R2. `recorded_at` is nondecreasing along sequence, across the batch
  boundary and within a batch** (`_validate`, `RecordedOutOfOrder`;
  row 47; stated by the events review, item 2, but not by the proposal
  or the module doc). Sequence is the order the ledger learned things,
  so recorded time can stand still or advance along it but never step
  back; otherwise a cut by sequence and a cut by recorded time would
  disagree about what was knowable.
- **R3. `Expiry.settlement_price` is finite and non-negative**
  (`Expiry` constructor, `InvalidPrice`; row 22; the fill half, finite
  and positive, was already written in the `Fill` docstring and the
  fill review). A settlement print is a price of the underlying, which
  is never negative or non-finite, and refusing it at construction keeps
  `NonIntegralCash` for what it names.
- **R4. `Fill.order_leg_id` and `Fill.execution_id` are positive**
  (`Fill` constructor, `DanglingReference`, two field checks under one
  rule; row 23; the fill review's "nonempty" ids, read for integer
  ids). Nothing will ever mint an id
  of zero or below, so such a fill can never join an order leg or an
  execution report, and a placeholder zero would be exactly the
  sentinel design rule 7 forbids.

Not rule additions, only enforcement of what was already written: the
expiry outcome check (row 19, the writer's own docstring), the finite
positive fill price (row 21), and naming an unminted id at `event`
(row 5, a named failure in place of a `KeyError`).

## Table 2: deferred to slice 2 (or the slice named)

| # | Statement | Stated in | Slice | Why it cannot land now |
|---|---|---|---|---|
| D1 | Every fill joins exactly one order leg: an `OrderLegRecord` exists for `Fill.order_leg_id` | P Invariants, Order journal; FR 1; SR | 2 | there is no order journal yet; `order_leg_id` is an unchecked integer beyond positivity (row 23) |
| D2 | Fill contract, side, intent and group equal the leg's and the order's | P; FR 2 | 2 | needs the leg record |
| D3 | Cumulative fill quantity per leg never exceeds the ordered quantity | P; FR 3 | 2 | needs the leg record |
| D4 | `(source, execution_id)` is unique, so a retried live fill is idempotent; stale duplicates are accepted today on purpose (row 62) | P; FR 4; SR; WT 7.5 | 2 | ownership of execution ids is a slice 2 decision |
| D5 | A simulated fill has exactly one selected quote observation and one spot observation, matching contract and underlying, not after the decision cut | FR 5; P Order journal | 2 | observations live in the order journal |
| D6 | The required quote side is present and `Fill.price == fill_rule(quote, side)` | FR 6; P | 2 | needs the observation |
| D7 | Under `BrokerExecution` observations are optional and never alter the reported price | FR 7 | 2, live adapter later | needs the fill rule vocabulary |
| D8 | The order journal: order, order leg, status and observation records; the order record carries the ledger sequence at the decision | P Order journal; FR | 2 | the journal itself |
| D9 | `record_order!`: a structure lands whole or not at all; every leg validated before the order journal and the ledger events are written as one transaction; group minting rolled back on failure | P Tick order; SR 6.4; WT 6.4; FXR | 2 | the `@test_broken` at the end of `test_append.jl` waits for it |
| D10 | Tick order lifecycle, then decide, then fill, as an explicit invariant; the as-known boundary a decision saw comes from its order record | ER 6; P Tick order | 2, 3 | engine work |
| D11 | All legs of an all-or-none structure share one effective instant, so `(group, closed_at)` sampling is one sample per structure | SR open interpretation | 2 | the engine books the structure |
| D12 | Persistence validates the fill-to-order join on write and on load; a load fails on a dangling fill, a duplicate execution id, a cardinality violation, a field mismatch or an invalid simulated price | P Order journal; FR | 6 | persistence of the ledger |
| D13 | `Assignment`, `Exercise`, `CashMovement` events | ER Event kinds; P Not events | later, research extensions and the broker adapter | named deferral in the proposal |
| D14 | When an expiry is booked: the lifecycle model with the session calendar; a gap is a named failure; the lot stays open | P Lifecycle; WT 7.3 | 3 | the ledger only refuses the impossible (rows 17, 18) |
| D15 | `window_end_spot` and `n_unmarked` leave `PnLSeries` | MD adapter | 5 | metrics move onto the structure series |
| D16 | Resolved `ContractSpec`, `ExecutionModel` and `LifecycleModel` project into identity | P decision 7 | 4 | identity work |

## Where the review-findings tests live

`test/ledger/test_review_findings.jl` is deleted and its include line
removed from `test/runtests.jl`. Its testsets, names intact:

| Finding | Testset | Now in |
|---|---|---|
| 6.1 | "ledger promise: every public write replays to its incremental book" | `test_book.jl`, restated with the three-argument `commit!` and `!hasmethod` for the four-argument one |
| 6.2 | "ledger promise: accepted equal-time lifecycle events replay safely" | `test_book.jl` |
| 6.3 | "ledger promise: validated batches enforce FIFO" | `test_append.jl` |
| 6.4 | "ledger promise: a structure lands whole or not at all" (`@test_broken`, comment intact) | end of `test_append.jl` |
| 7.1 | "ledger promise: consumption is not effective before its open" | `test_append.jl` |
| 7.2 | "ledger promise: expiry consumes the whole remaining lot" | `test_append.jl` |
| 7.3 | "ledger promise: expiry is not effective before contract expiry" | `test_append.jl` |
| 7.4 | "ledger promise: incremental book exactly equals effective replay" | `test_book.jl` |

The `_lg_commit_review!` helper is gone; every moved test calls
`commit!(L, book, batch)` directly.

## Interpretations made this round

- The two recorded-time checks sit in `_validate`, not in the
  `EventHeader` constructor: the nondecreasing half needs the ledger,
  and keeping both under one failure in one place reads better than
  splitting them. A header with recorded time before effective time
  can therefore be built as a value but never committed.
- `RecordedOutOfOrder(id, recorded_at, bound)` keeps the shape the brief
  suggested; the message says which of the two bounds `bound` is.
- The fill review's "nonempty `order_leg_id` and `execution_id`" has no
  direct meaning for integer ids; it is read as positivity and reuses
  `DanglingReference` (a reference that can point to nothing) rather
  than a third new name.
- `NothingToClose` is the writer's failure. A hand-built close through
  `commit!` with no matches is `MatchMismatch`, since the validator
  sees a close whose matches consume zero.
- `FillAfterExpiry` is thrown by the `Fill` constructor (the fill
  review's construction-time list; codex's High finding on the first
  pass of this round) and checked again in `_validate`. The append-time
  check cannot be reached through the constructor; it stands for an
  event that bypasses it, such as one deserialised in slice 6, and so
  has no direct test.
- Opening order within a key is ascending opening fill id, which is
  sequence order, per the slice 1 review's open interpretation.
- `book_as_known(L, k)` with `k` past the last sequence returns the
  whole fold; row 3's test pins that as "a boundary past the end cuts
  nothing".
- `_lg_check_book` also asserts that an emptied key is dropped, a step
  beyond the brief's three conditions; `apply!`'s docstring promises it
  and "book: apply! per kind" already tested it once.
- Every fixture journal is effective-monotone, so the per-instant
  agreement of the two replays (row 44) holds on all of them; the
  non-monotone case is the moved 7.4 test.
- The brief lists `NothingToClose` on same-side lots as a gap; "append:
  case 3, two groups on one contract" already covered it. The dedicated
  testset was added anyway for the three-part check.
- The module-boundary scan asserts only that `src/ledger/` has source
  files to scan, not how many (codex, Low): the count was repository
  shape, not the boundary.
