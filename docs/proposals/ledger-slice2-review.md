# Slice 2 review

Reviewed against `47cec92`. I read the requested design/proposal material in order, inspected the full diff (including deletions and untracked additions), recomputed the new test literals, and ran fresh-process focused suites: execution + engine + append, 687/687 passed; persistence, 220/220 passed. I did not run the full gate.

## Findings (ranked by severity)

### High

1. **The public structure writer can return a ledger that violates the fill/order join.** `record_order!` accepts arbitrary observations, prices and `fill_rule`, commits the events, and only afterward constructs/pushes the order record; it never runs the fill review's cross-record checks (`src/ledger/append.jl:309-355`). Thus, for example, a `:cross_spread` long fill at 0.85 against an observed ask of 0.86 succeeds, as do observations after `decided_at` and unknown simulated rules. `check_join` would reject these at `src/backtest/engine.jl:172-182`, but only the engine's final assertion and load call it. This breaks the binding “at engine atomic append” invariant and means the promised single validated writer is not actually validated. Validate the proposed record and batch together before mutating either container (or make one transactional primitive own both).

2. **Persistence does not validate the join on write.** `save_run` writes `events`, `orders`, and `order_legs` directly after checking only config identity (`src/persistence/store.jl:202-230`); `check_join` appears only after reconstruction on load (`src/persistence/store.jl:564-613`). A corrupt hand-built/mutated `Ledger` is therefore persisted successfully and fails only when read back. This misses the proposal/fill-review requirement that the foreign-key/join contract be checked both on write and load.

### Medium

3. **The claimed journal/event atomicity is not literal.** `commit!` mutates events, counters, index, and book before `OrderRecord` allocation, observation collection, `push!`, and journal-counter updates (`src/ledger/append.jl:347-354`). Those later allocations can throw, leaving economic events/book committed with no order record. The brief prescribes this ordering, but its premise that “nothing after `commit!` can fail” is false in Julia. The same inaccurate guarantee is repeated in `docs/modules/ledger.md:103-111,195-199` and `docs/modules/backtest.md:7-11,152-154` (design rules 1 and 6).

### Low

4. **The broker-execution exception is narrower than the stated invariant.** The fill review says observations are optional for `BrokerExecution`; the representation requires one `LegObservation` per order leg and `check_join` rejects a missing vector before its broker bypass (`src/ledger/types.jl:250-257`, `src/backtest/engine.jl:147-170`). This is acceptable only as the explicitly deferred live-adapter portion of D7; module prose should say that this slice permits missing quote sides, not an absent observation record (`docs/modules/backtest.md:124-133`).

## Requested checks

- `record_order!`: normal validation failures occur before `commit!`; group/order/leg counters do not advance; event id/sequence/execution counters advance correctly on success. FIFO correctly includes earlier same-order opens (`src/ledger/append.jl:233-275,330-345`). Fees follow all fills/matches, reference the right fills, and zero fees are omitted. Findings 1 and 3 qualify the atomicity result.
- Venue: long/Short rounding is respectively ceiling/floor on the tick with noise tolerance (`src/backtest/execution.jl:25-31`). IBKR tiers and boundaries are correct: 25c below 0.05, 50c in [0.05,0.10), 65c at/above 0.10, $1/order minimum. Cumulative allocation is exact; `[25,65] -> [28,72]`, and three 25c legs -> `[33,34,33]` (`src/backtest/execution.jl:75-100`).
- Engine: `known_to` is captured after `decide` and before the tick's first fill, then shared by all orders (`src/backtest/engine.jl:220-228`). The policy receives the incrementally folded book.
- Join/load: `check_join` covers contiguous record/leg ids, fill foreign key, contract/side/intent/group, cumulative quantity, observation times, executable side, known rule, and recomputed price (`src/backtest/engine.jl:140-184`). Load constructs every event, calls one batch `commit!`, rebuilds counters, then calls `check_join`; schema is 3 (`src/persistence/store.jl:35,564-613`). Finding 4 is the only cross-record gap.
- Positions are removed: source/tests/module doc deleted; remaining `positions.parquet` mentions are historical/schema assertions only. No forbidden old API symbol remains outside proposals.
- Hand calculations all checked: 19,370; 9,240; trip PnLs 4,370 and 4,870; 92.40 USD; open/trim cash 13,000 and trip 4,500; single open/close cash -1,200 and PnL -12.00; persistence ten events/two orders; regression 189,900; tick examples 1.07/1.02/0.82/0.85/1.10/1.05; all commission examples and tier edges are correct.
- Scope matches the brief. Only the two pre-existing untracked review files are outside its list; this review is the requested third. `src/backtest/execution.jl` and its test are correctly new, position files are correctly deleted, and prohibited ledger/book/identity/dependency files did not change.
- Module docs are architecturally organized and the backtest conventions table provides traceable primary-source decisions (rules 5 and 6), but finding 3 makes the atomicity descriptions incoherent with code (rule 1).

## Named-failure audit

| Failure | Fires for the named case | Ledger/book untouched | Prints name |
|---|---|---|---|
| `UnpriceableLeg` (`no_quote`, `no_executable_side`, `no_spot`) | yes | yes (venue is pure) | yes |
| `UnservedSelector` | yes | yes | yes |
| `NothingToClose`, `ExceedsOpen` | yes | yes | yes |
| `InvalidPrice`, `NonIntegralCash`, `UnknownContract`, `FillAfterExpiry` | yes | yes | yes |
| `DanglingReference(:group, ...)` | yes | yes | yes |
| `DuplicateExecution` | yes | yes | yes |
| `JoinViolation` / dangling order-leg reference | yes in `check_join` | check is read-only | yes |

The focused tests explicitly assert unchanged snapshots and printed names for the writer failures; engine pricing/selector failures happen before a writer call. `check_join` failures cannot mutate either ledger or book. Unknown rule/model cases are intentionally ordinary `ErrorException`s, not named exception types, but their messages name the requested and available symbols.

## Brief coverage

| Brief section | Status | Note |
|---|---|---|
| Ledger types/journal/counters | landed | Required shape and lookup present |
| `record_order!`, FIFO, groups, fees | partial | Happy/failure paths land; findings 1 and 3 |
| Venue and IBKR costs | landed | Rules, examples, shared minimum correct |
| Engine/orders/book/`known_to` | landed | Correct tick boundary; lifecycle remains named deferred slot |
| `fill_legs` and `check_join` | partial | Research checks land; truly absent broker observation deferred |
| Policies/agents/experiment/metrics | landed | Order/Book conversion and open-window behavior agree |
| Persistence v3 | partial | Correct round trip/load validation; missing write validation |
| Positions removal and scope/docs | partial | Removal/scope complete; atomicity docs overstate code |
| Deferred slices 3-6/later adapters | landed as deferral | Exactly two intended broken lifecycle tests remain described |

## Decisions the brief left to the code

The implementation chose to omit zero-valued `Fee` events; use ties-to-even for cumulative cent allocation; reconstruct the first occurrence of each group as the minting order on load; retain a placeholder `LegObservation` even for `:broker_execution`; use ordinary symbol-table errors for unknown venue choices; and perform join validation at engine completion/load rather than inside `record_order!` (the last choice causes finding 1).

**Verdict: not mergeable — the order/event transaction and persistence write path do not enforce the binding cross-record contract atomically.**
