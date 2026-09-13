# Ledger rebuild: orchestration handoff

Instructions for the session that drives the implementation. The
human wants to do nothing between review points. Your job is to run
the loop below, one slice at a time, and surface a report at each
review point.

## State as of 2026-09-13, morning

- Branch `claude/ledger-system-review-h9c8md`. Origin holds it at
  `3f3d7c9`; locally two docs commits follow (`47cec92`, the slice 2
  brief, and the commit carrying this handoff, the slice 2 review and
  the fix brief). Slice 2's code sits **uncommitted in the working
  tree** (43 files, `positions` staged for deletion): gate observed
  2026-09-12 night at 2795 passed, 0 failed, 0 errored, 2 broken, the
  two Broken being the slice 3 placeholders named in the brief.
- Codex's review of slice 2 (`ledger-slice2-review.md`): not mergeable
  on four findings, two High (the writer commits before the
  cross-record checks run; `save_run` never runs `check_join`), one
  Medium (the "nothing after `commit!` can fail" claim), one Low (a
  doc sentence). Every literal and the scope were confirmed. The fix
  round is `ledger-slice2-fix.md`; run it as a mission (fresh agent,
  gate, codex re-check), then commit the slice as one commit.
- Two review inputs, `ledger-events-review.md` and
  `ledger-fill-review.md`, are still untracked pending the human's
  decision.
- The `julia` window holds a REPL with Revise and the package loaded, no
  state worth keeping; the `codex` window is at a bash prompt. The
  headless codex runner pattern below worked again for slice 2.

## Decisions taken on 2026-09-12 (binding for later slices)

- **Cash is an integer number of USD cents** inside the ledger;
  `contract_cents` is the one rounding point; `NonIntegralCash` refuses
  what is not whole cents. Fee shares by cumulative rounding.
- **The simulated venue is shaped like Interactive Brokers**: a combo
  order fills in whole units or not at all (`GuaranteedCombo`), one
  fill per leg, commissions per contract as `Fee` events with a
  per-order minimum (numbers cited from IBKR's page when the model
  lands); net-price allocation and partial fills in whole units are
  later models; margin and rejections out of scope. Proposal section
  "Contract, venue, simplifications" has the text.
- **`record_order!` is slice 2's central piece**: plan every leg's
  fills and matches against the book as it would be after the earlier
  legs, validate all, then one `commit!`; the group is minted inside the
  transaction and not consumed on failure. Codex's proposed signature:
  `record_order!(L, book, order::Order; prices, effective_at,
  recorded_at, order_leg_ids, fill_rule)` with per-leg vectors.
- **Tests live beside the source they test**, one file per source file
  in the mirrored folder (`test/metrics/test_ledger_series.jl` stays).
- **Named failures, never bare errors**; every failure test checks it
  fires, leaves ledger and book untouched, and prints its name.
- **Commit at each checkpoint before the next mission**, split so that
  the human's doc decisions and a round's code are separate commits;
  the human confirmed this workflow. Nothing is pushed without the
  human saying so.
- Rule additions R1 to R4 (recorded time at or after effective time and
  nondecreasing; settlement price finite and non-negative; positive
  join ids) are in force; codex kept all four.

## Decisions taken on 2026-09-12, night (binding for later slices)

- **The engine computes, the ledger records.** The engine turns a
  decision into immutable inputs and makes one call; every id, the
  group, the order record and the events are minted inside
  `record_order!`. The engine holds no state beyond the ledger and the
  book it folds. The first slice 2 draft, which minted ids in two places
  and kept a journal in sync with the ledger, was rejected by the human
  as breaking the codebase's simplicity and near-immutability.
- **The order journal lives inside the `Ledger` container**, as
  `L.orders` beside `L.events`: two records, `OrderRecord` (embedding
  the `Order` the policy emitted, with `first_leg_id` and `known_to`)
  and `LegObservation`. Replays and cash fold `events` only; the ledger
  module still knows no quotes, spots or time cut.
- **No type hierarchy for the venue.** The price rule and the cost
  model are symbols dispatched through two tables in the
  `_METRIC_TABLE` style; the tick is an integer; slice 4 puts the three
  values in config and identity. Before proposing a struct, ask whether
  a symbol, a function or an existing type does the job.
- **R5: fill prices are on the venue's tick, rounded away from the
  trader** (USD 0.01 for SPY, QQQ, IWM).
- **Persistence landed with slice 2**: `events`, `orders`, `order_legs`
  parquet replace `positions.parquet`, schema version 3, `load_run`
  rebuilds through one `commit!` and `check_join`. Slice 6 keeps the
  derived tables (`round_trips`, `marks`, `equity`, `failures`), the
  completeness flag and `compare_runs.jl` over them.
- **Two `@test_broken` placeholders wait for slice 3** (an expiry inside
  the window is booked; the PR #9 regression becomes an `Expiry`
  against the lot's own underlying). They flip when slice 3 lands.
- Smaller: a commission of zero books no `Fee`; a duplicate execution
  id is `DuplicateExecution`; an unminted group is
  `DanglingReference(:group, g)`; a malformed `record_order!` call is
  `ArgumentError`; `DailyShortStrangle.quantity` is an `Int` and the
  config accepts `1` and `1.0`.

## Pull requests

Four PRs over the seven slices, cut where `master` is usable and the
diff reviewable. Agreed with the human on 2026-09-13.

1. **Slice 1 alone.** Cut a branch `ledger-slice1` at `3f3d7c9` (slice
   1, its fix and hardening rounds, this handoff as it then stood; it
   is exactly what origin holds for this branch) and open it against
   `master`. Additive: the ledger landed beside `positions`.
2. **Slices 2 and 3 together**, from this branch. Slice 2 alone would
   leave `master` with a strangle run that opens and never closes,
   since expiries are booked only in slice 3; the auditable strangle
   run at the end of slice 3 is the PR's own evidence. Open it as a
   draft with base `ledger-slice1` as soon as slice 2 is committed, so
   its diff shows slice 2 only and review can start; GitHub retargets
   it to `master` when PR 1 merges.
3. **Slice 4 alone.** The run-id break; small, and easy to point at
   later.
4. **Slices 5, 6 and 7 together.** Outputs only: the structure series
   and equity curve, the remaining tables, the docs cleanup that
   deletes the proposal, the briefs and the reviews.

Seven PRs would be too fine (item 2), one would be some 8,000 lines.
Each PR carries its codex review files and module docs current with
the code. Pushing is the human's call, as before.

## Read before acting

1. `docs/design.md`, all seven rules.
2. `docs/proposals/ledger.md`, the whole thing. Sections 2 and 3 are
   binding.
3. `docs/proposals/ledger-slice1.md`, the brief for the first slice.
4. `docs/status.md`, the in-flight entry and the backlog.

## The loop, per slice

1. **Brief.** Slice 1 has one. For each later slice, write
   `docs/proposals/ledger-slice<N>.md` in the same shape as slice 1's:
   read first, scope with an explicit do-not-touch list, files and
   public surface, tests with hand-computed literals, how to run here,
   done means. Derive it from the proposal's slice list and from the
   code the previous slice actually landed, not from memory.
2. **Implement** with a fresh agent, never a fork: the Agent tool with
   `subagent_type: "general-purpose"`, prompt = "Read and execute
   docs/proposals/ledger-slice<N>.md. Report what landed, the test
   count, and anything in the brief or proposal you had to interpret
   or found wrong." Run it on the branch, not in a worktree, so the
   `ws` tooling and the human's editor see the same tree. One agent
   at a time; the box has 3.7 GB.
3. **Gate.** When the agent reports, run the full suite yourself. Use
   `ws run "JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=. -e 'using Pkg; Pkg.test()'"`
   in the shell window (`ws test` sets no precompile variable). Do not
   wait on echoed text (`ws wait` false-matches the command); wait on
   the julia process exiting, then `ws capture shell 60`. Check `free -m` first; under about 1.2 GB
   available, exit the REPL in the `julia` window before running.
   Green means every existing test still passes plus the new ones.
4. **Codex review.** Write the review prompt to a file under the
   scratchpad, then in the codex window run
   `codex "$(cat <file>)"` via `tmux send-keys -t dev:codex ... Enter`
   from a bash prompt (if a codex session is already open, the text
   lands in its chat box instead; use `tmux load-buffer` +
   `paste-buffer -p` in that case, or exit the session first). Ask it
   to review the diff against the brief and the proposal for
   correctness, invariant coverage, and anything the brief left open,
   and to write its answer to `docs/proposals/ledger-slice<N>-review.md`
   in under 150 lines. Wait for the file with a Monitor that polls for
   the file and for the codex process; do not use a foreground sleep
   loop, it gets OOM-killed on this box.
5. **Report** to the human. They are usually on Remote Control, not
   attached to tmux (`tmux list-clients` is empty), so anything drawn
   in a pane is invisible: send files with SendUserFile. The report
   is: what landed (files, test count), the gate output's last lines,
   codex's verdict, your own read of the diff against the brief, and
   a proposed commit split. Then stop and wait for their go. Commit
   only when they say so, with the attribution trailer this session
   uses.
6. On their go, update `docs/status.md` (rule 4) if the agent did not,
   commit if asked, and start the next slice.

Slice 1 is reviewed together with the human. Later slices still stop
at the report; the human said they will wait for review time, not that
review is skipped.

## Slice order and where the checkpoints matter

1. `ledger` module (pure). Cheapest place to catch a misreading.
2. Engine switches to orders and the book; `positions` retired.
3. Lifecycle in the tick loop; then one auditable strangle run on the
   stored config before slice 4 breaks run ids. Show the human that run.
4. Config and identity; schema bump. Stored runs become unreadable.
5. Metrics on the structure series and the equity curve.
6. Persistence tables and load validation.
7. Docs; proposal, briefs and reviews deleted once landed.

## Box and tooling facts the agents need

- 2 cores, 3.7 GB, no sudo. `JULIA_NUM_PRECOMPILE_TASKS=1` for
  precompilation. Julia 1.12 via juliaup.
- Data lives at `~/data/massive/{options_1min,spots_1min}`; the stored
  ten-year strangle run is `5700d3f242f8132e`.
- The `ws` command drives the tmux windows; `ws open <file>:<line>`
  puts a file in the human's Neovim only when they are attached.
- Long jobs go in a pane, not the foreground. Background bash loops
  polling every few seconds have been killed by memory pressure;
  Monitor with a 10 s poll survived.
