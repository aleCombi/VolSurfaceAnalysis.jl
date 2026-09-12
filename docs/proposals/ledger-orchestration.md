# Ledger rebuild: orchestration handoff

Instructions for the session that drives the implementation. The
human wants to do nothing between review points. Your job is to run
the loop below, one slice at a time, and surface a report at each
review point.

## State as of 2026-09-12, evening

- Branch `claude/ledger-system-review-h9c8md`, tracking origin, six
  commits ahead, nothing pushed. Tree clean except two untracked review
  inputs, `docs/proposals/ledger-events-review.md` and
  `ledger-fill-review.md`, folded into the proposal long ago; the human
  has not yet said to delete them.
- Landed and committed: slice 1 (the pure `ledger` module), its fix
  round (`ledger-slice1-fix.md`, codex review `-fix-review.md`), and a
  hardening round (`ledger-slice1-hardening.md`, codex review
  `-hardening-review.md`, inventory `ledger-slice1-coverage.md`). Gate:
  2306 passed, 0 failed, 1 broken. The one Broken is deliberate: the
  `@test_broken` structure-atomicity testset at the end of
  `test/ledger/test_append.jl`, which waits for slice 2's
  `record_order!`; when that writer lands the test records an
  unexpected pass and must be flipped to `@test`.
- Next: slice 2. Write `docs/proposals/ledger-slice2.md` in the shape of
  the slice 1 brief, derived from the code as it stands, then run the
  loop. See "Decisions taken on 2026-09-12" below; they are binding.
- The `julia` tmux window holds a REPL with Revise and the package
  loaded, no state worth keeping. Exit it before the gate when under
  about 1.3 GB available; relaunch it after
  (`julia --project=. -e 'using Revise' -i`, then `using VolSurfaceAnalysis`).
- The `codex` tmux window is at a bash prompt. Headless codex worked
  well: `codex exec --dangerously-bypass-approvals-and-sandbox "$(cat
  prompt)"` from a runner script that tees to a log and appends a
  `CODEX_EXIT=` marker, watched by a Monitor; codex writes its review to
  `docs/proposals/ledger-slice<N>-review.md`.

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
   `ws test` in the shell window. Do not wait on echoed text (`ws wait`
   false-matches the command); wait on the julia process exiting, then
   `ws capture shell 60`. Check `free -m` first; under about 1.2 GB
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
