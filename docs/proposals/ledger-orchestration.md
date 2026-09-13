# Ledger rebuild: orchestration handoff

Instructions for the session that drives the implementation. The human
wants to do nothing between review points. Your job is to run the loop
below, one slice at a time, and surface a report at each review point.

## State as of 2026-09-13

- **PR #11, the ledger module**, is the branch `ledger-slice1` against
  `master`: slice 1 and its fix and hardening rounds, then the order
  journal (`L.orders`, `OrderRecord`, `LegObservation`,
  `record_order!`) and the ledger-owned book. Additive -- `positions`
  still drives the backtest -- so `master` stays usable when it lands.
- **The wiring** sits on `claude/ledger-system-review-h9c8md`: the
  engine, the simulated venue, `fill_legs`, the join check, policies,
  agents, persistence at schema 3, and the retirement of `positions`.
  It is cut into a pull request only once PR #11 merges, so it is cut
  once, against the ledger that actually landed. PR #12 held an earlier
  form of it and was closed for that reason.
- `backup/pre-split` holds the tip as it stood before the two were
  separated. The wiring delta is `git diff <ledger tip> backup/pre-split`.
- The `julia` window holds a REPL with Revise and the package loaded, no
  state worth keeping. The `codex` window is at a bash prompt.

## Decisions taken on 2026-09-12 (binding)

- **Cash is an integer number of USD cents** inside the ledger;
  `contract_cents` is the one rounding point; `NonIntegralCash` refuses
  what is not whole cents. Fee shares by cumulative rounding.
- **The simulated venue is shaped like Interactive Brokers**: a combo
  order fills in whole units or not at all, one fill per leg,
  commissions per contract as `Fee` events with a per-order minimum;
  net-price allocation and partial fills in whole units are later
  models; margin and rejections out of scope.
- **`record_order!` is the structure writer**: plan every leg's fills
  and matches against the book as it would be after the earlier legs,
  validate all, then one `commit!`; the group is minted inside the
  transaction and not consumed on failure.
- **Tests live beside the source they test**, one file per source file
  in the mirrored folder.
- **Named failures, never bare errors**; every failure test checks it
  fires, leaves the ledger untouched, and prints its name.
- **Commit at each checkpoint before the next mission**, splitting the
  human's doc decisions from a round's code. Nothing is pushed without
  the human saying so.
- Rule additions R1 to R4: recorded time at or after effective time and
  nondecreasing; settlement price finite and non-negative; positive
  join ids.

## Decisions taken on 2026-09-12, night (binding)

- **The engine computes, the ledger records.** The engine turns a
  decision into immutable inputs and makes one call; every id, the
  group, the order record and the events are minted inside
  `record_order!`. An earlier draft that minted ids in two places and
  kept a journal in sync with the ledger was rejected as breaking the
  codebase's simplicity.
- **The order journal lives inside the `Ledger` container**, as
  `L.orders` beside `L.events`. Replays and cash fold `events` only;
  the ledger module knows no quotes, spots or time cut.
- **No type hierarchy for the venue.** The price rule and the cost
  model are symbols dispatched through two tables; the tick is an
  integer. Before proposing a struct, ask whether a symbol, a function
  or an existing type does the job.
- **R5: fill prices are on the venue's tick, rounded away from the
  trader** (USD 0.01 for SPY, QQQ, IWM).
- **Persistence at schema 3**: `events`, `orders`, `order_legs` parquet
  replace `positions.parquet`; `load_run` rebuilds through one
  `commit!`; the join is validated on write and on load.
- Smaller: a commission of zero books no `Fee`; a duplicate execution
  id is `DuplicateExecution`; an unminted group is
  `DanglingReference(:group, g)`; a malformed `record_order!` call is
  `ArgumentError`.

## Decisions taken on 2026-09-13 (binding)

- **The `Ledger` owns its book.** No writer takes a `Book`; a ledger
  built from events folds its own. The read side is unchanged: `Book`
  stays a type and both replays return standalone values. `apply!` is
  internal.
- **Describe the boundary, never claim impossibility.** Three reviews
  in a row caught this codebase claiming an absolute the code cannot
  deliver -- "nothing after `commit!` can fail", the incremental book
  equalling the replay "by construction", divergence being
  unrepresentable. Julia has no private fields. State the guarantee and
  its edge.
- **The join is checked at every engine append**, per record, not once
  at the end of a run: the whole-ledger form is quadratic through
  `order_leg`'s scan. The whole-ledger form stays for `save_run`,
  `load_run` and the tests.
- **Pull requests are cut by module, not by slice** (below).

## Pull requests

Cut where `master` stays usable and the diff is one thing.

1. **The ledger module.** Everything that defines the ledger: the event
   journal, both replays, round trips, the order journal, the owned
   book, `test/ledger/`, `docs/modules/ledger.md`. Open as #11.
2. **The wiring.** Everything that uses it: engine, venue, `fill_legs`,
   the join check, policies, agents, persistence, the metrics adapter,
   the retirement of `positions` -- plus the lifecycle, since the
   wiring alone would leave `master` with a strangle run that opens and
   never closes. The auditable strangle run at its end is its own
   evidence. Cut after #11 merges.
3. **Config and identity.** The run-id break; small, and easy to point
   at later.
4. **Outputs.** The structure series and equity curve, the remaining
   derived tables, and the docs sweep that deletes this proposal.

An earlier plan cut these by slice, which put ledger changes in three
different pull requests and had a reviewer reviewing a shape that never
shipped. Working notes are retired in the PR that lands their work, not
carried to `master` for a later sweep.

## Read before acting

1. `docs/design.md`, all seven rules.
2. `docs/proposals/ledger.md`, sections 2 and 3.
3. `docs/status.md`, the in-flight entry and the backlog.
4. `docs/modules/ledger.md`, for the module as it now stands.

## The loop, per slice

1. **Brief.** Write `docs/proposals/ledger-<name>.md`: read first, scope
   with an explicit do-not-touch list, files and public surface, tests
   with hand-computed literals, how to run here, done means. Derive it
   from the code the previous round actually landed, not from memory.
2. **Implement** with a fresh agent on the branch, never a fork and
   never a worktree, so the `ws` tooling and the human's editor see the
   same tree. One at a time; the box has 3.7 GB.
3. **Gate.** Run the full suite yourself:
   `ws run "JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=. -e 'using Pkg; Pkg.test()'"`
   (`ws test` sets no precompile variable). Do not wait on pane text --
   a grep for your own marker matches the echoed command line. Wait on
   the julia process exiting, then `ws capture shell 60`. Check
   `free -m` first; under about 1.3 GB available, exit the REPL in the
   `julia` window before running, and relaunch it after.
4. **Codex review.** Write the prompt to a file under the scratchpad,
   then in the codex window run
   `codex exec --dangerously-bypass-approvals-and-sandbox "$(cat <file>)"`
   from a runner script that tees to a log and appends a `CODEX_EXIT=`
   marker, watched by a Monitor. Tell it what the brief decided and
   why, or it re-files closed findings. Ask for a verdict line first.
5. **Report** to the human. They are usually on Remote Control, not
   attached to tmux (`tmux list-clients` is empty), so anything drawn
   in a pane is invisible: send files with SendUserFile. The report is
   what landed, the gate's last lines, codex's verdict, your own read
   of the diff, and a proposed commit split. Then stop.
6. On their go, update `docs/status.md` (rule 4), commit, and start the
   next round.

## What remains

1. The wiring, with the lifecycle: `Expiry` in the tick loop, the
   session calendar, window-end lifecycle at the evaluation endpoint.
   Then one auditable strangle run on the stored config.
2. Config and identity; schema bump. Stored runs become unreadable.
3. Metrics on the structure series and the equity curve.
4. The remaining persistence tables and the completeness flag.
5. Docs; this proposal and the briefs deleted once landed.

## Box and tooling facts

- 2 cores, 3.7 GB, no sudo. `JULIA_NUM_PRECOMPILE_TASKS=1` for
  precompilation. Julia 1.12 via juliaup.
- Data lives at `~/data/massive/{options_1min,spots_1min}`; the stored
  ten-year strangle run is `5700d3f242f8132e`.
- The `ws` command drives the tmux windows; `ws open <file>:<line>`
  puts a file in the human's Neovim only when they are attached.
- Long jobs go in a pane, not the foreground. Background bash loops
  polling every few seconds have been killed by memory pressure;
  Monitor with a 20 s poll survives.
