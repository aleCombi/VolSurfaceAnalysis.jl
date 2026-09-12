# Ledger rebuild: orchestration handoff

Instructions for the session that drives the implementation. The
human wants to do nothing between review points. Your job is to run
the loop below, one slice at a time, and surface a report at each
review point.

## State when this was written (2026-09-11)

- Branch `claude/ledger-system-review-h9c8md` is checked out, tracking
  origin. Uncommitted: `docs/proposals/ledger.md` (the design, revised
  after four reviews) and the in-flight entry in `docs/status.md`.
- Untracked: `docs/proposals/ledger-slice1.md` (the slice 1 brief),
  `docs/proposals/ledger-events-review.md` and
  `docs/proposals/ledger-fill-review.md` (two codex reviews whose
  content is folded into the proposal; delete before the first commit),
  and this file.
- Nothing is committed and no code exists. The human decides commits.
- The `julia` tmux window holds a REPL with the package loaded and no
  state worth keeping. It may be exited to free memory; say so in the
  report when you do.
- The `codex` tmux window has an interactive codex session open from
  the reviews. Exit it (`/quit`) before starting a new one.

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
