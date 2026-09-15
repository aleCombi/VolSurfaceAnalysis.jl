# Persistence: keep the run, then reproduce it

This round gives the run folder two jobs: keep the inputs needed to run
an experiment again, and keep the outputs needed to inspect what it
produced. It finishes the persistence part of
[ledger-outputs.md](ledger-outputs.md) and brings the reproducibility
backlog into the library. The important addition is evidence: a rerun
must have an earlier result it can disagree with.

This is one round, one branch and one pull request to finish the ledger
rebuild. All three pieces land together, as three commits in this order:

1. Fix metric-parameter identity to use effective defaults, so omitted
   and explicit defaults share an output identity before persistence
   records witnesses under it.
2. Split persistence into loading the stored record and reproducing it,
   as specified below.
3. Sweep the docs, moving surviving proposal decisions into module docs
   and unfinished work into the backlog before retiring all proposals.
   The sweep is the last commit so the module docs describe what the
   branch actually landed.

The identity fix is twenty lines and the docs sweep is deletions and doc
moves; both sit in the same corner of the codebase as the persistence
split. Separate pull requests would cost three review cycles and two
rebases for no migration benefit, since nothing live is stored. The last
rebase produced four conflicts. The ordering belongs inside the branch.

## What the store says today

`src/persistence/store.jl` writes six files: verbatim `config.toml`, a
one-row `manifest.parquet`, long-form `metrics.parquet`, and the ledger's
`events.parquet`, `orders.parquet` and `order_legs.parquet`. Rendered
artifacts live separately under `artifacts/`. There is no stored curve,
failure table or dependency manifest.

On load, the config rebuilds the experiment and the three ledger tables
rebuild the ledger through its constructors, one batch through `commit!`,
and `check_join`. The loader then opens the current market data, builds
the marked curve and computes every metric again. It never reads
`metrics.parquet`. Only failure to open the data degrades the result:
the curve becomes `nothing` and all optional metrics disappear,
including `profit_factor`. Failures after opening propagate.

The manifest reader selects every column but retains only
`schema_version`, after checking that there is exactly one row. The
other fields mix code provenance and save time with identity, the human
name, the window and six derived counts. None of those counts is checked
against the loaded result. They are not entirely unused, though:
`scripts/compare_runs.jl` reads them and the saved metrics directly. The
notes' claim that no code reads the metrics is too strong. That script
compares two stored runs; it does not rerun either or compare a curve.

These files serve no single principle. The ledger is an output of the
experiment but an input to today's loader. Metrics are saved for SQL
while another equally consequential output, the curve, is discarded.
The manifest records summaries the loader ignores, while the dependency
versions needed to explain the result are missing. In particular,
`backtest/settlement.jl` uses the NYSE calendar supplied by `BusinessDays`:
a code commit alone does not pin the calendar the run consulted.

## Why the outputs must survive

Reproducibility needs inputs: the config, code version, dependency
versions and dataset identity. Outputs are not needed to produce the
answer again; they are needed to verify that it is the same answer.
Inspectability needs those outputs too, so a reader can see what the run
actually reported without running it again. Files belong here when they
serve one of these two goals; the manifest indexes them and records
their provenance.

**The reason to store outputs is evidence. Portability is explicitly
not a concern.** Loading on a machine without the data tree is not the
case driving this design. If loading first recomputes the curve and
failures, a reproduction check has lost the original witness. Two fresh
computations against today's code and tree can agree perfectly while
both differ from the recorded run. Agreement then proves nothing about
that run.

This deliberately reverses the recompute-on-load decision of
2026-09-14. Its ground was that a loaded result must agree with the
inputs available now. The new ground is that loading must preserve what
was observed then, and reproduction must test whether today's inputs
still produce it. The proposed rule change belongs explicitly in
`persistence.md` and `experiment.md`, including their key decisions and
failure descriptions, when the implementation lands. It strengthens the
vision's accumulated research record by keeping a claim available for
challenge instead of replacing it during inspection.

## The record this round writes

Keep two input documents, both in existing formats. `config.toml` stays
verbatim. Add `Manifest.toml`, copied verbatim from the environment that
produced the run, rather than resolved anew at save time. Together with
the recorded code checkout and its `Project.toml`, it lets Pkg restore
the dependency environment with `Pkg.instantiate`; persistence needs no
bespoke environment reader. A missing manifest must be named before
writing, not silently replaced with an empty document. `commit_sha` and
`dirty` remain scalar columns in `manifest.parquet`. A dirty flag records
uncertainty; it does not preserve an uncommitted patch.

This follows [Pkg's Project and Manifest documentation](https://pkgdocs.julialang.org/v1/toml-files/),
which describes the manifest as the resolved dependency record used
alongside the project. It is an environment record, not an archive of
local path dependencies or dirty source trees. Record this convention in
the persistence module's *Conventions consulted* section at implementation.

The output files are the three existing ledger tables, `metrics.parquet`,
and two new tables: `curve.parquet` and `failures.parquet`. The ledger
tables retain decisions, observations and events exactly as now. Metrics
retain their names and values, including meaningful NaN and infinities;
absence of a metric remains different from any numeric value.

`curve.parquet` records each marked session and its profit in USD, and
each unmarked instant and its reason. An unmarked row has no profit; it
is not zero, NaN or a carried-forward price. This is the existing
`MarkedCurve` in stored form, preserving its two pairs of vectors. It
replaces the older proposal's separate marks and equity exports. There
is no new per-lot mark history or account-value model.

`failures.parquet` records every unanswered question retained by a
completed run: its instant, subject and named reason. Settlement
failures must reach it from both lifecycle passes in `run_backtest`,
including the window-end pass. Both currently take only
`settlements(...).settled`; `.unsettled` is warned about and then lost.
It is a runtime observation, not something a ledger replay can recover.
Carry it through the engine result into `ExperimentResult` without
inventing ledger events for things that did not happen.

Marking needs the same care. Today's builder stops at the first
unpriceable lot and keeps only its reason on the session. Retain the
subjects and reasons while marking, and examine the remaining lots for
failures without reporting a partial portfolio value. A missing session
close names the underlying and session; an unpriceable lot names the
lot or contract with enough lineage to distinguish it. The curve still
has one unmarked entry per failed session, even when several lots failed.
Its unmarked entries and the corresponding failure records must agree.
Unexpected errors still propagate; this is no general exception catcher
and does not turn aborted runs into successful records with empty outputs.

Do not add `round_trips.parquet` in this round. It would serve a
legitimate goal if a concrete cross-run trade query needed it, but the
brief has no such consumer to justify another schema and comparison
surface. The ledger already records the trade facts, and the stored
metrics witness their reported reductions. This accepts a limit: this
round does not independently freeze every intermediate result of the
round-trip algorithm. Revisit the export when a real query earns its
place, not merely because it appeared in the earlier four-table list.

The manifest remains the run's index: run id, core hash, name, window,
counts, write time and schema version, with the existing code provenance.
Those fields do not require separate documents. Keep the six counts and
**check them on load** against the reconstructed ledger and stored curve.
Their cost is small and they can expose a truncated table or a mixed
save that still passes individual constructors. A mismatch must name the
column and both values. NULL curve counts mean no curve was recorded;
zero means a recorded curve has no entries of that kind. An absent curve
and a present but empty curve must remain distinguishable on disk.
Counts are consistency checks, not proof that prices or metrics are right.

Do not add the older proposal's completeness flag. No unmarked sessions
and no retained failures does not assert that every open lot was valued
at the window endpoint: the curve samples whole session closes, and the
endpoint need not be one. The counts and failure records say exactly
what was answered. A stronger endpoint assertion would require new
valuation work and belongs outside this persistence round.

## Loading and checking are different operations

`load_run` continues to return `ExperimentResult` and to validate the
ledger through its existing write path and join checks. It reads the
stored curve, failures and metrics into that result. It does not reopen
market data or recompute outputs, and missing required output files are
named load failures. It checks the manifest counts and structural
relationships, including curve/failure agreement. A plausible but stale
price can still load; checking its production is the next operation.

Add `reproduce(store, run_id)`. It reads the saved input documents and
provenance, rebuilds the experiment, reruns it against live data, and
compares the fresh outputs with the stored witness field by field. It
uses the running code and environment, reporting their provenance
alongside the recorded one; it does not secretly switch a checkout or
instantiate packages inside the caller's process. The copied manifest
makes a separate rerun under the recorded environment possible.

Comparison covers event order and fields, order records and observations,
curve instants and profits, failure subjects and reasons, and metric
names and values. Missing and extra rows count as divergence. Identify
each difference by output, row identity and field, with stored and fresh
values; a bare boolean or a generic failure is insufficient. Integers,
timestamps, identifiers and reasons compare exactly. Use a documented
absolute tolerance of `1e-9` for finite floating values, as the existing
comparison script does, with matching NaNs and same-sign infinities
handled explicitly. Non-finite values must never pass through an ordinary
subtraction test.

The report names success, divergence, or inability to reproduce. Missing
data is inability, never successful comparison of empty outputs. A
changed identity projection is a named identity mismatch, with the
projection regenerated for diagnosis rather than silently looking in a
new folder. `reproduce` never saves the fresh result over the witness.
Keep it read-only in this round; the backlog's optional refresh of
`commit_sha` and `dirty` after success can follow as an explicit utility.
Refreshing must never be a side effect of divergence, nor replace output
evidence or the originating dependency document.

The plain function name fits [Julia's convention](https://docs.julialang.org/en/v1/manual/style-guide/)
of reserving `!` for functions that modify their arguments. Keep it in
persistence, beside save and load, and put its tests in the matching
`test/persistence/` file. Record the naming convention in the module doc
when the API lands; no new module or type hierarchy is needed to describe
the report.

The Massive OHLCV trees are trusted as stable, and their schema is
trusted. Dataset versioning is dropped, not deferred: there will be no
fingerprint or dataset version in identity. Under that decision, a
divergence attributes to code or dependencies by elimination. The run
records both through `commit_sha`, `dirty` and the stored `Manifest.toml`;
a controlled rerun can separate their effects. This is attribution under
the data contract, not a content check performed by `reproduce`, and a
dirty flag still does not preserve the source changes it names.

## The boundary of the round

The persistence commit makes the schema **6**, with **no further identity
change or run id moves** after the metric-parameter identity fix in the
first commit. Older schemas are refused with instructions to rerun the
config;
there is no migration that can recover an unwritten curve or failure.
Re-saving still targets the same experiment folder. Atomic replacement
of that folder remains separate work; count checks do not make a
multi-file save atomic.

`experiment/identity.jl` makes the projection the real definition of
sameness. The hash is a lookup key over it: it can be computed before a
run, so asking whether a run exists is a folder-existence check and two
machines with the same resolved inputs agree without a coordinator.
Existence alone does not certify a complete save. The hash inherits the
projection's quality and hides its choices. Today the parquet specs'
`dataset` slot contains a root path, and that is the accepted identity
contract. Identical trees at different paths can have different ids; the
same path does not detect changed bytes. Neither code provenance nor
dependency versions enter the hashes.

Do not store a canonical projection file. It is deterministically
derived from the config by the corresponding code; regenerate it when
diagnosing an identity mismatch. Delete the dataset-fingerprint backlog
entry as decided rather than parking it again, and record the trusted-data
contract in the affected module docs in the final sweep. Metric-default
normalization lands in the first commit of this branch; the persistence
commit's no-identity-change promise is measured against that result.

This absorbs the second half's stored curve, failure table and engine
failure retention, and supersedes its export-only load contract. It
defers round-trip exports and the completeness flag for the reasons
above. Take the named-column parquet-write backlog here: name columns
at every insert site and remove event-specific positional NULL padding,
while keeping the single sparse events table and its column schema.
That cleanup does not itself require an identity or schema change.

The reproducibility harness becomes opt-in integration tests over this
function. The separate `scripts/revalidate_runs.jl` provenance-refresh
utility and extending `compare_runs.jl` to the new outputs remain later
work. Compute reuse, curation, official-close data, changed marking or
metric conventions, capital reporting and a second policy remain out of
scope. The final commit updates the affected module docs and status
entries, preserves unrelated unfinished work in the backlog, and retires
every proposal, this file included. `docs/proposals/` is empty at the end,
and `docs/status.md` records the ledger rebuild as finished.

## Tests

1. Save and load a small fixture with fills, matches, expiry, fees,
   orders and nullable observations. Check config and dependency
   manifest bytes verbatim, every ledger field, curve point, failure
   and metric. Include NaN and both infinities, a present empty curve,
   an explicitly absent curve, and an unmarked session with no profit.
   Loading must read the witness without invoking data or metric code.
2. Retain settlement failures from both lifecycle passes, multiple
   unpriceable lots in one session, and a missing session close. Check
   subject, instant and reason through save and load; count the session
   once and retain each failed question. No failure becomes a numeric
   placeholder, and unexpected exceptions still propagate.
3. Tamper with each manifest count, truncate a table, remove a required
   file and break curve/failure agreement. Loading must name the defect.
   Preserve the existing ledger corruption tests and exercise every
   event kind through the named-column writes. Reject schema 5 and a
   missing dependency manifest without fabricating evidence.
4. Reproduce an unchanged fixture, then independently change an event,
   observation, curve value, metric, failure reason and row membership.
   Check named field-level differences, tolerance boundaries and
   non-finite comparisons. Change live data while keeping the config
   fixed: load still returns the stored values and reproduction detects
   the change. Every path leaves all stored bytes untouched.
5. Pin both hashes for unchanged resolved experiments across the persistence
   commit, and check that provenance and dependency-document changes do not
   move them. Exercise the named identity-mismatch report without
   creating a replacement run. Opt-in tests rerun stored schema-6 runs
   where data is available; data-less environments explicitly skip,
   while an invoked reproduction reports inability rather than success.

Keep tests beside the corresponding source in the existing test tree.
For named exceptions, check the failure, unchanged state and printed
name. The box has 2 cores and 3.7 GB of RAM: use one Julia process at a
time, stopping any Julia REPL before a test or regression. Check available
memory with `free -m` first. Do not parallelize gates, targeted tests,
precompilation, regressions or artifact rendering. Finish other work
before launching a gate and limit activity to checking its log until it
exits.

For each code commit, use a fresh absolute log path in place
of `<log>`:

```sh
ws run "JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=. -e 'using Pkg; Pkg.test()' > <log> 2>&1; echo EXIT=\$? >> <log>"
```

Poll with `grep '^EXIT=' <log>` at short intervals so progress can still
be reported. No match means pending; only `EXIT=0` with a passing test
summary closes the gate. Never poll the tmux pane or use `pgrep` on a
pattern contained in the waiting command itself. The starting gate is
**3721 passed, 0 failed**; new tests should increase coverage, not be
removed to recover that number. Record the actual summary and log path
for each code commit.

After each code commit's gate exits, run the ten-year strangle separately,
without `--save`:

```sh
julia --project=. scripts/run_experiment.jl configs/strangle_spy_16d_1dte.local.toml
```

The supplied baseline is **13204 events, 2201 orders, USD 29942.23 cash,
sharpe 1.0389**, with `core_hash` `2bde5de695f9c90a` and `full_hash`
`6990a511c201c1aa`. Capture actual before and after ids during
implementation. Commit 1 changes `full_hash` and the run id, leaving
`core_hash` and schema 5 fixed. Commit 2's schema and hash boundary is
specified above. Both code commits must preserve all four numerical
figures; stored outputs and retained failure detail grow in commit 2,
without changing economic results. The documented curve has **2515
marked sessions and one unmarked session**, 2018-10-25T20:00:00 with
`:no_mark`; retaining more failure subjects must not increase the
session count. Commit 3 changes no runtime figures, identities or schema
and needs no Julia rerun.

These are supplied observations, not measurements made while writing
this brief. No Julia process is needed for this document consolidation,
and no implementation tests are claimed to have run.

## Done means

All three commits land together in one branch and one pull request:
metric-parameter identity fix, persistence split, then docs sweep.
Omitted and explicit metric defaults share an output identity; the fix
changes `full_hash` and the run id while preserving `core_hash`.

A saved run contains its original config and dependency manifest, its
ledger, curve, failures and metrics. Loading returns that record and
rejects structural inconsistencies, including wrong counts.
`reproduce` can disagree with the record by name and cannot overwrite it.
The tests demonstrate both unchanged reproduction and detected drift.

Schema 6 separates the persistence contract without further changing
either hash after the first commit.
Persistence, experiment and affected engine/metrics docs explain the
new boundaries. Surviving proposal decisions are in the module docs,
unrelated unfinished work is in the backlog, and `docs/proposals/` is
empty, this file included. `docs/status.md` records the ledger rebuild
as finished, with entry 7 and the relevant backlog entries reflecting
what landed and what remains. The dependency convention and API
naming have durable citations there. Both code commits' gates are green;
divergence attributes to code or dependencies by elimination under the
recorded trusted-data decision.
