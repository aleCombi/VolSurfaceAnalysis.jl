# `persistence` module

The knowledge base. Every run that lands here is written to a
content-addressed folder so prior experiments stay queryable and
comparable rather than evaporating into ad-hoc notebooks (see
[vision.md](../vision.md)).

## Storage shape

Hive-partitioned parquet under `<store_root>/runs/run_id=<hash>/`:

```
<root>/runs/
  run_id=deea7f6b1cf56779/
    config.toml          # INPUT:  raw input TOML bytes, verbatim
    Manifest.toml        # INPUT:  the resolved dependency record, verbatim
    manifest.parquet     # INDEX:  1 row of run-level metadata
    metrics.parquet      # OUTPUT: long: (run_id, metric_name, value)
    events.parquet       # OUTPUT: 1 row per ledger event, in sequence order
    orders.parquet       # OUTPUT: 1 row per order record
    order_legs.parquet   # OUTPUT: 1 row per order leg, with its observation
    curve.parquet        # OUTPUT: the marked curve; absent when there is none
    failures.parquet     # OUTPUT: what the run could not answer
    artifacts/           # rendered outputs (plots, ...), regenerable
      marked_curve.png
  run_id=8a1c.../
    ...
```

**A run folder has two jobs**, and every file serves one of them. It keeps
the *inputs* needed to run the experiment again -- the config, the
dependency record, and the code provenance on the manifest row -- and the
*outputs* needed to verify that a rerun produced the same answer. The
manifest indexes them and records their provenance. A file belongs here
when it serves one of those two goals and not otherwise.

The query substrate is parquet: every run-level table is parquet, so
any tool that reads it (DuckDB CLI, pandas, polars, a notebook in
another language) can browse the store. There is no separate index file
-- the manifest *is* the cross-run index, queried via the Hive glob.
Rendered binaries are quarantined under each run's `artifacts/` subdir
so they never sit as peers of the parquet; they are convenience
snapshots, regenerable from `load_run`, not the source of truth.

## Reader-shaped, but for output

`RunStore` mirrors the parquet readers' open/close pair: owns its
DuckDB connection, exposes a `close` / `with_run_store` scoped form,
and a finalizer cleans up. Same pattern, opposite direction (writes
instead of reads).

## Public surface

```julia
RunStore(root) :: RunStore
with_run_store(f, root)
Base.close(store) :: RunStore
Base.isopen(store) :: Bool

run_dir(store, run_id) :: String                       # path helper

save_run(store, result, config_toml;                   # returns run id = full_hash
         commit_sha="", dirty=true,
         manifest_toml=dependency_manifest()) :: String
load_run(store, run_id) :: ExperimentResult
reproduce(store, run_id) :: ReproductionReport
```

### The record is read back; reproducing it is a different operation

**The reason to store outputs is evidence.** Reproducibility needs inputs:
the config, the code version, the dependency versions. Outputs are not
needed to produce the answer again -- they are needed to verify that it is
the same answer. So `load_run` reads the record as it was written: the
ledger, the marked curve, the failures the run retained and the metrics it
reported. It recomputes nothing and **opens no market data**.

This reverses the recompute-on-load rule of 2026-09-14, deliberately. That
rule's ground was that a loaded result must agree with the inputs
available now. The new ground is that loading must preserve what was
observed *then*, because a reproduction check needs a witness it can
disagree with: two fresh computations against today's code and tree can
agree perfectly and both differ from the recorded run, and their agreement
would then prove nothing about that run. Under the old rule, inspecting
the witness destroyed it.

**Portability is explicitly not a concern.** Loading on a machine without
the data tree is not the case this design serves -- it simply works,
because nothing on the load path asks the data anything. There is no
degrade-by-piece path any more and no warning about an absent tree: a
loaded curve is the stored curve, or the run carried none.

The consequence to hold: a loaded result is a claim frozen at write time,
and a plausible but stale price still loads. Whether today's code and data
still produce it is what `reproduce` answers, and nothing else does.

`save_run` validates that `config_toml` rebuilds the saved
`result.experiment`: same `full_hash` and same human `name` label. That
keeps the manifest, saved config, and `load_run(...).experiment`
coherent even though `name` is deliberately excluded from identity. It
also runs the whole-ledger `check_join` before creating the run folder
or writing any file, and it takes the dependency manifest **verbatim from
the environment that produced the run** rather than re-resolving it at
save time. A missing one is `MissingManifest`, named before the folder
exists rather than silently written as an empty document.

### What the two output tables hold

`curve.parquet` is the existing `MarkedCurve` in stored form, preserving
both pairs of vectors: each marked session with its profit in USD, and
each unmarked instant with its reason. **An unmarked row has no profit** --
not zero, not NaN, not the previous session's number, but NULL. There is
no separate marks or equity export and no per-lot mark history: the curve
is the one path result.

`failures.parquet` records every unanswered question a *completed* run
retained -- its instant, its subject and its named reason. Two producers
fill it: the engine's lifecycle passes, for a lot left open because no
honest settlement price existed, and the marking pass, for a session close
the curve could not mark. Neither is a ledger event, because nothing
happened; no replay of the journal can recover one, so the run that
observed them is the only thing that can carry them out. The subject
carries enough lineage to tell two questions apart -- a lot names its
contract and its opening fill, a printless session names its underlying.

A broken session is counted **once** on the curve however many of its lots
failed, and each failed lot is its own row here. The two must agree on
instants **and** on reasons: the reason the curve carries for a session is
one of the reasons its lots gave, so it has to occur among the mark
failures recorded at that instant. `load_run` checks both. Instants alone
would accept a curve reporting `:no_mark` at a session whose every failure
says `:unexpected_gap` -- two tables describing two different runs.

### Counts, and NULL versus zero

The manifest keeps **one count per output table** and `load_run` **checks
every one** against what it rebuilt, naming the column and both values on a
mismatch. They are cheap and they catch the one defect the per-record
constructors cannot see: a truncated table, or a mixed save that every
individual check accepts. They are consistency checks, not proof that any
price or metric is right.

The coverage has to be complete, because a count is the only membership
evidence a table with no structural relationship has. The ledger counts
are checked against the rebuilt ledger and the curve counts against the
stored curve; `n_metrics` and `n_failures` are what keep an empty
`metrics.parquet` from reading as a run that reported no metrics, and a
deleted settlement failure -- which no curve entry answers for -- from
reading as a question nobody asked. Deleting one of two mark failures at a
session that keeps the other passes every structural check there is; only
the count knows. A truncated record is not an empty one (design rule 7).

NULL curve counts mean the run recorded no curve, and `curve.parquet` must
then be absent; zero means a recorded curve with no entries of that kind,
and the file must be present. An absent curve and a present-but-empty one
stay different facts on disk, in the file list and in SQL alike.
`n_metrics` and `n_failures` are never NULL: both tables are always
written, so zero means "asked, and nothing to record", and a manifest that
cannot say is itself the defect.

### `reproduce`: the other operation

`reproduce(store, run_id)` reads the saved inputs and provenance, rebuilds
the experiment, reruns it against live data, and compares the fresh
outputs with the stored witness field by field: event order and fields,
order records and their observations, curve instants and profits, failure
subjects and reasons, and metric names and values. A row present on one
side only is a divergence like any other. Each difference is identified by
output, row identity and field, with both values -- never a bare boolean.

Integers, timestamps, identifiers and reasons compare exactly. Finite
floating values use a documented absolute tolerance; NaN matches NaN and
an infinity matches the same-signed infinity, both settled **before** any
subtraction, because `NaN - NaN` answers nothing about whether two runs
agreed.

The report names one of three outcomes: reproduced, diverged, or could not
be reproduced. Inability is never reported as a successful comparison of
empty outputs. Data this machine cannot open is inability. So is a stored
config that no longer hashes to the folder it sits in: that is a named
identity mismatch carrying the regenerated projection for diagnosis, not a
quiet lookup of whatever run the new id points at.

It uses the running code and environment and reports **both** provenances
beside the recorded ones: the two commits, and the two dependency
documents by digest with every package whose version moved named. A report
that carried only the commits could not tell an unchanged rerun from one
whose calendar moved -- `backtest/settlement.jl` reads the NYSE calendar
`BusinessDays` ships, so identical commits over two environments are not
the same run. Differing dependencies are provenance, not divergence: they
never change the status, they say what a controlled rerun has to separate,
which is the whole of attribution-by-elimination under the trusted-data
decision. Reading the two documents is all it does with them: it does not
switch checkouts or instantiate packages inside the caller's process, and
the run's own `Manifest.toml` is what makes a separate rerun under the
recorded environment possible. **It never writes**: no fresh result is
saved over the witness, and refreshing a run's provenance after a
successful reproduction is a separate utility, never a side effect of
this one.

### The write and load paths validate, they never trust

The ledger is rebuilt through its own write path: every event is built
through its constructor in sequence order and committed to a fresh
`Ledger` as **one batch** through `commit!` (the book is empty, so FIFO
among batch-opened lots is sequence order), so a loaded ledger has
passed every append-time check the module makes. The order records are
rebuilt from the two order tables, every counter is set one past the
largest id seen (groups included), and `check_join` runs last. A load
that fails a check throws that check's named failure (`SequenceGap`,
`DuplicateExecution`, `MatchMismatch`, `DanglingReference`,
`JoinViolation`, ...); it never drops the join and never loads a
ledger that disagrees with itself.

A stored leg's own identity is checked against the order that claims it:
an order's legs are 1..n in `leg_idx`, and leg k's `order_leg_id` is the
order's `first_leg_id + k - 1` -- the arithmetic the writer used and the
ledger mints by. Both columns are stored facts, not sort keys, so a
rewritten id or indices shifted without changing their order is named
rather than sorted back into a ledger that looks untouched. Every leg row
is accounted for, too: a leg recorded against an order id no order claims
is a defect, not a row to drop, because dropping it would rebuild the
ledger from a subset of the table and return it as the record.

Whether the policy named an order's group or the ledger minted it is
not a column: the first record of a group (by order id) minted it,
since a group can only be named once minted, and only by a later order.

### Cross-run queries

There is no Julia query API by design. Read parquet with DuckDB
directly via the store's connection (or any other DuckDB session):

```julia
DBInterface.execute(store.con, """
    SELECT m.run_id, m.name, mt.value AS sharpe
    FROM '<root>/runs/*/manifest.parquet' m
    JOIN '<root>/runs/*/metrics.parquet'  mt USING (run_id)
    WHERE mt.metric_name = 'sharpe'
    ORDER BY sharpe DESC NULLS LAST
""")
```

`pandas.read_parquet("<root>/runs/*/manifest.parquet")` works
identically from Python.

## Run identity

The run id is `full_hash(result.experiment)` -- the canonical hash of
the *resolved* experiment (see [`experiment`](experiment.md)), not the
raw config bytes. Whitespace, comments, key order, the human `name`,
omitted-vs-explicit defaults, and machine cache knobs therefore do
**not** change the id. Re-saving the same experiment overwrites the
same folder in place.

The manifest also records `core_hash` -- the hash of the
backtest-determining inputs only (data, clock, agent, window, the venue's
two choices and the resolved contract facts). Two runs that
differ only in outputs (metrics / artifacts) share a `core_hash` but get
distinct `run_id`s, so output variations of one backtest are detectable
in a cross-run query:

```julia
DBInterface.execute(store.con, """
    SELECT run_id, name FROM '<root>/runs/*/manifest.parquet'
    WHERE core_hash = '<some core_hash>'
""")
```

Both hashes come from `to_dict`, an identity projection (in the
experiment module) that emits one entry per data kind, the clock, the
agent, the window, the venue's two choices and the contract facts
resolved for the experiment's underlying, omitting non-result-affecting
fields (cache sizes, readers, part order). They also project the bar-stamp
convention as the constant `"bar_end"`, which is what moved every run id
when bar-end visibility landed.

**The projection is the definition of sameness; the hash is a lookup key
over it.** It can be computed before a run, so asking whether a run
already exists is a folder-existence check and two machines with the same
resolved inputs agree on the id without a coordinator. Existence alone
does not certify a *complete* save, which is what the manifest counts are
for. The projection is not stored: it is deterministically derived from
the config by the code that reads it, so it is regenerated when diagnosing
an identity mismatch rather than kept as a file that could go stale.

**The `dataset` slot holds a root path, and that is the accepted
contract.** Dataset versioning is dropped, not deferred: the Massive OHLCV
trees are trusted as stable and their schema is trusted, so there will be
no fingerprint and no dataset version in identity. Live with both edges of
that: identical trees at two paths get different ids, and the same path
does not detect changed bytes. Under this decision a divergence found by
`reproduce` attributes to code or dependencies by elimination -- the run
records both, through `commit_sha`, `dirty` and the stored
`Manifest.toml`, and a controlled rerun separates their effects. That is
attribution under the data contract, not a content check `reproduce`
performs. Neither code provenance nor dependency versions enter either
hash.

**Runs written under bar-open visibility do not reproduce.** They keep
their own ids -- the corrected code hashes the same config to a different
one -- so nothing collides and no schema version separates them; the id
break *is* the separation. Their ledgers are results of a backtest whose
data layer served a minute before it had finished, and they cannot be
reused as results of the corrected one. The loss of reproduction is
accepted and recorded in [status](../status.md). The verbatim `config.toml` is still
stored -- for reading and for rebuilding the experiment on load -- but
it is not what identity is computed from.

**Schema versions.** The manifest carries a `schema_version` outside
the hash; `load_run` refuses a run whose version is absent or differs,
with a message that says to rerun its config. Version 2 was the
data-kinds migration (every run id changed with the identity
projection). Version 3 was the ledger: `positions.parquet` gave way to
`events`, `orders` and `order_legs`. Version 5 is the marked curve:
`pnl_series.parquet` and the manifest's `window_end_spot` left with the
`PnLSeries` type they exported and `n_marked` joined `n_unmarked`.
Version 6 is the persistence split: `Manifest.toml`, `curve.parquet` and
`failures.parquet` arrive and the load path stops recomputing. Version 7
completes the index over them: `n_metrics` and `n_failures` join the
manifest, so every output table has membership evidence and a truncated
one stops reading as an empty one. It refuses 5 and 6 like every other
older version, and there is no migration -- no earlier schema holds a
curve or a failure table to migrate *from*, a schema-6 manifest never
wrote down the two counts, and a run's witness cannot be invented after
the fact. Version 4 is the identity break --
the venue's two choices and the resolved contract facts joined
`core_hash`, so every stored run id moved. It is also what separates the
schema-3 tree from today's code: those runs were written before the
engine booked expiries, and their ids do not say so, since a run id
records the experiment and `commit_sha` records the code. The version is
the only thing on disk that refuses them. `scripts/compare_runs.jl`
compares runs written under the current version only. No migration
script: the store held one run each time.

## Responsibility boundaries

**Owns:** storage layout, parquet schemas for the output tables, the
`artifacts/` subdir, the DuckDB connection used for writing, the load
path's rebuilding of the ledger through the ledger's own checks, the
record's internal consistency (the manifest counts, the stored legs'
identity against the orders that claim them, and curve/failure agreement
on instants and reasons), and the comparison `reproduce` performs.

**Does NOT own:**

- Computing identity. `save_run` calls `full_hash` / `core_hash`; the
  canonical `to_dict` projection lives in the [`experiment`](experiment.md)
  module.
- The checks themselves. `commit!` and `check_join` belong to
  [`ledger`](ledger.md) and [`backtest`](backtest.md); this module only
  routes a loaded run through them.
- Querying. SQL on the parquet tree is the API.
- Code provenance capture. `commit_sha` / `dirty` are produced by
  `code_provenance` and passed into `save_run` by the caller.
- Producing the outputs it stores. The curve and the failures are the
  run's, built by [`metrics`](metrics.md) and [`backtest`](backtest.md)
  while the data was open; `reproduce` re-runs the experiment through
  `run_experiment` and does not reimplement any part of it.
- Restoring a dependency environment. It stores `Manifest.toml` verbatim
  and reads it for one thing only -- naming the recorded environment, and
  the packages whose versions moved, in a reproduction report.
  `Pkg.instantiate` against the recorded checkout is the restore
  operation, and it is performed outside the caller's process.
- Artifact rendering. The store records what was written; rendering is
  script-level (`scripts/lib/artifacts.jl`), so the core stays Plots-free.
- Atomicity guarantees beyond best-effort. A crash mid-`save_run` can
  leave a half-written folder; re-running the same experiment recovers
  it. Write-to-temp-then-rename is queued.

## Schemas

Timestamps are written with millisecond precision, so a loaded ledger
equals the saved one exactly. `group` is a SQL keyword: the column is
`group_id` everywhere.

### `manifest.parquet`

| column | type | notes |
|---|---|---|
| `run_id` | VARCHAR | `full_hash`, also the partition key |
| `core_hash` | VARCHAR | backtest-only identity (shared by output variations) |
| `name` | VARCHAR | from `Experiment.name` (label; not part of identity) |
| `from_ts` | TIMESTAMP | evaluation window start |
| `to_ts` | TIMESTAMP | evaluation window end |
| `n_events` | BIGINT | `length(result.ledger)` |
| `n_orders` | BIGINT | `length(result.ledger.orders)` |
| `n_opens` | BIGINT | `Open` fills in the ledger |
| `n_closes` | BIGINT | `Close` fills in the ledger |
| `n_marked` | BIGINT | session closes the run marked; NULL when it carries no curve |
| `n_unmarked` | BIGINT | session closes it could not mark; NULL when it carries no curve |
| `n_metrics` | BIGINT | metrics the run reported; never NULL |
| `n_failures` | BIGINT | questions the run left unanswered; never NULL |
| `commit_sha` | VARCHAR | git commit of the code that produced the run |
| `dirty` | BOOLEAN | working tree had uncommitted changes |
| `written_at` | TIMESTAMP | UTC time of the save |
| `schema_version` | INTEGER | manifest schema version (`RUN_SCHEMA_VERSION`, currently 7); outside the hash |

`n_marked` / `n_unmarked` are NULL rather than 0 when the saved result had
no curve: "no curve at all" and "a curve that marked nothing" are different
facts and must not read the same in SQL. `n_metrics` / `n_failures` have
no such case -- both tables are always written -- so a NULL there is a
defective index and `load_run` says so.

### `metrics.parquet`

Long form so the schema is stable as metrics come and go.

| column | type |
|---|---|
| `run_id` | VARCHAR |
| `metric_name` | VARCHAR |
| `value` | DOUBLE |

`NaN` and `±Infinity` are stored verbatim (cast via `'NaN'::DOUBLE` on
write), because a NaN sharpe is information about the run. Absence of a
row still means "not computed", which is different from every number
including NaN. `load_run` reads this table back as the metrics the run
reported; the column is DOUBLE, so an integer metric returns as a Float64
of the same value, which compares equal and is the whole of what a metric
is. A metric that does not reduce to a number is refused at save time
rather than written as a NaN that would claim it was computed.

### `curve.parquet`

The `MarkedCurve` in stored form, both pairs of vectors, marked rows
first. **Absent entirely when the run carried no curve**, which the
manifest's NULL counts say too.

| column | type | notes |
|---|---|---|
| `run_id` | VARCHAR | |
| `kind` | VARCHAR (`'marked'` / `'unmarked'`) | which pair the row belongs to |
| `instant` | TIMESTAMP | the session close (`at` is a DuckDB keyword) |
| `profit` | DOUBLE | marked profit in USD; **NULL on an unmarked row** |
| `reason` | VARCHAR | why it could not be marked; NULL on a marked row |

### `failures.parquet`

One row per unanswered question the run retained, in the run's canonical
order (instant, then stage, then subject, then reason). Always written;
empty means the run asked and nothing went unanswered.

| column | type | notes |
|---|---|---|
| `run_id` | VARCHAR | |
| `instant` | TIMESTAMP | when the question was asked |
| `stage` | VARCHAR (`'settlement'` / `'mark'`) | which pass asked it |
| `subject` | VARCHAR | the lot, contract or underlying it was about |
| `reason` | VARCHAR | the `UnpriceableLeg` name |

### `events.parquet`

One row per event in sequence order: the header, the kind, and the
kind's own columns; everything a kind lacks is NULL.

| column | type | fill | match | expiry | fee |
|---|---|---|---|---|---|
| `run_id` | VARCHAR | x | x | x | x |
| `sequence`, `id` | BIGINT | x | x | x | x |
| `kind` | VARCHAR (`'fill'`, `'match'`, `'expiry'`, `'fee'`) | x | x | x | x |
| `effective_at`, `recorded_at` | TIMESTAMP | x | x | x | x |
| `group_id` | BIGINT | x | x | x | |
| `order_leg_id`, `execution_id` | BIGINT | x | | | |
| `underlying` | VARCHAR | x | | x | |
| `strike` | DOUBLE | x | | x | |
| `expiry` | TIMESTAMP | x | | x | |
| `option_type` | VARCHAR (`'C'` / `'P'`) | x | | x | |
| `side` | VARCHAR (`'long'` / `'short'`) | x | | x | |
| `intent` | VARCHAR (`'open'` / `'close'`) | x | | | |
| `quantity` | BIGINT | x | x | x | |
| `price` | DOUBLE | x | | | |
| `fill_rule` | VARCHAR | x | | | |
| `open_fill_id` | BIGINT | | x | x | |
| `close_fill_id` | BIGINT | | x | | |
| `settlement_price` | DOUBLE | | | x | |
| `outcome` | VARCHAR (`'worthless'` / `'cash_settled'`) | | | x | |
| `source_id` | BIGINT | | | | x |
| `amount` | BIGINT (cents) | | | | x |

### `orders.parquet`

One row per order record.

| column | type |
|---|---|
| `run_id` | VARCHAR |
| `order_id` | BIGINT |
| `first_leg_id` | BIGINT |
| `label` | VARCHAR |
| `group_id` | BIGINT (the group minted or named) |
| `operation` | BIGINT (nullable) |
| `decided_at` | TIMESTAMP |
| `known_to` | BIGINT |

### `order_legs.parquet`

One row per order leg: the leg and the observation it was priced
against. `bid` and `ask` keep the source's nullability.

| column | type |
|---|---|
| `run_id` | VARCHAR |
| `order_id`, `order_leg_id`, `leg_idx` | BIGINT |
| `underlying` | VARCHAR |
| `strike` | DOUBLE |
| `expiry` | TIMESTAMP |
| `option_type` | VARCHAR (`'C'` / `'P'`) |
| `side` | VARCHAR (`'long'` / `'short'`) |
| `intent` | VARCHAR (`'open'` / `'close'`) |
| `quantity` | BIGINT |
| `quote_at` | TIMESTAMP |
| `bid`, `ask` | DOUBLE (nullable) |
| `spot` | DOUBLE |
| `spot_at` | TIMESTAMP |

## Key decisions

| Decision | Why |
|---|---|
| **Parquet + DuckDB-as-engine, no single-file DB** | Matches the `data` module's existing pattern. Files are inspectable from any parquet-aware tool; a single corrupt run doesn't take down the whole store; `rm -rf <run_dir>` is a valid delete. |
| **Hive partition `run_id=<hash>`** | DuckDB and pandas / polars all understand the layout natively. Cross-run queries are one parquet glob, no separate manifest table to keep in sync. |
| **The ledger stored as it is: events with the kind's own columns, plus the order journal** | The events are the facts and the journal is what each decision saw; storing a derived table *in place of* them would let a stored run disagree with its own replay. The curve and the failures are stored beside them, not instead. |
| **Outputs are stored as evidence, and read back rather than recomputed** | A stored derived table is a claim frozen at write time, and that is exactly what a reproduction check needs to disagree with. Recomputing on load destroys the witness: two fresh computations can agree perfectly and both differ from the recorded run. `load_run` returns what was observed then; `reproduce` asks whether today's code and data still produce it. Reverses the 2026-09-14 rule. |
| **No `round_trips.parquet`** | It would serve a real goal once a concrete cross-run trade query needs it, and there is no such consumer. The ledger already records the trade facts and the stored metrics witness their reported reductions. The limit accepted: this round does not independently freeze every intermediate result of the round-trip algorithm. Revisit when a query earns the schema, not because it appeared in an earlier four-table list. |
| **No completeness flag** | "No unmarked sessions and no retained failures" would not assert that every open lot was valued at the window endpoint: the curve samples whole session closes and the endpoint need not be one. The counts and the failure records already say exactly what was answered; a stronger endpoint assertion needs new valuation work. |
| **The dependency manifest is copied, not re-resolved** | `Manifest.toml` beside the project is Pkg's own record of the environment that ran. Re-resolving it at save time would record what today's registry picks, not what the run loaded. It is an input document this module does not interpret -- with the code checkout and its `Project.toml`, `Pkg.instantiate` restores the environment, so persistence needs no bespoke environment reader. |
| **A failure is a row, never an event** | A lot that could not be settled and a session that could not be marked are things that did *not* happen. Inventing ledger events for them would put fictions in the journal of facts, and warning about them loses them when the run ends. They ride out on the result and land in their own table. |
| **Named columns at every insert** | A positional `VALUES (...)` list makes a miscount write a value into the neighbouring column of the right type, and nothing in the type system catches it. Naming the columns removes the event table's per-kind NULL padding too, since DuckDB nulls what an insert does not name. The bytes written are identical, so it is not a schema change. |
| **Write and load validate the join; load rebuilds through `commit!`** | `save_run` checks before creating a folder. A load must fail on a dangling fill, duplicate execution id, field mismatch or invalid simulated price, never drop the join. Reusing `commit!` means append invariants are not copied. |
| **`load_run` returns `ExperimentResult`, not a separate `StoredRun`** | Same type as `run_experiment` means same recipes / `show` / downstream consumers, and it is what lets `reproduce` compare the two sides with one set of accessors. Specs are pure values, so the rebuilt experiment only touches the data when it is actually run. |
| **The reproduction report is two flat records, not a hierarchy** | A `ReproductionReport` holding `Divergence` rows says everything the report has to say: an outcome, why when there is no outcome to compare, both code provenances, both dependency environments with the packages that moved, and each difference by output, row and field. No abstract type, no per-output subtype -- a table of differences is data, and a list of changed versions is strings. |
| **A reproduction report names both environments, not only both commits** | Dataset versioning is deliberately out of identity, so a divergence attributes to code or dependencies *by elimination* -- which fails outright if the report cannot tell two environments apart. Identical commits with different dependencies are ordinary here: the settlement rule consults the calendar a dependency ships. The digests identify each document the way a sha identifies a checkout, and the version map is what lets a difference be named package by package instead of as two opaque hashes. Differing dependencies are provenance, never divergence: the status still reports whether the outputs agreed. |
| **Long-form `metrics.parquet`** | Optional metrics come and go per run; a wide schema would force columns to NULL across runs and break naive `UNION ALL` reads. Long form is stable and trivially pivotable. |
| **NaN / Inf preserved, not nulled** | A NaN sharpe (e.g. one trade, zero variance) is meaningful information about that run; collapsing it to NULL would lose the distinction from "metric not requested." |
| **Caller passes the TOML bytes** | The TOML is the source of truth for what was run; pushing the bytes through `save_run` keeps the persistence layer ignorant of how the `Experiment` was built and avoids stashing config strings on `Experiment` itself. |
| **DuckDB connection per store, exposed as `store.con`** | Same as the parquet readers. Lets viz / notebook code query without spawning a second DuckDB session. |
| **Best-effort atomicity** | A write-to-tmp-then-rename pass is the right fix, but it complicates the first slice. The recovery story is "re-run the same config," which works because identity is content-addressed. |

## Future work

- Opt-in, data-gated integration tests that `reproduce` every stored
  schema-7 run, skipping cleanly where the source data is absent.
- `scripts/revalidate_runs.jl`: refresh a run's `commit_sha` / `dirty`
  after a *successful* reproduction. Read-only stays the rule for
  `reproduce` itself -- a refresh must never be a side effect of
  divergence, and must never replace the output evidence or the
  originating dependency document.
- `compare_runs.jl` over the curve and the failures, not only the manifest
  and the metrics.
- Write-to-temp-then-rename for atomic saves. The manifest counts catch a
  mixed save on load; they do not make a multi-file save atomic.
- Compute reuse: on a `core_hash` + `commit_sha` hit with a clean tree,
  load the cached ledger and recompute only the outputs instead of
  re-running the backtest.
- A curation gate (`status` draft / accepted / retracted with
  `accept_run` / `retract_run`), cross-run queries defaulting to accepted.
- Cross-run viz recipes built on the SQL queries demonstrated above.
- Delete / archive helpers (`drop_run(store, id)`). For now a manual
  `rm -rf <run_dir>` is the API.

## Conventions consulted

| Decision | Source checked | Finding |
|---|---|---|
| `Manifest.toml` stored verbatim as the run's dependency record, beside the project rather than as a bespoke environment dump | [Pkg's Project and Manifest documentation](https://pkgdocs.julialang.org/v1/toml-files/) (checked 2026-09-15) | Pkg describes the manifest as the *resolved* dependency record used alongside the project: the exact versions an environment loaded. It is an environment record, not an archive of local path dependencies or dirty source trees -- so copying it records what ran, and `Pkg.instantiate` against it plus the recorded checkout restores that environment. A missing one is therefore a real gap, named rather than written as an empty document. |
| Reading a stored `Manifest.toml` for provenance: which key holds the resolved versions | [Pkg's Project and Manifest documentation](https://pkgdocs.julialang.org/v1/toml-files/) (checked 2026-09-15) | `manifest_format` 2.0 nests the resolved packages under a `deps` table (`[[deps.<Name>]]` blocks, each with `version` for a registered package and none for a stdlib); format 1.0 puts the same blocks at the top level, and both carry `julia_version`. So a report reads `deps` when it is there and the top level otherwise, and a package with no version contributes nothing to name. This is provenance only -- restoring an environment is still `Pkg.instantiate` outside the process. |
| `reproduce` as a plain verb, no `!`, living in persistence beside save and load | [Julia style guide](https://docs.julialang.org/en/v1/manual/style-guide/) (checked 2026-09-15) | `!` is reserved for functions that modify their arguments. `reproduce` reads the store, reruns, and writes nothing, so the bang would be a false promise in the other direction. Its tests sit in the matching `test/persistence/` file, following the mirrored-tree layout the rest of the suite uses. |
