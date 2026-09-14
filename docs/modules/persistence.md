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
    config.toml          # raw input TOML bytes, verbatim
    manifest.parquet     # 1 row of run-level metadata
    metrics.parquet      # long: (run_id, metric_name, value)
    events.parquet       # 1 row per ledger event, in sequence order
    orders.parquet       # 1 row per order record
    order_legs.parquet   # 1 row per order leg, with its observation
    artifacts/           # rendered outputs (plots, ...), regenerable
      marked_curve.png
  run_id=8a1c.../
    ...
```

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
         commit_sha="", dirty=true) :: String
load_run(store, run_id) :: ExperimentResult
```

### Inputs are read back; anything derived is recomputed

The ledger and the config are the run's **authoritative inputs** and are
read back as written. Everything derived from them -- the marked curve,
the per-trade dollars, every metric -- is **recomputed** on load and never
hydrated from a file. `metrics.parquet` is therefore an export for
cross-run SQL, not an input to an `ExperimentResult`: a loaded result is
truthful to the ledger and the market data it just read, and cannot report
a number its own inputs no longer produce.

**Loading degrades by piece.** The ledger and the always-on core metrics
are functions of the ledger, so they always come back. The marked curve is
not -- marking an open lot needs the run's market data -- so `load_run`
reopens `exp.data`. On a machine where that data is absent the curve is
`nothing` and the **whole** optional metric set is omitted, `:profit_factor`
included even though it reads trades and needs no curve: every metric takes
both inputs, so the dispatch table does not record which one each reads and
the omission cannot be selective. The path
metrics are absent from `metrics` (absent, not `NaN`: see
[metrics](metrics.md)), and the rest loads normally with a warning naming
the cause.

That boundary makes the backlog item **Dataset fingerprint in identity**
load-bearing. Previously stale or re-collected data could make a *rerun*
differ under one id; now a plain load can differ while appearing to read
recorded history. This round names that risk and does not solve it.

`save_run` validates that `config_toml` rebuilds the saved
`result.experiment`: same `full_hash` and same human `name` label. That
keeps the manifest, saved config, and `load_run(...).experiment`
coherent even though `name` is deliberately excluded from identity. It
also runs the whole-ledger `check_join` before creating the run folder
or writing any file.

`load_run` reads the ledger tables plus the saved `config.toml`, rebuilds
the live `Experiment` via `load_experiment_str`, rebuilds the ledger, and
then recomputes the marked curve and the metrics. The rebuilt
`Experiment.data` holds provider specs, pure values, so the ledger and the
trade metrics load on any machine; only the curve needs the data tree to be
present.

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
fields (cache sizes, readers, part order). The parquet specs' root sits in a
reserved `dataset` slot of that projection, the place a logical
dataset id and version would go; they also project the bar-stamp
convention as the constant `"bar_end"`, which is what moved every run id
when bar-end visibility landed.

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
`PnLSeries` type they exported, `n_marked` joined `n_unmarked`, and the
derived tables that replace the series belong to the second half of the
outputs round. Version 4 is the identity break --
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
`artifacts/` subdir, the DuckDB connection used for writing, and the
load path's rebuilding of the ledger through the ledger's own checks.

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
| `commit_sha` | VARCHAR | git commit of the code that produced the run |
| `dirty` | BOOLEAN | working tree had uncommitted changes |
| `written_at` | TIMESTAMP | UTC time of the save |
| `schema_version` | INTEGER | manifest schema version (`RUN_SCHEMA_VERSION`, currently 5); outside the hash |

`n_marked` / `n_unmarked` are NULL rather than 0 when the saved result had
no curve: "no curve at all" and "a curve that marked nothing" are different
facts and must not read the same in SQL.

### `metrics.parquet`

Long form so the schema is stable as metrics come and go.

| column | type |
|---|---|
| `run_id` | VARCHAR |
| `metric_name` | VARCHAR |
| `value` | DOUBLE |

`NaN` and `±Infinity` are stored verbatim (cast via `'NaN'::DOUBLE` on
write). An export only: `load_run` recomputes every metric and never reads
this file.

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
| **The ledger stored as it is: events with the kind's own columns, plus the order journal** | The events are the facts and the journal is what each decision saw; storing derived tables instead would let a stored run disagree with its own replay. Round trips, marks, equity and failures arrive as their own export tables in the second half of the outputs round. |
| **Derived results are recomputed on load, never read** | A stored derived table is a claim frozen at write time. Recomputing from the ledger and the market data means a loaded result cannot disagree with its own inputs -- and it turns "the data moved" into a visible, named degradation instead of a silently stale number. |
| **Write and load validate the join; load rebuilds through `commit!`** | `save_run` checks before creating a folder. A load must fail on a dangling fill, duplicate execution id, field mismatch or invalid simulated price, never drop the join. Reusing `commit!` means append invariants are not copied. |
| **`load_run` returns `ExperimentResult`, not a separate `StoredRun`** | Same type as `run_experiment` means same recipes / `show` / downstream consumers. Specs are pure values, so the rebuilt experiment only touches the data when run, while the ledger / pnl / metrics remain inspectable. |
| **Long-form `metrics.parquet`** | Optional metrics come and go per run; a wide schema would force columns to NULL across runs and break naive `UNION ALL` reads. Long form is stable and trivially pivotable. |
| **NaN / Inf preserved, not nulled** | A NaN sharpe (e.g. one trade, zero variance) is meaningful information about that run; collapsing it to NULL would lose the distinction from "metric not requested." |
| **Caller passes the TOML bytes** | The TOML is the source of truth for what was run; pushing the bytes through `save_run` keeps the persistence layer ignorant of how the `Experiment` was built and avoids stashing config strings on `Experiment` itself. |
| **DuckDB connection per store, exposed as `store.con`** | Same as the parquet readers. Lets viz / notebook code query without spawning a second DuckDB session. |
| **Best-effort atomicity** | A write-to-tmp-then-rename pass is the right fix, but it complicates the first slice. The recovery story is "re-run the same config," which works because identity is content-addressed. |

## Future work

- The second half of the outputs round: `round_trips`, `marks`, `equity`
  and `failures` export tables, a completeness flag in the manifest, and
  `compare_runs.jl` over them. Exports only -- the load path will keep
  recomputing.
- Write-to-temp-then-rename for atomic saves.
- Compute reuse: on a `core_hash` + `commit_sha` hit with a clean tree,
  load the cached ledger and recompute only the outputs instead of
  re-running the backtest.
- A curation gate (`status` draft / accepted / retracted with
  `accept_run` / `retract_run`), cross-run queries defaulting to accepted.
- Cross-run viz recipes built on the SQL queries demonstrated above.
- Delete / archive helpers (`drop_run(store, id)`). For now a manual
  `rm -rf <run_dir>` is the API.
