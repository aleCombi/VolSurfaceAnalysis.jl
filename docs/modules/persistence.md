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
    config.toml         # raw input TOML bytes, verbatim
    manifest.parquet    # 1 row of run-level metadata
    metrics.parquet     # long: (run_id, metric_name, value)
    positions.parquet   # 1 row per leg
    pnl_series.parquet  # 1 row per round trip (+ residuals)
  run_id=8a1c.../
    ...
```

The whole tree is just parquet. Any tool that reads parquet (DuckDB
CLI, pandas, polars, even a notebook in another language) can browse
it. There is no separate index file -- the manifest *is* the cross-run
index, queried via the Hive glob.

## `DataSource`-shaped, but for output

`RunStore` mirrors `ParquetDataSource`: owns its DuckDB connection,
exposes a `close` / `with_run_store` scoped form, and finalizer cleans
up. Same pattern, opposite direction (writes instead of reads).

## Public surface

```julia
RunStore(root) :: RunStore
with_run_store(f, root)
Base.close(store) :: RunStore
Base.isopen(store) :: Bool

run_id(config_toml) :: String                          # pure, no I/O
run_dir(store, run_id) :: String                       # path helper

save_run(store, result, config_toml) :: String         # returns run_id
load_run(store, run_id) :: ExperimentResult
```

`load_run` reads the four parquets plus the saved `config.toml`,
rebuilds the live `Experiment` via `load_experiment_str`, and
reconstructs `positions`, `pnl_series`, and the `metrics` `NamedTuple`
(integer types preserved for `n_round_trips` / `n_opens` / `n_closes`,
NaN / Inf preserved verbatim). The rebuilt `Experiment.source` does
**not** need its data on disk -- `ParquetDataSource` validates roots
lazily, so inspecting `positions` / `pnl_series` / `metrics` always
works; only an actual `get_chain` / `get_spot` against a missing root
throws.

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

`run_id = sha2_256(config_toml)[1:16]` (hex). Pure function of the
bytes the caller hands `save_run`. Re-saving the same TOML overwrites
the same folder in place.

**Known limitation, accepted on purpose:** whitespace, comments, and
key order all change the hash. Two semantically identical configs
spelled differently produce two folders. Canonicalization (a per-type
`to_dict` symmetric to the existing `build_*`, hashed as canonical
JSON) is deferred until spurious "new runs" become a real nuisance.
Documented here so the workaround is obvious: run the same file, or
run the same string.

## Responsibility boundaries

**Owns:** storage layout, parquet schemas for the four output tables,
content-hash identity, the DuckDB connection used for writing.

**Does NOT own:**

- Loading runs back into Julia values (`load_run` is the next slice).
- Querying. SQL on the parquet tree is the API.
- Canonical config serialization. Today the caller is the one
  responsible for handing in the TOML bytes they want hashed.
- Atomicity guarantees beyond best-effort. A crash mid-`save_run` can
  leave a half-written folder; re-running the same config recovers it.
  Write-to-temp-then-rename is queued.

## Schemas

### `manifest.parquet`

| column | type | notes |
|---|---|---|
| `run_id` | VARCHAR | content hash, also in the partition key |
| `name` | VARCHAR | from `Experiment.name` |
| `from_ts` | TIMESTAMP | evaluation window start |
| `to_ts` | TIMESTAMP | evaluation window end |
| `n_positions` | BIGINT | `length(result.positions)` |
| `n_opens` | BIGINT | from `PnLSeries` |
| `n_closes` | BIGINT | from `PnLSeries` |
| `settlement_spot` | DOUBLE | spot used to mark residuals |
| `written_at` | TIMESTAMP | UTC time of the save |

### `metrics.parquet`

Long form so the schema is stable as metrics come and go.

| column | type |
|---|---|
| `run_id` | VARCHAR |
| `metric_name` | VARCHAR |
| `value` | DOUBLE |

`NaN` and `±Infinity` are stored verbatim (cast via `'NaN'::DOUBLE` on
write).

### `positions.parquet`

`Trade` and `Position` flattened together.

| column | type |
|---|---|
| `run_id` | VARCHAR |
| `leg_idx` | BIGINT |
| `underlying` | VARCHAR |
| `strike` | DOUBLE |
| `expiry` | TIMESTAMP |
| `option_type` | VARCHAR (`'C'` / `'P'`) |
| `direction` | INTEGER (`+1` / `-1`) |
| `quantity` | DOUBLE |
| `entry_price` | DOUBLE |
| `entry_spot` | DOUBLE |
| `entry_bid` | DOUBLE (nullable) |
| `entry_ask` | DOUBLE (nullable) |
| `entry_timestamp` | TIMESTAMP |

### `pnl_series.parquet`

One row per round trip, post-sort by timestamp (matches `PnLSeries`).

| column | type |
|---|---|
| `run_id` | VARCHAR |
| `idx` | BIGINT |
| `timestamp` | TIMESTAMP |
| `pnl` | DOUBLE |

## Key decisions

| Decision | Why |
|---|---|
| **Parquet + DuckDB-as-engine, no single-file DB** | Matches the `data` module's existing pattern. Files are inspectable from any parquet-aware tool; a single corrupt run doesn't take down the whole store; `rm -rf <run_dir>` is a valid delete. |
| **Hive partition `run_id=<hash>`** | DuckDB and pandas / polars all understand the layout natively. Cross-run queries are one parquet glob, no separate manifest table to keep in sync. |
| **`load_run` returns `ExperimentResult`, not a separate `StoredRun`** | Same type as `run_experiment` means same recipes / `show` / downstream consumers. The "what about missing source data?" objection is resolved by `ParquetDataSource`'s lazy root validation -- the rebuilt source just throws on first read if its data is gone, while positions / pnl / metrics remain inspectable. No separate stored-vs-live type needed. |
| **Long-form `metrics.parquet`** | Optional metrics come and go per run; a wide schema would force columns to NULL across runs and break naive `UNION ALL` reads. Long form is stable and trivially pivotable. |
| **NaN / Inf preserved, not nulled** | A NaN sharpe (e.g. one trade, zero variance) is meaningful information about that run; collapsing it to NULL would lose the distinction from "metric not requested." |
| **Caller passes the TOML bytes** | The TOML is the source of truth for what was run; pushing the bytes through `save_run` keeps the persistence layer ignorant of how the `Experiment` was built and avoids stashing config strings on `Experiment` itself. |
| **DuckDB connection per store, exposed as `store.con`** | Same as `ParquetDataSource`. Lets viz / notebook code query without spawning a second DuckDB session. |
| **Best-effort atomicity** | A write-to-tmp-then-rename pass is the right fix, but it complicates the first slice. The recovery story is "re-run the same config," which works because identity is content-addressed. |

## Future work

- Write-to-temp-then-rename for atomic saves.
- Canonical `to_dict(::Experiment)` to make identity insensitive to
  whitespace / key order. Becomes worthwhile when spurious folder
  duplication actually starts hurting.
- Cross-run viz recipes built on the SQL queries demonstrated above.
- Delete / archive helpers (`drop_run(store, id)`). For now a manual
  `rm -rf <run_dir>` is the API.
