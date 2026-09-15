# `persistence` module: the knowledge base.
#
# Every run that lands here writes its inputs and its outputs to disk under
# a content-addressed folder, so prior experiments stay queryable and
# comparable rather than evaporating into ad-hoc notebooks (see vision.md).
#
# Storage shape: Hive-partitioned parquet under `<root>/runs/run_id=<hash>/`.
# Cross-run queries are just DuckDB SQL against the partitioned trees --
# this module does not invent a query API. Same DuckDB-as-engine /
# parquet-as-storage pattern the `data` module uses for input data.
#
# **A run folder has two jobs.** It keeps the *inputs* needed to run the
# experiment again -- `config.toml` and `Manifest.toml`, both verbatim,
# plus the code provenance in the manifest row -- and the *outputs* needed
# to verify that a rerun produced the same answer: the three ledger tables,
# the metrics, the marked curve and the failures the run retained. The
# manifest indexes them and records their provenance.
#
# **The reason to store outputs is evidence, and portability is explicitly
# not a concern.** Loading on a machine without the data tree is not the
# case this design serves. If loading recomputed the curve and the
# failures, a reproduction check would have lost the original witness: two
# fresh computations against today's code and tree can agree perfectly and
# both differ from the recorded run, and their agreement would then prove
# nothing about that run. So `load_run` reads the record and opens no
# market data, and `reproduce` is the separate operation that reruns the
# experiment and disagrees with the record by name.
#
# The ledger is written as it is: one row per event with the kind's own
# columns and NULLs elsewhere, one row per order and one per order leg
# with its observation. `load_run` rebuilds it through the ledger's own
# validated write path and the fill-to-order join, so a stored run that
# breaks an invariant fails to load by name rather than loading wrong. The
# manifest's counts are checked against what was rebuilt, and the curve's
# unmarked entries against the failures, so a truncated table or a mixed
# save is caught even when every individual constructor is satisfied.
#
# Run identity is `full_hash(result.experiment)` -- the canonical hash of
# the resolved experiment (see experiment/identity.jl), not the raw TOML
# bytes. The verbatim `config.toml` is still stored for human reading and
# to rebuild the experiment on load. The manifest also records `core_hash`
# (the backtest-only identity, shared by output variations of one backtest)
# and code provenance (`commit_sha` / `dirty`). Neither the code
# provenance nor the dependency manifest enters either hash.

using DuckDB
using DuckDB: DBInterface

# Manifest schema version, outside the run hash. `load_run` refuses a run
# written under another version rather than rebuilding a result its files
# cannot describe; the message says to rerun the config, because no
# migration can recover a curve or a failure table that was never written.
const RUN_SCHEMA_VERSION = 6

"""
    RunStore

A directory-backed knowledge base of past runs. Owns its DuckDB
connection for writing parquet; downstream readers (notebooks, viz, ad
hoc analysis) can `DBInterface.execute(store.con, "SELECT ...")`
directly against the run files for cross-run queries.

# Fields
- `root::String` -- absolute path to the store root. The `runs/`
  subdirectory below it is the partitioned tree.
- `con::DuckDB.DB`
- `closed::Bool`
"""
mutable struct RunStore
    root::String
    con::DuckDB.DB
    closed::Bool
end

"""
    RunStore(root::AbstractString)

Open (or create) a store rooted at `root`. The directory is created if
absent; the `runs/` subdirectory is created lazily on first save.
"""
function RunStore(root::AbstractString)
    mkpath(String(root))
    store = RunStore(abspath(String(root)), DuckDB.DB(":memory:"), false)
    finalizer(_close_store_con, store)
    store
end

function _close_store_con(s::RunStore)
    if !s.closed
        try
            DBInterface.close!(s.con)
        catch
        end
        s.closed = true
    end
end

Base.isopen(s::RunStore) = !s.closed

function Base.close(s::RunStore)
    _close_store_con(s)
    return s
end

"""
    with_run_store(f, root)

Open a `RunStore`, call `f(store)`, then close the store in a `finally`
block. Mirrors `with_data` for resource-scoped use.
"""
function with_run_store(f::Function, root::AbstractString)
    s = RunStore(root)
    try
        return f(s)
    finally
        close(s)
    end
end

# --- run paths -----------------------------------------------------------
# A run's id is its `full_hash` (computed in save_run); `run_dir` just maps
# an id string to its Hive-partitioned folder.

run_dir(store::RunStore, run_id::AbstractString)::String =
    joinpath(store.root, "runs", "run_id=" * String(run_id))

# --- write helpers -------------------------------------------------------

_sql_pq_path(p::AbstractString) = replace(String(p), "\\" => "/")

_assert_open(s::RunStore) = isopen(s) || throw(ArgumentError("RunStore is closed"))

# Write a Tables.jl-compatible row iterator to a parquet file via DuckDB.
# Registers the rows as a temp view, COPYs to parquet, unregisters the
# view. `schema_sql` declares the parquet column types so DuckDB doesn't
# have to guess (and so empty inputs still get a typed file).
function _write_parquet(store::RunStore, path::AbstractString,
                        schema_sql::AbstractString,
                        insert_sqls::AbstractVector{<:AbstractString})
    mkpath(dirname(path))
    DBInterface.execute(store.con, "CREATE OR REPLACE TEMP TABLE _writebuf $schema_sql")
    try
        for sql in insert_sqls
            DBInterface.execute(store.con, sql)
        end
        DBInterface.execute(store.con,
            "COPY _writebuf TO '$(_sql_pq_path(path))' (FORMAT PARQUET)")
    finally
        DBInterface.execute(store.con, "DROP TABLE IF EXISTS _writebuf")
    end
end

# Milliseconds, so a loaded ledger equals the saved one exactly.
_dt_sql(d::DateTime) = "TIMESTAMP '$(Dates.format(d, "yyyy-mm-dd HH:MM:SS.sss"))'"
_str_sql(s::AbstractString) = "'" * replace(String(s), "'" => "''") * "'"
_otype_sql(t::OptionType) = t == Call ? "'C'" : "'P'"
_side_sql(s::Side) = s == Long ? "'long'" : "'short'"
_intent_sql(i::Intent) = i == Open ? "'open'" : "'close'"
_outcome_sql(o::ExpiryOutcome) = o == Worthless ? "'worthless'" : "'cash_settled'"
_opt_sql(x, f) = x === nothing || x === missing ? "NULL" : f(x)

_otype_from(s) = String(s) == "C" ? Call : Put
_side_from(s) = String(s) == "long" ? Long : Short
_intent_from(s) = String(s) == "open" ? Open : Close
_outcome_from(s) = String(s) == "worthless" ? Worthless : CashSettled
_contract_from(r) = ContractKey(Underlying(String(r.underlying)), Float64(r.strike),
                                DateTime(r.expiry), _otype_from(r.option_type))
_opt_from(x, f) = x === missing ? missing : f(x)

# DuckDB SQL has no bare NaN / Infinity literals -- they parse as
# identifiers. Round-trip non-finite floats via a quoted cast.
function _f_sql(x::Real)::String
    f = Float64(x)
    if isnan(f)
        return "'NaN'::DOUBLE"
    elseif isinf(f)
        return f > 0 ? "'Infinity'::DOUBLE" : "'-Infinity'::DOUBLE"
    else
        return string(f)
    end
end

# --- named-column inserts ------------------------------------------------
# Every insert names its columns. A positional `VALUES (...)` list makes a
# miscount write a value into the neighbouring column of the right type,
# which nothing in the type system catches; naming them also removes the
# per-kind NULL padding in the events table, since DuckDB nulls what an
# insert does not name. The file written is byte-identical either way, so
# this is not a schema change.

_insert_sql(cols::AbstractVector{<:Pair}) =
    "INSERT INTO _writebuf (" * join((first(c) for c in cols), ", ") *
    ") VALUES (" * join((last(c) for c in cols), ", ") * ")"

# --- save_run ------------------------------------------------------------

"""
    save_run(store::RunStore, result::ExperimentResult,
             config_toml::AbstractString;
             commit_sha::AbstractString="", dirty::Bool=true,
             manifest_toml::AbstractString=dependency_manifest()) -> String

Persist `result` plus the inputs that would let it be run again, under
`<store.root>/runs/run_id=<full_hash>/`. Returns the 16-hex run id, which
is `full_hash(result.experiment)` -- the identity of the resolved
experiment, independent of how its config was spelled. `config_toml`
must rebuild `result.experiment` (same `full_hash` and same human
`name`) or this throws `ArgumentError`, keeping the persisted config
faithful to the saved result. The ledger's event/order join is validated
before the run directory is created or any file is written.

`commit_sha` / `dirty` are the code provenance of the run (see
[`code_provenance`](@ref)); they default to `("", true)` so a caller that
does not supply provenance records an unknown, uncacheable run.
`manifest_toml` is the resolved dependency record, copied verbatim from
the environment that produced the run rather than re-resolved here; its
absence is [`MissingManifest`](@ref), raised before the run folder exists.

Writes, as two inputs and five outputs:

- `config.toml`, `Manifest.toml` -- the bytes passed in, verbatim.
- `manifest.parquet` -- one row indexing the run: both hashes, the name,
  the window, six counts, the code provenance, the write time and the
  schema version.
- `metrics.parquet`, `events.parquet`, `orders.parquet`,
  `order_legs.parquet`, `failures.parquet` -- the run's outputs.
- `curve.parquet` -- the marked curve, **only when the result carries
  one**. Its absence and a present-but-empty curve are different facts and
  stay different on disk; the manifest's NULL-versus-zero curve counts say
  which.

If a folder for this id already exists, its contents are overwritten:
same resolved experiment means same id, so re-saving is idempotent in
intent (the latest run of that exact experiment wins). A stale
`curve.parquet` from a previous save that had a curve is removed when this
one does not, so the folder never describes two different runs at once.

Atomicity is best-effort today: a crash mid-write can leave a
half-written folder, and the manifest's counts do not make a multi-file
save atomic. Re-running the same experiment recovers it. A
write-to-temp-then-rename pass is queued for the next iteration.
"""
function save_run(store::RunStore, result::ExperimentResult,
                  config_toml::AbstractString;
                  commit_sha::AbstractString="", dirty::Bool=true,
                  manifest_toml::AbstractString=dependency_manifest())::String
    _assert_open(store)
    id = full_hash(result.experiment)
    # Integrity: the config we persist must rebuild the experiment being
    # saved, so load_run reproduces it faithfully. `name` is not part of
    # full_hash (label only), so compare it explicitly.
    config_exp = load_experiment_str(config_toml)           # specs only, nothing to close
    config_id = full_hash(config_exp)
    config_id == id || throw(ArgumentError(
        "save_run: config_toml does not describe result.experiment " *
        "(config full_hash=$config_id, result full_hash=$id)"))
    config_exp.name == result.experiment.name || throw(ArgumentError(
        "save_run: config_toml name \"$(config_exp.name)\" does not match " *
        "result.experiment name \"$(result.experiment.name)\""))
    check_join(result.ledger)
    dir = run_dir(store, id)
    mkpath(dir)

    open(joinpath(dir, "config.toml"), "w") do io
        write(io, String(config_toml))
    end
    open(joinpath(dir, "Manifest.toml"), "w") do io
        write(io, String(manifest_toml))
    end

    _write_manifest(store, dir, id, result; commit_sha=commit_sha, dirty=dirty)
    _write_metrics(store, dir, id, result)
    _write_events(store, dir, id, result.ledger)
    _write_orders(store, dir, id, result.ledger)
    _write_order_legs(store, dir, id, result.ledger)
    _write_curve(store, dir, id, result.curve)
    _write_failures(store, dir, id, result.failures)

    return id
end

function _write_manifest(store::RunStore, dir::AbstractString, id::AbstractString,
                         result::ExperimentResult;
                         commit_sha::AbstractString, dirty::Bool)
    exp = result.experiment
    c   = result.curve
    schema = """(
        run_id VARCHAR,
        core_hash VARCHAR,
        name VARCHAR,
        from_ts TIMESTAMP,
        to_ts TIMESTAMP,
        n_events BIGINT,
        n_orders BIGINT,
        n_opens BIGINT,
        n_closes BIGINT,
        n_marked BIGINT,
        n_unmarked BIGINT,
        commit_sha VARCHAR,
        dirty BOOLEAN,
        written_at TIMESTAMP,
        schema_version INTEGER
    )"""
    # `n_marked` / `n_unmarked` are NULL when the run carries no curve at
    # all, which is a different fact from a curve that marked nothing.
    cols = [
        "run_id"         => _str_sql(id),
        "core_hash"      => _str_sql(core_hash(exp)),
        "name"           => _str_sql(exp.name),
        "from_ts"        => _dt_sql(exp.from),
        "to_ts"          => _dt_sql(exp.to),
        "n_events"       => string(length(result.ledger)),
        "n_orders"       => string(length(result.ledger.orders)),
        "n_opens"        => string(n_opens(result.ledger)),
        "n_closes"       => string(n_closes(result.ledger)),
        "n_marked"       => c === nothing ? "NULL" : string(n_marked(c)),
        "n_unmarked"     => c === nothing ? "NULL" : string(n_unmarked(c)),
        "commit_sha"     => _str_sql(commit_sha),
        "dirty"          => dirty ? "TRUE" : "FALSE",
        "written_at"     => _dt_sql(Dates.now(UTC)),
        "schema_version" => string(RUN_SCHEMA_VERSION),
    ]
    _write_parquet(store, joinpath(dir, "manifest.parquet"), schema, [_insert_sql(cols)])
end

function _write_metrics(store::RunStore, dir::AbstractString, id::AbstractString,
                        result::ExperimentResult)
    schema = "(run_id VARCHAR, metric_name VARCHAR, value DOUBLE)"
    inserts = String[]
    for (k, v) in pairs(result.metrics)
        # A metric that is not a number has no honest cell in a DOUBLE
        # column, and `NaN` would claim it was computed and undefined.
        v isa Real || throw(ArgumentError(
            "save_run: metric :$k is a $(typeof(v)), which metrics.parquet " *
            "cannot record; every metric must reduce to a number"))
        push!(inserts, _insert_sql([
            "run_id"      => _str_sql(id),
            "metric_name" => _str_sql(String(k)),
            "value"       => _f_sql(Float64(v)),
        ]))
    end
    _write_parquet(store, joinpath(dir, "metrics.parquet"), schema, inserts)
end

# One row per event in sequence order: the header, the kind, and the
# kind's own columns; a column the kind lacks is simply not named, and
# DuckDB nulls it. `group` is a SQL keyword, so the column is `group_id`.
const _EVENTS_SCHEMA = """(
    run_id VARCHAR,
    sequence BIGINT,
    id BIGINT,
    kind VARCHAR,
    effective_at TIMESTAMP,
    recorded_at TIMESTAMP,
    group_id BIGINT,
    order_leg_id BIGINT,
    execution_id BIGINT,
    underlying VARCHAR,
    strike DOUBLE,
    expiry TIMESTAMP,
    option_type VARCHAR,
    side VARCHAR,
    intent VARCHAR,
    quantity BIGINT,
    price DOUBLE,
    fill_rule VARCHAR,
    open_fill_id BIGINT,
    close_fill_id BIGINT,
    settlement_price DOUBLE,
    outcome VARCHAR,
    source_id BIGINT,
    amount BIGINT
)"""

# The contract columns, shared by the two kinds that carry one.
_contract_columns(c::ContractKey) = [
    "underlying"  => _str_sql(ticker(c.underlying)),
    "strike"      => _f_sql(c.strike),
    "expiry"      => _dt_sql(c.expiry),
    "option_type" => _otype_sql(c.option_type),
]

# The kind-specific columns of one event: exactly the columns that kind
# has, named.
_event_columns(e::Fill) = vcat([
    "group_id"     => string(e.group),
    "order_leg_id" => string(e.order_leg_id),
    "execution_id" => string(e.execution_id),
], _contract_columns(e.contract), [
    "side"      => _side_sql(e.side),
    "intent"    => _intent_sql(e.intent),
    "quantity"  => string(e.quantity),
    "price"     => _f_sql(e.price),
    "fill_rule" => _str_sql(String(e.fill_rule)),
])
_event_columns(e::Match) = [
    "group_id"      => string(e.group),
    "quantity"      => string(e.quantity),
    "open_fill_id"  => string(e.open_fill_id),
    "close_fill_id" => string(e.close_fill_id),
]
_event_columns(e::Expiry) = vcat([
    "group_id" => string(e.group),
], _contract_columns(e.contract), [
    "side"             => _side_sql(e.side),
    "quantity"         => string(e.quantity),
    "open_fill_id"     => string(e.open_fill_id),
    "settlement_price" => _f_sql(e.settlement_price),
    "outcome"          => _outcome_sql(e.outcome),
])
_event_columns(e::Fee) = [
    "source_id" => string(e.source_id),
    "amount"    => string(e.amount),
]

_kind_sql(::Fill) = "'fill'"
_kind_sql(::Match) = "'match'"
_kind_sql(::Expiry) = "'expiry'"
_kind_sql(::Fee) = "'fee'"

function _write_events(store::RunStore, dir::AbstractString, id::AbstractString, L::Ledger)
    inserts = String[]
    for e in L.events
        cols = vcat([
            "run_id"       => _str_sql(id),
            "sequence"     => string(sequence(e)),
            "id"           => string(event_id(e)),
            "kind"         => _kind_sql(e),
            "effective_at" => _dt_sql(effective_at(e)),
            "recorded_at"  => _dt_sql(recorded_at(e)),
        ], _event_columns(e))
        push!(inserts, _insert_sql(cols))
    end
    _write_parquet(store, joinpath(dir, "events.parquet"), _EVENTS_SCHEMA, inserts)
end

function _write_orders(store::RunStore, dir::AbstractString, id::AbstractString, L::Ledger)
    schema = """(
        run_id VARCHAR,
        order_id BIGINT,
        first_leg_id BIGINT,
        label VARCHAR,
        group_id BIGINT,
        operation BIGINT,
        decided_at TIMESTAMP,
        known_to BIGINT
    )"""
    inserts = String[]
    for r in L.orders
        push!(inserts, _insert_sql([
            "run_id"       => _str_sql(id),
            "order_id"     => string(r.order_id),
            "first_leg_id" => string(r.first_leg_id),
            "label"        => _str_sql(String(r.order.label)),
            "group_id"     => string(r.group),
            "operation"    => _opt_sql(r.order.operation, string),
            "decided_at"   => _dt_sql(r.decided_at),
            "known_to"     => string(r.known_to),
        ]))
    end
    _write_parquet(store, joinpath(dir, "orders.parquet"), schema, inserts)
end

function _write_order_legs(store::RunStore, dir::AbstractString, id::AbstractString, L::Ledger)
    schema = """(
        run_id VARCHAR,
        order_id BIGINT,
        order_leg_id BIGINT,
        leg_idx BIGINT,
        underlying VARCHAR,
        strike DOUBLE,
        expiry TIMESTAMP,
        option_type VARCHAR,
        side VARCHAR,
        intent VARCHAR,
        quantity BIGINT,
        quote_at TIMESTAMP,
        bid DOUBLE,
        ask DOUBLE,
        spot DOUBLE,
        spot_at TIMESTAMP
    )"""
    inserts = String[]
    for r in L.orders, (k, leg) in enumerate(r.order.legs)
        o = r.observations[k]
        cols = vcat([
            "run_id"       => _str_sql(id),
            "order_id"     => string(r.order_id),
            "order_leg_id" => string(r.first_leg_id + k - 1),
            "leg_idx"      => string(k),
        ], _contract_columns(leg.contract), [
            "side"     => _side_sql(leg.side),
            "intent"   => _intent_sql(leg.intent),
            "quantity" => string(leg.quantity),
            "quote_at" => _dt_sql(o.quote_at),
            "bid"      => _opt_sql(o.bid, _f_sql),
            "ask"      => _opt_sql(o.ask, _f_sql),
            "spot"     => _f_sql(o.spot),
            "spot_at"  => _dt_sql(o.spot_at),
        ])
        push!(inserts, _insert_sql(cols))
    end
    _write_parquet(store, joinpath(dir, "order_legs.parquet"), schema, inserts)
end

# The `MarkedCurve` in stored form: both pairs of vectors, one row each,
# marked rows first. An unmarked row has a `reason` and **no profit** --
# NULL, not zero, not NaN and not the previous session's value. The column
# is `instant` rather than `at`, which DuckDB spells as a keyword.
const _CURVE_SCHEMA = """(
    run_id VARCHAR,
    kind VARCHAR,
    instant TIMESTAMP,
    profit DOUBLE,
    reason VARCHAR
)"""

function _write_curve(store::RunStore, dir::AbstractString, id::AbstractString,
                      c::Union{MarkedCurve,Nothing})
    path = joinpath(dir, "curve.parquet")
    if c === nothing
        # No curve at all is the absence of the file, matched by the
        # manifest's NULL counts. A previous save of this id may have
        # written one; leaving it would make the folder describe two runs.
        rm(path; force = true)
        return nothing
    end
    inserts = String[]
    for (t, p) in zip(c.timestamps, c.profit)
        push!(inserts, _insert_sql([
            "run_id"  => _str_sql(id),
            "kind"    => "'marked'",
            "instant" => _dt_sql(t),
            "profit"  => _f_sql(p),
        ]))
    end
    for (t, r) in zip(c.unmarked_at, c.unmarked_reason)
        push!(inserts, _insert_sql([
            "run_id"  => _str_sql(id),
            "kind"    => "'unmarked'",
            "instant" => _dt_sql(t),
            "reason"  => _str_sql(String(r)),
        ]))
    end
    _write_parquet(store, path, _CURVE_SCHEMA, inserts)
end

# Every unanswered question the run retained, in the run's own canonical
# order. A runtime observation: nothing happened, so no ledger replay can
# recover one, and this table is the only place it survives.
const _FAILURES_SCHEMA = """(
    run_id VARCHAR,
    instant TIMESTAMP,
    stage VARCHAR,
    subject VARCHAR,
    reason VARCHAR
)"""

function _write_failures(store::RunStore, dir::AbstractString, id::AbstractString,
                         failures::AbstractVector{RunFailure})
    inserts = String[_insert_sql([
        "run_id"  => _str_sql(id),
        "instant" => _dt_sql(f.at),
        "stage"   => _str_sql(String(f.stage)),
        "subject" => _str_sql(f.subject),
        "reason"  => _str_sql(String(f.reason)),
    ]) for f in failures]
    _write_parquet(store, joinpath(dir, "failures.parquet"), _FAILURES_SCHEMA, inserts)
end

# --- load_run ------------------------------------------------------------

"""
    load_run(store::RunStore, run_id::AbstractString) -> ExperimentResult

Read a previously [`save_run`](@ref)-saved run back as the
`ExperimentResult` it was: the ledger, the marked curve, the failures it
retained and the metrics it reported. **It recomputes nothing and opens no
market data.** A loaded run is the recorded witness -- what that run
observed then -- not a fresh answer that happens to be produced from the
same config now. Checking whether today's code and data still produce it
is [`reproduce`](@ref), which needs this record to have something to
disagree with.

The ledger is still rebuilt through its own write path: every event is
built through its constructor in sequence order and committed to a fresh
`Ledger` as one batch through `commit!` (the book is empty, so FIFO
among batch-opened lots is sequence order), so a loaded ledger has passed
every append-time check; the order records are rebuilt from the two order
tables, every counter is set one past the largest id seen (groups
included), and `check_join` runs last. A load that fails a check throws
that check's named failure; it never drops the join.

Beyond the per-record checks it checks the record against itself: each of
the manifest's six counts against what was rebuilt, naming the column and
both values on a mismatch; and the curve's unmarked entries against the
failures, which must agree instant for instant. NULL curve counts mean
the run carried no curve and `curve.parquet` must be absent; zero counts
mean a recorded curve with no entries of that kind, and the file must be
there. Counts are consistency checks -- they expose a truncated table or
a mixed save that every individual constructor would accept -- not proof
that any price or metric is right; a plausible but stale price loads.

Throws `ArgumentError` if the run folder or any required file is missing,
or if the manifest's `schema_version` is absent or differs from
`RUN_SCHEMA_VERSION`: rerun its config to regenerate it. There is no
migration -- no earlier schema holds a curve or a failure table to
migrate from.
"""
function load_run(store::RunStore, run_id::AbstractString)::ExperimentResult
    _assert_open(store)
    dir = run_dir(store, run_id)
    isdir(dir) || throw(ArgumentError("load_run: no run folder for id $run_id at $dir"))

    manifest = _load_manifest(store, dir)
    manifest.schema_version == RUN_SCHEMA_VERSION || throw(ArgumentError(
        "load_run: run $run_id was written with manifest schema_version " *
        "$(manifest.schema_version); this store reads version " *
        "$RUN_SCHEMA_VERSION only -- rerun the config to regenerate it"))

    cfg_path = joinpath(dir, "config.toml")
    isfile(cfg_path) || throw(ArgumentError("load_run: missing config.toml in $dir"))
    # Read, not parsed: the dependency record is an input document for a
    # controlled rerun, not something this module interprets. Its absence
    # is still a defect in the record and is named as one.
    dep_path = joinpath(dir, "Manifest.toml")
    isfile(dep_path) || throw(ArgumentError(
        "load_run: missing Manifest.toml in $dir; the run's dependency " *
        "record is part of the record -- rerun the config to regenerate it"))
    exp = load_experiment_str(read(cfg_path, String))

    ledger   = _load_ledger(store, dir)
    curve    = _load_curve(store, dir, manifest)
    failures = _load_failures(store, dir)
    metrics  = _load_metrics(store, dir)

    _check_counts(run_id, manifest, ledger, curve)
    _check_curve_failure_agreement(run_id, curve, failures)

    return ExperimentResult(exp, ledger, curve, metrics, failures)
end

# A count that disagrees names the column and both values: the manifest is
# the index, and an index that disagrees with the tables it indexes is the
# one defect the per-record constructors cannot see.
function _count_mismatch(run_id, column, stored, actual)
    stored == actual || throw(ArgumentError(
        "load_run: run $run_id manifest column $column says $stored, " *
        "but the stored tables hold $actual"))
    return nothing
end

function _check_counts(run_id, manifest, L::Ledger, curve::Union{MarkedCurve,Nothing})
    _count_mismatch(run_id, "n_events", manifest.n_events, length(L))
    _count_mismatch(run_id, "n_orders", manifest.n_orders, length(L.orders))
    _count_mismatch(run_id, "n_opens", manifest.n_opens, n_opens(L))
    _count_mismatch(run_id, "n_closes", manifest.n_closes, n_closes(L))
    if curve === nothing
        manifest.n_marked === nothing && manifest.n_unmarked === nothing || throw(ArgumentError(
            "load_run: run $run_id manifest records a curve " *
            "(n_marked=$(manifest.n_marked), n_unmarked=$(manifest.n_unmarked)) " *
            "but no curve.parquet is stored"))
    else
        manifest.n_marked === nothing && throw(ArgumentError(
            "load_run: run $run_id stores a curve.parquet but its manifest " *
            "records no curve (n_marked is NULL)"))
        _count_mismatch(run_id, "n_marked", manifest.n_marked, n_marked(curve))
        _count_mismatch(run_id, "n_unmarked", manifest.n_unmarked, n_unmarked(curve))
    end
    return nothing
end

# The curve says a session could not be marked; the failures say which
# questions that session left unanswered. One implies the other: a session
# is unmarked exactly when at least one of its lots could not be priced,
# and a mark failure is recorded exactly at a session the curve left
# unmarked. A truncation of either table breaks the agreement.
function _check_curve_failure_agreement(run_id, curve::Union{MarkedCurve,Nothing},
                                        failures::AbstractVector{RunFailure})
    marked_failure_instants = Set(f.at for f in failures if f.stage === :mark)
    unmarked = curve === nothing ? Set{DateTime}() : Set(curve.unmarked_at)
    missing_rows = sort!(collect(setdiff(unmarked, marked_failure_instants)))
    isempty(missing_rows) || throw(ArgumentError(
        "load_run: run $run_id has unmarked curve instants with no failure " *
        "recorded against them: $(missing_rows)"))
    extra_rows = sort!(collect(setdiff(marked_failure_instants, unmarked)))
    isempty(extra_rows) || throw(ArgumentError(
        "load_run: run $run_id records mark failures at instants the curve " *
        "does not report unmarked: $(extra_rows)"))
    return nothing
end

function _select_rows(store::RunStore, path::AbstractString, sql::AbstractString)
    isfile(path) || throw(ArgumentError("load_run: missing $(basename(path)) at $path"))
    return collect(DBInterface.execute(store.con, sql))
end

# `SELECT *` so a manifest written without `schema_version` (an old run)
# still reads; the missing column reports as version 0, which no store
# accepts. Everything else is read as written; the curve counts keep their
# NULL, which is the only way "no curve" survives a round trip.
function _load_manifest(store::RunStore, dir::AbstractString)
    path = joinpath(dir, "manifest.parquet")
    rows = _select_rows(store, path, "SELECT * FROM '$(_sql_pq_path(path))'")
    length(rows) == 1 ||
        throw(ArgumentError("load_run: manifest.parquet must have exactly 1 row, got $(length(rows))"))
    r = first(rows)
    has(col) = col in propertynames(r) && getproperty(r, col) !== missing
    version = has(:schema_version) ? Int(r.schema_version) : 0
    version == RUN_SCHEMA_VERSION || return (schema_version = version,)
    num(col) = has(col) ? Int(getproperty(r, col)) : nothing
    return (schema_version = version,
            core_hash  = String(r.core_hash),
            n_events   = num(:n_events),
            n_orders   = num(:n_orders),
            n_opens    = num(:n_opens),
            n_closes   = num(:n_closes),
            n_marked   = num(:n_marked),
            n_unmarked = num(:n_unmarked),
            commit_sha = has(:commit_sha) ? String(r.commit_sha) : "",
            dirty      = has(:dirty) ? Bool(r.dirty) : true)
end

# One event from its row, through the kind's constructor so every
# construction-time check runs on a stored event too.
function _event_from(r)::LedgerEvent
    h = EventHeader(Int(r.id), DateTime(r.effective_at), DateTime(r.recorded_at), Int(r.sequence))
    kind = String(r.kind)
    if kind == "fill"
        return Fill(h, Int(r.group_id), Int(r.order_leg_id), Int(r.execution_id), _contract_from(r),
                    _side_from(r.side), _intent_from(r.intent), Int(r.quantity), Float64(r.price),
                    Symbol(String(r.fill_rule)))
    elseif kind == "match"
        return Match(h, Int(r.group_id), Int(r.open_fill_id), Int(r.close_fill_id), Int(r.quantity))
    elseif kind == "expiry"
        return Expiry(h, Int(r.group_id), Int(r.open_fill_id), _contract_from(r), _side_from(r.side),
                      Int(r.quantity), Float64(r.settlement_price), _outcome_from(r.outcome))
    elseif kind == "fee"
        return Fee(h, Int(r.source_id), Int(r.amount))
    end
    throw(ArgumentError("load_run: unknown event kind \"$kind\" at sequence $(r.sequence)"))
end

function _load_ledger(store::RunStore, dir::AbstractString)::Ledger
    events_path = joinpath(dir, "events.parquet")
    rows = _select_rows(store, events_path,
        "SELECT * FROM '$(_sql_pq_path(events_path))' ORDER BY sequence")
    events = LedgerEvent[_event_from(r) for r in rows]
    # One batch through the validated write path: ids and sequences must
    # continue a fresh ledger's counters, references must point backward,
    # matches must be FIFO, cash must be whole cents -- as when written.
    L = Ledger(events)

    orders_path = joinpath(dir, "orders.parquet")
    order_rows = _select_rows(store, orders_path,
        "SELECT * FROM '$(_sql_pq_path(orders_path))' ORDER BY order_id")
    legs_path = joinpath(dir, "order_legs.parquet")
    leg_rows = _select_rows(store, legs_path,
        "SELECT * FROM '$(_sql_pq_path(legs_path))' ORDER BY order_id, leg_idx")
    legs_by_order = Dict{Int,Vector{Any}}()
    for r in leg_rows
        push!(get!(() -> Any[], legs_by_order, Int(r.order_id)), r)
    end
    # Whether the policy named the group or the ledger minted it is not a
    # column: the first record of a group (by order id) minted it, since
    # a group can only be named once minted, and only by a later order.
    minted = Set{Int}()
    for r in order_rows
        oid = Int(r.order_id)
        g = Int(r.group_id)
        own_group = g in minted ? g : nothing
        push!(minted, g)
        lrows = get(legs_by_order, oid, Any[])
        legs = Leg[Leg(_contract_from(l), _side_from(l.side), Int(l.quantity), _intent_from(l.intent))
                   for l in lrows]
        obs = LegObservation[LegObservation(DateTime(l.quote_at), _opt_from(l.bid, Float64),
                                            _opt_from(l.ask, Float64), Float64(l.spot),
                                            DateTime(l.spot_at)) for l in lrows]
        order = Order(Symbol(String(r.label)), legs; group = own_group,
                      operation = r.operation === missing ? nothing : Int(r.operation))
        push!(L.orders, OrderRecord(oid, Int(r.first_leg_id), g, DateTime(r.decided_at),
                                    Int(r.known_to), order, obs))
    end
    # Every counter one past the largest id seen (`commit!` moved the
    # event, sequence and execution counters already).
    groups = Int[group(e) for e in L.events if !(e isa Fee)]
    append!(groups, Int[r.group for r in L.orders])
    L.next_group    = isempty(groups) ? 1 : maximum(groups) + 1
    L.next_order_id = isempty(L.orders) ? 1 : maximum(r.order_id for r in L.orders) + 1
    L.next_leg_id   = isempty(L.orders) ? 1 :
                      maximum(r.first_leg_id + length(r.order.legs) - 1 for r in L.orders) + 1
    check_join(L)
    return L
end

# The curve as stored, or `nothing` when the run carried none. Absence of
# the file is the absence of a curve, and the manifest's NULL counts are
# checked against it; a file that is there while the manifest says NULL,
# or missing while it says a number, is a mixed save and is named.
function _load_curve(store::RunStore, dir::AbstractString, manifest)::Union{MarkedCurve,Nothing}
    path = joinpath(dir, "curve.parquet")
    isfile(path) || return nothing
    rows = _select_rows(store, path,
        "SELECT * FROM '$(_sql_pq_path(path))' ORDER BY kind, instant")
    ts, profit = DateTime[], Float64[]
    at_, reason = DateTime[], Symbol[]
    for r in rows
        k = String(r.kind)
        if k == "marked"
            r.profit === missing && throw(ArgumentError(
                "load_run: curve.parquet marks $(r.instant) with no profit"))
            push!(ts, DateTime(r.instant)); push!(profit, Float64(r.profit))
        elseif k == "unmarked"
            # An unmarked point carries a reason and no value. A profit
            # here would be the carried-forward number the curve exists to
            # refuse.
            r.reason === missing && throw(ArgumentError(
                "load_run: curve.parquet leaves $(r.instant) unmarked with no reason"))
            r.profit === missing || throw(ArgumentError(
                "load_run: curve.parquet gives the unmarked instant $(r.instant) " *
                "a profit of $(r.profit); an unmarked session has no value"))
            push!(at_, DateTime(r.instant)); push!(reason, Symbol(String(r.reason)))
        else
            throw(ArgumentError("load_run: unknown curve row kind \"$k\" at $(r.instant)"))
        end
    end
    return MarkedCurve(ts, profit, at_, reason)
end

# In the run's own canonical order, which is what makes a stored failure
# table and a freshly produced one comparable row by row.
function _load_failures(store::RunStore, dir::AbstractString)::Vector{RunFailure}
    path = joinpath(dir, "failures.parquet")
    rows = _select_rows(store, path,
        "SELECT * FROM '$(_sql_pq_path(path))' ORDER BY instant, stage, subject, reason")
    return RunFailure[RunFailure(DateTime(r.instant), Symbol(String(r.stage)),
                                 String(r.subject), Symbol(String(r.reason))) for r in rows]
end

# The metrics the run reported, read back in the order it wrote them. The
# column is DOUBLE, so an integer metric comes back as a Float64 of the
# same value; absence of a key still means "not computed", which is what
# keeps it different from any number including NaN.
function _load_metrics(store::RunStore, dir::AbstractString)::NamedTuple
    path = joinpath(dir, "metrics.parquet")
    rows = _select_rows(store, path, "SELECT * FROM '$(_sql_pq_path(path))'")
    names, values = Symbol[], Float64[]
    for r in rows
        name = Symbol(String(r.metric_name))
        name in names && throw(ArgumentError(
            "load_run: metrics.parquet records :$name twice; a metric has one value"))
        push!(names, name); push!(values, Float64(r.value))
    end
    return NamedTuple{Tuple(names)}(Tuple(values))
end

# --- reproduce -----------------------------------------------------------

# The tolerance for finite floating comparison, the same absolute bound
# `scripts/compare_runs.jl` uses. Non-finite values never reach it: NaN
# matches NaN and an infinity matches the same-signed infinity, checked
# before any subtraction, because `NaN - NaN` and `Inf - Inf` answer
# nothing about whether two runs agreed.
const REPRODUCE_TOL = 1e-9

"""
    Divergence

One field on which a rerun disagreed with the record: which `output`
table, which `row` inside it, which `field`, and both values as they
print. `field` is `:presence` when the row exists on one side only.
A difference is never a bare boolean -- naming what moved is the whole
point of keeping the witness.
"""
struct Divergence
    output :: Symbol
    row    :: String
    field  :: Symbol
    stored :: String
    fresh  :: String
end

Base.show(io::IO, d::Divergence) = print(io,
    d.output, " ", d.row, ": ", d.field, " stored ", d.stored, ", fresh ", d.fresh)

"""
    ReproductionReport

What [`reproduce`](@ref) found. `status` is one of:

- `:reproduced` -- the rerun agreed with the record on every compared field.
- `:diverged` -- it disagreed; `divergences` names where.
- `:unreproducible` -- the question could not be asked at all, and
  `detail` says why. Missing data is this, never a successful comparison
  of empty outputs.

`stored_code` and `fresh_code` are the recorded and the running
`(commit_sha, dirty)`. Dataset versioning is deliberately not in identity
(see the module doc), so a divergence attributes to code or dependencies
by elimination; these two, with the run's stored `Manifest.toml`, are what
a controlled rerun separates.
"""
struct ReproductionReport
    run_id      :: String
    status      :: Symbol
    detail      :: String
    stored_code :: Tuple{String,Bool}
    fresh_code  :: Tuple{String,Bool}
    divergences :: Vector{Divergence}
end

function Base.show(io::IO, ::MIME"text/plain", r::ReproductionReport)
    println(io, "ReproductionReport(", r.run_id, "): ", r.status)
    isempty(r.detail) || println(io, "  ", r.detail)
    println(io, "  stored code ", isempty(r.stored_code[1]) ? "(unknown)" : r.stored_code[1],
            r.stored_code[2] ? " (dirty)" : "",
            ", running code ", isempty(r.fresh_code[1]) ? "(unknown)" : r.fresh_code[1],
            r.fresh_code[2] ? " (dirty)" : "")
    for d in r.divergences
        println(io, "  - ", d)
    end
end

# Exact for everything that is an identity, a name or an instant; the
# documented absolute tolerance for finite floats only.
_same_value(a::Integer, b::Integer) = a == b
function _same_value(a::Real, b::Real)
    x, y = Float64(a), Float64(b)
    (isnan(x) || isnan(y)) && return isnan(x) && isnan(y)
    (isinf(x) || isinf(y)) && return x == y            # same sign, no subtraction
    return abs(x - y) <= REPRODUCE_TOL
end
_same_value(a, b) = isequal(a, b)

_show_value(x) = x === missing ? "missing" : x === nothing ? "nothing" : string(x)

# Compare two keyed row sets. A key present on one side only is a
# `:presence` divergence; matched rows compare field by field. Duplicate
# keys pair off in order, so two rows that share a key are still two rows.
function _compare_rows!(ds::Vector{Divergence}, output::Symbol,
                        stored::AbstractVector, fresh::AbstractVector,
                        keyof::Function, fieldsof::Function)
    index(rows) = begin
        d = Dict{String,Vector{Any}}()
        for x in rows
            push!(get!(() -> Any[], d, keyof(x)), x)
        end
        d
    end
    si, fi = index(stored), index(fresh)
    for key in sort!(collect(union(keys(si), keys(fi))))
        srows = get(si, key, Any[])
        frows = get(fi, key, Any[])
        for k in 1:max(length(srows), length(frows))
            row = length(srows) > 1 || length(frows) > 1 ? "$key #$k" : key
            if k > length(frows)
                push!(ds, Divergence(output, row, :presence, "present", "absent"))
            elseif k > length(srows)
                push!(ds, Divergence(output, row, :presence, "absent", "present"))
            else
                for ((sname, sv), (fname, fv)) in zip(fieldsof(srows[k]), fieldsof(frows[k]))
                    if sname !== fname
                        # The two rows do not have the same fields at all --
                        # two different event kinds at one sequence, say. The
                        # first field that disagrees is what happened; every
                        # field after it compares two unrelated things.
                        push!(ds, Divergence(output, row, :fields,
                                             string(sname), string(fname)))
                        break
                    end
                    _same_value(sv, fv) || push!(ds, Divergence(
                        output, row, sname, _show_value(sv), _show_value(fv)))
                end
            end
        end
    end
    return ds
end

# The comparable fields of one event, in the order they are written. The
# kind comes first: two events with the same sequence and different kinds
# disagree about what happened, and nothing below that is comparable.
_event_fields(e) = vcat(Pair{Symbol,Any}[
    :kind         => _kind_sql(e),
    :id           => event_id(e),
    :effective_at => effective_at(e),
    :recorded_at  => recorded_at(e),
], _event_kind_fields(e))

_event_kind_fields(e::Fill) = Pair{Symbol,Any}[
    :group => e.group, :order_leg_id => e.order_leg_id, :execution_id => e.execution_id,
    :contract => e.contract, :side => e.side, :intent => e.intent,
    :quantity => e.quantity, :price => e.price, :fill_rule => e.fill_rule]
_event_kind_fields(e::Match) = Pair{Symbol,Any}[
    :group => e.group, :open_fill_id => e.open_fill_id,
    :close_fill_id => e.close_fill_id, :quantity => e.quantity]
_event_kind_fields(e::Expiry) = Pair{Symbol,Any}[
    :group => e.group, :open_fill_id => e.open_fill_id, :contract => e.contract,
    :side => e.side, :quantity => e.quantity,
    :settlement_price => e.settlement_price, :outcome => e.outcome]
_event_kind_fields(e::Fee) = Pair{Symbol,Any}[
    :source_id => e.source_id, :amount => e.amount]

_order_fields(r::OrderRecord) = Pair{Symbol,Any}[
    :label => r.order.label, :group => r.group, :operation => r.order.operation,
    :first_leg_id => r.first_leg_id, :decided_at => r.decided_at,
    :known_to => r.known_to, :n_legs => length(r.order.legs)]

# One row per leg, carrying the observation it was priced against: what
# the decision saw is as much of the record as the fill it produced.
_order_leg_rows(L::Ledger) = [(r.order_id, k, leg, r.observations[k])
                              for r in L.orders for (k, leg) in enumerate(r.order.legs)]
_order_leg_fields(x) = begin
    (_, _, leg, o) = x
    Pair{Symbol,Any}[:contract => leg.contract, :side => leg.side, :intent => leg.intent,
                     :quantity => leg.quantity, :quote_at => o.quote_at, :bid => o.bid,
                     :ask => o.ask, :spot => o.spot, :spot_at => o.spot_at]
end

_curve_rows(c::Union{MarkedCurve,Nothing}) = c === nothing ? Tuple{Symbol,DateTime,Any}[] :
    vcat(Tuple{Symbol,DateTime,Any}[(:marked, t, p) for (t, p) in zip(c.timestamps, c.profit)],
         Tuple{Symbol,DateTime,Any}[(:unmarked, t, r) for (t, r) in zip(c.unmarked_at, c.unmarked_reason)])

"""
    reproduce(store::RunStore, run_id::AbstractString) -> ReproductionReport

Rerun a stored run against today's code and data and compare the fresh
outputs with the stored witness, field by field. Reads the saved inputs
and provenance, rebuilds the experiment from the stored `config.toml`,
runs it, and reports where the two disagree: event order and fields, order
records and their observations, curve instants and profits, failure
subjects and reasons, and metric names and values. A row present on one
side only is a divergence like any other.

It uses the running code and environment and reports their provenance
beside the recorded one; it does not switch checkouts or instantiate
packages inside the caller's process. The run's own `Manifest.toml` is
what makes a separate rerun under the recorded environment possible.

**It never writes.** No result is saved over the witness, and no stored
byte is touched; refreshing a run's provenance after a successful
reproduction is a separate utility, never a side effect of this one.

Returns a [`ReproductionReport`](@ref) naming success, divergence or
inability. Inability covers market data this machine cannot open -- which
is never reported as a successful comparison of empty outputs -- and a
stored config that no longer hashes to the folder it sits in, which is
named as an identity mismatch with the regenerated projection, rather than
quietly reproducing whatever run the new id points at. Anything else
propagates: this is not a general exception catcher.
"""
function reproduce(store::RunStore, run_id::AbstractString)::ReproductionReport
    _assert_open(store)
    # Load first: a folder that is not there, or a schema this store does
    # not read, is a load failure with its own message, not a report.
    stored_result = load_run(store, run_id)
    stored_manifest = _load_manifest(store, run_dir(store, run_id))
    stored_code = (stored_manifest.commit_sha, stored_manifest.dirty)
    fresh_code = code_provenance()
    report(status, detail, ds = Divergence[]) =
        ReproductionReport(String(run_id), status, detail, stored_code, fresh_code, ds)

    exp = stored_result.experiment
    id_now = full_hash(exp)
    if id_now != run_id
        # The projection that defines sameness has changed under this
        # config. Regenerate it here rather than look up the new id: the
        # run at that id, if any, is a different run.
        return report(:unreproducible,
            "identity mismatch: the stored config now hashes to $id_now, not $run_id. " *
            "The identity projection has changed since this run was saved; its " *
            "regenerated form is: " * _canonical(_full_dict(exp)))
    end

    # Openability is the declared boundary. Probing it first separates "the
    # data is not on this machine" -- an inability -- from a failure inside
    # the run, which is a real defect and propagates.
    d = try
        open_data(exp.data)
    catch e
        return report(:unreproducible,
            "the run's market data could not be opened on this machine: " * sprint(showerror, e))
    end
    _close_quietly(d)

    fresh = run_experiment(exp)
    ds = Divergence[]
    _compare_rows!(ds, :events, stored_result.ledger.events, fresh.ledger.events,
                   e -> "sequence " * string(sequence(e)), _event_fields)
    _compare_rows!(ds, :orders, stored_result.ledger.orders, fresh.ledger.orders,
                   r -> "order " * string(r.order_id), _order_fields)
    _compare_rows!(ds, :order_legs, _order_leg_rows(stored_result.ledger),
                   _order_leg_rows(fresh.ledger),
                   x -> "order $(x[1]) leg $(x[2])", _order_leg_fields)
    if (stored_result.curve === nothing) != (fresh.curve === nothing)
        push!(ds, Divergence(:curve, "curve", :presence,
                             stored_result.curve === nothing ? "absent" : "present",
                             fresh.curve === nothing ? "absent" : "present"))
    else
        _compare_rows!(ds, :curve, _curve_rows(stored_result.curve), _curve_rows(fresh.curve),
                       x -> string(x[1]) * " " * string(x[2]),
                       x -> Pair{Symbol,Any}[(x[1] === :marked ? :profit : :reason) => x[3]])
    end
    _compare_rows!(ds, :failures, stored_result.failures, fresh.failures,
                   f -> "$(f.at) $(f.stage) $(f.subject)",
                   f -> Pair{Symbol,Any}[:reason => f.reason])
    _compare_rows!(ds, :metrics,
                   collect(pairs(stored_result.metrics)), collect(pairs(fresh.metrics)),
                   p -> "metric " * string(first(p)),
                   p -> Pair{Symbol,Any}[:value => last(p)])

    return isempty(ds) ? report(:reproduced, "") : report(:diverged, "", ds)
end
