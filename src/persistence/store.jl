# `persistence` module: the knowledge base.
#
# Every run that lands here writes its config and its outcome to disk under
# a content-addressed folder, so prior experiments stay queryable and
# comparable rather than evaporating into ad-hoc notebooks (see vision.md).
#
# Storage shape: Hive-partitioned parquet under `<root>/runs/run_id=<hash>/`.
# Each run folder holds the verbatim input TOML plus six parquet files
# (manifest, metrics, events, orders, order_legs, pnl_series). Cross-run
# queries are just DuckDB SQL against the partitioned trees -- this module
# does not invent a query API. Same DuckDB-as-engine / parquet-as-storage
# pattern the `data` module uses for input data.
#
# The ledger is written as it is: one row per event with the kind's own
# columns and NULLs elsewhere, one row per order and one per order leg
# with its observation. `load_run` rebuilds it through the ledger's own
# validated write path and the fill-to-order join, so a stored run that
# breaks an invariant fails to load by name rather than loading wrong.
#
# Run identity is `full_hash(result.experiment)` -- the canonical hash of
# the resolved experiment (see experiment/identity.jl), not the raw TOML
# bytes. The verbatim config.toml is still stored for human reading and to
# rebuild the experiment on load. The manifest also records `core_hash`
# (the backtest-only identity, shared by output variations of one backtest)
# and code provenance (`commit_sha` / `dirty`).

using DuckDB
using DuckDB: DBInterface

# Manifest schema version, outside the run hash. Bumped by the data-kinds
# migration (every run id changed with the identity projection) and again
# when the ledger replaced `positions.parquet` (slice 2 of the ledger
# rebuild); `load_run` refuses a run written under another version rather
# than rebuilding a result its files cannot describe.
const RUN_SCHEMA_VERSION = 3

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

# --- save_run ------------------------------------------------------------

"""
    save_run(store::RunStore, result::ExperimentResult,
             config_toml::AbstractString;
             commit_sha::AbstractString="", dirty::Bool=true) -> String

Persist `result` plus its originating TOML config under
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

Writes:
- `config.toml` -- the bytes passed in, verbatim.
- `manifest.parquet` -- one row of run-level metadata (incl. `core_hash`,
  `commit_sha`, `dirty`, `n_events`, `n_orders`).
- `metrics.parquet` -- long form, one row per `(metric_name, value)`.
- `events.parquet` -- one row per ledger event in sequence order.
- `orders.parquet` -- one row per order record.
- `order_legs.parquet` -- one row per order leg with its observation.
- `pnl_series.parquet` -- one row per sample in `result.pnl_series`.

If a folder for this id already exists, its contents are overwritten:
same resolved experiment means same id, so re-saving is idempotent in
intent (the latest run of that exact experiment wins).

Atomicity is best-effort today: a crash mid-write can leave a
half-written folder. Re-running the same experiment recovers it. A
write-to-temp-then-rename pass is queued for the next iteration.
"""
function save_run(store::RunStore, result::ExperimentResult,
                  config_toml::AbstractString;
                  commit_sha::AbstractString="", dirty::Bool=true)::String
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

    _write_manifest(store, dir, id, result; commit_sha=commit_sha, dirty=dirty)
    _write_metrics(store, dir, id, result)
    _write_events(store, dir, id, result.ledger)
    _write_orders(store, dir, id, result.ledger)
    _write_order_legs(store, dir, id, result.ledger)
    _write_pnl_series(store, dir, id, result)

    return id
end

function _write_manifest(store::RunStore, dir::AbstractString, id::AbstractString,
                         result::ExperimentResult;
                         commit_sha::AbstractString, dirty::Bool)
    exp = result.experiment
    s   = result.pnl_series
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
        n_unmarked BIGINT,
        window_end_spot DOUBLE,
        commit_sha VARCHAR,
        dirty BOOLEAN,
        written_at TIMESTAMP,
        schema_version INTEGER
    )"""
    insert = "INSERT INTO _writebuf VALUES (" * join([
        _str_sql(id),
        _str_sql(core_hash(exp)),
        _str_sql(exp.name),
        _dt_sql(exp.from),
        _dt_sql(exp.to),
        string(length(result.ledger)),
        string(length(result.ledger.orders)),
        string(s.n_opens),
        string(s.n_closes),
        string(s.n_unmarked),
        _f_sql(s.window_end_spot),
        _str_sql(commit_sha),
        dirty ? "TRUE" : "FALSE",
        _dt_sql(Dates.now(UTC)),
        string(RUN_SCHEMA_VERSION),
    ], ", ") * ")"
    _write_parquet(store, joinpath(dir, "manifest.parquet"), schema, [insert])
end

function _write_metrics(store::RunStore, dir::AbstractString, id::AbstractString,
                        result::ExperimentResult)
    schema = "(run_id VARCHAR, metric_name VARCHAR, value DOUBLE)"
    inserts = String[]
    for (k, v) in pairs(result.metrics)
        val = v isa Real ? Float64(v) : NaN
        push!(inserts,
              "INSERT INTO _writebuf VALUES (" *
              join([_str_sql(id), _str_sql(String(k)), _f_sql(val)], ", ") * ")")
    end
    _write_parquet(store, joinpath(dir, "metrics.parquet"), schema, inserts)
end

# One row per event in sequence order: the header, the kind, and the
# kind's own columns; everything the kind lacks is NULL. `group` is a SQL
# keyword, so the column is `group_id`.
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

# The kind-specific columns of one event, in schema order after
# `recorded_at`: group_id .. amount.
_event_columns(e::Fill) = [
    string(e.group), string(e.order_leg_id), string(e.execution_id),
    _str_sql(ticker(e.contract.underlying)), _f_sql(e.contract.strike),
    _dt_sql(e.contract.expiry), _otype_sql(e.contract.option_type),
    _side_sql(e.side), _intent_sql(e.intent), string(e.quantity), _f_sql(e.price),
    _str_sql(String(e.fill_rule)),
    "NULL", "NULL", "NULL", "NULL", "NULL", "NULL",
]
_event_columns(e::Match) = [
    string(e.group), "NULL", "NULL",
    "NULL", "NULL", "NULL", "NULL",
    "NULL", "NULL", string(e.quantity), "NULL", "NULL",
    string(e.open_fill_id), string(e.close_fill_id), "NULL", "NULL", "NULL", "NULL",
]
_event_columns(e::Expiry) = [
    string(e.group), "NULL", "NULL",
    _str_sql(ticker(e.contract.underlying)), _f_sql(e.contract.strike),
    _dt_sql(e.contract.expiry), _otype_sql(e.contract.option_type),
    _side_sql(e.side), "NULL", string(e.quantity), "NULL", "NULL",
    string(e.open_fill_id), "NULL", _f_sql(e.settlement_price), _outcome_sql(e.outcome),
    "NULL", "NULL",
]
_event_columns(e::Fee) = [
    "NULL", "NULL", "NULL",
    "NULL", "NULL", "NULL", "NULL",
    "NULL", "NULL", "NULL", "NULL", "NULL",
    "NULL", "NULL", "NULL", "NULL", string(e.source_id), string(e.amount),
]

_kind_sql(::Fill) = "'fill'"
_kind_sql(::Match) = "'match'"
_kind_sql(::Expiry) = "'expiry'"
_kind_sql(::Fee) = "'fee'"

function _write_events(store::RunStore, dir::AbstractString, id::AbstractString, L::Ledger)
    inserts = String[]
    for e in L.events
        cols = vcat([_str_sql(id), string(sequence(e)), string(event_id(e)), _kind_sql(e),
                     _dt_sql(effective_at(e)), _dt_sql(recorded_at(e))],
                    _event_columns(e))
        push!(inserts, "INSERT INTO _writebuf VALUES (" * join(cols, ", ") * ")")
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
        push!(inserts,
              "INSERT INTO _writebuf VALUES (" *
              join([
                  _str_sql(id),
                  string(r.order_id),
                  string(r.first_leg_id),
                  _str_sql(String(r.order.label)),
                  string(r.group),
                  _opt_sql(r.order.operation, string),
                  _dt_sql(r.decided_at),
                  string(r.known_to),
              ], ", ") * ")")
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
        push!(inserts,
              "INSERT INTO _writebuf VALUES (" *
              join([
                  _str_sql(id),
                  string(r.order_id),
                  string(r.first_leg_id + k - 1),
                  string(k),
                  _str_sql(ticker(leg.contract.underlying)),
                  _f_sql(leg.contract.strike),
                  _dt_sql(leg.contract.expiry),
                  _otype_sql(leg.contract.option_type),
                  _side_sql(leg.side),
                  _intent_sql(leg.intent),
                  string(leg.quantity),
                  _dt_sql(o.quote_at),
                  _opt_sql(o.bid, _f_sql),
                  _opt_sql(o.ask, _f_sql),
                  _f_sql(o.spot),
                  _dt_sql(o.spot_at),
              ], ", ") * ")")
    end
    _write_parquet(store, joinpath(dir, "order_legs.parquet"), schema, inserts)
end

function _write_pnl_series(store::RunStore, dir::AbstractString, id::AbstractString,
                           result::ExperimentResult)
    schema = "(run_id VARCHAR, idx BIGINT, timestamp TIMESTAMP, pnl DOUBLE)"
    s = result.pnl_series
    inserts = String[]
    for i in eachindex(s.timestamps)
        push!(inserts,
              "INSERT INTO _writebuf VALUES (" *
              join([
                  _str_sql(id),
                  string(i),
                  _dt_sql(s.timestamps[i]),
                  _f_sql(s.pnl[i]),
              ], ", ") * ")")
    end
    _write_parquet(store, joinpath(dir, "pnl_series.parquet"), schema, inserts)
end

# --- load_run ------------------------------------------------------------

# Always-on metrics in `compute_metrics` are emitted in this fixed order;
# `n_round_trips`, `n_opens`, and `n_closes` are integers, everything
# else is Float64. The load path uses this to round-trip the NamedTuple
# faithfully (integers stay integers, key order matches `compute_metrics`).
const _ALWAYS_ON_METRIC_KEYS = (:total_pnl, :n_round_trips, :n_opens, :n_closes, :hit_rate)
const _INT_METRIC_KEYS = (:n_round_trips, :n_opens, :n_closes)

"""
    load_run(store::RunStore, run_id::AbstractString) -> ExperimentResult

Rehydrate a previously [`save_run`](@ref)-saved run back into an
`ExperimentResult`. Reads the six parquet artifacts plus the saved
`config.toml`, rebuilds the live `Experiment` via
[`load_experiment_str`](@ref), rebuilds the ledger, and reconstructs
`pnl_series` and the `metrics` NamedTuple (preserving the integer types
of `n_round_trips`, `n_opens`, `n_closes`).

The ledger is rebuilt through its own write path: every event is built
through its constructor in sequence order and committed to a fresh
`Ledger` as one batch through `commit!` (the book is empty, so FIFO
among batch-opened lots is sequence order), so a loaded ledger has
passed every append-time check; the order records are rebuilt from the
two order tables, every counter is set one past the largest id seen
(groups included), and `check_join` runs last. A load that fails a
check throws that check's named failure; it never drops the join.

The data declared by the saved config does not need to be present on
disk: `Experiment.data` holds provider specs, which are pure values, so
the rebuilt experiment only fails at `open_data` (i.e. at
`run_experiment`) if the data has moved. Inspecting the persisted
fields (`ledger`, `pnl_series`, `metrics`) needs no data at all.

Throws `ArgumentError` if the run folder or any of the expected files
is missing, or if the manifest's `schema_version` is absent or differs
from `RUN_SCHEMA_VERSION` (a run written before the ledger replaced
`positions.parquet`): rerun its config to regenerate it.
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
    config_toml = read(cfg_path, String)
    exp = load_experiment_str(config_toml)

    ledger = _load_ledger(store, dir)
    series = _load_pnl_series(store, dir, manifest)
    metrics = _load_metrics(store, dir, exp.outputs.metrics)

    return ExperimentResult(exp, ledger, series, metrics)
end

function _select_rows(store::RunStore, path::AbstractString, sql::AbstractString)
    isfile(path) || throw(ArgumentError("load_run: missing $(basename(path)) at $path"))
    return collect(DBInterface.execute(store.con, sql))
end

# `SELECT *` so a manifest written without `schema_version` (an old run)
# still reads; the missing column reports as version 0.
function _load_manifest(store::RunStore, dir::AbstractString)
    path = joinpath(dir, "manifest.parquet")
    rows = _select_rows(store, path, "SELECT * FROM '$(_sql_pq_path(path))'")
    length(rows) == 1 ||
        throw(ArgumentError("load_run: manifest.parquet must have exactly 1 row, got $(length(rows))"))
    r = first(rows)
    version = :schema_version in propertynames(r) && r.schema_version !== missing ?
        Int(r.schema_version) : 0
    return (window_end_spot=Float64(r.window_end_spot),
            n_opens=Int(r.n_opens),
            n_closes=Int(r.n_closes),
            n_unmarked=Int(r.n_unmarked),
            schema_version=version)
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
    L, book = Ledger(), Book()
    # One batch through the validated write path: ids and sequences must
    # continue a fresh ledger's counters, references must point backward,
    # matches must be FIFO, cash must be whole cents -- as when written.
    commit!(L, book, LedgerEvent[_event_from(r) for r in rows])

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

function _load_pnl_series(store::RunStore, dir::AbstractString,
                          manifest::NamedTuple)::PnLSeries
    path = joinpath(dir, "pnl_series.parquet")
    rows = _select_rows(store, path,
        "SELECT timestamp, pnl FROM '$(_sql_pq_path(path))' ORDER BY idx")
    timestamps = DateTime[DateTime(r.timestamp) for r in rows]
    pnl        = Float64[Float64(r.pnl)         for r in rows]
    return PnLSeries(timestamps, pnl,
                     manifest.window_end_spot,
                     manifest.n_opens, manifest.n_closes, manifest.n_unmarked)
end

function _load_metrics(store::RunStore, dir::AbstractString,
                       requested::Vector{Symbol})::NamedTuple
    path = joinpath(dir, "metrics.parquet")
    rows = _select_rows(store, path,
        "SELECT metric_name, value FROM '$(_sql_pq_path(path))'")
    raw = Dict{Symbol,Float64}()
    for r in rows
        raw[Symbol(r.metric_name)] = Float64(r.value)
    end
    # Build the NamedTuple in canonical order: always-on first, then
    # optional in the requested order (matches `compute_metrics`).
    keys = Symbol[]
    vals = Any[]
    for k in _ALWAYS_ON_METRIC_KEYS
        haskey(raw, k) || continue
        push!(keys, k)
        push!(vals, k in _INT_METRIC_KEYS ? Int(raw[k]) : raw[k])
    end
    for k in requested
        haskey(raw, k) || continue   # caller-requested metric absent (skip rather than error)
        k in _ALWAYS_ON_METRIC_KEYS && continue   # don't double-add
        push!(keys, k)
        push!(vals, raw[k])
    end
    return NamedTuple{Tuple(keys)}(Tuple(vals))
end
