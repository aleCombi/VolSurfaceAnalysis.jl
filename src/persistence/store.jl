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
# breaks an invariant fails to load by name rather than loading wrong. Each
# stored leg's own identity is checked against the order that claims it and
# every leg row is accounted for, so a changed `order_leg_id`, a shifted
# `leg_idx` or a leg belonging to no order is named rather than sorted
# away. The manifest's counts are checked against what was rebuilt -- every
# output table has membership evidence, and the failures have one count per
# stage, so recorded absence, truncation and a row moved from one stage to
# another stay different facts (design rule 7) -- and the curve's unmarked
# entries against the mark-stage failures, instant *and* reason, so a
# truncated table, a mixed save or contradictory evidence is caught even
# when every individual constructor is satisfied.
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
using SHA: sha256
using TOML

# Manifest schema version, outside the run hash. `load_run` refuses a run
# written under another version rather than rebuilding a result its files
# cannot describe; the message says to rerun the config, because no
# migration can recover a curve or a failure table that was never written.
const RUN_SCHEMA_VERSION = 8

# The manifest column carrying the membership evidence for one failure
# stage. One count per stage, not one for the table: a single total is
# blind to a row that moved from one stage to another, and the two stages
# are guarded by different things -- the mark rows by the curve they must
# agree with, the settlement rows by nothing else at all.
_failure_count_column(stage::Symbol)::String = "n_" * String(stage) * "_failures"

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

# A failure outside the closed stage vocabulary is a producer bug, and it is
# refused BEFORE any file is touched. Refusing it inside `_write_failures`
# was too late: by then every other file had been overwritten, and an
# existing folder for the same id kept its old `failures.parquet` under a
# manifest that no longer described it -- a mixed record that passed every
# load check and silently lost the failure. So this runs beside
# `check_join`, in the preflight, where a refusal leaves the folder as it was.
function _check_failure_stages(failures::AbstractVector{RunFailure})
    for f in failures
        f.stage in RUN_FAILURE_STAGES || throw(ArgumentError(
            "save_run: failure at $(f.at) names the stage :$(f.stage), but a " *
            "run's failures come from the stages " *
            "$(join((":" * String(s) for s in RUN_FAILURE_STAGES), ", ")); " *
            "no pass asks a question under :$(f.stage)"))
    end
    return nothing
end

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

A failure naming a stage no pass emits is refused rather than written: the
manifest counts the failures stage by stage, so such a row would land in
the record with no count covering it.

Writes, as two inputs and five outputs:

- `config.toml`, `Manifest.toml` -- the bytes passed in, verbatim.
- `manifest.parquet` -- one row indexing the run: both hashes, the name,
  the window, the membership counts (the ledger's four, the curve's two,
  the metrics' one, and one per failure stage), the code provenance, the
  write time and the schema version.
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
    _check_failure_stages(result.failures)      # before mkpath: a refusal touches nothing
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
        n_metrics BIGINT,
        $(join(("$(_failure_count_column(s)) BIGINT" for s in RUN_FAILURE_STAGES), ",\n        ")),
        commit_sha VARCHAR,
        dirty BOOLEAN,
        written_at TIMESTAMP,
        schema_version INTEGER
    )"""
    # `n_marked` / `n_unmarked` are NULL when the run carries no curve at
    # all, which is a different fact from a curve that marked nothing.
    # `n_metrics` and the per-stage failure counts are never NULL: both
    # tables are always written, so zero means "asked, and nothing to
    # record" while a missing count would mean the index cannot say -- and
    # an index that cannot say is exactly what lets a truncated table read
    # as an empty one. The failures are counted per stage, because one
    # total cannot tell a settlement failure from a mark failure and so
    # accepts a record whose rows changed stage.
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
        "n_metrics"      => string(length(result.metrics)),
    ]
    for s in RUN_FAILURE_STAGES
        push!(cols, _failure_count_column(s) =>
                    string(count(f -> f.stage === s, result.failures)))
    end
    append!(cols, [
        "commit_sha"     => _str_sql(commit_sha),
        "dirty"          => dirty ? "TRUE" : "FALSE",
        "written_at"     => _dt_sql(Dates.now(UTC)),
        "schema_version" => string(RUN_SCHEMA_VERSION),
    ])
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
    # The manifest counts the failures stage by stage, so a stage no pass
    # emits would be written with no count covering it -- and a row nothing
    # counts is exactly the membership hole the counts exist to close.
    _check_failure_stages(failures)              # preflighted in save_run; cheap to repeat
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
the manifest's counts against what was rebuilt, naming the column and both
values on a mismatch; every stored leg against the order that claims it,
so a changed `order_leg_id`, a shifted `leg_idx` or a leg no order owns is
named rather than sorted away; and the curve's unmarked entries against
the **mark-stage** failures, which must agree instant *and* reason. Every
output table has membership evidence, so a truncated table and a table
that recorded nothing stay different facts (design rule 7): an empty
`metrics.parquet` is a defect unless the run reported no metrics. The
failures are counted **per stage** and their `stage` is a closed
vocabulary, so a row retagged from one stage to another, or one naming a
stage no pass emits, is refused too: a single total would accept both, and
the curve answers for the mark rows only. NULL curve counts mean the run carried
no curve and `curve.parquet` must be absent; zero counts mean a recorded
curve with no entries of that kind, and the file must be there. Counts are
consistency checks -- they expose a truncated table or a mixed save that
every individual constructor would accept -- not proof that any price or
metric is right; a plausible but stale price loads.

Throws `ArgumentError` if the run folder or any required file is missing,
or if the manifest's `schema_version` is absent or differs from
`RUN_SCHEMA_VERSION`: rerun its config to regenerate it. There is no
migration -- an older manifest never wrote today's counts down, and a
run's witness cannot be invented after the fact.
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
    failures = _load_failures(store, dir, run_id)
    metrics  = _load_metrics(store, dir)

    _check_counts(run_id, manifest, ledger, curve, metrics, failures)
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

function _check_counts(run_id, manifest, L::Ledger, curve::Union{MarkedCurve,Nothing},
                       metrics::NamedTuple, failures::AbstractVector{RunFailure})
    _count_mismatch(run_id, "n_events", manifest.n_events, length(L))
    _count_mismatch(run_id, "n_orders", manifest.n_orders, length(L.orders))
    _count_mismatch(run_id, "n_opens", manifest.n_opens, n_opens(L))
    _count_mismatch(run_id, "n_closes", manifest.n_closes, n_closes(L))
    # Membership evidence for the two tables no structural relationship
    # covers. Without it an empty `metrics.parquet` reads as a run that
    # reported no metrics, and a deleted settlement failure -- or one of two
    # mark failures at a session that keeps the other -- reads as a question
    # nobody asked.
    #
    # The failures are counted **per stage**. A single total is blind to
    # membership *within* the table: it accepts a record that keeps the
    # number of rows and moves one between stages, and the two stages are
    # answered for by different things -- a mark row by the curve instant it
    # must agree with, a settlement row by nothing else at all. So retagging
    # one of two mark failures as a settlement failure, or deleting a
    # settlement row and duplicating a mark row, would both pass a total and
    # every structural check there is; only a count per stage knows.
    counts = Pair{String,Int}["n_metrics" => length(metrics)]
    for s in RUN_FAILURE_STAGES
        push!(counts, _failure_count_column(s) => count(f -> f.stage === s, failures))
    end
    for (col, actual) in counts
        stored = getproperty(manifest, Symbol(col))
        stored === nothing && throw(ArgumentError(
            "load_run: run $run_id manifest records no $col; without it a " *
            "truncated table reads as one that recorded nothing -- rerun " *
            "the config to regenerate the run"))
        _count_mismatch(run_id, col, stored, actual)
    end
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

# The curve says a session could not be marked and why; the failures say
# which questions that session left unanswered and with what named reason.
# One implies the other: a session is unmarked exactly when at least one of
# its lots could not be priced, and a mark failure is recorded exactly at a
# session the curve left unmarked. The reason the curve carries is one of
# the reasons its lots gave, so it must occur among them -- matching
# instants alone would accept a curve reporting `:no_mark` at a session
# whose every failure says `:unexpected_gap`, which is two tables
# describing two different runs. A truncation, or a rewritten reason on
# either side, breaks the agreement.
#
# It sees the **mark** rows only, deliberately: a settlement failure is
# about a lot the lifecycle could not close, and no curve entry answers for
# one. Their membership is `n_settlement_failures`, and the mark rows'
# is `n_mark_failures`, which is what stops a row from being moved between
# the two stages under a total that never changes.
function _check_curve_failure_agreement(run_id, curve::Union{MarkedCurve,Nothing},
                                        failures::AbstractVector{RunFailure})
    reasons_at = Dict{DateTime,Vector{Symbol}}()
    for f in failures
        f.stage === :mark || continue
        push!(get!(() -> Symbol[], reasons_at, f.at), f.reason)
    end
    unmarked = curve === nothing ? DateTime[] : curve.unmarked_at
    curve_reasons = curve === nothing ? Symbol[] : curve.unmarked_reason
    missing_rows = sort!(collect(setdiff(Set(unmarked), keys(reasons_at))))
    isempty(missing_rows) || throw(ArgumentError(
        "load_run: run $run_id has unmarked curve instants with no failure " *
        "recorded against them: $(missing_rows)"))
    for (t, r) in zip(unmarked, curve_reasons)
        r in reasons_at[t] || throw(ArgumentError(
            "load_run: run $run_id leaves $t unmarked for reason :$r, but " *
            "the mark failures recorded at that instant give only " *
            "$(sort(unique(reasons_at[t]))); the curve and the failures " *
            "describe different runs"))
    end
    extra_rows = sort!(collect(setdiff(keys(reasons_at), Set(unmarked))))
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
    # One entry per failure stage, under the column the writer names it by,
    # so a stage added to `RUN_FAILURE_STAGES` needs no second edit here.
    stage_counts = NamedTuple{Tuple(Symbol(_failure_count_column(s)) for s in RUN_FAILURE_STAGES)}(
        Tuple(num(Symbol(_failure_count_column(s))) for s in RUN_FAILURE_STAGES))
    return merge((schema_version = version,
            core_hash  = String(r.core_hash),
            n_events   = num(:n_events),
            n_orders   = num(:n_orders),
            n_opens    = num(:n_opens),
            n_closes   = num(:n_closes),
            n_marked   = num(:n_marked),
            n_unmarked = num(:n_unmarked),
            n_metrics  = num(:n_metrics),
            commit_sha = has(:commit_sha) ? String(r.commit_sha) : "",
            dirty      = has(:dirty) ? Bool(r.dirty) : true), stage_counts)
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
    consumed = 0
    for r in order_rows
        oid = Int(r.order_id)
        g = Int(r.group_id)
        own_group = g in minted ? g : nothing
        push!(minted, g)
        lrows = get(legs_by_order, oid, Any[])
        consumed += length(lrows)
        # `leg_idx` and `order_leg_id` are stored facts, not sort keys. The
        # legs of an order are 1..n in order, and leg k's id is the order's
        # `first_leg_id + k - 1` -- the same arithmetic the writer used and
        # the ledger mints by. Checking them here is what stops a rewritten
        # id, or indices shifted without changing their order, from being
        # sorted back into a ledger that looks untouched.
        first_leg = Int(r.first_leg_id)
        for (k, l) in enumerate(lrows)
            Int(l.leg_idx) == k || throw(ArgumentError(
                "load_run: order_legs.parquet gives order $oid a leg_idx of " *
                "$(l.leg_idx) where leg $k was expected; an order's legs are " *
                "1..n in order"))
            Int(l.order_leg_id) == first_leg + k - 1 || throw(ArgumentError(
                "load_run: order_legs.parquet gives order $oid leg $k the " *
                "order_leg_id $(l.order_leg_id), but the order's first_leg_id " *
                "$first_leg makes it $(first_leg + k - 1)"))
        end
        legs = Leg[Leg(_contract_from(l), _side_from(l.side), Int(l.quantity), _intent_from(l.intent))
                   for l in lrows]
        obs = LegObservation[LegObservation(DateTime(l.quote_at), _opt_from(l.bid, Float64),
                                            _opt_from(l.ask, Float64), Float64(l.spot),
                                            DateTime(l.spot_at)) for l in lrows]
        order = Order(Symbol(String(r.label)), legs; group = own_group,
                      operation = r.operation === missing ? nothing : Int(r.operation))
        push!(L.orders, OrderRecord(oid, first_leg, g, DateTime(r.decided_at),
                                    Int(r.known_to), order, obs))
    end
    # Every input row is accounted for. A leg whose order_id names no order
    # would otherwise be dropped in silence -- the loader would build a
    # ledger from a strict subset of the table and call it the record.
    if consumed != length(leg_rows)
        orphans = sort!(collect(setdiff(keys(legs_by_order),
                                        Set(Int(r.order_id) for r in order_rows))))
        throw(ArgumentError(
            "load_run: order_legs.parquet holds $(length(leg_rows)) rows but " *
            "the stored orders account for $consumed" *
            (isempty(orphans) ? "; orders.parquet claims an order id twice" :
             "; legs are recorded against order ids no order claims: $orphans")))
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
#
# `stage` is a closed vocabulary, checked here: a row naming a stage no
# pass emits is a defect, not a question this run asked. It has to be
# refused rather than counted, because the per-stage counts cover exactly
# the stages the writer emits -- a row outside them is a row nothing counts.
function _load_failures(store::RunStore, dir::AbstractString,
                        run_id::AbstractString)::Vector{RunFailure}
    path = joinpath(dir, "failures.parquet")
    rows = _select_rows(store, path,
        "SELECT * FROM '$(_sql_pq_path(path))' ORDER BY instant, stage, subject, reason")
    out = RunFailure[RunFailure(DateTime(r.instant), Symbol(String(r.stage)),
                                String(r.subject), Symbol(String(r.reason))) for r in rows]
    for f in out
        f.stage in RUN_FAILURE_STAGES || throw(ArgumentError(
            "load_run: run $run_id records a failure at $(f.at) under the " *
            "stage :$(f.stage), but a run's failures come from the stages " *
            "$(join((":" * String(s) for s in RUN_FAILURE_STAGES), ", ")); " *
            "no pass asks a question under :$(f.stage)"))
    end
    return out
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
`(commit_sha, dirty)`; `stored_deps` and `fresh_deps` are the digests of
the two dependency documents and `deps_changed` names every package whose
version moved between them, matched by UUID rather than by name, because
two distinct packages may share a name. Two documents that differ while
every recorded version agrees say that instead. Dataset versioning is deliberately not in
identity (see the module doc), so a divergence attributes to code or
dependencies by elimination -- which needs **both** environments named,
not only both commits. Identical commits with different dependencies are a
real and ordinary case here: `backtest/settlement.jl` consults the NYSE
calendar `BusinessDays` ships, so the code alone does not pin the calendar
a run read. Differing dependencies are provenance, not divergence: they do
not change the status, they say what a controlled rerun has to separate.
"""
struct ReproductionReport
    run_id       :: String
    status       :: Symbol
    detail       :: String
    stored_code  :: Tuple{String,Bool}
    fresh_code   :: Tuple{String,Bool}
    stored_deps  :: String
    fresh_deps   :: String
    deps_changed :: Vector{String}
    divergences  :: Vector{Divergence}
end

function Base.show(io::IO, ::MIME"text/plain", r::ReproductionReport)
    println(io, "ReproductionReport(", r.run_id, "): ", r.status)
    isempty(r.detail) || println(io, "  ", r.detail)
    println(io, "  stored code ", isempty(r.stored_code[1]) ? "(unknown)" : r.stored_code[1],
            r.stored_code[2] ? " (dirty)" : "",
            ", running code ", isempty(r.fresh_code[1]) ? "(unknown)" : r.fresh_code[1],
            r.fresh_code[2] ? " (dirty)" : "")
    println(io, "  stored deps ", r.stored_deps, ", running deps ", r.fresh_deps,
            isempty(r.deps_changed) ? " (same document)" : " (differ)")
    for c in r.deps_changed
        println(io, "  ~ ", c)
    end
    for d in r.divergences
        println(io, "  - ", d)
    end
end

# --- dependency provenance ----------------------------------------------
# The two environments, as Pkg records them. `reproduce` reads the stored
# `Manifest.toml` and the running one for provenance only: it still
# instantiates nothing and switches no checkout. A digest identifies each
# document the way a commit sha identifies a checkout; the version map is
# what lets a difference be named package by package instead of as two
# opaque hashes.

_deps_digest(toml::AbstractString)::String =
    isempty(toml) ? "(unknown)" : bytes2hex(sha256(codeunits(String(toml))))[1:16]

# UUID => (name, resolved version), plus the Julia version the environment
# was resolved for. Manifest format 2.0 nests packages under `deps`;
# format 1.0 puts them at the top level. A stdlib entry carries no version
# and contributes nothing to name.
#
# The key is the **UUID**, not the name. Pkg allows two distinct packages
# with the same name in one environment (see the Conventions section of the
# module doc), and a map keyed by name collapses them: the later entry
# overwrites the earlier one, so a version change in the earlier package
# disappears from the report entirely. The name stays, for display.
function _deps_versions(toml::AbstractString)::Dict{String,Tuple{String,String}}
    out = Dict{String,Tuple{String,String}}()
    isempty(toml) && return out
    parsed = try
        TOML.parse(String(toml))
    catch
        return out
    end
    jv = get(parsed, "julia_version", nothing)
    jv isa AbstractString && (out["julia_version"] = ("julia", String(jv)))
    entries = get(parsed, "deps", parsed)
    entries isa AbstractDict || return out
    for (name, es) in entries
        es isa AbstractVector || continue
        for e in es
            e isa AbstractDict || continue
            v = get(e, "version", nothing)
            v isa AbstractString || continue
            u = get(e, "uuid", nothing)
            # A package with no uuid at all is not something Pkg writes; key
            # it by name so it is still reported rather than dropped.
            out[u isa AbstractString ? String(u) : "name:" * String(name)] =
                (String(name), String(v))
        end
    end
    return out
end

# One line per package that moved, `absent` naming either side that does
# not have it at all. Packages are matched by UUID; the uuid joins the line
# only when the name alone would not say which package moved, which is the
# whole reason the map is not keyed by name.
#
# Two documents that differ while their version maps agree say exactly
# that: many versions may be recorded and none of them moved, and only a
# comment, the project hash or a dependency path changed.
function _deps_changes(stored::AbstractString, fresh::AbstractString)::Vector{String}
    s, f = _deps_versions(stored), _deps_versions(fresh)
    ks = collect(union(keys(s), keys(f)))
    name_of(k) = first(get(s, k, get(f, k, ("?", ""))))
    version_of(d, k) = haskey(d, k) ? last(d[k]) : "absent"
    shared = Set(n for n in unique(name_of.(ks)) if count(==(n), name_of.(ks)) > 1)
    sort!(ks; by = k -> (name_of(k), k))
    changed = [name_of(k) * (name_of(k) in shared ? " [$k]" : "") * " " *
               version_of(s, k) * " -> " * version_of(f, k)
               for k in ks if version_of(s, k) != version_of(f, k)]
    if isempty(changed) && _deps_digest(stored) != _deps_digest(fresh)
        return ["the two dependency documents differ; no recorded version changes"]
    end
    return changed
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

It uses the running code and environment and reports **both** provenances
beside the recorded ones: the two commits, and the two dependency
documents by digest with every package whose version moved named. Two
identical commits with different dependencies are otherwise
indistinguishable in a report, and they are not the same run -- the
settlement rule reads the calendar a dependency ships. Differing
dependencies never change the status; they say what a controlled rerun has
to separate. Reading the two documents is all it does with them: it does
not switch checkouts or instantiate packages inside the caller's process.
The run's own `Manifest.toml` is what makes a separate rerun under the
recorded environment possible.

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
    dir = run_dir(store, run_id)
    stored_manifest = _load_manifest(store, dir)
    stored_code = (stored_manifest.commit_sha, stored_manifest.dirty)
    fresh_code = code_provenance()
    # Both environments, on every path: a report that names two commits and
    # one environment cannot tell an unchanged rerun from one whose calendar
    # moved. `load_run` has already named an absent stored document; a
    # running environment with no manifest is unknown, not empty.
    stored_deps_toml = read(joinpath(dir, "Manifest.toml"), String)
    fresh_deps_toml = try
        dependency_manifest()
    catch e
        e isa MissingManifest ? "" : rethrow()
    end
    stored_deps = _deps_digest(stored_deps_toml)
    fresh_deps = _deps_digest(fresh_deps_toml)
    deps_changed = _deps_changes(stored_deps_toml, fresh_deps_toml)
    report(status, detail, ds = Divergence[]) =
        ReproductionReport(String(run_id), status, detail, stored_code, fresh_code,
                           stored_deps, fresh_deps, deps_changed, ds)

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
