# `persistence` module: the knowledge base.
#
# Every run that lands here writes its config and its outcome to disk under
# a content-addressed folder, so prior experiments stay queryable and
# comparable rather than evaporating into ad-hoc notebooks (see vision.md).
#
# Storage shape: Hive-partitioned parquet under `<root>/runs/run_id=<hash>/`.
# Each run folder holds the verbatim input TOML plus four parquet files
# (manifest, metrics, positions, pnl_series). Cross-run queries are just
# DuckDB SQL against the partitioned trees -- this module does not invent
# a query API. Same DuckDB-as-engine / parquet-as-storage pattern the
# `data` module uses for input data.
#
# Run identity is SHA-256 of the raw TOML bytes (truncated to 16 hex
# chars). Whitespace / comment / key-order changes therefore produce
# distinct ids today; canonicalization is deliberately deferred until
# spurious "new runs" actually become a nuisance.

using SHA
using DuckDB
using DuckDB: DBInterface

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
block. Mirrors `with_parquet_source` for resource-scoped use.
"""
function with_run_store(f::Function, root::AbstractString)
    s = RunStore(root)
    try
        return f(s)
    finally
        close(s)
    end
end

# --- run identity --------------------------------------------------------

const _RUN_ID_HEX_LEN = 16

"""
    run_id(config_toml::AbstractString) -> String

Content hash of the raw TOML bytes (SHA-256, truncated to 16 hex
characters). Pure function -- no I/O.
"""
run_id(config_toml::AbstractString)::String =
    bytes2hex(sha2_256(codeunits(String(config_toml))))[1:_RUN_ID_HEX_LEN]

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

_dt_sql(d::DateTime) = "TIMESTAMP '$(Dates.format(d, "yyyy-mm-dd HH:MM:SS"))'"
_str_sql(s::AbstractString) = "'" * replace(String(s), "'" => "''") * "'"
_otype_sql(t::OptionType) = t == Call ? "'C'" : "'P'"

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
             config_toml::AbstractString) -> String

Persist `result` plus its originating TOML config under
`<store.root>/runs/run_id=<hash>/`. Returns the 16-hex `run_id`.

Writes:
- `config.toml` -- the bytes passed in, verbatim.
- `manifest.parquet` -- one row of run-level metadata.
- `metrics.parquet` -- long form, one row per `(metric_name, value)`.
- `positions.parquet` -- one row per leg in `result.positions`.
- `pnl_series.parquet` -- one row per round trip in `result.pnl_series`.

If a folder for this `run_id` already exists, its contents are
overwritten. Same content hash means same config means same expected
result -- rewriting is idempotent in intent (the bytes may differ if
the run is non-deterministic, in which case the latest wins).

Atomicity is best-effort today: a crash mid-write can leave a
half-written folder. Re-running the same config recovers it. A
write-to-temp-then-rename pass is queued for the next iteration.
"""
function save_run(store::RunStore, result::ExperimentResult,
                  config_toml::AbstractString)::String
    _assert_open(store)
    id = run_id(config_toml)
    dir = run_dir(store, id)
    mkpath(dir)

    open(joinpath(dir, "config.toml"), "w") do io
        write(io, String(config_toml))
    end

    _write_manifest(store, dir, id, result)
    _write_metrics(store, dir, id, result)
    _write_positions(store, dir, id, result)
    _write_pnl_series(store, dir, id, result)

    return id
end

function _write_manifest(store::RunStore, dir::AbstractString, id::AbstractString,
                         result::ExperimentResult)
    exp = result.experiment
    s   = result.pnl_series
    schema = """(
        run_id VARCHAR,
        name VARCHAR,
        from_ts TIMESTAMP,
        to_ts TIMESTAMP,
        n_positions BIGINT,
        n_opens BIGINT,
        n_closes BIGINT,
        n_unmarked BIGINT,
        window_end_spot DOUBLE,
        written_at TIMESTAMP
    )"""
    insert = "INSERT INTO _writebuf VALUES (" * join([
        _str_sql(id),
        _str_sql(exp.name),
        _dt_sql(exp.from),
        _dt_sql(exp.to),
        string(length(result.positions)),
        string(s.n_opens),
        string(s.n_closes),
        string(s.n_unmarked),
        _f_sql(s.window_end_spot),
        _dt_sql(Dates.now(UTC)),
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

function _write_positions(store::RunStore, dir::AbstractString, id::AbstractString,
                          result::ExperimentResult)
    schema = """(
        run_id VARCHAR,
        leg_idx BIGINT,
        underlying VARCHAR,
        strike DOUBLE,
        expiry TIMESTAMP,
        option_type VARCHAR,
        direction INTEGER,
        quantity DOUBLE,
        entry_price DOUBLE,
        entry_spot DOUBLE,
        entry_bid DOUBLE,
        entry_ask DOUBLE,
        entry_timestamp TIMESTAMP
    )"""
    inserts = String[]
    for (i, p) in enumerate(result.positions)
        bid = ismissing(p.entry_bid) ? "NULL" : _f_sql(p.entry_bid)
        ask = ismissing(p.entry_ask) ? "NULL" : _f_sql(p.entry_ask)
        push!(inserts,
              "INSERT INTO _writebuf VALUES (" *
              join([
                  _str_sql(id),
                  string(i),
                  _str_sql(ticker(p.trade.underlying)),
                  _f_sql(p.trade.strike),
                  _dt_sql(p.trade.expiry),
                  _otype_sql(p.trade.option_type),
                  string(p.trade.direction),
                  _f_sql(p.trade.quantity),
                  _f_sql(p.entry_price),
                  _f_sql(p.entry_spot),
                  bid, ask,
                  _dt_sql(p.entry_timestamp),
              ], ", ") * ")")
    end
    _write_parquet(store, joinpath(dir, "positions.parquet"), schema, inserts)
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
`ExperimentResult`. Reads the four parquet artifacts plus the saved
`config.toml`, rebuilds the live `Experiment` via
[`load_experiment_str`](@ref), and reconstructs `positions`,
`pnl_series`, and the `metrics` NamedTuple (preserving the integer
types of `n_round_trips`, `n_opens`, `n_closes`).

The underlying data source declared by the saved config does not need
to be present on disk -- `ParquetDataSource` validates roots lazily, so
the rebuilt `Experiment.source` simply throws on first read if the
data has moved. Inspecting the persisted fields (`positions`,
`pnl_series`, `metrics`) needs no source data at all.

Throws `ArgumentError` if the run folder or any of the expected files
is missing.
"""
function load_run(store::RunStore, run_id::AbstractString)::ExperimentResult
    _assert_open(store)
    dir = run_dir(store, run_id)
    isdir(dir) || throw(ArgumentError("load_run: no run folder for id $run_id at $dir"))

    cfg_path = joinpath(dir, "config.toml")
    isfile(cfg_path) || throw(ArgumentError("load_run: missing config.toml in $dir"))
    config_toml = read(cfg_path, String)
    exp = load_experiment_str(config_toml)

    manifest = _load_manifest(store, dir)
    positions = _load_positions(store, dir, exp.source.chain_source.underlying)
    series = _load_pnl_series(store, dir, manifest)
    metrics = _load_metrics(store, dir, exp.metrics)

    return ExperimentResult(exp, positions, series, metrics)
end

function _select_rows(store::RunStore, path::AbstractString, sql::AbstractString)
    isfile(path) || throw(ArgumentError("load_run: missing $(basename(path)) at $path"))
    return collect(DBInterface.execute(store.con, sql))
end

function _load_manifest(store::RunStore, dir::AbstractString)
    path = joinpath(dir, "manifest.parquet")
    rows = _select_rows(store, path,
        "SELECT window_end_spot, n_opens, n_closes, n_unmarked FROM '$(_sql_pq_path(path))'")
    length(rows) == 1 ||
        throw(ArgumentError("load_run: manifest.parquet must have exactly 1 row, got $(length(rows))"))
    r = first(rows)
    return (window_end_spot=Float64(r.window_end_spot),
            n_opens=Int(r.n_opens),
            n_closes=Int(r.n_closes),
            n_unmarked=Int(r.n_unmarked))
end

function _load_positions(store::RunStore, dir::AbstractString,
                         underlying::Underlying)::Vector{Position}
    path = joinpath(dir, "positions.parquet")
    rows = _select_rows(store, path,
        "SELECT leg_idx, underlying, strike, expiry, option_type, direction, " *
        "quantity, entry_price, entry_spot, entry_bid, entry_ask, entry_timestamp " *
        "FROM '$(_sql_pq_path(path))' ORDER BY leg_idx")
    out = Position[]
    for r in rows
        # `underlying` from the parquet should match the source's; trust the
        # row but build a per-row Underlying so this stays correct if a future
        # multi-symbol persistence schema lands.
        u = String(r.underlying) == ticker(underlying) ? underlying :
            Underlying(String(r.underlying))
        otype = String(r.option_type) == "C" ? Call : Put
        trade = Trade(u, Float64(r.strike), DateTime(r.expiry), otype;
                      direction=Int(r.direction), quantity=Float64(r.quantity))
        bid = r.entry_bid === missing ? missing : Float64(r.entry_bid)
        ask = r.entry_ask === missing ? missing : Float64(r.entry_ask)
        push!(out, Position(trade,
                            Float64(r.entry_price),
                            Float64(r.entry_spot),
                            bid, ask,
                            DateTime(r.entry_timestamp)))
    end
    return out
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
