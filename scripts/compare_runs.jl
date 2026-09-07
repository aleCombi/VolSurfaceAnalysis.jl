using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DuckDB
using DuckDB: DBInterface

# Compare two saved runs table by table and exit non-zero on any difference.
#
#   julia --project=. scripts/compare_runs.jl <store_root> <run_id_a> <run_id_b>
#
# Reads `<store_root>/runs/run_id=<id>/{manifest,metrics,positions,pnl_series}.parquet`
# straight through DuckDB, deliberately *without* `load_run`, so runs written
# under an older manifest schema stay comparable across the data-kinds
# migration (the reproduction gate between the step-0 baseline and every later
# run). Nothing from VolSurfaceAnalysis is loaded.
#
# What is compared (`run_id` and `written_at` are ignored everywhere):
# - positions.parquet   joined on `leg_idx`; strings / timestamps / integers
#                       exactly, doubles within TOL (NULL == NULL).
# - pnl_series.parquet  in canonical order (rank over `timestamp, pnl`), not
#                       on the stored `idx`: runs written before the
#                       canonical order landed (metrics.md) ordered samples
#                       at one timestamp by Dict iteration, which depended
#                       on the package build. `timestamp` exactly, `pnl`
#                       within TOL.
# - metrics.parquet     joined on `metric_name`; `value` within TOL, NaN == NaN,
#                       except `max_drawdown`, which is path-dependent: it is
#                       recomputed from each run's canonical series and those
#                       are compared (stored values are printed).
# - manifest.parquet    `n_opens`, `n_closes`, `n_unmarked` exactly,
#                       `window_end_spot` within TOL.
# A row present on one side only is a difference.

const TOL = 1e-9
const MAX_SHOWN = 10

function _usage()
    println(stderr,
        "usage: julia --project=. scripts/compare_runs.jl <store_root> <run_id_a> <run_id_b>")
    exit(2)
end

length(ARGS) == 3 || _usage()
store_root, id_a, id_b = ARGS

_run_dir(root, id) = joinpath(abspath(root), "runs", "run_id=" * id)
_pq(path) = "read_parquet('" * replace(path, "\\" => "/", "'" => "''") * "')"

const dir_a = _run_dir(store_root, id_a)
const dir_b = _run_dir(store_root, id_b)

for d in (dir_a, dir_b), f in ("manifest", "metrics", "positions", "pnl_series")
    p = joinpath(d, f * ".parquet")
    if !isfile(p)
        println(stderr, "compare_runs: missing $p")
        exit(1)
    end
end

const con = DuckDB.DB(":memory:")
_rows(sql) = collect(DBInterface.execute(con, sql))

# A result row as a plain NamedTuple, so a differing row prints on one line.
_compact(r) = NamedTuple{Tuple(propertynames(r))}(Tuple(getproperty(r, k) for k in propertynames(r)))

# SQL predicates for "column c differs between the two sides".
_exact_diff(c) = "(a.$c IS DISTINCT FROM b.$c)"
_approx_diff(c) = "(NOT coalesce((a.$c IS NULL AND b.$c IS NULL) OR " *
                  "(isnan(a.$c) AND isnan(b.$c)) OR abs(a.$c - b.$c) <= $TOL, false))"

# Full outer join on `key`, report every row where any listed column differs
# or the row is missing on one side. Returns the number of differing rows.
# `source(path)` turns a parquet path into the SELECT to compare.
_select_all(path) = "SELECT * FROM " * _pq(path)

function compare_table(name, key, exact, approx; source = _select_all)
    pa = "(" * source(joinpath(dir_a, name * ".parquet")) * ")"
    pb = "(" * source(joinpath(dir_b, name * ".parquet")) * ")"
    n_a = first(_rows("SELECT count(*) AS n FROM $pa AS a")).n
    n_b = first(_rows("SELECT count(*) AS n FROM $pb AS b")).n
    cols = vcat(exact, approx)
    preds = vcat(["a.$key IS NULL", "b.$key IS NULL"],
                 [_exact_diff(c) for c in exact],
                 [_approx_diff(c) for c in approx])
    sel = join(vcat(["coalesce(a.$key, b.$key) AS $key"],
                    ["a.$c AS $(c)_a" for c in cols],
                    ["b.$c AS $(c)_b" for c in cols]), ", ")
    sql = "SELECT $sel FROM $pa a FULL OUTER JOIN $pb b ON a.$key = b.$key " *
          "WHERE " * join(preds, " OR ") * " ORDER BY $key"
    diffs = _rows(sql)
    if isempty(diffs)
        println("$name: OK ($n_a rows)")
    else
        println("$name: $(length(diffs)) differing rows (a has $n_a, b has $n_b)")
        for r in Iterators.take(diffs, MAX_SHOWN)
            println("  ", _compact(r))
        end
        length(diffs) > MAX_SHOWN && println("  ... $(length(diffs) - MAX_SHOWN) more")
    end
    return length(diffs)
end

# The series in canonical order: rank over (timestamp, pnl).
_canonical_series(path) =
    "SELECT row_number() OVER (ORDER BY timestamp, pnl) AS rk, timestamp, pnl FROM " * _pq(path)

# metrics.parquet without the path-dependent metric.
_metrics_no_dd(path) = "SELECT * FROM " * _pq(path) * " WHERE metric_name <> 'max_drawdown'"

# max_drawdown as metrics/optional.jl defines it, over the canonical series.
function _drawdown(dir)
    rows = _rows("SELECT pnl FROM (" * _canonical_series(joinpath(dir, "pnl_series.parquet")) * ") ORDER BY rk")
    isempty(rows) && return 0.0
    eq = cumsum(Float64[r.pnl for r in rows])
    peak, max_dd = eq[1], 0.0
    for v in eq
        peak = max(peak, v)
        max_dd = max(max_dd, peak - v)
    end
    max_dd
end

function _stored_drawdown(dir)
    rows = _rows("SELECT value FROM " * _pq(joinpath(dir, "metrics.parquet")) * " WHERE metric_name = 'max_drawdown'")
    isempty(rows) ? missing : Float64(first(rows).value)
end

function compare_drawdown()
    da, db = _drawdown(dir_a), _drawdown(dir_b)
    sa, sb = _stored_drawdown(dir_a), _stored_drawdown(dir_b)
    ok = abs(da - db) <= TOL
    println("max_drawdown (recomputed on the canonical series): ", ok ? "OK" : "DIFFERS",
            " (a = $da, b = $db; stored a = $sa, b = $sb)")
    ok ? 0 : 1
end

function compare_manifest()
    fields = ["n_opens", "n_closes", "n_unmarked", "window_end_spot"]
    ra = _rows("SELECT $(join(fields, ", ")) FROM $(_pq(joinpath(dir_a, "manifest.parquet")))")
    rb = _rows("SELECT $(join(fields, ", ")) FROM $(_pq(joinpath(dir_b, "manifest.parquet")))")
    if length(ra) != 1 || length(rb) != 1
        println("manifest: expected 1 row each, got $(length(ra)) and $(length(rb))")
        return 1
    end
    a, b = first(ra), first(rb)
    bad = String[]
    for f in ("n_opens", "n_closes", "n_unmarked")
        getproperty(a, Symbol(f)) == getproperty(b, Symbol(f)) ||
            push!(bad, "$f: $(getproperty(a, Symbol(f))) vs $(getproperty(b, Symbol(f)))")
    end
    wa, wb = Float64(a.window_end_spot), Float64(b.window_end_spot)
    ((isnan(wa) && isnan(wb)) || abs(wa - wb) <= TOL) ||
        push!(bad, "window_end_spot: $wa vs $wb")
    if isempty(bad)
        println("manifest: OK")
    else
        println("manifest: $(length(bad)) differing fields")
        foreach(m -> println("  ", m), bad)
    end
    return length(bad)
end

println("comparing $id_a vs $id_b under $(abspath(store_root))")
n_bad = 0
n_bad += compare_table("positions", "leg_idx",
    ["underlying", "expiry", "option_type", "direction", "entry_timestamp"],
    ["strike", "quantity", "entry_price", "entry_spot", "entry_bid", "entry_ask"])
n_bad += compare_table("pnl_series", "rk", ["timestamp"], ["pnl"]; source = _canonical_series)
n_bad += compare_table("metrics", "metric_name", String[], ["value"]; source = _metrics_no_dd)
n_bad += compare_drawdown()
n_bad += compare_manifest()

DBInterface.close!(con)

if n_bad == 0
    println("runs match")
else
    println("runs differ")
    exit(1)
end
