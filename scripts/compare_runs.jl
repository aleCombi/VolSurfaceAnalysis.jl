using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DuckDB
using DuckDB: DBInterface

# Compare two saved runs table by table and exit non-zero on any difference.
#
#   julia --project=. scripts/compare_runs.jl <store_root> <run_id_a> <run_id_b>
#
# Reads `<store_root>/runs/run_id=<id>/{manifest,metrics,events,orders,order_legs}.parquet`
# straight through DuckDB, deliberately *without* `load_run`, so two runs are
# compared as written (the reproduction gate between a baseline and every
# later run). Nothing from VolSurfaceAnalysis is loaded. Runs written under
# manifest schema version 2 or earlier (`positions.parquet`) are no longer
# comparable: the ledger replaced positions in slice 2 of the ledger rebuild.
#
# What is compared (`run_id` and `written_at` are ignored everywhere):
# - events.parquet      joined on `sequence`; exact on every column but
#                       `strike`, `price` and `settlement_price`, which are
#                       within TOL (NULL == NULL).
# - orders.parquet      joined on `order_id`; every column exactly.
# - order_legs.parquet  joined on `order_leg_id`; doubles within TOL, the
#                       rest exactly (NULL == NULL).
# - metrics.parquet     joined on `metric_name`; `value` within TOL, NaN == NaN.
#                       Every metric is now a deterministic function of the
#                       ledger and the marked curve, so `max_drawdown` needs
#                       no special case; the curve itself is not exported
#                       under schema 5 and so is not compared here.
# - manifest.parquet    `n_events`, `n_orders`, `n_opens`, `n_closes`,
#                       `n_marked`, `n_unmarked`, `n_metrics`,
#                       `n_mark_failures` and `n_settlement_failures`
#                       exactly (NULL == NULL).
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

for d in (dir_a, dir_b), f in ("manifest", "metrics", "events", "orders", "order_legs")
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

function compare_manifest()
    fields = ["n_events", "n_orders", "n_opens", "n_closes", "n_marked", "n_unmarked",
              "n_metrics", "n_mark_failures", "n_settlement_failures"]
    ra = _rows("SELECT $(join(fields, ", ")) FROM $(_pq(joinpath(dir_a, "manifest.parquet")))")
    rb = _rows("SELECT $(join(fields, ", ")) FROM $(_pq(joinpath(dir_b, "manifest.parquet")))")
    if length(ra) != 1 || length(rb) != 1
        println("manifest: expected 1 row each, got $(length(ra)) and $(length(rb))")
        return 1
    end
    a, b = first(ra), first(rb)
    bad = String[]
    for f in fields
        getproperty(a, Symbol(f)) === getproperty(b, Symbol(f)) ||
            push!(bad, "$f: $(getproperty(a, Symbol(f))) vs $(getproperty(b, Symbol(f)))")
    end
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
n_bad += compare_table("events", "sequence",
    ["id", "kind", "effective_at", "recorded_at", "group_id", "order_leg_id", "execution_id",
     "underlying", "expiry", "option_type", "side", "intent", "quantity", "fill_rule",
     "open_fill_id", "close_fill_id", "outcome", "source_id", "amount"],
    ["strike", "price", "settlement_price"])
n_bad += compare_table("orders", "order_id",
    ["first_leg_id", "label", "group_id", "operation", "decided_at", "known_to"], String[])
n_bad += compare_table("order_legs", "order_leg_id",
    ["order_id", "leg_idx", "underlying", "expiry", "option_type", "side", "intent", "quantity",
     "quote_at", "spot_at"],
    ["strike", "bid", "ask", "spot"])
n_bad += compare_table("metrics", "metric_name", String[], ["value"])
n_bad += compare_manifest()

DBInterface.close!(con)

if n_bad == 0
    println("runs match")
else
    println("runs differ")
    exit(1)
end
