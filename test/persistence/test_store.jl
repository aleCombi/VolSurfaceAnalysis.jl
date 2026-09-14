# Tests for the RunStore persistence layer.
#
# A run's id is `full_hash(result.experiment)`, and `save_run` validates
# that the persisted config.toml rebuilds that same experiment. So these
# tests save *config-buildable* experiments (parquet specs, pure values,
# so no data tree is needed) paired with a hand-built ledger: the strangle
# of test/ledger/fixtures.jl cases 7 and 8 (ten events, two orders, cash
# 9240 cents), built through `record_order!`. The ledger exercises the
# serialization layer directly -- it need not come from a real backtest,
# and an in-memory source could not be hashed/saved anyway.

using DuckDB
using DuckDB: DBInterface

# A buildable parquet + noop config. Roots are nonexistent on purpose:
# provider specs are pure values, so the experiment builds and hashes
# without any data on disk; only open_data (a run) would throw.
function _smoke_config(; name="persist-smoke", metrics="[\"sharpe\", \"max_drawdown\"]")
    """
    name  = "$name"
    from  = 2024-01-15T15:30:00
    to    = 2024-01-15T15:32:00
    clock = { kind = "option_quote", underlying = "SPY" }

    [outputs]
    metrics = $metrics

    [data.option_bar]
    type = "parquet_option_bars"
    root = "/nonexistent/opts"

    [data.option_quote]
    type = "from_bars"
    synthesizer = { type = "ohlcv_spread", lambda = 0.7 }

    [data.spot_price]
    type = "parquet_spots"
    root = "/nonexistent/spot"

    [data.rate_curve]
    type = "constant"
    currency = "USD"
    value = 0.04

    [data.div_curve]
    type = "constant"
    underlying = "SPY"
    value = 0.015

    [data.vol_surface]
    type = "surface_from"
    currency = "USD"

    [agent]
    type = "static"

    [agent.policy]
    type = "noop"
    """
end

const _SMOKE_CONFIG = _smoke_config()

# A hand-built marked curve standing in for the one a run would produce:
# the smoke config's data roots do not exist, so nothing here could mark a
# book, and the curve is an input to the writer rather than something the
# writer derives.
_st_curve(profit::Vector{Float64}) = MarkedCurve(
    [DateTime(2024, 1, 16, 21, 0) + Day(i - 1) for i in 1:length(profit)],
    profit, DateTime[], Symbol[])

# Config-buildable experiment + the hand-built strangle ledger, opened and
# closed as two orders: one structure trade of 92.40 USD at the close.
function _build_smoke_result(config=_SMOKE_CONFIG; curve=_st_curve([0.0, 40.0, 92.40]))
    exp = load_experiment_str(config)
    L, _ = _lg_case_strangle_closed()
    ExperimentResult(exp, L, curve, compute_metrics(L, curve, exp.outputs.metrics))
end

# Config-buildable experiment with an empty ledger (folder / identity tests).
function _empty_result(config)
    exp = load_experiment_str(config)
    L = Ledger()
    curve = _st_curve(Float64[])
    ExperimentResult(exp, L, curve, compute_metrics(L, curve, exp.outputs.metrics))
end

_st_pq(path) = replace(path, "\\" => "/")
_st_rows(store, path, sql="SELECT * FROM '$(_st_pq(path))'") = collect(DBInterface.execute(store.con, sql))

# Two order records equal field by field (a record holds vectors, so the
# default `==` would compare them by identity).
function _st_same_record(a::OrderRecord, b::OrderRecord)
    a.order_id == b.order_id && a.first_leg_id == b.first_leg_id && a.group == b.group &&
    a.decided_at == b.decided_at && a.known_to == b.known_to &&
    a.order.label == b.order.label && a.order.group == b.order.group &&
    a.order.operation == b.order.operation && a.order.legs == b.order.legs &&
    length(a.observations) == length(b.observations) &&
    all(isequal(x.quote_at, y.quote_at) && isequal(x.bid, y.bid) && isequal(x.ask, y.ask) &&
        isequal(x.spot, y.spot) && isequal(x.spot_at, y.spot_at)
        for (x, y) in zip(a.observations, b.observations))
end

_st_counters(L::Ledger) = (L.next_id, L.next_sequence, L.next_group, L.next_execution,
                           L.next_order_id, L.next_leg_id)

@testset "save id is the experiment full_hash (16 hex chars)" begin
    res = _build_smoke_result()
    id = full_hash(res.experiment)
    @test length(id) == 16
    @test all(c -> c in "0123456789abcdef", id)
end

@testset "code_provenance: returns (sha::String, dirty::Bool)" begin
    sha, dirty = code_provenance()
    @test sha isa String
    @test dirty isa Bool
end

@testset "RunStore: construct creates root, close is idempotent" begin
    mktempdir() do tmp
        root = joinpath(tmp, "kb")
        store = RunStore(root)
        @test isdir(root)
        @test isopen(store)
        close(store)
        @test !isopen(store)
        close(store)  # safe to call again
    end
end

@testset "save_run: writes config.toml + 5 parquet files under runs/run_id=<hash>/" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            @test id == full_hash(res.experiment)

            dir = run_dir(store, id)
            @test isdir(dir)
            @test isfile(joinpath(dir, "config.toml"))
            for f in ("manifest", "metrics", "events", "orders", "order_legs")
                @test isfile(joinpath(dir, f * ".parquet"))
            end
            @test !isfile(joinpath(dir, "positions.parquet"))
            # The derived series left with the type it exported; the tables
            # that replace it belong to the second half of the outputs round.
            @test !isfile(joinpath(dir, "pnl_series.parquet"))

            @test read(joinpath(dir, "config.toml"), String) == _SMOKE_CONFIG
        end
        GC.gc()
    end
end

@testset "save_run: rejects a config that does not describe the result" begin
    mktempdir() do tmp
        res = _build_smoke_result()                        # built from _SMOKE_CONFIG
        mismatched = _smoke_config(metrics="[\"sortino\"]")  # different outputs -> different full_hash
        with_run_store(joinpath(tmp, "kb")) do store
            @test_throws ArgumentError save_run(store, res, mismatched)
        end
        GC.gc()
    end
end

@testset "save_run: rejects a config with a different name label" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        mismatched = _smoke_config(name="other-label")       # same full_hash, different label
        with_run_store(joinpath(tmp, "kb")) do store
            @test_throws ArgumentError save_run(store, res, mismatched)
        end
        GC.gc()
    end
end

@testset "save_run: refuses a broken join before writing a run folder" begin
    mktempdir() do tmp
        with_run_store(joinpath(tmp, "kb")) do store
            for observations in ([_lg_seen(0.86), _lg_seen(1.10)], LegObservation[])
                res = _build_smoke_result()
                r = first(res.ledger.orders)
                res.ledger.orders[1] = OrderRecord(r.order_id, r.first_leg_id, r.group,
                    r.decided_at, r.known_to, r.order, observations)
                err = try save_run(store, res, _SMOKE_CONFIG); nothing catch e; e end
                expected = isempty(observations) ? :observations : :price
                @test err isa JoinViolation && err.field == expected
                @test occursin("JoinViolation", sprint(showerror, err))
                @test !isdir(run_dir(store, full_hash(res.experiment)))
            end
        end
        GC.gc()
    end
end

@testset "save_run: manifest row carries name/window/counts and the schema version" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            rows = _st_rows(store, joinpath(run_dir(store, id), "manifest.parquet"))
            @test length(rows) == 1
            r = first(rows)
            @test r.run_id == id
            @test r.name == "persist-smoke"
            @test r.n_events == 10
            @test r.n_orders == 2
            @test r.n_opens == n_opens(res.ledger) == 2
            @test r.n_closes == n_closes(res.ledger) == 2
            @test r.n_marked == n_marked(res.curve) == 3
            @test r.n_unmarked == n_unmarked(res.curve) == 0
            @test r.schema_version == 5
            @test !(:n_positions in propertynames(r))
            @test !(:window_end_spot in propertynames(r))
        end
        GC.gc()
    end
end

@testset "save_run: manifest records core_hash + commit_sha + dirty" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG; commit_sha="abc123def456", dirty=true)
            path = joinpath(run_dir(store, id), "manifest.parquet")
            r = first(_st_rows(store, path, "SELECT core_hash, commit_sha, dirty FROM '$(_st_pq(path))'"))
            @test r.core_hash == core_hash(res.experiment)
            @test r.commit_sha == "abc123def456"
            @test r.dirty == true
        end
        GC.gc()
    end
end

@testset "save_run: metrics.parquet has one row per (name, value)" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            path = joinpath(run_dir(store, id), "metrics.parquet")
            rows = _st_rows(store, path, "SELECT metric_name, value FROM '$(_st_pq(path))'")
            names = Set(r.metric_name for r in rows)
            # Always-on core metrics plus the two requested optionals.
            @test "total_pnl"     in names
            @test "n_round_trips" in names
            @test "hit_rate"      in names
            @test "sharpe"        in names
            @test "max_drawdown"  in names
            total_pnl_row = first(r for r in rows if r.metric_name == "total_pnl")
            @test total_pnl_row.value ≈ 92.40                # 9240 cents, one structure sample
        end
        GC.gc()
    end
end

@testset "save_run: events.parquet has one row per event with the kind's own columns" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            path = joinpath(run_dir(store, id), "events.parquet")
            rows = _st_rows(store, path, "SELECT * FROM '$(_st_pq(path))' ORDER BY sequence")
            @test length(rows) == 10
            @test [r.kind for r in rows] == ["fill", "fill", "fee", "fee", "fill", "match", "fill", "match", "fee", "fee"]
            @test [r.sequence for r in rows] == 1:10
            @test [r.id for r in rows] == 1:10
            @test all(r.run_id == id for r in rows)
            f1 = rows[1]
            @test f1.group_id == 1 && f1.order_leg_id == 1 && f1.execution_id == 1
            @test f1.underlying == "SPY" && f1.strike == 470.0 && f1.option_type == "P"
            @test f1.side == "short" && f1.intent == "open" && f1.quantity == 1 && f1.price == 0.85
            @test f1.fill_rule == "cross_spread"
            @test f1.expiry == _LG_EXPIRY_A
            @test f1.effective_at == _LG_T_OPEN && f1.recorded_at == _LG_T_OPEN
            @test f1.open_fill_id === missing && f1.source_id === missing && f1.amount === missing
            fee = rows[3]
            @test fee.source_id == 1 && fee.amount == -65 && fee.group_id === missing
            m = rows[6]
            @test m.open_fill_id == 1 && m.close_fill_id == 5 && m.quantity == 1 && m.group_id == 1
            @test m.price === missing && m.underlying === missing
            c = rows[5]
            @test c.side == "long" && c.intent == "close" && c.price == 0.40 && c.execution_id == 3
        end
        GC.gc()
    end
end

@testset "save_run: orders.parquet and order_legs.parquet hold the order journal" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            opath = joinpath(run_dir(store, id), "orders.parquet")
            orders = _st_rows(store, opath, "SELECT * FROM '$(_st_pq(opath))' ORDER BY order_id")
            @test length(orders) == 2
            o1, o2 = orders
            @test o1.order_id == 1 && o1.first_leg_id == 1 && o1.label == "strangle" && o1.group_id == 1
            @test o1.operation === missing && o1.decided_at == _LG_T_OPEN && o1.known_to == 0
            @test o2.order_id == 2 && o2.first_leg_id == 3 && o2.label == "close" && o2.group_id == 1
            @test o2.decided_at == _LG_T_CLOSE && o2.known_to == 4
            lpath = joinpath(run_dir(store, id), "order_legs.parquet")
            legs = _st_rows(store, lpath, "SELECT * FROM '$(_st_pq(lpath))' ORDER BY order_leg_id")
            @test length(legs) == 4
            @test [l.order_leg_id for l in legs] == 1:4
            @test [l.order_id for l in legs] == [1, 1, 2, 2]
            @test [l.leg_idx for l in legs] == [1, 2, 1, 2]
            l1 = legs[1]
            @test l1.underlying == "SPY" && l1.strike == 470.0 && l1.option_type == "P" && l1.expiry == _LG_EXPIRY_A
            @test l1.side == "short" && l1.intent == "open" && l1.quantity == 1
            @test l1.quote_at == _LG_T_OPEN && l1.bid == 0.85 && l1.ask == 0.85 && l1.spot == 480.0 && l1.spot_at == _LG_T_OPEN
            l4 = legs[4]
            @test l4.side == "long" && l4.intent == "close" && l4.strike == 490.0 && l4.option_type == "C"
            @test l4.bid == 0.60 && l4.quote_at == _LG_T_CLOSE
        end
        GC.gc()
    end
end

@testset "save_run: manifest carries NULL counts when the run has no curve" begin
    mktempdir() do tmp
        exp = load_experiment_str(_SMOKE_CONFIG)
        L, _ = _lg_case_strangle_closed()
        res = ExperimentResult(exp, L, nothing, compute_metrics(L, nothing, exp.outputs.metrics))
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            r = first(_st_rows(store, joinpath(run_dir(store, id), "manifest.parquet")))
            # NULL, not 0: "no curve at all" and "a curve that marked nothing"
            # are different facts and must not read the same in SQL.
            @test r.n_marked === missing
            @test r.n_unmarked === missing
        end
        GC.gc()
    end
end

@testset "save_run: idempotent re-save of same experiment overwrites in place" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id1 = save_run(store, res, _SMOKE_CONFIG)
            id2 = save_run(store, res, _SMOKE_CONFIG)
            @test id1 == id2
            runs_root = joinpath(store.root, "runs")
            @test length(readdir(runs_root)) == 1
        end
        GC.gc()
    end
end

@testset "save_run: experiments differing only in outputs -> 2 folders, shared core_hash" begin
    mktempdir() do tmp
        c1 = _smoke_config(metrics="[\"sharpe\"]")
        c2 = _smoke_config(metrics="[\"sharpe\", \"sortino\"]")
        r1 = _empty_result(c1)
        r2 = _empty_result(c2)
        with_run_store(joinpath(tmp, "kb")) do store
            id1 = save_run(store, r1, c1)
            id2 = save_run(store, r2, c2)
            @test id1 != id2                                       # outputs change full_hash
            @test core_hash(r1.experiment) == core_hash(r2.experiment)  # ... but not the backtest
            @test isdir(run_dir(store, id1))
            @test isdir(run_dir(store, id2))
        end
        GC.gc()
    end
end

@testset "save_run: an empty ledger -> empty, typed ledger tables" begin
    mktempdir() do tmp
        res = _empty_result(_SMOKE_CONFIG)
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            for f in ("events", "orders", "order_legs")
                path = joinpath(run_dir(store, id), f * ".parquet")
                @test isfile(path)
                @test isempty(_st_rows(store, path))
            end
            r = first(_st_rows(store, joinpath(run_dir(store, id), "manifest.parquet")))
            @test r.n_events == 0 && r.n_orders == 0
        end
        GC.gc()
    end
end

@testset "cross-run query: SELECT across runs/*/manifest.parquet" begin
    mktempdir() do tmp
        c1 = _smoke_config(metrics="[\"sharpe\"]")
        c2 = _smoke_config(metrics="[\"sortino\"]")
        with_run_store(joinpath(tmp, "kb")) do store
            save_run(store, _empty_result(c1), c1)
            save_run(store, _empty_result(c2), c2)
            glob = replace(joinpath(store.root, "runs", "*", "manifest.parquet"),
                           "\\" => "/")
            rows = collect(DBInterface.execute(store.con,
                "SELECT run_id, name, core_hash FROM '$glob' ORDER BY run_id"))
            @test length(rows) == 2
            @test all(r -> r.name == "persist-smoke", rows)
            @test rows[1].core_hash == rows[2].core_hash   # same backtest, different outputs
        end
        GC.gc()
    end
end

@testset "save_run on closed store throws" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        store = RunStore(joinpath(tmp, "kb"))
        close(store)
        @test_throws ArgumentError save_run(store, res, _SMOKE_CONFIG)
        GC.gc()
    end
end

# ---- load_run: round-trip + edge cases ---------------------------------

@testset "load_run: round-trip ExperimentResult (ledger, orders, pnl, metrics)" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            loaded = load_run(store, id)

            @test loaded isa ExperimentResult
            L, S = loaded.ledger, res.ledger

            # the events equal the saved ones
            @test length(L) == length(S) == 10
            for (e, s) in zip(L.events, S.events)
                @test typeof(e) === typeof(s)
                @test event_id(e) == event_id(s)
                @test sequence(e) == sequence(s)
                @test effective_at(e) == effective_at(s)
                @test recorded_at(e) == recorded_at(s)
                @test cash(e) == cash(s)
                @test group(e) == group(s)
            end
            @test [e.price for e in L.events if e isa Fill] == [0.85, 1.10, 0.40, 0.60]
            @test [e.execution_id for e in L.events if e isa Fill] == 1:4
            @test [e.fill_rule for e in L.events if e isa Fill] == fill(:cross_spread, 4)

            # the order records equal the saved ones field by field
            @test length(L.orders) == length(S.orders) == 2
            @test all(_st_same_record(a, b) for (a, b) in zip(L.orders, S.orders))
            @test L.orders[1].order.group === nothing            # minted: as the policy emitted it
            @test L.orders[2].order.group == 1                   # named

            # the counters equal the saved ledger's
            @test _st_counters(L) == _st_counters(S) == (11, 11, 2, 5, 3, 5)

            # the book at the last sequence equals the saved book
            @test L.book == S.book
            @test L.book == book_as_known(L, last_sequence(L))
            @test L.book.cash == 9240
            @test check_join(L) === nothing

            # This config's roots do not exist, so the curve cannot be
            # recomputed and degrades by the declared boundary.
            @test loaded.curve === nothing
            @test keys(loaded.metrics) ==
                  (:total_pnl, :n_round_trips, :n_opens, :n_closes, :hit_rate)
            @test loaded.metrics.n_round_trips isa Int
            @test loaded.metrics.n_opens isa Int
            @test loaded.metrics.total_pnl ≈ res.metrics.total_pnl ≈ 92.40
            @test loaded.metrics.n_opens == 2 && loaded.metrics.n_closes == 2
        end
        GC.gc()
    end
end

@testset "load_run: the loaded ledger replays and derives like the saved one" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            L = load_run(store, id).ledger
            @test [r.pnl for r in round_trips(L)] == [4370, 4870]
            @test trade_pnl(L) ≈ [92.40]
            @test book_effective(L, _LG_FAR) == book_effective(res.ledger, _LG_FAR)
            @test book_as_known(L, 4) == book_as_known(res.ledger, 4)
            @test book_as_known(L, 4).cash == 19370
            @test order_leg(L, 3)[1].order_id == 2
        end
        GC.gc()
    end
end

@testset "load_run: rebuilds live Experiment via load_experiment_str" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            loaded = load_run(store, id)
            @test loaded.experiment isa Experiment
            @test loaded.experiment.name == "persist-smoke"
            @test loaded.experiment.outputs.metrics == [:sharpe, :max_drawdown]
            @test loaded.experiment.agent isa StaticAgent
            @test loaded.experiment.data isa MarketData
            @test loaded.experiment.clock == Clock{OptionQuote}(Underlying("SPY"))
        end
        GC.gc()
    end
end

@testset "load_run: works when the data is absent (specs are pure values)" begin
    mktempdir() do tmp
        # _SMOKE_CONFIG points at nonexistent roots: the specs rebuild and
        # the persisted fields load, but opening the data throws.
        res = _build_smoke_result()
        store_root = joinpath(tmp, "kb")
        id = with_run_store(store_root) do store
            save_run(store, res, _SMOKE_CONFIG)
        end
        with_run_store(store_root) do store
            loaded = load_run(store, id)
            @test length(loaded.ledger) == length(res.ledger)
            @test loaded.metrics.total_pnl ≈ res.metrics.total_pnl
            @test_throws ArgumentError open_data(loaded.experiment.data)
            @test_throws ArgumentError run_experiment(loaded.experiment)
        end
        GC.gc()
    end
end

@testset "load_run: a run whose order journal lost a leg fails the join by name" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            path = _st_pq(joinpath(run_dir(store, id), "order_legs.parquet"))
            @test load_run(store, id) isa ExperimentResult
            # the last leg of the second order is gone: the fill on leg 4 joins nothing
            DBInterface.execute(store.con, "CREATE OR REPLACE TABLE ol AS SELECT * FROM '$path' WHERE order_leg_id <> 4")
            DBInterface.execute(store.con, "COPY ol TO '$path' (FORMAT PARQUET)")
            err = try load_run(store, id); nothing catch e; e end
            @test err isa DanglingReference && err.field == :order_leg_id && err.id == 4
            @test occursin("DanglingReference", sprint(showerror, err))
            # the second leg of the first order is gone: the leg ids are no longer contiguous
            save_run(store, res, _SMOKE_CONFIG)                    # restore
            DBInterface.execute(store.con, "CREATE OR REPLACE TABLE ol AS SELECT * FROM '$path' WHERE order_leg_id <> 2")
            DBInterface.execute(store.con, "COPY ol TO '$path' (FORMAT PARQUET)")
            err = try load_run(store, id); nothing catch e; e end
            @test err isa JoinViolation && err.field == :first_leg_id
            @test occursin("JoinViolation", sprint(showerror, err))
            # a stored fill whose price is not what the rule gives from its observation
            save_run(store, res, _SMOKE_CONFIG)                    # restore
            epath = _st_pq(joinpath(run_dir(store, id), "events.parquet"))
            DBInterface.execute(store.con, "CREATE OR REPLACE TABLE ev AS SELECT * REPLACE (CASE WHEN id = 1 THEN 0.84 ELSE price END AS price) FROM '$epath'")
            DBInterface.execute(store.con, "COPY ev TO '$epath' (FORMAT PARQUET)")
            err = try load_run(store, id); nothing catch e; e end
            @test err isa JoinViolation && err.field == :price && err.id == 1
        end
        GC.gc()
    end
end

@testset "manifest schema_version: written as 5, and load_run refuses other versions" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            path = replace(joinpath(run_dir(store, id), "manifest.parquet"), "\\" => "/")
            r = first(collect(DBInterface.execute(store.con, "SELECT schema_version FROM '$path'")))
            @test r.schema_version == VolSurfaceAnalysis.RUN_SCHEMA_VERSION == 5
            @test load_run(store, id) isa ExperimentResult

            # a manifest written before the column existed
            DBInterface.execute(store.con, "CREATE OR REPLACE TABLE m AS SELECT * EXCLUDE (schema_version) FROM '$path'")
            DBInterface.execute(store.con, "COPY m TO '$path' (FORMAT PARQUET)")
            err = try load_run(store, id); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("schema_version 0", err.msg) && occursin("rerun the config", err.msg)

            # an explicit older version: the positions-era store
            DBInterface.execute(store.con, "CREATE OR REPLACE TABLE m AS SELECT *, 2::INTEGER AS schema_version FROM '$path'")
            DBInterface.execute(store.con, "COPY m TO '$path' (FORMAT PARQUET)")
            err = try load_run(store, id); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("schema_version 2", err.msg)

            # and the version before this one. A schema-3 run's id was
            # computed without the venue or the contract facts, and the tree
            # holds runs made before the lifecycle booked expiries at all --
            # an id that matches one of those says nothing about whether the
            # same code produced it, so the version is what refuses them.
            DBInterface.execute(store.con, "CREATE OR REPLACE TABLE m AS SELECT * EXCLUDE (schema_version), 3::INTEGER AS schema_version FROM '$path'")
            DBInterface.execute(store.con, "COPY m TO '$path' (FORMAT PARQUET)")
            err = try load_run(store, id); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("schema_version 3", err.msg) && occursin("rerun the config", err.msg)
        end
        GC.gc()
    end
end

@testset "load_run: unknown id throws clearly" begin
    mktempdir() do tmp
        with_run_store(joinpath(tmp, "kb")) do store
            @test_throws ArgumentError load_run(store, "deadbeef00000000")
        end
    end
end

@testset "load_run: missing config.toml inside an existing run dir throws" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            rm(joinpath(run_dir(store, id), "config.toml"))
            @test_throws ArgumentError load_run(store, id)
        end
        GC.gc()
    end
end

@testset "load_run: an empty result round-trips as an empty ledger" begin
    mktempdir() do tmp
        res = _empty_result(_SMOKE_CONFIG)
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            loaded = load_run(store, id)
            @test isempty(loaded.ledger)
            @test isempty(loaded.ledger.orders)
            @test _st_counters(loaded.ledger) == (1, 1, 1, 1, 1, 1)
            @test isempty(trade_pnl(loaded.ledger))
            @test loaded.metrics.n_round_trips == 0
        end
        GC.gc()
    end
end

# ---- load_run recomputes: nothing derived is read back ------------------

# A one-session parquet tree: SPY spot prints at the open and the close of
# Fri 2024-01-19, plus one option bar so the bar root is a real tree. The
# strangle ledger is flat from 2024-01-18, so the book needs no mark at
# that session's close and the curve is the realised total.
#
# Rows are written in VENDOR time and read back in visibility time, one
# minute later (`bar_visible_at`). So the 14:29 row is the 09:30 ET open and
# the 20:59 row is the 16:00 ET close: written a minute earlier than the
# instant they stand for. Writing 21:00 here would put the close outside the
# 09:30-16:00 window, which is the whole point of the convention.
function _st_session_tree(root::AbstractString)
    options_root = joinpath(root, "options_1min")
    spot_root    = joinpath(root, "spots_1min")
    odir = joinpath(options_root, "date=2024-01-19", "symbol=SPY")
    sdir = joinpath(spot_root,    "date=2024-01-19", "symbol=SPY")
    mkpath(odir); mkpath(sdir)
    db = DuckDB.DB(":memory:")
    try
        opath = replace(joinpath(odir, "data.parquet"), "\\" => "/")
        DBInterface.execute(db, """
            COPY (SELECT
                'O:SPY240216C00480000'          AS ticker,
                5.05::DOUBLE                    AS close,
                TIMESTAMP '2024-01-19 15:30:00' AS timestamp,
                100.0::DOUBLE                   AS volume,
                'SPY'                           AS parsed_underlying,
                DATE '2024-02-16'               AS parsed_expiry,
                480.0::DOUBLE                   AS parsed_strike,
                'C'                             AS parsed_option_type
            ) TO '$opath' (FORMAT PARQUET);
        """)
        spath = replace(joinpath(sdir, "data.parquet"), "\\" => "/")
        DBInterface.execute(db, """
            COPY (SELECT * FROM (VALUES
                (TIMESTAMP '2024-01-19 14:29:00', 480.0::DOUBLE),
                (TIMESTAMP '2024-01-19 20:59:00', 483.0::DOUBLE)
            ) AS t(timestamp, close)) TO '$spath' (FORMAT PARQUET);
        """)
    finally
        DBInterface.close!(db)
    end
    return (options_root=options_root, spot_root=spot_root)
end

_st_session_config(tree) = """
name  = "session-smoke"
from  = 2024-01-19T00:00:00
to    = 2024-01-19T23:59:00
clock = { kind = "option_quote", underlying = "SPY" }

[outputs]
metrics = ["max_drawdown", "profit_factor"]

[data.option_bar]
type = "parquet_option_bars"
root = "$(replace(tree.options_root, "\\" => "/"))"

[data.option_quote]
type = "from_bars"
synthesizer = { type = "ohlcv_spread", lambda = 0.7 }

[data.spot_price]
type = "parquet_spots"
root = "$(replace(tree.spot_root, "\\" => "/"))"

[data.rate_curve]
type = "constant"
currency = "USD"
value = 0.04

[data.div_curve]
type = "constant"
underlying = "SPY"
value = 0.015

[data.vol_surface]
type = "surface_from"
currency = "USD"

[agent]
type = "static"

[agent.policy]
type = "noop"
"""

@testset "load_run: with data present, the curve and every metric are recomputed" begin
    mktempdir() do tmp
        tree = _st_session_tree(tmp)
        cfg  = _st_session_config(tree)
        exp  = load_experiment_str(cfg)
        L, _ = _lg_case_strangle_closed()
        # Save a result carrying a deliberately wrong curve: if the load path
        # read anything derived back, this is the number that would return.
        wrong = _st_curve([-999.0, -999.0])
        res = ExperimentResult(exp, L, wrong, compute_metrics(L, wrong, exp.outputs.metrics))
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, cfg)
            # Remove the derived export outright: the load path must not need it.
            rm(joinpath(run_dir(store, id), "metrics.parquet"))
            loaded = load_run(store, id)

            @test loaded.curve isa MarkedCurve
            @test loaded.curve.timestamps == [DateTime(2024, 1, 19, 21, 0)]
            @test n_unmarked(loaded.curve) == 0
            @test loaded.curve.profit ≈ [92.40]          # flat book: the realised total
            @test loaded.curve != wrong
            @test loaded.metrics.max_drawdown == 0.0     # not the saved -999 curve's
            @test loaded.metrics.total_pnl ≈ 92.40
            @test loaded.metrics.profit_factor ≈ profit_factor(trade_pnl(L), nothing)
            @test haskey(loaded.metrics, :max_drawdown)  # the curve is available
        end
        GC.gc()
    end
end

# The degradation boundary is one line wide: opening the data. A failure
# *inside* the curve build is a real defect in the run or the tree, and
# turning it into "market data unavailable" would hand back an apparently
# valid partial result under a warning naming the wrong cause -- the exact
# masking design rule 7 forbids, and the defect this whole round removes.
@testset "load_run: a failure inside the curve build propagates, never degrades" begin
    mktempdir() do tmp
        tree = _st_session_tree(tmp)
        cfg  = _st_session_config(tree)
        exp  = load_experiment_str(cfg)
        L, _ = _lg_case_strangle_closed()
        good = _st_curve([92.40])
        res  = ExperimentResult(exp, L, good, compute_metrics(L, good, exp.outputs.metrics))
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, cfg)
            # The tree still opens; it now disagrees with itself at one instant,
            # the real defect found in the SPY tree while this round was built.
            sdir = joinpath(tree.spot_root, "date=2024-01-19", "symbol=SPY")
            db = DuckDB.DB(":memory:")
            try
                spath = replace(joinpath(sdir, "data.parquet"), "\\" => "/")
                DBInterface.execute(db, """
                    COPY (SELECT * FROM (VALUES
                        (TIMESTAMP '2024-01-19 14:29:00', 480.0::DOUBLE),
                        (TIMESTAMP '2024-01-19 20:59:00', 483.0::DOUBLE),
                        (TIMESTAMP '2024-01-19 20:59:00', 483.5::DOUBLE)
                    ) AS t(timestamp, close)) TO '$spath' (FORMAT PARQUET);
                """)
            finally
                DBInterface.close!(db)
            end
            err = try; load_run(store, id); nothing; catch e; e; end
            @test err isa ConflictingRecords
            @test err !== nothing
            println("  propagated, did not degrade: ", typeof(err))
        end
        GC.gc()
    end
end
