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

# A curve with a broken session, and the failure record that must agree
# with it: the store checks the two against each other on load.
_st_broken_curve(t::DateTime, reason::Symbol=:no_mark) = MarkedCurve(
    [DateTime(2024, 1, 16, 21, 0)], [0.0], [t], [reason])
_st_mark_failure(t::DateTime, subject="SPY 2024-01-19T21:00:00 480.0P lot@1",
                 reason::Symbol=:no_mark) = RunFailure(t, :mark, subject, reason)

# Config-buildable experiment + the hand-built strangle ledger, opened and
# closed as two orders: one structure trade of 92.40 USD at the close.
function _build_smoke_result(config=_SMOKE_CONFIG; curve=_st_curve([0.0, 40.0, 92.40]),
                             failures=RunFailure[])
    exp = load_experiment_str(config)
    L, _ = _lg_case_strangle_closed()
    ExperimentResult(exp, L, curve, compute_metrics(L, curve, exp.outputs.metrics),
                     canonical_failures(failures))
end

# Config-buildable experiment with an empty ledger (folder / identity tests).
function _empty_result(config)
    exp = load_experiment_str(config)
    L = Ledger()
    curve = _st_curve(Float64[])
    ExperimentResult(exp, L, curve, compute_metrics(L, curve, exp.outputs.metrics),
                     RunFailure[])
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

@testset "save_run: writes both input documents and every output table" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            @test id == full_hash(res.experiment)

            dir = run_dir(store, id)
            @test isdir(dir)
            # Two input documents: what to run, and what it ran against.
            @test isfile(joinpath(dir, "config.toml"))
            @test isfile(joinpath(dir, "Manifest.toml"))
            # ... and every output the run produced.
            for f in ("manifest", "metrics", "events", "orders", "order_legs",
                      "curve", "failures")
                @test isfile(joinpath(dir, f * ".parquet"))
            end
            @test !isfile(joinpath(dir, "positions.parquet"))
            @test !isfile(joinpath(dir, "pnl_series.parquet"))
            # Deliberately not written: the ledger already records the trade
            # facts and the stored metrics witness their reported reductions,
            # so a round-trip table has no consumer to justify it yet.
            @test !isfile(joinpath(dir, "round_trips.parquet"))

            # Both input documents are the bytes handed in, verbatim.
            @test read(joinpath(dir, "config.toml"), String) == _SMOKE_CONFIG
            @test read(joinpath(dir, "Manifest.toml"), String) == dependency_manifest()
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
            # Every output table has membership evidence: the ones the
            # ledger and the curve do not cover are what keep a truncated
            # metrics or failures table different from one that recorded
            # nothing. The failures are counted stage by stage, so a row
            # cannot change stage under a total that never moves.
            @test r.n_metrics == length(res.metrics) == 7
            @test r.n_mark_failures == 0 && r.n_settlement_failures == 0
            @test !(:n_failures in propertynames(r))
            @test r.schema_version == 8
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
        res = ExperimentResult(exp, L, nothing,
                               compute_metrics(L, nothing, exp.outputs.metrics), RunFailure[])
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

            # The curve and the metrics come back as the run reported them.
            # This config's roots do not exist, so nothing here could have
            # been recomputed even if the loader wanted to.
            @test loaded.curve isa MarkedCurve
            @test loaded.curve.timestamps == res.curve.timestamps
            @test loaded.curve.profit == res.curve.profit
            @test n_unmarked(loaded.curve) == 0
            @test isempty(loaded.failures)
            @test keys(loaded.metrics) == keys(res.metrics)
            @test loaded.metrics.total_pnl ≈ res.metrics.total_pnl ≈ 92.40
            @test loaded.metrics.n_opens == 2 && loaded.metrics.n_closes == 2
            @test loaded.metrics.sharpe ≈ res.metrics.sharpe || isnan(res.metrics.sharpe)
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
            # The record loads in full without the tree; only a *rerun*
            # needs it, which is what reproduce reports it cannot do.
            @test loaded.curve isa MarkedCurve
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

@testset "manifest schema_version: written as 8, and load_run refuses other versions" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            path = replace(joinpath(run_dir(store, id), "manifest.parquet"), "\\" => "/")
            r = first(collect(DBInterface.execute(store.con, "SELECT schema_version FROM '$path'")))
            @test r.schema_version == VolSurfaceAnalysis.RUN_SCHEMA_VERSION == 8
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

            # And schema 5: it stored no curve and no failure table at all,
            # so there is nothing to migrate from -- the run has to be rerun.
            DBInterface.execute(store.con, "CREATE OR REPLACE TABLE m AS SELECT * EXCLUDE (schema_version), 5::INTEGER AS schema_version FROM '$path'")
            DBInterface.execute(store.con, "COPY m TO '$path' (FORMAT PARQUET)")
            err = try load_run(store, id); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("schema_version 5", err.msg) && occursin("rerun the config", err.msg)

            # And schema 6, the version just before this one: its manifest
            # index carries no metric or failure count, so a schema-6 run
            # cannot tell a truncated output table from an empty one. That
            # is a fact about the record, not about the reader, and no
            # migration can add evidence a save never wrote down.
            DBInterface.execute(store.con, "CREATE OR REPLACE TABLE m AS SELECT * EXCLUDE (schema_version, n_metrics, n_mark_failures, n_settlement_failures), 6::INTEGER AS schema_version FROM '$path'")
            DBInterface.execute(store.con, "COPY m TO '$path' (FORMAT PARQUET)")
            err = try load_run(store, id); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("schema_version 6", err.msg) && occursin("rerun the config", err.msg)

            # And schema 7, the version just before this one: it counted the
            # failures as one total, which cannot tell a settlement failure
            # from a mark failure. A manifest that only ever wrote the total
            # down is exempt from the per-stage check unless the version
            # refuses it outright, so it is refused outright.
            id2 = save_run(store, res, _SMOKE_CONFIG)
            path2 = _st_pq(joinpath(run_dir(store, id2), "manifest.parquet"))
            DBInterface.execute(store.con,
                "CREATE OR REPLACE TABLE m AS SELECT * EXCLUDE (schema_version, " *
                "n_mark_failures, n_settlement_failures), " *
                "(n_mark_failures + n_settlement_failures)::BIGINT AS n_failures, " *
                "7::INTEGER AS schema_version FROM '$path2'")
            DBInterface.execute(store.con, "COPY m TO '$path2' (FORMAT PARQUET)")
            err = try load_run(store, id2); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("schema_version 7", err.msg) && occursin("rerun the config", err.msg)
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

# ---- load_run reads the record; reproduce is what reruns it ------------

# A one-session parquet tree: SPY spot prints at the open and the close of
# Fri 2024-01-19, plus one option bar so the bar root is a real tree. It is
# what the reproduction tests rerun against -- a config that can actually
# be run twice, which is the only way a witness can be disagreed with.
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


# A real run of that tree: a noop policy, so the ledger is empty and the
# curve is the one session's realised zero. Small, but genuinely produced
# by `run_experiment`, which is what makes it rerunnable.
_st_session_result(cfg) = run_experiment(load_experiment_str(cfg))

@testset "load_run: reads the stored outputs, and opens no market data" begin
    mktempdir() do tmp
        tree = _st_session_tree(tmp)
        cfg  = _st_session_config(tree)
        exp  = load_experiment_str(cfg)
        L, _ = _lg_case_strangle_closed()
        # A curve and a metric set that today's data would never produce.
        # If the loader recomputed anything, these are what would vanish --
        # and the whole point of storing them is that they do not.
        witness = _st_curve([-999.0, -998.0])
        res = ExperimentResult(exp, L, witness,
                               (total_pnl = 92.40, made_up = -999.0), RunFailure[])
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, cfg)
            loaded = load_run(store, id)
            @test loaded.curve.profit == [-999.0, -998.0]
            @test loaded.curve.timestamps == witness.timestamps
            @test loaded.metrics.made_up == -999.0
            @test loaded.metrics.total_pnl ≈ 92.40
            @test keys(loaded.metrics) == (:total_pnl, :made_up)

            # Move the tree out from under it: the record still loads whole,
            # because loading never asked the data anything.
            mv(tree.spot_root, tree.spot_root * ".gone")
            again = load_run(store, id)
            @test again.curve.profit == [-999.0, -998.0]
            @test again.metrics.made_up == -999.0
        end
        GC.gc()
    end
end

@testset "save/load: the curve keeps both pairs, and an unmarked row has no value" begin
    mktempdir() do tmp
        broken = DateTime(2024, 1, 17, 21, 0)
        curve = MarkedCurve([DateTime(2024, 1, 16, 21, 0), DateTime(2024, 1, 18, 21, 0)],
                            [0.0, 92.40], [broken], [:no_mark])
        res = _build_smoke_result(; curve = curve,
                                  failures = [_st_mark_failure(broken)])
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            path = joinpath(run_dir(store, id), "curve.parquet")
            rows = _st_rows(store, path, "SELECT * FROM '$(_st_pq(path))' ORDER BY kind, instant")
            @test [r.kind for r in rows] == ["marked", "marked", "unmarked"]
            # NULL, not 0.0, not NaN and not the previous session's number.
            @test rows[3].profit === missing
            @test rows[3].reason == "no_mark"
            @test rows[1].reason === missing

            loaded = load_run(store, id)
            @test loaded.curve.timestamps == curve.timestamps
            @test loaded.curve.profit == curve.profit
            @test loaded.curve.unmarked_at == [broken]
            @test loaded.curve.unmarked_reason == [:no_mark]
            @test n_marked(loaded.curve) == 2 && n_unmarked(loaded.curve) == 1
        end
        GC.gc()
    end
end

@testset "save/load: an absent curve and an empty curve stay different on disk" begin
    mktempdir() do tmp
        exp = load_experiment_str(_SMOKE_CONFIG)
        L, _ = _lg_case_strangle_closed()
        none  = ExperimentResult(exp, L, nothing,
                                 compute_metrics(L, nothing, exp.outputs.metrics), RunFailure[])
        empty = _build_smoke_result(; curve = _st_curve(Float64[]))
        with_run_store(joinpath(tmp, "kb")) do store
            # Same experiment, so the same folder: save the empty curve, then
            # the absent one, and the stale file must not survive.
            id = save_run(store, empty, _SMOKE_CONFIG)
            @test isfile(joinpath(run_dir(store, id), "curve.parquet"))
            e = load_run(store, id)
            @test e.curve isa MarkedCurve && n_marked(e.curve) == 0

            @test save_run(store, none, _SMOKE_CONFIG) == id
            @test !isfile(joinpath(run_dir(store, id), "curve.parquet"))
            r = first(_st_rows(store, joinpath(run_dir(store, id), "manifest.parquet")))
            @test r.n_marked === missing && r.n_unmarked === missing
            @test load_run(store, id).curve === nothing
        end
        GC.gc()
    end
end

@testset "save/load: metrics keep NaN, both infinities and their absence" begin
    mktempdir() do tmp
        exp = load_experiment_str(_SMOKE_CONFIG)
        L, _ = _lg_case_strangle_closed()
        curve = _st_curve([0.0, 92.40])
        res = ExperimentResult(exp, L, curve,
                               (total_pnl = 92.40, sharpe = NaN,
                                up = Inf, down = -Inf), RunFailure[])
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            m = load_run(store, id).metrics
            @test isnan(m.sharpe)
            @test m.up == Inf && m.down == -Inf
            # Absence is still different from every number, NaN included: a
            # metric nobody computed has no key at all.
            @test !haskey(m, :sortino)
        end
        GC.gc()
    end
end

@testset "save_run: a metric that is not a number is refused, not NaN-ed" begin
    mktempdir() do tmp
        exp = load_experiment_str(_SMOKE_CONFIG)
        L, _ = _lg_case_strangle_closed()
        curve = _st_curve([0.0])
        res = ExperimentResult(exp, L, curve, (total_pnl = 92.40, label = "nope"),
                               RunFailure[])
        with_run_store(joinpath(tmp, "kb")) do store
            err = try save_run(store, res, _SMOKE_CONFIG); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("label", err.msg)
        end
        GC.gc()
    end
end

@testset "save/load: failures round-trip with instant, stage, subject and reason" begin
    mktempdir() do tmp
        broken = DateTime(2024, 1, 17, 21, 0)
        fs = [RunFailure(DateTime(2024, 1, 16, 14, 0), :settlement,
                         "SPY 2024-01-16T21:00:00 470.0P lot@2", :unexpected_gap),
              _st_mark_failure(broken, "SPY 2024-01-19T21:00:00 480.0P lot@1"),
              _st_mark_failure(broken, "SPY 2024-01-19T21:00:00 490.0C lot@3")]
        curve = MarkedCurve([DateTime(2024, 1, 16, 21, 0)], [0.0], [broken], [:no_mark])
        res = _build_smoke_result(; curve = curve, failures = fs)
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            loaded = load_run(store, id)
            @test loaded.failures == canonical_failures(fs)
            @test [f.stage for f in loaded.failures] == [:settlement, :mark, :mark]
            # Two lots of one session are two questions; the session is still
            # counted once on the curve.
            @test n_unmarked(loaded.curve) == 1
            @test count(f -> f.stage === :mark, loaded.failures) == 2
        end
        GC.gc()
    end
end

@testset "load_run: every manifest count is checked against the stored tables" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        for (col, bad) in (("n_events", 9), ("n_orders", 3), ("n_opens", 1),
                           ("n_closes", 5), ("n_marked", 2), ("n_unmarked", 7),
                           ("n_metrics", 4), ("n_mark_failures", 2),
                           ("n_settlement_failures", 2))
            with_run_store(joinpath(tmp, "kb_" * col)) do store
                id = save_run(store, res, _SMOKE_CONFIG)
                path = _st_pq(joinpath(run_dir(store, id), "manifest.parquet"))
                DBInterface.execute(store.con,
                    "CREATE OR REPLACE TABLE m AS SELECT * EXCLUDE ($col), " *
                    "$(bad)::BIGINT AS $col FROM '$path'")
                DBInterface.execute(store.con, "COPY m TO '$path' (FORMAT PARQUET)")
                err = try load_run(store, id); nothing catch e; e end
                @test err isa ArgumentError
                # The message names the column and both values, so the defect
                # is diagnosable without opening the parquet by hand.
                @test occursin(col, err.msg) && occursin(string(bad), err.msg)
            end
            GC.gc()
        end
    end
end

@testset "load_run: a truncated table is caught by the counts" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            path = _st_pq(joinpath(run_dir(store, id), "curve.parquet"))
            DBInterface.execute(store.con,
                "CREATE OR REPLACE TABLE c AS SELECT * FROM '$path' LIMIT 2")
            DBInterface.execute(store.con, "COPY c TO '$path' (FORMAT PARQUET)")
            err = try load_run(store, id); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("n_marked", err.msg) && occursin("3", err.msg)
        end
        GC.gc()
    end
end

@testset "load_run: a mixed save of curve file and curve counts is named" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            rm(joinpath(run_dir(store, id), "curve.parquet"))
            err = try load_run(store, id); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("no curve.parquet is stored", err.msg)
        end
        GC.gc()
    end
end

@testset "load_run: the curve's unmarked entries and the failures must agree" begin
    mktempdir() do tmp
        broken = DateTime(2024, 1, 17, 21, 0)
        curve = MarkedCurve([DateTime(2024, 1, 16, 21, 0)], [0.0], [broken], [:no_mark])
        # An unmarked session with nothing to say why.
        orphan = ExperimentResult(load_experiment_str(_SMOKE_CONFIG),
                                  first(_lg_case_strangle_closed()), curve,
                                  (total_pnl = 92.40,), RunFailure[])
        with_run_store(joinpath(tmp, "kb1")) do store
            id = save_run(store, orphan, _SMOKE_CONFIG)
            err = try load_run(store, id); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("no failure recorded", err.msg) && occursin("2024-01-17", err.msg)
        end
        # ... and the other direction: a mark failure at a session the curve
        # says it marked.
        stray = _build_smoke_result(;
            failures = [_st_mark_failure(DateTime(2024, 1, 16, 21, 0))])
        with_run_store(joinpath(tmp, "kb2")) do store
            id = save_run(store, stray, _SMOKE_CONFIG)
            err = try load_run(store, id); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("does not report unmarked", err.msg)
        end
        GC.gc()
    end
end

@testset "load_run: a curve reason must occur among its own mark failures" begin
    mktempdir() do tmp
        broken = DateTime(2024, 1, 17, 21, 0)
        curve = MarkedCurve([DateTime(2024, 1, 16, 21, 0)], [0.0], [broken], [:no_mark])
        res = _build_smoke_result(; curve = curve, failures = [_st_mark_failure(broken)])
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            # The instants still line up exactly; only the reasons now
            # contradict each other. A curve that says :no_mark while every
            # failure at that session says :unexpected_gap is two tables
            # describing two different runs, and matching timestamps alone
            # would accept it.
            path = _st_pq(joinpath(run_dir(store, id), "failures.parquet"))
            DBInterface.execute(store.con,
                "CREATE OR REPLACE TABLE f AS SELECT * EXCLUDE (reason), " *
                "'unexpected_gap' AS reason FROM '$path'")
            DBInterface.execute(store.con, "COPY f TO '$path' (FORMAT PARQUET)")
            err = try load_run(store, id); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("no_mark", err.msg) && occursin("unexpected_gap", err.msg)
            @test occursin("2024-01-17", err.msg)
        end
        GC.gc()
    end
end

@testset "load_run: metric and failure membership is protected by its own count" begin
    mktempdir() do tmp
        broken = DateTime(2024, 1, 17, 21, 0)
        # One settlement failure with no curve entry to answer for it, and
        # two mark failures at one broken session -- the two shapes the
        # instant-for-instant agreement cannot see.
        fs = [RunFailure(DateTime(2024, 1, 16, 14, 0), :settlement,
                         "SPY 2024-01-16T21:00:00 470.0P lot@2", :unexpected_gap),
              _st_mark_failure(broken, "SPY 2024-01-19T21:00:00 480.0P lot@1"),
              _st_mark_failure(broken, "SPY 2024-01-19T21:00:00 490.0C lot@3")]
        curve = MarkedCurve([DateTime(2024, 1, 16, 21, 0)], [0.0], [broken], [:no_mark])
        res = _build_smoke_result(; curve = curve, failures = fs)
        @test length(res.metrics) == 7 && length(res.failures) == 3

        # An empty but perfectly schema-valid metrics.parquet is not a run
        # that reported no metrics (design rule 7).
        with_run_store(joinpath(tmp, "kb_metrics")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            path = _st_pq(joinpath(run_dir(store, id), "metrics.parquet"))
            DBInterface.execute(store.con,
                "CREATE OR REPLACE TABLE m AS SELECT * FROM '$path' LIMIT 0")
            DBInterface.execute(store.con, "COPY m TO '$path' (FORMAT PARQUET)")
            err = try load_run(store, id); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("n_metrics", err.msg) && occursin("hold 0", err.msg)
        end
        GC.gc()

        # The settlement failures have no curve entry to miss them: delete
        # them and every structural relationship still holds.
        with_run_store(joinpath(tmp, "kb_settlement")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            path = _st_pq(joinpath(run_dir(store, id), "failures.parquet"))
            DBInterface.execute(store.con,
                "CREATE OR REPLACE TABLE f AS SELECT * FROM '$path' " *
                "WHERE stage <> 'settlement'")
            DBInterface.execute(store.con, "COPY f TO '$path' (FORMAT PARQUET)")
            err = try load_run(store, id); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("n_settlement_failures", err.msg) && occursin("hold 0", err.msg)
        end
        GC.gc()

        # ... and one of the two mark failures of one session, keeping the
        # other: the instants agree, the reasons agree, and only the count
        # knows a question was dropped.
        with_run_store(joinpath(tmp, "kb_mark")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            path = _st_pq(joinpath(run_dir(store, id), "failures.parquet"))
            DBInterface.execute(store.con,
                "CREATE OR REPLACE TABLE f AS SELECT * FROM '$path' " *
                "WHERE subject NOT LIKE '%480.0P%'")
            DBInterface.execute(store.con, "COPY f TO '$path' (FORMAT PARQUET)")
            err = try load_run(store, id); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("n_mark_failures", err.msg) && occursin("hold 1", err.msg)
        end
        GC.gc()
    end
end

@testset "load_run: failure membership is not interchangeable across stages" begin
    mktempdir() do tmp
        broken = DateTime(2024, 1, 17, 21, 0)
        # One settlement failure and two mark failures at one broken
        # session: the shape a single total cannot police, because every
        # rearrangement below keeps the number of rows exactly.
        fs = [RunFailure(DateTime(2024, 1, 16, 14, 0), :settlement,
                         "SPY 2024-01-16T21:00:00 470.0P lot@2", :unexpected_gap),
              _st_mark_failure(broken, "SPY 2024-01-19T21:00:00 480.0P lot@1"),
              _st_mark_failure(broken, "SPY 2024-01-19T21:00:00 490.0C lot@3")]
        curve = MarkedCurve([DateTime(2024, 1, 16, 21, 0)], [0.0], [broken], [:no_mark])
        res = _build_smoke_result(; curve = curve, failures = fs)

        # (1) One of the two mark failures retagged as a settlement failure.
        # The total is unchanged, the unmarked session still keeps a mark
        # failure carrying its reason, and the agreement check -- which sees
        # mark rows only -- is satisfied. Only the per-stage counts know a
        # question was moved from one pass to the other.
        with_run_store(joinpath(tmp, "kb_retag")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            path = _st_pq(joinpath(run_dir(store, id), "failures.parquet"))
            DBInterface.execute(store.con,
                "CREATE OR REPLACE TABLE f AS SELECT * EXCLUDE (stage), " *
                "CASE WHEN subject LIKE '%480.0P%' THEN 'settlement' ELSE stage END " *
                "AS stage FROM '$path'")
            DBInterface.execute(store.con, "COPY f TO '$path' (FORMAT PARQUET)")
            # The row count is exactly what it was.
            @test length(_st_rows(store, joinpath(run_dir(store, id), "failures.parquet"))) == 3
            err = try load_run(store, id); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("n_mark_failures", err.msg) &&
                  occursin("says 2", err.msg) && occursin("hold 1", err.msg)
        end
        GC.gc()

        # (2) The settlement failure deleted and an existing mark failure
        # duplicated in its place. Same number of rows again, the curve
        # still agrees with the mark rows instant and reason, and the
        # settlement question simply vanishes. Both per-stage counts are
        # violated -- one row too many under :mark, one too few under
        # :settlement -- and the first of them names the defect. (Deleting
        # the settlement row on its own is the case above, which names
        # `n_settlement_failures`.)
        with_run_store(joinpath(tmp, "kb_swap")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            path = _st_pq(joinpath(run_dir(store, id), "failures.parquet"))
            DBInterface.execute(store.con,
                "CREATE OR REPLACE TABLE f AS " *
                "SELECT * FROM '$path' WHERE stage <> 'settlement' " *
                "UNION ALL SELECT * FROM '$path' WHERE subject LIKE '%490.0C%'")
            DBInterface.execute(store.con, "COPY f TO '$path' (FORMAT PARQUET)")
            @test length(_st_rows(store, joinpath(run_dir(store, id), "failures.parquet"))) == 3
            err = try load_run(store, id); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("n_mark_failures", err.msg) &&
                  occursin("says 2", err.msg) && occursin("hold 3", err.msg)
        end
        GC.gc()

        # And a stage no pass emits is refused outright: the counts cover
        # exactly the stages the writer names, so a row outside them is a
        # row nothing counts.
        with_run_store(joinpath(tmp, "kb_stage")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            path = _st_pq(joinpath(run_dir(store, id), "failures.parquet"))
            DBInterface.execute(store.con,
                "CREATE OR REPLACE TABLE f AS SELECT * EXCLUDE (stage), " *
                "CASE WHEN stage = 'settlement' THEN 'rollover' ELSE stage END " *
                "AS stage FROM '$path'")
            DBInterface.execute(store.con, "COPY f TO '$path' (FORMAT PARQUET)")
            err = try load_run(store, id); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("rollover", err.msg) && occursin(":mark", err.msg) &&
                  occursin(":settlement", err.msg)
        end
        GC.gc()

        # The writer refuses one too -- and BEFORE it touches the folder. The
        # same experiment is first saved cleanly, then re-saved with a bad
        # stage: the refusal must leave every file byte-identical, not a
        # fresh manifest over a stale failures table.
        with_run_store(joinpath(tmp, "kb_write")) do store
            # The curve has an unmarked instant, so a consistent record carries
            # the mark failure that explains it.
            good = _build_smoke_result(; curve = curve,
                failures = [RunFailure(broken, :mark, "SPY", :no_mark)])
            id = save_run(store, good, _SMOKE_CONFIG)
            files = filter(f -> isfile(joinpath(run_dir(store, id), f)),
                           readdir(run_dir(store, id)))
            before = Dict(f => read(joinpath(run_dir(store, id), f)) for f in files)

            bad = _build_smoke_result(; curve = curve,
                failures = [RunFailure(broken, :rollover, "SPY", :no_mark)])
            @test full_hash(bad.experiment) == id      # it targets the same folder
            err = try save_run(store, bad, _SMOKE_CONFIG); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("rollover", err.msg) && occursin(":mark", err.msg)
            println("  refused before writing: ", err.msg)

            after = Dict(f => read(joinpath(run_dir(store, id), f)) for f in files)
            @test Set(readdir(run_dir(store, id))) == Set(readdir(run_dir(store, id)))
            @test after == before                     # nothing in the folder moved
            @test load_run(store, id).failures == good.failures   # and it still loads as saved
        end
        GC.gc()
    end
end

@testset "load_run: a stored leg's identity is checked, and every row accounted for" begin
    mktempdir() do tmp
        res = _build_smoke_result()          # two orders, two legs each
        legs_of(store, id) = _st_pq(joinpath(run_dir(store, id), "order_legs.parquet"))

        # A rewritten order_leg_id. `leg_idx` sorts the rows and the loader
        # used to mint the ids from first_leg_id, so the stored column could
        # say anything at all and the ledger still loaded as if untouched.
        with_run_store(joinpath(tmp, "kb_id")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            path = legs_of(store, id)
            DBInterface.execute(store.con,
                "CREATE OR REPLACE TABLE l AS SELECT * EXCLUDE (order_leg_id), " *
                "CASE WHEN order_id = 1 AND leg_idx = 2 THEN 77 ELSE order_leg_id END" *
                "::BIGINT AS order_leg_id FROM '$path'")
            DBInterface.execute(store.con, "COPY l TO '$path' (FORMAT PARQUET)")
            err = try load_run(store, id); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("order_leg_id", err.msg) && occursin("77", err.msg)
        end
        GC.gc()

        # Indices shifted without changing their order: 1,2 becomes 2,3, so
        # the ORDER BY sees the same rows in the same order and only the
        # stored index says otherwise.
        with_run_store(joinpath(tmp, "kb_idx")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            path = legs_of(store, id)
            DBInterface.execute(store.con,
                "CREATE OR REPLACE TABLE l AS SELECT * EXCLUDE (leg_idx), " *
                "CASE WHEN order_id = 1 THEN leg_idx + 1 ELSE leg_idx END" *
                "::BIGINT AS leg_idx FROM '$path'")
            DBInterface.execute(store.con, "COPY l TO '$path' (FORMAT PARQUET)")
            err = try load_run(store, id); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("leg_idx", err.msg)
        end
        GC.gc()

        # A leg recorded against an order id no order claims. The loader
        # indexed the legs by order id and read only the ids it knew, so an
        # extra row was dropped in silence and the subset loaded as the record.
        with_run_store(joinpath(tmp, "kb_orphan")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            path = legs_of(store, id)
            DBInterface.execute(store.con,
                "CREATE OR REPLACE TABLE l AS SELECT * FROM '$path' UNION ALL BY NAME " *
                "(SELECT * EXCLUDE (order_id), 99::BIGINT AS order_id FROM '$path' LIMIT 1)")
            DBInterface.execute(store.con, "COPY l TO '$path' (FORMAT PARQUET)")
            err = try load_run(store, id); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("no order claims", err.msg) && occursin("99", err.msg)
        end
        GC.gc()

        # The untampered run still loads: these checks refuse defects, not legs.
        with_run_store(joinpath(tmp, "kb_ok")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            @test load_run(store, id) isa ExperimentResult
        end
        GC.gc()
    end
end

@testset "load_run: a required output file or the dependency manifest is named" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        for f in ("failures.parquet", "metrics.parquet", "events.parquet",
                  "orders.parquet", "order_legs.parquet", "Manifest.toml")
            with_run_store(joinpath(tmp, "kb_" * f)) do store
                id = save_run(store, res, _SMOKE_CONFIG)
                rm(joinpath(run_dir(store, id), f))
                err = try load_run(store, id); nothing catch e; e end
                @test err isa ArgumentError
                @test occursin(f, err.msg)
            end
            GC.gc()
        end
    end
end

@testset "save_run: every event kind survives the named-column writes" begin
    mktempdir() do tmp
        # The strangle ledger holds fills, matches and fees; settling the
        # open strangle adds the fourth kind, with the order journal intact
        # so the join still holds. Each kind writes only its own columns
        # now, so one that gained or lost a column would surface here rather
        # than as a value silently landing in a neighbouring one.
        expired = let (L, _) = _lg_case_strangle_order()
            for lot in collect(open_lots(L.book))
                record_expiry!(L, lot; settlement_price = 468.0,
                               effective_at = lot.contract.expiry, recorded_at = _LG_T_NEXT)
            end
            L
        end
        for L in (first(_lg_case_strangle_closed()), expired)
            exp = load_experiment_str(_SMOKE_CONFIG)
            curve = _st_curve([0.0])
            res = ExperimentResult(exp, L, curve,
                                   compute_metrics(L, curve, exp.outputs.metrics),
                                   RunFailure[])
            with_run_store(joinpath(tmp, "kb_" * string(length(L)))) do store
                id = save_run(store, res, _SMOKE_CONFIG)
                loaded = load_run(store, id)
                @test length(loaded.ledger) == length(L)
                for (a, b) in zip(loaded.ledger.events, L.events)
                    @test typeof(a) === typeof(b)
                    @test cash(a) == cash(b)
                end
                @test loaded.ledger.book == L.book
            end
            GC.gc()
        end
    end
end

# ---- reproduce ---------------------------------------------------------

@testset "reproduce: an unchanged run reproduces, and nothing is written" begin
    mktempdir() do tmp
        tree = _st_session_tree(tmp)
        cfg  = _st_session_config(tree)
        res  = _st_session_result(cfg)
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, cfg; commit_sha="abc123", dirty=false)
            dir = run_dir(store, id)
            before = Dict(f => read(joinpath(dir, f)) for f in readdir(dir))

            rep = reproduce(store, id)
            @test rep.status === :reproduced
            @test isempty(rep.divergences)
            @test rep.run_id == id
            @test rep.stored_code == ("abc123", false)
            @test rep.fresh_code == code_provenance()
            @test occursin("reproduced", sprint(show, MIME"text/plain"(), rep))

            # Read-only: every stored byte is exactly as it was.
            after = Dict(f => read(joinpath(dir, f)) for f in readdir(dir))
            @test keys(after) == keys(before)
            @test all(after[f] == before[f] for f in keys(before))
        end
        GC.gc()
    end
end

@testset "reproduce: a changed curve value is named by row and field" begin
    mktempdir() do tmp
        tree = _st_session_tree(tmp)
        cfg  = _st_session_config(tree)
        res  = _st_session_result(cfg)
        @test n_marked(res.curve) >= 1
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, cfg)
            path = _st_pq(joinpath(run_dir(store, id), "curve.parquet"))
            DBInterface.execute(store.con,
                "CREATE OR REPLACE TABLE c AS SELECT run_id, kind, instant, " *
                "CASE WHEN kind = 'marked' THEN 7.5 ELSE profit END AS profit, reason " *
                "FROM '$path'")
            DBInterface.execute(store.con, "COPY c TO '$path' (FORMAT PARQUET)")

            rep = reproduce(store, id)
            @test rep.status === :diverged
            d = only(d for d in rep.divergences if d.output === :curve)
            @test d.field === :profit
            @test d.stored == "7.5"
            @test occursin("marked", d.row)
            @test occursin("profit", sprint(show, MIME"text/plain"(), rep))
        end
        GC.gc()
    end
end

@testset "reproduce: missing and extra rows are divergence, not silence" begin
    mktempdir() do tmp
        tree = _st_session_tree(tmp)
        cfg  = _st_session_config(tree)
        res  = _st_session_result(cfg)
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, cfg)
            # A metric the record has and the rerun does not. The manifest is
            # the index over the record, so a row added to a table is added
            # to its count too -- otherwise this is a truncation defect and
            # `load_run` refuses it before `reproduce` can compare anything.
            mpath = _st_pq(joinpath(run_dir(store, id), "metrics.parquet"))
            DBInterface.execute(store.con,
                "CREATE OR REPLACE TABLE m AS SELECT * FROM '$mpath' UNION ALL " *
                "SELECT '$id', 'invented', 1.0::DOUBLE")
            DBInterface.execute(store.con, "COPY m TO '$mpath' (FORMAT PARQUET)")
            npath = _st_pq(joinpath(run_dir(store, id), "manifest.parquet"))
            DBInterface.execute(store.con,
                "CREATE OR REPLACE TABLE n AS SELECT * EXCLUDE (n_metrics), " *
                "(n_metrics + 1)::BIGINT AS n_metrics FROM '$npath'")
            DBInterface.execute(store.con, "COPY n TO '$npath' (FORMAT PARQUET)")

            rep = reproduce(store, id)
            @test rep.status === :diverged
            d = only(d for d in rep.divergences if d.output === :metrics)
            @test d.field === :presence
            @test d.stored == "present" && d.fresh == "absent"
            @test occursin("invented", d.row)
        end
        GC.gc()
    end
end

@testset "reproduce: live data that moved is detected, while the load does not" begin
    mktempdir() do tmp
        tree = _st_session_tree(tmp)
        cfg  = _st_session_config(tree)
        res  = _st_session_result(cfg)
        stored_profit = copy(res.curve.profit)
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, cfg)
            # Same config, different world: the session's close moves.
            sdir = joinpath(tree.spot_root, "date=2024-01-19", "symbol=SPY")
            spath = _st_pq(joinpath(sdir, "data.parquet"))
            db = DuckDB.DB(":memory:")
            try
                DBInterface.execute(db, """
                    COPY (SELECT * FROM (VALUES
                        (TIMESTAMP '2024-01-19 14:29:00', 480.0::DOUBLE),
                        (TIMESTAMP '2024-01-19 19:59:00', 483.0::DOUBLE)
                    ) AS t(timestamp, close)) TO '$spath' (FORMAT PARQUET);
                """)
            finally
                DBInterface.close!(db)
            end

            # The record is untouched by the world moving under it.
            @test load_run(store, id).curve.profit == stored_profit
            # The rerun sees a different session close, so the curve's
            # instants move and reproduction says so.
            rep = reproduce(store, id)
            @test rep.status === :diverged
            @test any(d -> d.output === :curve, rep.divergences)
        end
        GC.gc()
    end
end

@testset "reproduce: data this machine cannot open is inability, not success" begin
    mktempdir() do tmp
        tree = _st_session_tree(tmp)
        cfg  = _st_session_config(tree)
        res  = _st_session_result(cfg)
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, cfg)
            mv(tree.spot_root, tree.spot_root * ".gone")
            rep = reproduce(store, id)
            # Never :reproduced on empty outputs, and never :diverged either:
            # the question could not be asked.
            @test rep.status === :unreproducible
            @test occursin("could not be opened", rep.detail)
            @test isempty(rep.divergences)
        end
        GC.gc()
    end
end

@testset "reproduce: a moved identity projection is a named mismatch" begin
    mktempdir() do tmp
        tree = _st_session_tree(tmp)
        cfg  = _st_session_config(tree)
        res  = _st_session_result(cfg)
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, cfg)
            # Stand the run in a folder its own config no longer hashes to,
            # which is what a changed projection looks like from here.
            moved = "0123456789abcdef"
            mv(run_dir(store, id), run_dir(store, moved))
            rep = reproduce(store, moved)
            @test rep.status === :unreproducible
            @test occursin("identity mismatch", rep.detail)
            @test occursin(id, rep.detail)
            # The projection is regenerated for diagnosis rather than stored,
            # and no run at the new id is invented.
            @test occursin("\"outputs\"", rep.detail)
            @test !isdir(run_dir(store, id))
        end
        GC.gc()
    end
end

@testset "reproduce: the comparison is exact for names and bounded for floats" begin
    same = VolSurfaceAnalysis._same_value
    tol  = VolSurfaceAnalysis.REPRODUCE_TOL
    @test tol == 1e-9
    # Finite floats: the documented absolute bound, and its edge.
    @test same(1.0, 1.0 + tol / 2)
    @test !same(1.0, 1.0 + 10 * tol)
    # Non-finite values never reach a subtraction: NaN - NaN and Inf - Inf
    # answer nothing about whether two runs agreed.
    @test same(NaN, NaN)
    @test !same(NaN, 0.0)
    @test same(Inf, Inf) && same(-Inf, -Inf)
    @test !same(Inf, -Inf)
    @test !same(Inf, 1e308)
    # Integers, names and instants compare exactly.
    @test same(3, 3) && !same(3, 4)
    @test same(:no_mark, :no_mark) && !same(:no_mark, :unexpected_gap)
    @test same(DateTime(2024, 1, 1), DateTime(2024, 1, 1))
    @test !same(DateTime(2024, 1, 1), DateTime(2024, 1, 1, 0, 0, 0, 1))
    @test same(missing, missing) && !same(missing, 1.0)
end

@testset "reproduce: an event field difference is named by sequence and field" begin
    # The comparison over two hand-built ledgers: the integration tests above
    # run a noop policy, so this is where a *field* on an event is moved.
    ds = VolSurfaceAnalysis.Divergence[]
    a, _ = _lg_case_strangle_closed()
    b, _ = _lg_case_strangle_closed()
    VolSurfaceAnalysis._compare_rows!(ds, :events, a.events, b.events,
        e -> "sequence " * string(sequence(e)), VolSurfaceAnalysis._event_fields)
    @test isempty(ds)

    fills = [e for e in b.events if e isa Fill]
    @test !isempty(fills)
    moved = [e isa Fill && event_id(e) == event_id(fills[1]) ?
             Fill(header(e), e.group, e.order_leg_id, e.execution_id, e.contract,
                  e.side, e.intent, e.quantity, e.price + 0.25, e.fill_rule) : e
             for e in b.events]
    VolSurfaceAnalysis._compare_rows!(ds, :events, a.events, moved,
        e -> "sequence " * string(sequence(e)), VolSurfaceAnalysis._event_fields)
    d = only(ds)
    @test d.output === :events && d.field === :price
    @test occursin("sequence " * string(sequence(fills[1])), d.row)
    @test d.stored != d.fresh
end

@testset "reproduce: both dependency environments are named, package by package" begin
    mktempdir() do tmp
        tree = _st_session_tree(tmp)
        cfg  = _st_session_config(tree)
        res  = _st_session_result(cfg)
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, cfg; commit_sha="abc123", dirty=false)
            dep = joinpath(run_dir(store, id), "Manifest.toml")

            rep = reproduce(store, id)
            @test rep.status === :reproduced
            @test rep.stored_deps == rep.fresh_deps != "(unknown)"
            @test isempty(rep.deps_changed)
            @test occursin(rep.stored_deps, sprint(show, MIME"text/plain"(), rep))

            # Same commit, different environment. `backtest/settlement.jl`
            # reads the NYSE calendar `BusinessDays` ships, so a code sha
            # alone does not pin what a run consulted: two identical commits
            # over two environments must not produce the same report.
            stored_toml = read(dep, String)
            swapped = replace(stored_toml,
                r"(\[\[deps\.BusinessDays\]\][\s\S]*?version = \")([^\"]+)" => s"\g<1>0.0.1")
            @test swapped != stored_toml
            write(dep, swapped)

            rep2 = reproduce(store, id)
            # The outputs still agree -- differing dependencies are
            # provenance, not divergence -- but the report says so by name.
            @test rep2.status === :reproduced
            @test isempty(rep2.divergences)
            @test rep2.stored_deps != rep2.fresh_deps
            @test rep2.fresh_deps == rep.fresh_deps
            changed = only(rep2.deps_changed)
            @test occursin("BusinessDays", changed) && occursin("0.0.1", changed)
            printed = sprint(show, MIME"text/plain"(), rep2)
            @test occursin("BusinessDays", printed) && occursin("differ", printed)

            # And it is still read-only: the document it read is unchanged.
            @test read(dep, String) == swapped
        end
        GC.gc()
    end
end

@testset "dependency provenance: packages are matched by UUID, not by name" begin
    # Pkg allows two distinct packages with the same name in one
    # environment. Keyed by name, the later block overwrites the earlier
    # one and a version move in the earlier package leaves the report
    # entirely -- the report would say the documents agree on versions
    # while one of them moved.
    stored = """
    julia_version = "1.12.0"
    manifest_format = "2.0"

    [[deps.Widget]]
    uuid = "11111111-1111-1111-1111-111111111111"
    version = "1.0.0"

    [[deps.Widget]]
    uuid = "22222222-2222-2222-2222-222222222222"
    version = "2.0.0"
    """
    fresh = replace(stored, "version = \"1.0.0\"" => "version = \"1.5.0\"")
    changed = VolSurfaceAnalysis._deps_changes(stored, fresh)
    @test length(changed) == 1
    moved = only(changed)
    @test occursin("Widget", moved) && occursin("1.0.0", moved) && occursin("1.5.0", moved)
    # The name alone does not say which Widget moved, so the uuid is on the
    # line; the package that did not move is not reported at all.
    @test occursin("11111111-1111-1111-1111-111111111111", moved)
    @test !occursin("2.0.0", moved)

    # A name is still just a display label when it is unambiguous.
    one = """
    [[deps.BusinessDays]]
    uuid = "33333333-3333-3333-3333-333333333333"
    version = "0.9.25"
    """
    @test only(VolSurfaceAnalysis._deps_changes(one,
            replace(one, "0.9.25" => "0.9.26"))) == "BusinessDays 0.9.25 -> 0.9.26"
end

@testset "dependency provenance: documents that differ with no version change say so" begin
    # Two documents that name many versions, none of which moved: a
    # comment, the project hash or a dependency path changed. Reporting
    # that "neither names a version" would be false -- both name one.
    stored = """
    julia_version = "1.12.0"
    project_hash = "aaaa"

    [[deps.BusinessDays]]
    uuid = "33333333-3333-3333-3333-333333333333"
    version = "0.9.25"
    """
    fresh = replace(stored, "aaaa" => "bbbb")
    msg = only(VolSurfaceAnalysis._deps_changes(stored, fresh))
    @test occursin("documents differ", msg) && occursin("no recorded version changes", msg)
    @test !occursin("neither names a version", msg)
    # Identical documents report nothing at all.
    @test isempty(VolSurfaceAnalysis._deps_changes(stored, stored))
end
