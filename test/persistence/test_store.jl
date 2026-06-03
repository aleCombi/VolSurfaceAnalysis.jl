# Tests for the RunStore persistence layer.
#
# A run's id is `full_hash(result.experiment)`, and `save_run` validates
# that the persisted config.toml rebuilds that same experiment. So these
# tests save *config-buildable* experiments (parquet source, validated
# lazily so no data tree is needed) paired with a hand-built ledger. The
# hand-built positions exercise the serialization layer directly -- they
# need not come from a real backtest, and an in-memory source could not be
# hashed/saved anyway.

using DuckDB
using DuckDB: DBInterface

# A buildable parquet + noop config. Roots are nonexistent on purpose:
# ParquetDataSource validates lazily, so the experiment builds and hashes
# without any data on disk; only an actual get_chain/get_spot would throw.
function _smoke_config(; name="persist-smoke", metrics="[\"sharpe\", \"max_drawdown\"]")
    """
    name = "$name"
    from = 2024-01-15T15:30:00
    to   = 2024-01-15T15:32:00

    [outputs]
    metrics = $metrics

    [source]
    type = "parquet"
    underlying = "SPY"
    options_root = "/nonexistent/opts"
    spot_root = "/nonexistent/spot"

    [source.synthesizer]
    type = "ohlcv_spread"
    lambda = 0.7

    [source.rate]
    type = "flat"
    value = 0.04

    [source.div]
    type = "flat"
    value = 0.015

    [agent]
    type = "static"

    [agent.policy]
    type = "noop"
    """
end

const _SMOKE_CONFIG = _smoke_config()

# Config-buildable experiment + a hand-built ledger (one long 480 call
# settling to -5.10). Positions are constructed directly: we are testing
# serialization, not the backtest.
function _build_smoke_result(config=_SMOKE_CONFIG)
    exp = load_experiment_str(config)
    trd = Trade(Underlying("SPY"), 480.0, DateTime(2024, 2, 16, 21, 0), Call;
                direction=1, quantity=1.0)
    pos = Position(trd, 5.10, 480.0, 5.00, 5.10, DateTime(2024, 1, 15, 15, 31))
    series = PnLSeries([DateTime(2024, 2, 16, 21, 0)], [-5.10], 480.0, 1, 0, 0)
    ExperimentResult(exp, [pos], series, compute_metrics(series, exp.outputs.metrics))
end

# Config-buildable experiment with an empty ledger (folder / identity tests).
function _empty_result(config)
    exp = load_experiment_str(config)
    series = PnLSeries(DateTime[], Float64[], 480.0, 0, 0, 0)
    ExperimentResult(exp, Position[], series, compute_metrics(series, exp.outputs.metrics))
end

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

@testset "save_run: writes config.toml + 4 parquet files under runs/run_id=<hash>/" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            @test id == full_hash(res.experiment)

            dir = run_dir(store, id)
            @test isdir(dir)
            @test isfile(joinpath(dir, "config.toml"))
            @test isfile(joinpath(dir, "manifest.parquet"))
            @test isfile(joinpath(dir, "metrics.parquet"))
            @test isfile(joinpath(dir, "positions.parquet"))
            @test isfile(joinpath(dir, "pnl_series.parquet"))

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

@testset "save_run: manifest row carries name/window/spot/counts" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            path = joinpath(run_dir(store, id), "manifest.parquet")
            rows = collect(DBInterface.execute(store.con,
                "SELECT * FROM '$(replace(path, "\\" => "/"))'"))
            @test length(rows) == 1
            r = first(rows)
            @test r.run_id == id
            @test r.name == "persist-smoke"
            @test r.n_positions == length(res.positions)
            @test r.n_opens == res.pnl_series.n_opens
            @test r.n_closes == res.pnl_series.n_closes
            @test r.window_end_spot == res.pnl_series.window_end_spot
            @test r.n_unmarked == res.pnl_series.n_unmarked
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
            r = first(collect(DBInterface.execute(store.con,
                "SELECT core_hash, commit_sha, dirty FROM '$(replace(path, "\\" => "/"))'")))
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
            rows = collect(DBInterface.execute(store.con,
                "SELECT metric_name, value FROM '$(replace(path, "\\" => "/"))'"))
            names = Set(r.metric_name for r in rows)
            # Always-on core metrics plus the two requested optionals.
            @test "total_pnl"     in names
            @test "n_round_trips" in names
            @test "hit_rate"      in names
            @test "sharpe"        in names
            @test "max_drawdown"  in names
            total_pnl_row = first(r for r in rows if r.metric_name == "total_pnl")
            @test total_pnl_row.value ≈ -5.10
        end
        GC.gc()
    end
end

@testset "save_run: positions.parquet flattens legs with bid/ask + entry snapshot" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            path = joinpath(run_dir(store, id), "positions.parquet")
            rows = collect(DBInterface.execute(store.con,
                "SELECT * FROM '$(replace(path, "\\" => "/"))'"))
            @test length(rows) == length(res.positions)
            r = first(rows)
            p = first(res.positions)
            @test r.run_id == id
            @test r.leg_idx == 1
            @test r.underlying == "SPY"
            @test r.strike == p.trade.strike
            @test r.option_type == (p.trade.option_type == Call ? "C" : "P")
            @test r.direction == p.trade.direction
            @test r.quantity == p.trade.quantity
            @test r.entry_price == p.entry_price
            @test r.entry_spot == p.entry_spot
            @test r.entry_bid == p.entry_bid
            @test r.entry_ask == p.entry_ask
        end
        GC.gc()
    end
end

@testset "save_run: pnl_series.parquet has one row per round trip" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            path = joinpath(run_dir(store, id), "pnl_series.parquet")
            rows = collect(DBInterface.execute(store.con,
                "SELECT idx, pnl FROM '$(replace(path, "\\" => "/"))' ORDER BY idx"))
            @test length(rows) == length(res.pnl_series.pnl)
            @test [Float64(r.pnl) for r in rows] ≈ res.pnl_series.pnl
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

@testset "save_run: empty positions -> empty positions.parquet still readable" begin
    mktempdir() do tmp
        res = _empty_result(_SMOKE_CONFIG)
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            path = joinpath(run_dir(store, id), "positions.parquet")
            @test isfile(path)
            rows = collect(DBInterface.execute(store.con,
                "SELECT * FROM '$(replace(path, "\\" => "/"))'"))
            @test isempty(rows)
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

@testset "load_run: round-trip ExperimentResult (positions, pnl, metrics)" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            loaded = load_run(store, id)

            @test loaded isa ExperimentResult

            @test length(loaded.positions) == length(res.positions)
            for (p, q) in zip(loaded.positions, res.positions)
                @test p.trade.strike == q.trade.strike
                @test p.trade.expiry == q.trade.expiry
                @test p.trade.option_type == q.trade.option_type
                @test p.trade.direction == q.trade.direction
                @test p.trade.quantity == q.trade.quantity
                @test ticker(p.trade.underlying) == ticker(q.trade.underlying)
                @test p.entry_price == q.entry_price
                @test p.entry_spot == q.entry_spot
                @test p.entry_bid == q.entry_bid
                @test p.entry_ask == q.entry_ask
                @test p.entry_timestamp == q.entry_timestamp
            end

            @test loaded.pnl_series.timestamps == res.pnl_series.timestamps
            @test loaded.pnl_series.pnl ≈ res.pnl_series.pnl
            @test loaded.pnl_series.window_end_spot == res.pnl_series.window_end_spot
            @test loaded.pnl_series.n_opens == res.pnl_series.n_opens
            @test loaded.pnl_series.n_closes == res.pnl_series.n_closes
            @test loaded.pnl_series.n_unmarked == res.pnl_series.n_unmarked

            # Metrics: keys match, types preserved (Int stays Int), NaN preserved
            @test keys(loaded.metrics) == keys(res.metrics)
            @test loaded.metrics.n_round_trips isa Int
            @test loaded.metrics.n_opens isa Int
            @test loaded.metrics.total_pnl ≈ res.metrics.total_pnl
            @test isnan(loaded.metrics.sharpe) == isnan(res.metrics.sharpe)
            @test loaded.metrics.max_drawdown == res.metrics.max_drawdown
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
            @test loaded.experiment.source isa ModelDataSource
        end
        GC.gc()
    end
end

@testset "load_run: works when source data is absent (lazy root validation)" begin
    mktempdir() do tmp
        # _SMOKE_CONFIG points at nonexistent roots: the source rebuilds and
        # the persisted fields load, but an actual chain read throws.
        res = _build_smoke_result()
        store_root = joinpath(tmp, "kb")
        id = with_run_store(store_root) do store
            save_run(store, res, _SMOKE_CONFIG)
        end
        with_run_store(store_root) do store
            loaded = load_run(store, id)
            @test length(loaded.positions) == length(res.positions)
            @test loaded.metrics.total_pnl ≈ res.metrics.total_pnl
            @test_throws Exception get_chain(loaded.experiment.source.chain_source,
                                             DateTime(2024, 1, 15, 15, 30))
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

@testset "load_run: empty positions round-trip as empty Vector{Position}" begin
    mktempdir() do tmp
        res = _empty_result(_SMOKE_CONFIG)
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _SMOKE_CONFIG)
            loaded = load_run(store, id)
            @test isempty(loaded.positions)
            @test isempty(loaded.pnl_series.pnl)
        end
        GC.gc()
    end
end
