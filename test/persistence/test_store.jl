# Tests for the RunStore persistence layer.
#
# These tests reuse the experiment-module fixture (`_ex_fixture` /
# `_ExOpenOnceAt`) defined in test/experiment/test_experiment.jl, which
# runs earlier in the suite. They construct a real ExperimentResult,
# save it, then read the parquet artifacts back via DuckDB SQL to verify
# what landed on disk.

using DuckDB
using DuckDB: DBInterface

const _PERSIST_TOML = """
name = "persist-smoke"
from = 2024-01-15T15:30:00
to   = 2024-01-15T15:32:00

[source]
type = "noop"
"""

# Build a fresh ExperimentResult with one filled trade (residual at settlement).
function _build_smoke_result()
    f = _ex_fixture()
    trd = Trade(_EX_UND, 480.0, f.expiry, Call)
    exp = Experiment(name="persist-smoke",
                     agent=StaticAgent(_ExOpenOnceAt(f.ts2, trd)),
                     source=f.mds, from=f.ts1, to=f.ts3,
                     metrics=[:sharpe, :max_drawdown])
    return run_experiment(exp)
end

@testset "run_id: deterministic SHA-256 of TOML bytes, 16 hex chars" begin
    id1 = run_id(_PERSIST_TOML)
    id2 = run_id(_PERSIST_TOML)
    @test id1 == id2
    @test length(id1) == 16
    @test all(c -> c in "0123456789abcdef", id1)
    # Different bytes -> different id.
    @test run_id(_PERSIST_TOML * " ") != id1
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
            id = save_run(store, res, _PERSIST_TOML)
            @test id == run_id(_PERSIST_TOML)

            dir = run_dir(store, id)
            @test isdir(dir)
            @test isfile(joinpath(dir, "config.toml"))
            @test isfile(joinpath(dir, "manifest.parquet"))
            @test isfile(joinpath(dir, "metrics.parquet"))
            @test isfile(joinpath(dir, "positions.parquet"))
            @test isfile(joinpath(dir, "pnl_series.parquet"))

            @test read(joinpath(dir, "config.toml"), String) == _PERSIST_TOML
        end
        GC.gc()
    end
end

@testset "save_run: manifest row carries name/window/spot/counts" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _PERSIST_TOML)
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
            @test r.settlement_spot == res.pnl_series.settlement_spot
        end
        GC.gc()
    end
end

@testset "save_run: metrics.parquet has one row per (name, value)" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _PERSIST_TOML)
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
            id = save_run(store, res, _PERSIST_TOML)
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
            id = save_run(store, res, _PERSIST_TOML)
            path = joinpath(run_dir(store, id), "pnl_series.parquet")
            rows = collect(DBInterface.execute(store.con,
                "SELECT idx, pnl FROM '$(replace(path, "\\" => "/"))' ORDER BY idx"))
            @test length(rows) == length(res.pnl_series.pnl)
            @test [Float64(r.pnl) for r in rows] ≈ res.pnl_series.pnl
        end
        GC.gc()
    end
end

@testset "save_run: idempotent re-save of same config overwrites in place" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id1 = save_run(store, res, _PERSIST_TOML)
            id2 = save_run(store, res, _PERSIST_TOML)
            @test id1 == id2
            # Still exactly one run folder.
            runs_root = joinpath(store.root, "runs")
            @test length(readdir(runs_root)) == 1
        end
        GC.gc()
    end
end

@testset "save_run: different TOML bytes -> different run folders" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            id1 = save_run(store, res, _PERSIST_TOML)
            id2 = save_run(store, res, _PERSIST_TOML * "\n# comment\n")
            @test id1 != id2
            @test isdir(run_dir(store, id1))
            @test isdir(run_dir(store, id2))
        end
        GC.gc()
    end
end

@testset "save_run: empty positions -> empty positions.parquet still readable" begin
    mktempdir() do tmp
        f = _ex_fixture()
        exp = Experiment(name="empty", agent=StaticAgent(NoOpPolicy()),
                         source=f.mds, from=f.ts1, to=f.ts3)
        res = run_experiment(exp)
        with_run_store(joinpath(tmp, "kb")) do store
            id = save_run(store, res, _PERSIST_TOML)
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
        res = _build_smoke_result()
        with_run_store(joinpath(tmp, "kb")) do store
            save_run(store, res, _PERSIST_TOML)
            save_run(store, res, _PERSIST_TOML * "\n# v2\n")
            glob = replace(joinpath(store.root, "runs", "*", "manifest.parquet"),
                           "\\" => "/")
            rows = collect(DBInterface.execute(store.con,
                "SELECT run_id, name FROM '$glob' ORDER BY run_id"))
            @test length(rows) == 2
            @test all(r -> r.name == "persist-smoke", rows)
        end
        GC.gc()
    end
end

@testset "save_run on closed store throws" begin
    mktempdir() do tmp
        res = _build_smoke_result()
        store = RunStore(joinpath(tmp, "kb"))
        close(store)
        @test_throws ArgumentError save_run(store, res, _PERSIST_TOML)
        GC.gc()
    end
end
