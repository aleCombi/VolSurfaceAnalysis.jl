# Tests for the TOML config loader. Most assertions hit the small
# Dict->object builders directly so we don't need real parquet data on
# disk; one end-to-end `load_experiment` test writes a tiny TOML file
# pointing at a temp `ParquetDataSource` tree.

using TOML
using DuckDB

@testset "build_curve: flat from Dict" begin
    c = build_curve(Dict{String,Any}("type" => "flat", "value" => 0.04))
    @test c isa FlatCurve
    @test c(DateTime(2024, 1, 1)) == 0.04
end

@testset "build_curve: pc from Dict" begin
    c = build_curve(Dict{String,Any}(
        "type"   => "pc",
        "knots"  => [DateTime(2024, 1, 1), DateTime(2024, 6, 1)],
        "values" => [0.04, 0.05],
    ))
    @test c isa PCCurve
    @test c(DateTime(2024, 3, 1)) == 0.04
    @test c(DateTime(2024, 7, 1)) == 0.05
end

@testset "build_curve: unknown type errors with known list" begin
    @test_throws ErrorException build_curve(Dict{String,Any}("type" => "bogus", "value" => 0.0))
end

@testset "build_curve: missing type errors" begin
    @test_throws ErrorException build_curve(Dict{String,Any}("value" => 0.0))
end

@testset "build_synthesizer: ohlcv_spread from Dict" begin
    s = build_synthesizer(Dict{String,Any}("type" => "ohlcv_spread", "lambda" => 0.7))
    @test s isa SpreadFromOHLCV
    @test s.lambda == 0.7
end

@testset "build_synthesizer: missing lambda errors" begin
    @test_throws ErrorException build_synthesizer(Dict{String,Any}("type" => "ohlcv_spread"))
end

@testset "build_synthesizer: unknown type errors with known list" begin
    @test_throws ErrorException build_synthesizer(
        Dict{String,Any}("type" => "bogus", "lambda" => 0.7))
end

@testset "build_data_source: parquet requires [source.synthesizer]" begin
    # `_require` errors before any filesystem touch, so no parquet tree needed.
    @test_throws ErrorException build_data_source(Dict{String,Any}(
        "type"         => "parquet",
        "underlying"   => "SPY",
        "options_root" => "/nonexistent/opts",
        "spot_root"    => "/nonexistent/spot",
    ))
end

@testset "build_policy: noop" begin
    @test build_policy(Dict{String,Any}("type" => "noop")) isa NoOpPolicy
end

@testset "build_policy: daily_short_strangle constructs from Dict" begin
    p = build_policy(Dict{String,Any}(
        "type"        => "daily_short_strangle",
        "underlying"  => "SPY",
        "entry_time"  => Time(15, 45),
        "expiry_days" => 1,
        "put_delta"   => 0.20,
        "call_delta"  => 0.20,
        "quantity"    => 2.0,
    ))
    @test p isa DailyShortStrangle
    @test p.underlying == Underlying("SPY")
    @test p.entry_time == Time(15, 45)
    @test p.expiry_interval == Day(1)
    @test p.put_delta == 0.20
    @test p.call_delta == 0.20
    @test p.quantity == 2.0
end

@testset "build_policy: daily_short_strangle accepts string entry_time" begin
    p = build_policy(Dict{String,Any}(
        "type"        => "daily_short_strangle",
        "underlying"  => "SPY",
        "entry_time"  => "15:45:00",
        "expiry_days" => 1,
        "put_delta"   => 0.20,
        "call_delta"  => 0.20,
    ))
    @test p.entry_time == Time(15, 45)
    @test p.quantity == 1.0
end

@testset "build_policy: daily_short_strangle parses TOML literal" begin
    cfg = TOML.parse("""
        type        = "daily_short_strangle"
        underlying  = "SPY"
        entry_time  = 15:45:00
        expiry_days = 1
        put_delta   = 0.20
        call_delta  = 0.20
        """)
    p = build_policy(cfg)
    @test p isa DailyShortStrangle
    @test p.entry_time == Time(15, 45)
    @test p.expiry_interval == Day(1)
end

@testset "build_policy: daily_short_strangle missing required field errors" begin
    @test_throws ErrorException build_policy(Dict{String,Any}(
        "type"        => "daily_short_strangle",
        "underlying"  => "SPY",
        # entry_time missing
        "expiry_days" => 1,
        "put_delta"   => 0.20,
        "call_delta"  => 0.20,
    ))
end

@testset "build_agent: static wraps the inner policy" begin
    a = build_agent(Dict{String,Any}(
        "type"   => "static",
        "policy" => Dict{String,Any}("type" => "noop"),
    ))
    @test a isa StaticAgent
    @test a.policy isa NoOpPolicy
end

@testset "build_agent: missing policy table errors" begin
    @test_throws ErrorException build_agent(Dict{String,Any}("type" => "static"))
end

# ---- end-to-end: write a tiny parquet tree, load + run ----

# Minimal parquet fixture: one date, one timestamp, one option row + one
# spot row. The runner's NoOpPolicy never trades; we only need the
# loader to construct a working ModelDataSource and the engine to find
# at least one timestamp so settlement resolves.
function _write_smoke_parquet_tree(root::AbstractString)
    options_root = joinpath(root, "options_1min")
    spot_root    = joinpath(root, "spots_1min")
    odir = joinpath(options_root, "date=2024-01-15", "symbol=SPY")
    sdir = joinpath(spot_root,    "date=2024-01-15", "symbol=SPY")
    mkpath(odir)
    mkpath(sdir)

    db = DuckDB.DB(":memory:")
    try
        ts = "2024-01-15 15:30:00"
        opath = replace(joinpath(odir, "data.parquet"), "\\" => "/")
        DBInterface.execute(db, """
            COPY (SELECT
                'O:SPY240216C00480000'        AS ticker,
                5.05::DOUBLE                  AS close,
                TIMESTAMP '$ts'               AS timestamp,
                100.0::DOUBLE                 AS volume,
                'SPY'                         AS parsed_underlying,
                DATE '2024-02-16'             AS parsed_expiry,
                480.0::DOUBLE                 AS parsed_strike,
                'C'                           AS parsed_option_type
            ) TO '$opath' (FORMAT PARQUET);
        """)
        spath = replace(joinpath(sdir, "data.parquet"), "\\" => "/")
        DBInterface.execute(db, """
            COPY (SELECT
                TIMESTAMP '$ts'   AS timestamp,
                480.0::DOUBLE     AS close
            ) TO '$spath' (FORMAT PARQUET);
        """)
    finally
        DBInterface.close!(db)
    end
    return (options_root=options_root, spot_root=spot_root)
end

@testset "load_experiment: end-to-end TOML -> run_experiment" begin
    mktempdir() do tmp
        tree = _write_smoke_parquet_tree(tmp)
        cfg_path = joinpath(tmp, "noop.toml")
        # Use the explicit options_root / spot_root form so we don't
        # depend on the default subdir convention.
        open(cfg_path, "w") do io
            print(io, """
            name = "noop_loader_smoke"
            from = 2024-01-15T15:30:00
            to   = 2024-01-15T15:30:00
            metrics = ["sharpe"]

            [source]
            type         = "parquet"
            underlying   = "SPY"
            options_root = "$(replace(tree.options_root, "\\" => "/"))"
            spot_root    = "$(replace(tree.spot_root,    "\\" => "/"))"

            [source.synthesizer]
            type   = "ohlcv_spread"
            lambda = 0.7

            [source.rate]
            type  = "flat"
            value = 0.04

            [source.div]
            type  = "flat"
            value = 0.015

            [agent]
            type = "static"

            [agent.policy]
            type = "noop"
            """)
        end

        exp = load_experiment(cfg_path)
        @test exp.name == "noop_loader_smoke"
        @test exp.from == DateTime(2024, 1, 15, 15, 30)
        @test exp.to   == DateTime(2024, 1, 15, 15, 30)
        @test exp.metrics == [:sharpe]
        @test exp.agent isa StaticAgent
        @test exp.agent.policy isa NoOpPolicy
        @test exp.source isa ModelDataSource

        res = run_experiment(exp)
        @test isempty(res.positions)
        @test res.metrics.total_pnl == 0.0
        @test res.experiment === exp

        # Release the DuckDB handle so mktempdir cleanup doesn't trip on
        # locked parquet files (Windows). Explicit close + GC.gc() drops
        # both the connection and any lingering Tables cursors.
        close(exp.source.chain_source)
        exp = nothing
        res = nothing
        GC.gc()
    end
end

@testset "Base.show(ExperimentResult): renders header + metrics" begin
    # Reuse the experiment-module fixture for a result with one trade.
    f = _ex_fixture()
    trd = Trade(_EX_UND, 480.0, f.expiry, Call)
    exp = Experiment(name="show-test",
                     agent=StaticAgent(_ExOpenOnceAt(f.ts2, trd)),
                     source=f.mds, from=f.ts1, to=f.ts3,
                     metrics=[:sharpe])
    res = run_experiment(exp)
    io = IOBuffer()
    show(io, MIME"text/plain"(), res)
    s = String(take!(io))
    @test occursin("ExperimentResult: show-test", s)
    @test occursin("Metrics:", s)
    @test occursin("total_pnl", s)
    @test occursin("sharpe", s)
    @test occursin("Settlement spot", s)
end
