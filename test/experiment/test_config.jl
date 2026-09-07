# Tests for the TOML config loader. Most assertions hit the small
# Dict->object builders directly so we don't need real parquet data on
# disk; one end-to-end `load_experiment` test writes a tiny TOML file
# pointing at a temp parquet tree.

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

# ---- [data.*] provider builders and load-time checks ----

const _CFG_DATA = Dict{String,Any}(
    "option_bar"   => Dict{String,Any}("type" => "parquet_option_bars", "root" => "/x/options_1min"),
    "option_quote" => Dict{String,Any}("type" => "from_bars",
                                       "synthesizer" => Dict{String,Any}("type" => "ohlcv_spread", "lambda" => 0.7)),
    "spot_price"   => Dict{String,Any}("type" => "parquet_spots", "root" => "/x/spots_1min"),
    "rate_curve"   => Dict{String,Any}("type" => "constant", "currency" => "USD", "value" => 0.04),
    "div_curve"    => Dict{String,Any}("type" => "constant", "underlying" => "SPY", "value" => 0.015),
    "vol_surface"  => Dict{String,Any}("type" => "surface_from", "currency" => "USD"),
)
_cfg_data(; overrides...) = (d = deepcopy(_CFG_DATA); for (k, v) in overrides; d[String(k)] = v; end; d)

@testset "build_market_data: every builder from Dicts" begin
    m = build_market_data(_cfg_data())
    @test m isa MarketData
    @test entry(m, OptionBar) == ParquetOptionBars("/x/options_1min")
    @test entry(m, SpotPrice) == ParquetSpots("/x/spots_1min")
    @test entry(m, OptionQuote).synthesizer == SpreadFromOHLCV(0.7)
    @test entry(m, RateCurve) == Constant(RateCurve(Currency("USD"), FlatCurve(0.04)))
    @test entry(m, DivCurve) == Constant(DivCurve(Underlying("SPY"), FlatCurve(0.015)))
    @test entry(m, VolatilitySurface) == SurfaceFrom(currency=Currency("USD"))
    @test kind_name(kind(entry(m, VolatilitySurface))) == "vol_surface"
end

@testset "build_market_data: constant with a curve table, surface_from with spot_for, by_selector" begin
    d = _cfg_data(
        rate_curve = Dict{String,Any}("type" => "constant", "currency" => "usd",
            "curve" => Dict{String,Any}("type" => "pc", "knots" => [DateTime(2024, 1, 1)], "values" => [0.05])),
        vol_surface = Dict{String,Any}("type" => "surface_from", "currency" => "USD",
            "spot_for" => Dict{String,Any}("SPY" => "SPX")),
        spot_price = Dict{String,Any}("type" => "by_selector",
            "SPY" => Dict{String,Any}("type" => "parquet_spots", "root" => "/x/spots_1min"),
            "SPX" => Dict{String,Any}("type" => "parquet_spots", "root" => "/y/spots_1min")))
    m = build_market_data(d)
    rc = entry(m, RateCurve).record
    @test rc.currency == Currency("USD") && rc.curve isa PCCurve && rc.curve(DateTime(2025)) == 0.05
    @test entry(m, VolatilitySurface).spot_for == Dict(Underlying("SPY") => Underlying("SPX"))
    bs = entry(m, SpotPrice)
    @test bs isa BySelector{SpotPrice}
    @test first.(bs.parts) == (Underlying("SPX"), Underlying("SPY"))       # sorted by selector
    @test last(bs.parts[2]) == ParquetSpots("/x/spots_1min")
end

@testset "build_market_data: load-time checks" begin
    # unknown table name
    @test_throws ErrorException build_market_data(_cfg_data(dividends = Dict{String,Any}("type" => "constant")))
    # wrong kind under a table name
    @test_throws ErrorException build_market_data(
        _cfg_data(spot_price = Dict{String,Any}("type" => "parquet_option_bars", "root" => "/x")))
    # derived input missing
    d = _cfg_data(); delete!(d, "option_bar")
    @test_throws ErrorException build_market_data(d)
    d = _cfg_data(); delete!(d, "rate_curve")
    @test_throws ErrorException build_market_data(d)
    # unknown provider type, missing type, constant on a non-curve kind, bad by_selector part
    @test_throws ErrorException build_market_data(_cfg_data(spot_price = Dict{String,Any}("type" => "csv_spots", "path" => "/x")))
    @test_throws ErrorException build_market_data(_cfg_data(spot_price = Dict{String,Any}("root" => "/x")))
    @test_throws ErrorException build_market_data(_cfg_data(spot_price = Dict{String,Any}("type" => "constant", "underlying" => "SPY", "value" => 1.0)))
    @test_throws ErrorException build_market_data(_cfg_data(spot_price = Dict{String,Any}("type" => "by_selector", "SPY" => "not a table")))
    @test_throws ErrorException build_market_data(_cfg_data(spot_price = Dict{String,Any}("type" => "by_selector")))
    @test_throws ErrorException build_market_data(Dict{String,Any}())
    # constant needs value or curve, and the right selector key
    @test_throws ErrorException build_market_data(_cfg_data(rate_curve = Dict{String,Any}("type" => "constant", "currency" => "USD")))
    @test_throws ErrorException build_market_data(_cfg_data(rate_curve = Dict{String,Any}("type" => "constant", "underlying" => "SPY", "value" => 0.04)))
end

@testset "build_market_data: a statically demanded selector nobody serves fails at load" begin
    # the surface selects its rate curve by currency; the only one built is USD
    d = _cfg_data(vol_surface = Dict{String,Any}("type" => "surface_from", "currency" => "EUR"))
    err = try build_market_data(d); nothing catch e; e end
    @test err isa ErrorException
    @test occursin("rate_curve", err.msg)
    @test occursin("EUR", err.msg) && occursin("USD", err.msg)

    # a spot_for remap onto a selector the spot entry does not route
    d2 = _cfg_data(
        vol_surface = Dict{String,Any}("type" => "surface_from", "currency" => "USD",
            "spot_for" => Dict{String,Any}("SPY" => "SPX")),
        spot_price = Dict{String,Any}("type" => "by_selector",
            "SPY" => Dict{String,Any}("type" => "parquet_spots", "root" => "/x/spots_1min")))
    @test_throws ErrorException build_market_data(d2)

    # ... and the check never touches the filesystem: every root above is
    # nonexistent, and a parquet spec answers `missing` rather than probing.
    @test build_market_data(_cfg_data()) isa MarketData
    @test build_market_data(_cfg_data(
        vol_surface = Dict{String,Any}("type" => "surface_from", "currency" => "USD",
            "spot_for" => Dict{String,Any}("SPY" => "SPX")))) isa MarketData
end

@testset "build_clock: kind name and typed selector" begin
    c = build_clock(Dict{String,Any}("kind" => "option_quote", "underlying" => "spy"))
    @test c == Clock{OptionQuote}(Underlying("SPY"))
    @test build_clock(Dict{String,Any}("kind" => "rate_curve", "currency" => "usd")) == Clock{RateCurve}(Currency("USD"))
    @test_throws ErrorException build_clock(Dict{String,Any}("kind" => "bogus", "underlying" => "SPY"))
    @test_throws ErrorException build_clock(Dict{String,Any}("kind" => "option_quote", "currency" => "USD"))
    @test_throws ErrorException build_clock(Dict{String,Any}("underlying" => "SPY"))
end

const _CFG_HEAD = """
name  = "x"
from  = 2024-01-15T15:30:00
to    = 2024-01-15T15:31:00
clock = { kind = "option_quote", underlying = "SPY" }
[data.option_bar]
type = "parquet_option_bars"
root = "/nonexistent/options_1min"
[data.option_quote]
type = "from_bars"
synthesizer = { type = "ohlcv_spread", lambda = 0.7 }
[data.spot_price]
type = "parquet_spots"
root = "/nonexistent/spots_1min"
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

@testset "load_experiment: [data.*] + clock, data absent fails only at open" begin
    e = load_experiment_str(_CFG_HEAD)
    @test e.data isa MarketData
    @test e.clock == Clock{OptionQuote}(Underlying("SPY"))
    @test entry(e.data, OptionBar).root == "/nonexistent/options_1min"
    @test_throws ArgumentError open_data(e.data)
    # clock kind without a table: drop vol_surface (nothing depends on it) and tick on it
    no_surface = replace(_CFG_HEAD, "[data.vol_surface]\ntype = \"surface_from\"\ncurrency = \"USD\"\n" => "")
    @test load_experiment_str(no_surface).data isa MarketData
    @test_throws ErrorException load_experiment_str(replace(no_surface,
        "clock = { kind = \"option_quote\", underlying = \"SPY\" }" => "clock = { kind = \"vol_surface\", underlying = \"SPY\" }"))
    # a clock on a kind that is present but whose selector type is wrong for that kind
    @test_throws ErrorException load_experiment_str(replace(_CFG_HEAD,
        "clock = { kind = \"option_quote\", underlying = \"SPY\" }" => "clock = { kind = \"option_quote\", currency = \"USD\" }"))
    @test_throws ErrorException load_experiment_str(replace(_CFG_HEAD, "clock = { kind = \"option_quote\", underlying = \"SPY\" }\n" => ""))
    @test_throws ErrorException load_experiment_str(replace(_CFG_HEAD, "[data.option_bar]" => "[source]"))
end

@testset "load_experiment: the old [source] table is rejected with a pointer" begin
    err = try load_experiment_str("""
        name = "old"
        from = 2024-01-15T15:30:00
        to   = 2024-01-15T15:31:00
        [source]
        type = "parquet"
        underlying = "SPY"
        root = "/x"
        [agent]
        type = "static"
        [agent.policy]
        type = "noop"
        """); nothing catch e; e end
    @test err isa ErrorException
    @test occursin("[data.<kind>]", err.msg) && occursin("experiment.md", err.msg)
end

@testset "load_experiment: top-level metrics errors clearly" begin
    @test_throws ErrorException load_experiment_str("""
        name = "old_metrics_shape"
        from = 2024-01-15T15:30:00
        to   = 2024-01-15T15:31:00
        metrics = ["sharpe"]
        """)
end

# ---- end-to-end: write a tiny parquet tree, load + run ----

# Minimal parquet fixture: one date, one timestamp, one option row + one
# spot row. The runner's NoOpPolicy never trades; we only need the
# loader to construct a working MarketData + Clock and the engine to find
# at least one clock tick so settlement resolves.
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
            name  = "noop_loader_smoke"
            from  = 2024-01-15T15:30:00
            to    = 2024-01-15T15:30:00
            clock = { kind = "option_quote", underlying = "SPY" }

            [outputs]
            metrics = ["sharpe"]

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
            """)
        end

        exp = load_experiment(cfg_path)
        @test exp.name == "noop_loader_smoke"
        @test exp.from == DateTime(2024, 1, 15, 15, 30)
        @test exp.to   == DateTime(2024, 1, 15, 15, 30)
        @test exp.outputs.metrics == [:sharpe]
        @test exp.agent isa StaticAgent
        @test exp.agent.policy isa NoOpPolicy
        @test exp.data isa MarketData
        @test exp.clock == Clock{OptionQuote}(Underlying("SPY"))
        @test Set(kind_name(kind(s)) for s in exp.data.entries) ==
              Set(["option_bar", "option_quote", "spot_price", "rate_curve", "div_curve", "vol_surface"])
        @test entry(exp.data, OptionBar).root == tree.options_root
        @test entry(exp.data, SpotPrice).root == tree.spot_root
        @test entry(exp.data, RateCurve).record.currency == Currency("USD")
        @test entry(exp.data, DivCurve).record.underlying == Underlying("SPY")

        res = run_experiment(exp)               # opens and closes the readers itself
        @test isempty(res.positions)
        @test res.metrics.total_pnl == 0.0
        @test res.experiment === exp
        @test res.pnl_series.window_end_spot == 480.0
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
                     data=f.data, clock=_EX_CLOCK, from=f.ts1, to=f.ts3,
                     outputs=OutputSpec(metrics=[:sharpe]))
    res = run_experiment(exp)
    io = IOBuffer()
    show(io, MIME"text/plain"(), res)
    s = String(take!(io))
    @test occursin("ExperimentResult: show-test", s)
    @test occursin("option_quote", s)
    @test occursin("clock", s)
    @test occursin("Metrics:", s)
    @test occursin("total_pnl", s)
    @test occursin("sharpe", s)
    @test occursin("Window-end spot", s)
end
