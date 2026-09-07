# Tests for canonical layered experiment identity (core_hash / full_hash).
#
# Experiments are built through the config loader pointed at nonexistent
# roots: specs are pure values, so nothing is opened and nothing to close.

# Data + clock + agent body shared by the structural tests. Explicit roots,
# so there is no platform path-separator ambiguity.
const _ID_SRC_TOML = """
clock = { kind = "option_quote", underlying = "SPY" }
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

_id_toml(; name="x", from="2024-01-15T15:30:00", to="2024-01-15T15:31:00") =
    "name = \"$name\"\nfrom = $from\nto = $to\n" * _ID_SRC_TOML

@testset "identity: hashes are 16 hex chars" begin
    e = load_experiment_str(_id_toml())
    for h in (core_hash(e), full_hash(e))
        @test length(h) == 16
        @test all(c -> c in "0123456789abcdef", h)
    end
end

@testset "identity: name excluded from both hashes" begin
    a = load_experiment_str(_id_toml(name="a"))
    b = load_experiment_str(_id_toml(name="b"))
    @test core_hash(a) == core_hash(b)
    @test full_hash(a) == full_hash(b)
end

@testset "identity: outputs change full_hash, not core_hash" begin
    base = load_experiment_str(_id_toml())
    src, clk, ag = base.data, base.clock, base.agent
    e1 = Experiment(name="x", agent=ag, data=src, clock=clk, from=base.from, to=base.to,
                    outputs=OutputSpec(metrics=[:sharpe]))
    e2 = Experiment(name="x", agent=ag, data=src, clock=clk, from=base.from, to=base.to,
                    outputs=OutputSpec(metrics=[:sharpe, :sortino]))
    @test core_hash(e1) == core_hash(e2)
    @test full_hash(e1) != full_hash(e2)
end

@testset "identity: metric order does not change full_hash" begin
    base = load_experiment_str(_id_toml())
    src, clk, ag = base.data, base.clock, base.agent
    e1 = Experiment(name="x", agent=ag, data=src, clock=clk, from=base.from, to=base.to,
                    outputs=OutputSpec(metrics=[:sharpe, :sortino]))
    e2 = Experiment(name="x", agent=ag, data=src, clock=clk, from=base.from, to=base.to,
                    outputs=OutputSpec(metrics=[:sortino, :sharpe]))
    @test full_hash(e1) == full_hash(e2)
end

@testset "identity: window change changes core_hash" begin
    base = load_experiment_str(_id_toml())
    src, clk, ag = base.data, base.clock, base.agent
    moved = Experiment(name="x", agent=ag, data=src, clock=clk,
                       from=base.from, to=DateTime(2024, 1, 15, 15, 32))
    @test core_hash(moved) != core_hash(base)
end

@testset "identity: invariant to whitespace, key order, name, omitted defaults" begin
    a = load_experiment_str(_id_toml(name="a"))   # outputs omitted -> default all
    b_toml = """
    to   =  2024-01-15T15:31:00
    from =  2024-01-15T15:30:00
    name = "b"
    clock = { underlying = "spy", kind = "option_quote" }

    [outputs]
    metrics = ["max_drawdown", "profit_factor", "sharpe", "sortino", "volatility"]
    artifacts = ["equity_curve"]

    [agent]
    type = "static"
    [agent.policy]
    type = "noop"

    [data.vol_surface]
    currency = "usd"
    type = "surface_from"
    [data.div_curve]
    value = 0.015
    underlying = "SPY"
    type = "constant"
    [data.rate_curve]
    curve = { type = "flat", value = 0.04 }
    currency = "USD"
    type = "constant"
    [data.spot_price]
    root = "/nonexistent/spot"
    type = "parquet_spots"
    [data.option_quote]
    synthesizer = { lambda = 0.7, type = "ohlcv_spread" }
    type = "from_bars"
    [data.option_bar]
    root = "/nonexistent/opts"
    type = "parquet_option_bars"
    """
    b = load_experiment_str(b_toml)
    @test core_hash(a) == core_hash(b)
    @test full_hash(a) == full_hash(b)
end

@testset "identity: spot_for and by_selector order in TOML do not change the hash" begin
    mk(spot_for, parts) = """
    name  = "a"
    from  = 2024-01-15T15:30:00
    to    = 2024-01-15T15:31:00
    clock = { kind = "option_quote", underlying = "SPY" }
    [data.option_bar]
    type = "parquet_option_bars"
    root = "/x/opts"
    [data.option_quote]
    type = "from_bars"
    synthesizer = { type = "ohlcv_spread", lambda = 0.7 }
    [data.spot_price]
    type = "by_selector"
    $parts
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
    spot_for = { $spot_for }
    [agent]
    type = "static"
    [agent.policy]
    type = "noop"
    """
    p1 = "SPY = { type = \"parquet_spots\", root = \"/x/spot\" }\nSPX = { type = \"parquet_spots\", root = \"/y/spot\" }"
    p2 = "SPX = { type = \"parquet_spots\", root = \"/y/spot\" }\nSPY = { type = \"parquet_spots\", root = \"/x/spot\" }"
    a = load_experiment_str(mk("SPY = \"SPX\", SPX = \"SPY\"", p1))
    b = load_experiment_str(mk("SPX = \"SPY\", SPY = \"SPX\"", p2))
    @test core_hash(a) == core_hash(b)
    @test full_hash(a) == full_hash(b)
    c = load_experiment_str(mk("SPY = \"SPX\"", p1))
    @test core_hash(c) != core_hash(a)
end

@testset "identity: in-memory providers are not hashable (so not saveable)" begin
    f = _ex_fixture()
    exp = Experiment(name="mem", agent=StaticAgent(NoOpPolicy()),
                     data=f.data, clock=_EX_CLOCK, from=f.ts1, to=f.ts3)
    @test_throws ErrorException core_hash(exp)
    @test_throws ErrorException full_hash(exp)
end

@testset "identity: to_dict(MarketData) has one entry per kind; dataset slot; clock" begin
    e = load_experiment_str(_id_toml())
    d = VolSurfaceAnalysis.to_dict(e.data)
    @test Set(keys(d["entries"])) ==
          Set(["option_bar", "option_quote", "spot_price", "rate_curve", "div_curve", "vol_surface"])
    @test d["entries"]["option_bar"]["dataset"]["root"] == "/nonexistent/opts"
    @test d["entries"]["spot_price"]["type"] == "parquet_spots"
    @test d["entries"]["option_quote"]["synthesizer"]["lambda"] == 0.7
    @test d["entries"]["rate_curve"]["selector"] == "USD"
    @test d["entries"]["div_curve"]["selector"] == "SPY"
    @test !haskey(d["entries"]["rate_curve"], "timestamp")
    @test VolSurfaceAnalysis.to_dict(e.clock) == Dict("kind" => "option_quote", "selector" => "SPY")
    # a stamped constant records its visibility time
    stamped = Constant(RateCurve(Currency("USD"), FlatCurve(0.04), DateTime(2024, 1, 1)))
    @test VolSurfaceAnalysis.to_dict(stamped)["timestamp"] == "2024-01-01T00:00:00"
end

@testset "identity: clock is part of core_hash; spot_for and BySelector order are not" begin
    base = load_experiment_str(_id_toml())
    other_clock = Experiment(name="x", agent=base.agent, data=base.data,
                             clock=Clock{SpotPrice}(Underlying("SPY")), from=base.from, to=base.to)
    @test core_hash(other_clock) != core_hash(base)

    spy, spx = Underlying("SPY"), Underlying("SPX")
    mk_data(spot_for, parts) = MarketData(
        ParquetOptionBars("/x/opts"), QuotesFromBars(SpreadFromOHLCV(0.7)),
        BySelector{SpotPrice}(parts...),
        Constant(RateCurve(Currency("USD"), FlatCurve(0.04))),
        Constant(DivCurve(spy, FlatCurve(0.015))),
        SurfaceFrom(currency=Currency("USD"), spot_for=spot_for))
    a = mk_data(Dict(spy => spx, spx => spy), (spy => ParquetSpots("/x/spot"), spx => ParquetSpots("/y/spot")))
    b = mk_data(Dict(spx => spy, spy => spx), (spx => ParquetSpots("/y/spot"), spy => ParquetSpots("/x/spot")))
    ea = Experiment(name="a", agent=base.agent, data=a, clock=base.clock, from=base.from, to=base.to)
    eb = Experiment(name="b", agent=base.agent, data=b, clock=base.clock, from=base.from, to=base.to)
    @test core_hash(ea) == core_hash(eb)
    @test VolSurfaceAnalysis.to_dict(a)["entries"]["spot_price"]["parts"][1]["selector"] == "SPX"
    @test VolSurfaceAnalysis.to_dict(a)["entries"]["vol_surface"]["spot_for"] == [["SPX", "SPY"], ["SPY", "SPX"]]
    # lookback_ticks changes which surface a policy sees, so it is core identity
    lb = MarketData(
        ParquetOptionBars("/x/opts"), QuotesFromBars(SpreadFromOHLCV(0.7)),
        BySelector{SpotPrice}(spy => ParquetSpots("/x/spot"), spx => ParquetSpots("/y/spot")),
        Constant(RateCurve(Currency("USD"), FlatCurve(0.04))),
        Constant(DivCurve(spy, FlatCurve(0.015))),
        SurfaceFrom(currency=Currency("USD"), spot_for=Dict(spy => spx, spx => spy),
                    lookback_ticks=5))
    elb = Experiment(name="lb", agent=base.agent, data=lb, clock=base.clock,
                     from=base.from, to=base.to)
    @test core_hash(elb) != core_hash(ea)
    @test VolSurfaceAnalysis.to_dict(a)["entries"]["vol_surface"]["lookback_ticks"] == 3
    c = mk_data(Dict(spy => spx), (spy => ParquetSpots("/x/spot"), spx => ParquetSpots("/z/spot")))
    ec = Experiment(name="c", agent=base.agent, data=c, clock=base.clock, from=base.from, to=base.to)
    @test core_hash(ec) != core_hash(ea)
end

@testset "identity: DailyShortStrangle expiry-interval unit is part of identity" begin
    # Same value (1), different unit -> must not collide. (Dates.value alone
    # would: value(Day(1)) == value(Week(1)) == 1.)
    p_day  = DailyShortStrangle(; underlying=Underlying("SPY"), entry_time=Time(15, 45),
                                expiry_interval=Day(1), put_delta=0.2, call_delta=0.2)
    p_week = DailyShortStrangle(; underlying=Underlying("SPY"), entry_time=Time(15, 45),
                                expiry_interval=Week(1), put_delta=0.2, call_delta=0.2)
    @test VolSurfaceAnalysis.to_dict(p_day)["expiry_interval"] !=
          VolSurfaceAnalysis.to_dict(p_week)["expiry_interval"]
end
