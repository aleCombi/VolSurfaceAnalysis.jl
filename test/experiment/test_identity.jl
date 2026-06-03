# Tests for canonical layered experiment identity (core_hash / full_hash).
#
# Experiments are built through the parquet config loader pointed at
# nonexistent roots (ParquetDataSource validates lazily, so construction
# only warns). to_dict reads fields without touching the DuckDB handle; we
# close the sources at the end of each testset to avoid leaking them.

# Source + agent body shared by the structural tests. Explicit roots, so
# there is no platform path-separator ambiguity.
const _ID_SRC_TOML = """
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

_id_toml(; name="x", from="2024-01-15T15:30:00", to="2024-01-15T15:31:00") =
    "name = \"$name\"\nfrom = $from\nto = $to\n" * _ID_SRC_TOML

@testset "identity: hashes are 16 hex chars" begin
    e = load_experiment_str(_id_toml())
    for h in (core_hash(e), full_hash(e))
        @test length(h) == 16
        @test all(c -> c in "0123456789abcdef", h)
    end
    close(e.source.chain_source)
end

@testset "identity: name excluded from both hashes" begin
    a = load_experiment_str(_id_toml(name="a"))
    b = load_experiment_str(_id_toml(name="b"))
    @test core_hash(a) == core_hash(b)
    @test full_hash(a) == full_hash(b)
    close(a.source.chain_source); close(b.source.chain_source)
end

@testset "identity: outputs change full_hash, not core_hash" begin
    base = load_experiment_str(_id_toml())
    src, ag = base.source, base.agent
    e1 = Experiment(name="x", agent=ag, source=src, from=base.from, to=base.to,
                    outputs=OutputSpec(metrics=[:sharpe]))
    e2 = Experiment(name="x", agent=ag, source=src, from=base.from, to=base.to,
                    outputs=OutputSpec(metrics=[:sharpe, :sortino]))
    @test core_hash(e1) == core_hash(e2)
    @test full_hash(e1) != full_hash(e2)
    close(src.chain_source)
end

@testset "identity: metric order does not change full_hash" begin
    base = load_experiment_str(_id_toml())
    src, ag = base.source, base.agent
    e1 = Experiment(name="x", agent=ag, source=src, from=base.from, to=base.to,
                    outputs=OutputSpec(metrics=[:sharpe, :sortino]))
    e2 = Experiment(name="x", agent=ag, source=src, from=base.from, to=base.to,
                    outputs=OutputSpec(metrics=[:sortino, :sharpe]))
    @test full_hash(e1) == full_hash(e2)
    close(src.chain_source)
end

@testset "identity: window change changes core_hash" begin
    base = load_experiment_str(_id_toml())
    src, ag = base.source, base.agent
    moved = Experiment(name="x", agent=ag, source=src,
                       from=base.from, to=DateTime(2024, 1, 15, 15, 32))
    @test core_hash(moved) != core_hash(base)
    close(src.chain_source)
end

@testset "identity: invariant to whitespace, key order, name, cache knobs, omitted defaults" begin
    a = load_experiment_str(_id_toml(name="a"))   # outputs omitted -> default all
    b_toml = """
    to   =  2024-01-15T15:31:00
    from =  2024-01-15T15:30:00
    name = "b"

    [outputs]
    metrics = ["max_drawdown", "profit_factor", "sharpe", "sortino", "volatility"]
    artifacts = ["equity_curve"]

    [agent]
    type = "static"
    [agent.policy]
    type = "noop"

    [source]
    type = "parquet"
    underlying = "SPY"
    options_root = "/nonexistent/opts"
    spot_root = "/nonexistent/spot"
    max_days_cached = 99
    [source.synthesizer]
    type = "ohlcv_spread"
    lambda = 0.7
    [source.rate]
    type = "flat"
    value = 0.04
    [source.div]
    type = "flat"
    value = 0.015
    """
    b = load_experiment_str(b_toml)
    @test core_hash(a) == core_hash(b)
    @test full_hash(a) == full_hash(b)
    close(a.source.chain_source); close(b.source.chain_source)
end

@testset "identity: cache knobs excluded from hashes" begin
    mk(mdc) = """
    name = "a"
    from = 2024-01-15T15:30:00
    to   = 2024-01-15T15:31:00
    [source]
    type = "parquet"
    underlying = "SPY"
    options_root = "/x/opts"
    spot_root = "/x/spot"
    max_days_cached = $mdc
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
    a = load_experiment_str(mk(3))
    b = load_experiment_str(mk(99))
    @test core_hash(a) == core_hash(b)
    @test full_hash(a) == full_hash(b)
    close(a.source.chain_source); close(b.source.chain_source)
end

@testset "identity: in-memory sources are not hashable (so not saveable)" begin
    f = _ex_fixture()
    exp = Experiment(name="mem", agent=StaticAgent(NoOpPolicy()),
                     source=f.mds, from=f.ts1, to=f.ts3)
    @test_throws ErrorException core_hash(exp)
    @test_throws ErrorException full_hash(exp)
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
