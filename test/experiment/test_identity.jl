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
    artifacts = ["marked_curve"]

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

@testset "identity: a strangle quantity of 1 and of 1.0 is one experiment" begin
    # the loader builds an Int either way and the canonical form collapses
    # 1 and 1.0, so no run id moves with the integer quantity
    strangle(q) = """
        name = "q"
        from = 2024-01-15T15:30:00
        to = 2024-01-15T15:31:00
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
        type = "daily_short_strangle"
        underlying = "SPY"
        entry_time = 15:45:00
        expiry_days = 1
        put_delta = 0.20
        call_delta = 0.20
        $q
        """
    a = load_experiment_str(strangle("quantity = 1"))
    b = load_experiment_str(strangle("quantity = 1.0"))
    c = load_experiment_str(strangle(""))                 # the default is 1
    @test a.agent.policy.quantity === b.agent.policy.quantity === c.agent.policy.quantity === 1
    @test full_hash(a) == full_hash(b) == full_hash(c)
    @test core_hash(a) == core_hash(b) == core_hash(c)
    @test full_hash(a) != full_hash(load_experiment_str(strangle("quantity = 2")))
end

@testset "identity: the venue's two choices are in core_hash" begin
    base = load_experiment_str(_id_toml())
    @test VolSurfaceAnalysis._core_dict(base)["venue"] ==
          Dict("fill_rule" => "cross_spread", "cost_model" => "ibkr_pro_us_options")
    # A different cost model is a different backtest: the fees are ledger
    # events, so every round trip's cash moves with it.
    free = load_experiment_str(_id_toml() * "\n[venue]\ncost_model = \"none\"\n")
    @test free.cost_model === :none
    @test core_hash(free) != core_hash(base)
    @test full_hash(free) != full_hash(base)
    # Direct construction for the fill rule, because the loader only accepts
    # rules the venue has: identity is computed on an `Experiment`, not on
    # TOML, and it must fork on any value the field can hold.
    other = Experiment(name="x", agent=base.agent, data=base.data, clock=base.clock,
                       from=base.from, to=base.to, fill_rule=:mid)
    @test core_hash(other) != core_hash(base)
    @test full_hash(other) != full_hash(base)
end

@testset "identity: an omitted [venue] and an explicit default are one run id" begin
    # The standing omitted-vs-explicit invariant: identity is projected from
    # the resolved experiment, so how a config spells a default cannot move
    # the id. Both spellings must also survive a `name` change.
    base = load_experiment_str(_id_toml(name="a"))
    explicit = load_experiment_str(_id_toml(name="b") *
        "\n[venue]\nfill_rule = \"cross_spread\"\ncost_model = \"ibkr_pro_us_options\"\n")
    @test core_hash(explicit) == core_hash(base)
    @test full_hash(explicit) == full_hash(base)
end

@testset "identity: the resolved contract facts are in core_hash" begin
    base = load_experiment_str(_id_toml())
    d = VolSurfaceAnalysis._core_dict(base)
    # The resolved spec for the experiment's one underlying, not the table:
    # projecting the table would fork every run id on an entry the run never
    # touches.
    @test d["contract"] == Dict{String,Any}("multiplier" => 100, "exercise" => "American",
                                            "settlement" => "PMSettled", "delivery" => "Physical")
    @test !haskey(d["contract"], "SPY") && !haskey(d["contract"], "QQQ")
    # A spec differing in any one field is a different id. Projected
    # directly rather than by mutating `_CONTRACT_TABLE`, which every other
    # testset in the run shares.
    spec = contract_spec(Underlying("SPY"))
    variants = [ContractSpec(10, spec.exercise, spec.settlement, spec.delivery),
                ContractSpec(spec.multiplier, European, spec.settlement, spec.delivery),
                ContractSpec(spec.multiplier, spec.exercise, AMSettled, spec.delivery),
                ContractSpec(spec.multiplier, spec.exercise, spec.settlement, Cash)]
    hashes = Set{String}()
    for v in variants
        forked = copy(d)
        forked["contract"] = VolSurfaceAnalysis.to_dict(v)
        push!(hashes, VolSurfaceAnalysis._hash16(VolSurfaceAnalysis._canonical(forked)))
    end
    @test length(hashes) == length(variants)
    @test !(core_hash(base) in hashes)
    @test VolSurfaceAnalysis._hash16(VolSurfaceAnalysis._canonical(d)) == core_hash(base)
end

@testset "identity: a policy trading another underlying is refused, not hashed" begin
    # `Experiment` is a public kwarg constructor, so a config is not the only
    # way to build one and the loader's one-underlying assertion is not the
    # only gate that matters. Identity projects the *clock* underlying's
    # contract facts; a policy trading a different one would be hashed
    # against the wrong multiplier -- a wrong run id, not a coarse one. The
    # id must not exist rather than be wrong.
    base = load_experiment_str(_id_toml())
    elsewhere = StaticAgent(DailyShortStrangle(underlying = Underlying("QQQ"),
                                               entry_time = Time(15, 45),
                                               expiry_interval = Day(1),
                                               put_delta = 0.2, call_delta = 0.2,
                                               quantity = 1))
    mismatched = Experiment(name="crossed", agent=elsewhere, data=base.data,
                            clock=base.clock, from=base.from, to=base.to)
    for f in (core_hash, full_hash)
        err = try f(mismatched); nothing catch e; e end
        @test err isa ErrorException
        @test occursin("QQQ", err.msg) && occursin("SPY", err.msg)
    end
    # and the agreeing spelling of the same experiment hashes.
    agreed = Experiment(name="crossed", agent=base.agent, data=base.data,
                        clock=base.clock, from=base.from, to=base.to)
    @test core_hash(agreed) isa String
end

@testset "identity: a clock that names no underlying is the runner's error" begin
    # There are no contract facts to resolve for a currency, and the failure
    # a reader should see is the one `run_experiment` gives for the same
    # experiment, not a `MethodError` out of `contract_spec`.
    base = load_experiment_str(_id_toml())
    on_rates = Experiment(name="rates", agent=base.agent, data=base.data,
                          clock=Clock{RateCurve}(Currency("USD")),
                          from=base.from, to=base.to)
    for f in (core_hash, full_hash)
        err = try f(on_rates); nothing catch e; e end
        @test err isa ErrorException
        @test occursin("must be an Underlying", err.msg) && occursin("rates", err.msg)
    end
end

@testset "identity: the bar-end convention is in the core projection" begin
    # Both parquet readers map a vendor minute bar to a record visible at
    # bar END. That determines which minute every decision reads, so it is
    # part of what the spec serves and must be in the hash -- and it forks
    # every id away from the runs made under bar-open visibility, which the
    # corrected code cannot reproduce.
    e = load_experiment_str(_id_toml())
    d = VolSurfaceAnalysis.to_dict(e.data)
    @test d["entries"]["option_bar"]["stamp"] == "bar_end"
    @test d["entries"]["spot_price"]["stamp"] == "bar_end"

    # Pinned against the prior core projection: the same resolved experiment
    # hashed without the stamp is what the pre-correction code produced, and
    # both hashes move.
    core = VolSurfaceAnalysis._core_dict(e)
    full = copy(core); full["outputs"] = VolSurfaceAnalysis.to_dict(e.outputs)
    prior = deepcopy(core)
    for k in ("option_bar", "spot_price")
        delete!(prior["data"]["entries"][k], "stamp")
    end
    prior_full = deepcopy(prior); prior_full["outputs"] = VolSurfaceAnalysis.to_dict(e.outputs)
    h(x) = VolSurfaceAnalysis._hash16(VolSurfaceAnalysis._canonical(x))
    @test h(core) == core_hash(e) && h(full) == full_hash(e)
    @test core_hash(e) != h(prior)
    @test full_hash(e) != h(prior_full)

    # Output variations still share the one corrected core.
    e2 = Experiment(name="x", agent=e.agent, data=e.data, clock=e.clock,
                    from=e.from, to=e.to, outputs=OutputSpec(metrics=[:sharpe]))
    @test core_hash(e2) == core_hash(e)
    @test full_hash(e2) != full_hash(e)
end

@testset "identity: no config key can restore bar-open visibility" begin
    # The convention is a constant in code, not a setting. A `stamp` key in
    # a parquet data table is an unknown key like any other -- ignored, not
    # honoured -- so the hash does not move and no experiment can select the
    # clock that leaks the future.
    base = load_experiment_str(_id_toml())
    with_key = load_experiment_str(replace(_id_toml(),
        "type = \"parquet_option_bars\"\nroot = \"/nonexistent/opts\"" =>
        "type = \"parquet_option_bars\"\nroot = \"/nonexistent/opts\"\nstamp = \"bar_open\""))
    @test core_hash(with_key) == core_hash(base)
    @test entry(with_key.data, OptionBar) == entry(base.data, OptionBar)
    # and there is no builder that would accept one either
    @test VolSurfaceAnalysis.to_dict(ParquetOptionBars("/x"))["stamp"] == "bar_end"
    @test VolSurfaceAnalysis.to_dict(ParquetSpots("/x"))["stamp"] == "bar_end"
end

@testset "identity: the marked curve moves full_hash and leaves core_hash alone" begin
    # Marks are output-side. A mark cannot alter an event, so an existing
    # ledger stays reusable and `core_hash` must not move; what did move is
    # the default artifact, the realised equity curve having become the
    # marked curve, and that lives in `OutputSpec`.
    base = load_experiment_str(_id_toml())
    old = Experiment(name="x", agent=base.agent, data=base.data, clock=base.clock,
                     from=base.from, to=base.to,
                     outputs=OutputSpec(artifacts=[:equity_curve]))
    @test base.outputs.artifacts == [:marked_curve]
    @test core_hash(base) == core_hash(old)
    @test full_hash(base) != full_hash(old)
    @test VolSurfaceAnalysis.to_dict(base.outputs)["artifacts"] == ["marked_curve"]

    # The core projection is exactly the pre-round set of keys: nothing the
    # outputs own has leaked into it, which is what keeps a stored ledger
    # addressable by the same core identity.
    d = VolSurfaceAnalysis._core_dict(base)
    @test Set(keys(d)) == Set(["from", "to", "data", "clock", "agent", "venue", "contract"])
    @test !haskey(d, "outputs")
    @test haskey(VolSurfaceAnalysis._full_dict(base), "outputs")
end

@testset "identity: metric params project as effective, not as spelled" begin
    # The projection is what the metric will actually run under. Naming a
    # parameter at its table default and omitting it are the same
    # experiment, so they are one run id; the override map as spelled is
    # not, because two spellings of one computation are not two results.
    # `[outputs]` goes last: the shared body ends in a table, so a new
    # top-level table can only follow it.
    mk(outputs) = load_experiment_str(_id_toml() * "\n" * outputs)

    omitted = mk("[outputs]\nmetrics = [\"sharpe\"]")
    explicit = mk("""
    [outputs]
    metrics = ["sharpe"]
    [outputs.metric_params.sharpe]
    periods_per_year = 252
    risk_free = 0.0
    """)
    partial = mk("""
    [outputs]
    metrics = ["sharpe"]
    [outputs.metric_params.sharpe]
    periods_per_year = 252
    """)
    @test full_hash(omitted) == full_hash(explicit)
    @test full_hash(omitted) == full_hash(partial)

    # A genuinely different value is a different output and still forks,
    # while the backtest underneath is untouched.
    moved = mk("""
    [outputs]
    metrics = ["sharpe"]
    [outputs.metric_params.sharpe]
    periods_per_year = 12
    """)
    @test full_hash(moved) != full_hash(omitted)
    @test core_hash(moved) == core_hash(omitted)

    # Overriding one key leaves the rest of that metric's defaults in the
    # projection, so a partial override cannot be read as "only this key".
    mp = VolSurfaceAnalysis.to_dict(moved.outputs)["metric_params"]
    @test mp["sharpe"] == Dict("periods_per_year" => 12, "risk_free" => 0.0)

    # One entry per metric the run will compute, emitted resolved -- a
    # metric with no parameters at all is present and empty. An override
    # for a metric this experiment does not compute reaches no result, so
    # it is not in the identity.
    plain = mk("[outputs]\nmetrics = [\"max_drawdown\"]")
    @test VolSurfaceAnalysis.to_dict(plain.outputs)["metric_params"] ==
          Dict("max_drawdown" => Dict())
    unused = mk("""
    [outputs]
    metrics = ["max_drawdown"]
    [outputs.metric_params.sharpe]
    periods_per_year = 12
    """)
    @test full_hash(unused) == full_hash(plain)

    # An unknown requested metric still hashes: naming it is
    # `compute_metrics`' failure at run time, not identity's.
    bogus = Experiment(name="x", agent=omitted.agent, data=omitted.data,
                       clock=omitted.clock, from=omitted.from, to=omitted.to,
                       outputs=OutputSpec(metrics=[:nonsense_metric],
                                          metric_params=Dict(:nonsense_metric => (a=1,))))
    @test length(full_hash(bogus)) == 16
    @test VolSurfaceAnalysis.to_dict(bogus.outputs)["metric_params"] ==
          Dict("nonsense_metric" => Dict("a" => 1))
end

@testset "identity: every table default is reachable through the projection" begin
    # One entry per registered metric, carrying that metric's table
    # defaults, so a default added to the table without thought about
    # identity shows up here rather than silently forking ids the next time
    # a config spells it out.
    e = load_experiment_str(_id_toml())          # outputs omitted -> all metrics
    mp = VolSurfaceAnalysis.to_dict(e.outputs)["metric_params"]
    for (sym, entry) in VolSurfaceAnalysis._METRIC_TABLE
        @test mp[string(sym)] ==
              Dict{String,Any}(string(k) => v for (k, v) in pairs(entry.defaults))
    end
end
