# Tests for the Experiment orchestrator.

const _EX_UND = Underlying("SPY")

function _ex_fixture()
    ts1 = DateTime(2024, 1, 15, 15, 30)
    ts2 = DateTime(2024, 1, 15, 15, 31)
    ts3 = DateTime(2024, 1, 15, 15, 32)
    spot = 480.0; r = 0.04; q = 0.015
    expiry = DateTime(2024, 2, 16, 21, 0)
    mk_quote(ts, strike, otype, bid, ask) = OptionQuote(
        "X", _EX_UND, expiry, strike, otype,
        bid, ask, (bid + ask) / 2, missing, missing, missing, ts,
    )
    mk_chain(ts) = [
        mk_quote(ts, 480.0, Call, 5.00, 5.10),
        mk_quote(ts, 480.0, Put,  4.80, 4.90),
    ]
    chains = Dict(ts1 => mk_chain(ts1), ts2 => mk_chain(ts2), ts3 => mk_chain(ts3))
    spots  = Dict(ts1 => spot,           ts2 => spot,           ts3 => spot)
    inner  = InMemoryDataSource(_EX_UND; chains=chains, spots=spots)
    mds    = ModelDataSource(inner; rate=FlatCurve(r), div=FlatCurve(q))
    (mds=mds, ts1=ts1, ts2=ts2, ts3=ts3, expiry=expiry, spot=spot)
end

# Policy that opens one long call at a chosen tick and does nothing else.
struct _ExOpenOnceAt <: Policy
    when::DateTime
    trade::Trade
end

function VolSurfaceAnalysis.decide(s::_ExOpenOnceAt, t::DateTime,
                                   ::TimeCutModelDataSource,
                                   ::AbstractVector{Position})::Vector{Trade}
    return t == s.when ? Trade[s.trade] : Trade[]
end

@testset "Experiment: kwarg constructor round-trips fields" begin
    f = _ex_fixture()
    exp = Experiment(name="smoke", agent=StaticAgent(NoOpPolicy()),
                     source=f.mds, from=f.ts1, to=f.ts3,
                     metrics=[:sharpe])
    @test exp.name == "smoke"
    @test exp.agent isa StaticAgent
    @test exp.source === f.mds
    @test exp.from == f.ts1
    @test exp.to == f.ts3
    @test exp.metrics == [:sharpe]
end

@testset "Experiment: default metrics is empty" begin
    f = _ex_fixture()
    exp = Experiment(name="default", agent=StaticAgent(NoOpPolicy()),
                     source=f.mds, from=f.ts1, to=f.ts3)
    @test exp.metrics == Symbol[]
end

@testset "run_experiment: NoOpPolicy -> empty result, provenance carried" begin
    f = _ex_fixture()
    exp = Experiment(name="noop", agent=StaticAgent(NoOpPolicy()),
                     source=f.mds, from=f.ts1, to=f.ts3)
    res = run_experiment(exp)
    @test res.experiment === exp
    @test isempty(res.positions)
    @test isempty(res.pnl_series.pnl)
    @test res.metrics.total_pnl == 0.0
    @test res.metrics.n_round_trips == 0
    @test isnan(res.metrics.hit_rate)
end

@testset "run_experiment: single-fill policy produces residual PnL at settlement" begin
    f = _ex_fixture()
    trd = Trade(_EX_UND, 480.0, f.expiry, Call)   # long call @ 5.10 ask
    exp = Experiment(name="single",
                     agent=StaticAgent(_ExOpenOnceAt(f.ts2, trd)),
                     source=f.mds, from=f.ts1, to=f.ts3)
    res = run_experiment(exp)
    @test length(res.positions) == 1
    @test length(res.pnl_series.pnl) == 1
    # spot at ts3 = 480 (ATM) -> payoff_unit = 0; cost_unit = 5.10 * +1 = 5.10
    # pnl = (0 - 5.10) * 1 = -5.10
    @test res.pnl_series.pnl[1] ≈ -5.10
    @test res.metrics.total_pnl ≈ -5.10
    @test res.pnl_series.timestamps[1] == f.ts3    # marked at window end
    @test res.pnl_series.settlement_spot == f.spot
end

@testset "run_experiment: requested optional metric appears in result" begin
    f = _ex_fixture()
    exp = Experiment(name="with-sharpe",
                     agent=StaticAgent(NoOpPolicy()),
                     source=f.mds, from=f.ts1, to=f.ts3,
                     metrics=[:sharpe, :max_drawdown])
    res = run_experiment(exp)
    @test haskey(res.metrics, :sharpe)
    @test haskey(res.metrics, :max_drawdown)
    @test isnan(res.metrics.sharpe)               # empty series
    @test res.metrics.max_drawdown == 0.0
end

@testset "run_experiment: unknown metric symbol errors" begin
    f = _ex_fixture()
    exp = Experiment(name="bogus", agent=StaticAgent(NoOpPolicy()),
                     source=f.mds, from=f.ts1, to=f.ts3,
                     metrics=[:nonsense_metric])
    @test_throws ErrorException run_experiment(exp)
end

@testset "run_experiment: empty time window errors" begin
    f = _ex_fixture()
    # Window strictly before any available ts.
    early = DateTime(2024, 1, 1, 0, 0)
    later = DateTime(2024, 1, 2, 0, 0)
    exp = Experiment(name="empty-window",
                     agent=StaticAgent(NoOpPolicy()),
                     source=f.mds, from=early, to=later)
    @test_throws ErrorException run_experiment(exp)
end

@testset "run_experiment: provenance allows rerun via result.experiment" begin
    f = _ex_fixture()
    trd = Trade(_EX_UND, 480.0, f.expiry, Call)
    exp = Experiment(name="rerun",
                     agent=StaticAgent(_ExOpenOnceAt(f.ts2, trd)),
                     source=f.mds, from=f.ts1, to=f.ts3)
    res1 = run_experiment(exp)
    res2 = run_experiment(res1.experiment)
    @test res1.metrics.total_pnl == res2.metrics.total_pnl
    @test length(res1.positions) == length(res2.positions)
end
