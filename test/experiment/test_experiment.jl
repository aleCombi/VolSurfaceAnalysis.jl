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
                     outputs=OutputSpec(metrics=[:sharpe]))
    @test exp.name == "smoke"
    @test exp.agent isa StaticAgent
    @test exp.source === f.mds
    @test exp.from == f.ts1
    @test exp.to == f.ts3
    @test exp.outputs.metrics == [:sharpe]
end

@testset "Experiment: default outputs = all registered metrics" begin
    f = _ex_fixture()
    exp = Experiment(name="default", agent=StaticAgent(NoOpPolicy()),
                     source=f.mds, from=f.ts1, to=f.ts3)
    @test Set(exp.outputs.metrics) ==
          Set([:sharpe, :sortino, :max_drawdown, :volatility, :profit_factor])
    @test exp.outputs.artifacts == [:equity_curve]
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

@testset "run_experiment: single-fill leg past window-end -> case 1 mark at window-end spot" begin
    # The leg's expiry (Feb 16) is past the window end (ts3 = Jan 15).
    # _build_settle case 1 returns window_end_spot = f.spot. Timestamp is
    # stamped at the leg's expiry, not at the window end (the new contract:
    # residual entries are honestly stamped at the leg's own expiry).
    f = _ex_fixture()
    trd = Trade(_EX_UND, 480.0, f.expiry, Call)
    exp = Experiment(name="case1-mark",
                     agent=StaticAgent(_ExOpenOnceAt(f.ts2, trd)),
                     source=f.mds, from=f.ts1, to=f.ts3)
    res = run_experiment(exp)
    @test length(res.positions) == 1
    @test length(res.pnl_series.pnl) == 1
    # ATM payoff at window-end spot 480 = 0; cost = ask 5.10. PnL = -5.10.
    @test res.pnl_series.pnl[1] ≈ -5.10
    @test res.metrics.total_pnl ≈ -5.10
    @test res.pnl_series.timestamps[1] == f.expiry   # stamped at leg's own expiry
    @test res.pnl_series.window_end_spot == f.spot
    @test res.pnl_series.n_unmarked == 0
end

@testset "run_experiment: held-to-expiry leg inside window settles at expiry spot (case 2)" begin
    # Build a tiny fixture whose source has a real bar at the leg's expiry,
    # so `get_spot(source, expiry)` returns a number and case 2 succeeds.
    ts1 = DateTime(2024, 1, 15, 15, 30)
    ts2 = DateTime(2024, 1, 15, 15, 31)
    ts3 = DateTime(2024, 1, 15, 15, 32)
    expiry = ts3                                       # leg expires at window end
    spot   = 480.0
    mk_q(ts, K) = OptionQuote("X", _EX_UND, expiry, K, Call,
                              5.00, 5.10, 5.05, missing, missing, missing, ts)
    chains = Dict(ts1 => [mk_q(ts1, 480.0)],
                  ts2 => [mk_q(ts2, 480.0)],
                  ts3 => [mk_q(ts3, 480.0)])
    spots  = Dict(ts1 => spot, ts2 => spot, ts3 => spot)
    inner  = InMemoryDataSource(_EX_UND; chains=chains, spots=spots)
    mds    = ModelDataSource(inner; rate=FlatCurve(0.04), div=FlatCurve(0.015))

    trd = Trade(_EX_UND, 480.0, expiry, Call)
    exp = Experiment(name="case2-held",
                     agent=StaticAgent(_ExOpenOnceAt(ts2, trd)),
                     source=mds, from=ts1, to=ts3)
    res = run_experiment(exp)
    @test length(res.positions) == 1
    @test length(res.pnl_series.pnl) == 1
    @test res.pnl_series.timestamps[1] == expiry
    @test res.pnl_series.pnl[1] ≈ -5.10                # ATM 480 payoff = 0; cost = 5.10
    @test res.pnl_series.n_unmarked == 0
end

@testset "run_experiment: requested optional metric appears in result" begin
    f = _ex_fixture()
    exp = Experiment(name="with-sharpe",
                     agent=StaticAgent(NoOpPolicy()),
                     source=f.mds, from=f.ts1, to=f.ts3,
                     outputs=OutputSpec(metrics=[:sharpe, :max_drawdown]))
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
                     outputs=OutputSpec(metrics=[:nonsense_metric]))
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

# ---- DailyShortStrangle e2e ------------------------------------------------

# Multi-strike, two-expiry fixture priced from flat 20% BS so the surface
# inverts cleanly and `invert_delta` has a wide observed bracket.
function _strangle_ex_fixture()
    entry_ts = DateTime(2024, 6, 3, 15, 45)             # the entry tick
    pre_ts   = DateTime(2024, 6, 3, 15, 44)             # one minute before
    end_ts   = DateTime(2024, 6, 3, 15, 46)             # window end
    spot, r, q, sigma = 480.0, 0.045, 0.013, 0.20
    e_target = DateTime(2024, 6, 4, 20, 0)              # first expiry on/after entry+1d
    e_far    = DateTime(2024, 6, 7, 20, 0)
    strikes = 440.0:5.0:520.0

    function mk_q(ts, K, expiry, otype)
        T = time_to_expiry(expiry, ts)
        mark = bs_price(spot, K, T, sigma, otype; r=r, q=q)
        spread = max(0.02, 0.01 * mark)
        OptionQuote("X", _EX_UND, expiry, K, otype,
                    mark - spread / 2, mark + spread / 2, mark,
                    missing, missing, missing, ts)
    end

    mk_chain(ts) = OptionQuote[
        mk_q(ts, K, e, K >= spot ? Call : Put)
        for K in strikes, e in (e_target, e_far)
    ] |> vec
    chains = Dict(pre_ts => mk_chain(pre_ts),
                  entry_ts => mk_chain(entry_ts),
                  end_ts  => mk_chain(end_ts))
    spots  = Dict(pre_ts => spot, entry_ts => spot, end_ts => spot)
    inner  = InMemoryDataSource(_EX_UND; chains=chains, spots=spots)
    mds    = ModelDataSource(inner; rate=FlatCurve(r), div=FlatCurve(q))
    (mds=mds, pre_ts=pre_ts, entry_ts=entry_ts, end_ts=end_ts,
     spot=spot, e_target=e_target)
end

@testset "run_experiment: DailyShortStrangle opens two legs at entry tick" begin
    f = _strangle_ex_fixture()
    policy = DailyShortStrangle(; underlying=_EX_UND,
                                entry_time=Time(15, 45),
                                expiry_interval=Day(1),
                                put_delta=0.20, call_delta=0.20,
                                quantity=1.0)
    exp = Experiment(name="strangle-e2e",
                     agent=StaticAgent(policy),
                     source=f.mds, from=f.pre_ts, to=f.end_ts)
    res = run_experiment(exp)

    # The gate fires exactly once over [pre_ts, end_ts] -> 2 short legs opened.
    @test length(res.positions) == 2
    @test all(p.trade.direction == -1 for p in res.positions)
    @test all(p.trade.expiry == f.e_target for p in res.positions)

    # e_target = 2024-06-04 is past end_ts = 2024-06-03 -> case 1 (mark at
    # window-end spot). Both legs settle to a finite PnL, stamped at their
    # own expiry, and n_unmarked stays 0.
    @test res.pnl_series.n_unmarked == 0
    @test length(res.pnl_series.pnl) == 2
    @test all(ts == f.e_target for ts in res.pnl_series.timestamps)
    @test isfinite(res.metrics.total_pnl)
end
