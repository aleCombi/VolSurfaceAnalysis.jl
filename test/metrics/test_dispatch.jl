# `compute_metrics`: the two honest inputs, the per-metric input column of
# the dispatch table, and what happens when the curve is unavailable.

const _ALWAYS_ON_KEYS = (:total_pnl, :n_round_trips, :n_opens, :n_closes, :hit_rate)

_di_curve(profit::Vector{Float64}) = MarkedCurve(
    [DateTime(2024, 1, 2, 21, 0) + Day(i - 1) for i in 1:length(profit)],
    profit, DateTime[], Symbol[])

@testset "compute_metrics: empty requested -> exactly the always-on keys" begin
    L, _ = _lg_case_strangle_closed()
    out = compute_metrics(L, _di_curve([0.0, 1.0, 3.0]), Symbol[])
    @test keys(out) == _ALWAYS_ON_KEYS
    @test out.total_pnl ≈ total_pnl(trade_pnl(L))
    @test out.n_round_trips == 1       # one structure, both legs closed at once
    @test out.n_opens == n_opens(L) == 2
    @test out.n_closes == n_closes(L) == 2
end

@testset "compute_metrics: the table says which input each metric consumes" begin
    L, _ = _lg_case_strangle_closed()
    c = _di_curve([0.0, 1.0, 3.0, 2.0, 6.0])
    out = compute_metrics(L, c, [:sharpe, :profit_factor, :max_drawdown])
    @test keys(out) == (_ALWAYS_ON_KEYS..., :sharpe, :profit_factor, :max_drawdown)
    @test out.sharpe ≈ sharpe(Float64[], c)                             # curve metric
    @test out.max_drawdown ≈ max_drawdown(Float64[], c)                 # curve metric
    @test out.profit_factor ≈ profit_factor(trade_pnl(L), nothing)    # trade metric
    # Every entry is (fn, defaults) and nothing else: the table records no
    # per-metric input, because every metric takes both.
    for (sym, entry) in VolSurfaceAnalysis._METRIC_TABLE
        @test propertynames(entry) == (:fn, :defaults)
    end
end

@testset "every optional metric accepts both inputs" begin
    L, _ = _lg_case_strangle_closed()
    c = _di_curve([0.0, 1.0, 3.0, 2.0, 6.0])
    trades = trade_pnl(L)
    # Uniform arity is the contract that lets the dispatcher hand both to
    # every entry without knowing which one is read.
    for (sym, entry) in VolSurfaceAnalysis._METRIC_TABLE
        @test entry.fn(trades, c; entry.defaults...) isa Float64
    end
end

@testset "compute_metrics: requested metrics appear in request order" begin
    L, _ = _lg_case_strangle_closed()
    c = _di_curve([0.0, 1.0, 3.0, 2.0])
    @test keys(compute_metrics(L, c, [:max_drawdown, :profit_factor])) ==
          (_ALWAYS_ON_KEYS..., :max_drawdown, :profit_factor)
    @test keys(compute_metrics(L, c, [:profit_factor, :max_drawdown])) ==
          (_ALWAYS_ON_KEYS..., :profit_factor, :max_drawdown)
end

@testset "compute_metrics: unknown symbol errors loudly" begin
    L = Ledger()
    err = try; compute_metrics(L, _di_curve([1.0, 2.0]), [:nonsense]); nothing; catch e; e; end
    @test err isa ErrorException
    @test occursin("nonsense", err.msg)
    @test occursin("Known", err.msg)
    println("  refused unknown metric: ", err.msg)
end

@testset "compute_metrics: default kwargs apply, partial overrides merge" begin
    L, _ = _lg_case_strangle_closed()
    c = _di_curve([0.0, 1.0, 3.0, 2.0, 6.0])
    @test compute_metrics(L, c, [:sharpe]).sharpe ≈
          sharpe(Float64[], c; periods_per_year=252, risk_free=0.0)
    over = Dict(:sharpe => (periods_per_year=4,))
    @test compute_metrics(L, c, [:sharpe]; kwargs=over).sharpe ≈
          sharpe(Float64[], c; periods_per_year=4, risk_free=0.0)
end

@testset "compute_metrics: no curve omits every optional metric, never NaNs them" begin
    L, _ = _lg_case_strangle_closed()
    out = compute_metrics(L, nothing, [:sharpe, :profit_factor, :max_drawdown, :volatility])
    # The always-on core is a function of the ledger alone and survives.
    @test keys(out) == _ALWAYS_ON_KEYS
    @test out.total_pnl ≈ total_pnl(trade_pnl(L))
    @test out.hit_rate ≈ hit_rate(trade_pnl(L))
    # An absent key says "not computed"; a NaN would claim "computed, and
    # undefined", which is a different and false answer.
    @test !haskey(out, :sharpe)
    @test !haskey(out, :max_drawdown)
    @test !haskey(out, :volatility)
    # Dropped wholesale: the table does not record which metrics need the
    # curve, so a trade metric goes with them even though it does not.
    @test !haskey(out, :profit_factor)
end

@testset "compute_metrics: an unknown symbol still errors with no curve" begin
    L = Ledger()
    err = try; compute_metrics(L, nothing, [:nonsense]); nothing; catch e; e; end
    @test err isa ErrorException
    @test occursin("nonsense", err.msg)
    println("  refused unknown metric with no curve: ", err.msg)
end

@testset "compute_metrics: an empty ledger stays consistent" begin
    L = Ledger()
    out = compute_metrics(L, _di_curve(Float64[]), [:sharpe, :max_drawdown, :profit_factor])
    @test out.total_pnl == 0.0
    @test out.n_round_trips == 0
    @test out.n_opens == 0 && out.n_closes == 0
    @test isnan(out.hit_rate)
    @test isnan(out.sharpe)
    @test out.max_drawdown == 0.0
    @test isnan(out.profit_factor)
end

@testset "PnLSeries and its placeholders are gone" begin
    @test !isdefined(VolSurfaceAnalysis, :PnLSeries)
    @test !isdefined(VolSurfaceAnalysis, :pnl_series)
    @test !isdefined(VolSurfaceAnalysis, :equity_curve)
    # What replaced them, and the home the two counts now have.
    @test isdefined(VolSurfaceAnalysis, :MarkedCurve)
    @test isdefined(VolSurfaceAnalysis, :trade_pnl)
    @test !any(f -> :window_end_spot in f, (fieldnames(MarkedCurve),))
    @test fieldnames(MarkedCurve) ==
          (:timestamps, :profit, :unmarked_at, :unmarked_reason)
end
