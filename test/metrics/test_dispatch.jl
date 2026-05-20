# Tests for compute_metrics symbol dispatch.

_di_series(pnl::Vector{Float64}) = PnLSeries(
    [DateTime(2024, 1, i, 16, 0) for i in 1:length(pnl)],
    copy(pnl),
    100.0,
    length(pnl),
    length(pnl),
)

const _ALWAYS_ON_KEYS = (:total_pnl, :n_round_trips, :n_opens, :n_closes, :hit_rate)

@testset "compute_metrics: empty requested -> exactly the always-on keys" begin
    s = _di_series([1.0, -1.0, 2.0])
    out = compute_metrics(s, Symbol[])
    @test keys(out) == _ALWAYS_ON_KEYS
    @test out.total_pnl ≈ 2.0
    @test out.n_round_trips == 3
end

@testset "compute_metrics: requested metrics appear in request order" begin
    s = _di_series([1.0, -0.5, 1.5, -0.25])
    out = compute_metrics(s, [:max_drawdown, :profit_factor])
    @test keys(out) == (_ALWAYS_ON_KEYS..., :max_drawdown, :profit_factor)

    out2 = compute_metrics(s, [:profit_factor, :max_drawdown])
    @test keys(out2) == (_ALWAYS_ON_KEYS..., :profit_factor, :max_drawdown)
end

@testset "compute_metrics: unknown symbol errors loudly" begin
    s = _di_series([1.0, -1.0])
    err = try
        compute_metrics(s, [:nonsense])
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("nonsense", err.msg)
    @test occursin("Known", err.msg)
end

@testset "compute_metrics: default kwargs apply when no override given" begin
    s = _di_series([1.0, 2.0, 3.0, 4.0])
    out = compute_metrics(s, [:sharpe])
    @test out.sharpe ≈ sharpe(s; periods_per_year=252, risk_free=0.0)
end

@testset "compute_metrics: per-metric kwargs override merges with defaults" begin
    s = _di_series([1.0, 2.0, 3.0, 4.0])
    overrides = Dict(:sharpe => (periods_per_year=1,))
    out = compute_metrics(s, [:sharpe]; kwargs=overrides)
    @test out.sharpe ≈ sharpe(s; periods_per_year=1, risk_free=0.0)

    # Partial override: change ppy, keep default risk_free.
    overrides2 = Dict(:sharpe => (periods_per_year=4,))
    out2 = compute_metrics(s, [:sharpe]; kwargs=overrides2)
    @test out2.sharpe ≈ sharpe(s; periods_per_year=4, risk_free=0.0)
end

@testset "compute_metrics: empty series stays consistent" begin
    s = _di_series(Float64[])
    out = compute_metrics(s, [:sharpe, :max_drawdown, :profit_factor])
    @test out.total_pnl == 0.0
    @test out.n_round_trips == 0
    @test isnan(out.hit_rate)
    @test isnan(out.sharpe)
    @test out.max_drawdown == 0.0
    @test isnan(out.profit_factor)
end
