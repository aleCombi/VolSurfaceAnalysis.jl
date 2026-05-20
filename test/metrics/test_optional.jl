# Tests for the optional symbol-addressable metrics.

# Build a PnLSeries directly to keep tests independent of positions /
# the round-trip aggregation path -- the metric functions are pure
# functions over the series.
_op_series(pnl::Vector{Float64}) = PnLSeries(
    [DateTime(2024, 1, i, 16, 0) for i in 1:length(pnl)],
    copy(pnl),
    100.0,
    length(pnl),
    length(pnl),
)

@testset "sharpe: <2 trades or zero variance -> NaN" begin
    @test isnan(sharpe(_op_series(Float64[])))
    @test isnan(sharpe(_op_series([1.0])))
    @test isnan(sharpe(_op_series([1.0, 1.0, 1.0])))   # zero variance
end

@testset "sharpe: zero-mean series -> 0" begin
    @test sharpe(_op_series([1.0, -1.0, 1.0, -1.0])) ≈ 0.0 atol=1e-12
end

@testset "sharpe: annualization scales by sqrt(periods_per_year)" begin
    pnl = [1.0, 2.0, 3.0, 4.0]
    s1 = sharpe(_op_series(pnl); periods_per_year=1)
    s4 = sharpe(_op_series(pnl); periods_per_year=4)
    @test s4 / s1 ≈ 2.0           # sqrt(4) / sqrt(1)
end

@testset "sharpe: risk_free subtracts per-period" begin
    pnl = [2.0, 2.0, 2.0, 2.0]
    # Constant series -> NaN regardless of risk_free (zero variance).
    @test isnan(sharpe(_op_series(pnl); risk_free=0.5))
    # Non-constant; with risk_free=0 the mean is 1.5, with risk_free=ppy the
    # excess mean drops by exactly 1.
    pnl2 = [1.0, 2.0, 1.0, 2.0]
    base = sharpe(_op_series(pnl2); periods_per_year=1, risk_free=0.0)
    rf1  = sharpe(_op_series(pnl2); periods_per_year=1, risk_free=1.0)
    @test base > rf1
end

@testset "sortino: no downside -> NaN" begin
    @test isnan(sortino(_op_series([1.0, 2.0, 3.0])))
end

@testset "sortino: hand-computed" begin
    pnl = [2.0, -1.0, 2.0, -1.0]
    # excess (rf=0, ppy=1) = pnl. mean = 0.5.
    # downside = [-1, -1]; dd = sqrt((1 + 1) / 4) = sqrt(0.5)
    # sortino = 0.5 / sqrt(0.5) * sqrt(1) = sqrt(0.5)
    @test sortino(_op_series(pnl); periods_per_year=1) ≈ sqrt(0.5)
end

@testset "max_drawdown: known curve" begin
    # pnl = [1, 2, -5, 3, -3] -> equity = [1, 3, -2, 1, -2]
    # peaks =  [1, 3,  3, 3,  3]; drops = [0, 0, 5, 2, 5]; max = 5
    @test max_drawdown(_op_series([1.0, 2.0, -5.0, 3.0, -3.0])) ≈ 5.0
end

@testset "max_drawdown: monotonically increasing -> 0" begin
    @test max_drawdown(_op_series([1.0, 1.0, 1.0])) == 0.0
end

@testset "max_drawdown: empty -> 0" begin
    @test max_drawdown(_op_series(Float64[])) == 0.0
end

@testset "volatility: <2 -> NaN, matches std*sqrt(ppy) otherwise" begin
    @test isnan(volatility(_op_series([1.0])))
    pnl = [1.0, 2.0, 3.0, 4.0]
    using_std = sqrt(sum((x - 2.5)^2 for x in pnl) / 3)
    @test volatility(_op_series(pnl); periods_per_year=4) ≈ using_std * 2.0
end

@testset "profit_factor: wins and losses" begin
    @test profit_factor(_op_series([3.0, -1.0, 2.0, -1.0])) ≈ 2.5     # 5 / 2
    @test profit_factor(_op_series([1.0, 2.0])) == Inf                 # no losses
    @test isnan(profit_factor(_op_series(Float64[])))                 # nothing
    @test isnan(profit_factor(_op_series([0.0, 0.0])))                # all breakevens
end
