# The optional metrics. The four path metrics are pure functions over a
# `MarkedCurve`; `profit_factor` is a pure function over per-trade dollars.
# Curves are built directly here so the metrics are tested apart from the
# marking that produces them.

_op_curve(profit::Vector{Float64}; unmarked=DateTime[]) = MarkedCurve(
    [DateTime(2024, 1, 2, 21, 0) + Day(i - 1) for i in 1:length(profit)],
    profit, unmarked, Symbol[:no_mark for _ in unmarked])

@testset "sharpe: the hand-computed answer on known session changes" begin
    # profit levels 0, 1, 3, 2, 6 -> session changes 1, 2, -1, 4.
    # mean = 6/4 = 1.5; deviations -0.5, 0.5, -2.5, 2.5; sum of squares 13;
    # corrected variance 13/3; sharpe = 1.5 / sqrt(13/3) * sqrt(252).
    c = _op_curve([0.0, 1.0, 3.0, 2.0, 6.0])
    @test session_changes(c) ≈ [1.0, 2.0, -1.0, 4.0]
    @test sharpe(Float64[], c) ≈ (1.5 / sqrt(13 / 3)) * sqrt(252)
    @test sharpe(Float64[], c; periods_per_year=1) ≈ 1.5 / sqrt(13 / 3)
end

@testset "sharpe: annualisation is by sessions, and it scales by sqrt" begin
    c = _op_curve([0.0, 1.0, 3.0, 6.0, 10.0])
    @test sharpe(Float64[], c; periods_per_year=4) / sharpe(Float64[], c; periods_per_year=1) ≈ 2.0
end

@testset "sharpe: capital cancels -- 1 and 100,000 give the same ratio" begin
    # Dividing every session's dollar change by a constant capital base is
    # exactly dividing the curve by it, and it scales the mean and the
    # standard deviation equally, so the zero-rate ratio is unchanged. That
    # is why capital is fixed at 1 and is not an argument here at all.
    profit = [0.0, 12.5, -4.0, 30.0, 18.0, 41.0]
    base = sharpe(Float64[], _op_curve(profit))
    @test sharpe(Float64[], _op_curve(profit ./ 100_000)) ≈ base
    @test sharpe(Float64[], _op_curve(profit ./ 1_000_000)) ≈ base
    # On a power-of-two base the float division is exact and so is the ratio.
    @test sharpe(Float64[], _op_curve(profit ./ 1024)) == base
    @test volatility(Float64[], _op_curve(profit ./ 1024)) ≈ volatility(Float64[], _op_curve(profit)) / 1024
    @test sortino(Float64[], _op_curve(profit ./ 1024)) == sortino(Float64[], _op_curve(profit))
end

@testset "sharpe: fewer than two changes or zero variance -> NaN" begin
    @test isnan(sharpe(Float64[], _op_curve(Float64[])))
    @test isnan(sharpe(Float64[], _op_curve([1.0])))
    @test isnan(sharpe(Float64[], _op_curve([1.0, 2.0])))          # one change
    @test isnan(sharpe(Float64[], _op_curve([0.0, 1.0, 2.0, 3.0])))  # constant changes
end

@testset "sharpe: risk_free subtracts per session" begin
    profit = [0.0, 1.0, 3.0, 4.0, 6.0]
    base = sharpe(Float64[], _op_curve(profit); periods_per_year=1, risk_free=0.0)
    rf   = sharpe(Float64[], _op_curve(profit); periods_per_year=1, risk_free=1.0)
    @test base > rf
end

@testset "sortino: hand-computed, and NaN with no downside" begin
    # changes 2, -1, 2, -1 -> mean 0.5; downside RMS sqrt((1+1)/4).
    c = _op_curve([0.0, 2.0, 1.0, 3.0, 2.0])
    @test session_changes(c) ≈ [2.0, -1.0, 2.0, -1.0]
    @test sortino(Float64[], c; periods_per_year=1) ≈ 0.5 / sqrt(0.5)
    @test isnan(sortino(Float64[], _op_curve([0.0, 1.0, 3.0, 6.0])))
end

@testset "volatility: annualised std of session changes" begin
    @test isnan(volatility(Float64[], _op_curve([1.0, 2.0])))
    c = _op_curve([0.0, 1.0, 3.0, 6.0, 10.0])          # changes 1, 2, 3, 4
    hand = sqrt(sum((x - 2.5)^2 for x in (1.0, 2.0, 3.0, 4.0)) / 3)
    @test volatility(Float64[], c; periods_per_year=4) ≈ hand * 2.0
end

@testset "max_drawdown: moves while a position is open" begin
    # The defect this round fixes: a book that is marked down and then closes
    # flat has a real drawdown, and a curve of closed trades reports zero.
    c = _op_curve([0.0, -25.0, -60.0, -10.0, 0.0])
    @test max_drawdown(Float64[], c) ≈ 60.0
    @test max_drawdown(Float64[], _op_curve([1.0, 3.0, 2.0, -2.0, 4.0])) ≈ 5.0
    @test max_drawdown(Float64[], _op_curve([1.0, 1.0, 1.0])) == 0.0
    @test max_drawdown(Float64[], _op_curve(Float64[])) == 0.0
    # The trade side of the same run: one round trip closing flat, no drawdown
    # anywhere in it. That is the number the old implementation reported.
    @test max_drawdown(Float64[], _op_curve([0.0, 0.0])) == 0.0
end

@testset "max_drawdown: an unmarked session costs an observation, not the peak" begin
    ts = [DateTime(2024, 1, 2, 21, 0) + Day(i - 1) for i in 1:4]
    c = MarkedCurve([ts[1], ts[2], ts[4]], [0.0, 50.0, 10.0], [ts[3]], [:no_mark])
    @test max_drawdown(Float64[], c) ≈ 40.0       # over what was seen; a lower bound
    @test n_unmarked(c) == 1           # and the curve says how much it could not see
    # The 1 -> 2 step is adjacent and survives; the 2 -> 4 step straddles the
    # break and is dropped rather than scaled over two sessions.
    @test session_changes(c) ≈ [50.0]
end

@testset "profit_factor: a trade metric on per-trade dollars" begin
    @test profit_factor([3.0, -1.0, 2.0, -1.0], nothing) ≈ 2.5
    @test profit_factor([1.0, 2.0], nothing) == Inf
    @test isnan(profit_factor(Float64[], nothing))
    @test isnan(profit_factor([0.0, 0.0], nothing))
end
