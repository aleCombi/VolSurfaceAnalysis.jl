# The always-on core metrics. Three are pure functions over per-trade
# dollars, two are counts over the ledger's own events.

@testset "core metrics: nothing closed" begin
    t = Float64[]
    @test total_pnl(t) == 0.0
    @test n_round_trips(t) == 0
    @test isnan(hit_rate(t))
end

@testset "core metrics: all winners, all losers, mixed" begin
    @test total_pnl([1.0]) ≈ 1.0 && hit_rate([1.0]) == 1.0
    @test total_pnl([-1.0]) ≈ -1.0 && hit_rate([-1.0]) == 0.0
    t = [1.0, -0.5]
    @test total_pnl(t) ≈ 0.5
    @test n_round_trips(t) == 2
    @test hit_rate(t) == 0.5
end

@testset "core metrics: a breakeven trade is not a win" begin
    @test total_pnl([0.0]) == 0.0
    @test n_round_trips([0.0]) == 1
    @test hit_rate([0.0]) == 0.0
end

@testset "total_pnl stays the realised total, not the curve's last level" begin
    L, _ = _lg_case_open_at_end()
    trades = trade_pnl(L)
    @test total_pnl(trades) ≈ 50.0            # the one closed structure, and only it
    c = marked_curve(L, _mk_data(), _MK_UND, _MK_FROM, _MK_TO).curve
    # The open put is worth something at every session close, so the curve's
    # last level is a different number by construction. Widening total_pnl to
    # mean that is exactly what decision 8 forbids.
    @test !(c.profit[end] ≈ total_pnl(trades))
end

@testset "n_opens / n_closes: counts of the ledger's own fills" begin
    L, _ = _lg_case_strangle_closed()
    @test n_opens(L) == 2
    @test n_closes(L) == 2
    @test n_opens(Ledger()) == 0 && n_closes(Ledger()) == 0
    # They are functions of the ledger, so they survive the wrapper that used
    # to carry them as fields.
    @test applicable(n_opens, L) && applicable(n_closes, L)
end
