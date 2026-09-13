# Tests for the PnLSeries intermediate and the equity_curve helper. The
# series is built from a ledger by `pnl_series(::Ledger)`, tested in
# test_ledger_series.jl with the canonical-order promise; here the struct
# and its one derived view.

@testset "PnLSeries: construction carries samples, counts and the two placeholders" begin
    ts = [DateTime(2024, 1, 15, 16, 0), DateTime(2024, 1, 16, 16, 0)]
    s = PnLSeries(ts, [1.5, -0.5], NaN, 2, 2, 0)
    @test s.timestamps == ts
    @test s.pnl == [1.5, -0.5]
    @test isnan(s.window_end_spot)               # placeholder until slice 5
    @test s.n_opens == 2
    @test s.n_closes == 2
    @test s.n_unmarked == 0                      # placeholder until slice 5
    @test !ismutable(s)
    empty = PnLSeries(DateTime[], Float64[], NaN, 0, 0, 0)
    @test isempty(empty.timestamps) && isempty(empty.pnl)
end

@testset "equity_curve: cumsum of pnl, empty stays empty" begin
    @test isempty(equity_curve(PnLSeries(DateTime[], Float64[], NaN, 0, 0, 0)))
    ts = [DateTime(2024, 1, 15, 16, 0) + Day(i) for i in 0:3]
    s = PnLSeries(ts, [1.0, -2.0, 0.5, 3.0], NaN, 4, 4, 0)
    @test equity_curve(s) == [1.0, -1.0, -0.5, 2.5]
    @test equity_curve(s) ≈ cumsum(s.pnl)
    @test length(equity_curve(s)) == length(s.pnl)
end
