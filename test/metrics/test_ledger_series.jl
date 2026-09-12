# The adapter from the ledger to PnLSeries (case 11): the existing metrics
# run unchanged on a ledger. Scenarios come from test/ledger/fixtures.jl.

@testset "pnl_series(ledger): structure level gives one sample per (group, closed_at)" begin
    L, book = _lg_case_split()
    s = pnl_series(L)
    @test s isa PnLSeries
    @test length(s.pnl) == 1
    @test s.pnl[1] ≈ 140.0
    @test s.timestamps == [_LG_T_CLOSE]
    @test s.n_opens == 2
    @test s.n_closes == 1
    @test s.n_unmarked == 0
    @test isnan(s.window_end_spot)
    @test total_pnl(s) ≈ 140.0
    @test n_round_trips(s) == 1
    @test hit_rate(s) == 1.0
    @test equity_curve(s) ≈ [140.0]
end

@testset "pnl_series(ledger): leg level gives one sample per round trip, ordered by (timestamp, pnl)" begin
    L, _ = _lg_case_split()
    s = pnl_series(L; unit=:leg)
    @test length(s.pnl) == 2
    @test s.pnl ≈ [50.0, 90.0]                   # one instant: ascending pnl, losses first
    @test s.timestamps == [_LG_T_CLOSE, _LG_T_CLOSE]
    @test s.n_opens == 2 && s.n_closes == 1
    @test equity_curve(s) ≈ [50.0, 140.0]
end

@testset "pnl_series(ledger): ordering matches pnl_series(positions)" begin
    # two groups closed at different instants, the earlier one a loss
    L, book = Ledger(), Book()
    g1 = mint_group!(L)
    g2 = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470,  Short, Open,  1, 0.85, g1; at=_LG_T_OPEN,  leg_id=1)
    _lg_fill!(L, book, _LG_CALL490, Short, Open,  1, 1.10, g2; at=_LG_T_OPEN,  leg_id=2)
    _lg_fill!(L, book, _LG_CALL490, Long,  Close, 1, 1.60, g2; at=_LG_T_OPEN2, leg_id=3)   # -50 at T_OPEN2
    _lg_fill!(L, book, _LG_PUT470,  Long,  Close, 1, 0.40, g1; at=_LG_T_CLOSE, leg_id=4)   # +45 at T_CLOSE
    s = pnl_series(L)
    @test s.timestamps == [_LG_T_OPEN2, _LG_T_CLOSE]
    @test s.pnl ≈ [-50.0, 45.0]
    @test s.n_opens == 2 && s.n_closes == 2
    @test hit_rate(s) == 0.5
    @test profit_factor(s) ≈ 0.9
    @test total_pnl(s) ≈ -5.0
    @test pnl_series(L; unit=:leg).pnl ≈ s.pnl    # one leg per structure here
end

@testset "pnl_series(ledger): legs of one group closed at one instant are one sample" begin
    L, book = Ledger(), Book()
    g = mint_group!(L)
    _lg_fill!(L, book, _LG_PUT470,  Short, Open,  1, 0.85, g; at=_LG_T_OPEN,  leg_id=1)
    _lg_fill!(L, book, _LG_CALL490, Short, Open,  1, 1.10, g; at=_LG_T_OPEN,  leg_id=2)
    _lg_fill!(L, book, _LG_PUT470,  Long,  Close, 1, 1.00, g; at=_LG_T_CLOSE, leg_id=3)   # -15
    _lg_fill!(L, book, _LG_CALL490, Long,  Close, 1, 0.60, g; at=_LG_T_CLOSE, leg_id=4)   # +50
    @test pnl_series(L).pnl ≈ [35.0]
    @test pnl_series(L; unit=:leg).pnl ≈ [-15.0, 50.0]
    @test hit_rate(pnl_series(L)) == 1.0
    @test hit_rate(pnl_series(L; unit=:leg)) == 0.5
end

@testset "pnl_series(ledger): an open lot contributes nothing; an expired lot samples at its instant" begin
    L, _ = _lg_case_open_at_end()
    s = pnl_series(L)
    @test s.pnl ≈ [50.0]
    @test s.n_opens == 2 && s.n_closes == 1

    L, _ = _lg_case_mixed_expiries()
    s = pnl_series(L)
    @test s.timestamps == [_LG_EXPIRY_A]
    @test s.pnl ≈ [-115.0]
    @test s.n_opens == 2 && s.n_closes == 0
end

@testset "pnl_series(ledger): an empty ledger, and an unknown unit" begin
    s = pnl_series(Ledger())
    @test isempty(s.pnl) && isempty(s.timestamps)
    @test s.n_opens == 0 && s.n_closes == 0 && s.n_unmarked == 0
    @test isnan(s.window_end_spot)
    @test_throws ArgumentError pnl_series(Ledger(); unit=:contract)
end
