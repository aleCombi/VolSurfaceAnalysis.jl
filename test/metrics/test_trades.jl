# `trade_pnl`: the ledger's round trips as plain per-trade dollars.
# Scenarios come from test/ledger/fixtures.jl; the structure-level grouping
# is the one the trade metrics have always used.

@testset "trade_pnl: structure level gives one entry per (group, closed_at)" begin
    L, _ = _lg_case_split()
    t = trade_pnl(L)
    @test t isa Vector{Float64}
    @test t ≈ [140.0]
    @test total_pnl(t) ≈ 140.0
    @test n_round_trips(t) == 1
    @test hit_rate(t) == 1.0
    @test n_opens(L) == 2 && n_closes(L) == 1
end

@testset "trade_pnl: leg level gives one entry per round trip, losses first" begin
    L, _ = _lg_case_split()
    @test trade_pnl(L; unit=:leg) ≈ [50.0, 90.0]
end

@testset "trade_pnl: legs of one group closed at one instant are one trade" begin
    L = Ledger()
    g = mint_group!(L)
    _lg_fill!(L, _LG_PUT470,  Short, Open,  1, 0.85, g; at=_LG_T_OPEN,  leg_id=1)
    _lg_fill!(L, _LG_CALL490, Short, Open,  1, 1.10, g; at=_LG_T_OPEN,  leg_id=2)
    _lg_fill!(L, _LG_PUT470,  Long,  Close, 1, 1.00, g; at=_LG_T_CLOSE, leg_id=3)   # -15
    _lg_fill!(L, _LG_CALL490, Long,  Close, 1, 0.60, g; at=_LG_T_CLOSE, leg_id=4)   # +50
    @test trade_pnl(L) ≈ [35.0]
    @test trade_pnl(L; unit=:leg) ≈ [-15.0, 50.0]
    @test hit_rate(trade_pnl(L)) == 1.0
    @test hit_rate(trade_pnl(L; unit=:leg)) == 0.5
end

@testset "trade_pnl: canonical (closed_at, pnl) order" begin
    L = Ledger()
    g1 = mint_group!(L)
    g2 = mint_group!(L)
    _lg_fill!(L, _LG_PUT470,  Short, Open,  1, 0.85, g1; at=_LG_T_OPEN,  leg_id=1)
    _lg_fill!(L, _LG_CALL490, Short, Open,  1, 1.10, g2; at=_LG_T_OPEN,  leg_id=2)
    _lg_fill!(L, _LG_CALL490, Long,  Close, 1, 1.60, g2; at=_LG_T_OPEN2, leg_id=3)   # -50 first
    _lg_fill!(L, _LG_PUT470,  Long,  Close, 1, 0.40, g1; at=_LG_T_CLOSE, leg_id=4)   # +45 second
    @test trade_pnl(L) ≈ [-50.0, 45.0]
    @test hit_rate(trade_pnl(L)) == 0.5
    @test profit_factor(trade_pnl(L), nothing) ≈ 0.9
    @test total_pnl(trade_pnl(L)) ≈ -5.0
end

@testset "trade_pnl: an open lot contributes nothing; an expired lot does" begin
    L, _ = _lg_case_open_at_end()
    @test trade_pnl(L) ≈ [50.0]
    @test n_opens(L) == 2 && n_closes(L) == 1

    L, _ = _lg_case_mixed_expiries()
    @test trade_pnl(L) ≈ [-115.0]
    @test n_opens(L) == 2 && n_closes(L) == 0      # an expiry is not a closing fill
end

@testset "trade_pnl: an empty ledger, and an unknown unit" begin
    @test isempty(trade_pnl(Ledger()))
    @test n_opens(Ledger()) == 0 && n_closes(Ledger()) == 0
    err = try; trade_pnl(Ledger(); unit=:contract); nothing; catch e; e; end
    @test err isa ArgumentError
    @test occursin(":structure", err.msg) && occursin(":leg", err.msg)
    println("  refused unknown unit: ", err.msg)
end

@testset "trade_pnl: cents cross to dollars exactly once, after summing" begin
    # Two legs of one structure at +8913 and +4957 cents: summing in cents
    # and converting once is exact, converting each and adding is not
    # guaranteed to be.
    L, _ = _lg_case_fees()
    @test trade_pnl(L) ≈ [cents_to_usd(8913 + 4957)]
    @test trade_pnl(L; unit=:leg) ≈ [cents_to_usd(4957), cents_to_usd(8913)]
end
