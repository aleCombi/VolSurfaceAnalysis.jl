# The cash rules, pinned with an explicit spec and resolved from the table.

@testset "cash: intrinsic per share" begin
    @test intrinsic(_LG_PUT470, 468.0) == 2.0
    @test intrinsic(_LG_PUT470, 470.0) == 0.0
    @test intrinsic(_LG_PUT470, 475.0) == 0.0
    @test intrinsic(_LG_CALL490, 495.0) == 5.0
    @test intrinsic(_LG_CALL490, 490.0) == 0.0
    @test intrinsic(_LG_CALL490, 480) == 0.0
end

@testset "cash: a fill is minus side times price times quantity times multiplier" begin
    h = EventHeader(1, _LG_T_OPEN, _LG_T_OPEN, 1)
    short = Fill(h, 1, 1, 1, _LG_PUT470, Short, Open,  2, 0.85, :cross_spread)
    long  = Fill(h, 1, 1, 1, _LG_PUT470, Long,  Close, 2, 0.40, :cross_spread)
    @test cash(short, _LG_SPEC) ≈ 170.0
    @test cash(long,  _LG_SPEC) ≈ -80.0
    @test cash(short, ContractSpec(10.0, American, PMSettled, Physical)) ≈ 17.0
    @test cash(short) ≈ 170.0                       # resolved from the SPY row
end

@testset "cash: a match moves no cash" begin
    m = Match(EventHeader(3, _LG_T_CLOSE, _LG_T_CLOSE, 3), 1, 1, 2, 2)
    @test cash(m, _LG_SPEC) == 0.0
    @test cash(m) == 0.0
end

@testset "cash: an expiry is side times intrinsic times quantity times multiplier" begin
    h = EventHeader(4, _LG_EXPIRY_A, _LG_T_NEXT, 4)
    itm      = Expiry(h, 1, 1, _LG_PUT470, Short, 1, 468.0, CashSettled)
    otm      = Expiry(h, 1, 1, _LG_PUT470, Short, 1, 475.0, Worthless)
    long_itm = Expiry(h, 1, 1, _LG_PUT470, Long,  3, 468.0, CashSettled)
    @test cash(itm, _LG_SPEC) ≈ -200.0
    @test cash(otm, _LG_SPEC) == 0.0
    @test cash(long_itm, _LG_SPEC) ≈ 600.0
    @test cash(itm) ≈ -200.0
    @test cash(itm, ContractSpec(50.0, European, AMSettled, Cash)) ≈ -100.0
end

@testset "cash: a fee is its amount" begin
    fee = Fee(EventHeader(5, _LG_T_CLOSE, _LG_T_CLOSE, 5), 2, -1.30)
    @test cash(fee, _LG_SPEC) == -1.30
    @test cash(fee) == -1.30
end

@testset "cash: an unlisted underlying needs an explicit spec" begin
    spx = ContractKey(Underlying("SPX"), 4700.0, _LG_EXPIRY_A, Put)
    f = Fill(EventHeader(1, _LG_T_OPEN, _LG_T_OPEN, 1), 1, 1, 1, spx, Short, Open, 1, 10.0, :cross_spread)
    @test_throws UnknownContract cash(f)
    @test cash(f, ContractSpec(100.0, European, AMSettled, Cash)) ≈ 1000.0
end
