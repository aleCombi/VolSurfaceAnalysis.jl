# The cash rules, pinned with an explicit spec and resolved from the table.
# Cash is whole USD cents: 0.85 per share on a 100-multiplier contract is
# 0.85 * 100 * 100 = 8500 cents.

@testset "cash: intrinsic per share" begin
    @test intrinsic(_LG_PUT470, 468.0) == 2.0
    @test intrinsic(_LG_PUT470, 470.0) == 0.0
    @test intrinsic(_LG_PUT470, 475.0) == 0.0
    @test intrinsic(_LG_CALL490, 495.0) == 5.0
    @test intrinsic(_LG_CALL490, 490.0) == 0.0
    @test intrinsic(_LG_CALL490, 480) == 0.0
end

@testset "cash: contract_cents is the one rounding point" begin
    cents = VolSurfaceAnalysis.contract_cents
    @test cents(0.85, _LG_SPEC) == 8500                       # 0.85 * 100 * 100
    @test cents(0.07, _LG_SPEC) == 700
    @test cents(2, _LG_SPEC) == 20000
    @test cents(0.0, _LG_SPEC) == 0
    @test cents(0.0001, _LG_SPEC) == 1                        # a sub-penny price that is whole cents per contract
    @test cents(0.85, ContractSpec(10, American, PMSettled, Physical)) == 850
    @test cents(intrinsic(_LG_PUT470, 469.83), _LG_SPEC) == 1700   # float noise on 470 - 469.83 is absorbed
    @test cents(0.85, _LG_SPEC) isa Int
    @test_throws NonIntegralCash cents(0.123456, _LG_SPEC)    # 1234.56 cents per contract
    @test_throws NonIntegralCash cents(0.00005, _LG_SPEC)     # half a cent per contract
    @test_throws NonIntegralCash cents(Inf, _LG_SPEC)         # not whole cents either: the named failure, not InexactError
    @test_throws NonIntegralCash cents(NaN, _LG_SPEC)
    @test_throws NonIntegralCash cents(1e300, _LG_SPEC)       # beyond what an Int holds
    err = try cents(0.123456, _LG_SPEC); nothing catch e; e end
    @test err isa NonIntegralCash && err.value ≈ 1234.56
    @test occursin("NonIntegralCash", sprint(showerror, err))
end

@testset "cash: a fill is minus side times contract cents times quantity" begin
    h = EventHeader(1, _LG_T_OPEN, _LG_T_OPEN, 1)
    short = Fill(h, 1, 1, 1, _LG_PUT470, Short, Open,  2, 0.85, :cross_spread)
    long  = Fill(h, 1, 1, 1, _LG_PUT470, Long,  Close, 2, 0.40, :cross_spread)
    @test cash(short, _LG_SPEC) == 17000                      # +8500 * 2
    @test cash(long,  _LG_SPEC) == -8000                      # -4000 * 2
    @test cash(short, ContractSpec(10, American, PMSettled, Physical)) == 1700   # 850 * 2
    @test cash(short) == 17000                                # resolved from the SPY row
    @test cash(short) isa Int
end

@testset "cash: a match moves no cash" begin
    m = Match(EventHeader(3, _LG_T_CLOSE, _LG_T_CLOSE, 3), 1, 1, 2, 2)
    @test cash(m, _LG_SPEC) == 0
    @test cash(m) == 0
end

@testset "cash: an expiry is side times contract cents of intrinsic times quantity" begin
    h = EventHeader(4, _LG_EXPIRY_A, _LG_T_NEXT, 4)
    itm      = Expiry(h, 1, 1, _LG_PUT470, Short, 1, 468.0, CashSettled)
    otm      = Expiry(h, 1, 1, _LG_PUT470, Short, 1, 475.0, Worthless)
    long_itm = Expiry(h, 1, 1, _LG_PUT470, Long,  3, 468.0, CashSettled)
    @test cash(itm, _LG_SPEC) == -20000                       # -(2.00 * 100 * 100) * 1
    @test cash(otm, _LG_SPEC) == 0
    @test cash(long_itm, _LG_SPEC) == 60000                   # +20000 * 3
    @test cash(itm) == -20000
    @test cash(itm, ContractSpec(50, European, AMSettled, Cash)) == -10000   # -(2.00 * 50 * 100)
end

@testset "cash: a fee is its amount in cents" begin
    fee = Fee(EventHeader(5, _LG_T_CLOSE, _LG_T_CLOSE, 5), 2, -130)   # 1.30 USD
    @test cash(fee, _LG_SPEC) == -130
    @test cash(fee) == -130
end

@testset "cash: an unlisted underlying needs an explicit spec" begin
    spx = ContractKey(Underlying("SPX"), 4700.0, _LG_EXPIRY_A, Put)
    f = Fill(EventHeader(1, _LG_T_OPEN, _LG_T_OPEN, 1), 1, 1, 1, spx, Short, Open, 1, 10.0, :cross_spread)
    @test_throws UnknownContract cash(f)
    @test cash(f, ContractSpec(100, European, AMSettled, Cash)) == 100000   # 10.00 * 100 * 100
end
