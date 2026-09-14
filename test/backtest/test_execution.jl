# The simulated venue: the price rule table and the cost model table.
# Cash literals are whole USD cents. Needs test/ledger/fixtures.jl for
# _LG_SPEC.

@testset "fill_price(:cross_spread): the side the trader crosses to, on the tick" begin
    # Long takes the ask, Short the bid
    @test fill_price(:cross_spread, 1.02, 1.065, Long,  1) == 1.07   # ask 1.065 rounded up
    @test fill_price(:cross_spread, 1.02, 1.065, Short, 1) == 1.02   # bid 1.02, on the tick
    # a synthesized quote off the tick: a sale rounds down, a buy rounds up
    @test fill_price(:cross_spread, 0.827261, 0.847261, Short, 1) == 0.82
    @test fill_price(:cross_spread, 0.827261, 0.847261, Long,  1) == 0.85
    # whole-cent quotes are unchanged: 1.07 * 100 is 107.00000000000001 in binary
    @test fill_price(:cross_spread, 1.02, 1.07, Long,  1) == 1.07
    @test fill_price(:cross_spread, 1.02, 1.07, Short, 1) == 1.02
    @test fill_price(:cross_spread, 0.85, 1.10, Short, 1) == 0.85
    @test fill_price(:cross_spread, 4.80, 4.90, Long,  1) == 4.90
    @test fill_price(:cross_spread, 5.00, 5.10, Short, 1) == 5.00
    # a wider tick
    @test fill_price(:cross_spread, 1.06, 1.06, Long,  5) == 1.10
    @test fill_price(:cross_spread, 1.06, 1.06, Short, 5) == 1.05
    @test fill_price(:cross_spread, 1.05, 1.10, Long,  5) == 1.10
    # the required side missing returns missing; the other side missing is fine
    @test ismissing(fill_price(:cross_spread, 1.02, missing, Long, 1))
    @test ismissing(fill_price(:cross_spread, missing, 1.07, Short, 1))
    @test fill_price(:cross_spread, missing, 1.065, Long,  1) == 1.07
    @test fill_price(:cross_spread, 1.02, missing, Short, 1) == 1.02
    # every result passes the ledger's one rounding point for a listed underlying
    for (bid, ask) in ((1.02, 1.065), (0.827261, 0.847261), (1.07, 1.07), (0.055, 0.065), (123.456789, 123.987654)),
        side in (Long, Short), tick in (1, 5)
        p = fill_price(:cross_spread, bid, ask, side, tick)
        @test VolSurfaceAnalysis.contract_cents(p, _LG_SPEC) == round(Int, p * 10000)
        @test p > 0
        # rounded away from the trader: a buy never below the ask, a sale never above the bid
        @test side == Long ? p >= ask - 1e-9 : p <= bid + 1e-9
    end
    # a bid below the tick floors to zero: the rule is arithmetic, and the
    # ledger then refuses the fill as InvalidPrice, so no such fill lands
    @test fill_price(:cross_spread, 0.005, 0.015, Short, 1) == 0.0
    @test fill_price(:cross_spread, 0.005, 0.015, Long,  1) == 0.02
    # an unknown rule errors naming the known ones
    err = try fill_price(:mid, 1.02, 1.07, Long, 1); nothing catch e; e end
    @test err isa ErrorException
    @test occursin("cross_spread", err.msg) && occursin("mid", err.msg)
    @test_throws ArgumentError fill_price(:cross_spread, 1.02, 1.07, Long, 0)
    # :broker_execution is not a rule of ours
    @test_throws ErrorException fill_price(:broker_execution, 1.02, 1.07, Long, 1)
end

@testset "commission(:ibkr_pro_us_options): the page's examples, cents per leg" begin
    # https://www.interactivebrokers.com/en/pricing/commissions-options.php, 2026-09-12
    @test commission(:ibkr_pro_us_options, [2.00],  [1]) == [100]   # 65 raised to the USD 1.00 minimum
    @test commission(:ibkr_pro_us_options, [5.00],  [2]) == [130]   # 2 * 65
    @test commission(:ibkr_pro_us_options, [0.075], [3]) == [150]   # 3 * 50
    @test commission(:ibkr_pro_us_options, [0.03],  [5]) == [125]   # 5 * 25
    # tier edges: 0.05 is the 50-cent tier, 0.10 the 65-cent tier
    @test commission(:ibkr_pro_us_options, [0.05],   [3]) == [150]
    @test commission(:ibkr_pro_us_options, [0.10],   [2]) == [130]
    @test commission(:ibkr_pro_us_options, [0.0499], [3]) == [100]  # 75 raised to the minimum
    @test commission(:ibkr_pro_us_options, [0.0999], [1]) == [100]  # 50 raised to the minimum
    # a strangle: 65 + 65 = 130, above the minimum, one share each
    @test commission(:ibkr_pro_us_options, [0.85, 1.10], [1, 1]) == [65, 65]
    # the minimum shared: 25 + 65 = 90 raised to 100; round(100 * 25 / 90) = 28, then 100 - 28 = 72
    @test commission(:ibkr_pro_us_options, [0.03, 0.50], [1, 1]) == [28, 72]
    # shares always sum to the order's commission
    for (prices, qs) in (([0.03, 0.03, 0.03], [1, 1, 1]), ([0.04, 0.50, 2.0], [2, 1, 3]), ([0.07], [1]))
        shares = commission(:ibkr_pro_us_options, prices, qs)
        raw = sum(VolSurfaceAnalysis._ibkr_rate_cents(p) * q for (p, q) in zip(prices, qs))
        @test sum(shares) == max(raw, 100)
        @test all(s -> s >= 0, shares)
    end
    @test commission(:ibkr_pro_us_options, [0.03, 0.03, 0.03], [1, 1, 1]) == [33, 34, 33]  # 75 raised to 100: 33, 67 - 33, 100 - 67
    # :none gives zeros; an empty order gives an empty vector
    @test commission(:none, [0.85, 1.10], [1, 1]) == [0, 0]
    @test commission(:none, Float64[], Int[]) == Int[]
    @test commission(:ibkr_pro_us_options, Float64[], Int[]) == Int[]
    # an unknown model errors naming the known ones; a malformed call is an ArgumentError
    err = try commission(:free_lunch, [1.0], [1]); nothing catch e; e end
    @test err isa ErrorException
    @test occursin("ibkr_pro_us_options", err.msg) && occursin("none", err.msg) && occursin("free_lunch", err.msg)
    @test_throws ArgumentError commission(:none, [1.0, 2.0], [1])
    @test_throws ArgumentError commission(:none, [1.0], [0])
end

@testset "the venue's tables hold what the docs say" begin
    @test collect(keys(VolSurfaceAnalysis._FILL_RULES)) == [:cross_spread]
    @test Set(keys(VolSurfaceAnalysis._COST_MODELS)) == Set([:none, :ibkr_pro_us_options])
    @test !haskey(VolSurfaceAnalysis._FILL_RULES, :broker_execution)
end

@testset "TICK_CENTS: the penny tick, defaulted rather than passed" begin
    # The tick is a constant because every underlying the contract table
    # lists trades in penny increments at every premium, so nothing about
    # an experiment can vary it; it stays an argument only so the join
    # check can state the tick it recomputes a fill against.
    @test TICK_CENTS == 1
    for u in ("SPY", "QQQ", "IWM")
        @test contract_spec(Underlying(u)) isa ContractSpec
    end
    # Defaulting it is the same rule applied: a buy at an ask of 0.857
    # rounds up to 0.86, a sale at a bid of 0.857 down to 0.85.
    @test fill_price(:cross_spread, 0.857, 0.857, Long)  ==
          fill_price(:cross_spread, 0.857, 0.857, Long,  TICK_CENTS) == 0.86
    @test fill_price(:cross_spread, 0.857, 0.857, Short) ==
          fill_price(:cross_spread, 0.857, 0.857, Short, TICK_CENTS) == 0.85
end
