# MarketData: entry by kind, construction checks, inference.

@testset "MarketData: entry by kind" begin
    bars  = InMemory(_md_bars())
    spots = InMemory(_md_spots())
    q     = QuotesFromBars(SpreadFromOHLCV(0.7))
    m = MarketData(bars, q, spots)
    @test entry(m, OptionBar) === bars
    @test entry(m, OptionQuote) === q
    @test entry(m, SpotPrice) === spots
    err = try entry(m, RawSurface); nothing catch e; e end
    @test err isa ErrorException
    @test occursin("RawSurface", err.msg)
    @test MarketData((bars, q)) isa typeof(MarketData(bars, q))
end

@testset "MarketData: construction rejects empty and duplicate kinds" begin
    @test_throws ArgumentError MarketData(())
    @test_throws ArgumentError MarketData(InMemory(_md_bars()), InMemory(OptionBar[]))
    @test_throws ArgumentError MarketData(InMemory(_md_spots()),
                                          Constant(SpotPrice(_MD_SPY, 1.0, typemin(DateTime))))
end

@testset "MarketData: entry and routed at infer" begin
    m = MarketData(InMemory(_md_bars()), QuotesFromBars(SpreadFromOHLCV(0.7)), InMemory(_md_spots()))
    @test @inferred(entry(m, OptionBar)) === m.entries[1]
    @test @inferred(entry(m, SpotPrice)) === m.entries[3]
    @test @inferred(at(m, OptionQuote, _MD_SPY, _MD_T1)) isa Vector{OptionQuote}
    @test @inferred(at(m, SpotPrice, _MD_SPY, _MD_T1)) isa Vector{SpotPrice}
    @test @inferred(asof(m, OptionBar, _MD_SPY, _MD_T1)) isa Vector{OptionBar}
end
