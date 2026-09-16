# Clock: typed selector, checked at construction; enumerates the grid.

@testset "Clock: selector type checked, concretely typed" begin
    c = Clock{OptionQuote}(_MD_SPY)
    @test typeof(c).parameters[2] === Underlying
    @test c.sel === _MD_SPY
    @test kind(c) === OptionQuote
    @test_throws ArgumentError Clock{OptionQuote}(Currency("USD"))
    @test_throws ArgumentError Clock{SpotPrice}("SPY")
end

@testset "Clock: timestamps on a map" begin
    m = MarketData(InMemory(_md_bars()), QuotesFromBars(SpreadFromOHLCV(0.7)), InMemory(_md_spots()))
    c = Clock{OptionQuote}(_MD_SPY)
    @test timestamps(m, c, _MD_T1, _MD_T3) == [_MD_T1, _MD_T2, _MD_T3]
    @test timestamps(m, c, _MD_T2, _MD_T2) == [_MD_T2]
    @test timestamps(m, Clock{SpotPrice}(_MD_SPX), _MD_T1, _MD_T1) == [_MD_T1]
end
