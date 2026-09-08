# TimeCut: every shape masked at the cutoff, including reads a derived
# provider makes through the cut.

function _md_cut_fixture()
    m = MarketData(InMemory(_md_bars()), QuotesFromBars(SpreadFromOHLCV(0.7)), InMemory(_md_spots()))
    (m = m, cut = TimeCut(m, _MD_T1))
end

@testset "TimeCut: raw shapes masked past the cutoff" begin
    f = _md_cut_fixture()
    cut = f.cut
    @test entry(cut, OptionBar) === entry(f.m, OptionBar)
    @test at(cut, SpotPrice, _MD_SPY, _MD_T1) == at(f.m, SpotPrice, _MD_SPY, _MD_T1)
    @test at(cut, SpotPrice, _MD_SPY, _MD_T2) == SpotPrice[]
    @test [r.timestamp for r in between(cut, SpotPrice, _MD_SPY, _MD_T1, _MD_T3)] == [_MD_T1]
    @test between(cut, SpotPrice, _MD_SPY, _MD_T2, _MD_T3) == SpotPrice[]
    @test only_or_missing(asof(cut, SpotPrice, _MD_SPY, _MD_T3)).timestamp == _MD_T1   # clamped
    @test timestamps(cut, SpotPrice, _MD_SPY, _MD_T1, _MD_T3) == [_MD_T1]
    @test timestamps(cut, SpotPrice, _MD_SPY, _MD_T2, _MD_T3) == DateTime[]
end

@testset "TimeCut: cut through a derived provider" begin
    f = _md_cut_fixture()
    cut = f.cut
    @test at(cut, OptionQuote, _MD_SPY, _MD_T2) == OptionQuote[]
    @test length(at(cut, OptionQuote, _MD_SPY, _MD_T1)) == 2
    latest = asof(cut, OptionQuote, _MD_SPY, _MD_T3)
    @test length(latest) == 2
    @test all(q -> q.timestamp == _MD_T1, latest)          # the OptionBar read went through the cut
    @test [q.timestamp for q in between(cut, OptionQuote, _MD_SPY, _MD_T1, _MD_T3)] == [_MD_T1, _MD_T1]
    @test timestamps(cut, Clock{OptionQuote}(_MD_SPY), _MD_T1, _MD_T3) == [_MD_T1]
    # a cut past the data is transparent
    open_cut = TimeCut(f.m, _MD_T3 + Day(1))
    @test at(open_cut, OptionQuote, _MD_SPY, _MD_T3) == at(f.m, OptionQuote, _MD_SPY, _MD_T3)
end
