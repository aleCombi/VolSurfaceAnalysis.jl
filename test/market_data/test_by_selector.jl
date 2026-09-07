# BySelector: constructor invariants, routing per shape, context
# forwarding, and inference across a heterogeneous part list.

const _MD_QQQ = Underlying("QQQ")

function _md_by_selector_fixture()
    spy = [s for s in _md_spots() if s.underlying === _MD_SPY]
    spx = [s for s in _md_spots() if s.underlying === _MD_SPX]
    # B also carries an SPY row that must never be reachable: SPY routes to A.
    a = InMemory(spy)
    b = InMemory(vcat(spx, [_md_spot(_MD_SPY, _MD_T1, 999.0)]))
    (a = a, b = b, m = MarketData(BySelector{SpotPrice}(_MD_SPY => a, _MD_SPX => b)))
end

@testset "BySelector: constructor rejections" begin
    spots = InMemory(_md_spots())
    bars  = InMemory(_md_bars())
    @test_throws ArgumentError BySelector{SpotPrice}()
    @test_throws ArgumentError BySelector{SpotPrice}(_MD_SPY => spots, _MD_SPX => bars)     # mixed kind
    @test_throws ArgumentError BySelector{SpotPrice}(_MD_SPY => spots, _MD_SPY => spots)    # duplicate
    @test_throws ArgumentError BySelector{SpotPrice}(Currency("USD") => spots)              # wrong selector type
    @test_throws ArgumentError BySelector{SpotPrice}("SPY" => spots)
    bs = BySelector{SpotPrice}(_MD_SPY => spots)
    @test kind(bs) === SpotPrice
    @test inputs(bs) == ()
end

@testset "BySelector: routes every shape, unknown selector throws" begin
    f = _md_by_selector_fixture()
    m = f.m
    @test only_or_missing(at(m, SpotPrice, _MD_SPY, _MD_T1)).price == 481.0     # A, not B's 999
    @test only_or_missing(at(m, SpotPrice, _MD_SPX, _MD_T1)).price == 4801.0
    @test collect(between(m, SpotPrice, _MD_SPX, _MD_T1, _MD_T3)) ==
          [s for s in f.b.rows if s.underlying === _MD_SPX]
    @test collect(between(m, SpotPrice, _MD_SPY, _MD_T1, _MD_T3)) == f.a.rows
    @test only_or_missing(asof(m, SpotPrice, _MD_SPY, _MD_T3 + Day(1))).timestamp == _MD_T3
    @test asof(m, SpotPrice, _MD_SPX, _MD_T1 - Day(1)) == SpotPrice[]
    @test timestamps(m, SpotPrice, _MD_SPX, _MD_T1, _MD_T2) == [_MD_T1, _MD_T2]
    @test_throws KeyError at(m, SpotPrice, _MD_QQQ, _MD_T1)
    @test_throws KeyError timestamps(m, SpotPrice, _MD_QQQ, _MD_T1, _MD_T3)
end

@testset "BySelector: forwards the context untouched" begin
    f = _md_by_selector_fixture()
    cut = TimeCut(f.m, _MD_T1)
    @test at(cut, SpotPrice, _MD_SPX, _MD_T2) == SpotPrice[]
    @test only_or_missing(asof(cut, SpotPrice, _MD_SPX, _MD_T3)).timestamp == _MD_T1
    # a derived provider reading OptionBar through a BySelector entry
    bars = BySelector{OptionBar}(
        _MD_SPY => InMemory([b for b in _md_bars() if b.underlying === _MD_SPY]),
        _MD_SPX => InMemory([b for b in _md_bars() if b.underlying === _MD_SPX]))
    m = MarketData(bars, QuotesFromBars(SpreadFromOHLCV(0.7)))
    @test length(at(m, OptionQuote, _MD_SPX, _MD_T1)) == 2
    @test timestamps(m, Clock{OptionQuote}(_MD_SPX), _MD_T1, _MD_T3) == [_MD_T1, _MD_T2, _MD_T3]
    @test at(TimeCut(m, _MD_T1), OptionQuote, _MD_SPX, _MD_T2) == OptionQuote[]
end

@testset "BySelector: heterogeneous parts infer to one return type" begin
    spy = InMemory([s for s in _md_spots() if s.underlying === _MD_SPY])
    spx = Constant(SpotPrice(_MD_SPX, 4800.0, typemin(DateTime)))
    m = MarketData(BySelector{SpotPrice}(_MD_SPY => spy, _MD_SPX => spx))
    @test @inferred(at(m, SpotPrice, _MD_SPY, _MD_T1)) isa Vector{SpotPrice}
    @test @inferred(asof(m, SpotPrice, _MD_SPX, _MD_T1)) == [spx.record]
    @test @inferred(asof(m, SpotPrice, _MD_SPY, _MD_T1)) isa Vector{SpotPrice}
    @test Base.return_types(at, (typeof(m), Type{SpotPrice}, Underlying, DateTime)) == [Vector{SpotPrice}]
    @test Base.return_types(asof, (typeof(m), Type{SpotPrice}, Underlying, DateTime)) == [Vector{SpotPrice}]
    @test Base.return_types(entry, (typeof(m), Type{SpotPrice})) == [typeof(m.entries[1])]
end
