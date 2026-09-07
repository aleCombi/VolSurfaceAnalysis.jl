# InMemory, Constant, QuotesFromBars against the protocol rules.

const _MD_SEC = Second(1)

@testset "InMemory{SpotPrice}: at / between / timestamps, selector-filtered" begin
    p = InMemory(reverse(_md_spots()))             # unsorted input is sorted on construction
    m = MarketData(p)
    @test kind(p) === SpotPrice
    @test issorted(p.rows; by = r -> r.timestamp)

    hit = at(m, SpotPrice, _MD_SPY, _MD_T1)
    @test hit isa Vector{SpotPrice}
    @test length(hit) == 1 && hit[1].price == 481.0
    @test only_or_missing(at(m, SpotPrice, _MD_SPX, _MD_T2)).price == 4802.0
    @test at(m, SpotPrice, _MD_SPY, _MD_T1 + _MD_SEC) == SpotPrice[]
    @test at(m, SpotPrice, Underlying("QQQ"), _MD_T1) == SpotPrice[]

    rng = collect(between(m, SpotPrice, _MD_SPX, _MD_T1, _MD_T2))
    @test [r.timestamp for r in rng] == [_MD_T1, _MD_T2]
    @test all(r -> r.underlying === _MD_SPX, rng)
    @test isempty(between(m, SpotPrice, _MD_SPY, _MD_T3 + _MD_SEC, _MD_T3 + Day(1)))

    for ts in (_MD_T1, _MD_T2, _MD_T3, _MD_T1 + _MD_SEC), u in (_MD_SPY, _MD_SPX)
        @test at(m, SpotPrice, u, ts) == collect(between(m, SpotPrice, u, ts, ts))
    end

    @test timestamps(m, SpotPrice, _MD_SPY, _MD_T1, _MD_T2) == [_MD_T1, _MD_T2]
    @test timestamps(m, SpotPrice, _MD_SPY, _MD_T1 + _MD_SEC, _MD_T3) == [_MD_T2, _MD_T3]
    @test timestamps(m, SpotPrice, _MD_SPY, _MD_T2 + _MD_SEC, _MD_T3) == [_MD_T3]
    @test timestamps(m, SpotPrice, _MD_SPY, _MD_T3 + _MD_SEC, _MD_T3 + Day(1)) == DateTime[]
end

@testset "InMemory: asof returns every record at the winning timestamp" begin
    m = MarketData(InMemory(_md_bars()))
    chain = asof(m, OptionBar, _MD_SPY, _MD_T2 + Minute(1))
    @test length(chain) == 2                       # both SPY bars at T2, none of SPX
    @test all(b -> b.timestamp == _MD_T2 && b.underlying === _MD_SPY, chain)
    @test asof(m, OptionBar, _MD_SPY, _MD_T2) == at(m, OptionBar, _MD_SPY, _MD_T2)
    @test asof(m, OptionBar, _MD_SPY, _MD_T1 - Minute(1)) == OptionBar[]
    @test asof(m, OptionBar, Underlying("QQQ"), _MD_T3) == OptionBar[]

    s = MarketData(InMemory(_md_spots()))
    @test only_or_missing(asof(s, SpotPrice, _MD_SPX, _MD_T3 + Day(30))).timestamp == _MD_T3
end

@testset "Constant{SpotPrice}: selector-checked, visible from the start of time" begin
    rec = SpotPrice(_MD_SPY, 100.0, typemin(DateTime))
    c = Constant(rec)
    m = MarketData(c)
    @test kind(c) === SpotPrice
    @test asof(m, SpotPrice, _MD_SPY, _MD_T1) == [rec]
    @test asof(m, SpotPrice, _MD_SPX, _MD_T1) == SpotPrice[]
    @test between(m, SpotPrice, _MD_SPY, DateTime(2024), DateTime(2025)) == SpotPrice[]
    @test between(m, SpotPrice, _MD_SPY, typemin(DateTime), _MD_T1) == [rec]
    @test between(m, SpotPrice, _MD_SPX, typemin(DateTime), _MD_T1) == SpotPrice[]
    @test timestamps(m, SpotPrice, _MD_SPY, DateTime(2024), DateTime(2025)) == DateTime[]
    @test at(m, SpotPrice, _MD_SPY, _MD_T1) == SpotPrice[]
    @test inputs(c) == ()
end

@testset "QuotesFromBars: reads OptionBar through the map" begin
    synth = SpreadFromOHLCV(0.7)
    bars = InMemory(_md_bars())
    q = QuotesFromBars(synth)
    m = MarketData(bars, q)
    @test kind(q) === OptionQuote
    @test inputs(q) == (OptionBar,)
    @test inputs(bars) == ()

    quotes = at(m, OptionQuote, _MD_SPY, _MD_T1)
    @test quotes isa Vector{OptionQuote}
    @test quotes == [synthesize(synth, b) for b in at(m, OptionBar, _MD_SPY, _MD_T1)]
    @test length(quotes) == 2
    @test quotes[1].bid ≈ 0.94 && quotes[1].ask ≈ 1.06 && quotes[1].mark == 1.00
    @test at(m, OptionQuote, _MD_SPY, _MD_T1 + _MD_SEC) == OptionQuote[]

    rng = collect(between(m, OptionQuote, _MD_SPX, _MD_T1, _MD_T2))
    @test rng == [synthesize(synth, b) for b in between(m, OptionBar, _MD_SPX, _MD_T1, _MD_T2)]
    @test asof(m, OptionQuote, _MD_SPY, _MD_T3 + Day(1)) ==
          [synthesize(synth, b) for b in asof(m, OptionBar, _MD_SPY, _MD_T3 + Day(1))]
    @test timestamps(m, OptionQuote, _MD_SPY, _MD_T1, _MD_T3) ==
          timestamps(m, OptionBar, _MD_SPY, _MD_T1, _MD_T3) == [_MD_T1, _MD_T2, _MD_T3]
end
