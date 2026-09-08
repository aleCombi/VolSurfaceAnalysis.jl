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
    @test at(m, SpotPrice, _MD_SPY, _MD_T1 + _MD_SEC) == SpotPrice[]     # temporal
    @test_throws UnservedSelector at(m, SpotPrice, Underlying("QQQ"), _MD_T1)   # structural

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
    @test_throws UnservedSelector asof(m, OptionBar, Underlying("QQQ"), _MD_T3)

    s = MarketData(InMemory(_md_spots()))
    @test only_or_missing(asof(s, SpotPrice, _MD_SPX, _MD_T3 + Day(30))).timestamp == _MD_T3
end

@testset "Constant{SpotPrice}: selector-checked, visible from the start of time" begin
    rec = SpotPrice(_MD_SPY, 100.0, typemin(DateTime))
    c = Constant(rec)
    m = MarketData(c)
    @test kind(c) === SpotPrice
    @test asof(m, SpotPrice, _MD_SPY, _MD_T1) == [rec]
    @test_throws UnservedSelector asof(m, SpotPrice, _MD_SPX, _MD_T1)
    @test between(m, SpotPrice, _MD_SPY, DateTime(2024), DateTime(2025)) == SpotPrice[]
    @test between(m, SpotPrice, _MD_SPY, typemin(DateTime), _MD_T1) == [rec]
    @test_throws UnservedSelector between(m, SpotPrice, _MD_SPX, typemin(DateTime), _MD_T1)
    @test timestamps(m, SpotPrice, _MD_SPY, DateTime(2024), DateTime(2025)) == DateTime[]
    @test at(m, SpotPrice, _MD_SPY, _MD_T1) == SpotPrice[]
    @test inputs(c) == ()
end

@testset "Constant: asof honours the visibility stamp" begin
    stamp = DateTime(2024, 6, 1)
    rec = RateCurve(_MD_USD, FlatCurve(0.04), stamp)
    m = MarketData(Constant(rec))

    @test asof(m, RateCurve, _MD_USD, stamp - Millisecond(1)) == RateCurve[]
    @test asof(m, RateCurve, _MD_USD, stamp) == [rec]
    @test asof(m, RateCurve, _MD_USD, stamp + Day(30)) == [rec]

    # between / timestamps are unchanged: the stamp must lie inside the range.
    @test between(m, RateCurve, _MD_USD, stamp, stamp) == [rec]
    @test between(m, RateCurve, _MD_USD, typemin(DateTime), stamp - Millisecond(1)) == RateCurve[]
    @test between(m, RateCurve, _MD_USD, stamp + Day(1), stamp + Day(2)) == RateCurve[]
    @test timestamps(m, RateCurve, _MD_USD, typemin(DateTime), stamp) == [stamp]
end

@testset "serves: structural absence is answerable, and three-valued" begin
    spots = InMemory(_md_spots())
    bars  = InMemory(_md_bars())
    q     = QuotesFromBars(SpreadFromOHLCV(0.7))
    c     = Constant(RateCurve(_MD_USD, FlatCurve(0.04)))
    qqq   = Underlying("QQQ")

    @test serves(spots, nothing, SpotPrice, _MD_SPY) === true
    @test serves(spots, nothing, SpotPrice, qqq) === false
    @test serves(InMemory(SpotPrice[]), nothing, SpotPrice, _MD_SPY) === false   # rows are the world
    @test serves(c, nothing, RateCurve, _MD_USD) === true
    @test serves(c, nothing, RateCurve, Currency("EUR")) === false
    @test serves(q, nothing, OptionQuote, _MD_SPY) === missing                   # derived: delegates

    m = MarketData(bars, q, spots)
    @test serves(m, SpotPrice, _MD_SPY) === true
    @test serves(m, SpotPrice, qqq) === false
    @test serves(m, OptionQuote, qqq) === missing        # waved through at the quote entry
    # ... and the OptionBar entry is where it stops, naming the real cause
    err = try at(m, OptionQuote, qqq, _MD_T1) catch e; e end
    @test err isa UnservedSelector && err.kind === OptionBar && err.selector === qqq
    @test occursin("QQQ", sprint(showerror, err))
    @test occursin("SPY", sprint(showerror, err))        # what the entry does serve

    # structural beats temporal: a cut cannot turn a broken config into silence
    cut = TimeCut(m, _MD_T1)
    @test serves(cut, SpotPrice, _MD_SPY) === true
    @test at(cut, SpotPrice, _MD_SPY, _MD_T3) == SpotPrice[]         # temporal, masked
    @test_throws UnservedSelector at(cut, SpotPrice, qqq, _MD_T3)    # structural, past the cutoff
    @test_throws UnservedSelector timestamps(cut, SpotPrice, qqq, _MD_T2, _MD_T3)
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
