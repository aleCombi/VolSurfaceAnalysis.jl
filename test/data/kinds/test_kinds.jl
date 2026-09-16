# Kinds, selector contract, and the export / name-clash guards for the
# market_data protocol.

@testset "Currency: uppercase-normalized value type" begin
    @test Currency("usd") == Currency("USD")
    @test Currency("usd").code == "USD"
    @test sprint(show, Currency("eur")) == "EUR"
    d = Dict(Currency("usd") => 1)
    @test d[Currency("USD")] == 1
    @test Currency("USD") != Underlying("USD")
    @test hash(Currency("usd")) == hash(Currency("USD"))
    @test hash(Currency("USD")) != hash(Underlying("USD"))
end

@testset "selector / selector_type per kind" begin
    bar  = _md_bar(_MD_SPY, _MD_T1, 470.0)
    spot = _md_spot(_MD_SPX, _MD_T1, 4801.0)
    q    = synthesize(SpreadFromOHLCV(0.7), bar)
    @test selector(bar)  === _MD_SPY
    @test selector(q)    === _MD_SPY
    @test selector(spot) === _MD_SPX
    @test selector_type(OptionBar)   === Underlying
    @test selector_type(OptionQuote) === Underlying
    @test selector_type(SpotPrice)   === Underlying
    @test selector(bar) isa selector_type(OptionBar)
end

@testset "snapshot trait: one record per selector per instant" begin
    @test snapshot(SpotPrice)
    @test snapshot(RateCurve)
    @test snapshot(DivCurve)
    @test snapshot(RawSurface)
    @test !snapshot(OptionBar)
    @test !snapshot(OptionQuote)
end

@testset "protocol: exported names, no Base collisions" begin
    exported = names(VolSurfaceAnalysis)
    for s in (:Currency, :selector, :selector_type, :snapshot, :at, :between, :asof,
              :timestamps, :kind, :only_or_missing, :by_timestamp)
        @test s in exported
    end
    # Base.between exists (unexported, Integer-only); ours must be a
    # separate generic function, never `import Base: between`.
    @test length(methods(Base.between)) == 1
    @test VolSurfaceAnalysis.between !== Base.between
    for f in (at, between, asof, timestamps, kind)
        @test f isa Function
    end
end

# --- records formerly covered by test/data/test_quotes.jl ---
@testset "Underlying" begin
    @test ticker(Underlying("spy")) == "SPY"
    @test ticker(Underlying("BTC")) == "BTC"
    @test sprint(show, Underlying("SPY")) == "SPY"
end

@testset "OptionQuote" begin
    q = OptionQuote(
        "O:SPY240129C00406000",
        Underlying("SPY"),
        DateTime(2024, 1, 29, 21, 0),
        406.0,
        Call,
        1.20, 1.25, 1.225, 18.5, 1234.0, 567.0,
        DateTime(2024, 1, 15, 15, 30),
    )
    @test q.strike == 406.0
    @test q.option_type == Call
    @test q.iv == 18.5
    @test q.bid == 1.20
end

@testset "OptionQuote with missings" begin
    q = OptionQuote(
        "X", Underlying("SPY"), DateTime(2024, 1, 29), 100.0, Put,
        missing, missing, missing, missing, missing, missing,
        DateTime(2024, 1, 15),
    )
    @test ismissing(q.bid)
    @test ismissing(q.ask)
    @test ismissing(q.iv)
    @test q.option_type == Put
end

@testset "SpotPrice" begin
    s = SpotPrice(Underlying("SPY"), 480.5, DateTime(2024, 1, 15, 15, 30))
    @test s.price == 480.5
    @test ticker(s.underlying) == "SPY"
end

@testset "Underlying: content hash and equality" begin
    @test Underlying("spy") == Underlying("SPY")
    @test hash(Underlying("spy")) == hash(Underlying("SPY"))
    @test hash(Underlying("SPY")) != hash(Underlying("SPX"))
    d = Dict(Underlying("SPY") => 1)
    @test d[Underlying("spy")] == 1
    @test isequal(Underlying("SPY"), Underlying("SPY"))
end
