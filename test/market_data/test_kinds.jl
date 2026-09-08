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

@testset "protocol: exported names, no Base collisions" begin
    exported = names(VolSurfaceAnalysis)
    for s in (:Currency, :selector, :selector_type, :at, :between, :asof,
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
