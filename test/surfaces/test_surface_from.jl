# SurfaceFrom on a full in-memory map: build, absence (cached), spot_for,
# bounded cache, cut independence, stepped rate, shapes.

const _SF_TS1 = DateTime(2024, 1, 15, 15, 30)
const _SF_TS2 = DateTime(2024, 1, 15, 15, 31)
const _SF_EXPIRY = DateTime(2024, 2, 16, 21, 0)
const _SF_SPOT = 480.0
const _SF_R = 0.04
const _SF_Q = 0.015

# A bar whose close is the BS price, so the synthesized mark is exact.
function _sf_bar(strike::Float64, ts::DateTime, sigma::Float64;
                 u::Underlying=_MD_SPY, expiry::DateTime=_SF_EXPIRY, spot=_SF_SPOT, r=_SF_R, q=_SF_Q)
    otype = strike >= spot ? Call : Put
    mark = bs_price(spot, strike, time_to_expiry(expiry, ts), sigma, otype; r=r, q=q)
    OptionBar("X$(strike)", u, expiry, strike, otype, mark, mark, mark, mark, 1.0, ts)
end

function _sf_bars(; u=_MD_SPY, expiry=_SF_EXPIRY)
    vcat([_sf_bar(k, _SF_TS1, s; u, expiry) for (k, s) in [(470.0, 0.21), (480.0, 0.20), (490.0, 0.19)]],
         [_sf_bar(k, _SF_TS2, s; u, expiry) for (k, s) in [(470.0, 0.22), (480.0, 0.21), (490.0, 0.20)]])
end

_sf_spots(; u=_MD_SPY, price=_SF_SPOT) = [SpotPrice(u, price, _SF_TS1), SpotPrice(u, price, _SF_TS2)]

# Counting wrapper over an OptionBar provider, to show absence is cached.
mutable struct _SF_CountingBars
    inner::InMemory{OptionBar}
    n::Int
end
VolSurfaceAnalysis.kind(::_SF_CountingBars) = OptionBar
VolSurfaceAnalysis.open_data(c::_SF_CountingBars) = c
VolSurfaceAnalysis.close_data!(::_SF_CountingBars) = nothing
function VolSurfaceAnalysis.at(c::_SF_CountingBars, m, ::Type{OptionBar}, u, ts::DateTime)
    c.n += 1
    at(c.inner, m, OptionBar, u, ts)
end
VolSurfaceAnalysis.between(c::_SF_CountingBars, m, ::Type{OptionBar}, u, from::DateTime, to::DateTime) =
    between(c.inner, m, OptionBar, u, from, to)
VolSurfaceAnalysis.asof(c::_SF_CountingBars, m, ::Type{OptionBar}, u, ts::DateTime) =
    asof(c.inner, m, OptionBar, u, ts)
VolSurfaceAnalysis.timestamps(c::_SF_CountingBars, m, ::Type{OptionBar}, u, from::DateTime, to::DateTime) =
    timestamps(c.inner, m, OptionBar, u, from, to)

function _sf_map(; bars=InMemory(_sf_bars()), spots=InMemory(_sf_spots()),
                   rate=Constant(RateCurve(_MD_USD, FlatCurve(_SF_R))),
                   div=Constant(DivCurve(_MD_SPY, FlatCurve(_SF_Q))),
                   surface=SurfaceFrom(currency=_MD_USD))
    MarketData(bars, QuotesFromBars(SpreadFromOHLCV(0.7)), spots, rate, div, surface)
end

@testset "SurfaceFrom: spec, kind, inputs, selector on surfaces" begin
    s = SurfaceFrom(currency=_MD_USD)
    @test kind(s) === VolatilitySurface
    @test inputs(s) == (OptionQuote, SpotPrice, RateCurve, DivCurve)
    # derived: it delegates the structural question rather than answering it
    @test serves(s, nothing, VolatilitySurface, _MD_SPY) === missing
    @test collect(demands(s)) == [(RateCurve, _MD_USD)]
    @test collect(demands(SurfaceFrom(currency=_MD_USD, spot_for=Dict(_MD_SPY => _MD_SPX)))) ==
          [(RateCurve, _MD_USD), (SpotPrice, _MD_SPX)]
    @test isempty(s.spot_for)
    @test s.lookback_ticks == 3
    @test SurfaceFrom(currency=_MD_USD, lookback_ticks=1).lookback_ticks == 1
    @test_throws ArgumentError SurfaceFrom(currency=_MD_USD, lookback_ticks=0)
    # lookback_ticks changes which surface a policy sees, so it is identity:
    # == and hash are hand-written and must carry it
    @test SurfaceFrom(currency=_MD_USD, lookback_ticks=1) != s
    @test hash(SurfaceFrom(currency=_MD_USD, lookback_ticks=1)) != hash(s)
    @test SurfaceFrom(currency=_MD_USD, lookback_ticks=3) == s
    @test hash(SurfaceFrom(currency=_MD_USD, lookback_ticks=3)) == hash(s)
    @test SurfaceFrom(currency=_MD_USD, spot_for=Dict(_MD_SPY => _MD_SPX)).spot_for[_MD_SPY] === _MD_SPX
    @test selector_type(VolatilitySurface) === Underlying
    @test selector_type(RawSurface) === Underlying
    @test Clock{VolatilitySurface}(_MD_SPY) isa Clock
    @test VolSurfaceAnalysis.has_lifecycle(s)
    r = open_data(s)
    @test kind(r) === VolatilitySurface && inputs(r) == inputs(s)
    @test close_data!(r) === nothing
end

@testset "SurfaceFrom: builds the surface from the map" begin
    with_data(_sf_map()) do d
        s = only_or_missing(at(d, VolatilitySurface, _MD_SPY, _SF_TS1))
        @test s isa RawSurface
        @test selector(s) === _MD_SPY
        @test s.timestamp == _SF_TS1
        @test expiries(s) == [_SF_EXPIRY]
        @test iv(s, _SF_EXPIRY, 480.0) ≈ 0.20 atol = 1e-4
        @test s.spot == _SF_SPOT && s.rate == _SF_R && s.div == _SF_Q
        s2 = only_or_missing(at(d, VolatilitySurface, _MD_SPY, _SF_TS2))
        @test iv(s2, _SF_EXPIRY, 480.0) ≈ 0.21 atol = 1e-4
        # SPX has no bars: the derived read fails on its input, and the error
        # names the input kind and selector, not the surface.
        err = try
            at(d, VolatilitySurface, _MD_SPX, _SF_TS1)
        catch e
            e
        end
        @test err isa UnservedSelector
        @test err.kind === OptionBar && err.selector === _MD_SPX
        @test at(d, VolatilitySurface, _MD_SPY, _SF_TS1) isa Vector{VolatilitySurface}
    end
end

@testset "SurfaceFrom: absent inputs give empty, cached" begin
    counting = _SF_CountingBars(InMemory(_sf_bars()), 0)
    with_data(_sf_map(bars=counting)) do d
        miss = DateTime(2024, 1, 15, 16, 0)
        @test at(d, VolatilitySurface, _MD_SPY, miss) == VolatilitySurface[]
        @test counting.n == 1
        @test at(d, VolatilitySurface, _MD_SPY, miss) == VolatilitySurface[]
        @test counting.n == 1                                   # not re-read
    end
    # no spot at ts2
    with_data(_sf_map(spots=InMemory([SpotPrice(_MD_SPY, _SF_SPOT, _SF_TS1)]))) do d
        @test !isempty(at(d, VolatilitySurface, _MD_SPY, _SF_TS1))
        @test at(d, VolatilitySurface, _MD_SPY, _SF_TS2) == VolatilitySurface[]
    end
    # rate for the wrong currency, div for the wrong underlying: structural,
    # so each throws naming the curve kind the surface asked for
    with_data(_sf_map(rate=Constant(RateCurve(Currency("EUR"), FlatCurve(_SF_R))))) do d
        err = try at(d, VolatilitySurface, _MD_SPY, _SF_TS1) catch e; e end
        @test err isa UnservedSelector && err.kind === RateCurve
        @test occursin("EUR", sprint(showerror, err))
    end
    with_data(_sf_map(div=Constant(DivCurve(_MD_SPX, FlatCurve(_SF_Q))))) do d
        err = try at(d, VolatilitySurface, _MD_SPY, _SF_TS1) catch e; e end
        @test err isa UnservedSelector && err.kind === DivCurve
        @test occursin("SPX", sprint(showerror, err))
    end
    # every expiry already passed: build_surface returns nothing -> empty
    with_data(_sf_map(bars=InMemory(_sf_bars(expiry=DateTime(2024, 1, 12, 21, 0))))) do d
        @test at(d, VolatilitySurface, _MD_SPY, _SF_TS1) == VolatilitySurface[]
    end
end

@testset "SurfaceFrom: spot_for remap uses the other underlying's spot" begin
    spots = InMemory(vcat(_sf_spots(u=_MD_SPY, price=_SF_SPOT), _sf_spots(u=_MD_SPX, price=4800.0)))
    with_data(_sf_map(spots=spots, surface=SurfaceFrom(currency=_MD_USD, spot_for=Dict(_MD_SPY => _MD_SPX)))) do d
        s = only_or_missing(at(d, VolatilitySurface, _MD_SPY, _SF_TS1))
        @test s.spot == 4800.0
        @test s.underlying === _MD_SPY
    end
    # remap to an underlying the spot entry does not serve -> structural throw
    with_data(_sf_map(surface=SurfaceFrom(currency=_MD_USD, spot_for=Dict(_MD_SPY => _MD_SPX)))) do d
        err = try at(d, VolatilitySurface, _MD_SPY, _SF_TS1) catch e; e end
        @test err isa UnservedSelector
        @test err.kind === SpotPrice && err.selector === _MD_SPX
    end
end

@testset "SurfaceFrom: cache bounded (max_surfaces=1)" begin
    m = _sf_map()
    r = open_data(SurfaceFrom(currency=_MD_USD); max_surfaces=1)
    d = MarketData(m.entries[1:5]..., r)               # the other specs are their own readers
    at(d, VolatilitySurface, _MD_SPY, _SF_TS1)
    at(d, VolatilitySurface, _MD_SPY, _SF_TS2)
    @test length(r.cache) == 1
    @test collect(keys(r.cache)) == [(_MD_SPY, _SF_TS2)]
end

@testset "SurfaceFrom: cut independence of the cache" begin
    with_data(_sf_map()) do d
        s1 = only_or_missing(at(d, VolatilitySurface, _MD_SPY, _SF_TS1))
        @test only_or_missing(at(TimeCut(d, _SF_TS1), VolatilitySurface, _MD_SPY, _SF_TS1)) === s1
        @test only_or_missing(at(TimeCut(d, _SF_TS2), VolatilitySurface, _MD_SPY, _SF_TS1)) === s1
        # masked past the cutoff, and the mask does not poison the cache
        @test at(TimeCut(d, _SF_TS1), VolatilitySurface, _MD_SPY, _SF_TS2) == VolatilitySurface[]
        @test !isempty(at(d, VolatilitySurface, _MD_SPY, _SF_TS2))
        # a surface first built through a cut is the same object later without it
        s2c = only_or_missing(at(TimeCut(d, _SF_TS2), VolatilitySurface, _MD_SPY, _SF_TS2))
        @test only_or_missing(at(d, VolatilitySurface, _MD_SPY, _SF_TS2)) === s2c
        @test only_or_missing(asof(TimeCut(d, _SF_TS1), VolatilitySurface, _MD_SPY, _SF_TS2)) === s1
    end
end

@testset "SurfaceFrom: stepped rate curve flows into surface.rate" begin
    stepped = Constant(RateCurve(_MD_USD, PCCurve([_SF_TS1, _SF_TS2], [0.04, 0.05])))
    with_data(_sf_map(rate=stepped)) do d
        @test only_or_missing(at(d, VolatilitySurface, _MD_SPY, _SF_TS1)).rate == 0.04
        @test only_or_missing(at(d, VolatilitySurface, _MD_SPY, _SF_TS2)).rate == 0.05
    end
end

# One chain that builds at TS1 and one that cannot at TS2 (every expiry
# already passed there), so `asof` at TS2 must walk back to TS1.
function _sf_walkback_map(; lookback_ticks::Int=3)
    good = _sf_bars()                                     # both timestamps, valid expiry
    stale = [_sf_bar(480.0, _SF_TS2, 0.20; expiry=_SF_TS2 - Day(1))]
    MarketData(InMemory(vcat([b for b in good if b.timestamp == _SF_TS1], stale)),
               QuotesFromBars(SpreadFromOHLCV(0.7)),
               InMemory(_sf_spots()),
               Constant(RateCurve(_MD_USD, FlatCurve(_SF_R))),
               Constant(DivCurve(_MD_SPY, FlatCurve(_SF_Q))),
               SurfaceFrom(currency=_MD_USD, lookback_ticks=lookback_ticks))
end

@testset "SurfaceFrom: asof walks back past an unbuildable chain, under a bound" begin
    with_data(_sf_walkback_map()) do d
        prior = at(d, VolatilitySurface, _MD_SPY, _SF_TS1)
        @test !isempty(prior)                                  # not a vacuous comparison
        @test at(d, VolatilitySurface, _MD_SPY, _SF_TS2) == VolatilitySurface[]
        @test asof(d, VolatilitySurface, _MD_SPY, _SF_TS2) == prior
        # the input grid is unchanged: timestamps over-estimates for a derived kind
        @test timestamps(d, VolatilitySurface, _MD_SPY, _SF_TS1, _SF_TS2) == [_SF_TS1, _SF_TS2]
    end

    # lookback_ticks = 1 examines only the newest quote timestamp -- and still
    # throws rather than reproducing the old empty result, deliberately
    with_data(_sf_walkback_map(lookback_ticks=1)) do d
        err = try asof(d, VolatilitySurface, _MD_SPY, _SF_TS2) catch e; e end
        @test err isa DerivationExhausted
        @test err.kind === VolatilitySurface && err.selector === _MD_SPY
        @test err.requested == _SF_TS2 && err.bound == 1
        @test occursin("VolatilitySurface", sprint(showerror, err))
    end

    # every chain within the bound is unbuildable: the bound is exhausted
    stale = [_sf_bar(480.0, _SF_TS1, 0.20; expiry=_SF_TS1 - Day(1)),
             _sf_bar(480.0, _SF_TS2, 0.20; expiry=_SF_TS2 - Day(1))]
    with_data(_sf_map(bars=InMemory(stale),
                      surface=SurfaceFrom(currency=_MD_USD, lookback_ticks=2))) do d
        err = try asof(d, VolatilitySurface, _MD_SPY, _SF_TS2) catch e; e end
        @test err isa DerivationExhausted
        @test err.bound == 2 && err.requested == _SF_TS2
        @test err.oldest == _SF_TS1 - Millisecond(1)      # the cursor after the last try
    end

    # running out of chain entirely is temporal, not exhaustion: empty
    with_data(_sf_map(bars=InMemory(stale))) do d          # default bound of 3
        @test asof(d, VolatilitySurface, _MD_SPY, _SF_TS2) == VolatilitySurface[]
    end
end

@testset "SurfaceFrom: between / asof / timestamps follow the quote grid" begin
    with_data(_sf_map()) do d
        @test timestamps(d, VolatilitySurface, _MD_SPY, _SF_TS1, _SF_TS2) == [_SF_TS1, _SF_TS2]
        @test timestamps(d, Clock{VolatilitySurface}(_MD_SPY), _SF_TS1, _SF_TS2 + Day(1)) == [_SF_TS1, _SF_TS2]
        surfaces = collect(between(d, VolatilitySurface, _MD_SPY, _SF_TS1, _SF_TS2))
        @test length(surfaces) == 2
        @test [s.timestamp for s in surfaces] == [_SF_TS1, _SF_TS2]
        @test surfaces[1] === only_or_missing(at(d, VolatilitySurface, _MD_SPY, _SF_TS1))
        latest = only_or_missing(asof(d, VolatilitySurface, _MD_SPY, _SF_TS2 + Minute(5)))
        @test latest === surfaces[2]
        @test asof(d, VolatilitySurface, _MD_SPY, _SF_TS1 - Minute(1)) == VolatilitySurface[]
    end
end
