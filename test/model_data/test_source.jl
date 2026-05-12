# Counting wrapper DataSource: forwards to an inner InMemoryDataSource
# and records each get_chain / get_spot call. Used to verify caching.
mutable struct _CountingSource <: DataSource
    inner :: InMemoryDataSource
    n_chain :: Int
    n_spot  :: Int
    _CountingSource(inner) = new(inner, 0, 0)
end

function VolSurfaceAnalysis.get_chain(s::_CountingSource, ts::DateTime)
    s.n_chain += 1
    return get_chain(s.inner, ts)
end

function VolSurfaceAnalysis.get_spot(s::_CountingSource, ts::DateTime)
    s.n_spot += 1
    return get_spot(s.inner, ts)
end

VolSurfaceAnalysis.available_timestamps(s::_CountingSource, from::DateTime, to::DateTime) =
    available_timestamps(s.inner, from, to)

# Helpers --------------------------------------------------------------------

const _MDS_UNDERLYING = Underlying("SPY")

function _mds_quote(strike::Float64, expiry::DateTime, ts::DateTime,
                    spot::Float64, sigma::Float64, r::Float64, q::Float64)
    otype = strike >= spot ? Call : Put
    T = time_to_expiry(expiry, ts)
    mark = bs_price(spot, strike, T, sigma, otype; r=r, q=q)
    OptionQuote("X", _MDS_UNDERLYING, expiry, strike, otype,
                missing, missing, mark, missing, missing, missing, ts)
end

function _mds_fixture(; ts1 = DateTime(2024, 1, 15, 15, 30),
                        ts2 = DateTime(2024, 1, 15, 15, 31),
                        spot = 480.0, r = 0.04, q = 0.015)
    expiry = DateTime(2024, 2, 16, 21, 0)
    quotes_ts1 = [_mds_quote(k, expiry, ts1, spot, sigma, r, q)
                  for (k, sigma) in [(470.0, 0.21), (480.0, 0.20), (490.0, 0.19)]]
    quotes_ts2 = [_mds_quote(k, expiry, ts2, spot, sigma, r, q)
                  for (k, sigma) in [(470.0, 0.22), (480.0, 0.21), (490.0, 0.20)]]
    chains = Dict(ts1 => quotes_ts1, ts2 => quotes_ts2)
    spots  = Dict(ts1 => spot, ts2 => spot)
    inner = InMemoryDataSource(_MDS_UNDERLYING; chains=chains, spots=spots)
    (inner=inner, ts1=ts1, ts2=ts2, spot=spot, r=r, q=q, expiry=expiry)
end

# Tests ----------------------------------------------------------------------

@testset "ModelDataSource: build and query surface" begin
    f = _mds_fixture()
    mds = ModelDataSource(f.inner; rate=FlatCurve(f.r), div=FlatCurve(f.q))
    s = get_surface(mds, f.ts1)
    @test s isa RawSurface
    @test expiries(s) == [f.expiry]
    @test iv(s, f.expiry, 480.0) ≈ 0.20 atol = 1e-4
    @test get_spot(mds, f.ts1) == f.spot
    @test get_rate(mds, f.ts1) == f.r
    @test get_div(mds, f.ts1)  == f.q
end

@testset "ModelDataSource: missing chain caches nothing" begin
    f = _mds_fixture()
    counting = _CountingSource(f.inner)
    mds = ModelDataSource(counting; rate=FlatCurve(f.r), div=FlatCurve(f.q))
    miss_ts = DateTime(2024, 1, 15, 16, 0)
    @test get_surface(mds, miss_ts) === nothing
    @test counting.n_chain == 1
    @test get_surface(mds, miss_ts) === nothing  # cached
    @test counting.n_chain == 1
end

@testset "ModelDataSource: missing spot caches nothing" begin
    ts1 = DateTime(2024, 1, 15, 15, 30)
    expiry = DateTime(2024, 2, 16, 21, 0)
    spot = 480.0
    chains = Dict(ts1 => [_mds_quote(480.0, expiry, ts1, spot, 0.20, 0.04, 0.015)])
    inner = InMemoryDataSource(_MDS_UNDERLYING; chains=chains)  # no spots
    mds = ModelDataSource(inner; rate=FlatCurve(0.04), div=FlatCurve(0.015))
    @test get_surface(mds, ts1) === nothing
end

@testset "ModelDataSource: cache prevents rebuild" begin
    f = _mds_fixture()
    counting = _CountingSource(f.inner)
    mds = ModelDataSource(counting; rate=FlatCurve(f.r), div=FlatCurve(f.q))
    s1 = get_surface(mds, f.ts1)
    s2 = get_surface(mds, f.ts1)
    @test s1 === s2
    @test counting.n_chain == 1
end

@testset "ModelDataSource: clear_cache! invalidates" begin
    f = _mds_fixture()
    counting = _CountingSource(f.inner)
    mds = ModelDataSource(counting; rate=FlatCurve(f.r), div=FlatCurve(f.q))
    get_surface(mds, f.ts1)
    @test counting.n_chain == 1
    clear_cache!(mds)
    get_surface(mds, f.ts1)
    @test counting.n_chain == 2
end

@testset "ModelDataSource: independent spot_source" begin
    f = _mds_fixture()
    # Build a separate spot-only source with a *different* spot value.
    alt_spot = 1234.5
    alt_source = InMemoryDataSource(_MDS_UNDERLYING; spots=Dict(f.ts1 => alt_spot))
    mds = ModelDataSource(f.inner; rate=FlatCurve(f.r), div=FlatCurve(f.q),
                          spot_source=alt_source)
    @test get_spot(mds, f.ts1) == alt_spot
    @test get_spot(f.inner, f.ts1) == f.spot      # chain source unchanged
    # get_surface uses alt_spot, so IV at the quoted strike will not match
    # the original; we only assert the spot threading.
    s = get_surface(mds, f.ts1)
    @test s !== nothing
    @test s.spot == alt_spot
end

@testset "ModelDataSource: curves evaluated at the right ts" begin
    f = _mds_fixture()
    # PCCurve with a step between ts1 and ts2; sanity-check it flows through.
    rate_curve = PCCurve([f.ts1, f.ts2], [0.04, 0.05])
    div_curve  = FlatCurve(f.q)
    mds = ModelDataSource(f.inner; rate=rate_curve, div=div_curve)
    @test get_rate(mds, f.ts1) == 0.04
    @test get_rate(mds, f.ts2) == 0.05
    # Surface at ts2 should be built with the stepped rate.
    s2 = get_surface(mds, f.ts2)
    @test s2.rate == 0.05
end

@testset "ModelDataSource: available_timestamps delegates to chain source" begin
    f = _mds_fixture()
    mds = ModelDataSource(f.inner; rate=FlatCurve(f.r), div=FlatCurve(f.q))
    @test available_timestamps(mds, f.ts1, f.ts2) == [f.ts1, f.ts2]
end
