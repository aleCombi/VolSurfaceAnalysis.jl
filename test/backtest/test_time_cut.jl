# Tests for TimeCutModelDataSource: every accessor must be masked at cutoff.

const _TC_UND = Underlying("SPY")

function _tc_quote(strike::Float64, expiry::DateTime, ts::DateTime,
                   spot::Float64, sigma::Float64, r::Float64, q::Float64)
    otype = strike >= spot ? Call : Put
    T = time_to_expiry(expiry, ts)
    mark = bs_price(spot, strike, T, sigma, otype; r=r, q=q)
    OptionQuote("X", _TC_UND, expiry, strike, otype,
                missing, missing, mark, missing, missing, missing, ts)
end

function _tc_fixture()
    ts1 = DateTime(2024, 1, 15, 15, 30)
    ts2 = DateTime(2024, 1, 15, 15, 31)
    ts3 = DateTime(2024, 1, 15, 15, 32)
    spot = 480.0; r = 0.04; q = 0.015
    expiry = DateTime(2024, 2, 16, 21, 0)
    mk(ts, sigma) = [_tc_quote(k, expiry, ts, spot, sigma, r, q)
                     for k in (470.0, 480.0, 490.0)]
    chains = Dict(ts1 => mk(ts1, 0.20), ts2 => mk(ts2, 0.21), ts3 => mk(ts3, 0.22))
    spots  = Dict(ts1 => spot, ts2 => spot, ts3 => spot)
    inner  = InMemoryDataSource(_TC_UND; chains=chains, spots=spots)
    mds    = ModelDataSource(inner; rate=FlatCurve(r), div=FlatCurve(q))
    (mds=mds, ts1=ts1, ts2=ts2, ts3=ts3, expiry=expiry, spot=spot, r=r, q=q)
end

@testset "TimeCutModelDataSource: ts <= cutoff passes through" begin
    f = _tc_fixture()
    cut = TimeCutModelDataSource(f.mds, f.ts2)

    @test get_chain(cut, f.ts1) !== nothing
    @test get_chain(cut, f.ts2) !== nothing
    @test get_spot(cut, f.ts1) == f.spot
    @test get_spot(cut, f.ts2) == f.spot
    @test get_surface(cut, f.ts1) isa RawSurface
    @test get_surface(cut, f.ts2) isa RawSurface
end

@testset "TimeCutModelDataSource: ts > cutoff masked" begin
    f = _tc_fixture()
    cut = TimeCutModelDataSource(f.mds, f.ts2)

    @test get_chain(cut, f.ts3) === nothing
    @test ismissing(get_spot(cut, f.ts3))
    @test get_surface(cut, f.ts3) === nothing
end

@testset "TimeCutModelDataSource: available_timestamps clips at cutoff" begin
    f = _tc_fixture()
    cut = TimeCutModelDataSource(f.mds, f.ts2)
    @test available_timestamps(cut, f.ts1, f.ts3) == [f.ts1, f.ts2]
    # Cutoff inside the upper bound still works.
    @test available_timestamps(cut, f.ts1, f.ts2) == [f.ts1, f.ts2]
end

@testset "TimeCutModelDataSource: rate and div passthrough" begin
    f = _tc_fixture()
    cut = TimeCutModelDataSource(f.mds, f.ts1)
    # Rate / div curves are math objects; the cut does not gate them.
    @test get_rate(cut, f.ts3) == f.r
    @test get_div(cut, f.ts3)  == f.q
end

@testset "TimeCutModelDataSource: surface cache shared with inner" begin
    f = _tc_fixture()
    cut = TimeCutModelDataSource(f.mds, f.ts2)
    s_via_cut = get_surface(cut, f.ts1)
    s_via_mds = get_surface(f.mds, f.ts1)
    @test s_via_cut === s_via_mds
end
