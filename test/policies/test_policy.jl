# Tests for the Policy abstraction and the NoOpPolicy base case.

const _PL_UND = Underlying("SPY")

function _pl_fixture()
    ts1 = DateTime(2024, 1, 15, 15, 30)
    spot = 480.0; r = 0.04; q = 0.015
    expiry = DateTime(2024, 2, 16, 21, 0)
    T = time_to_expiry(expiry, ts1)
    mark = bs_price(spot, 480.0, T, 0.20, Call; r=r, q=q)
    qte = OptionQuote("X", _PL_UND, expiry, 480.0, Call,
                      missing, missing, mark, missing, missing, missing, ts1)
    inner = InMemoryDataSource(_PL_UND;
        chains=Dict(ts1 => [qte]),
        spots=Dict(ts1 => spot))
    mds = ModelDataSource(inner; rate=FlatCurve(r), div=FlatCurve(q))
    (mds=mds, ts1=ts1)
end

@testset "NoOpPolicy: decide returns empty" begin
    f = _pl_fixture()
    cut = TimeCutModelDataSource(f.mds, f.ts1)
    @test decide(NoOpPolicy(), f.ts1, cut, Position[]) == Trade[]
end

# A custom Policy without a decide method must fall through to the
# error stub on the abstract type.
struct _UnimplementedPolicy <: Policy end

@testset "Policy: missing decide method errors" begin
    f = _pl_fixture()
    cut = TimeCutModelDataSource(f.mds, f.ts1)
    @test_throws ErrorException decide(_UnimplementedPolicy(), f.ts1, cut, Position[])
end

# ---- DailyShortStrangle ----------------------------------------------------

# Multi-strike, two-expiry InMemoryDataSource at a single timestamp, priced
# from a flat 20%-vol BS world so each slice cleanly inverts and the |Δ|
# bracket is wide.
function _strangle_fixture(; ts=DateTime(2024, 6, 3, 15, 45),
                            spot=480.0, r=0.045, q=0.013, sigma=0.20)
    e1 = DateTime(2024, 6, 4, 20, 0)   # ~1 DTE
    e2 = DateTime(2024, 6, 7, 20, 0)   # ~4 DTE
    strikes = 440.0:5.0:520.0          # 17 strikes spanning a wide delta range

    function mk_q(ts, K, expiry, otype)
        T = time_to_expiry(expiry, ts)
        mark = bs_price(spot, K, T, sigma, otype; r=r, q=q)
        spread = max(0.02, 0.01 * mark)
        bid, ask = mark - spread / 2, mark + spread / 2
        OptionQuote("X", _PL_UND, expiry, K, otype, bid, ask, mark,
                    missing, missing, missing, ts)
    end

    chain = OptionQuote[]
    for K in strikes, e in (e1, e2)
        otype = K >= spot ? Call : Put
        push!(chain, mk_q(ts, K, e, otype))
    end
    inner = InMemoryDataSource(_PL_UND;
        chains = Dict(ts => chain),
        spots  = Dict(ts => spot))
    mds = ModelDataSource(inner; rate=FlatCurve(r), div=FlatCurve(q))
    (mds=mds, ts=ts, e1=e1, e2=e2, spot=spot, strikes=collect(strikes))
end

@testset "DailyShortStrangle: constructor validation" begin
    und = _PL_UND
    @test_throws ArgumentError DailyShortStrangle(und, Time(15, 45), Day(1),
                                                  0.0,  0.20, 1.0)
    @test_throws ArgumentError DailyShortStrangle(und, Time(15, 45), Day(1),
                                                  1.0,  0.20, 1.0)
    @test_throws ArgumentError DailyShortStrangle(und, Time(15, 45), Day(1),
                                                  0.20, 1.5,  1.0)
    @test_throws ArgumentError DailyShortStrangle(und, Time(15, 45), Day(1),
                                                  0.20, 0.20, 0.0)
    @test_throws ArgumentError DailyShortStrangle(und, Time(15, 45), Day(0),
                                                  0.20, 0.20, 1.0)
    # Happy path with kwargs.
    p = DailyShortStrangle(; underlying=und, entry_time=Time(15, 45),
                           expiry_interval=Day(1),
                           put_delta=0.20, call_delta=0.20)
    @test p.quantity == 1.0
end

@testset "DailyShortStrangle: _first_expiry_on_or_after / _snap_to_sorted / _quoted_strikes" begin
    f = _strangle_fixture()
    surf = get_surface(f.mds, f.ts)
    @test VolSurfaceAnalysis._first_expiry_on_or_after(surf, f.ts) == f.e1
    @test VolSurfaceAnalysis._first_expiry_on_or_after(surf, f.e1) == f.e1
    @test VolSurfaceAnalysis._first_expiry_on_or_after(surf, f.e1 + Hour(1)) == f.e2
    @test VolSurfaceAnalysis._first_expiry_on_or_after(surf, f.e2 + Day(7)) === nothing

    strikes = [440.0, 445.0, 450.0, 480.0, 485.0, 520.0]
    @test VolSurfaceAnalysis._snap_to_sorted(strikes, 100.0)  == 440.0
    @test VolSurfaceAnalysis._snap_to_sorted(strikes, 999.0)  == 520.0
    @test VolSurfaceAnalysis._snap_to_sorted(strikes, 481.0)  == 480.0
    @test VolSurfaceAnalysis._snap_to_sorted(strikes, 483.0)  == 485.0
    @test VolSurfaceAnalysis._snap_to_sorted(strikes, 485.0)  == 485.0
    @test VolSurfaceAnalysis._snap_to_sorted(Float64[], 480.0) === nothing

    # _quoted_strikes filters chain by (expiry, underlying, option_type).
    chain = get_chain(f.mds, f.ts)
    puts_e1  = VolSurfaceAnalysis._quoted_strikes(chain, f.e1, _PL_UND, Put)
    calls_e1 = VolSurfaceAnalysis._quoted_strikes(chain, f.e1, _PL_UND, Call)
    @test all(K -> K <  f.spot, puts_e1)
    @test all(K -> K >= f.spot, calls_e1)
    @test issorted(puts_e1) && issorted(calls_e1)
end

@testset "DailyShortStrangle: gate skips non-entry timestamps cheaply" begin
    f = _strangle_fixture()
    p = DailyShortStrangle(; underlying=_PL_UND,
                           entry_time=Time(15, 45),
                           expiry_interval=Day(1),
                           put_delta=0.20, call_delta=0.20)
    off_ts = DateTime(2024, 6, 3, 15, 46)   # one minute off
    cut = TimeCutModelDataSource(f.mds, off_ts)
    @test decide(p, off_ts, cut, Position[]) == Trade[]
end

@testset "DailyShortStrangle: happy path opens two short legs at the right expiry" begin
    f = _strangle_fixture()
    p = DailyShortStrangle(; underlying=_PL_UND,
                           entry_time=Time(15, 45),
                           expiry_interval=Day(1),
                           put_delta=0.20, call_delta=0.20)
    cut = TimeCutModelDataSource(f.mds, f.ts)
    trades = decide(p, f.ts, cut, Position[])
    @test length(trades) == 2

    # Both legs short, same expiry == first slice after t + 1d.
    @test all(tr.direction == -1 for tr in trades)
    @test all(tr.quantity == 1.0  for tr in trades)
    @test all(tr.expiry == f.e1 for tr in trades)

    # One Put, one Call. Put strike < spot < Call strike.
    types = [tr.option_type for tr in trades]
    @test Call in types && Put in types
    put_K  = trades[findfirst(tr -> tr.option_type == Put,  trades)].strike
    call_K = trades[findfirst(tr -> tr.option_type == Call, trades)].strike
    @test put_K  < f.spot < call_K

    # Snap invariant: both strikes exist in the observed slice so
    # resolve_quote in the engine will match them.
    sl = get_slice(get_surface(f.mds, f.ts), f.e1)
    @test put_K  in sl.strikes
    @test call_K in sl.strikes

    # Chosen strikes should be the nearest-snapped to the |Δ|=0.20 targets.
    @test abs(delta(get_surface(f.mds, f.ts), f.e1, put_K,  Put))  ≈ 0.20 atol = 0.05
    @test abs(delta(get_surface(f.mds, f.ts), f.e1, call_K, Call)) ≈ 0.20 atol = 0.05
end

@testset "DailyShortStrangle: one-wing failure returns Trade[]" begin
    f = _strangle_fixture()
    # 0.99999 |Δ| exceeds even the deepest-ITM put's |Δ| on the slice
    # (~0.99996 at K=520, T~1d, σ=20%) -> put invert returns nothing,
    # so decide must skip the entry rather than trade only the call.
    p = DailyShortStrangle(; underlying=_PL_UND,
                           entry_time=Time(15, 45),
                           expiry_interval=Day(1),
                           put_delta=0.99999, call_delta=0.20)
    cut = TimeCutModelDataSource(f.mds, f.ts)
    @test decide(p, f.ts, cut, Position[]) == Trade[]
end

@testset "DailyShortStrangle: no surface available -> Trade[]" begin
    f = _strangle_fixture()
    later_ts = DateTime(2024, 6, 4, 15, 45)   # no chain at this ts in fixture
    p = DailyShortStrangle(; underlying=_PL_UND,
                           entry_time=Time(15, 45),
                           expiry_interval=Day(1),
                           put_delta=0.20, call_delta=0.20)
    cut = TimeCutModelDataSource(f.mds, later_ts)
    @test decide(p, later_ts, cut, Position[]) == Trade[]
end
