# Helpers to construct synthetic chains with known IVs.

const _SURF_UNDERLYING = Underlying("SPY")

function _synth_quote(strike::Float64, expiry::DateTime, otype::OptionType,
                      ts::DateTime, spot::Float64, sigma::Float64,
                      r::Float64, q::Float64)
    T = time_to_expiry(expiry, ts)
    mark = bs_price(spot, strike, T, sigma, otype; r=r, q=q)
    OptionQuote(
        "X", _SURF_UNDERLYING, expiry, strike, otype,
        missing, missing, mark, missing, missing, missing, ts,
    )
end

function _synth_chain(ts, spot, r, q, slices)
    # slices :: Vector{(expiry, Vector{(strike, sigma)})}
    out = OptionQuote[]
    for (expiry, ks) in slices
        for (k, sigma) in ks
            otype = k >= spot ? Call : Put
            push!(out, _synth_quote(k, expiry, otype, ts, spot, sigma, r, q))
        end
    end
    out
end

@testset "ExpirySlice: validation" begin
    e = DateTime(2024, 2, 15, 21, 0)
    @test_throws ArgumentError ExpirySlice(e, 0.1, Float64[], Float64[])
    @test_throws ArgumentError ExpirySlice(e, 0.1, [100.0, 105.0], [0.2])
    @test_throws ArgumentError ExpirySlice(e, 0.1, [105.0, 100.0], [0.2, 0.21])
    @test_throws ArgumentError ExpirySlice(e, 0.1, [100.0, 100.0], [0.2, 0.21])
end

@testset "build_surface: recovers IVs from synthetic chain" begin
    ts = DateTime(2024, 1, 15, 15, 30)
    spot, r, q = 480.0, 0.04, 0.015
    e1 = DateTime(2024, 2, 16, 21, 0)
    e2 = DateTime(2024, 3, 15, 21, 0)
    slices = [
        (e1, [(460.0, 0.22), (470.0, 0.20), (480.0, 0.18), (490.0, 0.19), (500.0, 0.21)]),
        (e2, [(460.0, 0.24), (480.0, 0.20), (500.0, 0.22)]),
    ]
    chain = _synth_chain(ts, spot, r, q, slices)

    s = build_surface(chain, spot, r, q)
    @test s isa RawSurface
    @test s.timestamp == ts
    @test s.spot == spot
    @test expiries(s) == [e1, e2]

    sl1 = get_slice(s, e1)
    @test sl1 !== nothing
    @test sl1.strikes == [460.0, 470.0, 480.0, 490.0, 500.0]
    for (k, sigma_true) in zip(sl1.strikes, [0.22, 0.20, 0.18, 0.19, 0.21])
        @test iv(s, e1, k) ≈ sigma_true atol = 1e-4
    end
end

@testset "build_surface: interpolation between strikes" begin
    ts = DateTime(2024, 1, 15, 15, 30)
    spot, r, q = 100.0, 0.0, 0.0
    e1 = DateTime(2024, 2, 16, 21, 0)
    slices = [(e1, [(90.0, 0.25), (110.0, 0.15)])]
    chain = _synth_chain(ts, spot, r, q, slices)
    s = build_surface(chain, spot, r, q)

    # log(90/100) ≈ -0.1054, log(110/100) ≈ 0.0953. Midpoint = log(99.499..)
    x_mid = 0.5 * (log(90/100) + log(110/100))
    k_mid = spot * exp(x_mid)
    iv_mid = iv(s, e1, k_mid)
    @test iv_mid ≈ 0.20 atol = 1e-6

    # Out-of-range flat-extrapolates
    @test iv(s, e1, 50.0) ≈ 0.25 atol = 1e-6
    @test iv(s, e1, 200.0) ≈ 0.15 atol = 1e-6
end

@testset "build_surface: drops expired expiries" begin
    ts = DateTime(2024, 1, 15, 15, 30)
    spot, r, q = 100.0, 0.04, 0.0
    e_past = DateTime(2024, 1, 10, 21, 0)
    e_future = DateTime(2024, 2, 16, 21, 0)
    slices = [
        (e_past,   [(100.0, 0.20)]),
        (e_future, [(100.0, 0.20)]),
    ]
    # Build chain manually because _synth_quote would compute tau<0 for past.
    chain = OptionQuote[]
    push!(chain, _synth_quote(100.0, e_future, Call, ts, spot, 0.20, r, q))
    # Past-expiry quote: dummy mark so it can't successfully be inverted.
    push!(chain, OptionQuote("Y", _SURF_UNDERLYING, e_past, 100.0, Call,
                             missing, missing, 1.0, missing, missing, missing, ts))
    s = build_surface(chain, spot, r, q)
    @test expiries(s) == [e_future]
end

@testset "build_surface: drops strikes with missing/invalid marks" begin
    ts = DateTime(2024, 1, 15, 15, 30)
    spot, r, q = 100.0, 0.04, 0.0
    e1 = DateTime(2024, 2, 16, 21, 0)
    good = _synth_quote(100.0, e1, Call, ts, spot, 0.20, r, q)
    missing_mark = OptionQuote("M", _SURF_UNDERLYING, e1, 105.0, Call,
                               missing, missing, missing, missing, missing, missing, ts)
    zero_mark = OptionQuote("Z", _SURF_UNDERLYING, e1, 110.0, Call,
                            missing, missing, 0.0, missing, missing, missing, ts)
    s = build_surface([good, missing_mark, zero_mark], spot, r, q)
    sl = get_slice(s, e1)
    @test sl.strikes == [100.0]
end

@testset "build_surface: empty chain throws" begin
    @test_throws ArgumentError build_surface(OptionQuote[], 100.0, 0.04, 0.0)
end

@testset "build_surface: all-expired returns nothing" begin
    ts = DateTime(2024, 1, 15, 15, 30)
    e_past = DateTime(2024, 1, 10, 21, 0)
    q = OptionQuote("X", _SURF_UNDERLYING, e_past, 100.0, Call,
                    missing, missing, 1.0, missing, missing, missing, ts)
    @test build_surface([q], 100.0, 0.04, 0.0) === nothing
end

@testset "Surface queries: error on unknown expiry" begin
    ts = DateTime(2024, 1, 15, 15, 30)
    spot, r, q = 100.0, 0.04, 0.0
    e1 = DateTime(2024, 2, 16, 21, 0)
    chain = [_synth_quote(100.0, e1, Call, ts, spot, 0.20, r, q)]
    s = build_surface(chain, spot, r, q)
    e_missing = DateTime(2025, 1, 1, 21, 0)
    @test get_slice(s, e_missing) === nothing
    @test_throws ArgumentError iv(s, e_missing, 100.0)
    @test_throws ArgumentError price(s, e_missing, 100.0, Call)
    @test_throws ArgumentError forward(s, e_missing)
end

@testset "Surface queries: price/delta/gamma/vega match BS at quoted strike" begin
    ts = DateTime(2024, 1, 15, 15, 30)
    spot, r, q = 100.0, 0.04, 0.015
    e1 = DateTime(2024, 4, 19, 20, 0)
    sigma = 0.25
    chain = [_synth_quote(100.0, e1, Call, ts, spot, sigma, r, q)]
    s = build_surface(chain, spot, r, q)
    sl = get_slice(s, e1)

    @test price(s, e1, 100.0, Call) ≈ bs_price(spot, 100.0, sl.tau, sl.ivs[1], Call; r=r, q=q) atol = 1e-6
    @test delta(s, e1, 100.0, Call) ≈ bs_delta(spot, 100.0, sl.tau, sl.ivs[1], Call; r=r, q=q) atol = 1e-6
    @test gamma(s, e1, 100.0)       ≈ bs_gamma(spot, 100.0, sl.tau, sl.ivs[1]; r=r, q=q) atol = 1e-6
    @test vega(s,  e1, 100.0)       ≈ bs_vega(spot,  100.0, sl.tau, sl.ivs[1]; r=r, q=q) atol = 1e-6
end

@testset "Surface queries: forward" begin
    ts = DateTime(2024, 1, 15, 15, 30)
    spot, r, q = 100.0, 0.04, 0.015
    e1 = DateTime(2024, 4, 19, 20, 0)
    chain = [_synth_quote(100.0, e1, Call, ts, spot, 0.25, r, q)]
    s = build_surface(chain, spot, r, q)
    sl = get_slice(s, e1)
    @test forward(s, e1) ≈ spot * exp((r - q) * sl.tau) atol = 1e-12
end
