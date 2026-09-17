@testset "BS: time_to_expiry" begin
    ts = DateTime(2024, 1, 15, 14, 30)
    @test time_to_expiry(ts + Day(365), ts) ≈ 365.0 / 365.25 atol = 1e-12
    @test time_to_expiry(ts, ts) == 0.0
    @test time_to_expiry(ts - Day(1), ts) < 0.0
end

@testset "BS: put-call parity" begin
    S, K, T, sigma, r, q = 100.0, 100.0, 0.5, 0.20, 0.04, 0.015
    C = bs_price(S, K, T, sigma, Call; r=r, q=q)
    P = bs_price(S, K, T, sigma, Put;  r=r, q=q)
    parity = S * exp(-q * T) - K * exp(-r * T)
    @test C - P ≈ parity atol = 1e-6
end

@testset "BS: intrinsic at expiry" begin
    S, K = 105.0, 100.0
    @test bs_price(S, K, 0.0, 0.20, Call; r=0.04, q=0.015) == 5.0
    @test bs_price(S, K, 0.0, 0.20, Put;  r=0.04, q=0.015) == 0.0
    @test bs_price(K, S, 0.0, 0.20, Call; r=0.04, q=0.015) == 0.0
    @test bs_price(K, S, 0.0, 0.20, Put;  r=0.04, q=0.015) == 5.0
end

@testset "BS: zero-vol degenerates to discounted intrinsic" begin
    S, K, T = 105.0, 100.0, 0.5
    r, q = 0.04, 0.015
    forward = S * exp((r - q) * T)
    expected_call = exp(-r * T) * max(forward - K, 0.0)
    @test bs_price(S, K, T, 0.0, Call; r=r, q=q) ≈ expected_call atol = 1e-12
end

@testset "BS: ATM call delta ~ exp(-qT) * 0.5+" begin
    S, K, T, sigma, r, q = 100.0, 100.0, 0.5, 0.20, 0.04, 0.015
    d = bs_delta(S, K, T, sigma, Call; r=r, q=q)
    @test 0.5 < d < 0.7
    d_put = bs_delta(S, K, T, sigma, Put; r=r, q=q)
    @test -0.5 < d_put < 0.0
    # Call delta - put delta = exp(-q*T)
    @test d - d_put ≈ exp(-q * T) atol = 1e-3
end

@testset "BS: gamma and vega positive ATM" begin
    S, K, T, sigma, r, q = 100.0, 100.0, 0.5, 0.20, 0.04, 0.015
    @test bs_gamma(S, K, T, sigma; r=r, q=q) > 0
    @test bs_vega(S, K, T, sigma;  r=r, q=q) > 0
end

@testset "implied_vol: round-trip" begin
    S, K, T, r, q = 100.0, 100.0, 0.5, 0.04, 0.015
    for sigma_true in (0.10, 0.20, 0.35, 0.60, 1.00)
        for ot in (Call, Put)
            p = bs_price(S, K, T, sigma_true, ot; r=r, q=q)
            sigma = implied_vol(p, S, K, T, ot; r=r, q=q)
            @test sigma !== nothing
            @test sigma ≈ sigma_true atol = 1e-4
        end
    end
end

@testset "implied_vol: out-of-bracket returns nothing" begin
    S, K, T, r, q = 100.0, 100.0, 0.5, 0.04, 0.015
    @test implied_vol(-1.0, S, K, T, Call; r=r, q=q) === nothing
    # Price above deep-vol limit (price for sigma=5.0)
    p_hi = bs_price(S, K, T, 5.0, Call; r=r, q=q)
    @test implied_vol(p_hi + 1.0, S, K, T, Call; r=r, q=q) === nothing
end

@testset "implied_vol: at-expiry returns nothing" begin
    @test implied_vol(5.0, 100.0, 95.0, 0.0, Call; r=0.04, q=0.015) === nothing
end
