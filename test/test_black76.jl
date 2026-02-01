@testset "Black-76 Pricing" begin
    # Test ATM call/put prices
    @testset "ATM Options" begin
        F = 100.0
        K = 100.0
        T = 1.0
        σ = 0.50
        
        call_price = black76_price(F, K, T, σ, Call)
        put_price = black76_price(F, K, T, σ, Put)
        
        # ATM call and put should have same price (put-call parity with r=0)
        @test call_price ≈ put_price atol=1e-10
        
        # Known value check: ATM with 50% vol, 1yr should be ~19.74% of forward
        @test call_price / F ≈ 0.1974 atol=0.001
    end
    
    # Test put-call parity: C - P = (F - K) * e^(-rT) for r=0 => C - P = F - K
    @testset "Put-Call Parity" begin
        F = 100.0
        K = 90.0  # ITM call
        T = 0.5
        σ = 0.30
        
        call_price = black76_price(F, K, T, σ, Call)
        put_price = black76_price(F, K, T, σ, Put)
        
        @test call_price - put_price ≈ (F - K) atol=1e-10
    end
    
    # Test edge cases
    @testset "Edge Cases" begin
        F = 100.0
        K = 100.0
        σ = 0.30
        
        # T = 0: should return intrinsic value
        @test black76_price(F, 90.0, 0.0, σ, Call) ≈ 10.0  # ITM call
        @test black76_price(F, 110.0, 0.0, σ, Call) ≈ 0.0  # OTM call
        @test black76_price(F, 110.0, 0.0, σ, Put) ≈ 10.0  # ITM put
        @test black76_price(F, 90.0, 0.0, σ, Put) ≈ 0.0   # OTM put
        
        # σ = 0: should return discounted intrinsic value
        @test black76_price(F, 90.0, 1.0, 0.0, Call) ≈ 10.0  # ITM call, no discounting since r=0
        @test black76_price(F, 110.0, 1.0, 0.0, Call) ≈ 0.0  # OTM call
    end
    
    # Test discounting with non-zero rate
    @testset "Discounting" begin
        F = 100.0
        K = 100.0
        T = 1.0
        σ = 0.30
        r = 0.05  # 5% interest rate
        
        price_no_r = black76_price(F, K, T, σ, Call)
        price_with_r = black76_price(F, K, T, σ, Call; r=r)
        
        # Price with discounting should be smaller
        @test price_with_r < price_no_r
        
        # Ratio should be approximately the discount factor
        @test price_with_r / price_no_r ≈ exp(-r * T) atol=0.01
    end
    
    # Test vol_to_price returns fraction of underlying
    @testset "vol_to_price" begin
        F = 100.0
        K = 100.0
        T = 1.0
        σ = 0.50
        
        price_frac = vol_to_price(σ, F, K, T, Call)
        abs_price = black76_price(F, K, T, σ, Call)
        
        @test price_frac ≈ abs_price / F atol=1e-10
    end
    
    # Test price_to_iv (inverse of vol_to_price)
    @testset "price_to_iv Round-Trip" begin
        F = 100.0
        K = 100.0
        T = 1.0
        
        # Test various volatilities
        for σ in [0.20, 0.50, 0.80, 1.20]
            price = vol_to_price(σ, F, K, T, Call)
            recovered_σ = price_to_iv(price, F, K, T, Call)
            @test recovered_σ ≈ σ atol=1e-6
        end
        
        # Test OTM put
        σ = 0.40
        price = vol_to_price(σ, F, 120.0, T, Put)
        recovered_σ = price_to_iv(price, F, 120.0, T, Put)
        @test recovered_σ ≈ σ atol=1e-6
        
        # Test deep OTM call
        σ = 0.60
        price = vol_to_price(σ, F, 150.0, T, Call)
        recovered_σ = price_to_iv(price, F, 150.0, T, Call)
        @test recovered_σ ≈ σ atol=1e-6
    end
    
    # Test black76_vega
    @testset "black76_vega" begin
        F = 100.0
        K = 100.0
        T = 1.0
        σ = 0.50
        
        # Vega should be positive
        vega = black76_vega(F, K, T, σ, Call)
        @test vega > 0
        
        # ATM vega is same for call and put
        vega_put = black76_vega(F, K, T, σ, Put)
        @test vega ≈ vega_put atol=1e-10
        
        # Vega numerical check
        ε = 0.0001
        price_up = black76_price(F, K, T, σ + ε, Call)
        price_down = black76_price(F, K, T, σ - ε, Call)
        numerical_vega = (price_up - price_down) / (2 * ε)
        @test abs(vega - numerical_vega) < 1e-4
    end
end

@testset "Mark IV to Mark Price Validation" begin
    # Load sample data (expiry is normalized to 08:00 UTC by the reader)
    data_path = joinpath(@__DIR__, "..", "data", "vols_20260117.parquet")
    if isfile(data_path)
        records = read_deribit_option_records(data_path; where="")
        
        # Filter valid records: non-missing mark_iv, mark_price, and T > 7 days
        sample = filter(records) do record
            !ismissing(record.mark_iv) && 
            !ismissing(record.mark_price) && 
            time_to_expiry(record.expiry, record.timestamp) > 7 / 365.25
        end
        
        # Test first 50 valid records
        test_sample = first(sample, 50)
        
        @testset "Sample options from $(length(test_sample)) records" for record in test_sample
            computed = vol_to_price(record)
            actual = record.mark_price
            
            # Allow 0.6% relative tolerance
            @test computed ≈ actual rtol=0.006
        end
    else
        @info "Test data not found at $data_path, skipping validation test"
    end
end

@testset "Bid/Ask IV Computation" begin
    data_path = joinpath(@__DIR__, "..", "data", "vols_20260117.parquet")
    if isfile(data_path)
        records = read_deribit_option_records(data_path; where="")

        # Find records with both bid and ask prices
        sample = filter(records) do record
            !ismissing(record.bid_price) && 
            !ismissing(record.ask_price) && 
            !ismissing(record.mark_iv) &&
            time_to_expiry(record.expiry, record.timestamp) > 7 / 365.25
        end
        
        test_sample = first(sample, 20)
        
        @testset "Bid/Ask IV for $(length(test_sample)) records" for record in test_sample
            bid_volatility = bid_iv(record)
            ask_volatility = ask_iv(record)
            mark_volatility = record.mark_iv / 100.0
            
            # Bid IV should exist and be reasonable (can be 0.0 if at intrinsic)
            @test !ismissing(bid_volatility)
            @test bid_volatility >= 0.0
            @test bid_volatility < 5.0  # Max reasonable vol
            
            # Ask IV should exist and be reasonable
            @test !ismissing(ask_volatility)
            @test ask_volatility >= 0.0
            @test ask_volatility < 5.0
            
            # Bid IV ≤ Mark IV ≤ Ask IV (bid-ask spread)
            # Using 2% tolerance since mark_iv is computed by Deribit with different methodology
            @test bid_volatility ≤ mark_volatility + 0.02
            @test ask_volatility ≥ mark_volatility - 0.02
        end
    else
        @info "Test data not found at $data_path, skipping Bid/Ask IV test"
    end
end
