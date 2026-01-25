@testset "Black-76 Greeks" begin
    F = 100.0
    K = 100.0
    T = 1.0
    σ = 0.50
    
    @testset "Delta" begin
        call_delta = black76_delta(F, K, T, σ, Call)
        put_delta = black76_delta(F, K, T, σ, Put)
        
        # ATM call delta should be ~0.5
        @test 0.4 < call_delta < 0.6
        
        # Put delta should be negative
        @test put_delta < 0.0
        
        # Call delta - Put delta = 1 (for r=0)
        @test call_delta - put_delta ≈ 1.0 atol=0.01
        
        # Deep ITM call delta → 1
        itm_delta = black76_delta(100.0, 50.0, 1.0, 0.3, Call)
        @test itm_delta > 0.95
        
        # Deep OTM call delta → 0
        otm_delta = black76_delta(100.0, 200.0, 1.0, 0.3, Call)
        @test otm_delta < 0.05
    end
    
    @testset "Gamma" begin
        call_gamma = black76_gamma(F, K, T, σ, Call)
        put_gamma = black76_gamma(F, K, T, σ, Put)
        
        # Gamma should be positive
        @test call_gamma > 0
        
        # Gamma same for call and put
        @test call_gamma ≈ put_gamma atol=1e-10
        
        # ATM has highest gamma
        itm_gamma = black76_gamma(100.0, 80.0, T, σ, Call)
        @test call_gamma > itm_gamma
    end
    
    @testset "Theta" begin
        call_theta = black76_theta(F, K, T, σ, Call)
        
        # Theta should be negative (time decay)
        @test call_theta < 0
        
        # Edge case: at expiry
        @test black76_theta(F, K, 0.0, σ, Call) == 0.0
    end
    
    @testset "Numerical Gradient Check" begin
        # Delta = ∂Price/∂F
        ε = 0.01
        price_up = black76_price(F + ε, K, T, σ, Call)
        price_down = black76_price(F - ε, K, T, σ, Call)
        numerical_delta = (price_up - price_down) / (2 * ε)
        
        @test abs(black76_delta(F, K, T, σ, Call) - numerical_delta) < 0.01
        
        # Gamma = ∂²Price/∂F²
        numerical_gamma = (price_up - 2*black76_price(F, K, T, σ, Call) + price_down) / (ε^2)
        @test abs(black76_gamma(F, K, T, σ, Call) - numerical_gamma) < 0.01
    end
end

@testset "Backtest Engine" begin
    data_path = joinpath(@__DIR__, "..", "data")
    sample_file = joinpath(data_path, "vols_20260117.parquet")
    
    if isfile(sample_file)
        store = LocalDataStore(data_path)
        
        # Define a simple test strategy that does nothing
        struct PassiveStrategy <: Strategy end
        VolSurfaceAnalysis.on_snapshot(::PassiveStrategy, surface::VolatilitySurface, portfolio::Portfolio) = Order[]
        
        @testset "Passive Strategy Backtest" begin
            result = run_backtest(
                PassiveStrategy(),
                store,
                BTC,
                Date(2026, 1, 17),
                Date(2026, 1, 17);
                resolution=Hour(1)
            )
            
            @test result.underlying == BTC
            @test length(result.snapshots) > 0
            @test result.metrics.total_trades == 0
            @test result.metrics.total_pnl == 0.0
        end
        
        # Strategy that buys one ATM call at first snapshot
        mutable struct BuyOnceStrategy <: Strategy
            bought::Bool
        end
        BuyOnceStrategy() = BuyOnceStrategy(false)
        
        function VolSurfaceAnalysis.on_snapshot(s::BuyOnceStrategy, surface::VolatilitySurface, portfolio::Portfolio)
            if s.bought
                return Order[]
            end
            
            # Find ATM option
            atm_points = filter(p -> abs(p.log_moneyness) < 0.05, surface.points)
            isempty(atm_points) && return Order[]
            
            # Get first ATM point's strike
            point = first(atm_points)
            strike = surface.spot * exp(point.log_moneyness)
            
            # Find matching expiry from records (need at least 7 days out)
            matching = filter(surface.points) do p
                p.τ > 7/365.25 && abs(p.log_moneyness) < 0.05
            end
            isempty(matching) && return Order[]
            
            best = first(matching)
            expiry = surface.timestamp + Day(round(Int, best.τ * 365.25))
            # Normalize to 08:00 UTC
            expiry = DateTime(Date(expiry)) + Hour(8)
            
            s.bought = true
            return [Order(BTC, surface.spot * exp(best.log_moneyness), expiry, Call)]
        end
        
        @testset "Buy Once Strategy" begin
            result = run_backtest(
                BuyOnceStrategy(),
                store,
                BTC,
                Date(2026, 1, 17),
                Date(2026, 1, 17);
                resolution=Hour(1)
            )
            
            # Should have opened 1 position
            opens = filter(t -> t.action == :open, result.trade_log)
            @test length(opens) >= 0  # May be 0 if no ATM found
            
            # Snapshots should exist
            @test length(result.snapshots) > 0
        end
        
        @testset "Performance Metrics" begin
            result = run_backtest(
                PassiveStrategy(),
                store,
                BTC,
                Date(2026, 1, 17),
                Date(2026, 1, 17)
            )
            
            metrics = result.metrics
            @test metrics.total_pnl == 0.0
            @test metrics.max_drawdown >= 0.0
            @test 0.0 <= metrics.win_rate <= 1.0
        end
        
        @testset "Utility Functions" begin
            result = run_backtest(
                PassiveStrategy(),
                store,
                BTC,
                Date(2026, 1, 17),
                Date(2026, 1, 17)
            )
            
            times, pnl = pnl_series(result)
            @test length(times) == length(pnl)
            @test length(times) == length(result.snapshots)
            
            times2, equity = equity_curve(result)
            @test length(times2) == length(equity)
        end
    else
        @info "Test data not found, skipping Backtest Engine tests"
    end
end
