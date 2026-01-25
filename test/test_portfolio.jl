@testset "Portfolio Management" begin
    # Create test data
    expiry = DateTime(2026, 2, 1, 8, 0, 0)  # 08:00 UTC
    now = DateTime(2026, 1, 25, 12, 0, 0)
    
    # Create a simple trade
    trade = Trade(BTC, 100000.0, expiry, Call, now; direction=1, quantity=1.0)
    
    @testset "Portfolio Construction" begin
        portfolio = Portfolio()
        @test num_positions(portfolio) == 0
        @test portfolio.cash == 0.0
        @test portfolio.realized_pnl == 0.0
    end
    
    @testset "Portfolio with Initial Cash" begin
        portfolio = Portfolio(initial_cash=1.0)
        @test portfolio.cash == 1.0
    end
    
    # Load real data for integration tests
    data_path = joinpath(@__DIR__, "..", "data")
    sample_file = joinpath(data_path, "vols_20260117.parquet")
    
    if isfile(sample_file)
        store = LocalDataStore(data_path)
        records = load_all(store; underlying=BTC)
        
        # Build a surface from first hour
        by_hour = split_by_timestamp(records, Hour(1))
        first_ts = minimum(keys(by_hour))
        surface = build_surface(by_hour[first_ts])
        
        # Get a valid strike/expiry from the surface
        point = first(surface.points)
        strike = surface.spot * exp(point.log_moneyness)
        
        # Find corresponding record for expiry
        sample_record = first(filter(r -> !ismissing(r.mark_iv), by_hour[first_ts]))
        test_expiry = sample_record.expiry
        test_strike = sample_record.strike
        
        test_trade = Trade(BTC, test_strike, test_expiry, Call, surface.timestamp; 
                           direction=1, quantity=1.0)
        
        @testset "Add Position" begin
            portfolio = Portfolio()

            # Check that we can find this option on the surface
            # Use :ask since test_trade is a buy (direction=1)
            σ = find_vol(surface, test_strike, test_expiry; side=:ask)
            if !ismissing(σ)
                pos = add_position!(portfolio, test_trade, surface)

                @test num_positions(portfolio) == 1
                @test pos.trade == test_trade
                @test pos.entry_vol == σ  # Now uses ask vol for buys
                @test pos.entry_price > 0
                @test portfolio.cash < 0  # We paid for the option
            end
        end
        
        @testset "Position Value & Greeks" begin
            portfolio = Portfolio()
            
            σ = find_vol(surface, test_strike, test_expiry)
            if !ismissing(σ)
                pos = add_position!(portfolio, test_trade, surface)
                
                # Position value should be positive for long call
                val = position_value(pos, surface)
                @test val >= 0
                
                # Delta should be between 0 and 1 for long call
                delta = position_delta(pos, surface)
                @test 0.0 <= delta <= 1.0
                
                # Vega should be positive for long option
                vega = position_vega(pos, surface)
                @test vega >= 0
            end
        end
        
        @testset "Mark to Market" begin
            portfolio = Portfolio(initial_cash=0.1)
            
            σ = find_vol(surface, test_strike, test_expiry)
            if !ismissing(σ)
                pos = add_position!(portfolio, test_trade, surface)
                
                snapshot = mark_to_market(portfolio, surface)
                
                @test snapshot.positions == 1
                @test snapshot.timestamp == surface.timestamp
                @test snapshot.realized_pnl == 0.0  # Nothing closed yet
                
                # With same surface, unrealized P&L should be ~0 (minus spread)
                @test abs(snapshot.unrealized_pnl) < surface.spot * 0.01
            end
        end
        
        @testset "Close Position" begin
            portfolio = Portfolio(initial_cash=0.1)
            
            σ = find_vol(surface, test_strike, test_expiry)
            if !ismissing(σ)
                pos = add_position!(portfolio, test_trade, surface)
                initial_cash = portfolio.cash
                
                # Close at same surface - P&L should be ~0
                pnl = close_position!(portfolio, pos, surface)
                
                @test num_positions(portfolio) == 0
                @test abs(pnl) < surface.spot * 0.01  # Small due to same prices
                @test length(portfolio.trade_log) == 2  # Open + close
            end
        end
        
        @testset "Trade Log" begin
            portfolio = Portfolio()
            
            σ = find_vol(surface, test_strike, test_expiry)
            if !ismissing(σ)
                pos = add_position!(portfolio, test_trade, surface)
                
                @test length(portfolio.trade_log) == 1
                @test portfolio.trade_log[1].action == :open
                @test portfolio.trade_log[1].position_id == pos.id
                
                close_position!(portfolio, pos, surface)
                
                @test length(portfolio.trade_log) == 2
                @test portfolio.trade_log[2].action == :close
            end
        end
        
        @testset "Record Snapshot History" begin
            portfolio = Portfolio()
            
            σ = find_vol(surface, test_strike, test_expiry)
            if !ismissing(σ)
                add_position!(portfolio, test_trade, surface)
                
                record_snapshot!(portfolio, surface)
                
                @test length(portfolio.history) == 1
                @test portfolio.history[1].positions == 1
            end
        end
        
        @testset "Position Queries" begin
            portfolio = Portfolio()
            
            σ = find_vol(surface, test_strike, test_expiry)
            if !ismissing(σ)
                add_position!(portfolio, test_trade, surface)
                
                # Query by underlying
                btc_pos = get_positions(portfolio; underlying=BTC)
                eth_pos = get_positions(portfolio; underlying=ETH)
                
                @test length(btc_pos) == 1
                @test length(eth_pos) == 0
                
                # Query by option type
                calls = get_positions(portfolio; option_type=Call)
                puts = get_positions(portfolio; option_type=Put)
                
                @test length(calls) == 1
                @test length(puts) == 0
            end
        end
    else
        @info "Test data not found at $sample_file, skipping Portfolio integration tests"
    end
end
