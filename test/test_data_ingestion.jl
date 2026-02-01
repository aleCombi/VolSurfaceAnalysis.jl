@testset "Data Ingestion" begin
    # ========================================================================
    # Core Type Tests
    # ========================================================================
    
    @testset "OptionType Enum" begin
        @test Call isa OptionType
        @test Put isa OptionType
        @test Call != Put
    end
    
    @testset "Underlying Type" begin
        btc = Underlying("BTC")
        eth = Underlying("ETH")
        spy = Underlying("SPY")
        
        @test ticker(btc) == "BTC"
        @test ticker(eth) == "ETH"
        @test ticker(spy) == "SPY"
        
        # Test string comparison
        @test btc == "BTC"
        @test "BTC" == btc
        @test btc == "btc"  # Case insensitive
        
        # Test case normalization
        @test Underlying("spy").ticker == "SPY"
        
        # Test display
        @test string(btc) == "BTC"
    end
    
    @testset "parse_option_type" begin
        @test VolSurfaceAnalysis.parse_option_type("C") == Call
        @test VolSurfaceAnalysis.parse_option_type("CALL") == Call
        @test VolSurfaceAnalysis.parse_option_type("call") == Call
        @test VolSurfaceAnalysis.parse_option_type("P") == Put
        @test VolSurfaceAnalysis.parse_option_type("PUT") == Put
        @test VolSurfaceAnalysis.parse_option_type("put") == Put
        
        @test_throws ErrorException VolSurfaceAnalysis.parse_option_type("X")
    end
    
    @testset "to_datetime" begin
        # DateTime passthrough
        dt = DateTime(2024, 1, 15, 10, 30)
        @test VolSurfaceAnalysis.to_datetime(dt) == dt
        
        # Date conversion
        d = Date(2024, 1, 15)
        @test VolSurfaceAnalysis.to_datetime(d) == DateTime(2024, 1, 15)
        
        # String conversion
        @test VolSurfaceAnalysis.to_datetime("2024-01-15T10:30:00") == DateTime(2024, 1, 15, 10, 30)
        
        # Unix milliseconds
        ms = Int64(1705314600000)  # 2024-01-15 10:30:00 UTC
        @test VolSurfaceAnalysis.to_datetime(ms) == DateTime(2024, 1, 15, 10, 30)
    end
    
    @testset "to_float_or_missing" begin
        @test VolSurfaceAnalysis.to_float_or_missing(5.0) == 5.0
        @test VolSurfaceAnalysis.to_float_or_missing(5) == 5.0
        @test ismissing(VolSurfaceAnalysis.to_float_or_missing(missing))
        @test ismissing(VolSurfaceAnalysis.to_float_or_missing(nothing))
    end
    
    # ========================================================================
    # Polygon Ticker Parsing
    # ========================================================================
    
    @testset "parse_polygon_ticker" begin
        # Standard SPY call
        underlying, expiry, option_type, strike = parse_polygon_ticker("O:SPY240129C00406000")
        @test underlying == "SPY"
        @test expiry == DateTime(2024, 1, 29, 21, 0, 0)  # 4pm ET = 21:00 UTC (EST)
        @test option_type == Call
        @test strike == 406.0
        
        # QQQ put
        underlying, expiry, option_type, strike = parse_polygon_ticker("O:QQQ240315P00380000")
        @test underlying == "QQQ"
        @test expiry == DateTime(2024, 3, 15, 20, 0, 0)  # 4pm ET = 20:00 UTC (EDT)
        @test option_type == Put
        @test strike == 380.0
        
        # Multi-letter ticker
        underlying, expiry, option_type, strike = parse_polygon_ticker("O:AAPL250117C00150000")
        @test underlying == "AAPL"
        @test expiry == DateTime(2025, 1, 17, 21, 0, 0)
        @test option_type == Call
        @test strike == 150.0
        
        # Fractional strike
        underlying, expiry, option_type, strike = parse_polygon_ticker("O:SPY240129C00406500")
        @test strike == 406.5
    end
    
    @testset "et_to_utc" begin
        # EST (winter) - 5 hours offset
        est_time = DateTime(2024, 1, 15, 16, 0, 0)  # 4pm ET
        utc_time = et_to_utc(est_time)
        @test utc_time == DateTime(2024, 1, 15, 21, 0, 0)
        
        # EDT (summer) - 4 hours offset
        edt_time = DateTime(2024, 7, 15, 16, 0, 0)  # 4pm ET
        utc_time = et_to_utc(edt_time)
        @test utc_time == DateTime(2024, 7, 15, 20, 0, 0)
    end
    
    # ========================================================================
    # SpotPrice and spot_dict
    # ========================================================================
    
    @testset "SpotPrice" begin
        spot = SpotPrice(Underlying("BTC"), 50000.0, DateTime(2024, 1, 15))
        @test spot.underlying == Underlying("BTC")
        @test spot.price == 50000.0
        @test spot.timestamp == DateTime(2024, 1, 15)
    end
    
    @testset "spot_dict" begin
        spots = [
            SpotPrice(Underlying("BTC"), 50000.0, DateTime(2024, 1, 15, 10, 0)),
            SpotPrice(Underlying("BTC"), 51000.0, DateTime(2024, 1, 15, 11, 0)),
            SpotPrice(Underlying("ETH"), 3000.0, DateTime(2024, 1, 15, 10, 0)),
        ]
        
        # All spots (BTC and ETH at 10:00 collide on the same key; last entry wins)
        dict_all = spot_dict(spots)
        @test length(dict_all) == 2
        @test dict_all[DateTime(2024, 1, 15, 10, 0)] == 3000.0
        
        # Filter by underlying
        dict_btc = spot_dict(spots; underlying="BTC")
        @test length(dict_btc) == 2
        @test dict_btc[DateTime(2024, 1, 15, 10, 0)] == 50000.0
        @test dict_btc[DateTime(2024, 1, 15, 11, 0)] == 51000.0
        
        dict_eth = spot_dict(spots; underlying=Underlying("ETH"))
        @test length(dict_eth) == 1
        @test dict_eth[DateTime(2024, 1, 15, 10, 0)] == 3000.0
    end
    
    # ========================================================================
    # Deribit Data Loading (if test data exists)
    # ========================================================================
    
    @testset "Deribit Data Loading" begin
        data_path = joinpath(@__DIR__, "..", "data", "vols_20260117.parquet")
        
        if isfile(data_path)
            @testset "read_deribit_parquet" begin
                quotes = read_deribit_parquet(data_path; where="underlying = 'BTC' LIMIT 10")
                @test length(quotes) <= 10
                @test all(q -> q isa DeribitQuote, quotes)
                @test all(q -> q.underlying == Underlying("BTC"), quotes)
                
                # Check expiry normalization to 08:00 UTC
                for q in quotes
                    @test Dates.hour(q.expiry) == 8
                    @test Dates.minute(q.expiry) == 0
                end
            end
            
            @testset "read_deribit_option_records" begin
                records = read_deribit_option_records(data_path; where="1=1 LIMIT 5")
                @test length(records) <= 5
                @test all(r -> r isa OptionRecord, records)
                
                # Check required fields
                for r in records
                    @test r.underlying isa Underlying
                    @test r.strike > 0
                    @test r.spot > 0
                    @test r.option_type in [Call, Put]
                    @test r.expiry > r.timestamp
                end
            end
            
            @testset "read_deribit_spot_prices" begin
                spots_dict = read_deribit_spot_prices(data_path; underlying="BTC")
                @test !isempty(spots_dict)
                @test all(v -> v > 0, values(spots_dict))
                @test all(k -> k isa DateTime, keys(spots_dict))
            end
            
            @testset "to_option_record(DeribitQuote)" begin
                quotes = read_deribit_parquet(data_path; where="1=1 LIMIT 1")
                if !isempty(quotes)
                    record = to_option_record(quotes[1])
                    @test record isa OptionRecord
                    @test record.instrument_name == quotes[1].instrument_name
                    @test record.underlying == quotes[1].underlying
                    @test record.strike == quotes[1].strike
                    @test record.expiry == quotes[1].expiry
                end
            end
        else
            @info "Deribit test data not found at $data_path, skipping Deribit tests"
        end
    end
    
    # ========================================================================
    # OptionRecord Structure
    # ========================================================================
    
    @testset "OptionRecord Construction" begin
        record = OptionRecord(
            "BTC-29DEC24-50000-C",
            Underlying("BTC"),
            DateTime(2024, 12, 29, 8, 0, 0),
            50000.0,
            Call,
            0.05,
            0.06,
            0.055,
            65.0,
            100.0,
            50.0,
            48000.0,
            DateTime(2024, 12, 1, 10, 0, 0)
        )
        
        @test record.instrument_name == "BTC-29DEC24-50000-C"
        @test record.underlying == Underlying("BTC")
        @test record.strike == 50000.0
        @test record.option_type == Call
        @test record.bid_price == 0.05
        @test record.ask_price == 0.06
        @test record.mark_price == 0.055
        @test record.mark_iv == 65.0
        @test record.spot == 48000.0
    end
    
    # ========================================================================
    # Integration: Full Pipeline Test
    # ========================================================================
    
    @testset "Data Pipeline Integration" begin
        data_path = joinpath(@__DIR__, "..", "data", "vols_20260117.parquet")
        
        if isfile(data_path)
            # Load records
            records = read_deribit_option_records(data_path; where="underlying = 'BTC' LIMIT 100")
            
            if !isempty(records)
                # Group by timestamp
                timestamps = unique(r.timestamp for r in records)
                @test length(timestamps) >= 1
                
                # Build surface from first timestamp
                ts = first(timestamps)
                ts_records = filter(r -> r.timestamp == ts, records)
                
                if length(ts_records) >= 4  # Need enough records for a surface
                    surface = build_surface(ts_records)
                    
                    @test surface isa VolatilitySurface
                    @test surface.spot > 0
                    @test surface.timestamp == ts
                    @test surface.underlying == Underlying("BTC")
                    @test !isempty(surface.points)
                    
                    # Verify surface points
                    for p in surface.points
                        @test p.τ > 0
                        @test p.vol > 0
                        @test p.vol < 10.0  # Reasonable IV bound
                    end
                end
            end
        else
            @info "Integration test skipped - no test data available"
        end
    end
end
