@testset "Local Data Store" begin
    if !isdir(DEFAULT_STORE.root)
        @warn "Skipping Local Data tests: $(DEFAULT_STORE.root) not found"
    else

    # Helper: find first options date that also has a spot file
    function _first_date_with_spot(dates::Vector{Date}, symbol::String)::Union{Date,Nothing}
        for d in dates
            isfile(polygon_spot_path(DEFAULT_STORE, d, symbol)) && return d
        end
        return nothing
    end

    # ================================================================
    # 1. Path derivation — smoke-check that known files resolve
    # ================================================================
    @testset "Path derivation" begin
        spy_dates = available_polygon_dates(DEFAULT_STORE, "SPY")
        if !isempty(spy_dates)
            d = spy_dates[1]
            p = polygon_options_path(DEFAULT_STORE, d, "SPY")
            @test isfile(p)
            # lowercase symbol normalises to upper
            @test polygon_options_path(DEFAULT_STORE, d, "spy") == p
        end

        # Polygon spot — find a date where the spot file actually exists
        spy_dates = available_polygon_dates(DEFAULT_STORE, "SPY")
        spot_date = _first_date_with_spot(spy_dates, "SPY")
        if spot_date !== nothing
            p = polygon_spot_path(DEFAULT_STORE, spot_date, "SPY")
            @test isfile(p)
        end

        # Deribit history
        deribit_dates = available_deribit_dates(DEFAULT_STORE)
        if !isempty(deribit_dates)
            d = deribit_dates[1]
            p = deribit_history_path(DEFAULT_STORE, d)
            @test isfile(p)
        end

        # Deribit delivery prices
        dp = deribit_delivery_path(DEFAULT_STORE)
        @test isfile(dp)
    end

    # ================================================================
    # 2. Polygon options read
    # ================================================================
    @testset "Polygon options read" begin
        spy_dates = available_polygon_dates(DEFAULT_STORE, "SPY")
        # Need a date where both options and spot exist (reader requires spot dict)
        spot_date = _first_date_with_spot(spy_dates, "SPY")
        if spot_date !== nothing
            opt_path  = polygon_options_path(DEFAULT_STORE, spot_date, "SPY")
            spot_path = polygon_spot_path(DEFAULT_STORE, spot_date, "SPY")
            spots = read_polygon_spot_prices(spot_path; underlying="SPY")

            records = read_polygon_option_records(opt_path, spots; where="", min_volume=0)
            @test length(records) > 0
            @test all(r -> r.spot > 0, records)
            @test all(r -> r.strike > 0, records)
            @test all(r -> r.option_type ∈ (Call, Put), records)
        end
    end

    # ================================================================
    # 3. Polygon spot read
    # ================================================================
    @testset "Polygon spot read" begin
        spy_dates = available_polygon_dates(DEFAULT_STORE, "SPY")
        spot_date = _first_date_with_spot(spy_dates, "SPY")
        if spot_date !== nothing
            path  = polygon_spot_path(DEFAULT_STORE, spot_date, "SPY")
            spots = read_polygon_spot_prices(path; underlying="SPY")
            @test length(spots) > 0
            @test all(v -> v > 0, values(spots))
        end
    end

    # ================================================================
    # 4. Deribit history read
    # ================================================================
    @testset "Deribit history read" begin
        deribit_dates = available_deribit_dates(DEFAULT_STORE)
        if !isempty(deribit_dates)
            path = deribit_history_path(DEFAULT_STORE, deribit_dates[1])
            records = read_deribit_option_records(path; where="")
            @test length(records) > 0
            @test all(r -> ticker(r.underlying) ∈ ("BTC", "ETH"), records)
        end
    end

    # ================================================================
    # 5. Date scanning
    # ================================================================
    @testset "Date scanning" begin
        spy_dates = available_polygon_dates(DEFAULT_STORE, "SPY")
        @test length(spy_dates) > 0
        @test spy_dates == sort(spy_dates)   # sorted ascending

        deribit_dates = available_deribit_dates(DEFAULT_STORE)
        @test length(deribit_dates) > 0
        @test deribit_dates == sort(deribit_dates)

        # Non-existent symbol yields empty
        @test available_polygon_dates(DEFAULT_STORE, "ZZZZZ") == Date[]
    end

    end # if isdir
end
