@testset "ShortStrangleStrategy" begin
    underlying = Underlying("SPY")
    spot = 500.0
    rate = 0.045
    div_yield = 0.013

    function make_rec(strike, otype, bid, ask, iv, ts, exp_ts)
        OptionRecord(
            "TEST", underlying, exp_ts, strike, otype,
            bid, ask, (bid + ask) / 2, iv,
            100.0, 50.0, spot, ts,
        )
    end

    function make_surface(entry_ts::DateTime, exp_ts::DateTime)
        recs = OptionRecord[]
        for K in 470:530
            iv = 18.0 + 0.05 * abs(K - 500)
            if K <= 500
                push!(recs, make_rec(Float64(K), Put,
                    0.001 + 0.0005 * (500 - K) / 30,
                    0.0015 + 0.0006 * (500 - K) / 30, iv, entry_ts, exp_ts))
            end
            if K >= 500
                push!(recs, make_rec(Float64(K), Call,
                    0.001 + 0.0005 * (K - 500) / 30,
                    0.0015 + 0.0006 * (K - 500) / 30, iv, entry_ts, exp_ts))
            end
        end
        build_surface(recs)
    end

    base_date = Date(2026, 3, 2)
    n_days = 3
    surfaces = Dict{DateTime,VolatilitySurface}()
    spots = Dict{DateTime,Float64}()
    entry_tss = DateTime[]
    for i in 0:(n_days - 1)
        d = base_date + Day(i)
        ents = DateTime(d) + Hour(14)
        exp_ts = ents + Hour(2)
        surfaces[ents] = make_surface(ents, exp_ts)
        spots[exp_ts] = spot + (i % 2 == 0 ? 0.5 : -0.5)
        push!(entry_tss, ents)
    end
    source = DictDataSource(surfaces, spots)

    @testset "Constructor defaults" begin
        sel = delta_strangle_selector(0.20, 0.05; rate=rate, div_yield=div_yield)
        strat = ShortStrangleStrategy(entry_tss, Hour(2), sel)
        @test strat.expiry_interval == Hour(2)
        @test strat.sizer isa FixedSize
        @test strat.sizer.quantity == 1.0
        @test entry_schedule(strat) == entry_tss
    end

    @testset "Runs end-to-end via backtest_strategy" begin
        sel = delta_strangle_selector(0.20, 0.05; rate=rate, div_yield=div_yield)
        strat = ShortStrangleStrategy(entry_tss, Hour(2), sel)
        result = backtest_strategy(strat, source)

        # Should place positions; 2 legs per entry
        @test length(result.positions) > 0
        @test length(result.positions) % 2 == 0
        @test length(result.positions) ÷ 2 == n_days

        # Verify legs are short put + short call
        for i in 1:2:length(result.positions)
            put_pos = result.positions[i]
            call_pos = result.positions[i + 1]
            @test put_pos.trade.option_type == Put
            @test put_pos.trade.direction == -1
            @test call_pos.trade.option_type == Call
            @test call_pos.trade.direction == -1
            # short put strike < short call strike
            @test put_pos.trade.strike < call_pos.trade.strike
        end
    end

    @testset "Custom sizer" begin
        sel = delta_strangle_selector(0.20, 0.05; rate=rate, div_yield=div_yield)
        strat = ShortStrangleStrategy(entry_tss, Hour(2), sel; sizer=FixedSize(2.5))
        result = backtest_strategy(strat, source)
        @test all(p.trade.quantity == 2.5 for p in result.positions)
    end

    @testset "Sizer returning 0 skips entry" begin
        sel = delta_strangle_selector(0.20, 0.05; rate=rate, div_yield=div_yield)
        zero_sizer = ctx -> 0.0
        strat = ShortStrangleStrategy(entry_tss, Hour(2), sel; sizer=zero_sizer)
        result = backtest_strategy(strat, source)
        @test isempty(result.positions)
    end

    @testset "Selector returning nothing skips entry" begin
        nothing_sel = ctx -> nothing
        strat = ShortStrangleStrategy(entry_tss, Hour(2), nothing_sel)
        result = backtest_strategy(strat, source)
        @test isempty(result.positions)
    end

    @testset "delta_strangle_selector picks correct delta sides" begin
        sel = delta_strangle_selector(0.20, 0.05; rate=rate, div_yield=div_yield)
        history_view = HistoricalView(source, DateTime(0))
        ctx = StrikeSelectionContext(surfaces[entry_tss[1]],
                                       entry_tss[1] + Hour(2), history_view)
        result = sel(ctx)
        @test result !== nothing
        sp_K, sc_K = result
        @test sp_K < spot
        @test sc_K > spot
    end
end
