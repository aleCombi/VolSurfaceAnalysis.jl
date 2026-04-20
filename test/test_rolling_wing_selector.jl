@testset "RollingWingCondorSelector" begin
    # Build a small fixture: 6 trading days of synthetic surfaces, each with
    # puts and calls dense enough for delta selection across multiple wings.
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
        # Strikes 470..530 in $1 increments, IV smile flat at 18%
        recs = OptionRecord[]
        for K in 470:530
            iv = 18.0 + 0.05 * abs(K - 500)   # mild smile
            if K <= 500
                push!(recs, make_rec(Float64(K), Put,  0.001 + 0.0005 * (500 - K) / 30,
                                                       0.0015 + 0.0006 * (500 - K) / 30, iv, entry_ts, exp_ts))
            end
            if K >= 500
                push!(recs, make_rec(Float64(K), Call, 0.001 + 0.0005 * (K - 500) / 30,
                                                       0.0015 + 0.0006 * (K - 500) / 30, iv, entry_ts, exp_ts))
            end
        end
        build_surface(recs)
    end

    base_date = Date(2026, 3, 2)   # Monday
    n_days = 6
    surfaces = Dict{DateTime,VolatilitySurface}()
    spots = Dict{DateTime,Float64}()
    entry_tss = DateTime[]
    for i in 0:(n_days - 1)
        d = base_date + Day(i)
        ents = DateTime(d) + Hour(14)
        exp_ts = ents + Hour(2)
        surfaces[ents] = make_surface(ents, exp_ts)
        spots[exp_ts] = spot + (i % 2 == 0 ? 0.5 : -0.5)   # tiny moves
        push!(entry_tss, ents)
    end
    source = DictDataSource(surfaces, spots)

    @testset "Constructor" begin
        sel = RollingWingCondorSelector(;
            put_delta=0.20, call_delta=0.05,
            wing_widths=[1.0, 2.0, 5.0],
            train_days=2, test_days=1,
            rate=rate, div_yield=div_yield,
            max_tau_days=0.5,
        )
        @test sel.config.train_days == 2
        @test sel.config.test_days == 1
        @test sel.config.step_days == 1   # defaults to test_days
        @test sel.state.current_wing == 5.0   # defaults to widest
        @test isempty(sel.state.history)
        @test sel.state.last_window_idx == -1
    end

    @testset "Runs end-to-end via IronCondorStrategy" begin
        sel = RollingWingCondorSelector(;
            put_delta=0.20, call_delta=0.05,
            wing_widths=[1.0, 2.0, 5.0],
            train_days=2, test_days=1,
            rate=rate, div_yield=div_yield,
            max_tau_days=0.5,
        )
        strat = IronCondorStrategy(entry_tss, Hour(2), sel)
        result = backtest_strategy(strat, source)

        # Should have placed positions on most/all entries (4 legs each)
        @test length(result.positions) > 0
        @test length(result.positions) % 4 == 0

        # History should have grown one entry per accepted entry timestamp
        n_entries = length(sel.state.history)
        @test n_entries == length(result.positions) ÷ 4
        @test n_entries >= 4   # at least most of the 6 days

        # Each history entry has the expected named-tuple fields
        h1 = sel.state.history[1]
        @test haskey(h1, :entry_ts)
        @test haskey(h1, :sp_K) && haskey(h1, :sc_K)
        @test haskey(h1, :wing_long_strikes)
        @test haskey(h1, :entry_credits_per_wing)
        @test length(h1.wing_long_strikes) == 3
        @test length(h1.entry_credits_per_wing) == 3
        @test h1.chosen_wing == 5.0   # initial wing before any refresh

        # first_entry_date should anchor to the first accepted entry
        @test sel.state.first_entry_date == base_date
    end

    @testset "max_tau filter" begin
        # Selector with max_tau_days=0 should reject every entry
        sel = RollingWingCondorSelector(;
            put_delta=0.20, call_delta=0.05,
            wing_widths=[2.0, 5.0],
            train_days=2, test_days=1,
            rate=rate, div_yield=div_yield,
            max_tau_days=0.0001,
        )
        strat = IronCondorStrategy(entry_tss, Hour(2), sel)
        result = backtest_strategy(strat, source)
        @test isempty(result.positions)
        @test isempty(sel.state.history)
        # first_entry_date should remain the sentinel — filter ran first
        @test sel.state.first_entry_date == Date(1900, 1, 1)
    end

    @testset "Wing refresh after training window" begin
        sel = RollingWingCondorSelector(;
            put_delta=0.20, call_delta=0.05,
            wing_widths=[1.0, 2.0, 5.0],
            train_days=2, test_days=1,
            rate=rate, div_yield=div_yield,
            max_tau_days=0.5,
        )
        strat = IronCondorStrategy(entry_tss, Hour(2), sel)
        backtest_strategy(strat, source)
        # After running 6 days with train=2/test=1, last_window_idx should
        # have advanced past the initial -1 sentinel.
        @test sel.state.last_window_idx >= 0
    end
end
