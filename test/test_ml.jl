using Flux

@testset "ML Module" begin
    # ================================================================
    # Fixture: minimal surface with puts and calls around spot
    # ================================================================
    entry_ts = DateTime(2026, 3, 10, 14, 0, 0)
    expiry_ts = DateTime(2026, 3, 11, 16, 0, 0)
    spot = 500.0
    underlying = Underlying("SPY")

    # Create synthetic option records at various strikes
    function make_rec(strike, otype, bid, ask, mark, iv)
        OptionRecord(
            "TEST", underlying, expiry_ts, strike, otype,
            bid, ask, mark, iv,
            100.0, 50.0,  # open_interest, volume
            spot, entry_ts
        )
    end

    records = OptionRecord[
        # Puts (OTM)
        make_rec(470.0, Put, 0.002, 0.004, 0.003, 25.0),
        make_rec(480.0, Put, 0.005, 0.008, 0.0065, 22.0),
        make_rec(490.0, Put, 0.010, 0.015, 0.0125, 20.0),
        make_rec(495.0, Put, 0.015, 0.022, 0.0185, 19.0),
        # ATM
        make_rec(500.0, Put, 0.030, 0.040, 0.035, 18.0),
        make_rec(500.0, Call, 0.030, 0.040, 0.035, 18.0),
        # Calls (OTM)
        make_rec(505.0, Call, 0.015, 0.022, 0.0185, 19.0),
        make_rec(510.0, Call, 0.010, 0.015, 0.0125, 20.0),
        make_rec(520.0, Call, 0.005, 0.008, 0.0065, 22.0),
        make_rec(530.0, Call, 0.002, 0.004, 0.003, 25.0),
    ]

    surface = build_surface(records)

    empty_source = DictDataSource(
        Dict{DateTime,VolatilitySurface}(),
        Dict{DateTime,Float64}()
    )
    history = HistoricalView(empty_source, DateTime(0))
    ctx = StrikeSelectionContext(surface, expiry_ts, history)

    # ================================================================
    # Feature extraction tests
    # ================================================================
    @testset "Surface Features" begin
        @testset "ATMImpliedVol" begin
            f = ATMImpliedVol(; rate=0.045, div_yield=0.015)
            val = f(ctx)
            @test val !== nothing
            @test val isa Float64
            @test 0.0 < val < 5.0  # reasonable IV (decimal)
        end

        @testset "DeltaSkew" begin
            f = DeltaSkew(0.25, :put; rate=0.045, div_yield=0.015)
            val = f(ctx)
            # May be nothing if delta resolution fails, but should be a number when it works
            if val !== nothing
                @test val isa Float64
            end
        end

        @testset "RiskReversal" begin
            f = RiskReversal(0.25; rate=0.045, div_yield=0.015)
            val = f(ctx)
            if val !== nothing
                @test val isa Float64
            end
        end

        @testset "Butterfly" begin
            f = Butterfly(0.25; rate=0.045, div_yield=0.015)
            val = f(ctx)
            if val !== nothing
                @test val isa Float64
            end
        end

        @testset "ATMSpread" begin
            val = ATMSpread()(ctx)
            @test val !== nothing
            @test val isa Float64
            @test val > 0.0  # spread should be positive
        end

        @testset "TotalVolume" begin
            val = TotalVolume()(ctx)
            @test val !== nothing
            @test val isa Float64
            @test val > 0.0
        end

        @testset "PutCallVolumeRatio" begin
            val = PutCallVolumeRatio()(ctx)
            @test val !== nothing
            @test val isa Float64
            @test val > 0.0
        end

        @testset "HourOfDay" begin
            val = HourOfDay()(ctx)
            @test val !== nothing
            @test val ≈ 14.0 / 24.0
        end

        @testset "DayOfWeek" begin
            val = DayOfWeek()(ctx)
            @test val !== nothing
            @test val isa Float64
        end

        @testset "TermSlope returns nothing with single expiry" begin
            f = TermSlope(; rate=0.045, div_yield=0.015)
            val = f(ctx)
            @test val === nothing  # only one expiry in our test data
        end

        @testset "History features return nothing with empty history" begin
            @test RealizedVol(; lookback=5)(ctx) === nothing
            @test VarianceRiskPremium(; lookback=5, rate=0.045, div_yield=0.015)(ctx) === nothing
            @test SpotMomentum(; lookback=5)(ctx) === nothing
            @test IVChange(; lookback=5, rate=0.045, div_yield=0.015)(ctx) === nothing
            @test IVPercentile(; lookback=5, rate=0.045, div_yield=0.015)(ctx) === nothing
        end
    end

    # ================================================================
    # History-based features with actual historical data
    # ================================================================
    @testset "History Features" begin
        # Build 25 historical surfaces with slightly different spots
        hist_surfaces = Dict{DateTime,VolatilitySurface}()
        hist_spots = Dict{DateTime,Float64}()
        for i in 0:24
            ts_i = entry_ts - Day(25 - i)
            spot_i = 490.0 + i * 0.5  # gentle uptrend: 490 → 502
            recs_i = OptionRecord[
                make_rec(spot_i - 10.0, Put, 0.010, 0.015, 0.0125, 20.0 + i * 0.1),
                make_rec(spot_i, Put, 0.030, 0.040, 0.035, 18.0 + i * 0.1),
                make_rec(spot_i, Call, 0.030, 0.040, 0.035, 18.0 + i * 0.1),
                make_rec(spot_i + 10.0, Call, 0.010, 0.015, 0.0125, 20.0 + i * 0.1),
            ]
            # Override spot in records
            recs_i = [OptionRecord(r.instrument_name, r.underlying, r.expiry, r.strike,
                r.option_type, r.bid_price, r.ask_price, r.mark_price, r.mark_iv,
                r.open_interest, r.volume, spot_i, ts_i) for r in recs_i]
            surf_i = build_surface(recs_i)
            hist_surfaces[ts_i] = surf_i
            hist_spots[ts_i] = spot_i
        end
        # Add current surface
        hist_surfaces[entry_ts] = surface
        hist_spots[entry_ts] = spot

        hist_source = DictDataSource(hist_surfaces, hist_spots)
        hist_view = HistoricalView(hist_source, entry_ts)
        hist_ctx = StrikeSelectionContext(surface, expiry_ts, hist_view)

        @testset "RealizedVol" begin
            val = RealizedVol(; lookback=20)(hist_ctx)
            @test val !== nothing
            @test val isa Float64
            @test val > 0.0
        end

        @testset "VarianceRiskPremium" begin
            val = VarianceRiskPremium(; lookback=20, rate=0.045, div_yield=0.015)(hist_ctx)
            @test val !== nothing
            @test val isa Float64
        end

        @testset "SpotMomentum" begin
            val5 = SpotMomentum(; lookback=5)(hist_ctx)
            @test val5 !== nothing
            @test val5 isa Float64
            val20 = SpotMomentum(; lookback=20)(hist_ctx)
            @test val20 !== nothing
            @test val20 isa Float64
        end

        @testset "IVChange" begin
            val = IVChange(; lookback=5, rate=0.045, div_yield=0.015)(hist_ctx)
            @test val !== nothing
            @test val isa Float64
        end

        @testset "IVPercentile" begin
            val = IVPercentile(; lookback=20, rate=0.045, div_yield=0.015)(hist_ctx)
            @test val !== nothing
            @test val isa Float64
            @test 0.0 <= val <= 1.0
        end

        @testset "SpotLogSig" begin
            f = SpotLogSig(; lookback=20, depth=3)
            @test logsig_dim(f) == 5  # 2 channels, depth 3

            val = f(hist_ctx)
            @test val !== nothing
            @test val isa Vector{Float64}
            @test length(val) == 5

            # With insufficient history, should return nothing
            short_f = SpotLogSig(; lookback=50, depth=3)
            @test short_f(hist_ctx) === nothing

            # Depth 2 should give fewer components
            f2 = SpotLogSig(; lookback=20, depth=2)
            @test logsig_dim(f2) == 3  # 2 channels, depth 2
            val2 = f2(hist_ctx)
            @test val2 !== nothing
            @test length(val2) == 3
        end
    end

    # ================================================================
    # Candidate feature tests
    # ================================================================
    @testset "Candidate Features" begin
        sp_K = 490.0
        sc_K = 510.0
        lp_K = 480.0
        lc_K = 520.0

        @testset "ShortPutDelta" begin
            f = ShortPutDelta(; rate=0.045, div_yield=0.015)
            val = f(ctx, sp_K, sc_K, lp_K, lc_K)
            if val !== nothing
                @test val isa Float64
                @test 0.0 < val < 1.0
            end
        end

        @testset "ShortCallDelta" begin
            f = ShortCallDelta(; rate=0.045, div_yield=0.015)
            val = f(ctx, sp_K, sc_K, lp_K, lc_K)
            if val !== nothing
                @test val isa Float64
                @test 0.0 < val < 1.0
            end
        end

        @testset "EntryCredit" begin
            val = EntryCredit()(ctx, sp_K, sc_K, lp_K, lc_K)
            @test val !== nothing
            @test val isa Float64
            # Credit = (sp_bid + sc_bid) - (lp_ask + lc_ask)
            # = (0.010 + 0.010) - (0.008 + 0.008) = 0.004
            @test val > 0.0
        end

        @testset "MaxLoss" begin
            val = MaxLoss()(ctx, sp_K, sc_K, lp_K, lc_K)
            @test val !== nothing
            @test val isa Float64
            @test val > 0.0
        end

        @testset "CreditToMaxLoss" begin
            val = CreditToMaxLoss()(ctx, sp_K, sc_K, lp_K, lc_K)
            @test val !== nothing
            @test val isa Float64
            @test val > 0.0
        end
    end

    # ================================================================
    # Model tests
    # ================================================================
    @testset "Model" begin
        @testset "create and forward pass" begin
            model = create_scoring_model(input_dim=22)
            # Random input
            x = randn(Float32, 22, 5)
            y = model(x)
            @test size(y) == (1, 5)
        end

        @testset "score_candidates" begin
            model = create_scoring_model(input_dim=22)
            sf = randn(Float32, 17)
            cf = randn(Float32, 5, 3)
            means = zeros(Float32, 22)
            stds = ones(Float32, 22)
            scores = score_candidates(model, sf, cf; feature_means=means, feature_stds=stds)
            @test length(scores) == 3
            @test eltype(scores) == Float32
        end
    end

    # ================================================================
    # Training data generation (small scale)
    # ================================================================
    @testset "generate_training_data" begin
        # Build a small data source with 2 surfaces
        ts1 = DateTime(2026, 3, 10, 14, 0, 0)
        ts2 = DateTime(2026, 3, 10, 15, 0, 0)
        settle_ts = DateTime(2026, 3, 11, 16, 0, 0)

        surfaces = Dict(
            ts1 => surface,
            ts2 => surface,
        )
        spots = Dict(
            settle_ts => 502.0,
        )
        source = DictDataSource(surfaces, spots)

        examples = generate_training_data(
            source, Day(1), [ts1, ts2];
            delta_grid=0.15:0.10:0.25,
            rate=0.045, div_yield=0.015,
            wing_objective=:roi
        )

        # Should generate some examples (may be 0 if delta resolution fails
        # on our sparse synthetic data, but the function should not error)
        @test examples isa Vector{TrainingExample}
        if !isempty(examples)
            @test length(examples[1].surface_features) > 0
            @test length(examples[1].candidate_features) > 0
            @test examples[1].label isa Float32
        end
    end

    # ================================================================
    # each_entry integration
    # ================================================================
    @testset "each_entry" begin
        ts1 = DateTime(2026, 3, 10, 14, 0, 0)
        settle_ts = DateTime(2026, 3, 11, 16, 0, 0)

        surfaces = Dict(ts1 => surface)
        spots = Dict(settle_ts => 502.0)
        source = DictDataSource(surfaces, spots)

        count = Ref(0)
        each_entry(source, Day(1), [ts1]) do ctx, settlement
            count[] += 1
            @test ctx isa StrikeSelectionContext
            @test ctx.surface === surface
        end
        @test count[] == 1
    end

    # ================================================================
    # Utility functions
    # ================================================================
    @testset "Utility functions" begin
        @test roi_utility(100.0, 500.0) ≈ 0.2
        @test pnl_utility(100.0, 500.0) ≈ 100.0
    end

    # ================================================================
    # Sizing policies
    # ================================================================
    @testset "Sizing policies" begin
        @testset "linear_sizing" begin
            policy = linear_sizing(; threshold=10.0, max_q=3.0, skip_negative=true)
            @test policy(0.0) == 0.0
            @test policy(-5.0) == 0.0
            @test policy(5.0) ≈ 0.5
            @test policy(10.0) ≈ 1.0
            @test policy(20.0) ≈ 2.0
            @test policy(100.0) == 3.0  # clamped to max_q

            # skip_negative=false: still trades on negative predictions
            policy_no_skip = linear_sizing(; threshold=10.0, max_q=3.0, skip_negative=false)
            @test policy_no_skip(-5.0) == 0.0  # clamped to 0
            @test policy_no_skip(5.0) ≈ 0.5
        end

        @testset "sigmoid_sizing" begin
            policy = sigmoid_sizing(; scale=1.0, max_q=2.0)
            @test policy(0.0) ≈ 1.0  # midpoint = max_q / 2
            @test policy(100.0) ≈ 2.0 atol=0.01  # large positive → max_q
            @test policy(-100.0) ≈ 0.0 atol=0.01  # large negative → 0
            @test 0.0 < policy(-1.0) < 1.0  # always positive
        end
    end

    # ================================================================
    # Sizing training data generation
    # ================================================================
    @testset "generate_sizing_training_data" begin
        ts1 = DateTime(2026, 3, 10, 14, 0, 0)
        settle_ts = DateTime(2026, 3, 11, 16, 0, 0)

        surfaces = Dict(ts1 => surface)
        spots = Dict(settle_ts => 502.0)
        source = DictDataSource(surfaces, spots)

        # Simple selector that returns fixed strikes from our test data
        fixed_selector = function(ctx)
            return (490.0, 510.0, 480.0, 520.0)
        end

        sf = Feature[ATMImpliedVol(; rate=0.045, div_yield=0.015)]

        examples = generate_sizing_training_data(
            source, Day(1), [ts1], fixed_selector;
            rate=0.045, div_yield=0.015,
            surface_features=sf
        )

        @test examples isa Vector{SizingTrainingExample}
        if !isempty(examples)
            @test length(examples) == 1
            @test length(examples[1].surface_features) == 1  # just ATMImpliedVol
            @test examples[1].pnl isa Float32
        end
    end

    # ================================================================
    # IronCondorStrategy with MLSizer
    # ================================================================
    @testset "IronCondorStrategy with MLSizer" begin
        ts1 = DateTime(2026, 3, 10, 14, 0, 0)
        settle_ts = DateTime(2026, 3, 11, 16, 0, 0)

        surfaces = Dict(ts1 => surface)
        spots = Dict(settle_ts => 502.0)
        source = DictDataSource(surfaces, spots)

        fixed_selector = function(ctx)
            return (490.0, 510.0, 480.0, 520.0)
        end

        sf = Feature[ATMImpliedVol(; rate=0.045, div_yield=0.015)]

        # Mock model: constant output of 5.0 (predicted PnL)
        # Use zero weights + bias=5.0 so output is always 5.0 regardless of input
        mock_model = Chain(Dense(zeros(Float32, 1, 1), Float32[5.0]))

        # linear_sizing with threshold=5 → quantity = 5/5 = 1.0
        strategy = IronCondorStrategy(
            [ts1], Day(1), fixed_selector;
            sizer=MLSizer(mock_model, Float32[0.0], Float32[1.0];
                surface_features=sf,
                policy=linear_sizing(; threshold=5.0, max_q=3.0)),
        )

        result = backtest_strategy(strategy, source)
        @test result isa BacktestResult

        # With a constant model predicting positive PnL, should produce trades
        if !isempty(result.positions)
            # Check quantity is 1.0 (5.0 / 5.0 threshold)
            @test result.positions[1].trade.quantity ≈ 1.0
        end

        # Test with negative prediction → skip entry
        neg_model = Chain(Dense(zeros(Float32, 1, 1), Float32[-5.0]))
        neg_strategy = IronCondorStrategy(
            [ts1], Day(1), fixed_selector;
            sizer=MLSizer(neg_model, Float32[0.0], Float32[1.0];
                surface_features=sf,
                policy=linear_sizing(; threshold=5.0, max_q=3.0, skip_negative=true)),
        )

        neg_result = backtest_strategy(neg_strategy, source)
        @test isempty(neg_result.positions)
    end
end
