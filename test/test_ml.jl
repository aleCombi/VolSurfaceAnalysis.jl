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
end
