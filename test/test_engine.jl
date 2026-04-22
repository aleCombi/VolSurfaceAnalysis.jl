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

using DuckDB

@testset "Backtest Engine" begin
    # ============================================================
    # Fixture: single BTC call, entry → expiry next day
    #
    # Strike      100 000   (ITM at entry)
    # Entry spot  105 000
    # Settlement  110 000   (deeper ITM)
    # ask         0.05      (fraction of underlying)
    #
    # entry_cost = ask × spot          = 0.05 × 105 000 = 5 250
    # payoff     = max(110k − 100k, 0)                  = 10 000
    # PnL        = payoff − entry_cost                  =  4 750
    # ============================================================
    entry_ts        = DateTime(2026, 1, 25, 12, 0, 0)
    expiry_ts       = DateTime(2026, 1, 26,  8, 0, 0)   # Deribit 08:00 UTC
    strike          = 100_000.0
    entry_spot      = 105_000.0
    settlement_spot = 110_000.0
    bid             = 0.04
    ask             = 0.05

    # Two snapshots in one parquet (mirrors real Deribit data layout):
    #   entry_ts  → live quote          → used to build the entry surface
    #   expiry_ts → settlement snapshot → read_deribit_spot_prices picks up the spot
    tmpfile = replace(joinpath(tempdir(), "test_backtest_btc.parquet"), "\\" => "/")
    con = DuckDB.DBInterface.connect(DuckDB.DB, ":memory:")
    try
        DuckDB.DBInterface.execute(con, """
            COPY (
                SELECT 'BTC-26JAN26-100000-C'::VARCHAR AS instrument_name,
                       'BTC'::VARCHAR                  AS underlying,
                       TIMESTAMP '2026-01-26 08:00:00' AS expiry,
                       $strike                         AS strike,
                       'C'::VARCHAR                   AS option_type,
                       $bid                           AS bid_price,
                       $ask                           AS ask_price,
                       0.045                          AS mark_price,
                       80.0                           AS mark_iv,
                       100.0                          AS open_interest,
                       10.0                           AS volume,
                       $entry_spot                    AS underlying_price,
                       TIMESTAMP '2026-01-25 12:00:00' AS ts
                UNION ALL
                SELECT 'BTC-26JAN26-100000-C'::VARCHAR,
                       'BTC'::VARCHAR,
                       TIMESTAMP '2026-01-26 08:00:00',
                       $strike,
                       'C'::VARCHAR,
                       0.10,
                       0.11,
                       0.105,
                       50.0,
                       90.0,
                       5.0,
                       $settlement_spot,
                       TIMESTAMP '2026-01-26 08:00:00'
            ) TO '$tmpfile' (FORMAT PARQUET)
        """)
    finally
        DuckDB.DBInterface.close!(con)
    end

    # ============================================================
    # 1. Load all spots (entry + settlement from same file)
    # ============================================================
    all_spots = read_deribit_spot_prices(tmpfile; underlying="BTC")

    # ============================================================
    # 2. Build surfaces via build_surfaces_for_timestamps
    #    (same pattern as backtest_polygon_iron_condor.jl)
    # ============================================================
    surfaces = build_surfaces_for_timestamps(
        [entry_ts];
        path_for_timestamp = _ -> tmpfile,
        read_records       = (path; where="") -> read_deribit_option_records(path; where=where),
        ts_col             = :ts
    )

    # ============================================================
    # 3. Strategy + schedule (schedule from surfaces, as scripts do)
    # ============================================================
    struct BuyTheOnlyCall <: ScheduledStrategy end
    schedule = sort(collect(keys(surfaces)))
    VolSurfaceAnalysis.entry_schedule(::BuyTheOnlyCall) = schedule
    function VolSurfaceAnalysis.entry_positions(::BuyTheOnlyCall, surf::VolatilitySurface)
        trade = Trade(Underlying("BTC"), strike, expiry_ts, Call; direction=1, quantity=1.0)
        return [open_position(trade, surf)]
    end

    # ============================================================
    # 4. Collect expiry timestamps → extract settlement spots
    #    (same pattern as scripts: dry-run entry_positions first)
    # ============================================================
    expiry_timestamps = unique([
        pos.trade.expiry
        for ts in schedule
        for pos in entry_positions(BuyTheOnlyCall(), surfaces[ts])
    ])
    settlement_spots = Dict(ts => all_spots[ts] for ts in expiry_timestamps)

    # ============================================================
    # 5. Run backtest
    # ============================================================
    result = backtest_strategy(BuyTheOnlyCall(), surfaces, settlement_spots)
    positions = result.positions
    pnl = result.pnl

    @testset "Position entry" begin
        @test length(positions) == 1

        pos = positions[1]
        @test pos.trade.strike      == strike
        @test pos.trade.option_type == Call
        @test pos.trade.direction   == 1
        @test pos.entry_price       == ask          # long → filled at ask
        @test pos.entry_spot        == entry_spot
        @test pos.entry_timestamp   == entry_ts
    end

    @testset "Settlement PnL" begin
        @test length(pnl) == 1
        @test !ismissing(pnl[1])

        expected_payoff  = max(settlement_spot - strike, 0.0)   # 10 000
        expected_cost    = ask * entry_spot                      #  5 250
        expected_pnl     = expected_payoff - expected_cost       #  4 750
        @test pnl[1] ≈ expected_pnl
    end

    # ============================================================
    # 6. DictDataSource — same result via protocol
    # ============================================================
    @testset "DictDataSource" begin
        source = DictDataSource(surfaces, settlement_spots)

        @test available_timestamps(source) == schedule
        @test get_surface(source, entry_ts) isa VolatilitySurface
        @test get_surface(source, DateTime(2099, 1, 1)) === nothing
        @test get_settlement_spot(source, expiry_ts) == settlement_spot
        @test ismissing(get_settlement_spot(source, DateTime(2099, 1, 1)))

        result2 = backtest_strategy(BuyTheOnlyCall(), source)

        @test length(result2.positions) == length(positions)
        @test length(result2.pnl) == length(pnl)
        @test result2.positions[1].trade.strike == positions[1].trade.strike
        @test result2.positions[1].entry_price  == positions[1].entry_price
        @test result2.pnl[1] ≈ pnl[1]
    end

    # ============================================================
    # 7. get_spot / get_spots on DictDataSource
    # ============================================================
    @testset "DictDataSource get_spot/get_spots" begin
        # DictDataSource.get_spot delegates to the spots dict (same as get_settlement_spot)
        source = DictDataSource(surfaces, settlement_spots)
        @test get_spot(source, expiry_ts) == settlement_spot
        @test ismissing(get_spot(source, DateTime(2099, 1, 1)))

        # get_spots returns all spots in range
        spots_range = get_spots(source, expiry_ts - Minute(1), expiry_ts + Minute(1))
        @test haskey(spots_range, expiry_ts)
        @test spots_range[expiry_ts] == settlement_spot

        # Empty range returns empty dict
        empty_range = get_spots(source, DateTime(2099, 1, 1), DateTime(2099, 1, 2))
        @test isempty(empty_range)
    end

    # ============================================================
    # 8. HistoricalView — time-filtered wrapper
    # ============================================================
    @testset "HistoricalView" begin
        source = DictDataSource(surfaces, settlement_spots)

        # View with cutoff at entry time: can see entry surface, not future
        view = HistoricalView(source, entry_ts)
        @test entry_ts in available_timestamps(view)
        @test get_surface(view, entry_ts) isa VolatilitySurface

        # Cannot see surface at a future time (if one existed)
        future_ts = entry_ts + Day(1)
        @test get_surface(view, future_ts) === nothing

        # Settlement spot at expiry is in the future relative to entry
        @test ismissing(get_settlement_spot(view, expiry_ts))
        @test ismissing(get_spot(view, expiry_ts))

        # get_spots clamps range to cutoff
        clamped = get_spots(view, entry_ts - Day(1), expiry_ts)
        for (ts, _) in clamped
            @test ts <= entry_ts
        end

        # View with cutoff after expiry: can see everything
        full_view = HistoricalView(source, expiry_ts + Day(1))
        @test get_spot(full_view, expiry_ts) == settlement_spot
    end

    # ============================================================
    # 9. 3-arg entry_positions backward compat (2-arg strategy works)
    # ============================================================
    @testset "entry_positions backward compat" begin
        source = DictDataSource(surfaces, settlement_spots)
        history = HistoricalView(source, entry_ts)

        # BuyTheOnlyCall only implements 2-arg; 3-arg should delegate to it
        pos3 = entry_positions(BuyTheOnlyCall(), surfaces[entry_ts], history)
        @test length(pos3) == 1
        @test pos3[1].trade.strike == strike
    end

    # ============================================================
    # 10. clear_cache!: no-op default, ParquetDataSource empties, idempotent
    # ============================================================
    @testset "clear_cache! default no-op" begin
        source = DictDataSource(surfaces, settlement_spots)
        @test VolSurfaceAnalysis.clear_cache!(source) === nothing
        # Still functional after "clearing"
        @test get_surface(source, entry_ts) isa VolatilitySurface
    end

    @testset "clear_cache! on ParquetDataSource empties caches" begin
        # Seed a ParquetDataSource directly — no parquet I/O needed.
        p = ParquetDataSource(
            [entry_ts];
            path_for_timestamp = _ -> "/nonexistent",
            read_records       = (path; where="") -> OptionRecord[],
            spot_root          = "/nonexistent",
            spot_symbol        = "BTC",
        )
        # Manually populate both caches
        p.surface_cache[entry_ts] = surfaces[entry_ts]
        p.spot_date_cache[Date(entry_ts)] = Dict(entry_ts => 1.0)
        @test !isempty(p.surface_cache)
        @test !isempty(p.spot_date_cache)

        VolSurfaceAnalysis.clear_cache!(p)
        @test isempty(p.surface_cache)
        @test isempty(p.spot_date_cache)

        # Idempotent
        @test VolSurfaceAnalysis.clear_cache!(p) === nothing
        @test isempty(p.surface_cache)
    end

    @testset "each_entry clear_cache kwarg drives clear_cache!" begin
        # Custom data source that counts clear_cache! invocations.
        mutable struct CountingDataSource <: BacktestDataSource
            inner::DictDataSource
            clears::Int
        end
        VolSurfaceAnalysis.available_timestamps(s::CountingDataSource) =
            available_timestamps(s.inner)
        VolSurfaceAnalysis.get_surface(s::CountingDataSource, ts::DateTime) =
            get_surface(s.inner, ts)
        VolSurfaceAnalysis.get_settlement_spot(s::CountingDataSource, ts::DateTime) =
            get_settlement_spot(s.inner, ts)
        VolSurfaceAnalysis.get_spot(s::CountingDataSource, ts::DateTime) =
            get_spot(s.inner, ts)
        VolSurfaceAnalysis.clear_cache!(s::CountingDataSource) = (s.clears += 1; nothing)

        inner = DictDataSource(surfaces, settlement_spots)

        # clear_cache=false (default): no clears
        cs1 = CountingDataSource(inner, 0)
        n_cb1 = 0
        each_entry(cs1, Day(1), [entry_ts]) do _, _
            n_cb1 += 1
        end
        @test n_cb1 == 1
        @test cs1.clears == 0

        # clear_cache=true: one clear per callback
        cs2 = CountingDataSource(inner, 0)
        n_cb2 = 0
        each_entry(cs2, Day(1), [entry_ts]; clear_cache=true) do _, _
            n_cb2 += 1
        end
        @test n_cb2 == 1
        @test cs2.clears == 1

        # Callback throws → finally still clears
        cs3 = CountingDataSource(inner, 0)
        @test_throws ErrorException each_entry(cs3, Day(1), [entry_ts]; clear_cache=true) do _, _
            error("boom")
        end
        @test cs3.clears == 1
    end

    # ============================================================
    # 11. delta_context + delta_strike public helpers
    # ============================================================
    @testset "delta_context + delta_strike" begin
        source = DictDataSource(surfaces, settlement_spots)
        history = HistoricalView(source, entry_ts)
        ctx = VolSurfaceAnalysis.StrikeSelectionContext(
            surfaces[entry_ts], expiry_ts, history,
        )

        # Happy path: ctx has only a call record in the fixture, so put-side
        # records are empty — delta_context returns nothing.
        @test delta_context(ctx) === nothing

        # Construct a two-record surface (one put + one call) inline to exercise
        # the happy path.
        instr_put  = "BTC-26JAN26-100000-P"
        instr_call = "BTC-26JAN26-100000-C"
        rec_put  = OptionRecord(instr_put,  Underlying("BTC"), expiry_ts, strike, Put,
                                0.05, 0.06, 0.055, 80.0, 10.0, 100.0, entry_spot, entry_ts)
        rec_call = OptionRecord(instr_call, Underlying("BTC"), expiry_ts, strike, Call,
                                bid,  ask,  0.045, 80.0, 10.0, 100.0, entry_spot, entry_ts)
        surf = build_surface([rec_put, rec_call])
        ctx_pc = VolSurfaceAnalysis.StrikeSelectionContext(surf, expiry_ts, history)

        dctx = delta_context(ctx_pc; rate=0.0, div_yield=0.0)
        @test dctx !== nothing
        @test length(dctx.put_recs)  == 1
        @test length(dctx.call_recs) == 1
        @test dctx.spot == entry_spot
        @test dctx.tau > 0.0
        # F = spot * exp(0) == spot when rate=div=0
        @test dctx.F == entry_spot

        # delta_strike: with only one strike, that strike is chosen
        @test delta_strike(dctx, -0.30, Put)  == strike
        @test delta_strike(dctx,  0.30, Call) == strike
    end

    # ============================================================
    # 12. nearest_otm_strike + extract_price
    # ============================================================
    @testset "nearest_otm_strike + extract_price" begin
        history = HistoricalView(DictDataSource(surfaces, settlement_spots), entry_ts)
        # Build a surface with OTM strikes on both sides of spot.
        K_short_put  = 100_000.0
        K_short_call = 110_000.0
        strikes_put  = [90_000.0, 95_000.0, 98_000.0, K_short_put]
        strikes_call = [K_short_call, 112_000.0, 115_000.0, 120_000.0]
        recs_multi = OptionRecord[]
        for K in strikes_put
            push!(recs_multi, OptionRecord("BTC-$K-P", Underlying("BTC"), expiry_ts, K, Put,
                                           0.05, 0.06, 0.055, 80.0, 10.0, 100.0, entry_spot, entry_ts))
        end
        for K in strikes_call
            push!(recs_multi, OptionRecord("BTC-$K-C", Underlying("BTC"), expiry_ts, K, Call,
                                           0.05, 0.06, 0.055, 80.0, 10.0, 100.0, entry_spot, entry_ts))
        end
        surf_multi = build_surface(recs_multi)
        ctx_multi = VolSurfaceAnalysis.StrikeSelectionContext(surf_multi, expiry_ts, history)
        dctx_multi = delta_context(ctx_multi)
        @test dctx_multi !== nothing

        # Put wing: below short put. width=2000 → target 98_000 → exact match
        @test nearest_otm_strike(dctx_multi, K_short_put, 2_000.0, Put)  == 98_000.0
        # Put wing: width=5000 → target 95_000 → exact match
        @test nearest_otm_strike(dctx_multi, K_short_put, 5_000.0, Put)  == 95_000.0
        # Put wing: width=6000 → target 94_000 → snap to nearest OTM (95_000)
        @test nearest_otm_strike(dctx_multi, K_short_put, 6_000.0, Put)  == 95_000.0

        # Call wing: above short call. width=2000 → target 112_000 → exact match
        @test nearest_otm_strike(dctx_multi, K_short_call, 2_000.0, Call) == 112_000.0
        # Call wing: width=11000 → target 121_000 → snap to nearest OTM (120_000)
        @test nearest_otm_strike(dctx_multi, K_short_call, 11_000.0, Call) == 120_000.0

        # Put wing with no OTM: reference below all strikes → nothing
        @test nearest_otm_strike(dctx_multi, 80_000.0, 5_000.0, Put) === nothing
        # Call wing with no OTM: reference above all strikes → nothing
        @test nearest_otm_strike(dctx_multi, 130_000.0, 5_000.0, Call) === nothing

        # extract_price: bid / ask / mark fallback
        rec_full = OptionRecord("BTC-F", Underlying("BTC"), expiry_ts, 100.0, Put,
                                0.03, 0.04, 0.035, 80.0, 10.0, 100.0, 100.0, entry_ts)
        @test extract_price(rec_full, :bid) == 0.03
        @test extract_price(rec_full, :ask) == 0.04

        rec_no_bid = OptionRecord("BTC-NB", Underlying("BTC"), expiry_ts, 100.0, Put,
                                  missing, 0.04, 0.035, 80.0, 10.0, 100.0, 100.0, entry_ts)
        @test extract_price(rec_no_bid, :bid) == 0.035   # fallback to mark
        @test extract_price(rec_no_bid, :ask) == 0.04    # primary

        rec_nothing = OptionRecord("BTC-NN", Underlying("BTC"), expiry_ts, 100.0, Put,
                                   missing, missing, missing, 80.0, 10.0, 100.0, 100.0, entry_ts)
        @test extract_price(rec_nothing, :bid) === nothing
        @test extract_price(rec_nothing, :ask) === nothing
    end

    # ============================================================
    # 13. open_condor_positions
    # ============================================================
    @testset "open_condor_positions" begin
        history = HistoricalView(DictDataSource(surfaces, settlement_spots), entry_ts)
        # Build a surface with 2 puts + 2 calls so a condor has 4 distinct legs.
        K_sp, K_lp = 100_000.0, 95_000.0   # short put, long put wing
        K_sc, K_lc = 110_000.0, 115_000.0  # short call, long call wing
        recs = OptionRecord[
            OptionRecord("BTC-SP", Underlying("BTC"), expiry_ts, K_sp, Put,
                         0.05, 0.06, 0.055, 80.0, 10.0, 100.0, entry_spot, entry_ts),
            OptionRecord("BTC-LP", Underlying("BTC"), expiry_ts, K_lp, Put,
                         0.02, 0.03, 0.025, 80.0, 10.0, 100.0, entry_spot, entry_ts),
            OptionRecord("BTC-SC", Underlying("BTC"), expiry_ts, K_sc, Call,
                         0.05, 0.06, 0.055, 80.0, 10.0, 100.0, entry_spot, entry_ts),
            OptionRecord("BTC-LC", Underlying("BTC"), expiry_ts, K_lc, Call,
                         0.02, 0.03, 0.025, 80.0, 10.0, 100.0, entry_spot, entry_ts),
        ]
        surf = build_surface(recs)
        ctx_condor = VolSurfaceAnalysis.StrikeSelectionContext(surf, expiry_ts, history)

        positions = open_condor_positions(ctx_condor, K_sp, K_sc, K_lp, K_lc)
        @test length(positions) == 4
        # Order: short put, short call, long put, long call
        @test positions[1].trade.option_type == Put  && positions[1].trade.direction == -1 && positions[1].trade.strike == K_sp
        @test positions[2].trade.option_type == Call && positions[2].trade.direction == -1 && positions[2].trade.strike == K_sc
        @test positions[3].trade.option_type == Put  && positions[3].trade.direction ==  1 && positions[3].trade.strike == K_lp
        @test positions[4].trade.option_type == Call && positions[4].trade.direction ==  1 && positions[4].trade.strike == K_lc
        # Short legs fill at bid, long legs at ask (trade.direction convention)
        @test positions[1].entry_price == 0.05
        @test positions[3].entry_price == 0.03

        # Missing record for one leg → empty vector
        empty = open_condor_positions(ctx_condor, K_sp, K_sc, 80_000.0, K_lc)
        @test isempty(empty)

        # Quantity propagates
        positions_q = open_condor_positions(ctx_condor, K_sp, K_sc, K_lp, K_lc; quantity=2.5)
        @test all(p.trade.quantity == 2.5 for p in positions_q)
    end

    # ============================================================
    # 14. open_strangle_positions + find_record_at_strike
    # ============================================================
    @testset "open_strangle_positions + find_record_at_strike" begin
        history = HistoricalView(DictDataSource(surfaces, settlement_spots), entry_ts)
        K_sp, K_sc = 100_000.0, 110_000.0
        recs = OptionRecord[
            OptionRecord("BTC-SP", Underlying("BTC"), expiry_ts, K_sp, Put,
                         0.05, 0.06, 0.055, 80.0, 10.0, 100.0, entry_spot, entry_ts),
            OptionRecord("BTC-SC", Underlying("BTC"), expiry_ts, K_sc, Call,
                         0.04, 0.05, 0.045, 80.0, 10.0, 100.0, entry_spot, entry_ts),
        ]
        surf = build_surface(recs)
        ctx_strangle = VolSurfaceAnalysis.StrikeSelectionContext(surf, expiry_ts, history)

        # Default direction = -1 (short). Short fills at bid.
        short_pos = open_strangle_positions(ctx_strangle, K_sp, K_sc)
        @test length(short_pos) == 2
        @test short_pos[1].trade.option_type == Put  && short_pos[1].trade.direction == -1
        @test short_pos[2].trade.option_type == Call && short_pos[2].trade.direction == -1
        @test short_pos[1].entry_price == 0.05   # bid
        @test short_pos[2].entry_price == 0.04   # bid

        # direction = +1 (long) fills at ask.
        long_pos = open_strangle_positions(ctx_strangle, K_sp, K_sc; direction=+1)
        @test length(long_pos) == 2
        @test all(p.trade.direction == +1 for p in long_pos)
        @test long_pos[1].entry_price == 0.06   # ask
        @test long_pos[2].entry_price == 0.05   # ask

        # Missing record → empty
        @test isempty(open_strangle_positions(ctx_strangle, 80_000.0, K_sc))

        # Quantity propagates
        qty_pos = open_strangle_positions(ctx_strangle, K_sp, K_sc; quantity=3.0)
        @test all(p.trade.quantity == 3.0 for p in qty_pos)

        # find_record_at_strike
        @test find_record_at_strike(recs, K_sp) === recs[1]
        @test find_record_at_strike(recs, K_sc) === recs[2]
        @test find_record_at_strike(recs, 99_999.0) === nothing
        @test find_record_at_strike(OptionRecord[], K_sp) === nothing
    end

    try rm(tmpfile; force=true) catch end
end
