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
    positions, pnl = backtest_strategy(BuyTheOnlyCall(), surfaces, settlement_spots)

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

    try rm(tmpfile; force=true) catch end
end
