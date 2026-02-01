@testset "Portfolio Management" begin
    data_path = joinpath(@__DIR__, "..", "data", "vols_20260117.parquet")

    if isfile(data_path)
        # Load records and build a surface from the first timestamp
        records = read_deribit_option_records(data_path; where="")
        timestamps = sort(unique(r.timestamp for r in records))
        ts = first(timestamps)
        ts_records = filter(r -> r.timestamp == ts, records)
        surface = build_surface(ts_records)

        # Pick a valid record that exists on the surface (has bid/ask and mark_iv)
        sample = first(filter(r ->
            !ismissing(r.mark_iv) &&
            !ismissing(r.bid_price) &&
            !ismissing(r.ask_price) &&
            r.bid_price > 0 &&
            r.expiry > r.timestamp,
            ts_records
        ))

        @testset "open_position long" begin
            trade = Trade(surface.underlying, sample.strike, sample.expiry, sample.option_type;
                          direction=1, quantity=1.0)
            pos = open_position(trade, surface)

            @test pos.trade == trade
            @test pos.entry_price == sample.ask_price   # long fills at ask
            @test pos.entry_spot == surface.spot
            @test pos.entry_timestamp == surface.timestamp
            @test !ismissing(pos.entry_bid)
            @test !ismissing(pos.entry_ask)
        end

        @testset "open_position short" begin
            trade = Trade(surface.underlying, sample.strike, sample.expiry, sample.option_type;
                          direction=-1, quantity=1.0)
            pos = open_position(trade, surface)

            @test pos.entry_price == sample.bid_price   # short fills at bid
        end

        @testset "entry_cost" begin
            # Long: positive cost (we pay ask * spot)
            trade_long = Trade(surface.underlying, sample.strike, sample.expiry, sample.option_type;
                               direction=1, quantity=1.0)
            pos_long = open_position(trade_long, surface)
            @test entry_cost(pos_long) > 0
            @test entry_cost(pos_long) ≈ sample.ask_price * surface.spot atol=1e-6

            # Short: negative cost (we receive bid * spot)
            trade_short = Trade(surface.underlying, sample.strike, sample.expiry, sample.option_type;
                                direction=-1, quantity=1.0)
            pos_short = open_position(trade_short, surface)
            @test entry_cost(pos_short) < 0
            @test entry_cost(pos_short) ≈ -sample.bid_price * surface.spot atol=1e-6
        end

        @testset "settle at expiry" begin
            trade = Trade(surface.underlying, sample.strike, sample.expiry, Call;
                          direction=1, quantity=1.0)

            # Find a call record on this surface for the trade
            call_rec = findfirst(r ->
                r.strike == sample.strike &&
                r.expiry == sample.expiry &&
                r.option_type == Call &&
                !ismissing(r.bid_price) && !ismissing(r.ask_price),
                ts_records
            )

            if call_rec !== nothing
                rec = ts_records[call_rec]
                trade = Trade(surface.underlying, rec.strike, rec.expiry, Call;
                              direction=1, quantity=1.0)
                pos = open_position(trade, surface)
                cost = entry_cost(pos)

                # Settle OTM (spot < strike): payoff = 0, PnL = -cost
                pnl_otm = settle(pos, rec.strike * 0.5)
                @test pnl_otm ≈ -cost atol=1e-6

                # Settle deep ITM: payoff = (spot - strike), PnL = payoff - cost
                deep_spot = rec.strike * 1.5
                pnl_itm = settle(pos, deep_spot)
                @test pnl_itm ≈ (deep_spot - rec.strike) - cost atol=1e-6

                # ITM PnL should be greater than OTM PnL
                @test pnl_itm > pnl_otm
            end
        end

        @testset "settle vector" begin
            # Long + short at same strike: payoffs cancel, PnL = spread cost
            trade_long  = Trade(surface.underlying, sample.strike, sample.expiry, sample.option_type;
                                direction=1, quantity=1.0)
            trade_short = Trade(surface.underlying, sample.strike, sample.expiry, sample.option_type;
                                direction=-1, quantity=1.0)
            positions = [open_position(trade_long, surface), open_position(trade_short, surface)]

            # At strike (ATM expiry), both payoffs are 0
            total_pnl = settle(positions, sample.strike)
            expected  = -(sample.ask_price - sample.bid_price) * surface.spot
            @test total_pnl ≈ expected atol=1e-6
        end
    else
        @info "Test data not found at $data_path, skipping Portfolio tests"
    end
end
