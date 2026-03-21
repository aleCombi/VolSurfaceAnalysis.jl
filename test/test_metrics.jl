using DataFrames

@testset "Metrics Helpers" begin
    # Build synthetic positions and pnls for testing
    underlying = Underlying("TEST")
    ts1 = DateTime(2025, 3, 1, 14, 0)
    ts2 = DateTime(2025, 3, 2, 14, 0)
    exp1 = DateTime(2025, 3, 2, 20, 0)
    exp2 = DateTime(2025, 3, 3, 20, 0)
    spot = 100.0

    function make_position(entry_ts, expiry, strike, opt_type, direction)
        trade = Trade(underlying, strike, expiry, opt_type; direction=direction, quantity=1.0)
        Position(trade, 0.01, spot, 0.009, 0.011, entry_ts)
    end

    # Create iron condor legs for day 1
    pos1_sp = make_position(ts1, exp1, 95.0, Put, -1)   # short put
    pos1_lp = make_position(ts1, exp1, 90.0, Put, 1)    # long put
    pos1_sc = make_position(ts1, exp1, 105.0, Call, -1)  # short call
    pos1_lc = make_position(ts1, exp1, 110.0, Call, 1)   # long call

    # Create iron condor legs for day 2
    pos2_sp = make_position(ts2, exp2, 96.0, Put, -1)
    pos2_lp = make_position(ts2, exp2, 91.0, Put, 1)
    pos2_sc = make_position(ts2, exp2, 104.0, Call, -1)
    pos2_lc = make_position(ts2, exp2, 109.0, Call, 1)

    positions = [pos1_sp, pos1_lp, pos1_sc, pos1_lc, pos2_sp, pos2_lp, pos2_sc, pos2_lc]
    pnls = Union{Missing,Float64}[1.5, -0.5, 2.0, -1.0, -3.0, 1.0, 0.5, -0.5]

    @testset "Formatting helpers" begin
        @test fmt_pnl(1234.5) == "\$1234"
        @test fmt_pnl(missing) == "n/a"

        @test fmt_ratio(1.234) == "1.23"
        @test fmt_ratio(missing) == "n/a"

        @test fmt_pct(0.654) == "65.4%"
        @test fmt_pct(missing) == "n/a"

        @test fmt_currency(1.234) == "\$1.23"
        @test fmt_currency(missing) == "n/a"

        @test fmt_metric(0.1234; pct=true) == "12.34%"
        @test fmt_metric(1.234; pct=false) == "1.23"
        @test fmt_metric(missing) == "n/a"
    end

    @testset "metrics_to_dataframe" begin
        metrics = performance_metrics(positions, pnls; margin_per_trade=500.0)
        df = metrics_to_dataframe(metrics)

        @test df isa DataFrame
        @test ncol(df) == 2
        @test nrow(df) == 17
        @test "Metric" in names(df)
        @test "Value" in names(df)
        @test df.Metric[1] == "count"
        @test df.Value[1] == metrics.count
        @test df.Metric[3] == "total_pnl"
        @test df.Value[3] == metrics.total_pnl
    end

    @testset "pnl_results_dataframe" begin
        df = pnl_results_dataframe(positions, pnls)

        @test df isa DataFrame
        @test ncol(df) == 3
        @test nrow(df) == 2  # Two trade groups (day 1 + day 2)
        @test "EntryDate" in names(df)
        @test "PnL" in names(df)
        @test "Result" in names(df)

        # Sorted by date
        @test df.EntryDate[1] <= df.EntryDate[2]

        # Day 1 net PnL: 1.5 + -0.5 + 2.0 + -1.0 = 2.0
        @test df.PnL[1] ≈ 2.0
        @test df.Result[1] == "Win"

        # Day 2 net PnL: -3.0 + 1.0 + 0.5 + -0.5 = -2.0
        @test df.PnL[2] ≈ -2.0
        @test df.Result[2] == "Loss"
    end

    @testset "pnl_results_dataframe with missing" begin
        pnls_with_missing = Union{Missing,Float64}[1.0, -0.5, 2.0, -1.0, missing, missing, missing, missing]
        df = pnl_results_dataframe(positions, pnls_with_missing)

        # Only day 1 has non-missing pnls
        @test nrow(df) == 1
        @test df.PnL[1] ≈ 1.5  # 1.0 + -0.5 + 2.0 + -1.0
    end

    @testset "format_backtest_report" begin
        metrics = performance_metrics(positions, pnls; margin_per_trade=500.0)
        lines = format_backtest_report(
            metrics;
            title="TEST REPORT",
            subtitle="Some warning",
            params=["Underlying" => "TEST", "Entry time" => "10:00 ET"],
            realized_pnls=[2.0, -2.0],
            n_scheduled=5,
            n_attempted=2,
            n_positions=8,
            n_missing=0,
            margin_description="fixed \$500"
        )

        @test lines isa Vector{String}
        @test any(l -> contains(l, "TEST REPORT"), lines)
        @test any(l -> contains(l, "Some warning"), lines)
        @test any(l -> contains(l, "Underlying: TEST"), lines)
        @test any(l -> contains(l, "Entry time: 10:00 ET"), lines)
        @test any(l -> contains(l, "Scheduled entries: 5"), lines)
        @test any(l -> contains(l, "Total positions: 8"), lines)
        @test any(l -> contains(l, "fixed \$500"), lines)
        @test any(l -> contains(l, "Total ROI:"), lines)
        @test any(l -> contains(l, "Sharpe:"), lines)
    end

    @testset "format_backtest_report minimal" begin
        metrics = performance_metrics(Position[], Union{Missing,Float64}[])
        lines = format_backtest_report(metrics; title="EMPTY")

        @test lines isa Vector{String}
        @test any(l -> contains(l, "EMPTY"), lines)
        @test any(l -> contains(l, "Count: 0"), lines)
    end

    @testset "performance_metrics(::BacktestResult)" begin
        result = BacktestResult(positions, pnls)
        m = performance_metrics(result)

        @test m isa PerformanceMetrics
        @test m.count == 2
        @test m.total_pnl ≈ 0.0  # 2.0 + (-2.0)
        @test !ismissing(m.total_roi)
        @test !ismissing(m.sharpe)

        # Empty result returns nothing
        empty_result = BacktestResult(Position[], Union{Missing,Float64}[])
        @test performance_metrics(empty_result) === nothing
    end
end
