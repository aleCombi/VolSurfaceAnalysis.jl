using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, DataFrames

# =============================================================================
# Parameters
# =============================================================================

SYMBOL          = "SPY"
SPOT_SYMBOL     = "SPY"
SPOT_MULTIPLIER = 1.0
ENTRY_TIME      = Time(10, 0)          # 10:00 ET
EXPIRY_INTERVAL = Day(1)               # 1DTE
SPREAD_LAMBDA   = 0.0                  # conservative (widest synthetic spread)

# In-sample / out-of-sample split
IS_START  = Date(2024, 2, 1)
IS_END    = Date(2025, 6, 30)
OOS_START = Date(2025, 7, 1)
OOS_END   = Date(2026, 3, 13)

# Selector parameters
PUT_DELTA      = 0.16
CALL_DELTA     = 0.16
RATE           = 0.045
DIV_YIELD      = 0.013
MAX_LOSS       = 5.0       # USD per condor
MAX_SPREAD_REL = 0.50      # reject if (ask-bid)/mid > 50%
MIN_DELTA_GAP  = 0.08

# =============================================================================
# Output directory
# =============================================================================

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "backtest_spy_iron_condor_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")

# =============================================================================
# Run one period
# =============================================================================

function run_period(label, start_date, end_date, run_dir)
    println("\n", "=" ^ 60)
    println("  $label: $start_date to $end_date")
    println("=" ^ 60)

    surfaces, _, settlement_spots = load_surfaces_and_spots(
        start_date, end_date;
        symbol=SYMBOL,
        spot_symbol=SPOT_SYMBOL,
        spot_multiplier=SPOT_MULTIPLIER,
        entry_times=ENTRY_TIME,
        spread_lambda=SPREAD_LAMBDA,
        expiry_interval=EXPIRY_INTERVAL
    )

    schedule = sort(collect(keys(surfaces)))
    println("  Schedule: $(length(schedule)) entries")

    selector = constrained_delta_selector(
        PUT_DELTA, CALL_DELTA;
        rate=RATE, div_yield=DIV_YIELD,
        max_loss=MAX_LOSS,
        max_spread_rel=MAX_SPREAD_REL,
        min_delta_gap=MIN_DELTA_GAP
    )

    strategy = IronCondorStrategy(schedule, EXPIRY_INTERVAL, selector)
    result = backtest_strategy(strategy, surfaces, settlement_spots)

    margin = condor_max_loss_by_key(result.positions)
    metrics = performance_metrics(result.positions, result.pnl; margin_by_key=margin)

    # Realized PnLs per condor
    pnl_by_key, _ = aggregate_pnl(result.positions, result.pnl)
    realized = sort(collect(pnl_by_key); by=first)
    realized_pnls = [v for (_, v) in realized]
    realized_dates = [Date(k[1]) for (k, _) in realized]

    # Report
    lines = format_backtest_report(
        metrics;
        title="$SYMBOL Iron Condor — $label",
        params=[
            "Period"       => "$start_date to $end_date",
            "Underlying"   => SYMBOL,
            "Entry time"   => "10:00 ET",
            "Expiry"       => "1DTE",
            "Deltas"       => "put=$PUT_DELTA, call=$CALL_DELTA",
            "Max loss"     => "\$$MAX_LOSS",
            "Max spread"   => "$(Int(MAX_SPREAD_REL * 100))%",
            "Spread lambda"=> "$SPREAD_LAMBDA"
        ],
        realized_pnls=realized_pnls,
        n_scheduled=length(schedule),
        n_attempted=length(realized_pnls),
        n_positions=length(result.positions),
        n_missing=metrics.missing,
        margin_description="per-condor max loss"
    )

    # Print and save report
    for l in lines; println(l); end
    prefix = joinpath(run_dir, lowercase(replace(label, " " => "_")))
    open(prefix * "_report.txt", "w") do io
        for l in lines; println(io, l); end
    end

    # Save CSVs
    df_metrics = metrics_to_dataframe(metrics)
    df_pnl = pnl_results_dataframe(result.positions, result.pnl)
    df_trades = condor_trade_table(result.positions, result.pnl)

    open(prefix * "_metrics.csv", "w") do io
        for i in 1:nrow(df_metrics)
            println(io, "$(df_metrics.Metric[i]),$(df_metrics.Value[i])")
        end
    end
    open(prefix * "_pnl.csv", "w") do io
        println(io, join(names(df_pnl), ","))
        for row in eachrow(df_pnl)
            println(io, join([row[c] for c in names(df_pnl)], ","))
        end
    end
    open(prefix * "_trades.csv", "w") do io
        println(io, join(names(df_trades), ","))
        for row in eachrow(df_trades)
            println(io, join([row[c] for c in names(df_trades)], ","))
        end
    end

    # Save plots
    plots_dir = joinpath(run_dir, "plots")
    mkpath(plots_dir)
    save_pnl_and_equity_curve(realized_dates, realized_pnls,
        joinpath(plots_dir, "$(lowercase(replace(label, " " => "_")))_equity.png");
        title_prefix="$SYMBOL $label")

    println("  Saved to $prefix*")
    return metrics
end

# =============================================================================
# Main
# =============================================================================

is_metrics  = run_period("In-Sample",      IS_START,  IS_END,  run_dir)
oos_metrics = run_period("Out-of-Sample",  OOS_START, OOS_END, run_dir)

println("\n", "=" ^ 60)
println("  COMPARISON")
println("=" ^ 60)
println("  In-sample  ROI: $(fmt_metric(is_metrics.total_roi; pct=true))  Sharpe: $(fmt_ratio(is_metrics.sharpe))  Win: $(fmt_pct(is_metrics.win_rate))")
println("  OOS        ROI: $(fmt_metric(oos_metrics.total_roi; pct=true))  Sharpe: $(fmt_ratio(oos_metrics.sharpe))  Win: $(fmt_pct(oos_metrics.win_rate))")
println("=" ^ 60)
