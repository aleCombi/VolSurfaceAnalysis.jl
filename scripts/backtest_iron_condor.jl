using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, DataFrames

# =============================================================================
# Parameters
# =============================================================================

ENTRY_TIME      = Time(10, 0)          # 10:00 ET
EXPIRY_INTERVAL = Day(1)               # 1DTE
SPREAD_LAMBDA   = 0.0                  # conservative (widest synthetic spread)
START_DATE      = Date(2024, 2, 1)
END_DATE        = Date(2025, 12, 31)

# Selector parameters
PUT_DELTA      = 0.16
CALL_DELTA     = 0.16
RATE           = 0.045
DIV_YIELD      = 0.013
MAX_LOSS       = 5.0       # USD per condor
MAX_SPREAD_REL = 0.50      # reject if (ask-bid)/mid > 50%
MIN_DELTA_GAP  = 0.08

# Symbols: (options_symbol, spot_symbol, spot_multiplier)
SYMBOLS = [
    ("SPY",  "SPY",  1.0),
    ("SPXW", "SPY", 10.0),
    ("QQQ",  "QQQ",  1.0),
    ("IWM",  "IWM",  1.0),
]

# =============================================================================
# Output directory
# =============================================================================

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "backtest_iron_condor_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")

# =============================================================================
# Run one symbol
# =============================================================================

function run_symbol(symbol, spot_symbol, spot_multiplier, run_dir)
    println("\n", "=" ^ 60)
    println("  $symbol (spot via $spot_symbol × $spot_multiplier)")
    println("=" ^ 60)

    # Scale max loss by spot multiplier (SPXW is ~10x SPY price level)
    scaled_max_loss = MAX_LOSS * spot_multiplier

    surfaces, _, settlement_spots = load_surfaces_and_spots(
        START_DATE, END_DATE;
        symbol=symbol,
        spot_symbol=spot_symbol,
        spot_multiplier=spot_multiplier,
        entry_times=ENTRY_TIME,
        spread_lambda=SPREAD_LAMBDA,
        expiry_interval=EXPIRY_INTERVAL
    )

    schedule = sort(collect(keys(surfaces)))
    println("  Schedule: $(length(schedule)) entries")

    selector = constrained_delta_selector(
        PUT_DELTA, CALL_DELTA;
        rate=RATE, div_yield=DIV_YIELD,
        max_loss=scaled_max_loss,
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
        title="$symbol Iron Condor",
        params=[
            "Period"        => "$START_DATE to $END_DATE",
            "Spot proxy"    => "$spot_symbol × $spot_multiplier",
            "Entry time"    => "10:00 ET",
            "Expiry"        => "1DTE",
            "Deltas"        => "put=$PUT_DELTA, call=$CALL_DELTA",
            "Max loss"      => "\$$scaled_max_loss",
            "Max spread"    => "$(Int(MAX_SPREAD_REL * 100))%",
            "Spread lambda" => "$SPREAD_LAMBDA"
        ],
        realized_pnls=realized_pnls,
        n_scheduled=length(schedule),
        n_attempted=length(realized_pnls),
        n_positions=length(result.positions),
        n_missing=metrics.missing,
        margin_description="per-condor max loss"
    )

    for l in lines; println(l); end

    # Save outputs
    prefix = joinpath(run_dir, lowercase(symbol))
    open(prefix * "_report.txt", "w") do io
        for l in lines; println(io, l); end
    end

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

    plots_dir = joinpath(run_dir, "plots")
    mkpath(plots_dir)
    save_pnl_and_equity_curve(realized_dates, realized_pnls,
        joinpath(plots_dir, "$(lowercase(symbol))_equity.png");
        title_prefix=symbol)

    return (symbol=symbol, metrics=metrics)
end

# =============================================================================
# Main
# =============================================================================

results = [run_symbol(sym, spot, mult, run_dir) for (sym, spot, mult) in SYMBOLS]

println("\n", "=" ^ 70)
println("  SUMMARY")
println("=" ^ 70)
@printf("  %-6s  %8s  %8s  %8s  %8s  %5s\n", "Symbol", "Trades", "ROI", "Sharpe", "Sortino", "Win%")
println("  ", "-" ^ 62)
for r in results
    m = r.metrics
    @printf("  %-6s  %8d  %8s  %8s  %8s  %5s\n",
        r.symbol, m.count,
        fmt_metric(m.total_roi; pct=true),
        fmt_ratio(m.sharpe),
        fmt_ratio(m.sortino),
        fmt_pct(m.win_rate))
end
println("=" ^ 70)
