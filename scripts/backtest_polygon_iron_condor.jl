# Iron Condor Backtest on Polygon Data (Multi-Symbol)
# Uses VolSurfaceAnalysis ScheduledStrategy interface and DuckDB readers
# Uses conservative OHLC bid/ask approximation from Polygon minute bars
# Excludes data on/after END_DATE_CUTOFF

using Pkg
Pkg.activate(@__DIR__)

using VolSurfaceAnalysis
using Dates
using CSV, DataFrames

# Configuration
const DEFAULT_TARGET_SYMBOLS = ["SPY", "QQQ", "IWM", "GLD", "SLV", "TLT", "USO", "HYG", "GDX", "XLE", "XLF", "VIX"]
const TARGET_SYMBOLS = ["SPY"]
const END_DATE_CUTOFF = Date(2025, 8, 1)

# Strategy parameters
const ENTRY_TIME_ET = Time(10, 0)  # 10:00 AM Eastern Time (DST-aware)
const EXPIRY_INTERVAL = Day(1)

const SHORT_SIGMAS = 0.7
const LONG_SIGMAS = 1.5
const SHORT_DELTA_ABS = 0.16
const MIN_DELTA_GAP = 0.08
const RISK_FREE_RATE = 0.045
const DIV_YIELD = 0.013
const QUANTITY = 1.0
const TAU_TOL = 1e-6
const MIN_VOLUME = 0
const SPREAD_LAMBDA = 0.0

# Wing selection: ROI-optimized (maximize credit / max_loss)
const WING_OBJECTIVE = :roi          # :roi, :target_max_loss, or :pnl
const PREFER_SYMMETRIC_WINGS = false
const CONDOR_MAX_LOSS_MIN = 5.0      # Min acceptable max loss ($)
const CONDOR_MAX_LOSS_MAX = 30.0     # Max acceptable max loss ($)
const CONDOR_MIN_CREDIT = 0.10       # Min entry credit ($)

# Output directory (scripts/runs/<script>_<timestamp>)
const RUN_ID = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
const RUN_DIR    = joinpath(@__DIR__, "runs", "backtest_polygon_iron_condor_$(RUN_ID)")
const LATEST_DIR = joinpath(@__DIR__, "latest_runs", "backtest_polygon_iron_condor")

# -----------------------------------------------------------------------------
# Backtest
# -----------------------------------------------------------------------------

function run_symbol_backtest(symbol::String)
    println("=" ^ 60)
    println("PROCESSING SYMBOL: $symbol")
    println("=" ^ 60)

    all_dates = available_polygon_dates(DEFAULT_STORE, symbol)
    filtered_dates = filter(d -> d < END_DATE_CUTOFF, all_dates)
    if isempty(filtered_dates)
        println("  No data found for $symbol")
        return
    end

    println("  Dates: $(length(filtered_dates)) range $(first(filtered_dates)) to $(last(filtered_dates))")

    entry_ts = build_entry_timestamps(filtered_dates, ENTRY_TIME_ET)
    entry_spots = read_polygon_spot_prices_for_timestamps(
        polygon_spot_root(DEFAULT_STORE),
        entry_ts;
        symbol=symbol
    )
    println("  Loaded entry spots: $(length(entry_spots))")

    path_for_ts = ts -> polygon_options_path(DEFAULT_STORE, Date(ts), symbol)
    read_records = (path; where="") -> read_polygon_option_records(
        path,
        entry_spots;
        where=where,
        min_volume=MIN_VOLUME,
        warn=true,
        spread_lambda=SPREAD_LAMBDA
    )
    surfaces = build_surfaces_for_timestamps(
        entry_ts;
        path_for_timestamp=path_for_ts,
        read_records=read_records
    )

    if isempty(surfaces)
        println("  No entry surfaces built for $symbol")
        return
    end

    schedule = sort(collect(keys(surfaces)))
    strike_selector = ctx -> begin
        # Short legs: delta-based placement
        shorts = VolSurfaceAnalysis._delta_strangle_strikes_asymmetric(
            ctx,
            SHORT_DELTA_ABS,
            SHORT_DELTA_ABS;
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD
        )
        shorts === nothing && return nothing
        short_put_K, short_call_K = shorts

        # Wings: ROI-optimized (maximize credit / max_loss)
        wings = VolSurfaceAnalysis._condor_wings_by_objective(
            ctx,
            short_put_K,
            short_call_K;
            objective=WING_OBJECTIVE,
            max_loss_min=CONDOR_MAX_LOSS_MIN,
            max_loss_max=CONDOR_MAX_LOSS_MAX,
            min_credit=CONDOR_MIN_CREDIT,
            min_delta_gap=MIN_DELTA_GAP,
            prefer_symmetric=PREFER_SYMMETRIC_WINGS,
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            debug=false
        )
        wings === nothing && return nothing
        long_put_K, long_call_K = wings
        return (short_put_K, short_call_K, long_put_K, long_call_K)
    end
    strategy = IronCondorStrategy(
        schedule,
        EXPIRY_INTERVAL,
        SHORT_SIGMAS,
        LONG_SIGMAS;
        rate=RISK_FREE_RATE,
        div_yield=DIV_YIELD,
        quantity=QUANTITY,
        tau_tol=TAU_TOL,
        debug=false,
        strike_selector=strike_selector
    )

    expiry_ts = DateTime[]
    for ts in schedule
        positions = entry_positions(strategy, surfaces[ts])
        for pos in positions
            push!(expiry_ts, pos.trade.expiry)
        end
    end
    expiry_ts = unique(expiry_ts)

    settlement_spots = read_polygon_spot_prices_for_timestamps(
        polygon_spot_root(DEFAULT_STORE),
        expiry_ts;
        symbol=symbol
    )
    println("  Loaded settlement spots: $(length(settlement_spots))")

    println("  Running backtest...")
    positions, pnls = backtest_strategy(strategy, surfaces, settlement_spots)
    println("  Positions: $(length(positions))")

    if isempty(positions)
        println("  No positions entered.")
        return
    end

    metrics = performance_metrics(positions, pnls; margin_by_key=condor_max_loss_by_key(positions))

    # Build per-leg trades CSV (script-specific detail)
    trades_csv = DataFrame(
        EntryTime = DateTime[],
        Expiry = DateTime[],
        Type = String[],
        Strike = Float64[],
        Direction = Int[],
        PnL = Float64[]
    )
    for (pos, pnl) in zip(positions, pnls)
        ismissing(pnl) && continue
        push!(trades_csv, (
            pos.entry_timestamp,
            pos.trade.expiry,
            string(pos.trade.option_type),
            pos.trade.strike,
            pos.trade.direction,
            pnl
        ))
    end

    # Aggregate PnL and build results
    pnl_by_key, missing_n = aggregate_pnl(positions, pnls)
    realized = [pnl_by_key[k] for k in sort(collect(keys(pnl_by_key)); by=k -> k[1])]
    n_attempted = length(pnl_by_key) + missing_n
    results_csv = pnl_results_dataframe(positions, pnls)

    total_pnl = isempty(realized) ? 0.0 : sum(realized)
    println("  Total P&L: \$$(round(total_pnl, digits=2))")
    println("  Missing settlements: $missing_n")

    # Generate text report
    lines = format_backtest_report(
        metrics;
        title="POLYGON IRON CONDOR BACKTEST RESULTS",
        subtitle="WARNING: SYNTHETIC BID/ASK SPREADS (CONSERVATIVE)\n  bid = low, ask = high from minute bars",
        params=[
            "Underlying" => "$symbol",
            "Data range" => "$(first(filtered_dates)) to $(last(filtered_dates))",
            "Excluded dates" => ">= $END_DATE_CUTOFF",
            "Entry time" => "$(ENTRY_TIME_ET) ET (DST-aware)",
            "Expiry" => "~$EXPIRY_INTERVAL from entry",
            "Short legs" => "$(round(SHORT_DELTA_ABS * 100, digits=0)) delta",
            "Wing selection" => "$(WING_OBJECTIVE) (max_loss \$$(CONDOR_MAX_LOSS_MIN)-\$$(CONDOR_MAX_LOSS_MAX), min credit \$$(CONDOR_MIN_CREDIT))",
            "Quantity per leg" => "$QUANTITY"
        ],
        realized_pnls=realized,
        n_scheduled=length(schedule),
        n_attempted=n_attempted,
        n_positions=length(positions),
        n_missing=missing_n,
        margin_description="per-condor max loss"
    )

    # Print prominent ROI summary to terminal
    println()
    println("  ===== ROI SUMMARY ($symbol) =====")
    println("  Total ROI:      $(fmt_metric(metrics.total_roi; pct=true))")
    println("  CAGR:           $(fmt_metric(metrics.annualized_roi_cagr; pct=true))")
    println("  Sharpe/Sortino: $(fmt_metric(metrics.sharpe)) / $(fmt_metric(metrics.sortino))")
    println("  Win Rate:       $(fmt_metric(metrics.win_rate; pct=true))")
    println("  =================================")

    results_txt_path = joinpath(RUN_DIR, "results_$(symbol)_results.txt")
    open(results_txt_path, "w") do io
        for line in lines
            println(io, line)
        end
    end

    pnl_path = joinpath(RUN_DIR, "results_$(symbol)_pnl.csv")
    trades_path = joinpath(RUN_DIR, "results_$(symbol)_trades.csv")
    CSV.write(pnl_path, results_csv)
    CSV.write(trades_path, trades_csv)

    println("  Results written to $results_txt_path")
    println("  Saved results to $pnl_path")

    metrics_csv_path = joinpath(RUN_DIR, "results_$(symbol)_metrics.csv")
    metrics_df = metrics_to_dataframe(metrics)
    CSV.write(metrics_csv_path, metrics_df)
    println("  Metrics saved to $metrics_csv_path")

    # Settlement zone analysis (first year only)
    zone_df = settlement_zone_analysis(positions, settlement_spots; first_year_only=true)
    if !isempty(zone_df)
        zone_summary = settlement_zone_summary(zone_df)
        zone_path = joinpath(RUN_DIR, "results_$(symbol)_zones.csv")
        zone_detail_path = joinpath(RUN_DIR, "results_$(symbol)_zones_detail.csv")
        CSV.write(zone_path, zone_summary)
        CSV.write(zone_detail_path, zone_df)
        println("  Settlement zone analysis (first year):")
        for row in eachrow(zone_summary)
            println("    $(row.zone): $(row.count) ($(row.percentage)%)")
        end
    end

    if !isempty(results_csv)
        plot_path = joinpath(RUN_DIR, "plots", "iron_condor_$(symbol)_pnl_distribution.png")
        save_pnl_and_equity_curve(
            results_csv.EntryDate,
            results_csv.PnL,
            plot_path;
            title_prefix="Iron Condor $(symbol)"
        )
        println("  P&L distribution plot saved to $plot_path")

        profit_path = joinpath(RUN_DIR, "plots", "iron_condor_$(symbol)_profit_curve.png")
        save_profit_curve(
            results_csv.EntryDate,
            results_csv.PnL,
            profit_path;
            title="Iron Condor $(symbol) - Profit per Trade"
        )
        println("  Profit curve saved to $profit_path")
    end

    if !isempty(entry_spots)
        spot_path = joinpath(RUN_DIR, "plots", "iron_condor_$(symbol)_spot_curve.png")
        save_spot_curve(entry_spots, spot_path; title="Spot Curve $(symbol)")
        println("  Spot curve saved to $spot_path")
    end
    println("-" ^ 60)
end

function main()
    mkpath(RUN_DIR)
    println("Output directory: $RUN_DIR")

    for symbol in TARGET_SYMBOLS
        try
            run_symbol_backtest(symbol)
        catch e
            println("ERROR processing $symbol: $e")
            Base.showerror(stdout, e, catch_backtrace())
        end
    end

    # Overwrite the no-timestamp "latest" copy
    isdir(LATEST_DIR) && rm(LATEST_DIR; recursive=true)
    cp(RUN_DIR, LATEST_DIR)
    println("Latest run: $LATEST_DIR")
end

main()
