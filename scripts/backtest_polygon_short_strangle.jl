# Short Strangle Backtest on Polygon SPY Data
# Uses VolSurfaceAnalysis ScheduledStrategy interface and DuckDB readers
# Sells short put + short call using a custom strike selector
# Uses conservative OHLC bid/ask approximation from Polygon minute bars

using Pkg
Pkg.activate(@__DIR__)

using VolSurfaceAnalysis
using Dates
using CSV, DataFrames

# Configuration
const UNDERLYING_SYMBOL = "SPY"
const START_DATE = Date(2024, 1, 29)
const END_DATE_CUTOFF = Date(2025, 8, 1)

# Strategy parameters
const ENTRY_TIME_ET = Time(10, 0)  # 10:00 AM Eastern Time (DST-aware)
const EXPIRY_INTERVAL = Day(1)

const SHORT_SIGMAS = 0.8
const FIXED_WING_PCT = 0.02  # +/- 2% from spot (overrides sigma-based if selector set)
const TARGET_DELTA = 0.15   # 15-delta strangle (custom selector)
const RISK_FREE_RATE = 0.045
const DIV_YIELD = 0.013
const QUANTITY = 1.0
const TAU_TOL = 1e-6
const MIN_VOLUME = 0
const SPREAD_LAMBDA = 0.0

# Output directory (scripts/runs/<script>_<timestamp>)
const RUN_ID = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
const RUN_DIR    = joinpath(@__DIR__, "runs", "backtest_polygon_short_strangle_$(RUN_ID)")
const LATEST_DIR = joinpath(@__DIR__, "latest_runs", "backtest_polygon_short_strangle")

# Fixed-wing strike selector (+/- pct from spot)
function _pick_otm_strike(
    strikes::Vector{Float64},
    spot::Float64,
    target::Float64;
    side::Symbol
)::Float64
    otm = side == :put ? filter(s -> s < spot, strikes) : filter(s -> s > spot, strikes)
    candidates = !isempty(otm) ? otm : strikes
    distances = abs.(candidates .- target)
    return candidates[argmin(distances)]
end

function _fixed_wing_strangle(ctx, pct::Float64)::Union{Nothing,Tuple{Float64,Float64}}
    put_strikes = ctx.put_strikes
    call_strikes = ctx.call_strikes
    isempty(put_strikes) && return nothing
    isempty(call_strikes) && return nothing

    target_put = ctx.surface.spot * (1.0 - pct)
    target_call = ctx.surface.spot * (1.0 + pct)

    short_put_K = _pick_otm_strike(put_strikes, ctx.surface.spot, target_put; side=:put)
    short_call_K = _pick_otm_strike(call_strikes, ctx.surface.spot, target_call; side=:call)
    return (short_put_K, short_call_K)
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function main()
    mkpath(RUN_DIR)

    println("=" ^ 80)
    println("POLYGON SPY SHORT STRANGLE BACKTEST")
    println("=" ^ 80)
    println("Output directory: $RUN_DIR")
    println("WARNING: Using synthetic bid/ask from OHLC (conservative)")
    println("  bid = low, ask = high")
    println()

    println("Scanning for available dates...")
    all_dates = available_polygon_dates(DEFAULT_STORE, UNDERLYING_SYMBOL)
    isempty(all_dates) && error("No Polygon data found for $(UNDERLYING_SYMBOL)")

    filtered_dates = filter(d -> d >= START_DATE && d < END_DATE_CUTOFF, all_dates)
    isempty(filtered_dates) && error("No dates after filtering")

    println("Dates: $(length(filtered_dates)) from $(first(filtered_dates)) to $(last(filtered_dates))")

    entry_ts = build_entry_timestamps(filtered_dates, ENTRY_TIME_ET)
    entry_spots = read_polygon_spot_prices_for_timestamps(
        polygon_spot_root(DEFAULT_STORE),
        entry_ts;
        symbol=UNDERLYING_SYMBOL
    )
    println("Loaded entry spots: $(length(entry_spots))")

    println("Loading entry surfaces...")
    path_for_ts = ts -> polygon_options_path(DEFAULT_STORE, Date(ts), UNDERLYING_SYMBOL)
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
    isempty(surfaces) && error("No surfaces built (check data and filters)")

    schedule = sort(collect(keys(surfaces)))
    strike_selector = ctx -> VolSurfaceAnalysis._delta_strangle_strikes(
        ctx,
        TARGET_DELTA;
        rate=RISK_FREE_RATE,
        div_yield=DIV_YIELD,
        debug=false
    )
    strategy = ShortStrangleStrategy(
        schedule,
        EXPIRY_INTERVAL,
        SHORT_SIGMAS;
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
        symbol=UNDERLYING_SYMBOL
    )
    println("Loaded settlement spots: $(length(settlement_spots))")

    println("Running backtest...")
    positions, pnls = backtest_strategy(strategy, surfaces, settlement_spots)

    # Build per-leg detailed results (script-specific)
    detailed_results = DataFrame(
        EntryTime = DateTime[],
        EntrySpot = Float64[],
        Expiry = DateTime[],
        Strike = Float64[],
        Type = String[],
        Direction = Int[],
        EntryPrice = Float64[],
        ExitSpot = Float64[],
        PnL = Float64[]
    )
    for (pos, pnl) in zip(positions, pnls)
        ismissing(pnl) && continue
        exit_spot = get(settlement_spots, pos.trade.expiry, NaN)
        push!(detailed_results, (
            pos.entry_timestamp, pos.entry_spot, pos.trade.expiry,
            pos.trade.strike, string(pos.trade.option_type), pos.trade.direction,
            pos.entry_price, exit_spot, pnl
        ))
    end

    # Aggregate PnL and build results
    pnl_by_key, missing_n = aggregate_pnl(positions, pnls)
    realized = [pnl_by_key[k] for k in sort(collect(keys(pnl_by_key)); by=k -> k[1])]
    n_attempted = length(pnl_by_key) + missing_n
    df_results = pnl_results_dataframe(positions, pnls)

    metrics = performance_metrics(positions, pnls; margin_per_trade=12000.0)

    # Generate text report
    lines = format_backtest_report(
        metrics;
        title="POLYGON SPY SHORT STRANGLE BACKTEST RESULTS",
        subtitle="WARNING: SYNTHETIC BID/ASK SPREADS (CONSERVATIVE)\n  bid = low, ask = high from minute bars",
        params=[
            "Underlying" => "$UNDERLYING_SYMBOL",
            "Data range" => "$(first(filtered_dates)) to $(last(filtered_dates))",
            "Excluded dates" => ">= $END_DATE_CUTOFF",
            "Entry time" => "$(ENTRY_TIME_ET) ET (DST-aware)",
            "Expiry" => "~$EXPIRY_INTERVAL from entry",
            "Short legs" => "+/-$(round(TARGET_DELTA * 100, digits=0)) delta",
            "Quantity per leg" => "$QUANTITY"
        ],
        realized_pnls=realized,
        n_scheduled=length(schedule),
        n_attempted=n_attempted,
        n_positions=length(positions),
        n_missing=missing_n,
        margin_description="\$12000 per trade"
    )

    output_file = joinpath(RUN_DIR, "results_$(UNDERLYING_SYMBOL)_results.txt")
    open(output_file, "w") do io
        for line in lines
            println(io, line)
        end
    end

    for line in lines
        println(line)
    end

    println()
    println("Results written to: $output_file")

    csv_path = joinpath(RUN_DIR, "results_$(UNDERLYING_SYMBOL)_pnl.csv")
    CSV.write(csv_path, df_results)
    println("Daily P&L saved to: $csv_path")

    trades_csv_path = joinpath(RUN_DIR, "results_$(UNDERLYING_SYMBOL)_trades.csv")
    CSV.write(trades_csv_path, detailed_results)
    println("Detailed trades saved to: $trades_csv_path")

    metrics_csv_path = joinpath(RUN_DIR, "results_$(UNDERLYING_SYMBOL)_metrics.csv")
    metrics_df = metrics_to_dataframe(metrics)
    CSV.write(metrics_csv_path, metrics_df)
    println("Metrics saved to: $metrics_csv_path")

    # Settlement zone analysis (first year only)
    zone_df = settlement_zone_analysis(positions, settlement_spots; first_year_only=true)
    if !isempty(zone_df)
        zone_summary = settlement_zone_summary(zone_df)
        zone_path = joinpath(RUN_DIR, "results_$(UNDERLYING_SYMBOL)_zones.csv")
        zone_detail_path = joinpath(RUN_DIR, "results_$(UNDERLYING_SYMBOL)_zones_detail.csv")
        CSV.write(zone_path, zone_summary)
        CSV.write(zone_detail_path, zone_df)
        println("Settlement zone analysis (first year):")
        for row in eachrow(zone_summary)
            println("  $(row.zone): $(row.count) ($(row.percentage)%)")
        end
    end

    if !isempty(df_results)
        plot_path = joinpath(RUN_DIR, "plots", "short_strangle_$(UNDERLYING_SYMBOL)_pnl_distribution.png")
        save_pnl_and_equity_curve(
            df_results.EntryDate,
            df_results.PnL,
            plot_path;
            title_prefix="Short Strangle"
        )
        println("P&L distribution plot saved to: $plot_path")

        profit_path = joinpath(RUN_DIR, "plots", "short_strangle_$(UNDERLYING_SYMBOL)_profit_curve.png")
        save_profit_curve(
            df_results.EntryDate,
            df_results.PnL,
            profit_path;
            title="Short Strangle $(UNDERLYING_SYMBOL) - Profit per Trade"
        )
        println("Profit curve saved to: $profit_path")
    end

    if !isempty(entry_spots)
        spot_path = joinpath(RUN_DIR, "plots", "short_strangle_$(UNDERLYING_SYMBOL)_spot_curve.png")
        save_spot_curve(entry_spots, spot_path; title="Spot Curve $(UNDERLYING_SYMBOL)")
        println("Spot curve saved to: $spot_path")
    end

    # Overwrite the no-timestamp "latest" copy
    isdir(LATEST_DIR) && rm(LATEST_DIR; recursive=true)
    cp(RUN_DIR, LATEST_DIR)
    println("Latest run: $LATEST_DIR")
end

main()
