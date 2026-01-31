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
const POLYGON_ROOT = raw"C:\repos\DeribitVols\data\massive_parquet\minute_aggs"
const SPOT_ROOT = raw"C:\repos\DeribitVols\data\massive_parquet\spot_1min"
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
const MIN_VOLUME = 5

# Output directory (scripts/runs/<script>_<timestamp>)
const RUN_ID = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
const SCRIPT_BASE = splitext(basename(PROGRAM_FILE))[1]
const RUN_DIR = joinpath(@__DIR__, "runs", "$(SCRIPT_BASE)_$(RUN_ID)")

# -----------------------------------------------------------------------------
# Data loading helpers (use src readers)
# -----------------------------------------------------------------------------

function available_dates(root::String, symbol::String)::Vector{Date}
    dirs = readdir(root)
    dates = Date[]
    for dir in dirs
        m = match(r"date=(\d{4})-(\d{2})-(\d{2})", dir)
        m === nothing && continue
        date = Date(parse(Int, m[1]), parse(Int, m[2]), parse(Int, m[3]))
        path = joinpath(root, dir, "underlying=$(symbol)", "data.parquet")
        isfile(path) && push!(dates, date)
    end
    return sort(dates)
end

function build_entry_timestamps(dates::Vector{Date})::Vector{DateTime}
    ts = DateTime[]
    for date in dates
        push!(ts, et_to_utc(date, ENTRY_TIME_ET))
    end
    return ts
end

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
    all_dates = available_dates(POLYGON_ROOT, UNDERLYING_SYMBOL)
    isempty(all_dates) && error("No Polygon data found for $(UNDERLYING_SYMBOL)")

    filtered_dates = filter(d -> d >= START_DATE && d < END_DATE_CUTOFF, all_dates)
    isempty(filtered_dates) && error("No dates after filtering")

    println("Dates: $(length(filtered_dates)) from $(first(filtered_dates)) to $(last(filtered_dates))")

    entry_ts = build_entry_timestamps(filtered_dates)
    entry_spots = read_polygon_spot_prices_for_timestamps(
        SPOT_ROOT,
        entry_ts;
        symbol=UNDERLYING_SYMBOL
    )
    println("Loaded entry spots: $(length(entry_spots))")

    println("Loading entry surfaces...")
    path_for_ts = ts -> begin
        date_str = Dates.format(Date(ts), "yyyy-mm-dd")
        joinpath(POLYGON_ROOT, "date=$date_str", "underlying=$UNDERLYING_SYMBOL", "data.parquet")
    end
    read_records = (path; where="") -> read_polygon_option_records(
        path,
        entry_spots;
        where=where,
        min_volume=MIN_VOLUME,
        warn=true
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
        SPOT_ROOT,
        expiry_ts;
        symbol=UNDERLYING_SYMBOL
    )
    println("Loaded settlement spots: $(length(settlement_spots))")

    println("Running backtest...")
    positions, pnls = backtest_strategy(strategy, surfaces, settlement_spots)

    attempted = Set{Tuple{DateTime, DateTime}}()
    pnl_by_key = Dict{Tuple{DateTime, DateTime}, Float64}()

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
        key = (pos.entry_timestamp, pos.trade.expiry)
        push!(attempted, key)

        if ismissing(pnl)
            continue
        end

        pnl_by_key[key] = get(pnl_by_key, key, 0.0) + pnl
        exit_spot = get(settlement_spots, pos.trade.expiry, NaN)

        push!(detailed_results, (
            pos.entry_timestamp,
            pos.entry_spot,
            pos.trade.expiry,
            pos.trade.strike,
            string(pos.trade.option_type),
            pos.trade.direction,
            pos.entry_price,
            exit_spot,
            pnl
        ))
    end

    realized_keys = collect(keys(pnl_by_key))
    realized = [pnl_by_key[k] for k in realized_keys]

    missing_n = length(attempted) - length(realized_keys)
    total_pnl = isempty(realized) ? 0.0 : sum(realized)
    avg_pnl = isempty(realized) ? 0.0 : total_pnl / length(realized)

    metrics = performance_metrics(positions, pnls; margin_per_trade=12000.0)

    lines = String[]
    push!(lines, "=" ^ 80)
    push!(lines, "POLYGON SPY SHORT STRANGLE BACKTEST RESULTS")
    push!(lines, "=" ^ 80)
    push!(lines, "")
    push!(lines, "WARNING: SYNTHETIC BID/ASK SPREADS (CONSERVATIVE)")
    push!(lines, "  bid = low, ask = high from minute bars")
    push!(lines, "")
    push!(lines, "Underlying: $UNDERLYING_SYMBOL")
    push!(lines, "Data range: $(first(filtered_dates)) to $(last(filtered_dates))")
    push!(lines, "Excluded dates >= $END_DATE_CUTOFF")
    push!(lines, "")
    push!(lines, "Strategy Parameters:")
    push!(lines, "  Entry time: $(ENTRY_TIME_ET) ET (DST-aware)")
    push!(lines, "  Expiry: ~$EXPIRY_INTERVAL from entry")
    push!(lines, "  Short legs: +/-$(round(TARGET_DELTA * 100, digits=0)) delta")
    push!(lines, "  Quantity per leg: $QUANTITY")
    push!(lines, "")
    push!(lines, "Results:")
    push!(lines, "  Scheduled entries: $(length(schedule))")
    push!(lines, "  Actual strangles: $(length(attempted))")
    push!(lines, "  Total positions: $(length(positions))")
    push!(lines, "  Missing settlements: $missing_n")
    push!(lines, "")
    push!(lines, "P&L (per strangle):")
    push!(lines, "  Total: \$$(round(total_pnl, digits=2))")
    push!(lines, "  Average: \$$(round(avg_pnl, digits=2))")
    push!(lines, "  Count: $(length(realized))")

    if !isempty(realized)
        push!(lines, "  Min: \$$(round(minimum(realized), digits=2))")
        push!(lines, "  Max: \$$(round(maximum(realized), digits=2))")
        winners = count(x -> x > 0, realized)
        win_rate = winners / length(realized) * 100
        push!(lines, "  Winners: $winners / $(length(realized)) ($(round(win_rate, digits=1))%)")
    end

    push!(lines, "")
    push!(lines, "Performance Metrics (margin per trade: \$12000):")
    push!(lines, "  Total ROI: $(metrics.total_roi === missing ? "n/a" : string(round(metrics.total_roi * 100, digits=2)) * "%")")
    push!(lines, "  Sharpe: $(metrics.sharpe === missing ? "n/a" : string(round(metrics.sharpe, digits=2)))")
    push!(lines, "  Sortino: $(metrics.sortino === missing ? "n/a" : string(round(metrics.sortino, digits=2)))")
    push!(lines, "  Win Rate: $(metrics.win_rate === missing ? "n/a" : string(round(metrics.win_rate * 100, digits=1)) * "%")")

    push!(lines, "")
    push!(lines, "=" ^ 80)

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

    df_results = DataFrame(
        EntryDate = Date[],
        EntryTime = DateTime[],
        ExpiryDate = Date[],
        PnL = Float64[],
        Result = String[]
    )

    for key in sort(realized_keys; by=k -> k[1])
        pnl_val = pnl_by_key[key]
        push!(df_results, (
            Date(key[1]),
            key[1],
            Date(key[2]),
            pnl_val,
            pnl_val > 0 ? "Win" : "Loss"
        ))
    end

    csv_path = joinpath(RUN_DIR, "results_$(UNDERLYING_SYMBOL)_pnl.csv")
    CSV.write(csv_path, df_results)
    println("Daily P&L saved to: $csv_path")

    trades_csv_path = joinpath(RUN_DIR, "results_$(UNDERLYING_SYMBOL)_trades.csv")
    CSV.write(trades_csv_path, detailed_results)
    println("Detailed trades saved to: $trades_csv_path")

    metrics_csv_path = joinpath(RUN_DIR, "results_$(UNDERLYING_SYMBOL)_metrics.csv")
    metrics_df = DataFrame(
        Metric = [
            "count", "missing", "total_pnl", "avg_pnl", "min_pnl", "max_pnl", "win_rate",
            "total_roi", "annualized_roi_simple", "annualized_roi_cagr",
            "avg_return", "volatility", "sharpe", "sortino",
            "duration_days", "duration_years"
        ],
        Value = [
            metrics.count,
            metrics.missing,
            metrics.total_pnl,
            metrics.avg_pnl,
            metrics.min_pnl,
            metrics.max_pnl,
            metrics.win_rate,
            metrics.total_roi,
            metrics.annualized_roi_simple,
            metrics.annualized_roi_cagr,
            metrics.avg_return,
            metrics.volatility,
            metrics.sharpe,
            metrics.sortino,
            metrics.duration_days,
            metrics.duration_years
        ]
    )
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
end

main()
