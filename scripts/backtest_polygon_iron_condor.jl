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
const MIN_VOLUME = 5
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
# Data loading helpers
# -----------------------------------------------------------------------------

function build_entry_timestamps(dates::Vector{Date})::Vector{DateTime}
    ts = DateTime[]
    for date in dates
        push!(ts, et_to_utc(date, ENTRY_TIME_ET))
    end
    return ts
end

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

    entry_ts = build_entry_timestamps(filtered_dates)
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

    attempted = Set{Tuple{DateTime, DateTime}}()
    pnl_by_key = Dict{Tuple{DateTime, DateTime}, Float64}()

    trades_csv = DataFrame(
        EntryTime = DateTime[],
        Expiry = DateTime[],
        Type = String[],
        Strike = Float64[],
        Direction = Int[],
        PnL = Float64[]
    )

    for (pos, pnl) in zip(positions, pnls)
        key = (pos.entry_timestamp, pos.trade.expiry)
        push!(attempted, key)

        if ismissing(pnl)
            continue
        end

        pnl_by_key[key] = get(pnl_by_key, key, 0.0) + pnl

        push!(trades_csv, (
            pos.entry_timestamp,
            pos.trade.expiry,
            string(pos.trade.option_type),
            pos.trade.strike,
            pos.trade.direction,
            pnl
        ))
    end

    realized_keys = sort(collect(keys(pnl_by_key)); by=k -> k[1])
    realized = [pnl_by_key[k] for k in realized_keys]

    missing_n = length(attempted) - length(realized_keys)
    total_pnl = isempty(realized) ? 0.0 : sum(realized)

    results_csv = DataFrame(EntryDate=Date[], PnL=Float64[], Result=String[])
    for key in realized_keys
        pnl_val = pnl_by_key[key]
        push!(results_csv, (Date(key[1]), pnl_val, pnl_val > 0 ? "Win" : "Loss"))
    end

    println("  Total P&L: \$$(round(total_pnl, digits=2))")
    println("  Missing settlements: $missing_n")

    lines = String[]
    push!(lines, "=" ^ 80)
    push!(lines, "POLYGON IRON CONDOR BACKTEST RESULTS")
    push!(lines, "=" ^ 80)
    push!(lines, "")
    push!(lines, "WARNING: SYNTHETIC BID/ASK SPREADS (CONSERVATIVE)")
    push!(lines, "  bid = low, ask = high from minute bars")
    push!(lines, "")
    push!(lines, "Underlying: $symbol")
    push!(lines, "Data range: $(first(filtered_dates)) to $(last(filtered_dates))")
    push!(lines, "Excluded dates >= $END_DATE_CUTOFF")
    push!(lines, "")
    push!(lines, "Strategy Parameters:")
    push!(lines, "  Entry time: $(ENTRY_TIME_ET) ET (DST-aware)")
    push!(lines, "  Expiry: ~$EXPIRY_INTERVAL from entry")
    push!(lines, "  Short legs: $(round(SHORT_DELTA_ABS * 100, digits=0)) delta")
    push!(lines, "  Wing selection: $(WING_OBJECTIVE) (max_loss \$$(CONDOR_MAX_LOSS_MIN)-\$$(CONDOR_MAX_LOSS_MAX), min credit \$$(CONDOR_MIN_CREDIT))")
    push!(lines, "  Quantity per leg: $QUANTITY")
    push!(lines, "")
    push!(lines, "Results:")
    push!(lines, "  Scheduled entries: $(length(schedule))")
    push!(lines, "  Actual condors: $(length(attempted))")
    push!(lines, "  Total positions: $(length(positions))")
    push!(lines, "  Missing settlements: $missing_n")
    push!(lines, "")
    push!(lines, "P&L (per condor):")
    push!(lines, "  Total: \$$(round(total_pnl, digits=2))")
    push!(lines, "  Count: $(length(realized))")

    if !isempty(realized)
        avg_pnl = total_pnl / length(realized)
        push!(lines, "  Average: \$$(round(avg_pnl, digits=2))")
        push!(lines, "  Min: \$$(round(minimum(realized), digits=2))")
        push!(lines, "  Max: \$$(round(maximum(realized), digits=2))")
        winners = count(x -> x > 0, realized)
        win_rate = winners / length(realized) * 100
        push!(lines, "  Winners: $winners / $(length(realized)) ($(round(win_rate, digits=1))%)")
    end

    push!(lines, "")
    push!(lines, "Performance Metrics (return basis: per-condor max loss):")
    push!(lines, "  Total ROI: $(metrics.total_roi === missing ? "n/a" : string(round(metrics.total_roi * 100, digits=2)) * "%")")
    push!(lines, "  Annualized ROI (CAGR): $(metrics.annualized_roi_cagr === missing ? "n/a" : string(round(metrics.annualized_roi_cagr * 100, digits=2)) * "%")")
    push!(lines, "  Sharpe: $(metrics.sharpe === missing ? "n/a" : string(round(metrics.sharpe, digits=2)))")
    push!(lines, "  Sortino: $(metrics.sortino === missing ? "n/a" : string(round(metrics.sortino, digits=2)))")
    push!(lines, "  Win Rate: $(metrics.win_rate === missing ? "n/a" : string(round(metrics.win_rate * 100, digits=1)) * "%")")
    push!(lines, "")
    push!(lines, "=" ^ 80)

    # Print prominent ROI summary to terminal
    _fmt(v, pct) = v === missing ? "n/a" : pct ? "$(round(v * 100, digits=2))%" : "$(round(v, digits=2))"
    println()
    println("  ===== ROI SUMMARY ($symbol) =====")
    println("  Total ROI:      $(_fmt(metrics.total_roi, true))")
    println("  CAGR:           $(_fmt(metrics.annualized_roi_cagr, true))")
    println("  Sharpe/Sortino: $(_fmt(metrics.sharpe, false)) / $(_fmt(metrics.sortino, false))")
    println("  Win Rate:       $(_fmt(metrics.win_rate, true))")
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
    metrics_df = DataFrame(
        Metric = [
            "count", "missing", "total_pnl", "avg_pnl", "min_pnl", "max_pnl", "win_rate",
            "avg_bid_ask_spread_rel",
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
            metrics.avg_bid_ask_spread_rel,
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
