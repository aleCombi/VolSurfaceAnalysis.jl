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
const POLYGON_ROOT = raw"C:\repos\DeribitVols\data\massive_parquet\minute_aggs"
const SPOT_ROOT = raw"C:\repos\DeribitVols\data\massive_parquet\spot_1min"
const DEFAULT_TARGET_SYMBOLS = ["SPY", "QQQ", "IWM", "GLD", "SLV", "TLT", "USO", "HYG", "GDX", "XLE", "XLF", "VIX"]
const TARGET_SYMBOLS = ["SPY"]
const END_DATE_CUTOFF = Date(2025, 8, 1)

# Strategy parameters
const ENTRY_TIME_ET = Time(10, 0)  # 10:00 AM Eastern Time (DST-aware)
const EXPIRY_INTERVAL = Day(1)

const SHORT_SIGMAS = 0.7
const LONG_SIGMAS = 1.5
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

function _condor_group_max_loss(group_positions::Vector{Position})::Union{Missing,Float64}
    short_puts = Float64[]
    long_puts = Float64[]
    short_calls = Float64[]
    long_calls = Float64[]
    total_entry_cost = 0.0

    for pos in group_positions
        total_entry_cost += entry_cost(pos)
        t = pos.trade
        if t.option_type == Put
            if t.direction < 0
                push!(short_puts, t.strike)
            else
                push!(long_puts, t.strike)
            end
        else
            if t.direction < 0
                push!(short_calls, t.strike)
            else
                push!(long_calls, t.strike)
            end
        end
    end

    if isempty(short_puts) || isempty(long_puts) || isempty(short_calls) || isempty(long_calls)
        return missing
    end

    short_put = maximum(short_puts)
    long_put = minimum(long_puts)
    short_call = minimum(short_calls)
    long_call = maximum(long_calls)

    put_width = abs(short_put - long_put)
    call_width = abs(long_call - short_call)
    max_width = max(put_width, call_width)

    credit = -total_entry_cost
    max_loss = max_width - credit
    return max(max_loss, 0.0)
end

function _first_condor_margin(
    positions::Vector{Position};
    buffer::Float64=0.3
)::Union{Nothing,Float64}
    groups = Dict{Tuple{DateTime,DateTime}, Vector{Position}}()
    for pos in positions
        key = (pos.entry_timestamp, pos.trade.expiry)
        push!(get!(groups, key, Position[]), pos)
    end
    isempty(groups) && return nothing

    for key in sort(collect(keys(groups)); by=k -> k[1])
        max_loss = _condor_group_max_loss(groups[key])
        if max_loss !== missing
            return max_loss * (1 + buffer)
        end
    end

    return nothing
end

function run_symbol_backtest(symbol::String)
    println("=" ^ 60)
    println("PROCESSING SYMBOL: $symbol")
    println("=" ^ 60)

    all_dates = available_dates(POLYGON_ROOT, symbol)
    filtered_dates = filter(d -> d < END_DATE_CUTOFF, all_dates)
    if isempty(filtered_dates)
        println("  No data found for $symbol")
        return
    end

    println("  Dates: $(length(filtered_dates)) range $(first(filtered_dates)) to $(last(filtered_dates))")

    entry_ts = build_entry_timestamps(filtered_dates)
    entry_spots = read_polygon_spot_prices_for_timestamps(
        SPOT_ROOT,
        entry_ts;
        symbol=symbol
    )
    println("  Loaded entry spots: $(length(entry_spots))")

    path_for_ts = ts -> begin
        date_str = Dates.format(Date(ts), "yyyy-mm-dd")
        joinpath(POLYGON_ROOT, "date=$date_str", "underlying=$symbol", "data.parquet")
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

    if isempty(surfaces)
        println("  No entry surfaces built for $symbol")
        return
    end

    schedule = sort(collect(keys(surfaces)))
    strategy = IronCondorStrategy(
        schedule,
        EXPIRY_INTERVAL,
        SHORT_SIGMAS,
        LONG_SIGMAS,
        RISK_FREE_RATE,
        DIV_YIELD,
        QUANTITY,
        TAU_TOL,
        false
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

    margin_per_trade = _first_condor_margin(positions; buffer=0.3)
    metrics = performance_metrics(positions, pnls; margin_per_trade=margin_per_trade)

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
    push!(lines, "  Short legs: +/-$(SHORT_SIGMAS) sigma from ATM IV")
    push!(lines, "  Long legs: +/-$(LONG_SIGMAS) sigma from ATM IV")
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
    margin_label = margin_per_trade === nothing ? "n/a" : "\$$(round(margin_per_trade, digits=2))"
    push!(lines, "Performance Metrics (margin per trade: $margin_label; first condor max loss + 30%):")
    push!(lines, "  Total ROI: $(metrics.total_roi === missing ? "n/a" : string(round(metrics.total_roi * 100, digits=2)) * "%")")
    push!(lines, "  Sharpe: $(metrics.sharpe === missing ? "n/a" : string(round(metrics.sharpe, digits=2)))")
    push!(lines, "  Sortino: $(metrics.sortino === missing ? "n/a" : string(round(metrics.sortino, digits=2)))")
    push!(lines, "  Win Rate: $(metrics.win_rate === missing ? "n/a" : string(round(metrics.win_rate * 100, digits=1)) * "%")")
    push!(lines, "")
    push!(lines, "=" ^ 80)

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
end

main()
