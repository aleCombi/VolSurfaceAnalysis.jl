# Iron-condor experiment helpers.

using Dates
using Printf
using Statistics

const CondorGridSummaryRow = @NamedTuple begin
    symbol::String
    delta::Float64
    max_loss_cap::Float64
    trades::Int
    win_rate::Float64
    mean_pnl::Float64
    std_pnl::Float64
    min_pnl::Float64
    p1_pnl::Float64
    p5_pnl::Float64
    p10_pnl::Float64
    p25_pnl::Float64
    median_pnl::Float64
    p75_pnl::Float64
    p90_pnl::Float64
    p95_pnl::Float64
    p99_pnl::Float64
    max_pnl::Float64
    total_pnl::Float64
    mean_maxloss::Float64
    mean_credit::Float64
    avg_win::Float64
    avg_loss::Float64
    skewness::Float64
end

function condor_trade_pnls(result)
    isempty(result.positions) && return Float64[], Float64[], Float64[]
    df = condor_trade_table(result.positions, result.pnl)
    isempty(df) && return Float64[], Float64[], Float64[]

    pnls = Float64[]
    maxlosses = Float64[]
    credits = Float64[]
    for row in eachrow(df)
        ismissing(row.PnL) && continue
        ismissing(row.MaxLoss) && continue
        row.MaxLoss <= 0 && continue
        push!(pnls, row.PnL)
        push!(maxlosses, row.MaxLoss)
        ismissing(row.Credit) || push!(credits, row.Credit)
    end
    return pnls, maxlosses, credits
end

function condor_grid_summary_row(;
    symbol::AbstractString,
    delta::Real,
    max_loss_cap::Real,
    pnls,
    maxlosses=Float64[],
    credits=Float64[],
)::CondorGridSummaryRow
    stats = pnl_distribution_stats(pnls)
    clean_maxlosses = finite_values(maxlosses)
    clean_credits = finite_values(credits)

    return (
        symbol=String(symbol),
        delta=Float64(delta),
        max_loss_cap=Float64(max_loss_cap),
        trades=stats.trades,
        win_rate=stats.win_rate,
        mean_pnl=stats.mean_pnl,
        std_pnl=stats.std_pnl,
        min_pnl=stats.min_pnl,
        p1_pnl=stats.p1_pnl,
        p5_pnl=stats.p5_pnl,
        p10_pnl=stats.p10_pnl,
        p25_pnl=stats.p25_pnl,
        median_pnl=stats.median_pnl,
        p75_pnl=stats.p75_pnl,
        p90_pnl=stats.p90_pnl,
        p95_pnl=stats.p95_pnl,
        p99_pnl=stats.p99_pnl,
        max_pnl=stats.max_pnl,
        total_pnl=stats.total_pnl,
        mean_maxloss=isempty(clean_maxlosses) ? 0.0 : mean(clean_maxlosses),
        mean_credit=isempty(clean_credits) ? 0.0 : mean(clean_credits),
        avg_win=stats.avg_win,
        avg_loss=stats.avg_loss,
        skewness=stats.skewness,
    )
end

function build_combo_wing_dataset(
    symbol,
    start_date,
    end_date,
    entry_time,
    expiry_interval,
    max_tau,
    delta_combos,
    wing_widths,
    rate,
    div_yield,
    spread_lambda,
)
    (; source, sched) = polygon_parquet_source(symbol;
        start_date=start_date,
        end_date=end_date,
        entry_time=entry_time,
        rate=rate,
        div_yield=div_yield,
        spread_lambda=spread_lambda,
    )

    n_combos = length(delta_combos)
    n_wings = length(wing_widths)
    dates = Date[]
    pnls = Vector{Matrix{Float64}}()
    n_total = 0
    n_skip = 0

    each_entry(source, expiry_interval, sched; clear_cache=true) do ctx, settlement
        ismissing(settlement) && return
        n_total += 1
        dctx = delta_context(ctx; rate=rate, div_yield=div_yield)
        dctx === nothing && return
        if dctx.tau * 365.25 > max_tau
            n_skip += 1
            return
        end
        spot = dctx.spot

        matrix = fill(NaN, n_combos, n_wings)
        for (ci, (pd, cd)) in enumerate(delta_combos)
            sp_K = delta_strike(dctx, -pd, Put)
            sc_K = delta_strike(dctx, cd, Call)
            (sp_K === nothing || sc_K === nothing) && continue
            for (wi, ww) in enumerate(wing_widths)
                lp_K = nearest_otm_strike(dctx, sp_K, ww, Put)
                lc_K = nearest_otm_strike(dctx, sc_K, ww, Call)
                (lp_K === nothing || lc_K === nothing) && continue
                positions = open_condor_positions(ctx, sp_K, sc_K, lp_K, lc_K)
                length(positions) == 4 || continue
                matrix[ci, wi] = settle(positions, Float64(settlement)) * spot
            end
        end
        push!(dates, Date(ctx.surface.timestamp))
        push!(pnls, matrix)
    end

    ord = sortperm(dates)
    return dates[ord], pnls[ord], n_total, n_skip
end

function print_condor_percentile_ladder(rows, symbol, deltas, max_loss_cap)
    println("\n  Full PnL percentile ladder at max_loss=$max_loss_cap -- $symbol:")
    @printf("  %8s  %5s  %7s  %7s  %7s  %7s  %7s  %7s  %7s  %7s  %7s  %7s  %7s  %6s  %6s\n",
        "delta", "n", "min", "p1", "p5", "p10", "p25", "med", "p75", "p90", "p95", "p99", "max", "skew", "w/l")
    println("  ", "-"^115)
    for delta in deltas
        row = nothing
        for r in rows
            if r.symbol == symbol && r.delta == delta && r.max_loss_cap == max_loss_cap
                row = r
                break
            end
        end
        row === nothing && continue
        ratio = row.avg_loss != 0.0 ? row.avg_win / abs(row.avg_loss) : 0.0
        @printf("  d=%.2f   %4d  %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f  %+5.2f  %5.2f\n",
            delta, row.trades,
            row.min_pnl, row.p1_pnl, row.p5_pnl, row.p10_pnl, row.p25_pnl,
            row.median_pnl, row.p75_pnl, row.p90_pnl, row.p95_pnl, row.p99_pnl, row.max_pnl,
            row.skewness, ratio)
    end
end

function print_condor_cross_symbol_comparison(rows, symbols, delta, max_loss_cap)
    println("\n\n", "="^100)
    println("  Cross-symbol comparison at d=$delta ml=$max_loss_cap (PnL normalized to base units)")
    println("="^100)
    @printf("  %-6s  %5s  %7s  %7s  %7s  %7s  %7s  %7s  %6s  %7s  %6s\n",
        "sym", "n", "mean", "std", "median", "p5", "p1", "total", "skew", "w_rate", "w/l")
    println("  ", "-"^90)
    for sym in symbols
        row = nothing
        for r in rows
            if r.symbol == sym && r.delta == delta && r.max_loss_cap == max_loss_cap
                row = r
                break
            end
        end
        row === nothing && continue
        ratio = row.avg_loss != 0.0 ? row.avg_win / abs(row.avg_loss) : 0.0
        @printf("  %-6s  %4d  %+7.4f  %7.4f  %+7.4f  %+7.3f  %+7.3f  %+7.2f  %+5.2f  %6.1f%%  %5.2f\n",
            sym, row.trades, row.mean_pnl, row.std_pnl, row.median_pnl,
            row.p5_pnl, row.p1_pnl, row.total_pnl, row.skewness, row.win_rate * 100, ratio)
    end
end

function run_condor_grid_experiment(;
    output_root::AbstractString,
    symbols=[
        ("SPY", "SPY", 1.0),
        ("QQQ", "QQQ", 1.0),
        ("IWM", "IWM", 1.0),
        ("SPXW", "SPY", 10.0),
    ],
    spread_lambda::Real=0.7,
    entry_time::Time=Time(10, 0),
    expiry_interval=Day(1),
    rate::Real=0.045,
    div_yield::Real=0.013,
    max_spread_rel::Real=0.50,
    start_date::Date=Date(2024, 1, 1),
    end_date::Date=Date(2025, 12, 31),
    delta_grid=[0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.25, 0.30],
    max_loss_grid=[2.0, 3.0, 5.0, 7.0, 10.0, 15.0, Inf],
    reference_delta::Real=0.16,
    reference_max_loss::Real=5.0,
    store=DEFAULT_STORE,
)
    run_dir = make_run_dir(output_root, "condor_grid")
    println("Output: $run_dir")
    all_rows = CondorGridSummaryRow[]

    function run_symbol(symbol, spot_sym, mult)
        println("\n", "="^70)
        println("  $symbol (spot via $spot_sym x $mult)")
        println("="^70)

        all_dates = available_polygon_dates(store, symbol)
        filtered = filter(d -> start_date <= d <= end_date, all_dates)
        if length(filtered) < 30
            println("  SKIP: only $(length(filtered)) dates")
            return
        end
        println("  Dates: $(length(filtered)) ($(first(filtered)) to $(last(filtered)))")

        entry_ts = build_entry_timestamps(filtered, [entry_time])
        entry_spots = read_polygon_spot_prices_for_timestamps(
            polygon_spot_root(store), entry_ts; symbol=spot_sym)
        if mult != 1.0
            for (k, v) in entry_spots
                entry_spots[k] = v * mult
            end
        end

        source = ParquetDataSource(entry_ts;
            path_for_timestamp=ts -> polygon_options_path(store, Date(ts), symbol),
            read_records=(path; where="") -> read_polygon_option_records(
                path, entry_spots; where=where, min_volume=0, warn=false,
                spread_lambda=spread_lambda),
            spot_root=polygon_spot_root(store),
            spot_symbol=spot_sym,
            spot_multiplier=mult)

        schedule = available_timestamps(source)
        println("  Schedule: $(length(schedule)) timestamps")

        total = length(delta_grid) * length(max_loss_grid)
        idx = 0
        for delta in delta_grid, max_loss in max_loss_grid
            idx += 1
            scaled_max_loss = max_loss * mult
            max_loss_label = isfinite(max_loss) ? @sprintf("%.1f", max_loss) : "Inf"
            @printf("  [%d/%d] d=%.2f ml=%s ... ", idx, total, delta, max_loss_label)

            selector = constrained_delta_selector(delta, delta;
                rate=rate,
                div_yield=div_yield,
                max_loss=scaled_max_loss,
                max_spread_rel=max_spread_rel)

            result = backtest_strategy(
                IronCondorStrategy(schedule, expiry_interval, selector), source)

            pnls, maxlosses, credits = condor_trade_pnls(result)
            pnls ./= mult
            maxlosses ./= mult
            credits ./= mult

            push!(all_rows, condor_grid_summary_row(
                symbol=symbol,
                delta=delta,
                max_loss_cap=max_loss,
                pnls=pnls,
                maxlosses=maxlosses,
                credits=credits,
            ))

            row = all_rows[end]
            @printf("n=%d mean=\$%.3f std=\$%.3f p5=\$%.3f skew=%.2f\n",
                row.trades, row.mean_pnl, row.std_pnl, row.p5_pnl, row.skewness)
        end
    end

    for (symbol, spot_sym, mult) in symbols
        run_symbol(symbol, spot_sym, mult)
    end

    csv_path = write_namedtuple_csv(joinpath(run_dir, "pnl_summary.csv"), all_rows)
    println("\nCSV: $csv_path")

    symbols_present = unique([row.symbol for row in all_rows])
    col_label(ml) = isfinite(ml) ? @sprintf("ml=%.0f", ml) : "ml=Inf"
    row_label(delta) = @sprintf("d=%.2f", delta)

    for sym in symbols_present
        println("\n\n", "#"^80)
        println("#  $sym")
        println("#"^80)
        filter_sym = row -> row.symbol == sym

        for (title, field, digits, pct, signed, value_fn) in [
            ("Mean PnL (\$)", :mean_pnl, 3, false, true, identity),
            ("Std PnL (\$)", :std_pnl, 4, false, false, identity),
            ("Median PnL (\$)", :median_pnl, 3, false, true, identity),
            ("5th Percentile PnL (\$)", :p5_pnl, 3, false, true, identity),
            ("1st Percentile PnL (\$)", :p1_pnl, 3, false, true, identity),
            ("Skewness", :skewness, 2, false, true, identity),
            ("Win Rate", :win_rate, 1, true, false, (x -> 100 * x)),
            ("Total PnL (\$)", :total_pnl, 2, false, true, identity),
            ("Avg Win (\$)", :avg_win, 3, false, true, identity),
            ("Avg Loss (\$)", :avg_loss, 3, false, true, identity),
            ("Mean Credit (\$)", :mean_credit, 4, false, false, identity),
            ("Mean MaxLoss (\$)", :mean_maxloss, 4, false, false, identity),
        ]
            print_metric_grid("$title -- $sym", all_rows, delta_grid, max_loss_grid;
                row_key=:delta,
                col_key=:max_loss_cap,
                value_key=field,
                col_label_fn=col_label,
                row_label_fn=row_label,
                filter_fn=filter_sym,
                digits=digits,
                pct=pct,
                signed=signed,
                value_fn=value_fn)
        end

        print_condor_percentile_ladder(all_rows, sym, delta_grid, reference_max_loss)
    end

    print_condor_cross_symbol_comparison(all_rows, symbols_present, reference_delta, reference_max_loss)
    println("\n\nDone.")
    return (run_dir=run_dir, rows=all_rows)
end
