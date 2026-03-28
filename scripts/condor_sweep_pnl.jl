using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, DataFrames

# =============================================================================
# Configuration
# =============================================================================

SYMBOLS = [
    ("SPY",  "SPY",  1.0),
    ("QQQ",  "QQQ",  1.0),
    ("IWM",  "IWM",  1.0),
    ("SPXW", "SPY", 10.0),
]

SPREAD_LAMBDA = 0.7
ENTRY_TIME    = Time(10, 0)
EXPIRY_INTERVAL = Day(1)
RATE          = 0.045
DIV_YIELD     = 0.013
MAX_SPREAD_REL = 0.50
START_DATE    = Date(2024, 1, 1)
END_DATE      = Date(2025, 12, 31)

DELTA_GRID    = [0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.25, 0.30]
# Base max loss grid (scaled by mult per symbol)
MAX_LOSS_GRID = [2.0, 3.0, 5.0, 7.0, 10.0, 15.0, Inf]

# =============================================================================
# Setup
# =============================================================================

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "condor_sweep_pnl_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")

store = DEFAULT_STORE

# =============================================================================
# Per-trade PnL extraction
# =============================================================================

function trade_pnls(result)
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

# =============================================================================
# Result type
# =============================================================================

const SummaryRow = @NamedTuple begin
    symbol::String
    delta::Float64
    max_loss_cap::Float64   # base (before mult scaling)
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

all_rows = SummaryRow[]

# =============================================================================
# Per-symbol sweep
# =============================================================================

function run_symbol(symbol, spot_sym, mult)
    println("\n", "=" ^ 70)
    println("  $symbol (spot via $spot_sym × $mult)")
    println("=" ^ 70)

    all_dates = available_polygon_dates(store, symbol)
    filtered = filter(d -> d >= START_DATE && d <= END_DATE, all_dates)
    if length(filtered) < 30
        println("  SKIP: only $(length(filtered)) dates")
        return
    end
    println("  Dates: $(length(filtered)) ($(first(filtered)) to $(last(filtered)))")

    entry_ts = build_entry_timestamps(filtered, [ENTRY_TIME])
    entry_spots = read_polygon_spot_prices_for_timestamps(
        polygon_spot_root(store), entry_ts; symbol=spot_sym)
    if mult != 1.0
        for (k, v) in entry_spots; entry_spots[k] = v * mult; end
    end

    source = ParquetDataSource(entry_ts;
        path_for_timestamp=ts -> polygon_options_path(store, Date(ts), symbol),
        read_records=(path; where="") -> read_polygon_option_records(
            path, entry_spots; where=where, min_volume=0, warn=false,
            spread_lambda=SPREAD_LAMBDA),
        spot_root=polygon_spot_root(store),
        spot_symbol=spot_sym,
        spot_multiplier=mult)

    schedule = available_timestamps(source)
    println("  Schedule: $(length(schedule)) timestamps")

    total = length(DELTA_GRID) * length(MAX_LOSS_GRID)
    idx = 0

    for delta in DELTA_GRID
        for ml in MAX_LOSS_GRID
            idx += 1
            scaled_ml = ml * mult
            ml_label = isfinite(ml) ? @sprintf("%.1f", ml) : "Inf"
            @printf("  [%d/%d] d=%.2f ml=%s ... ", idx, total, delta, ml_label)

            sel = constrained_delta_selector(delta, delta;
                rate=RATE, div_yield=DIV_YIELD, max_loss=scaled_ml,
                max_spread_rel=MAX_SPREAD_REL)

            result = backtest_strategy(
                IronCondorStrategy(schedule, EXPIRY_INTERVAL, sel), source)

            pnls, maxlosses, credits = trade_pnls(result)

            # Normalize PnL back to base units (divide by mult) for cross-symbol comparison
            pnls ./= mult
            maxlosses ./= mult
            credits ./= mult

            n = length(pnls)

            if n >= 2
                wins = filter(>(0), pnls)
                losses = filter(<=(0), pnls)
                wr = length(wins) / n
                s = std(pnls)
                m3 = s > 0 ? mean((pnls .- mean(pnls)).^3) / s^3 : 0.0

                push!(all_rows, (
                    symbol=symbol, delta=delta, max_loss_cap=ml, trades=n, win_rate=wr,
                    mean_pnl=mean(pnls), std_pnl=s,
                    min_pnl=minimum(pnls),
                    p1_pnl=quantile(pnls, 0.01),
                    p5_pnl=quantile(pnls, 0.05),
                    p10_pnl=quantile(pnls, 0.10),
                    p25_pnl=quantile(pnls, 0.25),
                    median_pnl=quantile(pnls, 0.50),
                    p75_pnl=quantile(pnls, 0.75),
                    p90_pnl=quantile(pnls, 0.90),
                    p95_pnl=quantile(pnls, 0.95),
                    p99_pnl=quantile(pnls, 0.99),
                    max_pnl=maximum(pnls),
                    total_pnl=sum(pnls),
                    mean_maxloss=mean(maxlosses),
                    mean_credit=isempty(credits) ? 0.0 : mean(credits),
                    avg_win=isempty(wins) ? 0.0 : mean(wins),
                    avg_loss=isempty(losses) ? 0.0 : mean(losses),
                    skewness=m3))
            else
                push!(all_rows, (
                    symbol=symbol, delta=delta, max_loss_cap=ml, trades=n, win_rate=0.0,
                    mean_pnl=0.0, std_pnl=0.0, min_pnl=0.0,
                    p1_pnl=0.0, p5_pnl=0.0, p10_pnl=0.0, p25_pnl=0.0,
                    median_pnl=0.0, p75_pnl=0.0, p90_pnl=0.0, p95_pnl=0.0,
                    p99_pnl=0.0, max_pnl=0.0, total_pnl=0.0,
                    mean_maxloss=0.0, mean_credit=0.0,
                    avg_win=0.0, avg_loss=0.0, skewness=0.0))
            end

            r = all_rows[end]
            @printf("n=%d mean=\$%.3f std=\$%.3f p5=\$%.3f skew=%.2f\n",
                r.trades, r.mean_pnl, r.std_pnl, r.p5_pnl, r.skewness)
        end
    end
end

for (symbol, spot_sym, mult) in SYMBOLS
    run_symbol(symbol, spot_sym, mult)
end

# =============================================================================
# CSV output
# =============================================================================

csv_path = joinpath(run_dir, "pnl_summary.csv")
open(csv_path, "w") do io
    println(io, "symbol,delta,max_loss_cap,trades,win_rate," *
                "mean_pnl,std_pnl,min_pnl,p1_pnl,p5_pnl,p10_pnl,p25_pnl," *
                "median_pnl,p75_pnl,p90_pnl,p95_pnl,p99_pnl,max_pnl,total_pnl," *
                "mean_maxloss,mean_credit,avg_win,avg_loss,skewness")
    for r in all_rows
        ml_str = isfinite(r.max_loss_cap) ? @sprintf("%.1f", r.max_loss_cap) : "Inf"
        @printf(io, "%s,%.2f,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
            r.symbol, r.delta, ml_str, r.trades, r.win_rate,
            r.mean_pnl, r.std_pnl, r.min_pnl, r.p1_pnl, r.p5_pnl, r.p10_pnl, r.p25_pnl,
            r.median_pnl, r.p75_pnl, r.p90_pnl, r.p95_pnl, r.p99_pnl, r.max_pnl, r.total_pnl,
            r.mean_maxloss, r.mean_credit, r.avg_win, r.avg_loss, r.skewness)
    end
end
println("\nCSV: $csv_path")

# =============================================================================
# Console tables — per symbol
# =============================================================================

symbols_present = unique([r.symbol for r in all_rows])

function _lookup(rows, sym, d, ml)
    r = filter(r -> r.symbol == sym && r.delta == d && r.max_loss_cap == ml, rows)
    isempty(r) ? nothing : r[1]
end

function print_table(title, sym, field::Symbol; fmt=Printf.Format("  %+8.4f"))
    println("\n  $title — $sym:")
    @printf("  %8s", "")
    for ml in MAX_LOSS_GRID; @printf("  %8s", isfinite(ml) ? @sprintf("ml=%.0f", ml) : "ml=Inf"); end
    println()
    for d in DELTA_GRID
        @printf("  d=%.2f ", d)
        for ml in MAX_LOSS_GRID
            r = _lookup(all_rows, sym, d, ml)
            if r === nothing
                @printf("  %8s", "-")
            else
                Printf.format(stdout, fmt, getfield(r, field))
            end
        end
        println()
    end
end

function print_pct_table(title, sym, field::Symbol)
    println("\n  $title — $sym:")
    @printf("  %8s", "")
    for ml in MAX_LOSS_GRID; @printf("  %8s", isfinite(ml) ? @sprintf("ml=%.0f", ml) : "ml=Inf"); end
    println()
    for d in DELTA_GRID
        @printf("  d=%.2f ", d)
        for ml in MAX_LOSS_GRID
            r = _lookup(all_rows, sym, d, ml)
            r === nothing ? @printf("  %8s", "-") : @printf("  %7.1f%%", getfield(r, field) * 100)
        end
        println()
    end
end

for sym in symbols_present
    println("\n\n", "#" ^ 80)
    println("#  $sym")
    println("#" ^ 80)

    print_table("Mean PnL (\$)", sym, :mean_pnl)
    print_table("Std PnL (\$)", sym, :std_pnl; fmt=Printf.Format("  %8.4f"))
    print_table("Median PnL (\$)", sym, :median_pnl)
    print_table("5th Percentile PnL (\$)", sym, :p5_pnl)
    print_table("1st Percentile PnL (\$)", sym, :p1_pnl)
    print_table("Skewness", sym, :skewness; fmt=Printf.Format("  %+8.2f"))
    print_pct_table("Win Rate", sym, :win_rate)
    print_table("Total PnL (\$)", sym, :total_pnl; fmt=Printf.Format("  %+8.2f"))
    print_table("Avg Win (\$)", sym, :avg_win)
    print_table("Avg Loss (\$)", sym, :avg_loss)
    print_table("Mean Credit (\$)", sym, :mean_credit; fmt=Printf.Format("  %8.4f"))
    print_table("Mean MaxLoss (\$)", sym, :mean_maxloss; fmt=Printf.Format("  %8.4f"))

    # Percentile ladder at ml=5
    println("\n  Full PnL percentile ladder at max_loss=5 — $sym:")
    @printf("  %8s  %5s  %7s  %7s  %7s  %7s  %7s  %7s  %7s  %7s  %7s  %7s  %7s  %6s  %6s\n",
        "delta", "n", "min", "p1", "p5", "p10", "p25", "med", "p75", "p90", "p95", "p99", "max", "skew", "w/l")
    println("  ", "-" ^ 115)
    for d in DELTA_GRID
        r = _lookup(all_rows, sym, d, 5.0)
        r === nothing && continue
        ratio = r.avg_loss != 0.0 ? r.avg_win / abs(r.avg_loss) : 0.0
        @printf("  d=%.2f   %4d  %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f  %+5.2f  %5.2f\n",
            d, r.trades,
            r.min_pnl, r.p1_pnl, r.p5_pnl, r.p10_pnl, r.p25_pnl,
            r.median_pnl, r.p75_pnl, r.p90_pnl, r.p95_pnl, r.p99_pnl, r.max_pnl,
            r.skewness, ratio)
    end
end

# =============================================================================
# Cross-symbol comparison at reference config (d=0.16, ml=5)
# =============================================================================

println("\n\n", "=" ^ 100)
println("  Cross-symbol comparison at d=0.16 ml=5 (PnL normalized to SPY-equivalent \$)")
println("=" ^ 100)
@printf("  %-6s  %5s  %7s  %7s  %7s  %7s  %7s  %7s  %6s  %7s  %6s\n",
    "sym", "n", "mean", "std", "median", "p5", "p1", "total", "skew", "w_rate", "w/l")
println("  ", "-" ^ 90)
for sym in symbols_present
    r = _lookup(all_rows, sym, 0.16, 5.0)
    r === nothing && continue
    ratio = r.avg_loss != 0.0 ? r.avg_win / abs(r.avg_loss) : 0.0
    @printf("  %-6s  %4d  %+7.4f  %7.4f  %+7.4f  %+7.3f  %+7.3f  %+7.2f  %+5.2f  %6.1f%%  %5.2f\n",
        sym, r.trades, r.mean_pnl, r.std_pnl, r.median_pnl,
        r.p5_pnl, r.p1_pnl, r.total_pnl, r.skewness, r.win_rate*100, ratio)
end

println("\n\nDone.")
