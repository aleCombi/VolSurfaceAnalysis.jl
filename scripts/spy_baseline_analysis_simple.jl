using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, DataFrames

# =============================================================================
# Configuration
# =============================================================================

START_DATE      = Date(2016, 3, 28)
END_DATE        = Date(2026, 1, 31)
ENTRY_TIME      = Time(10, 0)
EXPIRY_INTERVAL = Day(1)

SPREAD_LAMBDA   = 0.7
RATE            = 0.045
DIV_YIELD       = 0.013
BASE_MAX_LOSS   = 15.0
MAX_SPREAD_REL  = 0.50
PUT_DELTA       = 0.2
CALL_DELTA      = 0.2

# =============================================================================
# PnL distribution printer (from sizing_filter.jl)
# =============================================================================

function print_pnl_distribution(label, result)
    df = condor_trade_table(result.positions, result.pnl)
    pnls = Float64[r.PnL for r in eachrow(df) if !ismissing(r.PnL) && !ismissing(r.MaxLoss) && r.MaxLoss > 0]
    rors = Float64[r.ReturnOnRisk for r in eachrow(df) if !ismissing(r.ReturnOnRisk)]
    n = length(pnls)
    n < 2 && return

    wins = filter(>(0), pnls)
    losses = filter(<=(0), pnls)
    s = std(pnls)
    skew = s > 0 ? mean((pnls .- mean(pnls)).^3) / s^3 : 0.0

    println("\n  ── $label PnL Distribution ($n trades) ──")
    @printf("  Mean=\$%.4f  Std=\$%.4f  Skew=%.2f\n", mean(pnls), s, skew)
    @printf("  Median=\$%.4f  WinRate=%.1f%%  W/L=%.2f\n",
        quantile(pnls, 0.5), length(wins)/n*100,
        isempty(losses) ? Inf : mean(wins)/abs(mean(losses)))
    @printf("  AvgWin=\$%.4f  AvgLoss=\$%.4f\n",
        isempty(wins) ? 0.0 : mean(wins),
        isempty(losses) ? 0.0 : mean(losses))
    @printf("  Percentiles:  p1=\$%.3f  p5=\$%.3f  p10=\$%.3f  p25=\$%.3f  p50=\$%.3f  p75=\$%.3f  p90=\$%.3f  p95=\$%.3f  p99=\$%.3f\n",
        quantile(pnls, 0.01), quantile(pnls, 0.05), quantile(pnls, 0.10),
        quantile(pnls, 0.25), quantile(pnls, 0.50), quantile(pnls, 0.75),
        quantile(pnls, 0.90), quantile(pnls, 0.95), quantile(pnls, 0.99))
    @printf("  TotalPnL=\$%.2f  Min=\$%.4f  Max=\$%.4f\n", sum(pnls), minimum(pnls), maximum(pnls))

    # Histogram
    nbins = 20
    lo, hi = minimum(pnls), maximum(pnls)
    edges = range(lo, hi, length=nbins+1)
    counts = zeros(Int, nbins)
    for v in pnls
        b = clamp(searchsortedlast(collect(edges), v), 1, nbins)
        counts[b] += 1
    end
    maxc = maximum(counts)
    bw = 35
    for i in 1:nbins
        blen = round(Int, counts[i] / maxc * bw)
        bar = repeat("█", blen)
        @printf("    [%+7.3f,%+7.3f) %3d │%s\n", edges[i], edges[i+1], counts[i], bar)
    end

    big_losses = count(r -> r < -0.5, rors)
    @printf("  Big losses (ROI < -50%%): %d / %d (%.1f%%)\n", big_losses, n, big_losses/n*100)
end

# =============================================================================
# Setup
# =============================================================================

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "spy_baseline_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")

store = DEFAULT_STORE

# =============================================================================
# Data source
# =============================================================================

println("\nLoading SPY data from $START_DATE to $END_DATE...")
all_dates = available_polygon_dates(store, "SPY")
filtered = filter(d -> d >= START_DATE && d <= END_DATE, all_dates)
println("  $(length(filtered)) trading days with options data")

entry_ts = build_entry_timestamps(filtered, ENTRY_TIME)
entry_spots = read_polygon_spot_prices_for_timestamps(
    polygon_spot_root(store), entry_ts; symbol="SPY")
println("  $(length(entry_spots)) spot prices loaded")

source = ParquetDataSource(entry_ts;
    path_for_timestamp=ts -> polygon_options_path(store, Date(ts), "SPY"),
    read_records=(path; where="") -> read_polygon_option_records(
        path, entry_spots; where=where, min_volume=0, warn=false,
        spread_lambda=SPREAD_LAMBDA),
    spot_root=polygon_spot_root(store),
    spot_symbol="SPY")

sched = filter(t -> t in Set(build_entry_timestamps(filtered, ENTRY_TIME)),
    available_timestamps(source))
println("  $(length(sched)) entry timestamps")

# =============================================================================
# Strategy & backtest
# =============================================================================

println("\nRunning backtest: 16-delta condor, max_loss=\$$(BASE_MAX_LOSS), daily @ $(ENTRY_TIME)...")

selector = constrained_delta_selector(PUT_DELTA, CALL_DELTA;
    rate=RATE, div_yield=DIV_YIELD, max_loss=BASE_MAX_LOSS,
    max_spread_rel=MAX_SPREAD_REL)

strategy = IronCondorStrategy(sched, EXPIRY_INTERVAL, selector)
result = backtest_strategy(strategy, source)

# =============================================================================
# Overall metrics
# =============================================================================

m = performance_metrics(result)
if m === nothing
    println("ERROR: No valid trades produced.")
    exit(1)
end

println("\n", "=" ^ 70)
println("  SPY Systematic Iron Condor — $(START_DATE) to $(END_DATE)")
println("  16Δ put / 16Δ call, max_loss=\$$(BASE_MAX_LOSS), daily @ $(ENTRY_TIME)")
println("=" ^ 70)

@printf("\n  Trades:        %d\n", m.count)
@printf("  Missing:       %d\n", m.missing)
@printf("  Duration:      %s days (%.1f years)\n",
    ismissing(m.duration_days) ? "?" : string(m.duration_days),
    ismissing(m.duration_years) ? 0.0 : m.duration_years)
@printf("  Total PnL:     \$%.2f\n", m.total_pnl)
@printf("  Avg PnL:       \$%.4f\n", m.avg_pnl)
@printf("  Win Rate:      %s\n", fmt_pct(m.win_rate))
@printf("  Total ROI:     %s\n", fmt_metric(m.total_roi; pct=true))
@printf("  Ann. ROI:      %s (CAGR)\n", fmt_metric(m.annualized_roi_cagr; pct=true))
@printf("  Sharpe:        %s\n", fmt_ratio(m.sharpe))
@printf("  Sortino:       %s\n", fmt_ratio(m.sortino))
@printf("  Avg Spread:    %s (rel)\n",
    ismissing(m.avg_bid_ask_spread_rel) ? "n/a" : @sprintf("%.3f", m.avg_bid_ask_spread_rel))

# =============================================================================
# PnL distribution
# =============================================================================

print_pnl_distribution("Systematic Condor", result)

# =============================================================================
# Trade-level DataFrame for deeper analysis
# =============================================================================

df = condor_trade_table(result.positions, result.pnl)
df = filter(r -> !ismissing(r.PnL) && !ismissing(r.MaxLoss) && r.MaxLoss > 0, df)
df.RoR = [ismissing(r.ReturnOnRisk) ? missing : r.ReturnOnRisk for r in eachrow(df)]
df.Month = [Dates.format(Date(r.EntryTimestamp), "yyyy-mm") for r in eachrow(df)]
df.Year = [year(Date(r.EntryTimestamp)) for r in eachrow(df)]

# =============================================================================
# Tail risk
# =============================================================================

println("\n  ── Worst 20 Trades ──")
worst = sort(df, :PnL)[1:min(20, nrow(df)), :]
@printf("  %-20s  %10s  %10s  %8s\n", "Entry", "PnL", "MaxLoss", "RoR")
@printf("  %-20s  %10s  %10s  %8s\n", "-"^20, "-"^10, "-"^10, "-"^8)
for r in eachrow(worst)
    @printf("  %-20s  %+10.4f  %10.2f  %+7.1f%%\n",
        Dates.format(r.EntryTimestamp, "yyyy-mm-dd HH:MM"),
        r.PnL, r.MaxLoss,
        ismissing(r.ReturnOnRisk) ? 0.0 : r.ReturnOnRisk * 100)
end

# CVaR (expected shortfall at 5%)
sorted_pnl = sort(Float64.(df.PnL))
n_tail = max(1, floor(Int, 0.05 * length(sorted_pnl)))
cvar_5 = mean(sorted_pnl[1:n_tail])
@printf("\n  CVaR (5%%):  \$%.4f  (avg of worst %d trades)\n", cvar_5, n_tail)

# Max drawdown on cumulative PnL
dates_sorted = sort(df, :EntryTimestamp)
cum_pnl = cumsum(Float64.(dates_sorted.PnL))
peak = accumulate(max, cum_pnl)
drawdown = cum_pnl .- peak
max_dd = minimum(drawdown)
max_dd_idx = argmin(drawdown)
@printf("  Max Drawdown:  \$%.2f  (at trade #%d, %s)\n",
    max_dd, max_dd_idx,
    Dates.format(dates_sorted.EntryTimestamp[max_dd_idx], "yyyy-mm-dd"))

# =============================================================================
# Annual breakdown
# =============================================================================

println("\n  ── Annual Breakdown ──")
@printf("  %-6s  %6s  %10s  %8s  %8s  %8s\n",
    "Year", "Trades", "PnL", "AvgPnL", "WinRate", "AvgRoR")
@printf("  %-6s  %6s  %10s  %8s  %8s  %8s\n",
    "-"^6, "-"^6, "-"^10, "-"^8, "-"^8, "-"^8)

for yr in sort(unique(df.Year))
    ydf = filter(r -> r.Year == yr, df)
    nt = nrow(ydf)
    tp = sum(ydf.PnL)
    ap = mean(ydf.PnL)
    wr = count(>(0), ydf.PnL) / nt * 100
    rors = Float64[r for r in ydf.RoR if !ismissing(r)]
    ar = isempty(rors) ? 0.0 : mean(rors) * 100
    @printf("  %-6d  %6d  %+10.2f  %+8.4f  %7.1f%%  %+7.1f%%\n",
        yr, nt, tp, ap, wr, ar)
end

# =============================================================================
# Monthly breakdown (worst & best)
# =============================================================================

monthly = combine(groupby(df, :Month),
    :PnL => sum => :TotalPnL,
    :PnL => length => :Trades,
    :PnL => (x -> count(>(0), x) / length(x) * 100) => :WinRate)
sort!(monthly, :TotalPnL)

println("\n  ── Worst 10 Months ──")
@printf("  %-8s  %6s  %10s  %8s\n", "Month", "Trades", "PnL", "WinRate")
@printf("  %-8s  %6s  %10s  %8s\n", "-"^8, "-"^6, "-"^10, "-"^8)
for r in eachrow(monthly[1:min(10, nrow(monthly)), :])
    @printf("  %-8s  %6d  %+10.2f  %7.1f%%\n", r.Month, r.Trades, r.TotalPnL, r.WinRate)
end

println("\n  ── Best 10 Months ──")
@printf("  %-8s  %6s  %10s  %8s\n", "Month", "Trades", "PnL", "WinRate")
@printf("  %-8s  %6s  %10s  %8s\n", "-"^8, "-"^6, "-"^10, "-"^8)
for r in eachrow(sort(monthly, :TotalPnL, rev=true)[1:min(10, nrow(monthly)), :])
    @printf("  %-8s  %6d  %+10.2f  %7.1f%%\n", r.Month, r.Trades, r.TotalPnL, r.WinRate)
end

# =============================================================================
# Plots
# =============================================================================

println("\n  Saving plots...")

entry_dates = [DateTime(r.EntryTimestamp) for r in eachrow(dates_sorted)]
trade_pnls = Union{Missing,Float64}[r.PnL for r in eachrow(dates_sorted)]

save_pnl_and_equity_curve(entry_dates, trade_pnls,
    joinpath(run_dir, "equity_and_distribution.png");
    title_prefix="SPY 16Δ Condor")
println("    equity_and_distribution.png")

save_profit_curve(entry_dates, trade_pnls,
    joinpath(run_dir, "per_trade_pnl.png");
    title="SPY 16Δ Condor — Per-Trade P&L")
println("    per_trade_pnl.png")

# =============================================================================
# CSV export
# =============================================================================

csv_path = joinpath(run_dir, "trades.csv")
open(csv_path, "w") do io
    println(io, "EntryTimestamp,Expiry,PnL,Credit,MaxLoss,WidthPut,WidthCall,ReturnOnRisk")
    for r in eachrow(df)
        @printf(io, "%s,%s,%.6f,%.6f,%.4f,%.2f,%.2f,%.6f\n",
            r.EntryTimestamp, r.Expiry, r.PnL,
            ismissing(r.Credit) ? "" : r.Credit,
            ismissing(r.MaxLoss) ? "" : r.MaxLoss,
            ismissing(r.WidthPut) ? "" : r.WidthPut,
            ismissing(r.WidthCall) ? "" : r.WidthCall,
            ismissing(r.ReturnOnRisk) ? "" : r.ReturnOnRisk)
    end
end
println("    trades.csv ($(nrow(df)) rows)")

println("\nOutput: $run_dir")
println("Done.")
