# scripts/baseline_analysis.jl
#
# Systematic SPY iron-condor baseline analysis: full pipeline (MODE=full)
# runs the backtest, prints PnL distribution + per-month/per-year tables,
# and additionally trains an expanding-window IntradayLogSig ridge classifier
# on tail-loss labels with several skip thresholds. MODE=simple stops after
# the distribution + monthly/annual tables.
#
# Replaces spy_baseline_analysis.jl and spy_baseline_analysis_simple.jl.
#
# Presets (preserve original behavior):
#   # spy_baseline_analysis.jl (full):
#   MODE=full ENTRY_HOUR=12 END_DATE=2024-01-31 BASE_MAX_LOSS=8
#
#   # spy_baseline_analysis_simple.jl:
#   MODE=simple ENTRY_HOUR=10 END_DATE=2026-01-31 BASE_MAX_LOSS=15

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, DataFrames, Plots

# =============================================================================
# Configuration (override via ENV)
# =============================================================================

MODE            = lowercase(get(ENV, "MODE", "full"))
SYMBOL          = get(ENV, "SYM", "SPY")

START_DATE      = Date(get(ENV, "START_DATE", "2016-03-28"))
END_DATE        = Date(get(ENV, "END_DATE",   MODE == "simple" ? "2026-01-31" : "2024-01-31"))
ENTRY_TIME      = Time(parse(Int, get(ENV, "ENTRY_HOUR", MODE == "simple" ? "10" : "12")), 0)
EXPIRY_INTERVAL = Day(parse(Int, get(ENV, "EXPIRY_DAYS", "1")))

SPREAD_LAMBDA   = parse(Float64, get(ENV, "SPREAD_LAMBDA", "0.7"))
RATE            = parse(Float64, get(ENV, "RATE", "0.045"))
DIV_YIELD       = parse(Float64, get(ENV, "DIV", "0.013"))
BASE_MAX_LOSS   = parse(Float64, get(ENV, "BASE_MAX_LOSS", MODE == "simple" ? "15.0" : "8.0"))
MAX_SPREAD_REL  = parse(Float64, get(ENV, "MAX_SPREAD_REL", "0.50"))
PUT_DELTA       = parse(Float64, get(ENV, "PUT_DELTA", "0.2"))
CALL_DELTA      = parse(Float64, get(ENV, "CALL_DELTA", "0.2"))

# Tail loss label: RoR < this threshold (e.g. -0.5 = lost >50% of max_loss)
TAIL_CUTOFF_ROR = parse(Float64, get(ENV, "TAIL_CUTOFF_ROR", "-0.5"))

# =============================================================================
# PnL distribution printer
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
run_dir = joinpath(@__DIR__, "runs", "baseline_analysis_$(MODE)_$(SYMBOL)_$run_ts")
mkpath(run_dir)
println("Output: $run_dir   MODE=$MODE")

store = DEFAULT_STORE

# =============================================================================
# Data source
# =============================================================================

println("\nLoading $SYMBOL data from $START_DATE to $END_DATE...")
all_dates = available_polygon_dates(store, SYMBOL)
filtered = filter(d -> d >= START_DATE && d <= END_DATE, all_dates)
println("  $(length(filtered)) trading days with options data")

entry_ts = build_entry_timestamps(filtered, ENTRY_TIME)
entry_spots = read_polygon_spot_prices_for_timestamps(
    polygon_spot_root(store), entry_ts; symbol=SYMBOL)
println("  $(length(entry_spots)) spot prices loaded")

source = ParquetDataSource(entry_ts;
    path_for_timestamp=ts -> polygon_options_path(store, Date(ts), SYMBOL),
    read_records=(path; where="") -> read_polygon_option_records(
        path, entry_spots; where=where, min_volume=0, warn=false,
        spread_lambda=SPREAD_LAMBDA),
    spot_root=polygon_spot_root(store),
    spot_symbol=SYMBOL)

sched = filter(t -> t in Set(build_entry_timestamps(filtered, ENTRY_TIME)),
    available_timestamps(source))
println("  $(length(sched)) entry timestamps")

# =============================================================================
# Strategy & backtest
# =============================================================================

println("\nRunning backtest: $(round(Int,100*PUT_DELTA))-delta condor, max_loss=\$$(BASE_MAX_LOSS), daily @ $(ENTRY_TIME)...")

selector = constrained_delta_selector(PUT_DELTA, CALL_DELTA;
    rate=RATE, div_yield=DIV_YIELD, max_loss=BASE_MAX_LOSS,
    max_spread_rel=MAX_SPREAD_REL)

strategy = IronCondorStrategy(sched, EXPIRY_INTERVAL, selector)
result = backtest_strategy(strategy, source)

m = performance_metrics(result)
if m === nothing
    println("ERROR: No valid trades produced.")
    exit(1)
end

println("\n", "=" ^ 70)
println("  $SYMBOL Systematic Iron Condor — $(START_DATE) to $(END_DATE)")
println("  $(round(Int,100*PUT_DELTA))Δ put / $(round(Int,100*CALL_DELTA))Δ call, max_loss=\$$(BASE_MAX_LOSS), daily @ $(ENTRY_TIME)")
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

print_pnl_distribution("Systematic Condor", result)

# =============================================================================
# Trade-level DataFrame for deeper analysis
# =============================================================================

df = condor_trade_table(result.positions, result.pnl)
df = filter(r -> !ismissing(r.PnL) && !ismissing(r.MaxLoss) && r.MaxLoss > 0, df)
df.RoR = [ismissing(r.ReturnOnRisk) ? missing : r.ReturnOnRisk for r in eachrow(df)]
df.Month = [Dates.format(Date(r.EntryTimestamp), "yyyy-mm") for r in eachrow(df)]
df.Year = [year(Date(r.EntryTimestamp)) for r in eachrow(df)]

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

sorted_pnl = sort(Float64.(df.PnL))
n_tail = max(1, floor(Int, 0.05 * length(sorted_pnl)))
cvar_5 = mean(sorted_pnl[1:n_tail])
@printf("\n  CVaR (5%%):  \$%.4f  (avg of worst %d trades)\n", cvar_5, n_tail)

dates_sorted = sort(df, :EntryTimestamp)
cum_pnl = cumsum(Float64.(dates_sorted.PnL))
peak = accumulate(max, cum_pnl)
drawdown = cum_pnl .- peak
max_dd = minimum(drawdown)
max_dd_idx = argmin(drawdown)
@printf("  Max Drawdown:  \$%.2f  (at trade #%d, %s)\n",
    max_dd, max_dd_idx,
    Dates.format(dates_sorted.EntryTimestamp[max_dd_idx], "yyyy-mm-dd"))

# Annual breakdown
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

# Monthly
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

# Plots
println("\n  Saving plots...")
entry_dates = [DateTime(r.EntryTimestamp) for r in eachrow(dates_sorted)]
trade_pnls = Union{Missing,Float64}[r.PnL for r in eachrow(dates_sorted)]

save_pnl_and_equity_curve(entry_dates, trade_pnls,
    joinpath(run_dir, "equity_and_distribution.png");
    title_prefix="$SYMBOL $(round(Int,100*PUT_DELTA))Δ Condor")
println("    equity_and_distribution.png")

save_profit_curve(entry_dates, trade_pnls,
    joinpath(run_dir, "per_trade_pnl.png");
    title="$SYMBOL $(round(Int,100*PUT_DELTA))Δ Condor — Per-Trade P&L")
println("    per_trade_pnl.png")

# CSV export
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

if MODE == "simple"
    println("\nMODE=simple → stopping after distribution + tables.")
    println("\nOutput: $run_dir")
    println("Done.")
    exit(0)
end

# =============================================================================
# MODE=full: Expanding-window ridge classifier (IntradayLogSig features)
# =============================================================================

println("\n", "=" ^ 70)
println("  EXPANDING-WINDOW RIDGE CLASSIFIER (TAIL LABEL)")
println("  Features: IntradayLogSig(depth=3, 3ch) — spot + vol + skew")
println("  Training: $SYMBOL + QQQ + IWM pooled")
println("  Label: tail loss (RoR < $(TAIL_CUTOFF_ROR*100)%)")
println("=" ^ 70)

CLS_FEATURES = Feature[
    IntradayLogSig(; depth=3, channels=3, rate=RATE, div_yield=DIV_YIELD),
]
CLS_THRESHOLDS = [0.1, 0.15, 0.2, 0.3]

TRAIN_SYMBOLS = ["QQQ", "IWM"]
aux_sources = Dict{String, ParquetDataSource}()
aux_scheds = Dict{String, Vector{DateTime}}()
aux_selectors = Dict{String, Any}()

for sym in TRAIN_SYMBOLS
    sym_dates = filter(d -> d >= START_DATE && d <= END_DATE, available_polygon_dates(store, sym))
    if length(sym_dates) < 50
        println("  SKIP $sym: only $(length(sym_dates)) dates")
        continue
    end
    sym_entry_ts = build_entry_timestamps(sym_dates, ENTRY_TIME)
    sym_spots = read_polygon_spot_prices_for_timestamps(
        polygon_spot_root(store), sym_entry_ts; symbol=sym)

    aux_sources[sym] = ParquetDataSource(sym_entry_ts;
        path_for_timestamp=ts -> polygon_options_path(store, Date(ts), sym),
        read_records=(path; where="") -> read_polygon_option_records(
            path, sym_spots; where=where, min_volume=0, warn=false,
            spread_lambda=SPREAD_LAMBDA),
        spot_root=polygon_spot_root(store),
        spot_symbol=sym)

    aux_scheds[sym] = filter(t -> t in Set(sym_entry_ts), available_timestamps(aux_sources[sym]))
    aux_selectors[sym] = constrained_delta_selector(PUT_DELTA, CALL_DELTA;
        rate=RATE, div_yield=DIV_YIELD, max_loss=BASE_MAX_LOSS,
        max_spread_rel=MAX_SPREAD_REL)

    println("  $sym: $(length(aux_scheds[sym])) timestamps loaded")
end

first_full_year = year(START_DATE) + 1
last_test_year = year(END_DATE)
if month(END_DATE) < 6; last_test_year -= 1; end

windows = Tuple{Date,Date,Date,Date}[]
for test_yr in first_full_year:last_test_year
    train_end = Date(test_yr - 1, 12, 31)
    test_start = Date(test_yr, 1, 1)
    test_end = min(Date(test_yr, 12, 31), END_DATE)
    push!(windows, (START_DATE, train_end, test_start, test_end))
end

println("  $(length(windows)) windows:")
for (i, (ts, te, vs, ve)) in enumerate(windows)
    println("    Window $i: Train $ts → $te, Test $vs → $ve")
end

wf_dates = DateTime[]
wf_pnls_baseline = Float64[]
wf_rors_baseline = Float64[]
wf_pnls_cls = Dict{Float64, Vector{Float64}}(t => Float64[] for t in CLS_THRESHOLDS)
wf_rors_cls = Dict{Float64, Vector{Float64}}(t => Float64[] for t in CLS_THRESHOLDS)
wf_dates_cls = Dict{Float64, Vector{DateTime}}(t => DateTime[] for t in CLS_THRESHOLDS)

println("\n  ── Per-Window Results ──")
@printf("  %-8s  %7s  %4s  %10s  %6s  %7s  │", "Window", "Train", "Test", "BL PnL", "BL WR", "BL Shp")
for t in CLS_THRESHOLDS
    @printf("  p<%.2f PnL Trd  WR%%  Shp │", t)
end
println()
@printf("  %-8s  %7s  %4s  %10s  %6s  %7s  │", "-"^8, "-"^7, "-"^4, "-"^10, "-"^6, "-"^7)
for _ in CLS_THRESHOLDS
    @printf("  %8s  %3s  %4s  %4s │", "-"^8, "-"^3, "-"^4, "-"^4)
end
println()

for (i, (train_start, train_end, test_start, test_end)) in enumerate(windows)
    test_yr = year(test_start)

    train_sched = filter(t -> Date(t) >= train_start && Date(t) <= train_end, sched)
    test_sched = filter(t -> Date(t) >= test_start && Date(t) <= test_end, sched)

    if length(train_sched) < 20 || length(test_sched) < 10
        @printf("  %-8d  %5d  %5d  SKIP (too few timestamps)\n", test_yr, length(train_sched), length(test_sched))
        continue
    end

    bl_df = filter(r -> Date(r.EntryTimestamp) >= test_start && Date(r.EntryTimestamp) <= test_end, dates_sorted)
    bl_pnl = isempty(bl_df) ? 0.0 : sum(bl_df.PnL)
    bl_trades = nrow(bl_df)
    bl_wr = bl_trades > 0 ? count(>(0), bl_df.PnL) / bl_trades * 100 : 0.0

    bl_rors = Float64[]
    if bl_trades > 0
        bl_full = filter(r -> Date(r.EntryTimestamp) >= test_start && Date(r.EntryTimestamp) <= test_end, df)
        bl_rors = Float64[r for r in bl_full.RoR if !ismissing(r)]
    end
    bl_sharpe = length(bl_rors) > 1 && std(bl_rors) > 0 ? mean(bl_rors) / std(bl_rors) * sqrt(252) : 0.0

    bl_full = filter(r -> Date(r.EntryTimestamp) >= test_start && Date(r.EntryTimestamp) <= test_end, df)
    for r in eachrow(bl_full)
        push!(wf_dates, r.EntryTimestamp)
        push!(wf_pnls_baseline, r.PnL)
        push!(wf_rors_baseline, ismissing(r.RoR) ? 0.0 : r.RoR)
    end

    all_examples = VolSurfaceAnalysis.SizingTrainingExample[]

    spy_examples = generate_sizing_training_data(source, EXPIRY_INTERVAL,
        train_sched, selector; rate=RATE, div_yield=DIV_YIELD, surface_features=CLS_FEATURES)
    append!(all_examples, spy_examples)

    for sym in TRAIN_SYMBOLS
        haskey(aux_sources, sym) || continue
        sym_train = filter(t -> Date(t) >= train_start && Date(t) <= train_end, aux_scheds[sym])
        isempty(sym_train) && continue
        sym_ex = generate_sizing_training_data(aux_sources[sym], EXPIRY_INTERVAL,
            sym_train, aux_selectors[sym]; rate=RATE, div_yield=DIV_YIELD, surface_features=CLS_FEATURES)
        append!(all_examples, sym_ex)
    end

    if length(all_examples) < 20
        @printf("  %-8d  %5d  %5d  SKIP (only %d training examples)\n",
            test_yr, length(train_sched), length(test_sched), length(all_examples))
        continue
    end

    X = hcat([e.surface_features for e in all_examples]...)
    pnl_vec = Float32[e.pnl for e in all_examples]

    tail_pnl_cutoff = quantile(pnl_vec, 0.10)
    Y_cls = reshape(Float32[p < tail_pnl_cutoff ? 1.0f0 : 0.0f0 for p in pnl_vec], 1, :)
    n_spy = length(spy_examples)
    n_aux = length(all_examples) - n_spy
    n_tail = count(==(1.0f0), Y_cls)

    model, means, stds, _ = train_glmnet_classifier!(nothing, X, Y_cls; alpha=0.0)

    @printf("  %-8d  %3d+%3d %4d  %+10.2f  %5.1f%%  %+6.2f  │",
        test_yr, n_spy, n_aux, length(test_sched), bl_pnl, bl_wr, bl_sharpe)

    for t in CLS_THRESHOLDS
        skip_thresh = t
        policy = function(logit::Float64)
            p_loss = 1.0 / (1.0 + exp(-logit))
            return p_loss < skip_thresh ? 1.0 : 0.0
        end

        cls_strategy = IronCondorStrategy(test_sched, EXPIRY_INTERVAL, selector;
            sizer=MLSizer(model, means, stds; surface_features=CLS_FEATURES, policy=policy))
        cls_result = backtest_strategy(cls_strategy, source)
        cm = performance_metrics(cls_result)

        if cm === nothing
            @printf("  %8s  %3d  %4s  %4s │", "n/a", 0, "n/a", "n/a")
        else
            @printf("  %+8.2f  %3d  %4.1f  %+4.1f │",
                cm.total_pnl, cm.count, ismissing(cm.win_rate) ? 0.0 : cm.win_rate * 100,
                ismissing(cm.sharpe) ? 0.0 : cm.sharpe)

            cls_df = condor_trade_table(cls_result.positions, cls_result.pnl)
            cls_df = filter(r -> !ismissing(r.PnL) && !ismissing(r.MaxLoss) && r.MaxLoss > 0, cls_df)
            for r in eachrow(sort(cls_df, :EntryTimestamp))
                push!(wf_dates_cls[t], r.EntryTimestamp)
                push!(wf_pnls_cls[t], r.PnL)
                push!(wf_rors_cls[t], ismissing(r.ReturnOnRisk) ? 0.0 : r.ReturnOnRisk)
            end
        end
    end
    println()
end

# Walk-forward aggregate
println("\n  ── Walk-Forward Aggregate ──")
println("  (concatenated out-of-sample test periods)")

if !isempty(wf_pnls_baseline)
    bl_n = length(wf_pnls_baseline)
    bl_total = sum(wf_pnls_baseline)
    bl_wr = count(>(0), wf_pnls_baseline) / bl_n * 100
    bl_avg = mean(wf_pnls_baseline)
    bl_tail = count(r -> r < -0.5, wf_rors_baseline)
    bl_tail_pnl = sum(wf_pnls_baseline[i] for i in 1:bl_n if wf_rors_baseline[i] < -0.5; init=0.0)
    n5 = max(1, floor(Int, 0.05 * bl_n))
    bl_cvar = mean(sort(wf_pnls_baseline)[1:n5])
    @printf("\n  Baseline:    %d trades  PnL=%+.2f  WR=%.1f%%  Avg=%+.4f\n",
        bl_n, bl_total, bl_wr, bl_avg)
    @printf("               TailLosses(RoR<-50%%): %d (%.1f%%)  TailPnL=%+.2f  CVaR5%%=\$%.2f\n",
        bl_tail, bl_tail/bl_n*100, bl_tail_pnl, bl_cvar)
end

for t in CLS_THRESHOLDS
    pnls = wf_pnls_cls[t]
    rors = wf_rors_cls[t]
    isempty(pnls) && continue
    n = length(pnls)
    total = sum(pnls)
    wr = count(>(0), pnls) / n * 100
    avg = mean(pnls)
    skipped = length(wf_pnls_baseline) - n
    tail = count(r -> r < -0.5, rors)
    tail_pnl = sum(pnls[i] for i in 1:n if rors[i] < -0.5; init=0.0)
    n5 = max(1, floor(Int, 0.05 * n))
    cvar = mean(sort(pnls)[1:n5])
    @printf("  Cls p<%.2f:  %d trades  PnL=%+.2f  WR=%.1f%%  Avg=%+.4f  (skipped %d)\n",
        t, n, total, wr, avg, skipped)
    @printf("               TailLosses(RoR<-50%%): %d (%.1f%%)  TailPnL=%+.2f  CVaR5%%=\$%.2f\n",
        tail, tail/n*100, tail_pnl, cvar)
end

if !isempty(wf_pnls_baseline)
    best_t = CLS_THRESHOLDS[1]
    cls_dates_set = Set(wf_dates_cls[best_t])
    skipped_idx = [i for i in 1:length(wf_dates) if wf_dates[i] ∉ cls_dates_set]
    if !isempty(skipped_idx)
        sk_pnls = wf_pnls_baseline[skipped_idx]
        sk_rors = wf_rors_baseline[skipped_idx]
        sk_wins = count(>(0), sk_pnls)
        sk_losses = count(<=(0), sk_pnls)
        sk_tail = count(r -> r < -0.5, sk_rors)
        println("\n  ── Skipped Trade Analysis (p<$(best_t)) ──")
        @printf("  Skipped %d trades: %d wins, %d losses, %d tail losses (RoR<-50%%)\n",
            length(sk_pnls), sk_wins, sk_losses, sk_tail)
        @printf("  Skipped PnL: total=%+.2f  avg=%+.4f\n", sum(sk_pnls), mean(sk_pnls))
        @printf("  Skipped had %s avg PnL → filter was %s\n",
            mean(sk_pnls) < 0 ? "NEGATIVE" : "POSITIVE",
            mean(sk_pnls) < 0 ? "HELPFUL" : "HARMFUL")
        if sk_tail > 0
            @printf("  Tail losses avoided: %d / %d baseline tail losses (%.0f%%)\n",
                sk_tail, count(r -> r < -0.5, wf_rors_baseline),
                sk_tail / count(r -> r < -0.5, wf_rors_baseline) * 100)
        end
    end
end

println("\n  Saving walk-forward plots...")
if !isempty(wf_dates)
    order = sortperm(wf_dates)
    sorted_wf_dates = wf_dates[order]
    sorted_wf_pnls = wf_pnls_baseline[order]
    cum_bl = cumsum(sorted_wf_pnls)

    p = plot(Date.(sorted_wf_dates), cum_bl,
        label="Baseline", title="Walk-Forward Cumulative PnL (OOS)",
        xlabel="Date", ylabel="Cumulative PnL (USD)",
        linewidth=2, color=:gray, legend=:topleft, size=(900, 500))

    colors = [:blue, :orange, :green]
    for (i, t) in enumerate(CLS_THRESHOLDS)
        dates_t = wf_dates_cls[t]
        pnls_t = wf_pnls_cls[t]
        isempty(pnls_t) && continue
        order_t = sortperm(dates_t)
        cum_t = cumsum(pnls_t[order_t])
        plot!(p, Date.(dates_t[order_t]), cum_t,
            label="Cls p<$(t)", linewidth=2, color=colors[i])
    end

    hline!(p, [0], color=:black, linestyle=:dash, label="", linewidth=1)
    savefig(p, joinpath(run_dir, "walkforward_equity.png"))
    println("    walkforward_equity.png")
end

println("\nOutput: $run_dir")
println("Done.")
