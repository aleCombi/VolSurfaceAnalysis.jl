using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, DataFrames, Plots, StatsPlots

# =============================================================================
# Configuration
# =============================================================================

SYMBOL        = "SPY"
SPOT_SYM      = "SPY"
MULT          = 1.0
SPREAD_LAMBDA = 0.7
ENTRY_TIME    = Time(10, 0)
EXPIRY_INTERVAL = Day(1)
RATE          = 0.045
DIV_YIELD     = 0.013
MAX_SPREAD_REL = 0.50
START_DATE    = Date(2024, 1, 1)
END_DATE      = Date(2025, 12, 31)

DELTA_GRID    = [0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.25, 0.30]
MAX_LOSS_GRID = [2.0, 3.0, 5.0, 7.0, 10.0, 15.0, Inf]

# =============================================================================
# Setup
# =============================================================================

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "condor_sweep_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")

store = DEFAULT_STORE

# Build data source (once, reused for all grid points)
all_dates = available_polygon_dates(store, SYMBOL)
filtered = filter(d -> d >= START_DATE && d <= END_DATE, all_dates)
println("Available dates: $(length(filtered)) ($(first(filtered)) to $(last(filtered)))")

entry_ts = build_entry_timestamps(filtered, [ENTRY_TIME])
entry_spots = read_polygon_spot_prices_for_timestamps(
    polygon_spot_root(store), entry_ts; symbol=SPOT_SYM)

source = ParquetDataSource(entry_ts;
    path_for_timestamp=ts -> polygon_options_path(store, Date(ts), SYMBOL),
    read_records=(path; where="") -> read_polygon_option_records(
        path, entry_spots; where=where, min_volume=0, warn=false,
        spread_lambda=SPREAD_LAMBDA),
    spot_root=polygon_spot_root(store),
    spot_symbol=SPOT_SYM,
    spot_multiplier=MULT)

schedule = available_timestamps(source)
println("Schedule: $(length(schedule)) timestamps\n")

# =============================================================================
# Monthly ROI helper
# =============================================================================

function monthly_rois(result)
    isempty(result.positions) && return Dict{Tuple{Int,Int}, NamedTuple}()

    df = condor_trade_table(result.positions, result.pnl)
    isempty(df) && return Dict{Tuple{Int,Int}, NamedTuple}()

    months = Dict{Tuple{Int,Int}, NamedTuple}()
    for row in eachrow(df)
        ismissing(row.PnL) && continue
        ismissing(row.MaxLoss) && continue
        row.MaxLoss <= 0 && continue

        ym = (year(row.EntryTimestamp), month(row.EntryTimestamp))
        prev = get(months, ym, (pnl=0.0, margin=0.0, trades=0))
        months[ym] = (pnl=prev.pnl + row.PnL, margin=prev.margin + row.MaxLoss,
                      trades=prev.trades + 1)
    end

    # Convert to ROI
    return Dict(ym => (roi=v.pnl / v.margin, trades=v.trades, pnl=v.pnl, margin=v.margin)
                for (ym, v) in months if v.margin > 0)
end

# =============================================================================
# Grid sweep
# =============================================================================

# Collect results
const SweepRow = @NamedTuple begin
    delta::Float64
    max_loss::Float64   # Inf for unconstrained
    year_month::String
    trades::Int
    pnl::Float64
    margin::Float64
    roi::Float64
end

const SummaryRow = @NamedTuple begin
    delta::Float64
    max_loss::Float64
    total_trades::Int
    total_roi::Float64
    sharpe::Float64
    sortino::Float64
    win_rate::Float64
    mean_monthly_roi::Float64
    std_monthly_roi::Float64
    min_monthly_roi::Float64
    p5_monthly_roi::Float64
    p25_monthly_roi::Float64
    median_monthly_roi::Float64
    p75_monthly_roi::Float64
    p95_monthly_roi::Float64
    max_monthly_roi::Float64
end

detail_rows = SweepRow[]
summary_rows = SummaryRow[]

total = length(DELTA_GRID) * length(MAX_LOSS_GRID)
idx = 0

for delta in DELTA_GRID
    for ml in MAX_LOSS_GRID
        global idx += 1
        ml_label = isfinite(ml) ? @sprintf("%.1f", ml) : "Inf"
        @printf("[%d/%d] delta=%.2f max_loss=%s ... ", idx, total, delta, ml_label)

        sel = constrained_delta_selector(delta, delta;
            rate=RATE, div_yield=DIV_YIELD, max_loss=ml * MULT,
            max_spread_rel=MAX_SPREAD_REL)

        result = backtest_strategy(
            IronCondorStrategy(schedule, EXPIRY_INTERVAL, sel), source)

        # Overall metrics
        pm = performance_metrics(result)

        # Monthly breakdown
        mr = monthly_rois(result)
        for (ym, v) in sort(collect(mr))
            push!(detail_rows, (delta=delta, max_loss=ml,
                year_month=@sprintf("%04d-%02d", ym[1], ym[2]),
                trades=v.trades, pnl=v.pnl, margin=v.margin, roi=v.roi))
        end

        # Summary stats
        rois = [v.roi for v in values(mr)]
        n_trades = pm === nothing ? 0 : pm.count
        tot_roi = pm === nothing ? 0.0 : pm.total_roi
        sharpe = pm === nothing ? 0.0 : pm.sharpe
        sortino = pm === nothing ? 0.0 : pm.sortino
        wr = pm === nothing ? 0.0 : pm.win_rate

        if length(rois) >= 2
            sr = sort(rois)
            push!(summary_rows, (
                delta=delta, max_loss=ml,
                total_trades=n_trades, total_roi=tot_roi,
                sharpe=sharpe, sortino=sortino, win_rate=wr,
                mean_monthly_roi=mean(rois), std_monthly_roi=std(rois),
                min_monthly_roi=minimum(rois),
                p5_monthly_roi=quantile(sr, 0.05),
                p25_monthly_roi=quantile(sr, 0.25),
                median_monthly_roi=quantile(sr, 0.50),
                p75_monthly_roi=quantile(sr, 0.75),
                p95_monthly_roi=quantile(sr, 0.95),
                max_monthly_roi=maximum(rois)))
        elseif length(rois) == 1
            v = rois[1]
            push!(summary_rows, (
                delta=delta, max_loss=ml,
                total_trades=n_trades, total_roi=tot_roi,
                sharpe=sharpe, sortino=sortino, win_rate=wr,
                mean_monthly_roi=v, std_monthly_roi=0.0,
                min_monthly_roi=v, p5_monthly_roi=v, p25_monthly_roi=v,
                median_monthly_roi=v, p75_monthly_roi=v, p95_monthly_roi=v,
                max_monthly_roi=v))
        else
            push!(summary_rows, (
                delta=delta, max_loss=ml,
                total_trades=0, total_roi=0.0,
                sharpe=0.0, sortino=0.0, win_rate=0.0,
                mean_monthly_roi=0.0, std_monthly_roi=0.0,
                min_monthly_roi=0.0, p5_monthly_roi=0.0, p25_monthly_roi=0.0,
                median_monthly_roi=0.0, p75_monthly_roi=0.0, p95_monthly_roi=0.0,
                max_monthly_roi=0.0))
        end

        @printf("trades=%d roi=%.1f%% sharpe=%.2f monthly=%.1f%%±%.1f%%\n",
            n_trades, tot_roi * 100, sharpe,
            summary_rows[end].mean_monthly_roi * 100,
            summary_rows[end].std_monthly_roi * 100)
    end
end

# =============================================================================
# CSV output
# =============================================================================

detail_path = joinpath(run_dir, "monthly_detail.csv")
open(detail_path, "w") do io
    println(io, "delta,max_loss,year_month,trades,pnl,margin,roi")
    for r in sort(detail_rows; by=r -> (r.delta, r.max_loss, r.year_month))
        ml_str = isfinite(r.max_loss) ? @sprintf("%.1f", r.max_loss) : "Inf"
        @printf(io, "%.2f,%s,%s,%d,%.4f,%.4f,%.4f\n",
            r.delta, ml_str, r.year_month, r.trades, r.pnl, r.margin, r.roi)
    end
end
println("\nDetail CSV: $detail_path")

summary_path = joinpath(run_dir, "summary.csv")
open(summary_path, "w") do io
    println(io, "delta,max_loss,total_trades,total_roi,sharpe,sortino,win_rate," *
                "mean_monthly_roi,std_monthly_roi,min_monthly_roi," *
                "p5_monthly_roi,p25_monthly_roi,median_monthly_roi," *
                "p75_monthly_roi,p95_monthly_roi,max_monthly_roi")
    for r in summary_rows
        ml_str = isfinite(r.max_loss) ? @sprintf("%.1f", r.max_loss) : "Inf"
        @printf(io, "%.2f,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
            r.delta, ml_str, r.total_trades, r.total_roi, r.sharpe, r.sortino, r.win_rate,
            r.mean_monthly_roi, r.std_monthly_roi, r.min_monthly_roi,
            r.p5_monthly_roi, r.p25_monthly_roi, r.median_monthly_roi,
            r.p75_monthly_roi, r.p95_monthly_roi, r.max_monthly_roi)
    end
end
println("Summary CSV: $summary_path")

# =============================================================================
# Console summary table
# =============================================================================

println("\n", "=" ^ 120)
println("  Summary: mean monthly ROI ± std  |  [p5, median, p95]  |  trades  |  total ROI  |  sharpe")
println("=" ^ 120)

for delta in DELTA_GRID
    @printf("\n  delta = %.2f\n", delta)
    println("  ", "-" ^ 110)
    for r in filter(r -> r.delta == delta, summary_rows)
        ml_str = isfinite(r.max_loss) ? @sprintf("%5.1f", r.max_loss) : "  Inf"
        @printf("    ml=%s  │ %6.1f%% ± %5.1f%%  │ [%6.1f%%, %6.1f%%, %6.1f%%]  │ %4d trades  │ tot=%6.1f%%  │ sharpe=%5.2f\n",
            ml_str,
            r.mean_monthly_roi * 100, r.std_monthly_roi * 100,
            r.p5_monthly_roi * 100, r.median_monthly_roi * 100, r.p95_monthly_roi * 100,
            r.total_trades, r.total_roi * 100, r.sharpe)
    end
end
println("\n", "=" ^ 120)

# =============================================================================
# Plots
# =============================================================================

println("\nGenerating plots...")

# 1. Heatmap: mean monthly ROI by (delta, max_loss)
let
    deltas = DELTA_GRID
    mls = filter(isfinite, MAX_LOSS_GRID)  # skip Inf for heatmap
    ml_labels = [@sprintf("%.0f", m) for m in mls]
    delta_labels = [@sprintf("%.2f", d) for d in deltas]

    Z = zeros(length(deltas), length(mls))
    for (i, d) in enumerate(deltas)
        for (j, m) in enumerate(mls)
            row = filter(r -> r.delta == d && r.max_loss == m, summary_rows)
            Z[i, j] = isempty(row) ? NaN : row[1].mean_monthly_roi * 100
        end
    end

    p = heatmap(ml_labels, delta_labels, Z;
        title="Mean Monthly ROI (%) by Delta × Max Loss",
        xlabel="Max Loss (USD)", ylabel="Delta",
        color=:RdYlGn, clims=(-5, maximum(filter(!isnan, Z)) * 1.1),
        size=(700, 500))
    savefig(p, joinpath(run_dir, "heatmap_mean_roi.png"))
end

# 2. Heatmap: Sharpe
let
    deltas = DELTA_GRID
    mls = filter(isfinite, MAX_LOSS_GRID)
    ml_labels = [@sprintf("%.0f", m) for m in mls]
    delta_labels = [@sprintf("%.2f", d) for d in deltas]

    Z = zeros(length(deltas), length(mls))
    for (i, d) in enumerate(deltas)
        for (j, m) in enumerate(mls)
            row = filter(r -> r.delta == d && r.max_loss == m, summary_rows)
            Z[i, j] = isempty(row) ? NaN : row[1].sharpe
        end
    end

    p = heatmap(ml_labels, delta_labels, Z;
        title="Sharpe by Delta × Max Loss",
        xlabel="Max Loss (USD)", ylabel="Delta",
        color=:RdYlGn, size=(700, 500))
    savefig(p, joinpath(run_dir, "heatmap_sharpe.png"))
end

# 3. Heatmap: worst month (p5)
let
    deltas = DELTA_GRID
    mls = filter(isfinite, MAX_LOSS_GRID)
    ml_labels = [@sprintf("%.0f", m) for m in mls]
    delta_labels = [@sprintf("%.2f", d) for d in deltas]

    Z = zeros(length(deltas), length(mls))
    for (i, d) in enumerate(deltas)
        for (j, m) in enumerate(mls)
            row = filter(r -> r.delta == d && r.max_loss == m, summary_rows)
            Z[i, j] = isempty(row) ? NaN : row[1].p5_monthly_roi * 100
        end
    end

    p = heatmap(ml_labels, delta_labels, Z;
        title="5th Percentile Monthly ROI (%) — Tail Risk",
        xlabel="Max Loss (USD)", ylabel="Delta",
        color=:RdYlGn, size=(700, 500))
    savefig(p, joinpath(run_dir, "heatmap_p5_roi.png"))
end

# 4. Box plots: monthly ROI distribution by delta (at fixed max_loss=5.0)
let
    ref_ml = 5.0
    detail_at_ml = filter(r -> r.max_loss == ref_ml, detail_rows)
    if !isempty(detail_at_ml)
        deltas_present = sort(unique([r.delta for r in detail_at_ml]))
        groups = Int[]
        rois = Float64[]
        labels = String[]
        for (i, d) in enumerate(deltas_present)
            rows = filter(r -> r.delta == d, detail_at_ml)
            # Aggregate to monthly level
            monthly = Dict{String, Float64}()
            monthly_margin = Dict{String, Float64}()
            for r in rows
                monthly[r.year_month] = get(monthly, r.year_month, 0.0) + r.pnl
                monthly_margin[r.year_month] = get(monthly_margin, r.year_month, 0.0) + r.margin
            end
            for ym in keys(monthly)
                push!(groups, i)
                push!(rois, monthly[ym] / monthly_margin[ym] * 100)
                push!(labels, @sprintf("%.2f", d))
            end
        end
        p = boxplot(groups, rois;
            label="", title="Monthly ROI Distribution by Delta (max_loss=$ref_ml)",
            xlabel="Delta", ylabel="Monthly ROI (%)",
            xticks=(1:length(deltas_present), [@sprintf("%.2f", d) for d in deltas_present]),
            color=:steelblue, alpha=0.7, size=(800, 500))
        hline!(p, [0], color=:red, linestyle=:dash, label="", linewidth=1)
        savefig(p, joinpath(run_dir, "boxplot_delta_at_ml5.png"))
    end
end

# 5. Box plots: monthly ROI distribution by max_loss (at fixed delta=0.16)
let
    ref_delta = 0.16
    detail_at_d = filter(r -> r.delta == ref_delta && isfinite(r.max_loss), detail_rows)
    if !isempty(detail_at_d)
        mls_present = sort(unique([r.max_loss for r in detail_at_d]))
        groups = Int[]
        rois = Float64[]
        for (i, m) in enumerate(mls_present)
            rows = filter(r -> r.max_loss == m, detail_at_d)
            monthly = Dict{String, Float64}()
            monthly_margin = Dict{String, Float64}()
            for r in rows
                monthly[r.year_month] = get(monthly, r.year_month, 0.0) + r.pnl
                monthly_margin[r.year_month] = get(monthly_margin, r.year_month, 0.0) + r.margin
            end
            for ym in keys(monthly)
                push!(groups, i)
                push!(rois, monthly[ym] / monthly_margin[ym] * 100)
            end
        end
        p = boxplot(groups, rois;
            label="", title="Monthly ROI Distribution by Max Loss (delta=$ref_delta)",
            xlabel="Max Loss (USD)", ylabel="Monthly ROI (%)",
            xticks=(1:length(mls_present), [@sprintf("%.0f", m) for m in mls_present]),
            color=:darkorange, alpha=0.7, size=(800, 500))
        hline!(p, [0], color=:red, linestyle=:dash, label="", linewidth=1)
        savefig(p, joinpath(run_dir, "boxplot_ml_at_d16.png"))
    end
end

# 6. Line plot: mean ± 1 std monthly ROI vs delta for each max_loss
let
    mls = filter(isfinite, MAX_LOSS_GRID)
    p = plot(title="Mean Monthly ROI vs Delta by Max Loss",
        xlabel="Delta", ylabel="Monthly ROI (%)",
        legend=:topright, size=(800, 500))

    for ml in mls
        rows = filter(r -> r.max_loss == ml, summary_rows)
        isempty(rows) && continue
        ds = [r.delta for r in rows]
        means = [r.mean_monthly_roi * 100 for r in rows]
        stds = [r.std_monthly_roi * 100 for r in rows]
        plot!(p, ds, means; label=@sprintf("ml=%.0f", ml), linewidth=2, marker=:circle, markersize=3)
        plot!(p, ds, means .+ stds; label="", linestyle=:dot, linewidth=0.5, color=:gray, alpha=0.3)
        plot!(p, ds, means .- stds; label="", linestyle=:dot, linewidth=0.5, color=:gray, alpha=0.3)
    end
    hline!(p, [0]; color=:red, linestyle=:dash, label="", linewidth=1)
    savefig(p, joinpath(run_dir, "lines_roi_vs_delta.png"))
end

println("Plots saved to: $run_dir")
println("Done.")
