# scripts/strangle_2018_breakdown.jl
#
# Diagnoses the 2018 SPY 1-DTE short-strangle drawdown:
#  - Day-by-day PnL for the (0.20, 0.05) baseline through 2018
#  - Worst-N-day concentration: how much of the annual loss is in tail days
#  - Comparison: same metrics for a few alternative deltas

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, Plots

SYMBOL = "SPY"
START_DATE = Date(2017, 10, 1)   # buffer for context
END_DATE   = Date(2019, 3, 31)
ENTRY_TIME = Time(14, 0)
EXPIRY_INTERVAL = Day(1)
MAX_TAU_DAYS = 2.0
SPREAD_LAMBDA = 0.7
RATE = 0.045
DIV_YIELD = 0.013

# Compare a few combos
COMBOS = [(0.20, 0.05), (0.05, 0.05), (0.10, 0.10), (0.30, 0.05), (0.20, 0.20)]
n_combos = length(COMBOS)

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "strangle_2018_breakdown_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")

# --- Data source ------------------------------------------------------------
println("Loading SPY $START_DATE → $END_DATE ...")
(; source, sched) = polygon_parquet_source(SYMBOL;
    start_date=START_DATE, end_date=END_DATE, entry_time=ENTRY_TIME,
    rate=RATE, div_yield=DIV_YIELD, spread_lambda=SPREAD_LAMBDA,
)

dates = Date[]; pnl_rows = Vector{Vector{Float64}}()
spot_at_entry = Float64[]; spot_at_settle = Float64[]

println("\nBuilding dataset...")
each_entry(source, EXPIRY_INTERVAL, sched; clear_cache=true) do ctx, settlement
    ismissing(settlement) && return
    dctx = delta_context(ctx; rate=RATE, div_yield=DIV_YIELD)
    dctx === nothing && return
    dctx.tau * 365.25 > MAX_TAU_DAYS && return
    spot = dctx.spot
    spot_settle = Float64(settlement)

    row = fill(NaN, n_combos)
    for (i, (pd, cd)) in enumerate(COMBOS)
        sp_K = delta_strike(dctx, -pd, Put)
        sc_K = delta_strike(dctx,  cd, Call)
        (sp_K === nothing || sc_K === nothing) && continue
        sp_rec = nothing; sc_rec = nothing
        for r in dctx.put_recs;  r.strike == sp_K && (sp_rec = r; break); end
        for r in dctx.call_recs; r.strike == sc_K && (sc_rec = r; break); end
        (sp_rec === nothing || sc_rec === nothing) && continue
        sp_bid = extract_price(sp_rec, :bid)
        sc_bid = extract_price(sc_rec, :bid)
        (sp_bid === nothing || sc_bid === nothing) && continue
        credit_usd = (sp_bid + sc_bid) * spot
        intrinsic_usd = max(sp_K - spot_settle, 0.0) + max(spot_settle - sc_K, 0.0)
        row[i] = credit_usd - intrinsic_usd
    end
    push!(dates, Date(ctx.surface.timestamp))
    push!(pnl_rows, row)
    push!(spot_at_entry, spot)
    push!(spot_at_settle, spot_settle)
end
ord = sortperm(dates); dates = dates[ord]; pnl_rows = pnl_rows[ord]
spot_at_entry = spot_at_entry[ord]; spot_at_settle = spot_at_settle[ord]
PnL = permutedims(reduce(hcat, pnl_rows))
@printf "  %d entries kept\n" length(dates)

# --- Filter to 2018 ---------------------------------------------------------
mask_2018 = Dates.year.(dates) .== 2018
d18 = dates[mask_2018]
P18 = PnL[mask_2018, :]
sp_e18 = spot_at_entry[mask_2018]
sp_s18 = spot_at_settle[mask_2018]
@printf "\n  2018 entries: %d\n" length(d18)

# Per-combo summary in 2018
println("\n  Per-combo 2018 summary:")
@printf "    %-12s  %5s  %+10s  %+8s  %+8s  %+8s\n" "(pd, cd)" "n" "total" "Sharpe" "MeanPnL" "MinPnL"
for ci in 1:n_combos
    v = filter(!isnan, P18[:, ci])
    isempty(v) && continue
    sh = std(v) > 0 ? mean(v)/std(v)*sqrt(252) : 0.0
    @printf "    (%.2f, %.2f)  %5d  %+10.0f  %+8.2f  %+8.2f  %+8.2f\n" COMBOS[ci][1] COMBOS[ci][2] length(v) sum(v) sh mean(v) minimum(v)
end

# Concentration of losses for baseline (0.20, 0.05)
baseline_ci = findfirst(==((0.20, 0.05)), COMBOS)
v18 = P18[:, baseline_ci]
total_2018 = sum(filter(!isnan, v18))

println("\n  ── Loss concentration: baseline (0.20, 0.05) in 2018 ──")
sorted_idx = sortperm(v18)   # ascending → worst first
for k in [1, 3, 5, 10, 20]
    worst_k = sum(v18[sorted_idx[1:min(k, length(sorted_idx))]])
    @printf "    Worst %2d days sum: %+8.2f   (%.1f%% of annual %+.0f)\n" k worst_k 100*worst_k/total_2018 total_2018
end

# Identify the worst 10 days with date and spot move
println("\n  ── Top 10 worst days for baseline (0.20, 0.05) in 2018 ──")
@printf "    %-11s  %10s  %10s  %+9s  %+8s\n" "date" "spot_entry" "spot_settle" "spot_chg%" "PnL"
for k in 1:10
    i = sorted_idx[k]
    chg = 100 * (sp_s18[i] - sp_e18[i]) / sp_e18[i]
    @printf "    %-11s  %10.2f  %10.2f  %+9.2f  %+8.2f\n" d18[i] sp_e18[i] sp_s18[i] chg v18[i]
end

# Compare top 10 days across combos: do same days hit different combos?
println("\n  ── Same 10 worst dates (for baseline), PnL across combos ──")
@printf "    %-11s" "date"
for c in COMBOS; @printf "  (%.2f,%.2f)" c[1] c[2]; end
@printf "\n"
worst_dates_idx = sorted_idx[1:10]
for i in worst_dates_idx
    @printf "    %-11s" d18[i]
    for ci in 1:n_combos
        v = P18[i, ci]
        @printf "  %+9.2f" v
    end
    @printf "\n"
end

# Daily PnL bar chart for baseline in 2018
plt = bar(d18, v18; label="(0.20, 0.05)", color=ifelse.(v18 .>= 0, :seagreen, :firebrick),
    xlabel="date", ylabel="daily PnL (USD)", title="SPY 1-DTE strangle (0.20, 0.05) — 2018 daily PnL",
    size=(1100, 400), legend=false)
hline!(plt, [0]; color=:black, lw=1)
savefig(plt, joinpath(run_dir, "2018_daily_pnl_baseline.png"))
println("\n  Saved: 2018_daily_pnl_baseline.png")

# Cumulative PnL through 2018 for all combos
plt2 = plot(; xlabel="date", ylabel="cumulative PnL (USD)",
    title="SPY 1-DTE strangle — 2018 cumulative PnL by combo",
    size=(1100, 500), legend=:bottomleft)
colors = [:steelblue, :darkorange, :seagreen, :purple, :firebrick]
for (ci, c) in enumerate(COMBOS)
    v = P18[:, ci]
    keep = .!isnan.(v)
    plot!(plt2, d18[keep], cumsum(v[keep]); label="(pd=$(c[1]), cd=$(c[2]))", lw=2, color=colors[mod1(ci, length(colors))])
end
hline!(plt2, [0]; color=:gray, ls=:dash, label="")
savefig(plt2, joinpath(run_dir, "2018_cumulative.png"))
println("  Saved: 2018_cumulative.png")
