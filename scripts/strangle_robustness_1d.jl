# scripts/strangle_robustness_1d.jl
#
# Multi-year robustness analysis for SPY 1-DTE short strangles.
# Goal: find (put_delta, call_delta) combos that are *consistently* good across
# years, not just optimal over one period. Stability metrics beat raw Sharpe.

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, Plots

SYMBOL          = get(ENV, "SYM", "SPY")
START_DATE      = Date(parse(Int, get(ENV, "START_YEAR", "2014")), 6, 2)
END_DATE        = Date(2026, 3, 27)
ENTRY_TIME      = Time(14, 0)
EXPIRY_INTERVAL = Day(1)
MAX_TAU_DAYS    = 2.0
SPREAD_LAMBDA   = 0.7
RATE            = 0.045
DIV_YIELD       = parse(Float64, get(ENV, "DIV", "0.013"))

PUT_DELTAS  = collect(0.05:0.05:0.40)   # 8 values
CALL_DELTAS = collect(0.05:0.05:0.40)   # 8 values
COMBOS = [(pd, cd) for pd in PUT_DELTAS for cd in CALL_DELTAS]
n_combos = length(COMBOS)

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "strangle_robustness_1d_$(SYMBOL)_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")
println("\n  $SYMBOL  $START_DATE → $END_DATE   1-DTE   $(n_combos) (pd, cd) combos")

# --- Data source --------------------------------------------------------------
println("\nLoading $SYMBOL  $START_DATE → $END_DATE ...")
(; source, sched) = polygon_parquet_source(SYMBOL;
    start_date=START_DATE, end_date=END_DATE, entry_time=ENTRY_TIME,
    rate=RATE, div_yield=DIV_YIELD, spread_lambda=SPREAD_LAMBDA,
)

# --- Build dataset: PnL[entry] is Vector of length n_combos -----------------
dates = Date[]
pnl_rows = Vector{Vector{Float64}}()

n_total = 0; n_skip = 0

println("\nBuilding dataset (per-entry PnL × $n_combos combos)...")
each_entry(source, EXPIRY_INTERVAL, sched; clear_cache=true) do ctx, settlement
    ismissing(settlement) && return
    global n_total += 1
    dctx = delta_context(ctx; rate=RATE, div_yield=DIV_YIELD)
    dctx === nothing && (global n_skip += 1; return)
    dctx.tau * 365.25 > MAX_TAU_DAYS && (global n_skip += 1; return)
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
end
println()
@printf "  %d entries → kept %d  (skipped %d)\n" n_total length(dates) n_skip
length(dates) < 100 && error("Too few entries kept")

ord = sortperm(dates); dates = dates[ord]; pnl_rows = pnl_rows[ord]
PnL = permutedims(reduce(hcat, pnl_rows))   # rows = entries, cols = combos
years = Dates.year.(dates)
year_set = sort(unique(years))
@printf "  Years: %d → %d  (%d years)\n" first(year_set) last(year_set) length(year_set)

sharpe_of(v) = let c = filter(!isnan, v); isempty(c) || std(c) == 0 ? NaN : mean(c)/std(c)*sqrt(252); end
mean_of(v)   = let c = filter(!isnan, v); isempty(c) ? NaN : mean(c); end
winrate_of(v) = let c = filter(!isnan, v); isempty(c) ? NaN : mean(c .> 0); end
mdd_of(v) = let c = filter(!isnan, v)
    isempty(c) ? NaN :
        let cum = cumsum(c), rmax = accumulate(max, cum); -minimum(cum .- rmax); end
end

# --- Per (combo, year) stats -------------------------------------------------
n_years = length(year_set)
yearly_sharpe = fill(NaN, n_combos, n_years)
yearly_mean   = fill(NaN, n_combos, n_years)
yearly_n      = zeros(Int, n_combos, n_years)

for (yi, y) in enumerate(year_set)
    yr_mask = years .== y
    for ci in 1:n_combos
        v = PnL[yr_mask, ci]
        yearly_sharpe[ci, yi] = sharpe_of(v)
        yearly_mean[ci, yi]   = mean_of(v)
        yearly_n[ci, yi]      = count(!isnan, v)
    end
end

# Drop years with too-few trades for any combo (gives bad year-Sharpes)
yr_keep = [yi for yi in 1:n_years if minimum(yearly_n[:, yi]) >= 50]
yearly_sharpe = yearly_sharpe[:, yr_keep]
yearly_mean   = yearly_mean[:, yr_keep]
yearly_n      = yearly_n[:, yr_keep]
year_set      = year_set[yr_keep]
n_years       = length(year_set)
@printf "  After year-trade-count filter: %d years (%s)\n" n_years join(year_set, ",")

# --- Combo-level summary stats ----------------------------------------------
combo_stats = NamedTuple[]
for ci in 1:n_combos
    pd, cd = COMBOS[ci]
    yrs = filter(!isnan, yearly_sharpe[ci, :])
    n_yr = length(yrs)
    n_yr == 0 && continue
    full = filter(!isnan, PnL[:, ci])
    push!(combo_stats, (
        ci=ci, pd=pd, cd=cd,
        n_total = length(full),
        full_sharpe = sharpe_of(full),
        full_mean   = mean(full),
        full_total  = sum(full),
        full_mdd    = mdd_of(full),
        full_winrate = winrate_of(full),
        mean_yr_sharpe = mean(yrs),
        median_yr_sharpe = median(yrs),
        std_yr_sharpe  = n_yr > 1 ? std(yrs) : 0.0,
        min_yr_sharpe  = minimum(yrs),
        n_pos_years    = count(>(0), yrs),
        n_years        = n_yr,
        # Across-year stability: mean / std of yearly Sharpe (higher = more consistent)
        ic_stability   = (n_yr > 1 && std(yrs) > 0) ? mean(yrs)/std(yrs) : 0.0,
    ))
end

println("\n  ── Top 12 by MIN year-Sharpe (worst-case robustness) ──")
sort!(combo_stats; by=x->x.min_yr_sharpe, rev=true)
@printf "    %-7s  %-7s  %5s  %+8s  %+8s  %+8s  %+8s  %+5s\n" "putΔ" "callΔ" "n" "FullSh" "MeanYr" "MedYr" "MinYr" "+yrs"
for c in first(combo_stats, 12)
    @printf "    %-7.2f  %-7.2f  %5d  %+8.2f  %+8.2f  %+8.2f  %+8.2f  %2d/%-2d\n" c.pd c.cd c.n_total c.full_sharpe c.mean_yr_sharpe c.median_yr_sharpe c.min_yr_sharpe c.n_pos_years c.n_years
end

println("\n  ── Top 12 by stability ratio (mean / std of yearly Sharpe) ──")
sort!(combo_stats; by=x->x.ic_stability, rev=true)
@printf "    %-7s  %-7s  %5s  %+8s  %+8s  %+8s  %+8s  %+5s\n" "putΔ" "callΔ" "n" "FullSh" "MeanYr" "StdYr" "Stabil" "+yrs"
for c in first(combo_stats, 12)
    @printf "    %-7.2f  %-7.2f  %5d  %+8.2f  %+8.2f  %+8.2f  %+8.2f  %2d/%-2d\n" c.pd c.cd c.n_total c.full_sharpe c.mean_yr_sharpe c.std_yr_sharpe c.ic_stability c.n_pos_years c.n_years
end

println("\n  ── Top 12 by full-period Sharpe (for reference / overfit check) ──")
sort!(combo_stats; by=x->x.full_sharpe, rev=true)
@printf "    %-7s  %-7s  %5s  %+8s  %+8s  %+8s  %+8s  %+5s\n" "putΔ" "callΔ" "n" "FullSh" "MeanYr" "MinYr" "MaxDD" "+yrs"
for c in first(combo_stats, 12)
    @printf "    %-7.2f  %-7.2f  %5d  %+8.2f  %+8.2f  %+8.2f  %8.0f  %2d/%-2d\n" c.pd c.cd c.n_total c.full_sharpe c.mean_yr_sharpe c.min_yr_sharpe c.full_mdd c.n_pos_years c.n_years
end

# --- Heatmap: per-combo per-year Sharpe -------------------------------------
# Compact axis: combos labelled "p|c"
combo_labels = ["$(round(c.pd; digits=2))|$(round(c.cd; digits=2))" for c in [(pd=p, cd=c) for (p, c) in COMBOS]]
hm = heatmap(string.(year_set), combo_labels, yearly_sharpe;
    color=:RdBu, clims=(-3.0, 3.0),
    xlabel="year", ylabel="putΔ | callΔ",
    title="$SYMBOL 1-DTE strangle: per-(combo, year) Sharpe",
    size=(900, 1100),
)
savefig(hm, joinpath(run_dir, "yearly_sharpe_heatmap.png"))
println("\n  Saved: yearly_sharpe_heatmap.png")

# Per-year breakdown for the top stability winner
sort!(combo_stats; by=x->x.ic_stability, rev=true)
top = combo_stats[1]
println("\n  ── Per-year detail for top-stability combo: putΔ=$(top.pd)  callΔ=$(top.cd) ──")
@printf "    %-5s  %5s  %+8s  %+8s\n" "year" "n" "Sharpe" "mean"
for (yi, y) in enumerate(year_set)
    @printf "    %-5d  %5d  %+8.2f  %+8.2f\n" y yearly_n[top.ci, yi] yearly_sharpe[top.ci, yi] yearly_mean[top.ci, yi]
end

# Dump CSV for further analysis
csv_path = joinpath(run_dir, "combo_stats.csv")
open(csv_path, "w") do io
    println(io, "putDelta,callDelta,n_total,full_sharpe,full_mean,full_total,full_mdd,full_winrate,mean_yr_sharpe,median_yr_sharpe,std_yr_sharpe,min_yr_sharpe,n_pos_years,n_years,ic_stability")
    for c in combo_stats
        @printf io "%.2f,%.2f,%d,%.4f,%.4f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%.4f\n" c.pd c.cd c.n_total c.full_sharpe c.full_mean c.full_total c.full_mdd c.full_winrate c.mean_yr_sharpe c.median_yr_sharpe c.std_yr_sharpe c.min_yr_sharpe c.n_pos_years c.n_years c.ic_stability
    end
end
println("\n  Saved: combo_stats.csv")
