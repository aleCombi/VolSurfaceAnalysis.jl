# scripts/strangle_rolling_regularized_1d.jl
#
# Regularized rolling-delta selector for SPY 1-DTE strangles.
# Selector picks max of (Sharpe − z * SE(Sharpe)) over the training window.
# z=0 reproduces the naive selector. Higher z penalizes high-Sharpe / low-N
# combos more, shrinking toward conservative picks.
#
# Reports per-year performance for each z (not just one Sharpe number).

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, Plots

SYMBOL          = get(ENV, "SYM", "SPY")
START_DATE      = Date(2014, 6, 2)
END_DATE        = Date(2026, 3, 27)
ENTRY_TIME      = Time(14, 0)
EXPIRY_INTERVAL = Day(1)
MAX_TAU_DAYS    = 2.0
SPREAD_LAMBDA   = 0.7
RATE            = 0.045
DIV_YIELD       = 0.013

PUT_DELTAS  = collect(0.05:0.05:0.40)
CALL_DELTAS = collect(0.05:0.05:0.40)
COMBOS = [(pd, cd) for pd in PUT_DELTAS for cd in CALL_DELTAS]
n_combos = length(COMBOS)

TRAIN_DAYS = 90
TEST_DAYS  = 30
STEP_DAYS  = 30

Z_VALUES = [0.0, 0.1, 0.3, 1.0, 3.0]   # CVaR penalty λ
BASELINE_COMBO = (0.20, 0.05)
CVAR_ALPHA = 0.05   # worst 5% of training PnLs

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "strangle_rolling_regularized_1d_$(SYMBOL)_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")
println("\n  $SYMBOL  $START_DATE → $END_DATE   1-DTE strangle")
println("  Grid: $n_combos combos   train=$TRAIN_DAYS d / test=$TEST_DAYS d / step=$STEP_DAYS d")
println("  λ values (CVaR penalty weight): $Z_VALUES   α=$CVAR_ALPHA")

# --- Data source --------------------------------------------------------------
println("\nLoading $SYMBOL  $START_DATE → $END_DATE ...")
(; source, sched) = polygon_parquet_source(SYMBOL;
    start_date=START_DATE, end_date=END_DATE, entry_time=ENTRY_TIME,
    rate=RATE, div_yield=DIV_YIELD, spread_lambda=SPREAD_LAMBDA,
)

# --- Build dataset ----------------------------------------------------------
dates = Date[]; pnl_rows = Vector{Vector{Float64}}()
n_total = 0; n_skip = 0

println("\nBuilding dataset...")
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
ord = sortperm(dates); dates = dates[ord]; pnl_rows = pnl_rows[ord]
PnL = permutedims(reduce(hcat, pnl_rows))
@printf "  %d kept (%d skipped)\n" length(dates) n_skip
years = Dates.year.(dates)

# --- Selector helpers -------------------------------------------------------
sharpe_of(v) = let c = filter(!isnan, v); isempty(c) || std(c) == 0 ? NaN : mean(c)/std(c)*sqrt(252); end

# CVaR at α: mean of worst α-quantile of training PnLs (negative for losing tail)
function cvar_of(v, alpha)
    c = filter(!isnan, v)
    n = length(c); n < 2 && return -Inf
    k = max(1, ceil(Int, n * alpha))
    sorted = sort(c)
    mean(sorted[1:k])
end

function rolling_select(dates, PnL, COMBOS, lambda, TRAIN_DAYS, TEST_DAYS, STEP_DAYS, alpha)
    n_combos = length(COMBOS)
    oos_pnls = Float64[]; oos_dates = Date[]
    oos_combos = Tuple{Float64,Float64}[]
    folds = NamedTuple[]

    test_start = dates[1] + Day(TRAIN_DAYS); last_d = dates[end]
    fidx = 0
    while test_start <= last_d
        te_end = test_start + Day(TEST_DAYS) - Day(1)
        tr_start = test_start - Day(TRAIN_DAYS); tr_end = test_start - Day(1)
        tr_idx = findall(d -> tr_start <= d <= tr_end, dates)
        te_idx = findall(d -> test_start <= d <= te_end, dates)
        if length(tr_idx) < 30 || length(te_idx) < 5
            test_start += Day(STEP_DAYS); continue
        end

        # Score = mean(pnl) - λ * |CVaR_α(pnl)|.  Units: USD per trade.
        scores = fill(-Inf, n_combos)
        for ci in 1:n_combos
            v = PnL[tr_idx, ci]
            c = filter(!isnan, v)
            length(c) < 10 && continue
            m = mean(c)
            cv = cvar_of(c, alpha)
            scores[ci] = m - lambda * abs(cv)
        end
        best_ci = argmax(scores)
        chosen = COMBOS[best_ci]
        for i in te_idx
            v = PnL[i, best_ci]
            isnan(v) && continue
            push!(oos_pnls, v); push!(oos_dates, dates[i])
            push!(oos_combos, chosen)
        end
        fidx += 1
        push!(folds, (idx=fidx, test=(test_start, te_end), chosen=chosen, score=scores[best_ci]))
        test_start += Day(STEP_DAYS)
    end
    return oos_pnls, oos_dates, oos_combos, folds
end

# Baseline: fixed (0.20, 0.05) over the same OOS dates per z
function baseline_for_dates(target_dates, dates, PnL, COMBOS)
    bi = findfirst(==(BASELINE_COMBO), COMBOS)
    target_set = Set(target_dates)
    out_pnls = Float64[]; out_dates = Date[]
    for (i, d) in enumerate(dates)
        d in target_set || continue
        v = PnL[i, bi]
        isnan(v) && continue
        push!(out_pnls, v); push!(out_dates, d)
    end
    return out_pnls, out_dates
end

# --- Run all z values --------------------------------------------------------
results = Dict{Float64,Any}()
for z in Z_VALUES
    p, d, c, f = rolling_select(dates, PnL, COMBOS, z, TRAIN_DAYS, TEST_DAYS, STEP_DAYS, CVAR_ALPHA)
    bp, bd = baseline_for_dates(d, dates, PnL, COMBOS)
    results[z] = (pnls=p, dates=d, combos=c, folds=f, base_pnls=bp, base_dates=bd)
end

# Also a pure-baseline series over the union of all OOS dates (for context)
all_oos_dates = sort(collect(union([Set(results[z].dates) for z in Z_VALUES]...)))
baseline_full_p, baseline_full_d = baseline_for_dates(all_oos_dates, dates, PnL, COMBOS)

# --- Reports -----------------------------------------------------------------
function annual_sharpe(pnls, ds, year)
    mask = Dates.year.(ds) .== year
    v = filter(!isnan, pnls[mask])
    isempty(v) || std(v) == 0 ? NaN : mean(v)/std(v)*sqrt(252)
end
function annual_total(pnls, ds, year)
    mask = Dates.year.(ds) .== year
    v = filter(!isnan, pnls[mask])
    isempty(v) ? 0.0 : sum(v)
end

oos_years = sort(unique(Dates.year.(results[0.0].dates)))

println("\n" * "=" ^ 80)
println("  Per-year Sharpe by z (regularization strength)")
println("=" ^ 80)
@printf "  %-5s" "year"
for z in Z_VALUES; @printf "  z=%-5.1f" z; end
@printf "    %s\n" "baseline"
println("  " * "─"^(8 + 9 * length(Z_VALUES) + 12))
for y in oos_years
    @printf "  %-5d" y
    for z in Z_VALUES
        sh = annual_sharpe(results[z].pnls, results[z].dates, y)
        @printf "  %+7.2f" sh
    end
    bsh = annual_sharpe(baseline_full_p, baseline_full_d, y)
    @printf "    %+7.2f\n" bsh
end

println("\n" * "=" ^ 80)
println("  Per-year total \$ PnL by z")
println("=" ^ 80)
@printf "  %-5s" "year"
for z in Z_VALUES; @printf "  z=%-7.1f" z; end
@printf "    %s\n" "baseline"
println("  " * "─"^(8 + 11 * length(Z_VALUES) + 12))
for y in oos_years
    @printf "  %-5d" y
    for z in Z_VALUES
        t = annual_total(results[z].pnls, results[z].dates, y)
        @printf "  %+9.0f" t
    end
    bt = annual_total(baseline_full_p, baseline_full_d, y)
    @printf "    %+7.0f\n" bt
end

println("\n" * "=" ^ 80)
println("  Full-period summary by z")
println("=" ^ 80)
@printf "  %-6s  %5s  %+9s  %+8s  %+8s  %+8s  %5s\n" "z" "trades" "total" "Sharpe" "MeanYr" "MinYr" "+yrs"
for z in Z_VALUES
    p = results[z].pnls
    full_sh = isempty(p) || std(p) == 0 ? NaN : mean(p)/std(p)*sqrt(252)
    yr_sh = [annual_sharpe(p, results[z].dates, y) for y in oos_years]
    yr_sh_clean = filter(!isnan, yr_sh)
    @printf "  z=%-4.1f  %5d  %+9.0f  %+8.2f  %+8.2f  %+8.2f  %2d/%-2d\n" z length(p) sum(p) full_sh mean(yr_sh_clean) minimum(yr_sh_clean) count(>(0), yr_sh_clean) length(yr_sh_clean)
end
# Baseline row
bp = baseline_full_p
b_full_sh = isempty(bp) || std(bp) == 0 ? NaN : mean(bp)/std(bp)*sqrt(252)
b_yr_sh = filter(!isnan, [annual_sharpe(bp, baseline_full_d, y) for y in oos_years])
@printf "  %-6s  %5d  %+9.0f  %+8.2f  %+8.2f  %+8.2f  %2d/%-2d\n" "fixed" length(bp) sum(bp) b_full_sh mean(b_yr_sh) minimum(b_yr_sh) count(>(0), b_yr_sh) length(b_yr_sh)

# Combo concentration per z
println("\n" * "=" ^ 80)
println("  Combo selection diversity by z")
println("=" ^ 80)
@printf "  %-6s  %-5s  %-25s  %-25s\n" "z" "n_uniq" "top combo (% folds)" "(0.20, 0.05) picks"
for z in Z_VALUES
    cs = results[z].combos
    folds = results[z].folds
    counts = Dict{Tuple{Float64,Float64},Int}()
    for f in folds; counts[f.chosen] = get(counts, f.chosen, 0) + 1; end
    sorted = sort(collect(counts); by=x->x[2], rev=true)
    top, top_n = sorted[1]
    base_n = get(counts, BASELINE_COMBO, 0)
    @printf "  z=%-4.1f  %-5d  (%.2f, %.2f) %3d/%-3d (%.0f%%)   %d/%d (%.0f%%)\n" z length(counts) top[1] top[2] top_n length(folds) 100*top_n/length(folds) base_n length(folds) 100*base_n/length(folds)
end

# --- Plots ------------------------------------------------------------------
# Equity curves for each z + baseline
plt = plot(; xlabel="date", ylabel="cumulative PnL (USD)",
    title="$SYMBOL 1-DTE strangle — rolling regularized selector by z",
    size=(1100, 600), legend=:topleft)
colors = [:steelblue, :seagreen, :darkorange, :firebrick, :purple]
for (zi, z) in enumerate(Z_VALUES)
    od = sort(unique(results[z].dates))
    od_pos = sortperm(results[z].dates)
    plot!(plt, results[z].dates[od_pos], cumsum(results[z].pnls[od_pos]);
        label="z=$z", lw=2, color=colors[mod1(zi, length(colors))])
end
ord_b = sortperm(baseline_full_d)
plot!(plt, baseline_full_d[ord_b], cumsum(baseline_full_p[ord_b]);
    label="fixed (0.20, 0.05)", lw=2, color=:black, ls=:dash)
hline!(plt, [0]; color=:gray, ls=:dash, label="")
savefig(plt, joinpath(run_dir, "equity_curves.png"))
println("\n  Saved: equity_curves.png")

# Per-year Sharpe heatmap (rows = z, cols = year)
sh_matrix = [annual_sharpe(results[z].pnls, results[z].dates, y) for z in Z_VALUES, y in oos_years]
hm = heatmap(string.(oos_years), ["z=$z" for z in Z_VALUES], sh_matrix;
    color=:RdBu, clims=(-3.0, 5.0),
    xlabel="year", ylabel="regularization z",
    title="$SYMBOL 1-DTE strangle: per-year Sharpe by z",
    size=(900, 350),
)
savefig(hm, joinpath(run_dir, "yearly_sharpe_by_z.png"))
println("  Saved: yearly_sharpe_by_z.png")

# CSV dump
csv_path = joinpath(run_dir, "summary.csv")
open(csv_path, "w") do io
    println(io, "z,year,n,total,sharpe")
    for z in Z_VALUES, y in oos_years
        mask = Dates.year.(results[z].dates) .== y
        v = filter(!isnan, results[z].pnls[mask])
        isempty(v) && continue
        sh = std(v) == 0 ? NaN : mean(v)/std(v)*sqrt(252)
        @printf io "%.1f,%d,%d,%.2f,%.4f\n" z y length(v) sum(v) sh
    end
end
println("  Saved: summary.csv")
