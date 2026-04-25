# scripts/strangle_rolling.jl
#
# Rolling-delta selector for 1-DTE short strangle (no wings) — picks
# (put_delta, call_delta) per fold by maximizing
#     score = mean(PnL) − z * |CVaR_α(PnL)|
# over the training window. z=0 reproduces the naive max-mean selector;
# z>0 penalizes high-mean / heavy-left-tail combos.
#
# Replaces strangle_rolling_diagnostic_1d.jl and strangle_rolling_regularized_1d.jl.
# Behavior is selected by the Z_VALUES env (comma-separated):
#   Z_VALUES=0                              → diagnostic mode (single z=0; matches
#                                              the naive selector + per-fold detail
#                                              report comparing chosen vs baseline,
#                                              IS→OOS Spearman, etc.)
#   Z_VALUES=0,0.1,0.3,1,3 (default)       → regularized sweep (per-year + summary
#                                              tables across z, equity overlay,
#                                              per-year Sharpe heatmap)
#
# Note: the original diagnostic used a max-Sharpe IS scorer rather than mean-PnL.
# Both reduce to "rank combos by IS reward" and the rolling-delta diagnostic was
# itself only meaningful at z=0; the unified pipeline uses mean−z·|CVaR| so a
# single Z_VALUES list parameterizes everything. The chosen-combo statistics in
# diagnostic mode (overfit gap, IS→OOS Spearman, baseline rank, per-fold detail)
# are preserved when length(Z_VALUES)==1.

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, Plots
include(joinpath(@__DIR__, "lib", "experiment.jl"))

# =============================================================================
# Configuration
# =============================================================================

SYMBOL          = get(ENV, "SYM", "SPY")
START_DATE      = Date(get(ENV, "START_DATE", "2014-06-02"))
END_DATE        = Date(get(ENV, "END_DATE",   "2026-03-27"))
ENTRY_TIME      = Time(parse(Int, get(ENV, "ENTRY_HOUR", "14")), 0)
EXPIRY_INTERVAL = Day(parse(Int, get(ENV, "EXPIRY_DAYS", "1")))
MAX_TAU_DAYS    = parse(Float64, get(ENV, "MAX_TAU_DAYS", "2.0"))
SPREAD_LAMBDA   = parse(Float64, get(ENV, "SPREAD_LAMBDA", "0.7"))
RATE            = parse(Float64, get(ENV, "RATE", "0.045"))
DIV_YIELD       = parse(Float64, get(ENV, "DIV", "0.013"))

PUT_DELTAS  = collect(0.05:0.05:0.40)
CALL_DELTAS = collect(0.05:0.05:0.40)
COMBOS = [(pd, cd) for pd in PUT_DELTAS for cd in CALL_DELTAS]
n_combos = length(COMBOS)

TRAIN_DAYS = parse(Int, get(ENV, "TRAIN_DAYS", "90"))
TEST_DAYS  = parse(Int, get(ENV, "TEST_DAYS",  "30"))
STEP_DAYS  = parse(Int, get(ENV, "STEP_DAYS",  "30"))

Z_VALUES   = [parse(Float64, strip(z))
              for z in split(get(ENV, "Z_VALUES", "0,0.1,0.3,1,3"), ",")]
BASELINE_COMBO = (0.20, 0.05)
CVAR_ALPHA = parse(Float64, get(ENV, "CVAR_ALPHA", "0.05"))
DIAGNOSTIC = length(Z_VALUES) == 1

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
mode_tag = DIAGNOSTIC ? "diag" : "reg"
run_dir = joinpath(@__DIR__, "runs", "strangle_rolling_$(mode_tag)_$(SYMBOL)_$run_ts")
mkpath(run_dir)
println("Output: $run_dir   mode=$(DIAGNOSTIC ? "diagnostic" : "regularized sweep")")
println("\n  $SYMBOL  $START_DATE → $END_DATE   strangle (no wings)")
println("  Grid: $n_combos combos   train=$TRAIN_DAYS d / test=$TEST_DAYS d / step=$STEP_DAYS d")
println("  z values: $Z_VALUES   α=$CVAR_ALPHA")

# =============================================================================
# Data source + dataset
# =============================================================================

println("\nLoading $SYMBOL …")
(; source, sched) = polygon_parquet_source(SYMBOL;
    start_date=START_DATE, end_date=END_DATE, entry_time=ENTRY_TIME,
    rate=RATE, div_yield=DIV_YIELD, spread_lambda=SPREAD_LAMBDA,
)

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
        sp_rec = find_record_at_strike(dctx.put_recs,  sp_K)
        sc_rec = find_record_at_strike(dctx.call_recs, sc_K)
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

baseline_ci = findfirst(==(BASELINE_COMBO), COMBOS)

# =============================================================================
# Helpers
# =============================================================================

sharpe_of(v) = let c = filter(!isnan, v); isempty(c) || std(c) == 0 ? NaN : mean(c)/std(c)*sqrt(252); end

function cvar_of(v, alpha)
    c = filter(!isnan, v)
    n = length(c); n < 2 && return -Inf
    k = max(1, ceil(Int, n * alpha))
    sorted = sort(c)
    mean(sorted[1:k])
end

function spearman(x, y)
    n = length(x); n < 3 && return NaN
    rx = sortperm(sortperm(x)); ry = sortperm(sortperm(y))
    d2 = sum((rx .- ry) .^ 2)
    1 - 6 * d2 / (n * (n^2 - 1))
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

        scores = fill(-Inf, n_combos)
        tr_sh = fill(NaN, n_combos); te_sh = fill(NaN, n_combos)
        for ci in 1:n_combos
            v = PnL[tr_idx, ci]
            c = filter(!isnan, v)
            length(c) < 10 && continue
            m = mean(c)
            cv = cvar_of(c, alpha)
            scores[ci] = m - lambda * abs(cv)
            tr_sh[ci] = sharpe_of(PnL[tr_idx, ci])
            te_sh[ci] = sharpe_of(PnL[te_idx, ci])
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

        # diagnostic per-fold extras
        chosen_pnls = filter(!isnan, PnL[te_idx, best_ci])
        base_pnls   = filter(!isnan, PnL[te_idx, baseline_ci])
        valid = .!isnan.(tr_sh)
        valid_oos = .!isnan.(te_sh)
        common = valid .& valid_oos
        rho = spearman(tr_sh[common], te_sh[common])
        valid_idx = findall(valid)
        ranks_desc = sortperm(tr_sh[valid_idx]; rev=true)
        baseline_pos_in_valid = findfirst(==(baseline_ci), valid_idx)
        baseline_is_rank = baseline_pos_in_valid === nothing ? -1 :
                            findfirst(==(baseline_pos_in_valid), ranks_desc)

        push!(folds, (idx=fidx, test=(test_start, te_end), chosen=chosen,
                       chosen_ci=best_ci, score=scores[best_ci],
                       train_sh_chosen=tr_sh[best_ci], test_sh_chosen=te_sh[best_ci],
                       train_sh_base=tr_sh[baseline_ci], test_sh_base=te_sh[baseline_ci],
                       test_total_chosen=sum(chosen_pnls), test_total_base=sum(base_pnls),
                       n_te=length(te_idx),
                       baseline_is_rank=baseline_is_rank, spearman=rho))
        test_start += Day(STEP_DAYS)
    end
    return oos_pnls, oos_dates, oos_combos, folds
end

function baseline_for_dates(target_dates, dates, PnL, baseline_ci)
    target_set = Set(target_dates)
    out_pnls = Float64[]; out_dates = Date[]
    for (i, d) in enumerate(dates)
        d in target_set || continue
        v = PnL[i, baseline_ci]
        isnan(v) && continue
        push!(out_pnls, v); push!(out_dates, d)
    end
    return out_pnls, out_dates
end

# =============================================================================
# Run
# =============================================================================

results = Dict{Float64,Any}()
for z in Z_VALUES
    p, d, c, f = rolling_select(dates, PnL, COMBOS, z, TRAIN_DAYS, TEST_DAYS, STEP_DAYS, CVAR_ALPHA)
    bp, bd = baseline_for_dates(d, dates, PnL, baseline_ci)
    results[z] = (pnls=p, dates=d, combos=c, folds=f, base_pnls=bp, base_dates=bd)
end

all_oos_dates = sort(collect(union([Set(results[z].dates) for z in Z_VALUES]...)))
baseline_full_p, baseline_full_d = baseline_for_dates(all_oos_dates, dates, PnL, baseline_ci)

# =============================================================================
# Reports
# =============================================================================

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

oos_years = sort(unique(Dates.year.(results[Z_VALUES[1]].dates)))

println("\n" * "=" ^ 80)
println("  Per-year Sharpe by z")
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
    yr_sh = filter(!isnan, [annual_sharpe(p, results[z].dates, y) for y in oos_years])
    @printf "  z=%-4.1f  %5d  %+9.0f  %+8.2f  %+8.2f  %+8.2f  %2d/%-2d\n" z length(p) sum(p) full_sh mean(yr_sh) minimum(yr_sh) count(>(0), yr_sh) length(yr_sh)
end
bp = baseline_full_p
b_full_sh = isempty(bp) || std(bp) == 0 ? NaN : mean(bp)/std(bp)*sqrt(252)
b_yr_sh = filter(!isnan, [annual_sharpe(bp, baseline_full_d, y) for y in oos_years])
@printf "  %-6s  %5d  %+9.0f  %+8.2f  %+8.2f  %+8.2f  %2d/%-2d\n" "fixed" length(bp) sum(bp) b_full_sh mean(b_yr_sh) minimum(b_yr_sh) count(>(0), b_yr_sh) length(b_yr_sh)

# Combo concentration
println("\n" * "=" ^ 80)
println("  Combo selection diversity by z")
println("=" ^ 80)
@printf "  %-6s  %-5s  %-25s  %-25s\n" "z" "n_uniq" "top combo (% folds)" "(0.20, 0.05) picks"
for z in Z_VALUES
    folds = results[z].folds
    counts = Dict{Tuple{Float64,Float64},Int}()
    for f in folds; counts[f.chosen] = get(counts, f.chosen, 0) + 1; end
    sorted = sort(collect(counts); by=x->x[2], rev=true)
    top, top_n = sorted[1]
    base_n = get(counts, BASELINE_COMBO, 0)
    @printf "  z=%-4.1f  %-5d  (%.2f, %.2f) %3d/%-3d (%.0f%%)   %d/%d (%.0f%%)\n" z length(counts) top[1] top[2] top_n length(folds) 100*top_n/length(folds) base_n length(folds) 100*base_n/length(folds)
end

# =============================================================================
# Diagnostic mode: per-fold detail (only for length(Z_VALUES)==1)
# =============================================================================

if DIAGNOSTIC
    z = Z_VALUES[1]
    folds = results[z].folds
    n_chosen_eq_base = count(f -> f.chosen == BASELINE_COMBO, folds)
    @printf "\n  Rolling picked (%.2f, %.2f) in %d / %d folds (%.1f%%)\n" BASELINE_COMBO[1] BASELINE_COMBO[2] n_chosen_eq_base length(folds) 100*n_chosen_eq_base/length(folds)

    diff_test_sh = [f.test_sh_chosen - f.test_sh_base for f in folds if !isnan(f.test_sh_chosen) && !isnan(f.test_sh_base)]
    diff_total   = [f.test_total_chosen - f.test_total_base for f in folds]
    @printf "\n  Test-Sharpe diff (chosen − baseline): mean %+.3f   median %+.3f   wins %d/%d\n" mean(diff_test_sh) median(diff_test_sh) count(>(0), diff_test_sh) length(diff_test_sh)
    @printf "  Test-PnL diff (chosen − baseline):   mean %+.2f   total %+.0f   wins %d/%d\n" mean(diff_total) sum(diff_total) count(>(0), diff_total) length(diff_total)

    mean_tr_chosen = mean(f.train_sh_chosen for f in folds if !isnan(f.train_sh_chosen))
    mean_te_chosen = mean(f.test_sh_chosen  for f in folds if !isnan(f.test_sh_chosen))
    @printf "\n  Mean train Sharpe of chosen combo: %+.2f\n" mean_tr_chosen
    @printf "  Mean test  Sharpe of chosen combo: %+.2f   (gap %+.2f)\n" mean_te_chosen (mean_te_chosen - mean_tr_chosen)

    mean_tr_base = mean(f.train_sh_base for f in folds if !isnan(f.train_sh_base))
    mean_te_base = mean(f.test_sh_base  for f in folds if !isnan(f.test_sh_base))
    @printf "  Mean train Sharpe of baseline: %+.2f\n" mean_tr_base
    @printf "  Mean test  Sharpe of baseline: %+.2f   (gap %+.2f)\n" mean_te_base (mean_te_base - mean_tr_base)

    rhos = [f.spearman for f in folds if !isnan(f.spearman)]
    @printf "\n  IS→OOS Spearman ρ per fold:  mean %+.3f   median %+.3f   N=%d\n" mean(rhos) median(rhos) length(rhos)
    @printf "  Folds with ρ > 0:   %d / %d\n" count(>(0), rhos) length(rhos)
    @printf "  Folds with ρ > 0.3: %d / %d\n" count(>(0.3), rhos) length(rhos)

    baseline_ranks = [f.baseline_is_rank for f in folds if f.baseline_is_rank > 0]
    @printf "\n  Baseline (%.2f, %.2f) IS rank distribution:  median %d  mean %.1f  (out of %d)\n" BASELINE_COMBO[1] BASELINE_COMBO[2] median(baseline_ranks) mean(baseline_ranks) n_combos
    @printf "  In-sample top-5 contains baseline:  %d / %d  (%.1f%%)\n" count(<=(5), baseline_ranks) length(baseline_ranks) 100*count(<=(5), baseline_ranks)/length(baseline_ranks)

    # Per-fold detail
    println("\n  ── Per-fold detail ──")
    @printf "  %3s  %-23s  %-10s  %+8s  %+8s  %+8s  %+8s  %4s\n" "#" "test window" "chosen" "trS_ch" "teS_ch" "trS_b" "teS_b" "rkB"
    for f in folds
        @printf "  %3d  %s → %s  (%.2f,%.2f)  %+8.2f  %+8.2f  %+8.2f  %+8.2f  %4d\n" f.idx f.test[1] f.test[2] f.chosen[1] f.chosen[2] f.train_sh_chosen f.test_sh_chosen f.train_sh_base f.test_sh_base f.baseline_is_rank
    end

    csv_path = joinpath(run_dir, "fold_diagnostic.csv")
    open(csv_path, "w") do io
        println(io, "fold,test_start,test_end,chosen_pd,chosen_cd,train_sh_chosen,test_sh_chosen,train_sh_base,test_sh_base,test_total_chosen,test_total_base,baseline_is_rank,spearman")
        for f in folds
            @printf io "%d,%s,%s,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%d,%.4f\n" f.idx f.test[1] f.test[2] f.chosen[1] f.chosen[2] f.train_sh_chosen f.test_sh_chosen f.train_sh_base f.test_sh_base f.test_total_chosen f.test_total_base f.baseline_is_rank f.spearman
        end
    end
    println("\n  Saved: fold_diagnostic.csv")
end

# =============================================================================
# Plots
# =============================================================================

plt = plot(; xlabel="date", ylabel="cumulative PnL (USD)",
    title="$SYMBOL strangle — rolling regularized selector by z",
    size=(1100, 600), legend=:topleft)
colors = [:steelblue, :seagreen, :darkorange, :firebrick, :purple]
for (zi, z) in enumerate(Z_VALUES)
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

if !DIAGNOSTIC
    sh_matrix = [annual_sharpe(results[z].pnls, results[z].dates, y) for z in Z_VALUES, y in oos_years]
    hm = heatmap(string.(oos_years), ["z=$z" for z in Z_VALUES], sh_matrix;
        color=:RdBu, clims=(-3.0, 5.0),
        xlabel="year", ylabel="regularization z",
        title="$SYMBOL strangle: per-year Sharpe by z",
        size=(900, 350),
    )
    savefig(hm, joinpath(run_dir, "yearly_sharpe_by_z.png"))
    println("  Saved: yearly_sharpe_by_z.png")
end

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
