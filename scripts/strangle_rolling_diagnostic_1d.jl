# scripts/strangle_rolling_diagnostic_1d.jl
#
# Diagnostic: how does a rolling-delta selector behave on 1-DTE SPY strangles?
# Compares each fold's chosen (pd, cd) to a baseline (0.20, 0.05) and reports:
#  - what the selector picked
#  - chosen combo's train Sharpe vs test Sharpe (the overfit signature)
#  - chosen vs baseline test Sharpe per fold (did rolling beat fixed?)
#  - rank correlation IS→OOS across the full grid

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, Plots
include(joinpath(@__DIR__, "lib", "experiment.jl"))

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

BASELINE_COMBO = (0.20, 0.05)

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "strangle_rolling_diagnostic_1d_$(SYMBOL)_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")
println("\n  $SYMBOL  $START_DATE → $END_DATE   1-DTE strangle (no wings)")
println("  Grid: $n_combos combos   train=$TRAIN_DAYS d / test=$TEST_DAYS d / step=$STEP_DAYS d")

# --- Data source --------------------------------------------------------------
println("\nLoading $SYMBOL  $START_DATE → $END_DATE ...")
(; source, sched) = polygon_parquet_source(SYMBOL;
    start_date=START_DATE, end_date=END_DATE, entry_time=ENTRY_TIME,
    rate=RATE, div_yield=DIV_YIELD, spread_lambda=SPREAD_LAMBDA,
)

# --- Build dataset (PnL[entry, combo]) ---------------------------------------
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

baseline_ci = findfirst(==(BASELINE_COMBO), COMBOS)
@printf "  Baseline combo: (%.2f, %.2f) at index %d\n" BASELINE_COMBO[1] BASELINE_COMBO[2] baseline_ci

sharpe_of(v) = let c = filter(!isnan, v); isempty(c) || std(c) == 0 ? NaN : mean(c)/std(c)*sqrt(252); end

# --- Rolling-delta selector with detailed per-fold diagnostics ---------------
function rolling_diag(dates, PnL, COMBOS, baseline_ci, TRAIN_DAYS, TEST_DAYS, STEP_DAYS)
    n_combos = length(COMBOS)
    fold_results = NamedTuple[]
    wf_folds = build_folds(dates;
        train_days=TRAIN_DAYS, test_days=TEST_DAYS, step_days=STEP_DAYS,
        min_train=30, min_test=5,
    )
    function spearman(x, y)
        n = length(x); n < 3 && return NaN
        rx = sortperm(sortperm(x)); ry = sortperm(sortperm(y))
        d2 = sum((rx .- ry) .^ 2)
        1 - 6 * d2 / (n * (n^2 - 1))
    end
    for f in wf_folds
        tr_idx = findall(f.train_mask)
        te_idx = findall(f.test_mask)

        # Train Sharpe across all combos
        tr_sh = [sharpe_of(PnL[tr_idx, c]) for c in 1:n_combos]
        te_sh = [sharpe_of(PnL[te_idx, c]) for c in 1:n_combos]

        # Selector picks max IS Sharpe
        valid = .!isnan.(tr_sh)
        best_ci = findall(valid)[argmax(tr_sh[valid])]
        chosen = COMBOS[best_ci]

        # Test PnL realized for chosen vs baseline
        chosen_pnls = filter(!isnan, PnL[te_idx, best_ci])
        base_pnls   = filter(!isnan, PnL[te_idx, baseline_ci])

        # Rank of baseline in IS — where would the search rank our anchor?
        valid_idx = findall(valid)
        ranks_desc = sortperm(tr_sh[valid_idx]; rev=true)
        baseline_pos_in_valid = findfirst(==(baseline_ci), valid_idx)
        baseline_is_rank = baseline_pos_in_valid === nothing ? -1 :
                            findfirst(==(baseline_pos_in_valid), ranks_desc)

        # IS→OOS rank correlation (Spearman) over valid combos
        valid_oos = .!isnan.(te_sh)
        common = valid .& valid_oos
        rho = spearman(tr_sh[common], te_sh[common])

        push!(fold_results, (
            idx=f.idx, test=(f.test_start, f.test_end),
            chosen=chosen, chosen_ci=best_ci,
            train_sh_chosen = tr_sh[best_ci],
            test_sh_chosen  = te_sh[best_ci],
            train_sh_base   = tr_sh[baseline_ci],
            test_sh_base    = te_sh[baseline_ci],
            test_total_chosen = sum(chosen_pnls),
            test_total_base   = sum(base_pnls),
            n_te = length(te_idx),
            baseline_is_rank = baseline_is_rank,
            spearman = rho,
        ))
    end
    return fold_results
end

folds = rolling_diag(dates, PnL, COMBOS, baseline_ci, TRAIN_DAYS, TEST_DAYS, STEP_DAYS)
println("\n  $(length(folds)) folds")

# --- Aggregates --------------------------------------------------------------
n_chosen_eq_base = count(f -> f.chosen == BASELINE_COMBO, folds)
@printf "\n  Rolling picked (%.2f, %.2f) in %d / %d folds (%.1f%%)\n" BASELINE_COMBO[1] BASELINE_COMBO[2] n_chosen_eq_base length(folds) 100*n_chosen_eq_base/length(folds)

# Per-fold realized: chosen vs baseline
diff_test_sh = [f.test_sh_chosen - f.test_sh_base for f in folds if !isnan(f.test_sh_chosen) && !isnan(f.test_sh_base)]
diff_total   = [f.test_total_chosen - f.test_total_base for f in folds]
@printf "\n  Test-Sharpe diff (chosen − baseline): mean %+.3f   median %+.3f   wins %d/%d\n" mean(diff_test_sh) median(diff_test_sh) count(>(0), diff_test_sh) length(diff_test_sh)
@printf "  Test-PnL diff (chosen − baseline):   mean %+.2f   total %+.0f   wins %d/%d\n" mean(diff_total) sum(diff_total) count(>(0), diff_total) length(diff_total)

# Overfit signature: train sh(chosen) >> test sh(chosen)?
mean_tr_chosen = mean(f.train_sh_chosen for f in folds)
mean_te_chosen = mean(f.test_sh_chosen  for f in folds if !isnan(f.test_sh_chosen))
@printf "\n  Mean train Sharpe of chosen combo: %+.2f\n" mean_tr_chosen
@printf "  Mean test  Sharpe of chosen combo: %+.2f   (gap %+.2f)\n" mean_te_chosen (mean_te_chosen - mean_tr_chosen)

mean_tr_base = mean(f.train_sh_base for f in folds if !isnan(f.train_sh_base))
mean_te_base = mean(f.test_sh_base  for f in folds if !isnan(f.test_sh_base))
@printf "  Mean train Sharpe of baseline: %+.2f\n" mean_tr_base
@printf "  Mean test  Sharpe of baseline: %+.2f   (gap %+.2f)\n" mean_te_base (mean_te_base - mean_tr_base)

# IS→OOS rank correlation per fold
rhos = [f.spearman for f in folds if !isnan(f.spearman)]
@printf "\n  IS→OOS Spearman ρ per fold:  mean %+.3f   median %+.3f   N=%d\n" mean(rhos) median(rhos) length(rhos)
@printf "  Folds with ρ > 0:   %d / %d\n" count(>(0), rhos) length(rhos)
@printf "  Folds with ρ > 0.3: %d / %d\n" count(>(0.3), rhos) length(rhos)

# Where did baseline rank in IS?
baseline_ranks = [f.baseline_is_rank for f in folds if f.baseline_is_rank > 0]
@printf "\n  Baseline (%.2f, %.2f) IS rank distribution:  median %d  mean %.1f  (out of %d)\n" BASELINE_COMBO[1] BASELINE_COMBO[2] median(baseline_ranks) mean(baseline_ranks) n_combos
@printf "  In-sample top-5 contains baseline:  %d / %d  (%.1f%%)\n" count(<=(5), baseline_ranks) length(baseline_ranks) 100*count(<=(5), baseline_ranks)/length(baseline_ranks)

# Combo usage by rolling
combo_counts = Dict{Tuple{Float64,Float64},Int}()
for f in folds; combo_counts[f.chosen] = get(combo_counts, f.chosen, 0) + 1; end
println("\n  Top 8 combos picked by rolling:")
for (c, n) in first(sort(collect(combo_counts); by=x->x[2], rev=true), 8)
    @printf "    (%.2f, %.2f)  →  %d folds (%.1f%%)\n" c[1] c[2] n 100*n/length(folds)
end

# Per-fold detail
println("\n  ── Per-fold detail ──")
@printf "  %3s  %-23s  %-10s  %+8s  %+8s  %+8s  %+8s  %4s\n" "#" "test window" "chosen" "trS_ch" "teS_ch" "trS_b" "teS_b" "rkB"
for f in folds
    @printf "  %3d  %s → %s  (%.2f,%.2f)  %+8.2f  %+8.2f  %+8.2f  %+8.2f  %4d\n" f.idx f.test[1] f.test[2] f.chosen[1] f.chosen[2] f.train_sh_chosen f.test_sh_chosen f.train_sh_base f.test_sh_base f.baseline_is_rank
end

# Save CSV
csv_path = joinpath(run_dir, "fold_diagnostic.csv")
open(csv_path, "w") do io
    println(io, "fold,test_start,test_end,chosen_pd,chosen_cd,train_sh_chosen,test_sh_chosen,train_sh_base,test_sh_base,test_total_chosen,test_total_base,baseline_is_rank,spearman")
    for f in folds
        @printf io "%d,%s,%s,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%d,%.4f\n" f.idx f.test[1] f.test[2] f.chosen[1] f.chosen[2] f.train_sh_chosen f.test_sh_chosen f.train_sh_base f.test_sh_base f.test_total_chosen f.test_total_base f.baseline_is_rank f.spearman
    end
end
println("\n  Saved: fold_diagnostic.csv")
