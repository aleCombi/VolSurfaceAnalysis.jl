# scripts/condor_rolling.jl
#
# Unified rolling iron-condor experiment runner. Replaces 5 prior scripts via
# a MODE knob. Each MODE preserves the original logic; behavior is intentionally
# branched (not over-abstracted) so per-mode reporting matches the originals.
#
#   MODE=delta        — replaces condor_rolling_delta.jl
#                       Roll (put_delta, call_delta) with fixed wing (default \$12).
#                       Defaults: ENTRY=14:00, TENOR=2h, MAX_TAU=0.5d.
#
#   MODE=wing         — replaces condor_rolling_wing.jl
#                       Roll wing width with fixed (PUT_DELTA, CALL_DELTA)=(0.20, 0.05).
#                       Defaults: ENTRY=14:00, TENOR=2h, MAX_TAU=0.5d.
#
#   MODE=joint        — replaces condor_rolling_short_then_long_1d.jl
#                       Two-layer rolling: pick (Δ_p, Δ_c) at reference wing, then
#                       pick wing with chosen shorts fixed; both relayered every fold.
#                       Defaults: ENTRY=14:00, TENOR=1d, MAX_TAU=2.0d.
#
#   MODE=2stage       — replaces condor_2stage_delta_then_wing.jl
#                       Stage 1: pick best (Δ_p, Δ_c) on full IS window at reference wing.
#                       Stage 2: rolling-wing on OOS with that fixed Δ.
#                       Includes joint-fixed honest comparison.
#                       Defaults: ENTRY=14:00, TENOR=2h, MAX_TAU=0.5d, IS_END_YEAR=2020.
#
#   MODE=cross_tenor  — replaces condor_train_1d_apply_2h.jl
#                       Train rolling joint (Δ, wing) on TENOR_TRAIN, apply on TENOR_TEST.
#                       Defaults: TENOR_TRAIN=1d, TENOR_TEST=2h.
#
# Common ENV: SYM, START_YEAR, END_DATE, DIV, RATE, SPREAD_LAMBDA,
#             TRAIN_DAYS, TEST_DAYS, STEP_DAYS.

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, Plots
include(joinpath(@__DIR__, "lib", "experiment.jl"))

# =============================================================================
# Common configuration
# =============================================================================

MODE   = lowercase(get(ENV, "MODE", "delta"))
MODE in ("delta", "wing", "joint", "2stage", "cross_tenor") ||
    error("MODE must be delta|wing|joint|2stage|cross_tenor, got $MODE")

SYMBOL          = get(ENV, "SYM", "SPY")
START_DATE      = Date(parse(Int, get(ENV, "START_YEAR", "2017")), 1, 1)
END_DATE        = Date(get(ENV, "END_DATE", "2024-01-31"))
SPREAD_LAMBDA   = parse(Float64, get(ENV, "SPREAD_LAMBDA", "0.7"))
RATE            = parse(Float64, get(ENV, "RATE", "0.045"))
DIV_YIELD       = parse(Float64, get(ENV, "DIV", "0.013"))

TRAIN_DAYS = parse(Int, get(ENV, "TRAIN_DAYS", "90"))
TEST_DAYS  = parse(Int, get(ENV, "TEST_DAYS",  "30"))
STEP_DAYS  = parse(Int, get(ENV, "STEP_DAYS",  "30"))

ENTRY_TIME = Time(parse(Int, get(ENV, "ENTRY_HOUR", "14")), 0)

# Mode-specific tenor defaults
function _parse_tenor(s)
    s = lowercase(s)
    if endswith(s, "d"); return Day(parse(Int, s[1:end-1])); end
    if endswith(s, "h"); return Hour(parse(Int, s[1:end-1])); end
    error("Unknown tenor: $s")
end

if MODE == "joint"
    EXPIRY_INTERVAL = _parse_tenor(get(ENV, "TENOR", "1d"))
    MAX_TAU_DAYS    = parse(Float64, get(ENV, "MAX_TAU_DAYS", "2.0"))
else  # delta, wing, 2stage
    EXPIRY_INTERVAL = _parse_tenor(get(ENV, "TENOR", "2h"))
    MAX_TAU_DAYS    = parse(Float64, get(ENV, "MAX_TAU_DAYS", "0.5"))
end

PUT_DELTAS  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
CALL_DELTAS = [0.05, 0.10, 0.15, 0.20]
DELTA_COMBOS = [(pd, cd) for pd in PUT_DELTAS for cd in CALL_DELTAS]
WING_WIDTHS = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
SELECTION_WING_WIDTH = parse(Float64, get(ENV, "SELECTION_WING", "12.0"))

# Mode-specific deltas (for wing-only mode)
PUT_DELTA  = parse(Float64, get(ENV, "PUT_DELTA",  "0.20"))
CALL_DELTA = parse(Float64, get(ENV, "CALL_DELTA", "0.05"))

# 2stage
IN_SAMPLE_END = Date(parse(Int, get(ENV, "IS_END_YEAR", "2020")), 12, 31)

# cross_tenor
TENOR_TRAIN_STR = lowercase(get(ENV, "TENOR_TRAIN", "1d"))
TENOR_TEST_STR  = lowercase(get(ENV, "TENOR_TEST",  "2h"))
TENOR_TRAIN     = _parse_tenor(TENOR_TRAIN_STR)
TENOR_TEST      = _parse_tenor(TENOR_TEST_STR)
MAX_TAU_TRAIN   = parse(Float64, get(ENV, "MAX_TAU_TRAIN", endswith(TENOR_TRAIN_STR,"d") ? "2.0" : "0.5"))
MAX_TAU_TEST    = parse(Float64, get(ENV, "MAX_TAU_TEST",  endswith(TENOR_TEST_STR,"d")  ? "2.0" : "0.5"))
ENTRY_TRAIN     = Time(parse(Int, get(ENV, "ENTRY_HOUR_TRAIN", "14")), 0)
ENTRY_TEST      = Time(parse(Int, get(ENV, "ENTRY_HOUR_TEST",  "14")), 0)

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "condor_rolling_$(MODE)_$(SYMBOL)_$run_ts")
mkpath(run_dir)
println("Output: $run_dir   MODE=$MODE   $SYMBOL  $START_DATE → $END_DATE")
println("  train=$TRAIN_DAYS d / test=$TEST_DAYS d / step=$STEP_DAYS d")

# =============================================================================
# Common helpers
# =============================================================================

sharpe_with_nan(v::AbstractVector{<:Real}) = let c = filter(!isnan, v)
    isempty(c) ? -Inf : (std(c) > 0 ? mean(c)/std(c)*sqrt(252) : 0.0)
end
sharpe_of = sharpe_with_nan

function summary_for(pnls)
    clean = filter(!isnan, pnls)
    isempty(clean) && return (n=0, total=0.0, sharpe=0.0, mdd=0.0, avg=0.0)
    cum = cumsum(clean); rmax = accumulate(max, cum); mdd = -minimum(cum .- rmax)
    sh = std(clean) > 0 ? mean(clean)/std(clean)*sqrt(252) : 0.0
    return (n=length(clean), total=sum(clean), sharpe=sh, mdd=mdd, avg=mean(clean))
end

# Build a dataset where each entry has a Matrix[combo, wing] of Float64 PnL (NaN missing).
# Used by joint, 2stage, cross_tenor.
function build_combo_wing_dataset(symbol, start_date, end_date, entry_time, expiry_interval,
                                   max_tau, DELTA_COMBOS, WING_WIDTHS,
                                   RATE, DIV_YIELD, SPREAD_LAMBDA)
    (; source, sched) = polygon_parquet_source(symbol;
        start_date=start_date, end_date=end_date, entry_time=entry_time,
        rate=RATE, div_yield=DIV_YIELD, spread_lambda=SPREAD_LAMBDA,
    )

    n_combos = length(DELTA_COMBOS); n_wings = length(WING_WIDTHS)
    dates = Date[]; pnls = Vector{Matrix{Float64}}()
    n_total = Ref(0); n_skip = Ref(0)

    each_entry(source, expiry_interval, sched; clear_cache=true) do ctx, settlement
        ismissing(settlement) && return
        n_total[] += 1
        dctx = delta_context(ctx; rate=RATE, div_yield=DIV_YIELD)
        dctx === nothing && return
        dctx.tau * 365.25 > max_tau && (n_skip[] += 1; return)
        spot = dctx.spot

        M = fill(NaN, n_combos, n_wings)
        for (ci, (pd, cd)) in enumerate(DELTA_COMBOS)
            sp_K = delta_strike(dctx, -pd, Put)
            sc_K = delta_strike(dctx,  cd, Call)
            (sp_K === nothing || sc_K === nothing) && continue
            for (wi, ww) in enumerate(WING_WIDTHS)
                lp_K = nearest_otm_strike(dctx, sp_K, ww, Put)
                lc_K = nearest_otm_strike(dctx, sc_K, ww, Call)
                (lp_K === nothing || lc_K === nothing) && continue
                cp = open_condor_positions(ctx, sp_K, sc_K, lp_K, lc_K)
                length(cp) == 4 || continue
                M[ci, wi] = settle(cp, Float64(settlement)) * spot
            end
        end
        push!(dates, Date(ctx.surface.timestamp))
        push!(pnls, M)
    end
    ord = sortperm(dates)
    return dates[ord], pnls[ord], n_total[], n_skip[]
end

# =============================================================================
# MODE = delta : roll (put_delta, call_delta) with fixed wing
# =============================================================================

if MODE == "delta"
    WING_WIDTH = SELECTION_WING_WIDTH
    println("  Delta combos: $(length(DELTA_COMBOS))   wing fixed: \$$(WING_WIDTH)")

    println("\nLoading $SYMBOL …")
    (; source, sched) = polygon_parquet_source(SYMBOL;
        start_date=START_DATE, end_date=END_DATE, entry_time=ENTRY_TIME,
        rate=RATE, div_yield=DIV_YIELD, spread_lambda=SPREAD_LAMBDA,
    )

    dates  = Date[]
    pnls_by_combo = Vector{Vector{Float64}}()
    n_total = 0; n_skip = 0

    println("\nBuilding dataset (per-entry PnL × $(length(DELTA_COMBOS)) combos)...")
    each_entry(source, EXPIRY_INTERVAL, sched; clear_cache=true) do ctx, settlement
        ismissing(settlement) && return
        global n_total += 1
        dctx = delta_context(ctx; rate=RATE, div_yield=DIV_YIELD)
        dctx === nothing && (global n_skip += 1; return)
        dctx.tau * 365.25 > MAX_TAU_DAYS && (global n_skip += 1; return)
        spot = dctx.spot

        ps = Float64[]
        for (pd, cd) in DELTA_COMBOS
            sp_K = delta_strike(dctx, -pd, Put)
            sc_K = delta_strike(dctx,  cd, Call)
            if sp_K === nothing || sc_K === nothing
                push!(ps, NaN); continue
            end
            lp_K = nearest_otm_strike(dctx, sp_K, WING_WIDTH, Put)
            lc_K = nearest_otm_strike(dctx, sc_K, WING_WIDTH, Call)
            (lp_K === nothing || lc_K === nothing) && (push!(ps, NaN); continue)
            cp = open_condor_positions(ctx, sp_K, sc_K, lp_K, lc_K)
            length(cp) == 4 || (push!(ps, NaN); continue)
            push!(ps, settle(cp, Float64(settlement)) * spot)
        end
        push!(dates, Date(ctx.surface.timestamp))
        push!(pnls_by_combo, ps)
    end
    n_kept = length(dates)
    @printf "  %d entries → kept %d  (skipped %d)\n" n_total n_kept n_skip
    n_kept < 50 && error("Too few entries")

    ord = sortperm(dates); dates = dates[ord]; pnls_by_combo = pnls_by_combo[ord]
    PnL = hcat(pnls_by_combo...)'
    n_combos = length(DELTA_COMBOS)

    oos_pnls   = Float64[]; oos_dates = Date[]; oos_combos = Tuple{Float64,Float64}[]
    fold_choices = NamedTuple[]
    folds_meta = build_folds(dates;
        train_days=TRAIN_DAYS, test_days=TEST_DAYS, step_days=STEP_DAYS,
        min_train=10, min_test=1,
    )
    for f in folds_meta
        tr_mask = f.train_mask; te_mask = f.test_mask
        train_sharpes = [sharpe_with_nan(PnL[tr_mask, c]) for c in 1:n_combos]
        best_c = argmax(train_sharpes)
        chosen = DELTA_COMBOS[best_c]
        test_pnls = filter(!isnan, PnL[te_mask, best_c])
        test_dates_w = dates[te_mask][.!isnan.(PnL[te_mask, best_c])]
        append!(oos_pnls, test_pnls); append!(oos_dates, test_dates_w)
        append!(oos_combos, fill(chosen, length(test_pnls)))
        push!(fold_choices, (idx=f.idx, test=(f.test_start, f.test_end),
            n_tr=sum(tr_mask), n_te=sum(te_mask), chosen=chosen,
            train_sharpe=train_sharpes[best_c],
            test_sharpe=length(test_pnls) > 1 ? mean(test_pnls)/(std(test_pnls)+1e-12)*sqrt(252) : 0.0,
            test_total=sum(test_pnls)))
    end

    r = summary_for(oos_pnls)
    println("\n", "=" ^ 80)
    println("  RESULT — rolling-delta-combo OOS")
    println("=" ^ 80)
    @printf "  trades=%d   total=%+.0f   Sharpe=%+.2f   MaxDD=%.0f\n" r.n r.total r.sharpe r.mdd

    println("\n  Combo usage (top 10):")
    combo_counts = Dict{Tuple{Float64,Float64},Int}()
    for c in oos_combos; combo_counts[c] = get(combo_counts, c, 0) + 1; end
    for (c, n) in first(sort(collect(combo_counts); by=x->x[2], rev=true), min(10, length(combo_counts)))
        @printf "    (%.2f, %.2f)  →  %d trades (%.1f%%)\n" c[1] c[2] n 100*n/length(oos_combos)
    end

    println("\n  Reference: fixed-combo Sharpe over the OOS period")
    @printf "    %-7s  %-7s  %5s  %+8s  %+10s  %8s\n" "putΔ" "callΔ" "n" "Sharpe" "total" "MaxDD"
    println("    " * "─"^48)
    oos_idx = findall(d -> d in Set(oos_dates), dates)
    oos_pnl_matrix = PnL[oos_idx, :]
    combo_summaries = []
    for (i, (pd, cd)) in enumerate(DELTA_COMBOS)
        push!(combo_summaries, (pd=pd, cd=cd, summary=summary_for(oos_pnl_matrix[:, i])))
    end
    sort!(combo_summaries; by=x->x.summary.sharpe, rev=true)
    for (i, c) in enumerate(combo_summaries)
        i > 8 && break
        s = c.summary
        @printf "    %-7.2f  %-7.2f  %5d  %+8.2f  %+10.0f  %8.0f\n" c.pd c.cd s.n s.sharpe s.total s.mdd
    end

    println("\n  ── Per-month Sharpe (rolling-delta OOS) ──")
    @printf "  %-9s  %5s  %+10s  %+8s\n" "month" "n" "totalPnL" "Sharpe"
    println("  " * "─"^46)
    for m in sort(unique(yearmonth.(oos_dates)))
        mask = [ym == m for ym in yearmonth.(oos_dates)]
        ys = oos_pnls[mask]; n = length(ys); n < 3 && continue
        sh = std(ys) > 0 ? mean(ys)/std(ys)*sqrt(252) : 0.0
        @printf "  %4d-%02d   %5d  %+10.0f  %+8.2f\n" m[1] m[2] n sum(ys) sh
    end

    ord_d = sortperm(oos_dates)
    p = plot(oos_dates[ord_d], cumsum(oos_pnls[ord_d]);
        xlabel="date", ylabel="cumulative PnL (USD)",
        title="$SYMBOL $(EXPIRY_INTERVAL) iron condor — rolling delta selection (wing=\$$WING_WIDTH)",
        label="rolling-delta", lw=2, color=:steelblue, size=(1100, 500))
    ref_idx = findfirst(==((0.20, 0.05)), DELTA_COMBOS)
    if ref_idx !== nothing
        fixed_pnls = filter(!isnan, PnL[oos_idx, ref_idx])
        fixed_dates_keep = dates[oos_idx][.!isnan.(PnL[oos_idx, ref_idx])]
        ord_f = sortperm(fixed_dates_keep)
        plot!(p, fixed_dates_keep[ord_f], cumsum(fixed_pnls[ord_f]);
            label="fixed 20p/5c", lw=2, color=:darkorange, ls=:dash)
    end
    hline!(p, [0]; color=:gray, ls=:dash, label="")
    savefig(p, joinpath(run_dir, "cumulative.png"))
    println("\n  Saved: cumulative.png")
end

# =============================================================================
# MODE = wing : roll wing width with fixed shorts
# =============================================================================

if MODE == "wing"
    println("  Wing widths: $WING_WIDTHS   shorts fixed: ($PUT_DELTA, $CALL_DELTA)")

    println("\nLoading $SYMBOL …")
    (; source, sched) = polygon_parquet_source(SYMBOL;
        start_date=START_DATE, end_date=END_DATE, entry_time=ENTRY_TIME,
        rate=RATE, div_yield=DIV_YIELD, spread_lambda=SPREAD_LAMBDA,
    )

    dates  = Date[]
    pnls_by_width = Vector{Vector{Float64}}()
    n_total = 0; n_skip = 0

    println("\nBuilding dataset (per-entry PnL × $(length(WING_WIDTHS)) widths)...")
    each_entry(source, EXPIRY_INTERVAL, sched; clear_cache=true) do ctx, settlement
        ismissing(settlement) && return
        global n_total += 1
        dctx = delta_context(ctx; rate=RATE, div_yield=DIV_YIELD)
        dctx === nothing && (global n_skip += 1; return)
        dctx.tau * 365.25 > MAX_TAU_DAYS && (global n_skip += 1; return)
        spot = dctx.spot

        sp_K = delta_strike(dctx, -PUT_DELTA,  Put)
        sc_K = delta_strike(dctx,  CALL_DELTA, Call)
        (sp_K === nothing || sc_K === nothing) && (global n_skip += 1; return)

        ps = Float64[]; ok = true
        for ww in WING_WIDTHS
            lp_K = nearest_otm_strike(dctx, sp_K, ww, Put)
            lc_K = nearest_otm_strike(dctx, sc_K, ww, Call)
            (lp_K === nothing || lc_K === nothing) && (ok = false; break)
            cp = open_condor_positions(ctx, sp_K, sc_K, lp_K, lc_K)
            length(cp) == 4 || (ok = false; break)
            push!(ps, settle(cp, Float64(settlement)) * spot)
        end
        ok || (global n_skip += 1; return)
        push!(dates, Date(ctx.surface.timestamp))
        push!(pnls_by_width, ps)
    end
    n_kept = length(dates)
    @printf "  %d entries → kept %d  (skipped %d)\n" n_total n_kept n_skip
    n_kept < 50 && error("Too few entries")

    ord = sortperm(dates); dates = dates[ord]; pnls_by_width = pnls_by_width[ord]
    PnL = hcat(pnls_by_width...)'
    n_widths = length(WING_WIDTHS)

    oos_pnls = Float64[]; oos_dates = Date[]; oos_wings = Float64[]
    fold_choices = NamedTuple[]
    test_start = dates[1] + Day(TRAIN_DAYS); last_d = dates[end]; fold_idx = 0
    while test_start <= last_d
        te_end = test_start + Day(TEST_DAYS) - Day(1)
        tr_start = test_start - Day(TRAIN_DAYS); tr_end = test_start - Day(1)
        tr_mask = (dates .>= tr_start) .& (dates .<= tr_end)
        te_mask = (dates .>= test_start) .& (dates .<= te_end)
        n_tr = sum(tr_mask); n_te = sum(te_mask)
        if n_tr < 10 || n_te < 1
            test_start += Day(STEP_DAYS); continue
        end
        train_sharpes = [(s = std(filter(!isnan, PnL[tr_mask, w])); s > 0 ? mean(filter(!isnan, PnL[tr_mask, w]))/s*sqrt(252) : 0.0)
                         for w in 1:n_widths]
        best_w = argmax(train_sharpes)
        chosen_wing = WING_WIDTHS[best_w]
        test_pnls = PnL[te_mask, best_w]
        test_dates_w = dates[te_mask]
        append!(oos_pnls, test_pnls); append!(oos_dates, test_dates_w)
        append!(oos_wings, fill(chosen_wing, length(test_pnls)))
        fold_idx += 1
        push!(fold_choices, (idx=fold_idx, train=(tr_start, tr_end), test=(test_start, te_end),
            n_tr=n_tr, n_te=n_te, chosen_wing=chosen_wing,
            train_sharpe=train_sharpes[best_w],
            test_sharpe=(s = std(test_pnls); s > 0 ? mean(test_pnls)/s*sqrt(252) : 0.0),
            test_total=sum(test_pnls)))
        test_start += Day(STEP_DAYS)
    end

    println("\nFolds: $(length(fold_choices))")
    println("\n  ── Fold-by-fold rolling-wing choice ──")
    @printf "  %-3s  %-23s  %4s  %4s  %5s  %+8s  %+8s  %+8s\n" "f" "test_window" "n_tr" "n_te" "wing" "tr_Sh" "te_Sh" "te_USD"
    println("  " * "─"^85)
    for r in fold_choices
        @printf "  %-3d  %s → %s  %4d  %4d  %5.1f  %+8.2f  %+8.2f  %+8.0f\n" r.idx r.test[1] r.test[2] r.n_tr r.n_te r.chosen_wing r.train_sharpe r.test_sharpe r.test_total
    end

    n_oos = length(oos_pnls); total = sum(oos_pnls)
    sh_oos = (s = std(oos_pnls); s > 0 ? mean(oos_pnls)/s*sqrt(252) : 0.0)
    mdd = let cum = cumsum(oos_pnls); rmax = accumulate(max, cum); -minimum(cum .- rmax); end
    println("\n  ── Aggregate OOS (rolling-wing) ──")
    @printf "  trades = %d   total = %+.0f   AvgPnL = %+.2f   Sharpe = %+.2f   MaxDD = %.0f\n" n_oos total mean(oos_pnls) sh_oos mdd

    println("\n  Wing usage:")
    for w in WING_WIDTHS
        n = count(==(w), oos_wings)
        @printf "    wing %.1f → %d trades (%.0f%%)\n" w n 100*n/n_oos
    end

    println("\n  ── Per-month Sharpe (rolling-wing OOS) ──")
    @printf "  %-9s  %5s  %+10s  %+8s\n" "month" "n" "totalPnL" "Sharpe"
    println("  " * "─"^46)
    for m in sort(unique(yearmonth.(oos_dates)))
        mask = [ym == m for ym in yearmonth.(oos_dates)]
        ys = oos_pnls[mask]; n = length(ys); n < 3 && continue
        s = std(ys); @printf "  %4d-%02d   %5d  %+10.0f  %+8.2f\n" m[1] m[2] n sum(ys) (s > 0 ? mean(ys)/s*sqrt(252) : 0.0)
    end

    csv_path = joinpath(run_dir, "oos_series.csv")
    open(csv_path, "w") do io
        println(io, "date,chosen_wing,oos_pnl_usd")
        for i in eachindex(oos_dates)
            @printf io "%s,%.1f,%.6f\n" oos_dates[i] oos_wings[i] oos_pnls[i]
        end
    end
    println("\n  Saved: $csv_path  ($(length(oos_dates)) rows)")

    ord_d = sortperm(oos_dates)
    p1 = plot(oos_dates[ord_d], cumsum(oos_pnls[ord_d]);
        xlabel="date", ylabel="cumulative PnL (USD)",
        title="$SYMBOL $(EXPIRY_INTERVAL) iron condor — rolling wing selection",
        label="OOS cumulative", lw=2, color=:steelblue, size=(1100, 500))
    hline!(p1, [0]; color=:gray, ls=:dash, label="")
    savefig(p1, joinpath(run_dir, "cumulative.png"))

    fold_dates = [r.test[1] for r in fold_choices]
    fold_wings = [r.chosen_wing for r in fold_choices]
    p2 = scatter(fold_dates, fold_wings;
        xlabel="test-window start", ylabel="chosen wing width",
        title="$SYMBOL — wing chosen per test fold",
        ms=8, color=:darkorange, label="", size=(1100, 350))
    savefig(p2, joinpath(run_dir, "wing_over_time.png"))
    println("\n  Saved: cumulative.png, wing_over_time.png")
end

# =============================================================================
# MODE = joint : two-layer rolling Δ then wing (per fold)
# =============================================================================

if MODE == "joint"
    println("  Δ grid: $(length(DELTA_COMBOS))   wing grid: $(length(WING_WIDTHS))")
    println("\nLoading $SYMBOL …")

    dates, pnls_by_entry, t1, s1 = build_combo_wing_dataset(
        SYMBOL, START_DATE, END_DATE, ENTRY_TIME, EXPIRY_INTERVAL, MAX_TAU_DAYS,
        DELTA_COMBOS, WING_WIDTHS, RATE, DIV_YIELD, SPREAD_LAMBDA)
    @printf "  %d total → kept %d, skipped %d\n" t1 length(dates) s1
    length(dates) < 50 && error("Too few entries")

    sel_wi = findfirst(==(SELECTION_WING_WIDTH), WING_WIDTHS)
    sel_wi === nothing && error("SELECTION_WING_WIDTH=$SELECTION_WING_WIDTH not in WING_WIDTHS")

    n_combos = length(DELTA_COMBOS); n_wings = length(WING_WIDTHS)
    oos_pnls = Float64[]; oos_dates = Date[]
    oos_combos = Tuple{Float64,Float64}[]; oos_wings = Float64[]
    folds = NamedTuple[]

    test_start = dates[1] + Day(TRAIN_DAYS); last_d = dates[end]; fidx = 0
    while test_start <= last_d
        te_end = test_start + Day(TEST_DAYS) - Day(1)
        tr_start = test_start - Day(TRAIN_DAYS); tr_end = test_start - Day(1)
        tr_idx = findall(d -> tr_start <= d <= tr_end, dates)
        te_idx = findall(d -> test_start <= d <= te_end, dates)
        if length(tr_idx) < 10 || length(te_idx) < 1
            test_start += Day(STEP_DAYS); continue
        end

        combo_sharpes = [sharpe_of([pnls_by_entry[i][ci, sel_wi] for i in tr_idx]) for ci in 1:n_combos]
        best_ci = argmax(combo_sharpes); chosen_combo = DELTA_COMBOS[best_ci]

        wing_sharpes = [sharpe_of([pnls_by_entry[i][best_ci, wi] for i in tr_idx]) for wi in 1:n_wings]
        best_wi = argmax(wing_sharpes); chosen_wing = WING_WIDTHS[best_wi]

        for i in te_idx
            v = pnls_by_entry[i][best_ci, best_wi]
            isnan(v) && continue
            push!(oos_pnls, v); push!(oos_dates, dates[i])
            push!(oos_combos, chosen_combo); push!(oos_wings, chosen_wing)
        end
        fidx += 1
        push!(folds, (idx=fidx, test=(test_start, te_end),
            n_tr=length(tr_idx), n_te=length(te_idx),
            chosen_combo=chosen_combo, chosen_wing=chosen_wing,
            train_sharpe_A=combo_sharpes[best_ci], train_sharpe_B=wing_sharpes[best_wi]))
        test_start += Day(STEP_DAYS)
    end

    r = summary_for(oos_pnls)
    println("\n", "=" ^ 72)
    println("  OOS RESULT — rolling Δ then rolling wing")
    println("=" ^ 72)
    @printf "  trades=%d  total=%+.0f  AvgPnL=%+.2f  Sharpe=%+.2f  MaxDD=%.0f\n" r.n r.total r.avg r.sharpe r.mdd

    println("\n  Δ combo usage (top 8):")
    combo_counts = Dict{Tuple{Float64,Float64},Int}()
    for c in oos_combos; combo_counts[c] = get(combo_counts, c, 0) + 1; end
    for (c, n) in first(sort(collect(combo_counts); by=x->x[2], rev=true), min(8, length(combo_counts)))
        @printf "    (%.2f, %.2f)  →  %d trades (%.1f%%)\n" c[1] c[2] n 100*n/length(oos_combos)
    end

    println("\n  Wing usage:")
    for w in WING_WIDTHS
        n = count(==(w), oos_wings); n == 0 && continue
        @printf "    \$%.1f  →  %d trades  (%.1f%%)\n" w n 100*n/length(oos_wings)
    end

    # Reference + oracle
    println("\n  Reference over OOS period:")
    oos_idx_full = findall(d -> d in Set(oos_dates), dates)
    @printf "    %-30s  %5s  %+10s  %+8s\n" "strategy" "n" "total" "Sharpe"
    base_ci = findfirst(==((0.20, 0.05)), DELTA_COMBOS)
    ref_2010_12 = filter(!isnan, [pnls_by_entry[i][base_ci, sel_wi] for i in oos_idx_full])
    sr_2010 = summary_for(ref_2010_12)
    @printf "    %-30s  %5d  %+10.0f  %+8.2f\n" "fixed 20p/5c, wing=\$$(Int(SELECTION_WING_WIDTH))" sr_2010.n sr_2010.total sr_2010.sharpe

    best_sh = -Inf; best_label = ""; best_sum = nothing
    for (ci, (pd, cd)) in enumerate(DELTA_COMBOS), (wi, ww) in enumerate(WING_WIDTHS)
        v = filter(!isnan, [pnls_by_entry[i][ci, wi] for i in oos_idx_full])
        s = summary_for(v)
        if s.sharpe > best_sh
            best_sh = s.sharpe
            best_label = "ORACLE Δ=($(pd),$(cd)) wing=\$$(Int(ww))"
            best_sum = s
        end
    end
    @printf "    %-30s  %5d  %+10.0f  %+8.2f\n" best_label best_sum.n best_sum.total best_sum.sharpe

    println("\n  ── Per-fold choices ──")
    @printf "  %3s  %-23s  %4s  %4s  %-10s  %-5s  %+8s  %+8s\n" "#" "test window" "ntr" "nte" "Δ" "wing" "trS_A" "trS_B"
    for f in folds
        @printf "  %3d  %s → %s  %4d  %4d  (%.2f,%.2f)  \$%-4.0f  %+8.2f  %+8.2f\n" f.idx f.test[1] f.test[2] f.n_tr f.n_te f.chosen_combo[1] f.chosen_combo[2] f.chosen_wing f.train_sharpe_A f.train_sharpe_B
    end

    println("\n  ── Per-month Sharpe (OOS) ──")
    @printf "  %-9s  %5s  %+10s  %+8s\n" "month" "n" "totalPnL" "Sharpe"
    println("  " * "─"^46)
    for m in sort(unique(yearmonth.(oos_dates)))
        mask = [ym == m for ym in yearmonth.(oos_dates)]
        ys = oos_pnls[mask]; n = length(ys); n < 3 && continue
        sh = std(ys) > 0 ? mean(ys)/std(ys)*sqrt(252) : 0.0
        @printf "  %4d-%02d   %5d  %+10.0f  %+8.2f\n" m[1] m[2] n sum(ys) sh
    end

    ord_d = sortperm(oos_dates)
    p = plot(oos_dates[ord_d], cumsum(oos_pnls[ord_d]);
        xlabel="date", ylabel="cumulative PnL (USD)",
        title="$SYMBOL $(EXPIRY_INTERVAL) — rolling Δ then rolling wing",
        label="rolling Δ + rolling wing", lw=2, color=:steelblue, size=(1100, 500))
    hline!(p, [0]; color=:gray, ls=:dash, label="")
    savefig(p, joinpath(run_dir, "cumulative.png"))
    println("\n  Saved: cumulative.png")
end

# =============================================================================
# MODE = 2stage : Stage 1 IS-fix delta, Stage 2 rolling wing on OOS
# =============================================================================

if MODE == "2stage"
    println("  in-sample $START_DATE → $IN_SAMPLE_END   OOS $(IN_SAMPLE_END + Day(1)) → $END_DATE")

    # 2stage uses a finer Δ grid (matches original)
    PD_2STAGE = collect(0.050:0.025:0.300)
    CD_2STAGE = collect(0.025:0.025:0.225)
    DELTA_COMBOS_2S = [(pd, cd) for pd in PD_2STAGE for cd in CD_2STAGE]
    n_combos = length(DELTA_COMBOS_2S); n_wings = length(WING_WIDTHS)

    println("\nLoading $SYMBOL …")
    dates, pnls_by_entry, t1, s1 = build_combo_wing_dataset(
        SYMBOL, START_DATE, END_DATE, ENTRY_TIME, EXPIRY_INTERVAL, MAX_TAU_DAYS,
        DELTA_COMBOS_2S, WING_WIDTHS, RATE, DIV_YIELD, SPREAD_LAMBDA)
    @printf "  %d total → kept %d, skipped %d\n" t1 length(dates) s1
    length(dates) < 50 && error("Too few entries")

    is_idx = findall(<=(IN_SAMPLE_END), dates)
    oos_idx = findall(>(IN_SAMPLE_END), dates)
    println("\n  In-sample entries: $(length(is_idx))   OOS entries: $(length(oos_idx))")

    sel_wi = findfirst(==(SELECTION_WING_WIDTH), WING_WIDTHS)
    combo_sharpes = [sharpe_of([pnls_by_entry[i][ci, sel_wi] for i in is_idx]) for ci in 1:n_combos]
    best_ci = argmax(combo_sharpes)
    chosen_pd, chosen_cd = DELTA_COMBOS_2S[best_ci]
    println("\n  ── Stage 1 ──")
    println("  Top 8 in-sample combos (wing fixed at \$$SELECTION_WING_WIDTH):")
    @printf "    %-7s  %-7s  %+8s\n" "putΔ" "callΔ" "Sharpe"
    top_is = sort(collect(zip(DELTA_COMBOS_2S, combo_sharpes)); by=x->x[2], rev=true)
    for ((pd, cd), sh) in top_is[1:min(8, end)]
        @printf "    %-7.2f  %-7.2f  %+8.2f\n" pd cd sh
    end
    println("\n  → Selected: putΔ=$chosen_pd  callΔ=$chosen_cd")

    oos_dates_full = dates[oos_idx]
    PnL_oos = Matrix{Float64}(undef, length(oos_idx), n_wings)
    for (i, idx) in enumerate(oos_idx)
        PnL_oos[i, :] = pnls_by_entry[idx][best_ci, :]
    end

    # Rolling wing on OOS
    oos_p2 = Float64[]; oos_d2 = Date[]; oos_w2 = Float64[]; folds2 = NamedTuple[]
    test_start = oos_dates_full[1] + Day(TRAIN_DAYS); last_d = oos_dates_full[end]; fold_idx = 0
    while test_start <= last_d
        te_end = test_start + Day(TEST_DAYS) - Day(1)
        tr_start = test_start - Day(TRAIN_DAYS); tr_end = test_start - Day(1)
        tr_mask = (oos_dates_full .>= tr_start) .& (oos_dates_full .<= tr_end)
        te_mask = (oos_dates_full .>= test_start) .& (oos_dates_full .<= te_end)
        n_tr = sum(tr_mask); n_te = sum(te_mask)
        if n_tr < 10 || n_te < 1
            test_start += Day(STEP_DAYS); continue
        end
        train_sharpes = [sharpe_of(PnL_oos[tr_mask, w]) for w in 1:n_wings]
        best_w = argmax(train_sharpes)
        chosen_w = WING_WIDTHS[best_w]
        test_pnls = filter(!isnan, PnL_oos[te_mask, best_w])
        test_dates_w = oos_dates_full[te_mask][.!isnan.(PnL_oos[te_mask, best_w])]
        append!(oos_p2, test_pnls); append!(oos_d2, test_dates_w)
        append!(oos_w2, fill(chosen_w, length(test_pnls)))
        fold_idx += 1
        push!(folds2, (idx=fold_idx, test=(test_start, te_end), n_tr=n_tr, n_te=n_te,
            chosen=chosen_w, train_sharpe=train_sharpes[best_w]))
        test_start += Day(STEP_DAYS)
    end

    println("\n  ── Stage 2: rolling wing on OOS, fixed shorts (putΔ=$chosen_pd, callΔ=$chosen_cd) ──")
    println("\nFolds: $(length(folds2))")
    r2 = summary_for(oos_p2)
    println("\n", "=" ^ 70)
    println("  OOS RESULT — Stage 2 (rolling wing, fixed best delta from Stage 1)")
    println("=" ^ 70)
    @printf "  trades=%d  total=%+.0f  AvgPnL=%+.2f  Sharpe=%+.2f  MaxDD=%.0f\n" r2.n r2.total r2.avg r2.sharpe r2.mdd

    println("\n  Wing usage on OOS:")
    for w in WING_WIDTHS
        n = count(==(w), oos_w2); pct = 100 * n / length(oos_w2)
        @printf "    \$%.1f  →  %d trades  (%.1f%%)\n" w n pct
    end

    # Reference + honest fully-fixed
    println("\n  Reference — fixed strategies over OOS:")
    @printf "    %-40s  %5s  %+10s  %+8s\n" "strategy" "n" "total" "Sharpe"
    base_ci_2010 = findfirst(==((0.20, 0.05)), DELTA_COMBOS_2S)
    if base_ci_2010 !== nothing
        ref_pnl_2010 = filter(!isnan, [pnls_by_entry[i][base_ci_2010, sel_wi] for i in oos_idx])
        sr1 = summary_for(ref_pnl_2010)
        @printf "    %-40s  %5d  %+10.0f  %+8.2f\n" "fixed 20p/5c, wing=\$$(Int(SELECTION_WING_WIDTH)) (no search)" sr1.n sr1.total sr1.sharpe
    end
    ref_pnl_chosen = filter(!isnan, [pnls_by_entry[i][best_ci, sel_wi] for i in oos_idx])
    sr2 = summary_for(ref_pnl_chosen)
    @printf "    %-40s  %5d  %+10.0f  %+8.2f\n" "Δ-search only, wing=\$$(Int(SELECTION_WING_WIDTH))" sr2.n sr2.total sr2.sharpe

    println("\n  ── Honest fully-fixed (joint in-sample search over $(n_combos*n_wings) (Δ, wing) combos) ──")
    joint_sharpes = fill(-Inf, n_combos, n_wings)
    for ci in 1:n_combos, wi in 1:n_wings
        joint_sharpes[ci, wi] = sharpe_of([pnls_by_entry[i][ci, wi] for i in is_idx])
    end
    best_lin = argmax(joint_sharpes)
    best_ci_j, best_wi_j = best_lin.I
    best_pd_j, best_cd_j = DELTA_COMBOS_2S[best_ci_j]
    best_ww_j = WING_WIDTHS[best_wi_j]
    @printf "  In-sample winner: putΔ=%.2f  callΔ=%.2f  wing=\$%.0f   IS Sharpe=%+.2f\n" best_pd_j best_cd_j best_ww_j joint_sharpes[best_lin]

    println("  Top 8 in-sample (Δ, wing) combos:")
    @printf "    %-7s  %-7s  %-6s  %+8s\n" "putΔ" "callΔ" "wing" "Sharpe"
    flat = [((DELTA_COMBOS_2S[ci], WING_WIDTHS[wi]), joint_sharpes[ci, wi]) for ci in 1:n_combos, wi in 1:n_wings]
    flat_sorted = sort(vec(flat); by=x->x[2], rev=true)
    for (((pd, cd), ww), sh) in flat_sorted[1:min(8, end)]
        @printf "    %-7.2f  %-7.2f  %-6.0f  %+8.2f\n" pd cd ww sh
    end

    oos_fixed = filter(!isnan, [pnls_by_entry[i][best_ci_j, best_wi_j] for i in oos_idx])
    sr_fixed = summary_for(oos_fixed)
    println()
    @printf "  OOS fully-fixed (Δ=%.2f/%.2f, wing=\$%.0f): n=%d  total=%+.0f  Sharpe=%+.2f  MaxDD=%.0f\n" best_pd_j best_cd_j best_ww_j sr_fixed.n sr_fixed.total sr_fixed.sharpe sr_fixed.mdd

    println("\n  ── Honest comparison ──")
    @printf "    %-50s  %+8s\n" "variant" "Sharpe"
    @printf "    %-50s  %+8.2f\n" "Stage 2 (Δ in-sample + rolling wing OOS)" r2.sharpe
    @printf "    %-50s  %+8.2f\n" "Honest fully-fixed (joint Δ+wing in-sample)" sr_fixed.sharpe

    println("\n  ── Per-month Sharpe (Stage 2 OOS) ──")
    @printf "  %-9s  %5s  %+10s  %+8s\n" "month" "n" "totalPnL" "Sharpe"
    println("  " * "─"^46)
    for m in sort(unique(yearmonth.(oos_d2)))
        mask = [ym == m for ym in yearmonth.(oos_d2)]
        ys = oos_p2[mask]; n = length(ys); n < 3 && continue
        sh = std(ys) > 0 ? mean(ys)/std(ys)*sqrt(252) : 0.0
        @printf "  %4d-%02d   %5d  %+10.0f  %+8.2f\n" m[1] m[2] n sum(ys) sh
    end

    ord_d = sortperm(oos_d2)
    p = plot(oos_d2[ord_d], cumsum(oos_p2[ord_d]);
        xlabel="date", ylabel="cumulative PnL (USD)",
        title="$SYMBOL $(EXPIRY_INTERVAL) — Stage 2 rolling wing (Δ chosen IS $START_DATE→$IN_SAMPLE_END)",
        label="Stage 2 (rolling wing on chosen Δ)", lw=2, color=:steelblue, size=(1100, 500))
    hline!(p, [0]; color=:gray, ls=:dash, label="")
    savefig(p, joinpath(run_dir, "cumulative.png"))
    println("\n  Saved: cumulative.png")
end

# =============================================================================
# MODE = cross_tenor : train on TENOR_TRAIN, apply on TENOR_TEST
# =============================================================================

if MODE == "cross_tenor"
    println("  train tenor=$TENOR_TRAIN_STR (entry $ENTRY_TRAIN, max_tau=$MAX_TAU_TRAIN d)")
    println("  test  tenor=$TENOR_TEST_STR  (entry $ENTRY_TEST, max_tau=$MAX_TAU_TEST d)")

    println("\n[1/2] Building train-tenor dataset...")
    dates_train, pnls_train, t1, s1 = build_combo_wing_dataset(
        SYMBOL, START_DATE, END_DATE, ENTRY_TRAIN, TENOR_TRAIN, MAX_TAU_TRAIN,
        DELTA_COMBOS, WING_WIDTHS, RATE, DIV_YIELD, SPREAD_LAMBDA)
    @printf "  train: %d total, kept %d, skipped %d\n" t1 length(dates_train) s1

    println("\n[2/2] Building test-tenor dataset...")
    dates_apply, pnls_apply, t2, s2 = build_combo_wing_dataset(
        SYMBOL, START_DATE, END_DATE, ENTRY_TEST, TENOR_TEST, MAX_TAU_TEST,
        DELTA_COMBOS, WING_WIDTHS, RATE, DIV_YIELD, SPREAD_LAMBDA)
    @printf "  test:  %d total, kept %d, skipped %d\n" t2 length(dates_apply) s2

    n_combos = length(DELTA_COMBOS); n_wings = length(WING_WIDTHS)
    oos_pnls = Float64[]; oos_dates = Date[]
    oos_combos = Tuple{Float64,Float64}[]; oos_wings = Float64[]
    folds = NamedTuple[]

    test_start = max(dates_train[1], dates_apply[1]) + Day(TRAIN_DAYS)
    last_d = min(dates_train[end], dates_apply[end])
    fidx = 0
    while test_start <= last_d
        te_end = test_start + Day(TEST_DAYS) - Day(1)
        tr_start = test_start - Day(TRAIN_DAYS); tr_end = test_start - Day(1)
        tr_idx_train = findall(d -> tr_start <= d <= tr_end, dates_train)
        te_idx_apply = findall(d -> test_start <= d <= te_end, dates_apply)
        if length(tr_idx_train) < 10 || length(te_idx_apply) < 1
            test_start += Day(STEP_DAYS); continue
        end

        best_sh = -Inf; best_ci = 1; best_wi = 1
        for ci in 1:n_combos, wi in 1:n_wings
            sh = sharpe_of([pnls_train[i][ci, wi] for i in tr_idx_train])
            if sh > best_sh; best_sh = sh; best_ci = ci; best_wi = wi; end
        end
        chosen_combo = DELTA_COMBOS[best_ci]; chosen_wing = WING_WIDTHS[best_wi]

        for i in te_idx_apply
            v = pnls_apply[i][best_ci, best_wi]
            isnan(v) && continue
            push!(oos_pnls, v); push!(oos_dates, dates_apply[i])
            push!(oos_combos, chosen_combo); push!(oos_wings, chosen_wing)
        end
        fidx += 1
        push!(folds, (idx=fidx, test=(test_start, te_end),
            n_tr=length(tr_idx_train), n_te=length(te_idx_apply),
            chosen_combo=chosen_combo, chosen_wing=chosen_wing,
            train_sharpe=best_sh))
        test_start += Day(STEP_DAYS)
    end

    r = summary_for(oos_pnls)
    println("\n", "=" ^ 72)
    println("  OOS RESULT — train on $TENOR_TRAIN_STR, apply on $TENOR_TEST_STR")
    println("=" ^ 72)
    @printf "  trades=%d  total=%+.0f  AvgPnL=%+.2f  Sharpe=%+.2f  MaxDD=%.0f\n" r.n r.total r.avg r.sharpe r.mdd

    println("\n  Δ combo usage (top 8):")
    combo_counts = Dict{Tuple{Float64,Float64},Int}()
    for c in oos_combos; combo_counts[c] = get(combo_counts, c, 0) + 1; end
    for (c, n) in first(sort(collect(combo_counts); by=x->x[2], rev=true), min(8, length(combo_counts)))
        @printf "    (%.2f, %.2f)  →  %d trades (%.1f%%)\n" c[1] c[2] n 100*n/length(oos_combos)
    end

    println("\n  Wing usage:")
    for w in WING_WIDTHS
        n = count(==(w), oos_wings); n == 0 && continue
        @printf "    \$%.1f  →  %d trades  (%.1f%%)\n" w n 100*n/length(oos_wings)
    end

    println("\n  Reference (test-tenor OOS, same applied dates):")
    @printf "    %-30s  %5s  %+10s  %+8s\n" "strategy" "n" "total" "Sharpe"
    oos_idx_apply = findall(d -> d in Set(oos_dates), dates_apply)
    base_ci = findfirst(==((0.20, 0.05)), DELTA_COMBOS)
    base_wi = findfirst(==(12.0), WING_WIDTHS)
    if base_ci !== nothing && base_wi !== nothing
        ref_2010 = filter(!isnan, [pnls_apply[i][base_ci, base_wi] for i in oos_idx_apply])
        sr_2010 = summary_for(ref_2010)
        @printf "    %-30s  %5d  %+10.0f  %+8.2f\n" "fixed 20p/5c, wing=\$12" sr_2010.n sr_2010.total sr_2010.sharpe
    end

    println("\n  ── Per-fold choices ──")
    @printf "  %3s  %-23s  %4s  %4s  %-10s  %-5s  %+10s\n" "#" "test window" "ntr" "nte" "Δ" "wing" "trS"
    for f in folds
        @printf "  %3d  %s → %s  %4d  %4d  (%.2f,%.2f)  \$%-4.0f  %+10.2f\n" f.idx f.test[1] f.test[2] f.n_tr f.n_te f.chosen_combo[1] f.chosen_combo[2] f.chosen_wing f.train_sharpe
    end

    println("\n  ── Per-month Sharpe (test-tenor OOS) ──")
    @printf "  %-9s  %5s  %+10s  %+8s\n" "month" "n" "totalPnL" "Sharpe"
    println("  " * "─"^46)
    for m in sort(unique(yearmonth.(oos_dates)))
        mask = [ym == m for ym in yearmonth.(oos_dates)]
        ys = oos_pnls[mask]; n = length(ys); n < 3 && continue
        sh = std(ys) > 0 ? mean(ys)/std(ys)*sqrt(252) : 0.0
        @printf "  %4d-%02d   %5d  %+10.0f  %+8.2f\n" m[1] m[2] n sum(ys) sh
    end

    ord_d = sortperm(oos_dates)
    p = plot(oos_dates[ord_d], cumsum(oos_pnls[ord_d]);
        xlabel="date", ylabel="cumulative PnL (USD)",
        title="$SYMBOL — train (Δ, wing) on $TENOR_TRAIN_STR, apply on $TENOR_TEST_STR",
        label="cross-tenor", lw=2, color=:steelblue, size=(1100, 500))
    hline!(p, [0]; color=:gray, ls=:dash, label="")
    savefig(p, joinpath(run_dir, "cumulative.png"))
    println("\n  Saved: cumulative.png")
end
