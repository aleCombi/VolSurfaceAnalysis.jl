# scripts/condor_2stage_delta_then_wing.jl
#
# Two-stage selection:
#   Stage 1 (in-sample 2017→IN_SAMPLE_END):  pick best fixed (put_Δ, call_Δ)
#                                            with wing fixed at WING_WIDTH.
#   Stage 2 (OOS IN_SAMPLE_END+1 → END_DATE): roll wing width with that fixed
#                                              short-leg delta combo.
#
# Reports OOS Sharpe + per-month, comparable to `condor_rolling_wing.jl`.

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, Plots

SYMBOL              = get(ENV, "SYM", "SPY")
START_DATE          = Date(parse(Int, get(ENV, "START_YEAR", "2017")), 1, 1)
IN_SAMPLE_END       = Date(parse(Int, get(ENV, "IS_END_YEAR", "2020")), 12, 31)
END_DATE            = Date(2024, 1, 31)
ENTRY_TIME          = Time(14, 0)
EXPIRY_INTERVAL     = Hour(2)
SPREAD_LAMBDA       = 0.7
RATE                = 0.045
DIV_YIELD           = parse(Float64, get(ENV, "DIV", "0.013"))
MAX_TAU_DAYS        = 0.5

# Stage 1 grid
PUT_DELTAS  = collect(0.050:0.025:0.300)   # 11 values
CALL_DELTAS = collect(0.025:0.025:0.225)   # 9 values
DELTA_COMBOS = [(pd, cd) for pd in PUT_DELTAS for cd in CALL_DELTAS]
SELECTION_WING_WIDTH = 12.0   # fixed wing for Stage 1 delta-search

# Stage 2 wing widths
WING_WIDTHS = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
TRAIN_DAYS  = 90
TEST_DAYS   = 30
STEP_DAYS   = 30

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "condor_2stage_$(SYMBOL)_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")
println("\n  $SYMBOL   in-sample $START_DATE → $IN_SAMPLE_END   OOS $(IN_SAMPLE_END + Day(1)) → $END_DATE")

# --- Data source --------------------------------------------------------------
println("\nLoading $SYMBOL  $START_DATE → $END_DATE ...")
(; source, sched) = polygon_parquet_source(SYMBOL;
    start_date=START_DATE, end_date=END_DATE, entry_time=ENTRY_TIME,
    rate=RATE, div_yield=DIV_YIELD, spread_lambda=SPREAD_LAMBDA,
)

# --- Build dataset: PnL per (entry, delta_combo, wing_width) -----------------
# Layout: pnls[entry_idx] → Matrix[combo_idx, wing_idx] of Float64 (NaN if missing)
n_combos = length(DELTA_COMBOS)
n_wings  = length(WING_WIDTHS)

dates  = Date[]
pnls_by_entry = Vector{Matrix{Float64}}()

n_total = 0
n_skip  = 0

println("\nBuilding dataset (per-entry PnL × $n_combos combos × $n_wings wings = $(n_combos*n_wings) values)...")
each_entry(source, EXPIRY_INTERVAL, sched; clear_cache=true) do ctx, settlement
    ismissing(settlement) && return
    global n_total += 1
    dctx = delta_context(ctx; rate=RATE, div_yield=DIV_YIELD)
    dctx === nothing && return
    dctx.tau * 365.25 > MAX_TAU_DAYS && (global n_skip += 1; return)
    spot = dctx.spot

    M = fill(NaN, n_combos, n_wings)
    for (ci, (pd, cd)) in enumerate(DELTA_COMBOS)
        sp_K = delta_strike(dctx, -pd, Put)
        sc_K = delta_strike(dctx,  cd, Call)
        (sp_K === nothing || sc_K === nothing) && continue
        otm_p = filter(r -> r.strike < sp_K, dctx.put_recs)
        otm_c = filter(r -> r.strike > sc_K, dctx.call_recs)
        (isempty(otm_p) || isempty(otm_c)) && continue
        for (wi, ww) in enumerate(WING_WIDTHS)
            target_lp = sp_K - ww
            target_lc = sc_K + ww
            lp = otm_p[argmin(abs.([r.strike - target_lp for r in otm_p]))]
            lc = otm_c[argmin(abs.([r.strike - target_lc for r in otm_c]))]
            cp = Position[]
            ok2 = true
            for t in (Trade(ctx.surface.underlying, sp_K, ctx.expiry, Put;  direction=-1, quantity=1.0),
                      Trade(ctx.surface.underlying, sc_K, ctx.expiry, Call; direction=-1, quantity=1.0),
                      Trade(ctx.surface.underlying, lp.strike, ctx.expiry, Put;  direction=+1, quantity=1.0),
                      Trade(ctx.surface.underlying, lc.strike, ctx.expiry, Call; direction=+1, quantity=1.0))
                p = open_position(t, ctx.surface)
                p === nothing && (ok2 = false; break)
                push!(cp, p)
            end
            ok2 && length(cp) == 4 || continue
            M[ci, wi] = settle(cp, Float64(settlement)) * spot
        end
    end
    push!(dates, Date(ctx.surface.timestamp))
    push!(pnls_by_entry, M)
end
println()
n_kept = length(dates)
@printf "  %d entries → kept %d  (skipped %d)\n" n_total n_kept n_skip
n_kept < 50 && error("Too few entries")

ord = sortperm(dates)
dates = dates[ord]
pnls_by_entry = pnls_by_entry[ord]

# --- Stage 1: pick best delta combo on in-sample ----------------------------
sharpe_of(v) = let c = filter(!isnan, v); isempty(c) ? -Inf : (std(c) > 0 ? mean(c)/std(c)*sqrt(252) : 0.0); end

is_idx = findall(<=(IN_SAMPLE_END), dates)
oos_idx = findall(>(IN_SAMPLE_END), dates)
println("\n  In-sample entries: $(length(is_idx))   OOS entries: $(length(oos_idx))")

# For each combo, compute in-sample Sharpe at SELECTION_WING_WIDTH
sel_wi = findfirst(==(SELECTION_WING_WIDTH), WING_WIDTHS)
combo_sharpes = Float64[]
for ci in 1:n_combos
    pnls_is = [pnls_by_entry[i][ci, sel_wi] for i in is_idx]
    push!(combo_sharpes, sharpe_of(pnls_is))
end
best_ci = argmax(combo_sharpes)
chosen_pd, chosen_cd = DELTA_COMBOS[best_ci]
println("\n  ── Stage 1 ──")
println("  Top 8 in-sample combos (wing fixed at \$$SELECTION_WING_WIDTH):")
@printf "    %-7s  %-7s  %+8s\n" "putΔ" "callΔ" "Sharpe"
top_is = sort(collect(zip(DELTA_COMBOS, combo_sharpes)); by=x->x[2], rev=true)
for ((pd, cd), sh) in top_is[1:min(8, end)]
    @printf "    %-7.2f  %-7.2f  %+8.2f\n" pd cd sh
end
println("\n  → Selected: putΔ=$chosen_pd  callΔ=$chosen_cd")

# --- Stage 2: rolling-wing on OOS with chosen delta -------------------------
oos_dates = dates[oos_idx]
oos_pnls_per_wing = [pnls_by_entry[i][best_ci, :] for i in oos_idx]   # Vector of length-n_wings
PnL_oos = Matrix{Float64}(undef, length(oos_idx), n_wings)
for (i, v) in enumerate(oos_pnls_per_wing)
    PnL_oos[i, :] = v
end

function rolling_wing_select(dates, PnL, WING_WIDTHS, TRAIN_DAYS, TEST_DAYS, STEP_DAYS, sharpe_of)
    n_widths = length(WING_WIDTHS)
    oos_pnls = Float64[]; oos_d = Date[]; oos_wings = Float64[]
    fold_choices = NamedTuple[]
    test_start = dates[1] + Day(TRAIN_DAYS)
    last_d = dates[end]
    fold_idx = 0
    while test_start <= last_d
        te_end = test_start + Day(TEST_DAYS) - Day(1)
        tr_start = test_start - Day(TRAIN_DAYS)
        tr_end = test_start - Day(1)
        tr_mask = (dates .>= tr_start) .& (dates .<= tr_end)
        te_mask = (dates .>= test_start) .& (dates .<= te_end)
        n_tr = sum(tr_mask); n_te = sum(te_mask)
        if n_tr < 10 || n_te < 1
            test_start += Day(STEP_DAYS); continue
        end
        train_sharpes = [sharpe_of(PnL[tr_mask, w]) for w in 1:n_widths]
        best_w = argmax(train_sharpes)
        chosen_w = WING_WIDTHS[best_w]
        test_pnls = filter(!isnan, PnL[te_mask, best_w])
        test_dates_w = dates[te_mask][.!isnan.(PnL[te_mask, best_w])]
        append!(oos_pnls, test_pnls)
        append!(oos_d, test_dates_w)
        append!(oos_wings, fill(chosen_w, length(test_pnls)))
        fold_idx += 1
        push!(fold_choices, (idx=fold_idx, test=(test_start, te_end), n_tr=n_tr, n_te=n_te,
                             chosen=chosen_w, train_sharpe=train_sharpes[best_w]))
        test_start += Day(STEP_DAYS)
    end
    return oos_pnls, oos_d, oos_wings, fold_choices
end

println("\n  ── Stage 2: rolling wing on OOS, fixed shorts (putΔ=$chosen_pd, callΔ=$chosen_cd) ──")
oos_p2, oos_d2, oos_w2, folds2 = rolling_wing_select(oos_dates, PnL_oos, WING_WIDTHS,
                                                       TRAIN_DAYS, TEST_DAYS, STEP_DAYS, sharpe_of)
println("\nFolds: $(length(folds2))")

# --- Report -----------------------------------------------------------------
function summary_for(pnls)
    clean = filter(!isnan, pnls)
    isempty(clean) && return (n=0, total=0.0, sharpe=0.0, mdd=0.0, avg=0.0)
    cum = cumsum(clean); rmax = accumulate(max, cum); mdd = -minimum(cum .- rmax)
    sh = std(clean) > 0 ? mean(clean)/std(clean)*sqrt(252) : 0.0
    return (n=length(clean), total=sum(clean), sharpe=sh, mdd=mdd, avg=mean(clean))
end

r2 = summary_for(oos_p2)
println("\n", "=" ^ 70)
println("  OOS RESULT — Stage 2 (rolling wing, fixed best delta from Stage 1)")
println("=" ^ 70)
@printf "  trades=%d  total=%+.0f  AvgPnL=%+.2f  Sharpe=%+.2f  MaxDD=%.0f\n" r2.n r2.total r2.avg r2.sharpe r2.mdd

# Wing usage on OOS
println("\n  Wing usage on OOS:")
for w in WING_WIDTHS
    n = count(==(w), oos_w2)
    pct = 100 * n / length(oos_w2)
    @printf "    \$%.1f  →  %d trades  (%.1f%%)\n" w n pct
end

# Reference: fixed wing+delta over OOS for comparison
println("\n  Reference — fixed strategies over OOS:")
@printf "    %-40s  %5s  %+10s  %+8s\n" "strategy" "n" "total" "Sharpe"
ref_pnl_2010 = filter(!isnan, [pnls_by_entry[i][findfirst(==((0.20, 0.05)), DELTA_COMBOS), sel_wi] for i in oos_idx])
ref_pnl_chosen = filter(!isnan, [pnls_by_entry[i][best_ci, sel_wi] for i in oos_idx])
sr1 = summary_for(ref_pnl_2010)
sr2 = summary_for(ref_pnl_chosen)
@printf "    %-40s  %5d  %+10.0f  %+8.2f\n" "fixed 20p/5c, wing=\$$(Int(SELECTION_WING_WIDTH)) (no search)" sr1.n sr1.total sr1.sharpe
@printf "    %-40s  %5d  %+10.0f  %+8.2f\n" "Δ-search only, wing=\$$(Int(SELECTION_WING_WIDTH))" sr2.n sr2.total sr2.sharpe

# --- Honest fully-fixed: pick (Δ, wing) jointly on in-sample, run constant OOS
println("\n  ── Honest fully-fixed (joint in-sample search over $(n_combos*n_wings) (Δ, wing) combos) ──")
joint_sharpes = fill(-Inf, n_combos, n_wings)
for ci in 1:n_combos, wi in 1:n_wings
    pnls_is = [pnls_by_entry[i][ci, wi] for i in is_idx]
    joint_sharpes[ci, wi] = sharpe_of(pnls_is)
end
best_lin = argmax(joint_sharpes)
best_ci_j, best_wi_j = best_lin.I
best_pd_j, best_cd_j = DELTA_COMBOS[best_ci_j]
best_ww_j = WING_WIDTHS[best_wi_j]
@printf "  In-sample winner: putΔ=%.2f  callΔ=%.2f  wing=\$%.0f   IS Sharpe=%+.2f\n" best_pd_j best_cd_j best_ww_j joint_sharpes[best_lin]

println("  Top 8 in-sample (Δ, wing) combos:")
@printf "    %-7s  %-7s  %-6s  %+8s\n" "putΔ" "callΔ" "wing" "Sharpe"
flat = [((DELTA_COMBOS[ci], WING_WIDTHS[wi]), joint_sharpes[ci, wi]) for ci in 1:n_combos, wi in 1:n_wings]
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

# Per-month
println("\n  ── Per-month Sharpe (Stage 2 OOS) ──")
@printf "  %-9s  %5s  %+10s  %+8s\n" "month" "n" "totalPnL" "Sharpe"
println("  " * "─"^46)
for m in sort(unique(yearmonth.(oos_d2)))
    mask = [ym == m for ym in yearmonth.(oos_d2)]
    ys = oos_p2[mask]
    n = length(ys); n < 3 && continue
    sh = std(ys) > 0 ? mean(ys)/std(ys)*sqrt(252) : 0.0
    @printf "  %4d-%02d   %5d  %+10.0f  %+8.2f\n" m[1] m[2] n sum(ys) sh
end

# Plot
ord_d = sortperm(oos_d2)
p = plot(oos_d2[ord_d], cumsum(oos_p2[ord_d]);
    xlabel="date", ylabel="cumulative PnL (USD)",
    title="$SYMBOL 2pm/2h — Stage 2 rolling wing (Δ chosen in-sample $START_DATE→$IN_SAMPLE_END)",
    label="Stage 2 (rolling wing on chosen Δ)", lw=2, color=:steelblue, size=(1100, 500),
)
hline!(p, [0]; color=:gray, ls=:dash, label="")
savefig(p, joinpath(run_dir, "cumulative.png"))
println("\n  Saved: cumulative.png")
