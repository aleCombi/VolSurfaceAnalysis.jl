# scripts/condor_rolling_short_then_long_1d.jl
#
# Rolling-window two-layer selection on 1-DTE iron condors:
#   Layer A: pick best (put_delta, call_delta) on the trailing TRAIN_DAYS
#            (using a reference wing for the search)
#   Layer B: with the chosen shorts FIXED, pick best wing width on the same
#            training window
#   Apply (chosen_delta, chosen_wing) to the next TEST_DAYS test window.
#
# Both layers are recomputed every STEP_DAYS — no in-sample peek, no fixed
# hyperparameters. Reports OOS Sharpe, fold-by-fold choices, per-month.

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, Plots

SYMBOL              = get(ENV, "SYM", "SPY")
START_DATE          = Date(parse(Int, get(ENV, "START_YEAR", "2017")), 1, 1)
END_DATE            = Date(2024, 1, 31)
ENTRY_TIME          = Time(14, 0)
EXPIRY_INTERVAL     = Day(1)        # 1-DTE
SPREAD_LAMBDA       = 0.7
RATE                = 0.045
DIV_YIELD           = parse(Float64, get(ENV, "DIV", "0.013"))
MAX_TAU_DAYS        = 2.0           # accept 1d entries (~1 calendar day)

# Layer A grid (short legs)
PUT_DELTAS  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
CALL_DELTAS = [0.05, 0.10, 0.15, 0.20]
DELTA_COMBOS = [(pd, cd) for pd in PUT_DELTAS for cd in CALL_DELTAS]
SELECTION_WING_WIDTH = 12.0     # reference wing used during Layer A scoring

# Layer B grid (wings)
WING_WIDTHS = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]

TRAIN_DAYS  = 90
TEST_DAYS   = 30
STEP_DAYS   = 30

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "condor_rolling_short_then_long_1d_$(SYMBOL)_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")
println("\n  $SYMBOL  $START_DATE → $END_DATE   1-DTE   train=$TRAIN_DAYS d / test=$TEST_DAYS d / step=$STEP_DAYS d")
println("  Δ grid: $(length(DELTA_COMBOS)) combos   wing grid: $(length(WING_WIDTHS))")

# --- Data source --------------------------------------------------------------
println("\nLoading $SYMBOL  $START_DATE → $END_DATE ...")
(; source, sched) = polygon_parquet_source(SYMBOL;
    start_date=START_DATE, end_date=END_DATE, entry_time=ENTRY_TIME,
    rate=RATE, div_yield=DIV_YIELD, spread_lambda=SPREAD_LAMBDA,
)

# --- Build dataset: PnL[entry_idx] is Matrix[combo, wing] of Float64 (NaN = miss)
n_combos = length(DELTA_COMBOS)
n_wings  = length(WING_WIDTHS)

dates  = Date[]
pnls_by_entry = Vector{Matrix{Float64}}()

n_total = 0
n_skip  = 0

println("\nBuilding dataset (per-entry PnL × $n_combos combos × $n_wings wings)...")
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

sel_wi = findfirst(==(SELECTION_WING_WIDTH), WING_WIDTHS)
sel_wi === nothing && error("SELECTION_WING_WIDTH=$SELECTION_WING_WIDTH not in WING_WIDTHS")

sharpe_of(v) = let c = filter(!isnan, v); isempty(c) ? -Inf : (std(c) > 0 ? mean(c)/std(c)*sqrt(252) : 0.0); end

# --- Rolling two-layer selection ---------------------------------------------
function rolling_short_then_long(dates, pnls_by_entry, DELTA_COMBOS, WING_WIDTHS,
                                  sel_wi, TRAIN_DAYS, TEST_DAYS, STEP_DAYS)
    n_combos = length(DELTA_COMBOS); n_wings = length(WING_WIDTHS)
    oos_pnls = Float64[]; oos_dates = Date[]
    oos_combos = Tuple{Float64,Float64}[]; oos_wings = Float64[]
    folds = NamedTuple[]

    test_start = dates[1] + Day(TRAIN_DAYS)
    last_d = dates[end]
    fidx = 0
    while test_start <= last_d
        te_end   = test_start + Day(TEST_DAYS) - Day(1)
        tr_start = test_start - Day(TRAIN_DAYS)
        tr_end   = test_start - Day(1)
        tr_idx = findall(d -> tr_start <= d <= tr_end, dates)
        te_idx = findall(d -> test_start <= d <= te_end, dates)
        if length(tr_idx) < 10 || length(te_idx) < 1
            test_start += Day(STEP_DAYS); continue
        end

        # Layer A: pick best delta combo at sel_wi
        combo_sharpes = Float64[]
        for ci in 1:n_combos
            v = [pnls_by_entry[i][ci, sel_wi] for i in tr_idx]
            push!(combo_sharpes, sharpe_of(v))
        end
        best_ci = argmax(combo_sharpes); chosen_combo = DELTA_COMBOS[best_ci]

        # Layer B: with chosen shorts fixed, pick best wing
        wing_sharpes = Float64[]
        for wi in 1:n_wings
            v = [pnls_by_entry[i][best_ci, wi] for i in tr_idx]
            push!(wing_sharpes, sharpe_of(v))
        end
        best_wi = argmax(wing_sharpes); chosen_wing = WING_WIDTHS[best_wi]

        # Apply to test window
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
                       train_sharpe_A=combo_sharpes[best_ci],
                       train_sharpe_B=wing_sharpes[best_wi]))
        test_start += Day(STEP_DAYS)
    end
    return oos_pnls, oos_dates, oos_combos, oos_wings, folds
end

println("\nRolling two-layer selection...")
oos_pnls, oos_dates, oos_combos, oos_wings, folds = rolling_short_then_long(
    dates, pnls_by_entry, DELTA_COMBOS, WING_WIDTHS, sel_wi,
    TRAIN_DAYS, TEST_DAYS, STEP_DAYS,
)
println("  Folds: $(length(folds))")

# --- Report -----------------------------------------------------------------
function summary_for(pnls)
    clean = filter(!isnan, pnls)
    isempty(clean) && return (n=0, total=0.0, sharpe=0.0, mdd=0.0, avg=0.0)
    cum = cumsum(clean); rmax = accumulate(max, cum); mdd = -minimum(cum .- rmax)
    sh = std(clean) > 0 ? mean(clean)/std(clean)*sqrt(252) : 0.0
    return (n=length(clean), total=sum(clean), sharpe=sh, mdd=mdd, avg=mean(clean))
end

r = summary_for(oos_pnls)
println("\n", "=" ^ 72)
println("  OOS RESULT — rolling Δ then rolling wing (1-DTE)")
println("=" ^ 72)
@printf "  trades=%d  total=%+.0f  AvgPnL=%+.2f  Sharpe=%+.2f  MaxDD=%.0f\n" r.n r.total r.avg r.sharpe r.mdd

# Combo and wing usage on OOS
println("\n  Δ combo usage (top 8):")
combo_counts = Dict{Tuple{Float64,Float64},Int}()
for c in oos_combos; combo_counts[c] = get(combo_counts, c, 0) + 1; end
for (c, n) in first(sort(collect(combo_counts); by=x->x[2], rev=true), min(8, length(combo_counts)))
    @printf "    (%.2f, %.2f)  →  %d trades (%.1f%%)\n" c[1] c[2] n 100*n/length(oos_combos)
end

println("\n  Wing usage:")
for w in WING_WIDTHS
    n = count(==(w), oos_wings)
    n == 0 && continue
    @printf "    \$%.1f  →  %d trades  (%.1f%%)\n" w n 100*n/length(oos_wings)
end

# Reference: each fixed (combo, wing) over the OOS period — sanity / dominator check
println("\n  Reference over OOS period:")
oos_idx_full = findall(d -> d in Set(oos_dates), dates)

@printf "    %-30s  %5s  %+10s  %+8s\n" "strategy" "n" "total" "Sharpe"
ref_2010_12 = filter(!isnan, [pnls_by_entry[i][findfirst(==((0.20, 0.05)), DELTA_COMBOS), sel_wi] for i in oos_idx_full])
sr_2010 = summary_for(ref_2010_12)
@printf "    %-30s  %5d  %+10.0f  %+8.2f\n" "fixed 20p/5c, wing=\$12" sr_2010.n sr_2010.total sr_2010.sharpe

# Best fixed (combo, wing) over OOS — peeked oracle for upper bound
function find_oracle(DELTA_COMBOS, WING_WIDTHS, pnls_by_entry, oos_idx_full, summary_for)
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
    return best_label, best_sum
end
best_oracle_label, best_oracle_summary = find_oracle(DELTA_COMBOS, WING_WIDTHS, pnls_by_entry, oos_idx_full, summary_for)
@printf "    %-30s  %5d  %+10.0f  %+8.2f\n" best_oracle_label best_oracle_summary.n best_oracle_summary.total best_oracle_summary.sharpe

# Per-fold summary
println("\n  ── Per-fold choices ──")
@printf "  %3s  %-23s  %4s  %4s  %-10s  %-5s  %+8s  %+8s\n" "#" "test window" "ntr" "nte" "Δ" "wing" "trS_A" "trS_B"
for f in folds
    @printf "  %3d  %s → %s  %4d  %4d  (%.2f,%.2f)  \$%-4.0f  %+8.2f  %+8.2f\n" f.idx f.test[1] f.test[2] f.n_tr f.n_te f.chosen_combo[1] f.chosen_combo[2] f.chosen_wing f.train_sharpe_A f.train_sharpe_B
end

# Per-month
println("\n  ── Per-month Sharpe (OOS) ──")
@printf "  %-9s  %5s  %+10s  %+8s\n" "month" "n" "totalPnL" "Sharpe"
println("  " * "─"^46)
for m in sort(unique(yearmonth.(oos_dates)))
    mask = [ym == m for ym in yearmonth.(oos_dates)]
    ys = oos_pnls[mask]; n = length(ys)
    n < 3 && continue
    sh = std(ys) > 0 ? mean(ys)/std(ys)*sqrt(252) : 0.0
    @printf "  %4d-%02d   %5d  %+10.0f  %+8.2f\n" m[1] m[2] n sum(ys) sh
end

# Plot
ord_d = sortperm(oos_dates)
p = plot(oos_dates[ord_d], cumsum(oos_pnls[ord_d]);
    xlabel="date", ylabel="cumulative PnL (USD)",
    title="$SYMBOL 1-DTE — rolling Δ then rolling wing",
    label="rolling Δ + rolling wing", lw=2, color=:steelblue, size=(1100, 500),
)
hline!(p, [0]; color=:gray, ls=:dash, label="")
savefig(p, joinpath(run_dir, "cumulative.png"))
println("\n  Saved: cumulative.png")
