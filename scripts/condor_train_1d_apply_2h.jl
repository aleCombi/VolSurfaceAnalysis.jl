# scripts/condor_train_1d_apply_2h.jl
#
# Cross-tenor rolling experiment:
#   For each rolling window:
#     - Build training PnL grid on 1-DTE entries  (per (Δ, wing))
#     - Pick best (Δ, wing) by training Sharpe on 1-DTE
#     - Apply that (Δ, wing) to the 2-HOUR test window
#
# Hypothesis: 1-DTE has enough trades & signal for the search to generalize.
# If the chosen params reflect something stable about SPY (and not tenor-specific
# noise), they should transfer to the 2h horizon.

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, Plots

SYMBOL              = get(ENV, "SYM", "SPY")
START_DATE          = Date(parse(Int, get(ENV, "START_YEAR", "2017")), 1, 1)
END_DATE            = Date(2024, 1, 31)
SPREAD_LAMBDA       = 0.7
RATE                = 0.045
DIV_YIELD           = parse(Float64, get(ENV, "DIV", "0.013"))

ENTRY_TIME_1D       = Time(14, 0); EXPIRY_INTERVAL_1D = Day(1);  MAX_TAU_1D = 2.0
ENTRY_TIME_2H       = Time(14, 0); EXPIRY_INTERVAL_2H = Hour(2); MAX_TAU_2H = 0.5

PUT_DELTAS  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
CALL_DELTAS = [0.05, 0.10, 0.15, 0.20]
DELTA_COMBOS = [(pd, cd) for pd in PUT_DELTAS for cd in CALL_DELTAS]
WING_WIDTHS = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]

TRAIN_DAYS  = 90
TEST_DAYS   = 30
STEP_DAYS   = 30

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "condor_train_1d_apply_2h_$(SYMBOL)_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")
println("\n  $SYMBOL  $START_DATE → $END_DATE   train=1-DTE  apply=2-hr")

# --- Generic dataset builder -------------------------------------------------
function build_dataset(symbol, start_date, end_date, entry_time, expiry_interval, max_tau,
                       DELTA_COMBOS, WING_WIDTHS, RATE, DIV_YIELD, SPREAD_LAMBDA)
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
            otm_p = filter(r -> r.strike < sp_K, dctx.put_recs)
            otm_c = filter(r -> r.strike > sc_K, dctx.call_recs)
            (isempty(otm_p) || isempty(otm_c)) && continue
            for (wi, ww) in enumerate(WING_WIDTHS)
                target_lp = sp_K - ww
                target_lc = sc_K + ww
                lp = otm_p[argmin(abs.([r.strike - target_lp for r in otm_p]))]
                lc = otm_c[argmin(abs.([r.strike - target_lc for r in otm_c]))]
                cp = Position[]; ok2 = true
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
        push!(pnls, M)
    end
    ord = sortperm(dates)
    return dates[ord], pnls[ord], n_total[], n_skip[]
end

println("\n[1/2] Building 1-DTE dataset...")
dates_1d, pnls_1d, t1, s1 = build_dataset(SYMBOL, START_DATE, END_DATE,
    ENTRY_TIME_1D, EXPIRY_INTERVAL_1D, MAX_TAU_1D,
    DELTA_COMBOS, WING_WIDTHS, RATE, DIV_YIELD, SPREAD_LAMBDA)
@printf "  1-DTE: %d total, kept %d, skipped %d\n" t1 length(dates_1d) s1

println("\n[2/2] Building 2-hour dataset...")
dates_2h, pnls_2h, t2, s2 = build_dataset(SYMBOL, START_DATE, END_DATE,
    ENTRY_TIME_2H, EXPIRY_INTERVAL_2H, MAX_TAU_2H,
    DELTA_COMBOS, WING_WIDTHS, RATE, DIV_YIELD, SPREAD_LAMBDA)
@printf "  2-hr:  %d total, kept %d, skipped %d\n" t2 length(dates_2h) s2

n_combos = length(DELTA_COMBOS); n_wings = length(WING_WIDTHS)
sharpe_of(v) = let c = filter(!isnan, v); isempty(c) ? -Inf : (std(c) > 0 ? mean(c)/std(c)*sqrt(252) : 0.0); end

# --- Cross-tenor rolling: train 1d → apply 2h --------------------------------
function rolling_cross(dates_train, pnls_train, dates_apply, pnls_apply,
                       DELTA_COMBOS, WING_WIDTHS, TRAIN_DAYS, TEST_DAYS, STEP_DAYS)
    n_combos = length(DELTA_COMBOS); n_wings = length(WING_WIDTHS)
    oos_pnls = Float64[]; oos_dates = Date[]
    oos_combos = Tuple{Float64,Float64}[]; oos_wings = Float64[]
    folds = NamedTuple[]

    test_start = max(dates_train[1], dates_apply[1]) + Day(TRAIN_DAYS)
    last_d = min(dates_train[end], dates_apply[end])
    fidx = 0
    while test_start <= last_d
        te_end   = test_start + Day(TEST_DAYS) - Day(1)
        tr_start = test_start - Day(TRAIN_DAYS)
        tr_end   = test_start - Day(1)
        tr_idx_train = findall(d -> tr_start <= d <= tr_end, dates_train)
        te_idx_apply = findall(d -> test_start <= d <= te_end, dates_apply)
        if length(tr_idx_train) < 10 || length(te_idx_apply) < 1
            test_start += Day(STEP_DAYS); continue
        end

        # Joint Δ + wing search on 1-DTE training Sharpe
        best_sh = -Inf; best_ci = 1; best_wi = 1
        for ci in 1:n_combos, wi in 1:n_wings
            v = [pnls_train[i][ci, wi] for i in tr_idx_train]
            sh = sharpe_of(v)
            if sh > best_sh
                best_sh = sh; best_ci = ci; best_wi = wi
            end
        end
        chosen_combo = DELTA_COMBOS[best_ci]; chosen_wing = WING_WIDTHS[best_wi]

        # Apply to 2h test window
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
    return oos_pnls, oos_dates, oos_combos, oos_wings, folds
end

println("\nRolling cross-tenor selection...")
oos_pnls, oos_dates, oos_combos, oos_wings, folds = rolling_cross(
    dates_1d, pnls_1d, dates_2h, pnls_2h, DELTA_COMBOS, WING_WIDTHS,
    TRAIN_DAYS, TEST_DAYS, STEP_DAYS,
)
println("  Folds: $(length(folds))")

function summary_for(pnls)
    clean = filter(!isnan, pnls)
    isempty(clean) && return (n=0, total=0.0, sharpe=0.0, mdd=0.0, avg=0.0)
    cum = cumsum(clean); rmax = accumulate(max, cum); mdd = -minimum(cum .- rmax)
    sh = std(clean) > 0 ? mean(clean)/std(clean)*sqrt(252) : 0.0
    return (n=length(clean), total=sum(clean), sharpe=sh, mdd=mdd, avg=mean(clean))
end

r = summary_for(oos_pnls)
println("\n", "=" ^ 72)
println("  OOS RESULT — train on 1-DTE, apply on 2h")
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

# Reference comparisons on 2h OOS
println("\n  Reference (2h OOS, same applied dates):")
@printf "    %-30s  %5s  %+10s  %+8s\n" "strategy" "n" "total" "Sharpe"
oos_idx_2h = findall(d -> d in Set(oos_dates), dates_2h)
ref_2010 = filter(!isnan, [pnls_2h[i][findfirst(==((0.20, 0.05)), DELTA_COMBOS),
                                       findfirst(==(12.0), WING_WIDTHS)] for i in oos_idx_2h])
sr_2010 = summary_for(ref_2010)
@printf "    %-30s  %5d  %+10.0f  %+8.2f\n" "fixed 20p/5c, wing=\$12" sr_2010.n sr_2010.total sr_2010.sharpe

# Per-fold
println("\n  ── Per-fold choices ──")
@printf "  %3s  %-23s  %4s  %4s  %-10s  %-5s  %+10s\n" "#" "test window" "ntr" "nte" "Δ" "wing" "trS_1d"
for f in folds
    @printf "  %3d  %s → %s  %4d  %4d  (%.2f,%.2f)  \$%-4.0f  %+10.2f\n" f.idx f.test[1] f.test[2] f.n_tr f.n_te f.chosen_combo[1] f.chosen_combo[2] f.chosen_wing f.train_sharpe
end

# Per-month
println("\n  ── Per-month Sharpe (2h OOS) ──")
@printf "  %-9s  %5s  %+10s  %+8s\n" "month" "n" "totalPnL" "Sharpe"
println("  " * "─"^46)
for m in sort(unique(yearmonth.(oos_dates)))
    mask = [ym == m for ym in yearmonth.(oos_dates)]
    ys = oos_pnls[mask]; n = length(ys); n < 3 && continue
    sh = std(ys) > 0 ? mean(ys)/std(ys)*sqrt(252) : 0.0
    @printf "  %4d-%02d   %5d  %+10.0f  %+8.2f\n" m[1] m[2] n sum(ys) sh
end

# Plot
ord_d = sortperm(oos_dates)
p = plot(oos_dates[ord_d], cumsum(oos_pnls[ord_d]);
    xlabel="date", ylabel="cumulative PnL (USD)",
    title="$SYMBOL — train (Δ, wing) on 1-DTE, apply on 2h",
    label="cross-tenor", lw=2, color=:steelblue, size=(1100, 500),
)
hline!(p, [0]; color=:gray, ls=:dash, label="")
savefig(p, joinpath(run_dir, "cumulative.png"))
println("\n  Saved: cumulative.png")
