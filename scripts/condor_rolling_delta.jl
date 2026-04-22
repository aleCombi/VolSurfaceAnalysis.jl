# scripts/condor_rolling_delta.jl
#
# Rolling-window selection of (put_delta, call_delta) for the SPY 2pm/2h
# iron condor. Wing width fixed at 12 (the dominant choice from the
# rolling-wing experiment). Compare honest rolling-delta Sharpe against
# fixed (20p/5c) baseline.

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, Plots
include(joinpath(@__DIR__, "lib", "experiment.jl"))

SYMBOL              = get(ENV, "SYM", "SPY")
START_DATE          = Date(parse(Int, get(ENV, "START_YEAR", "2017")), 1, 1)
END_DATE            = Date(2024, 1, 31)
ENTRY_TIME          = Time(14, 0)
EXPIRY_INTERVAL     = Hour(2)
SPREAD_LAMBDA       = 0.7
RATE                = 0.045
DIV_YIELD           = parse(Float64, get(ENV, "DIV", "0.013"))
MAX_TAU_DAYS        = 0.5

# Delta grid — sweep over short-leg deltas
PUT_DELTAS  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
CALL_DELTAS = [0.05, 0.10, 0.15, 0.20]
# Cartesian product
DELTA_COMBOS = [(pd, cd) for pd in PUT_DELTAS for cd in CALL_DELTAS]

WING_WIDTH          = 12.0   # fixed (rolling-wing said 12 dominates 64% of folds)

TRAIN_DAYS          = 90
TEST_DAYS           = 30
STEP_DAYS           = 30

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "condor_rolling_delta_$(SYMBOL)_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")
println("\n  $SYMBOL  $START_DATE → $END_DATE   train=$TRAIN_DAYS d / test=$TEST_DAYS d / step=$STEP_DAYS d")
println("  Delta combos: $(length(DELTA_COMBOS))   wing fixed: \$$(WING_WIDTH)")

# --- Data source --------------------------------------------------------------
store = DEFAULT_STORE
all_dates = available_polygon_dates(store, SYMBOL)
filtered = filter(d -> START_DATE <= d <= END_DATE, all_dates)
println("\nLoading $SYMBOL  ($(length(filtered)) trading days)...")

entry_ts = build_entry_timestamps(filtered, ENTRY_TIME)
entry_spots = read_polygon_spot_prices_for_timestamps(
    polygon_spot_root(store), entry_ts; symbol=SYMBOL,
)
source = ParquetDataSource(entry_ts;
    path_for_timestamp = ts -> polygon_options_path(store, Date(ts), SYMBOL),
    read_records = (path; where="") -> read_polygon_option_records(
        path, entry_spots; where=where, min_volume=0, warn=false,
        spread_lambda=SPREAD_LAMBDA, rate=RATE, div_yield=DIV_YIELD,
    ),
    spot_root = polygon_spot_root(store),
    spot_symbol = SYMBOL,
)
sched = filter(t -> t in Set(entry_ts), available_timestamps(source))

# --- Build dataset: PnL per (entry, delta_combo) -----------------------------
dates  = Date[]
pnls_by_combo = Vector{Vector{Float64}}()   # entry_idx → length(DELTA_COMBOS) PnLs

n_total = 0
n_skip  = 0

println("\nBuilding dataset (per-entry PnL × $(length(DELTA_COMBOS)) combos)...")
each_entry(source, EXPIRY_INTERVAL, sched; clear_cache=true) do ctx, settlement
    ismissing(settlement) && return
    global n_total += 1
    dctx = delta_context(ctx; rate=RATE, div_yield=DIV_YIELD)
    dctx === nothing && (global n_skip += 1; return)
    dctx.tau * 365.25 > MAX_TAU_DAYS && (global n_skip += 1; return)
    spot = dctx.spot

    ps = Float64[]
    ok = true
    for (pd, cd) in DELTA_COMBOS
        sp_K = delta_strike(dctx, -pd, Put)
        sc_K = delta_strike(dctx,  cd, Call)
        if sp_K === nothing || sc_K === nothing
            push!(ps, NaN); continue
        end
        otm_p = filter(r -> r.strike < sp_K, dctx.put_recs)
        otm_c = filter(r -> r.strike > sc_K, dctx.call_recs)
        (isempty(otm_p) || isempty(otm_c)) && (push!(ps, NaN); continue)
        target_lp = sp_K - WING_WIDTH
        target_lc = sc_K + WING_WIDTH
        lp = otm_p[argmin(abs.([r.strike - target_lp for r in otm_p]))]
        lc = otm_c[argmin(abs.([r.strike - target_lc for r in otm_c]))]
        cp = Position[]
        for t in (Trade(ctx.surface.underlying, sp_K, ctx.expiry, Put;  direction=-1, quantity=1.0),
                  Trade(ctx.surface.underlying, sc_K, ctx.expiry, Call; direction=-1, quantity=1.0),
                  Trade(ctx.surface.underlying, lp.strike, ctx.expiry, Put;  direction=+1, quantity=1.0),
                  Trade(ctx.surface.underlying, lc.strike, ctx.expiry, Call; direction=+1, quantity=1.0))
            p = open_position(t, ctx.surface)
            p === nothing && (ok = false; break)
            push!(cp, p)
        end
        !ok && break
        length(cp) == 4 || (push!(ps, NaN); continue)
        push!(ps, settle(cp, Float64(settlement)) * spot)
    end
    ok || (global n_skip += 1; return)
    push!(dates, Date(ctx.surface.timestamp))
    push!(pnls_by_combo, ps)
end
println()
n_kept = length(dates)
@printf "  %d entries → kept %d  (skipped %d)\n" n_total n_kept n_skip
n_kept < 50 && error("Too few entries")

ord = sortperm(dates)
dates = dates[ord]
pnls_by_combo = pnls_by_combo[ord]
PnL = hcat(pnls_by_combo...)'   # rows = entries, cols = combos
n_combos = length(DELTA_COMBOS)

# --- Helpers ------------------------------------------------------------------
function sharpe_with_nan(v::AbstractVector{<:Real})
    clean = filter(!isnan, v)
    isempty(clean) && return -Inf
    s = std(clean)
    s > 0 ? mean(clean)/s*sqrt(252) : 0.0
end

# --- Rolling delta-combo selection -------------------------------------------
oos_pnls   = Float64[]
oos_dates  = Date[]
oos_combos = Tuple{Float64,Float64}[]
fold_choices = NamedTuple[]

function rolling_delta_select!(dates, PnL, DELTA_COMBOS, TRAIN_DAYS, TEST_DAYS, STEP_DAYS)
    n_combos = length(DELTA_COMBOS)
    oos_pnls = Float64[]; oos_dates = Date[]; oos_combos = Tuple{Float64,Float64}[]
    fold_choices = NamedTuple[]
    folds = build_folds(dates;
        train_days=TRAIN_DAYS, test_days=TEST_DAYS, step_days=STEP_DAYS,
        min_train=10, min_test=1,
    )
    for f in folds
        tr_mask = f.train_mask; te_mask = f.test_mask
        train_sharpes = [sharpe_with_nan(PnL[tr_mask, c]) for c in 1:n_combos]
        best_c = argmax(train_sharpes)
        chosen = DELTA_COMBOS[best_c]
        test_pnls = filter(!isnan, PnL[te_mask, best_c])
        test_dates_w = dates[te_mask][.!isnan.(PnL[te_mask, best_c])]
        append!(oos_pnls, test_pnls)
        append!(oos_dates, test_dates_w)
        append!(oos_combos, fill(chosen, length(test_pnls)))
        push!(fold_choices, (
            idx=f.idx, test=(f.test_start, f.test_end), n_tr=sum(tr_mask), n_te=sum(te_mask),
            chosen=chosen, train_sharpe=train_sharpes[best_c],
            test_sharpe=length(test_pnls) > 1 ? mean(test_pnls)/(std(test_pnls)+1e-12)*sqrt(252) : 0.0,
            test_total=sum(test_pnls),
        ))
    end
    return oos_pnls, oos_dates, oos_combos, fold_choices
end

oos_pnls, oos_dates, oos_combos, fold_choices = rolling_delta_select!(
    dates, PnL, DELTA_COMBOS, TRAIN_DAYS, TEST_DAYS, STEP_DAYS,
)

# --- Reference comparisons ----------------------------------------------------
function summary_for(pnls)
    clean = filter(!isnan, pnls)
    isempty(clean) && return (n=0, total=0.0, sharpe=0.0, mdd=0.0)
    cum = cumsum(clean); rmax = accumulate(max, cum)
    mdd = -minimum(cum .- rmax)
    sh = std(clean) > 0 ? mean(clean)/std(clean)*sqrt(252) : 0.0
    return (n=length(clean), total=sum(clean), sharpe=sh, mdd=mdd)
end

println("\n", "=" ^ 80)
println("  RESULT — rolling-delta-combo OOS")
println("=" ^ 80)
r = summary_for(oos_pnls)
@printf "  trades=%d   total=%+.0f   Sharpe=%+.2f   MaxDD=%.0f\n" r.n r.total r.sharpe r.mdd

# Combo usage
println("\n  Combo usage (top 10):")
combo_counts = Dict{Tuple{Float64,Float64},Int}()
for c in oos_combos
    combo_counts[c] = get(combo_counts, c, 0) + 1
end
sorted_combos = sort(collect(combo_counts); by=x->x[2], rev=true)
for (c, n) in first(sorted_combos, min(10, length(sorted_combos)))
    @printf "    (%.2f, %.2f)  →  %d trades (%.1f%%)\n" c[1] c[2] n 100*n/length(oos_combos)
end

# Reference: each fixed combo's full-sample Sharpe (over the OOS period)
println("\n  Reference: fixed-combo Sharpe over the OOS period")
@printf "    %-7s  %-7s  %5s  %+8s  %+10s  %8s\n" "putΔ" "callΔ" "n" "Sharpe" "total" "MaxDD"
println("    " * "─"^48)
oos_idx = findall(d -> d in Set(oos_dates), dates)
oos_pnl_matrix = PnL[oos_idx, :]
combo_summaries = []
for (i, (pd, cd)) in enumerate(DELTA_COMBOS)
    s = summary_for(oos_pnl_matrix[:, i])
    push!(combo_summaries, (pd=pd, cd=cd, summary=s))
end
# Sort by Sharpe
sort!(combo_summaries; by=x->x.summary.sharpe, rev=true)
for (i, c) in enumerate(combo_summaries)
    i > 8 && break
    s = c.summary
    @printf "    %-7.2f  %-7.2f  %5d  %+8.2f  %+10.0f  %8.0f\n" c.pd c.cd s.n s.sharpe s.total s.mdd
end

# Per-month
println("\n  ── Per-month Sharpe (rolling-delta OOS) ──")
@printf "  %-9s  %5s  %+10s  %+8s\n" "month" "n" "totalPnL" "Sharpe"
println("  " * "─"^46)
for m in sort(unique(yearmonth.(oos_dates)))
    mask = [ym == m for ym in yearmonth.(oos_dates)]
    ys = oos_pnls[mask]
    n = length(ys)
    n < 3 && continue
    sh = std(ys) > 0 ? mean(ys)/std(ys)*sqrt(252) : 0.0
    @printf "  %4d-%02d   %5d  %+10.0f  %+8.2f\n" m[1] m[2] n sum(ys) sh
end

# --- Plot ---------------------------------------------------------------------
ord_d = sortperm(oos_dates)
d_sorted = oos_dates[ord_d]
cum = cumsum(oos_pnls[ord_d])
p = plot(d_sorted, cum;
    xlabel="date", ylabel="cumulative PnL (USD)",
    title="$SYMBOL 2pm/2h iron condor — rolling delta selection (wing=\$$WING_WIDTH)",
    label="rolling-delta", lw=2, color=:steelblue, size=(1100, 500),
)
# Overlay fixed (20p/5c) for reference
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
