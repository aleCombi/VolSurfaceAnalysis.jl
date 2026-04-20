# scripts/condor_rolling_wing.jl
#
# Rolling forward-walk wing selection:
#   - Train window: TRAIN_DAYS (e.g. 90)
#   - Test window:  TEST_DAYS (e.g. 30)
#   - Step:         STEP_DAYS (= TEST_DAYS = no overlap)
#
# For each test window, pick the wing width that maximized Sharpe in the
# preceding TRAIN_DAYS, apply it through the test window, record PnL.
#
# Reports: aggregate Sharpe of the rolling strategy, per-test-window choice,
# per-month Sharpe of the OOS PnL series. No ML model.

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, Plots

# =============================================================================
# Configuration — set SYMBOL via env var SYM (default SPY)
# =============================================================================

SYMBOL              = get(ENV, "SYM", "SPY")
START_DATE          = Date(parse(Int, get(ENV, "START_YEAR", "2017")), 1, 1)
END_DATE            = Date(2024, 1, 31)
ENTRY_TIME          = Time(14, 0)        # 2pm ET
EXPIRY_INTERVAL     = Hour(2)            # 2h target → same-day 4pm
SPREAD_LAMBDA       = 0.7
RATE                = 0.045
DIV_YIELD           = parse(Float64, get(ENV, "DIV", "0.013"))
MAX_TAU_DAYS        = 0.5

PUT_DELTA           = 0.20
CALL_DELTA          = 0.05
WING_WIDTHS         = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]

TRAIN_DAYS          = 90    # 3 months train window
TEST_DAYS           = 30    # 1 month test window
STEP_DAYS           = 30    # advance 1 test window

# =============================================================================
# Output dir
# =============================================================================

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "condor_rolling_wing_$(SYMBOL)_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")
println("\n  $SYMBOL  $START_DATE → $END_DATE   train=$TRAIN_DAYS d / test=$TEST_DAYS d / step=$STEP_DAYS d")

# =============================================================================
# Data source
# =============================================================================

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

# =============================================================================
# Build dataset: PnL per (entry, wing_width)
# =============================================================================

dates  = Date[]
pnls_by_width = Vector{Vector{Float64}}()

n_total = 0
n_skip  = 0
function _evict!(src::ParquetDataSource)
    empty!(src.surface_cache); empty!(src.spot_date_cache); nothing
end

println("\nBuilding dataset (per-entry PnL × $(length(WING_WIDTHS)) widths)...")
each_entry(source, EXPIRY_INTERVAL, sched) do ctx, settlement
    try
        ismissing(settlement) && return
        global n_total += 1
        recs = VolSurfaceAnalysis._ctx_recs(ctx)
        tau  = VolSurfaceAnalysis._ctx_tau(ctx)
        tau <= 0.0 && return
        tau * 365.25 > MAX_TAU_DAYS && (global n_skip += 1; return)
        spot = ctx.surface.spot
        F = spot * exp((RATE - DIV_YIELD) * tau)

        put_recs  = filter(r -> r.option_type == Put,  recs)
        call_recs = filter(r -> r.option_type == Call, recs)
        sp_K = VolSurfaceAnalysis._best_delta_strike(put_recs,  -PUT_DELTA,  spot, :put,  F, tau, RATE)
        sc_K = VolSurfaceAnalysis._best_delta_strike(call_recs,  CALL_DELTA, spot, :call, F, tau, RATE)
        (sp_K === nothing || sc_K === nothing) && (global n_skip += 1; return)

        otm_put_recs  = filter(r -> r.strike < sp_K, put_recs)
        otm_call_recs = filter(r -> r.strike > sc_K, call_recs)
        (isempty(otm_put_recs) || isempty(otm_call_recs)) && (global n_skip += 1; return)

        ps = Float64[]
        ok = true
        for ww in WING_WIDTHS
            target_lp = sp_K - ww
            target_lc = sc_K + ww
            lp = otm_put_recs[argmin(abs.([r.strike - target_lp for r in otm_put_recs]))]
            lc = otm_call_recs[argmin(abs.([r.strike - target_lc for r in otm_call_recs]))]
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
            length(cp) == 4 || (ok = false; break)
            push!(ps, settle(cp, Float64(settlement)) * spot)
        end
        ok || (global n_skip += 1; return)
        push!(dates, Date(ctx.surface.timestamp))
        push!(pnls_by_width, ps)
    finally
        _evict!(source)
    end
end
println()
n_kept = length(dates)
@printf "  %d entries → kept %d  (skipped %d)\n" n_total n_kept n_skip
n_kept < 50 && error("Too few entries to do walk-forward")

# Sort by date
ord = sortperm(dates)
dates = dates[ord]
pnls_by_width = pnls_by_width[ord]
# Build matrix [n_entries × n_widths]
PnL = hcat(pnls_by_width...)'    # rows = entries, cols = widths
n_widths = length(WING_WIDTHS)

# =============================================================================
# Rolling-wing selection
# =============================================================================

sharpe_of(v) = std(v) > 0 ? mean(v)/std(v)*sqrt(252) : 0.0

# For each test window, pick wing using preceding TRAIN_DAYS
oos_pnls   = Float64[]
oos_dates  = Date[]
oos_wings  = Float64[]   # which wing was chosen for each test entry
fold_choices = NamedTuple[]

function rolling_wing_select(dates, PnL, WING_WIDTHS, TRAIN_DAYS, TEST_DAYS, STEP_DAYS, sharpe_of)
    n_widths = length(WING_WIDTHS)
    oos_pnls = Float64[]; oos_dates = Date[]; oos_wings = Float64[]
    fold_choices = NamedTuple[]
    test_start = dates[1] + Day(TRAIN_DAYS)
    last_d = dates[end]
    fold_idx = 0
    while test_start <= last_d
        te_end   = test_start + Day(TEST_DAYS) - Day(1)
        tr_start = test_start - Day(TRAIN_DAYS)
        tr_end   = test_start - Day(1)
        tr_mask  = (dates .>= tr_start) .& (dates .<= tr_end)
        te_mask  = (dates .>= test_start) .& (dates .<= te_end)
        n_tr = sum(tr_mask)
        n_te = sum(te_mask)
        if n_tr < 10 || n_te < 1
            test_start += Day(STEP_DAYS); continue
        end
        train_sharpes = [sharpe_of(PnL[tr_mask, w]) for w in 1:n_widths]
        best_w = argmax(train_sharpes)
        chosen_wing = WING_WIDTHS[best_w]
        test_pnls = PnL[te_mask, best_w]
        test_dates_w = dates[te_mask]
        append!(oos_pnls, test_pnls)
        append!(oos_dates, test_dates_w)
        append!(oos_wings, fill(chosen_wing, length(test_pnls)))
        fold_idx += 1
        push!(fold_choices, (
            idx=fold_idx,
            train=(tr_start, tr_end), test=(test_start, te_end),
            n_tr=n_tr, n_te=n_te,
            chosen_wing=chosen_wing, train_sharpe=train_sharpes[best_w],
            test_sharpe=sharpe_of(test_pnls),
            test_total=sum(test_pnls),
        ))
        test_start += Day(STEP_DAYS)
    end
    return (pnls=oos_pnls, dates=oos_dates, wings=oos_wings, folds=fold_choices)
end

result = rolling_wing_select(dates, PnL, WING_WIDTHS, TRAIN_DAYS, TEST_DAYS, STEP_DAYS, sharpe_of)
oos_pnls = result.pnls; oos_dates = result.dates; oos_wings = result.wings
fold_choices = result.folds

# =============================================================================
# Report
# =============================================================================

println("\nFolds: $(length(fold_choices))")
println("\n  ── Fold-by-fold rolling-wing choice ──")
@printf "  %-3s  %-23s  %4s  %4s  %5s  %+8s  %+8s  %+8s\n" "f" "test_window" "n_tr" "n_te" "wing" "tr_Sh" "te_Sh" "te_USD"
println("  " * "─"^85)
for r in fold_choices
    @printf "  %-3d  %s → %s  %4d  %4d  %5.1f  %+8.2f  %+8.2f  %+8.0f\n" (
        r.idx, r.test[1], r.test[2], r.n_tr, r.n_te,
        r.chosen_wing, r.train_sharpe, r.test_sharpe, r.test_total,
    )...
end

n_oos = length(oos_pnls)
total = sum(oos_pnls)
sh_oos = sharpe_of(oos_pnls)
mdd = let cum = cumsum(oos_pnls)
    rmax = accumulate(max, cum)
    -minimum(cum .- rmax)
end
println("\n  ── Aggregate OOS (rolling-wing) ──")
@printf "  trades = %d   total = %+.0f   AvgPnL = %+.2f   Sharpe = %+.2f   MaxDD = %.0f\n" n_oos total mean(oos_pnls) sh_oos mdd

# Wing-choice frequency
println("\n  Wing usage:")
for w in WING_WIDTHS
    n = count(==(w), oos_wings)
    pct = 100 * n / n_oos
    @printf "    wing %.1f → %d trades (%.0f%%)\n" w n pct
end

# Per-month Sharpe of the rolling-wing OOS series
println("\n  ── Per-month Sharpe (rolling-wing OOS) ──")
@printf "  %-9s  %5s  %+10s  %+8s\n" "month" "n" "totalPnL" "Sharpe"
println("  " * "─"^46)
for m in sort(unique(yearmonth.(oos_dates)))
    mask = [ym == m for ym in yearmonth.(oos_dates)]
    ys = oos_pnls[mask]
    n = length(ys)
    n < 3 && continue
    @printf "  %4d-%02d   %5d  %+10.0f  %+8.2f\n" m[1] m[2] n sum(ys) sharpe_of(ys)
end

# =============================================================================
# Dump OOS series to CSV — ground truth for cross-implementation comparison
# =============================================================================

csv_path = joinpath(run_dir, "oos_series.csv")
open(csv_path, "w") do io
    println(io, "date,chosen_wing,oos_pnl_usd")
    for i in eachindex(oos_dates)
        @printf io "%s,%.1f,%.6f\n" oos_dates[i] oos_wings[i] oos_pnls[i]
    end
end
println("\n  Saved: $csv_path  ($(length(oos_dates)) rows)")

# =============================================================================
# Plots
# =============================================================================

ord_d = sortperm(oos_dates)
d_sorted = oos_dates[ord_d]
cum = cumsum(oos_pnls[ord_d])
p1 = plot(d_sorted, cum;
    xlabel="date", ylabel="cumulative PnL (USD)",
    title="$SYMBOL 2pm/2h iron condor — rolling wing selection",
    label="OOS cumulative", lw=2, color=:steelblue, size=(1100, 500),
)
hline!(p1, [0]; color=:gray, ls=:dash, label="")
savefig(p1, joinpath(run_dir, "cumulative.png"))

# Wing choice over time
fold_dates = [r.test[1] for r in fold_choices]
fold_wings = [r.chosen_wing for r in fold_choices]
p2 = scatter(fold_dates, fold_wings;
    xlabel="test-window start", ylabel="chosen wing width",
    title="$SYMBOL — wing chosen per test fold",
    ms=8, color=:darkorange, label="", size=(1100, 350),
)
savefig(p2, joinpath(run_dir, "wing_over_time.png"))
println("\n  Saved: cumulative.png, wing_over_time.png")
