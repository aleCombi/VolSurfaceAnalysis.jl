# scripts/spy_sizing_condor_sweep.jl
#
# Sweep symmetric iron condor wing widths on SPY 1DTE 20p/5c.
# For each WING_WIDTH ∈ {3, 5, 7, 10, 15, 20}, compute the per-day PnL,
# run the SAME walk-forward + sizing-policy pipeline as before, and
# tabulate Sharpe / total PnL for the baseline + best-policy per width.

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, Random, Plots
include(joinpath(@__DIR__, "lib", "experiment.jl"))

# =============================================================================
# Configuration
# =============================================================================

SYMBOL              = "SPY"
START_DATE          = Date(2017, 1, 1)
END_DATE            = Date(2024, 1, 31)
ENTRY_TIME          = Time(14, 0)        # 2pm ET
EXPIRY_INTERVAL     = Hour(2)            # 2h tenor
SPREAD_LAMBDA       = 0.7
RATE                = 0.045
DIV_YIELD           = 0.013
MAX_TAU_DAYS        = 0.5                # filter to intraday only

PUT_DELTA           = 0.20
CALL_DELTA          = 0.05
WING_WIDTHS         = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]   # narrower than 1DTE — intraday strikes are tighter

FEATURES            = Feature[
    SpotMinuteLogSig(lookback_hours=5, depth=3, min_points=150),
]

TRAIN_WINDOW_DAYS   = 365
TEST_WINDOW_DAYS    = 90
STEP_DAYS           = 90
MAX_SIZE            = 2.0

# =============================================================================
# Output dir
# =============================================================================

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "spy_sizing_condor_rolling_2h_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")

# =============================================================================
# Data source
# =============================================================================

store = DEFAULT_STORE
all_dates = available_polygon_dates(store, SYMBOL)
filtered = filter(d -> START_DATE <= d <= END_DATE, all_dates)
println("\nLoading $SYMBOL  $START_DATE → $END_DATE  ($(length(filtered)) trading days)")

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
# Build dataset: for each entry compute PnL for ALL wing widths simultaneously
# =============================================================================

dates  = Date[]
feats  = Vector{Vector{Float32}}()
spots  = Float64[]
# pnls_by_width[entry_idx][width_idx] = pnl in USD
pnls_by_width = Vector{Vector{Float64}}()

n_total = 0
n_skip  = 0

println("\nBuilding dataset (per-entry PnL × $(length(WING_WIDTHS)) widths)...")
each_entry(source, EXPIRY_INTERVAL, sched; clear_cache=true) do ctx, settlement
    ismissing(settlement) && return
    global n_total += 1
    fvec = extract_surface_features(ctx, FEATURES)
    fvec === nothing && (global n_skip += 1; return)

    dctx = delta_context(ctx; rate=RATE, div_yield=DIV_YIELD)
    dctx === nothing && (global n_skip += 1; return)
    dctx.tau * 365.25 > MAX_TAU_DAYS && (global n_skip += 1; return)
    spot = dctx.spot

    sp_K = delta_strike(dctx, -PUT_DELTA,  Put)
    sc_K = delta_strike(dctx,  CALL_DELTA, Call)
    (sp_K === nothing || sc_K === nothing) && (global n_skip += 1; return)

    # Compute PnL for each wing width by snapping to nearest available OTM strike
    pnls_this_entry = Float64[]
    ok = true
    for ww in WING_WIDTHS
        lp_K = nearest_otm_strike(dctx, sp_K, ww, Put)
        lc_K = nearest_otm_strike(dctx, sc_K, ww, Call)
        (lp_K === nothing || lc_K === nothing) && (ok = false; break)
        condor_pos = Position[]
        for t in (Trade(ctx.surface.underlying, sp_K, ctx.expiry, Put;  direction=-1, quantity=1.0),
                  Trade(ctx.surface.underlying, sc_K, ctx.expiry, Call; direction=-1, quantity=1.0),
                  Trade(ctx.surface.underlying, lp_K, ctx.expiry, Put;  direction=+1, quantity=1.0),
                  Trade(ctx.surface.underlying, lc_K, ctx.expiry, Call; direction=+1, quantity=1.0))
            p = open_position(t, ctx.surface)
            p === nothing && (ok = false; break)
            push!(condor_pos, p)
        end
        !ok && break
        length(condor_pos) == 4 || (ok = false; break)
        push!(pnls_this_entry, settle(condor_pos, Float64(settlement)) * spot)
    end
    ok || (global n_skip += 1; return)

    push!(dates, Date(ctx.surface.timestamp))
    push!(feats, fvec)
    push!(spots, spot)
    push!(pnls_by_width, pnls_this_entry)
end
println()
n_kept = length(dates)
@printf "  %d entries → kept %d  (skipped %d)\n" n_total n_kept n_skip

# =============================================================================
# Helpers
# =============================================================================

sigmoid_fn(z) = 1 / (1 + exp(-z))

function run_walkforward(dates_sorted, X_all, Y_all, folds)
    all_pred  = Float64[]
    all_y     = Float64[]
    all_tμ    = Float64[]
    all_tσ    = Float64[]
    all_tmed  = Float64[]
    all_tq75  = Float64[]
    all_d     = Date[]

    for f in folds
        Random.seed!(42)
        Xtr = X_all[:, f.train_mask]
        Ytr = Y_all[:, f.train_mask]
        Xte = X_all[:, f.test_mask]
        Yte = Y_all[:, f.test_mask]
        Dte = dates_sorted[f.test_mask]

        model, μ, σ, _ = train_ridge!(nothing, Xtr, Ytr; alpha=0.0, val_fraction=0.2)
        σ_safe = [s > 1e-8 ? s : one(s) for s in σ]
        Xtr_norm = (Xtr .- Float32.(μ)) ./ Float32.(σ_safe)
        train_preds = Float64.(vec(model(Xtr_norm)))
        tμ = mean(train_preds); tσ = std(train_preds)
        tmed = median(train_preds)
        tq75 = sort(train_preds)[max(1, round(Int, 0.75 * length(train_preds)))]

        Xte_norm = (Xte .- Float32.(μ)) ./ Float32.(σ_safe)
        preds = Float64.(vec(model(Xte_norm)))
        n_te = length(preds)
        append!(all_pred, preds); append!(all_y, vec(Yte))
        append!(all_tμ, fill(tμ, n_te)); append!(all_tσ, fill(tσ, n_te))
        append!(all_tmed, fill(tmed, n_te)); append!(all_tq75, fill(tq75, n_te))
        append!(all_d, Dte)
    end

    return (preds=all_pred, y=all_y, μ=all_tμ, σ=all_tσ, med=all_tmed, q75=all_tq75, dates=all_d)
end

function policy_metrics(scores, ys, μs, σs, meds, q75s, max_size)
    policies = [
        ("baseline",      (p, μ, σ, m, q) -> 1.0),
        ("skip<med",      (p, μ, σ, m, q) -> p > m ? 1.0 : 0.0),
        ("topQ",          (p, μ, σ, m, q) -> p > q ? 1.0 : 0.0),
        ("linear",        (p, μ, σ, m, q) -> clamp((p - μ) / max(σ, 1e-9), 0.0, max_size)),
        ("sigmoid",       (p, μ, σ, m, q) -> max_size * sigmoid_fn((p - μ) / max(σ, 1e-9))),
    ]
    out = NamedTuple[]
    n = length(ys)
    for (name, fn) in policies
        sizes = [fn(scores[i], μs[i], σs[i], meds[i], q75s[i]) for i in 1:n]
        sized = sizes .* ys
        n_tr = count(>(0), sizes)
        cum = cumsum(sized); rmax = accumulate(max, cum); maxdd = -minimum(cum .- rmax)
        sh = std(sized) > 0 ? mean(sized)/std(sized)*sqrt(252) : 0.0
        push!(out, (name=name, total=sum(sized), avg_per_day=mean(sized),
                    sharpe=sh, n_trades=n_tr, maxdd=maxdd))
    end
    return out
end

# =============================================================================
# Sweep loop
# =============================================================================

# Sort dataset by date once
ord = sortperm(dates)
dates_sorted = dates[ord]
feats_sorted = feats[ord]
spots_sorted = spots[ord]
pnls_sorted  = pnls_by_width[ord]
X_all = Float32.(hcat(feats_sorted...))
folds = build_folds(dates_sorted;
    train_days=TRAIN_WINDOW_DAYS, test_days=TEST_WINDOW_DAYS, step_days=STEP_DAYS,
    min_train=80, min_test=20,
)
println("\nFolds: $(length(folds))")

println("\n", "=" ^ 110)
println("  WING WIDTH SWEEP — SPY 1DTE 20p/5c iron condor")
println("=" ^ 110)
@printf "  %5s  %-10s  %7s  %5s  %10s  %8s  %8s  %10s\n" (
    "wing", "policy", "trades", "rate", "totalPnL", "/day", "Sharpe", "maxDD"
)...
println("  ", "-" ^ 96)

best_per_width = Dict()
all_results = NamedTuple[]

for (wi, ww) in enumerate(WING_WIDTHS)
    Y_all = reshape(Float32.([p[wi] for p in pnls_sorted]), 1, length(pnls_sorted))
    res = run_walkforward(dates_sorted, X_all, Y_all, folds)
    metrics = policy_metrics(res.preds, res.y, res.μ, res.σ, res.med, res.q75, MAX_SIZE)
    for m in metrics
        @printf "  %5.1f  %-10s  %7d  %4.0f%%  %+10.0f  %+8.2f  %+8.2f  %10.0f\n" (
            ww, m.name, m.n_trades, 100*m.n_trades/length(res.y),
            m.total, m.avg_per_day, m.sharpe, m.maxdd,
        )...
        push!(all_results, (wing=ww, m...))
    end
    best = argmax([m.sharpe for m in metrics])
    best_per_width[ww] = metrics[best]
    println("  ", "-" ^ 96)
end

# =============================================================================
# Best policy per wing — summary
# =============================================================================

println("\n", "=" ^ 70)
println("  BEST POLICY PER WING WIDTH")
println("=" ^ 70)
@printf "  %5s  %-10s  %8s  %10s  %10s\n" "wing" "policy" "Sharpe" "total" "maxDD"
println("  ", "-" ^ 60)
for ww in WING_WIDTHS
    m = best_per_width[ww]
    @printf "  %5.1f  %-10s  %+8.2f  %+10.0f  %10.0f\n" ww m.name m.sharpe m.total m.maxdd
end

# =============================================================================
# Plot: Sharpe vs wing width, baseline + best-policy
# =============================================================================

ws    = [r.wing for r in all_results]
names = [r.name for r in all_results]
sharps = [r.sharpe for r in all_results]

p_sh = plot(;
    xlabel="wing width (USD)", ylabel="Sharpe (annualized)",
    title="SPY 1DTE 20p/5c condor — Sharpe vs wing width by policy",
    legend=:bottomright, size=(1100, 500),
)
for pol in ("baseline", "skip<med", "topQ", "linear", "sigmoid")
    mask = names .== pol
    plot!(p_sh, ws[mask], sharps[mask]; marker=:circle, ms=6, lw=2, label=pol)
end
hline!(p_sh, [0]; color=:gray, ls=:dash, label="")
savefig(p_sh, joinpath(run_dir, "sharpe_by_wing.png"))
println("\n  Saved: sharpe_by_wing.png")
