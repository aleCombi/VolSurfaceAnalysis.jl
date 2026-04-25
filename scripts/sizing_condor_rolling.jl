# scripts/sizing_condor_rolling.jl
#
# Walk-forward (rolling) iron-condor wing-width sweep with sizing policies.
# Replaces the per-symbol scripts spy_sizing_condor_rolling_2h.jl and
# qqq_sizing_condor_rolling_2h.jl. Every knob is ENV-overridable.
#
# Pipeline: short legs at fixed delta (default 20p/5c), wing widths swept,
# walk-forward ridge on SpotMinuteLogSig + base features predicts per-entry
# PnL, sizing policies (baseline / skip<med / topQ / linear / sigmoid) compared.
#
# Presets (preserve original behavior):
#   # SPY 2h 1DTE-style intraday baseline:
#   SYM=SPY DIV=0.013
#
#   # QQQ 2h:
#   SYM=QQQ DIV=0.006

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, Random, Plots
include(joinpath(@__DIR__, "lib", "experiment.jl"))

# =============================================================================
# Configuration (override via ENV)
# =============================================================================

SYMBOL              = get(ENV, "SYM", "SPY")
START_DATE          = Date(parse(Int, get(ENV, "START_YEAR", "2017")),
                           parse(Int, get(ENV, "START_MONTH", "1")),
                           parse(Int, get(ENV, "START_DAY",   "1")))
END_DATE            = Date(parse(Int, get(ENV, "END_YEAR",   "2024")),
                           parse(Int, get(ENV, "END_MONTH",  "1")),
                           parse(Int, get(ENV, "END_DAY",    "31")))

ENTRY_TIMES         = [Time(parse(Int, strip(h)), 0)
                       for h in split(get(ENV, "ENTRY_HOURS", "14"), ",")]

_tenor_str          = lowercase(get(ENV, "TENOR", "2h"))
EXPIRY_INTERVAL     = if endswith(_tenor_str, "d")
    Day(parse(Int, _tenor_str[1:end-1]))
elseif endswith(_tenor_str, "h")
    Hour(parse(Int, _tenor_str[1:end-1]))
else
    error("Unknown TENOR: $_tenor_str (use e.g. \"1d\" or \"2h\")")
end

SPREAD_LAMBDA       = parse(Float64, get(ENV, "SPREAD_LAMBDA", "0.7"))
RATE                = parse(Float64, get(ENV, "RATE", "0.045"))
DIV_YIELD           = parse(Float64, get(ENV, "DIV",  "0.013"))
MAX_TAU_DAYS        = parse(Float64, get(ENV, "MAX_TAU_DAYS", "0.5"))

PUT_DELTA           = parse(Float64, get(ENV, "PUT_DELTA",  "0.20"))
CALL_DELTA          = parse(Float64, get(ENV, "CALL_DELTA", "0.05"))
WING_WIDTHS         = [parse(Float64, strip(w))
                       for w in split(get(ENV, "WING_WIDTHS", "1,2,3,5,8,12"), ",")]

LOGSIG_LOOKBACK_HRS = parse(Int, get(ENV, "LOOKBACK", "5"))
LOGSIG_MIN_POINTS   = parse(Int, get(ENV, "MIN_PTS",  "150"))
FEATURES            = Feature[
    SpotMinuteLogSig(lookback_hours=LOGSIG_LOOKBACK_HRS, depth=3, min_points=LOGSIG_MIN_POINTS),
]

TRAIN_WINDOW_DAYS   = parse(Int, get(ENV, "TRAIN_DAYS", "365"))
TEST_WINDOW_DAYS    = parse(Int, get(ENV, "TEST_DAYS",  "90"))
STEP_DAYS           = parse(Int, get(ENV, "STEP_DAYS",  "90"))
MIN_TRAIN_ENTRIES   = parse(Int, get(ENV, "MIN_TRAIN",  "80"))
MIN_TEST_ENTRIES    = parse(Int, get(ENV, "MIN_TEST",   "20"))
MAX_SIZE            = parse(Float64, get(ENV, "MAX_SIZE", "2.0"))

# =============================================================================
# Output dir
# =============================================================================

run_tag = get(ENV, "TAG", "$(SYMBOL)_$(_tenor_str)")
run_ts  = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "sizing_condor_rolling_$(run_tag)_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")
println("\n  $SYMBOL  $START_DATE → $END_DATE  entries=$ENTRY_TIMES  tenor=$(EXPIRY_INTERVAL)")
println("  Δ_put=$PUT_DELTA  Δ_call=$CALL_DELTA  wings=$WING_WIDTHS  max_tau=$(MAX_TAU_DAYS)d")

# =============================================================================
# Data source
# =============================================================================

println("\nLoading $SYMBOL …")
(; source, sched) = polygon_parquet_source(SYMBOL;
    start_date=START_DATE, end_date=END_DATE, entry_time=ENTRY_TIMES,
    rate=RATE, div_yield=DIV_YIELD, spread_lambda=SPREAD_LAMBDA,
)
println("  $(length(sched)) entry timestamps")

# =============================================================================
# Build dataset: for each entry compute PnL for ALL wing widths simultaneously
# =============================================================================

dates         = Date[]
feats         = Vector{Vector{Float32}}()
spots         = Float64[]
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
    dctx === nothing && return
    dctx.tau * 365.25 > MAX_TAU_DAYS && (global n_skip += 1; return)
    spot = dctx.spot

    sp_K = delta_strike(dctx, -PUT_DELTA,  Put)
    sc_K = delta_strike(dctx,  CALL_DELTA, Call)
    (sp_K === nothing || sc_K === nothing) && (global n_skip += 1; return)

    pnls_this_entry = Float64[]
    ok = true
    for ww in WING_WIDTHS
        lp_K = nearest_otm_strike(dctx, sp_K, ww, Put)
        lc_K = nearest_otm_strike(dctx, sc_K, ww, Call)
        (lp_K === nothing || lc_K === nothing) && (ok = false; break)
        cp = open_condor_positions(ctx, sp_K, sc_K, lp_K, lc_K)
        length(cp) == 4 || (ok = false; break)
        push!(pnls_this_entry, settle(cp, Float64(settlement)) * spot)
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
        ("baseline", (p, μ, σ, m, q) -> 1.0),
        ("skip<med", (p, μ, σ, m, q) -> p > m ? 1.0 : 0.0),
        ("topQ",     (p, μ, σ, m, q) -> p > q ? 1.0 : 0.0),
        ("linear",   (p, μ, σ, m, q) -> clamp((p - μ) / max(σ, 1e-9), 0.0, max_size)),
        ("sigmoid",  (p, μ, σ, m, q) -> max_size * sigmoid_fn((p - μ) / max(σ, 1e-9))),
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

ord          = sortperm(dates)
dates_sorted = dates[ord]
feats_sorted = feats[ord]
spots_sorted = spots[ord]
pnls_sorted  = pnls_by_width[ord]
X_all        = Float32.(hcat(feats_sorted...))
folds = build_folds(dates_sorted;
    train_days=TRAIN_WINDOW_DAYS, test_days=TEST_WINDOW_DAYS, step_days=STEP_DAYS,
    min_train=MIN_TRAIN_ENTRIES, min_test=MIN_TEST_ENTRIES,
)
println("\nFolds: $(length(folds))")

println("\n", "=" ^ 110)
println("  WING WIDTH SWEEP — $SYMBOL $(EXPIRY_INTERVAL) $(round(Int,100*PUT_DELTA))p/$(round(Int,100*CALL_DELTA))c iron condor")
println("=" ^ 110)
@printf "  %5s  %-10s  %7s  %5s  %10s  %8s  %8s  %10s\n" (
    "wing", "policy", "trades", "rate", "totalPnL", "/day", "Sharpe", "maxDD"
)...
println("  ", "-" ^ 96)

best_per_width = Dict()
all_results    = NamedTuple[]

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
# Plot: Sharpe vs wing width by policy
# =============================================================================

ws    = [r.wing for r in all_results]
names = [r.name for r in all_results]
sharps = [r.sharpe for r in all_results]

p_sh = plot(;
    xlabel="wing width (USD)", ylabel="Sharpe (annualized)",
    title="$SYMBOL $(EXPIRY_INTERVAL) $(round(Int,100*PUT_DELTA))p/$(round(Int,100*CALL_DELTA))c condor — Sharpe vs wing by policy",
    legend=:bottomright, size=(1100, 500),
)
for pol in ("baseline", "skip<med", "topQ", "linear", "sigmoid")
    mask = names .== pol
    plot!(p_sh, ws[mask], sharps[mask]; marker=:circle, ms=6, lw=2, label=pol)
end
hline!(p_sh, [0]; color=:gray, ls=:dash, label="")
savefig(p_sh, joinpath(run_dir, "sharpe_by_wing.png"))
println("\n  Saved: sharpe_by_wing.png")

# =============================================================================
# Per-month Sharpe — focus on the best (wing, policy) combo
# =============================================================================

best_overall = argmax([r.sharpe for r in all_results])
best_ww   = all_results[best_overall].wing
best_pol  = all_results[best_overall].name
println("\n  Best overall: wing=$best_ww  policy=$best_pol  Sharpe=$(round(all_results[best_overall].sharpe, digits=2))")

best_ww_idx = findfirst(==(best_ww), WING_WIDTHS)
Y_best = reshape(Float32.([p[best_ww_idx] for p in pnls_sorted]), 1, length(pnls_sorted))
res_best = run_walkforward(dates_sorted, X_all, Y_best, folds)

size_fn = if best_pol == "baseline"
    (p, μ, σ, m, q) -> 1.0
elseif best_pol == "skip<med"
    (p, μ, σ, m, q) -> p > m ? 1.0 : 0.0
elseif best_pol == "topQ"
    (p, μ, σ, m, q) -> p > q ? 1.0 : 0.0
elseif best_pol == "linear"
    (p, μ, σ, m, q) -> clamp((p - μ)/max(σ, 1e-9), 0.0, MAX_SIZE)
else  # sigmoid
    (p, μ, σ, m, q) -> MAX_SIZE * sigmoid_fn((p - μ)/max(σ, 1e-9))
end
sizes_best = [size_fn(res_best.preds[i], res_best.μ[i], res_best.σ[i],
                      res_best.med[i], res_best.q75[i]) for i in eachindex(res_best.preds)]
sized_pnls_best = sizes_best .* res_best.y

println("\n  ──────────────────────────────────────────────────")
println("  PER-MONTH Sharpe — $SYMBOL  (wing=$best_ww  policy=$best_pol)")
println("  ──────────────────────────────────────────────────")
@printf "  %-9s  %5s  %+10s  %+8s\n" "month" "n" "totalPnL" "Sharpe"
println("  " * "─"^46)
months = unique(yearmonth.(res_best.dates))
for m in sort(months)
    mask = [ym == m for ym in yearmonth.(res_best.dates)]
    ys = sized_pnls_best[mask]
    n = length(ys)
    n < 3 && continue
    tot = sum(ys)
    s = std(ys)
    sh = s > 0 ? mean(ys) / s * sqrt(252) : 0.0
    @printf "  %4d-%02d   %5d  %+10.0f  %+8.2f\n" m[1] m[2] n tot sh
end
