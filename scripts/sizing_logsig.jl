# scripts/sizing_logsig.jl
#
# Translate a SpotMinuteLogSig-based ridge signal into a position-sizing strategy
# for a fixed delta short (strangle or iron condor). Walk-forward rolling train/test,
# compare cumulative PnL of fixed-size baseline vs ridge-driven sizing policies
# (skip<median, top-quartile, linear z-clip, sigmoid).
#
# Replaces 5 prior scripts:
#   spy_sizing_logsig.jl                (SPY 1-DTE strangle)
#   spy_sizing_logsig_2h.jl             (SPY 2h strangle)
#   spy_sizing_logsig_4h.jl             (SPY 4h strangle)
#   spy_sizing_logsig_condor.jl         (SPY 1-DTE condor, symmetric $5 wings)
#   spy_sizing_logsig_condor_asym.jl    (SPY 1-DTE condor, asymmetric wings)
#
# All knobs below can be overridden via ENV. Common presets:
#
#   # SPY 1-DTE strangle (canonical; mean IC ≈ +0.095)
#   SYM=SPY START_YEAR=2019 ENTRY_HOUR=12 TENOR=1d STRUCTURE=strangle \
#       LOOKBACK=3 MIN_PTS=100 TRAIN_DAYS=1095 MIN_TRAIN=200
#
#   # SPY 2h strangle
#   SYM=SPY ENTRY_HOUR=14 TENOR=2h MAX_TAU_DAYS=0.5 STRUCTURE=strangle
#
#   # SPY 4h strangle
#   SYM=SPY ENTRY_HOUR=12 TENOR=4h MAX_TAU_DAYS=0.5 STRUCTURE=strangle
#
#   # SPY 1-DTE symmetric condor ($5 wings)
#   SYM=SPY START_YEAR=2019 ENTRY_HOUR=12 TENOR=1d STRUCTURE=condor \
#       WING_WIDTH=5 TRAIN_DAYS=1095 MIN_TRAIN=200
#
#   # SPY 1-DTE asymmetric condor (put wing 10, call wing 3)
#   SYM=SPY STRUCTURE=condor WING_PUT=10 WING_CALL=3

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, Random, Plots
include(joinpath(@__DIR__, "lib", "experiment.jl"))

# =============================================================================
# Configuration
# =============================================================================

SYMBOL              = get(ENV, "SYM", "SPY")
START_DATE          = Date(parse(Int, get(ENV, "START_YEAR", "2019")), 1, 1)
END_DATE            = Date(parse(Int, get(ENV, "END_YEAR", "2024")),
                           parse(Int, get(ENV, "END_MONTH", "1")),
                           parse(Int, get(ENV, "END_DAY",   "31")))
ENTRY_TIME          = Time(parse(Int, get(ENV, "ENTRY_HOUR", "12")), 0)

_tenor_str          = lowercase(get(ENV, "TENOR", "1d"))
EXPIRY_INTERVAL     = if endswith(_tenor_str, "d")
    Day(parse(Int, _tenor_str[1:end-1]))
elseif endswith(_tenor_str, "h")
    Hour(parse(Int, _tenor_str[1:end-1]))
else
    error("Unknown TENOR: $_tenor_str")
end

SPREAD_LAMBDA       = parse(Float64, get(ENV, "SPREAD_LAMBDA", "0.7"))
RATE                = parse(Float64, get(ENV, "RATE", "0.045"))
DIV_YIELD           = parse(Float64, get(ENV, "DIV",  "0.013"))
# Default to a large value so daily tenor isn't filtered; intraday presets override.
MAX_TAU_DAYS        = parse(Float64, get(ENV, "MAX_TAU_DAYS", "5.0"))

PUT_DELTA           = parse(Float64, get(ENV, "PUT_DELTA",  "0.20"))
CALL_DELTA          = parse(Float64, get(ENV, "CALL_DELTA", "0.05"))

# Structure: "strangle" (naked 2-leg short) or "condor" (4-leg short inner + long wings).
STRUCTURE           = lowercase(get(ENV, "STRUCTURE", "strangle"))
STRUCTURE in ("strangle", "condor") || error("STRUCTURE must be strangle or condor")

# Wing widths (condor only). Symmetric via WING_WIDTH; asymmetric via WING_PUT/WING_CALL.
WING_WIDTH          = parse(Float64, get(ENV, "WING_WIDTH", "5.0"))
WING_PUT_WIDTH      = parse(Float64, get(ENV, "WING_PUT",   string(WING_WIDTH)))
WING_CALL_WIDTH     = parse(Float64, get(ENV, "WING_CALL",  string(WING_WIDTH)))

LOGSIG_LOOKBACK_HRS = parse(Int, get(ENV, "LOOKBACK", "3"))
LOGSIG_MIN_POINTS   = parse(Int, get(ENV, "MIN_PTS",  "100"))
FEATURES            = Feature[
    SpotMinuteLogSig(lookback_hours=LOGSIG_LOOKBACK_HRS, depth=3, min_points=LOGSIG_MIN_POINTS),
]

TRAIN_WINDOW_DAYS   = parse(Int, get(ENV, "TRAIN_DAYS", "1095"))
TEST_WINDOW_DAYS    = parse(Int, get(ENV, "TEST_DAYS",  "90"))
STEP_DAYS           = parse(Int, get(ENV, "STEP_DAYS",  "90"))
MIN_TRAIN_ENTRIES   = parse(Int, get(ENV, "MIN_TRAIN",  "200"))
MIN_TEST_ENTRIES    = parse(Int, get(ENV, "MIN_TEST",   "20"))

MAX_SIZE            = parse(Float64, get(ENV, "MAX_SIZE", "2.0"))

# =============================================================================
# Output dir
# =============================================================================

run_tag = get(ENV, "TAG", "$(SYMBOL)_$(STRUCTURE)_$(_tenor_str)")
run_ts  = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "sizing_logsig_$(run_tag)_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")
println("\n  $SYMBOL $STRUCTURE  $START_DATE → $END_DATE  entry $(ENTRY_TIME)  tenor=$(EXPIRY_INTERVAL)")
println("  Δ_put=$PUT_DELTA Δ_call=$CALL_DELTA  max_tau=$(MAX_TAU_DAYS)d")
if STRUCTURE == "condor"
    println("  wings: put=$(WING_PUT_WIDTH)  call=$(WING_CALL_WIDTH)")
end
println("  logsig lookback=$(LOGSIG_LOOKBACK_HRS)h  min_points=$(LOGSIG_MIN_POINTS)")
println("  walk-forward train=$(TRAIN_WINDOW_DAYS)d test=$(TEST_WINDOW_DAYS)d step=$(STEP_DAYS)d")

# =============================================================================
# Data source
# =============================================================================

println("\nLoading $SYMBOL …")
(; source, sched) = polygon_parquet_source(SYMBOL;
    start_date=START_DATE, end_date=END_DATE, entry_time=ENTRY_TIME,
    rate=RATE, div_yield=DIV_YIELD, spread_lambda=SPREAD_LAMBDA,
)

# =============================================================================
# Build dataset
# =============================================================================

dates  = Date[]
feats  = Vector{Vector{Float32}}()
pnls   = Float64[]
spots  = Float64[]
n_total     = 0
n_skip_feat = 0
n_skip_strk = 0

println("\nBuilding dataset ($STRUCTURE)...")
each_entry(source, EXPIRY_INTERVAL, sched; clear_cache=true) do ctx, settlement
    ismissing(settlement) && return
    global n_total += 1

    fvec = extract_surface_features(ctx, FEATURES)
    if fvec === nothing
        global n_skip_feat += 1
        return
    end

    dctx = delta_context(ctx; rate=RATE, div_yield=DIV_YIELD)
    dctx === nothing && return
    dctx.tau * 365.25 > MAX_TAU_DAYS && (global n_skip_strk += 1; return)
    spot = dctx.spot

    sp_K = delta_strike(dctx, -PUT_DELTA,  Put)
    sc_K = delta_strike(dctx,  CALL_DELTA, Call)
    (sp_K === nothing || sc_K === nothing) && (global n_skip_strk += 1; return)

    positions = if STRUCTURE == "strangle"
        open_strangle_positions(ctx, sp_K, sc_K; direction=-1)
    else  # condor
        lp_K = nearest_otm_strike(dctx, sp_K, WING_PUT_WIDTH,  Put)
        lc_K = nearest_otm_strike(dctx, sc_K, WING_CALL_WIDTH, Call)
        (lp_K === nothing || lc_K === nothing) && return
        open_condor_positions(ctx, sp_K, sc_K, lp_K, lc_K)
    end
    expected_legs = STRUCTURE == "strangle" ? 2 : 4
    if length(positions) != expected_legs
        global n_skip_strk += 1
        return
    end

    pnl_usd = settle(positions, Float64(settlement)) * spot
    push!(dates, Date(ctx.surface.timestamp))
    push!(feats, fvec)
    push!(pnls,  pnl_usd)
    push!(spots, spot)
end
println()
n_kept = length(dates)
@printf "  %d entries → kept %d (skipped: %d feat, %d strikes)\n" n_total n_kept n_skip_feat n_skip_strk
n_kept == 0 && error("No usable entries")

# =============================================================================
# Walk-forward fit + predict
# =============================================================================

ord   = sortperm(dates)
dates = dates[ord]
feats = feats[ord]
pnls  = pnls[ord]
spots = spots[ord]

X_all = Float32.(hcat(feats...))                # (n_features, n_kept)
Y_all = reshape(Float32.(pnls), 1, length(pnls))

folds = build_folds(dates;
    train_days=TRAIN_WINDOW_DAYS, test_days=TEST_WINDOW_DAYS, step_days=STEP_DAYS,
    min_train=MIN_TRAIN_ENTRIES, min_test=MIN_TEST_ENTRIES,
)
println("\nFolds: $(length(folds))")

all_dates_oos  = Date[]
all_y_oos      = Float64[]
all_pred_oos   = Float64[]
all_train_μ    = Float64[]
all_train_σ    = Float64[]
all_train_med  = Float64[]
all_train_q75  = Float64[]

for f in folds
    Random.seed!(42)
    Xtr = X_all[:, f.train_mask]
    Ytr = Y_all[:, f.train_mask]
    Xte = X_all[:, f.test_mask]
    Yte = Y_all[:, f.test_mask]
    Dte = dates[f.test_mask]

    model, μ, σ, _ = train_ridge!(nothing, Xtr, Ytr; alpha=0.0, val_fraction=0.2)
    σ_safe = [s > 1e-8 ? s : one(s) for s in σ]

    Xtr_norm = (Xtr .- Float32.(μ)) ./ Float32.(σ_safe)
    train_preds = Float64.(vec(model(Xtr_norm)))
    tμ = mean(train_preds); tσ = std(train_preds)
    tmed = median(train_preds)
    tq75 = sort(train_preds)[max(1, round(Int, 0.75 * length(train_preds)))]

    Xte_norm = (Xte .- Float32.(μ)) ./ Float32.(σ_safe)
    preds_te = Float64.(vec(model(Xte_norm)))
    n_te = length(preds_te)

    append!(all_dates_oos, Dte)
    append!(all_y_oos,   Float64.(vec(Yte)))
    append!(all_pred_oos, preds_te)
    append!(all_train_μ,   fill(tμ,   n_te))
    append!(all_train_σ,   fill(tσ,   n_te))
    append!(all_train_med, fill(tmed, n_te))
    append!(all_train_q75, fill(tq75, n_te))
end

n_oos = length(all_y_oos)
println("Pooled OOS: $n_oos samples  ($(all_dates_oos[1]) → $(all_dates_oos[end]))")

# =============================================================================
# Sizing policies
# =============================================================================

sigmoid_fn(z) = 1 / (1 + exp(-z))

policies = [
    ("baseline (size=1)",     (p, μ, σ, m, q) -> 1.0),
    ("binary skip-below-med", (p, μ, σ, m, q) -> p > m ? 1.0 : 0.0),
    ("binary top-quartile",   (p, μ, σ, m, q) -> p > q ? 1.0 : 0.0),
    ("linear z-clip[0,$MAX_SIZE]", (p, μ, σ, m, q) ->
        clamp((p - μ) / max(σ, 1e-9), 0.0, MAX_SIZE)),
    ("sigmoid $(MAX_SIZE)·σ(z)", (p, μ, σ, m, q) ->
        MAX_SIZE * sigmoid_fn((p - μ) / max(σ, 1e-9))),
]

sized_results = NamedTuple[]
for (name, fn) in policies
    sizes = [fn(all_pred_oos[i], all_train_μ[i], all_train_σ[i],
                all_train_med[i], all_train_q75[i]) for i in 1:n_oos]
    sized_pnls = sizes .* all_y_oos

    n_trades = count(s -> s > 0, sizes)
    avg_size = n_trades > 0 ? mean(sizes[sizes .> 0]) : 0.0
    cum = cumsum(sized_pnls)
    running_max = accumulate(max, cum)
    max_dd = -minimum(cum .- running_max)
    s = std(sized_pnls)
    sharpe = s > 0 ? mean(sized_pnls) / s * sqrt(252) : 0.0

    push!(sized_results, (
        name=name, sizes=sizes, sized_pnls=sized_pnls,
        n_trades=n_trades, trade_rate=n_trades / n_oos,
        avg_size=avg_size,
        total=sum(sized_pnls), avg_per_day=mean(sized_pnls),
        sharpe=sharpe, max_dd=max_dd,
    ))
end

# =============================================================================
# Print comparison
# =============================================================================

println("\n", "=" ^ 100)
println("  SIZING COMPARISON — $SYMBOL $STRUCTURE, pooled OOS over $(length(folds)) folds, $n_oos samples")
println("=" ^ 100)
@printf "  %-28s  %7s  %5s  %8s  %10s  %8s  %8s  %8s\n" "policy" "trades" "rate" "avg_sz" "total_PnL" "/day" "Sharpe" "maxDD"
println("  ", "-" ^ 96)
for r in sized_results
    @printf "  %-28s  %7d  %4.0f%%  %8.3f  %+10.0f  %+8.2f  %+8.2f  %8.0f\n" (
        r.name, r.n_trades, 100*r.trade_rate, r.avg_size,
        r.total, r.avg_per_day, r.sharpe, r.max_dd,
    )...
end

# =============================================================================
# Plot — cumulative PnL by policy
# =============================================================================

println("\n  Saving plots...")

ord_d = sortperm(all_dates_oos)
d_sorted = all_dates_oos[ord_d]

struct_label = STRUCTURE == "strangle" ? "strangle" :
    "condor (wings $(WING_PUT_WIDTH)p/$(WING_CALL_WIDTH)c)"
p_cum = plot(;
    xlabel="date", ylabel="cumulative PnL (USD)",
    title="$SYMBOL $(EXPIRY_INTERVAL) $(round(Int,100*PUT_DELTA))p/$(round(Int,100*CALL_DELTA))c $struct_label — sizing policies",
    legend=:topleft, size=(1300, 600),
)
colors = [:gray, :steelblue, :purple, :seagreen, :darkorange]
for (i, r) in enumerate(sized_results)
    cum = cumsum(r.sized_pnls[ord_d])
    plot!(p_cum, d_sorted, cum;
        label=@sprintf("%-26s  Sharpe=%+.2f", r.name, r.sharpe),
        lw=(startswith(r.name, "baseline") ? 3 : 2),
        color=colors[i], ls=(startswith(r.name, "baseline") ? :dash : :solid),
    )
end
hline!(p_cum, [0]; color=:black, ls=:dot, label="")
savefig(p_cum, joinpath(run_dir, "cumulative_pnl.png"))
println("    cumulative_pnl.png")

p_sz = plot(layout=(1, 2), size=(1200, 400),
    title=["sizes — linear" "sizes — sigmoid"],
)
linear_idx  = findfirst(r -> startswith(r.name, "linear"),  sized_results)
sigmoid_idx = findfirst(r -> startswith(r.name, "sigmoid"), sized_results)
linear_idx !== nothing && histogram!(p_sz[1], sized_results[linear_idx].sizes;
    bins=30, label="", color=:seagreen, alpha=0.7)
sigmoid_idx !== nothing && histogram!(p_sz[2], sized_results[sigmoid_idx].sizes;
    bins=30, label="", color=:darkorange, alpha=0.7)
savefig(p_sz, joinpath(run_dir, "size_histograms.png"))
println("    size_histograms.png")

println("\n  Output: $run_dir")
