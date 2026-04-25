# scripts/strangle_rolling.jl
#
# Rolling-delta selector for short strangles (no wings) — picks
#   (put_delta, call_delta) per fold by maximising
#       score = mean(PnL) − z · |CVaR_α(PnL)|
# over the trailing training window. z=0 reproduces the naive
# max-mean selector; z>0 penalises high-mean / heavy-left-tail combos.
#
# Replaces strangle_rolling_diagnostic_1d.jl and
# strangle_rolling_regularized_1d.jl. Behaviour is selected by Z_VALUES:
#
#   Z_VALUES=0                       → diagnostic mode (single z; per-fold
#                                       overfit gap, IS→OOS Spearman,
#                                       baseline rank, fold_diagnostic.csv)
#   Z_VALUES=0,0.1,0.3,1,3 (default) → regularised sweep (per-year + summary
#                                       tables across z, equity overlay,
#                                       per-year Sharpe heatmap)
#
# All the iteration / scoring / reporting lives in
# `VolSurfaceAnalysis.run_strangle_rolling_ensemble` +
# `VolSurfaceAnalysis.report_strangle_rolling`.

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates

# =============================================================================
# Configuration
# =============================================================================

SYMBOL          = get(ENV, "SYM", "SPY")
START_DATE      = Date(get(ENV, "START_DATE", "2014-06-02"))
END_DATE        = Date(get(ENV, "END_DATE",   "2026-03-27"))
ENTRY_TIME      = Time(parse(Int, get(ENV, "ENTRY_HOUR", "14")), 0)
EXPIRY_INTERVAL = Day(parse(Int, get(ENV, "EXPIRY_DAYS", "1")))
MAX_TAU_DAYS    = parse(Float64, get(ENV, "MAX_TAU_DAYS", "2.0"))
SPREAD_LAMBDA   = parse(Float64, get(ENV, "SPREAD_LAMBDA", "0.7"))
RATE            = parse(Float64, get(ENV, "RATE", "0.045"))
DIV_YIELD       = parse(Float64, get(ENV, "DIV", "0.013"))

PUT_DELTAS      = collect(0.05:0.05:0.40)
CALL_DELTAS     = collect(0.05:0.05:0.40)
TRAIN_DAYS      = parse(Int, get(ENV, "TRAIN_DAYS", "90"))
TEST_DAYS       = parse(Int, get(ENV, "TEST_DAYS",  "30"))
STEP_DAYS       = parse(Int, get(ENV, "STEP_DAYS",  "30"))

Z_VALUES        = parse.(Float64, strip.(split(get(ENV, "Z_VALUES", "0,0.1,0.3,1,3"), ",")))
CVAR_ALPHA      = parse(Float64, get(ENV, "CVAR_ALPHA", "0.05"))
BASELINE_COMBO  = (0.20, 0.05)

# =============================================================================
# Output dir + announce
# =============================================================================

mode_tag = length(Z_VALUES) == 1 ? "diag" : "reg"
run_ts   = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir  = joinpath(@__DIR__, "runs", "strangle_rolling_$(mode_tag)_$(SYMBOL)_$run_ts")
mkpath(run_dir)
println("Output: $run_dir   mode=$(mode_tag == "diag" ? "diagnostic" : "regularized sweep")")
println("\n  $SYMBOL  $START_DATE → $END_DATE   strangle (no wings)")
println("  grid: $(length(PUT_DELTAS))×$(length(CALL_DELTAS)) combos   train=$TRAIN_DAYS d / test=$TEST_DAYS d / step=$STEP_DAYS d")
println("  z values: $Z_VALUES   α=$CVAR_ALPHA   baseline=$BASELINE_COMBO")

# =============================================================================
# Run + report (no loops — all in src)
# =============================================================================

println("\nLoading $SYMBOL …")
(; source, sched) = polygon_parquet_source(SYMBOL;
    start_date=START_DATE, end_date=END_DATE, entry_time=ENTRY_TIME,
    rate=RATE, div_yield=DIV_YIELD, spread_lambda=SPREAD_LAMBDA,
)

println("\nRunning rolling ensemble …")
ensemble = run_strangle_rolling_ensemble(source, sched, EXPIRY_INTERVAL;
    put_deltas=PUT_DELTAS, call_deltas=CALL_DELTAS,
    train_days=TRAIN_DAYS, test_days=TEST_DAYS, step_days=STEP_DAYS,
    rate=RATE, div_yield=DIV_YIELD, max_tau_days=MAX_TAU_DAYS,
    z_values=Z_VALUES, cvar_alpha=CVAR_ALPHA,
    baseline_combo=BASELINE_COMBO,
)

report_strangle_rolling(ensemble;
    run_dir=run_dir, baseline_combo=BASELINE_COMBO,
    title_prefix="$SYMBOL strangle",
)
