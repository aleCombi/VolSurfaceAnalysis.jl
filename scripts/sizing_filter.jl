using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Flux, Statistics, Random, DataFrames

include("lib/experiment.jl")

function print_pnl_distribution(label, result)
    df = condor_trade_table(result.positions, result.pnl)
    pnls = Float64[r.PnL for r in eachrow(df) if !ismissing(r.PnL) && !ismissing(r.MaxLoss) && r.MaxLoss > 0]
    rors = Float64[r.ReturnOnRisk for r in eachrow(df) if !ismissing(r.ReturnOnRisk)]
    n = length(pnls)
    n < 2 && return

    wins = filter(>(0), pnls)
    losses = filter(<=(0), pnls)
    s = std(pnls)
    skew = s > 0 ? mean((pnls .- mean(pnls)).^3) / s^3 : 0.0

    println("\n  ── $label PnL Distribution ($n trades) ──")
    @printf("  Mean=\$%.4f  Std=\$%.4f  Skew=%.2f\n", mean(pnls), s, skew)
    @printf("  Median=\$%.4f  WinRate=%.1f%%  W/L=%.2f\n",
        quantile(pnls, 0.5), length(wins)/n*100,
        isempty(losses) ? Inf : mean(wins)/abs(mean(losses)))
    @printf("  AvgWin=\$%.4f  AvgLoss=\$%.4f\n",
        isempty(wins) ? 0.0 : mean(wins),
        isempty(losses) ? 0.0 : mean(losses))
    @printf("  Percentiles:  p1=\$%.3f  p5=\$%.3f  p10=\$%.3f  p25=\$%.3f  p50=\$%.3f  p75=\$%.3f  p90=\$%.3f  p95=\$%.3f  p99=\$%.3f\n",
        quantile(pnls, 0.01), quantile(pnls, 0.05), quantile(pnls, 0.10),
        quantile(pnls, 0.25), quantile(pnls, 0.50), quantile(pnls, 0.75),
        quantile(pnls, 0.90), quantile(pnls, 0.95), quantile(pnls, 0.99))
    @printf("  TotalPnL=\$%.2f  Min=\$%.4f  Max=\$%.4f\n", sum(pnls), minimum(pnls), maximum(pnls))

    # Histogram
    nbins = 20
    lo, hi = minimum(pnls), maximum(pnls)
    edges = range(lo, hi, length=nbins+1)
    counts = zeros(Int, nbins)
    for v in pnls
        b = clamp(searchsortedlast(collect(edges), v), 1, nbins)
        counts[b] += 1
    end
    maxc = maximum(counts)
    bw = 35
    for i in 1:nbins
        blen = round(Int, counts[i] / maxc * bw)
        bar = repeat("\u2588", blen)
        @printf("    [%+7.3f,%+7.3f) %3d \u2502%s\n", edges[i], edges[i+1], counts[i], bar)
    end

    # Losses the ML filter kept vs skipped (for comparison)
    if length(rors) >= 2
        big_losses = count(r -> r < -0.5, rors)
        @printf("  Big losses (ROI < -50%%): %d / %d (%.1f%%)\n", big_losses, n, big_losses/n*100)
    end
end

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "sizing_comparison"

SYMBOLS = [
    ("SPY",  "SPY",  1.0),
]

SPREAD_LAMBDA   = 0.7
SEEDS           = [42, 123, 7]

TRAIN_START = Date(2024, 2, 1)
TRAIN_END   = Date(2024, 12, 31)
TEST_START  = Date(2025, 1, 1)
TEST_END    = Date(2025, 12, 31)

ENTRY_TIME        = Time(10, 0)
TRAIN_ENTRY_TIMES = [Time(10, 0), Time(12, 0), Time(14, 0)]
EXPIRY_INTERVAL   = Day(1)

RATE           = 0.045
DIV_YIELD      = 0.013
BASE_MAX_LOSS  = 5.0
MAX_SPREAD_REL = 0.50
MIN_DELTA_GAP  = 0.08
PUT_DELTA      = 0.16
CALL_DELTA     = 0.16

FEATURES = Feature[
    ATMImpliedVol(; rate=RATE, div_yield=DIV_YIELD),
    DeltaSkew(0.25, :put; rate=RATE, div_yield=DIV_YIELD),
    ATMSpread(),
    SpotLogSig(; lookback=20, depth=3),
]
FEATURE_NAME = "minimal"

THRESHOLDS = [-0.5, -0.25, 0.0, 0.1, 0.2, 0.3, 0.5]
VARIANT_NAME = "Binary"

# =============================================================================
# Setup
# =============================================================================

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "$(EXPERIMENT_NAME)_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")

store = DEFAULT_STORE
results = ResultRow[]
input_dim = surface_feature_dim(FEATURES)

# =============================================================================
# Per-symbol experiment
# =============================================================================

function run_symbol(symbol, spot_sym, mult)
    println("\n", "=" ^ 60)
    println("  $symbol (spot via $spot_sym \u00d7 $mult)")
    println("=" ^ 60)

    scaled_ml = BASE_MAX_LOSS * mult

    # Data source
    all_dates = available_polygon_dates(store, symbol)
    filtered = filter(d -> d >= TRAIN_START && d <= TEST_END, all_dates)
    if length(filtered) < 50
        println("  SKIP: only $(length(filtered)) dates")
        return
    end

    all_entry_times = sort(unique([ENTRY_TIME; TRAIN_ENTRY_TIMES]))
    entry_ts = build_entry_timestamps(filtered, all_entry_times)
    entry_spots = read_polygon_spot_prices_for_timestamps(
        polygon_spot_root(store), entry_ts; symbol=spot_sym)
    if mult != 1.0
        for (k, v) in entry_spots; entry_spots[k] = v * mult; end
    end

    source = ParquetDataSource(entry_ts;
        path_for_timestamp=ts -> polygon_options_path(store, Date(ts), symbol),
        read_records=(path; where="") -> read_polygon_option_records(
            path, entry_spots; where=where, min_volume=0, warn=false,
            spread_lambda=SPREAD_LAMBDA),
        spot_root=polygon_spot_root(store),
        spot_symbol=spot_sym,
        spot_multiplier=mult)

    # Schedules
    all_ts = available_timestamps(source)
    train_sched = filter(t -> Date(t) <= TRAIN_END, all_ts)
    test_sched = filter(t -> t in Set(build_entry_timestamps(
        filter(d -> d >= TEST_START, filtered), ENTRY_TIME)), all_ts)
    println("  Train: $(length(train_sched)) | Test: $(length(test_sched))")

    # Baseline
    baseline_sel = constrained_delta_selector(PUT_DELTA, CALL_DELTA;
        rate=RATE, div_yield=DIV_YIELD, max_loss=scaled_ml,
        max_spread_rel=MAX_SPREAD_REL)

    baseline_result = backtest_strategy(
        IronCondorStrategy(test_sched, EXPIRY_INTERVAL, baseline_sel), source)
    bm = performance_metrics(baseline_result)
    push!(results, (symbol=symbol, features="\u2014", variant="Baseline", seed=0,
        sharpe=bm.sharpe, sortino=bm.sortino, roi=bm.total_roi,
        trades=bm.count, win_rate=bm.win_rate, pnl=bm.total_pnl))
    @printf("  Baseline: trades=%d sharpe=%.2f roi=%s\n",
        bm.count, bm.sharpe, fmt_metric(bm.total_roi; pct=true))
    print_pnl_distribution("Baseline", baseline_result)

    # Training data
    examples = generate_sizing_training_data(source, EXPIRY_INTERVAL,
        train_sched, baseline_sel; rate=RATE, div_yield=DIV_YIELD, surface_features=FEATURES)
    if isempty(examples)
        println("  No training data")
        return
    end
    X = hcat([e.surface_features for e in examples]...)
    Y_pnl = reshape(Float32[e.pnl for e in examples], 1, :)
    @printf("  %d training examples, %d dims\n", length(examples), input_dim)

    # Train + backtest per seed × threshold
    for seed in SEEDS
        Random.seed!(seed)
        model = Chain(Dense(input_dim => 32, relu), Dense(32 => 16, relu), Dense(16 => 1))
        model, means, stds, _ = train_model!(model, X, Y_pnl;
            epochs=200, lr=1e-3, batch_size=32, val_fraction=0.2, patience=20)

        for thresh in THRESHOLDS
            policy = binary_sizing(; threshold=thresh, quantity=1.0)
            variant = @sprintf("t=%.2f", thresh)

            strategy = IronCondorStrategy(test_sched, EXPIRY_INTERVAL, baseline_sel;
                sizer=MLSizer(model, means, stds; surface_features=FEATURES, policy=policy))

            ml_result = backtest_strategy(strategy, source)
            m = performance_metrics(ml_result)
            if m === nothing
                @printf("  %s seed=%d thresh=%.2f: no trades\n", VARIANT_NAME, seed, thresh)
                continue
            end
            push!(results, (symbol=symbol, features=FEATURE_NAME, variant=variant, seed=seed,
                sharpe=m.sharpe, sortino=m.sortino, roi=m.total_roi,
                trades=m.count, win_rate=m.win_rate, pnl=m.total_pnl))
            @printf("  seed=%d thresh=%+.2f: trades=%3d sharpe=%5.2f roi=%6.2f%% win=%.1f%% bigL=",
                seed, thresh, m.count, m.sharpe, m.total_roi*100, m.win_rate*100)

            # Quick big-loss count
            df = condor_trade_table(ml_result.positions, ml_result.pnl)
            rors = Float64[r.ReturnOnRisk for r in eachrow(df) if !ismissing(r.ReturnOnRisk)]
            big_l = count(r -> r < -0.5, rors)
            @printf("%d/%d (%.1f%%)\n", big_l, m.count, m.count > 0 ? big_l/m.count*100 : 0.0)
        end
        println()
    end
end

for (symbol, spot_sym, mult) in SYMBOLS
    run_symbol(symbol, spot_sym, mult)
end

# =============================================================================
# Summary
# =============================================================================

print_summary(results, [s[1] for s in SYMBOLS])

csv_path = joinpath(run_dir, "results.csv")
open(csv_path, "w") do io
    println(io, "symbol,features,variant,seed,sharpe,sortino,roi,trades,win_rate,pnl")
    for r in results
        @printf(io, "%s,%s,%s,%d,%.4f,%.4f,%.4f,%d,%.4f,%.4f\n",
            r.symbol, r.features, r.variant, r.seed,
            r.sharpe, r.sortino, r.roi, r.trades, r.win_rate, r.pnl)
    end
end
println("\nResults saved to: $csv_path")
println("Done.")
