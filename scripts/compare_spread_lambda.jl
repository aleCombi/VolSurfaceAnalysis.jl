using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Flux, Statistics, Random

# =============================================================================
# Parameters
# =============================================================================

ENTRY_TIME      = Time(10, 0)
TRAIN_ENTRY_TIMES = [Time(10, 0), Time(12, 0), Time(14, 0)]
EXPIRY_INTERVAL = Day(1)

TRAIN_START = Date(2024, 2, 1)
TRAIN_END   = Date(2024, 12, 31)
TEST_START  = Date(2025, 1, 1)
TEST_END    = Date(2025, 12, 31)

RATE           = 0.045
DIV_YIELD      = 0.013
BASE_MAX_LOSS  = 5.0
MAX_SPREAD_REL = 0.50
MIN_DELTA_GAP  = 0.08
PUT_DELTA      = 0.16
CALL_DELTA     = 0.16

SEEDS = [42, 123, 7]
LAMBDAS = [0.0, 0.7]

SYMBOLS = [
    ("SPY",  "SPY",  1.0),
    ("QQQ",  "QQQ",  1.0),
    ("IWM",  "IWM",  1.0),
    ("SPXW", "SPY", 10.0),
]

# Minimal feature set only (the cross-symbol winner)
function build_features(; rate, div_yield)
    Feature[
        ATMImpliedVol(; rate, div_yield),
        DeltaSkew(0.25, :put; rate, div_yield),
        ATMSpread(),
        SpotLogSig(; lookback=20, depth=3),
    ]
end

# =============================================================================
# Output
# =============================================================================

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "spread_lambda_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")

store = DEFAULT_STORE

# Collect: (symbol, lambda, seed, baseline_sharpe, binary_sharpe, baseline_roi, binary_roi, baseline_trades, binary_trades)
results = NamedTuple{(:symbol, :lambda, :seed, :base_sharpe, :bin_sharpe, :base_roi, :bin_roi, :base_trades, :bin_trades),
    Tuple{String, Float64, Int, Float64, Float64, Float64, Float64, Int, Int}}[]

for (symbol, spot_symbol, spot_multiplier) in SYMBOLS
    println("\n", "#" ^ 60)
    println("  $symbol")
    println("#" ^ 60)

    scaled_max_loss = BASE_MAX_LOSS * spot_multiplier

    all_dates = available_polygon_dates(store, symbol)
    filtered_dates = filter(d -> d >= TRAIN_START && d <= TEST_END, all_dates)
    length(filtered_dates) < 50 && continue

    all_entry_times = sort(unique([ENTRY_TIME; TRAIN_ENTRY_TIMES]))

    for lambda in LAMBDAS
        println("\n  --- lambda = $lambda ---")

        entry_ts = build_entry_timestamps(filtered_dates, all_entry_times)
        entry_spots = read_polygon_spot_prices_for_timestamps(
            polygon_spot_root(store), entry_ts; symbol=spot_symbol
        )
        if spot_multiplier != 1.0
            for (k, v) in entry_spots
                entry_spots[k] = v * spot_multiplier
            end
        end

        path_for_ts = ts -> polygon_options_path(store, Date(ts), symbol)
        read_records = (path; where="") -> read_polygon_option_records(
            path, entry_spots; where=where,
            min_volume=0, warn=false, spread_lambda=lambda
        )

        source = ParquetDataSource(
            entry_ts;
            path_for_timestamp=path_for_ts,
            read_records=read_records,
            spot_root=polygon_spot_root(store),
            spot_symbol=spot_symbol,
            spot_multiplier=spot_multiplier
        )

        all_timestamps = available_timestamps(source)
        test_dates = filter(d -> d >= TEST_START, filtered_dates)
        test_entry_ts = build_entry_timestamps(test_dates, ENTRY_TIME)
        test_entry_set = Set(test_entry_ts)

        train_schedule = filter(t -> Date(t) <= TRAIN_END, all_timestamps)
        test_schedule  = filter(t -> t in test_entry_set, all_timestamps)

        baseline_selector = constrained_delta_selector(
            PUT_DELTA, CALL_DELTA;
            rate=RATE, div_yield=DIV_YIELD,
            max_loss=scaled_max_loss,
            max_spread_rel=MAX_SPREAD_REL,
            min_delta_gap=MIN_DELTA_GAP
        )

        # Baseline
        baseline_strategy = IronCondorStrategy(test_schedule, EXPIRY_INTERVAL, baseline_selector)
        base_result = backtest_strategy(baseline_strategy, source)
        base_margin = condor_max_loss_by_key(base_result.positions)
        base_metrics = performance_metrics(base_result.positions, base_result.pnl; margin_by_key=base_margin)

        @printf("  Baseline: trades=%d sharpe=%.2f roi=%s\n",
            base_metrics.count, base_metrics.sharpe, fmt_metric(base_metrics.total_roi; pct=true))

        sf = build_features(; rate=RATE, div_yield=DIV_YIELD)
        input_dim = surface_feature_dim(sf)

        examples = generate_sizing_training_data(
            source, EXPIRY_INTERVAL, train_schedule, baseline_selector;
            rate=RATE, div_yield=DIV_YIELD,
            surface_features=sf
        )

        if isempty(examples)
            println("  SKIP: no training data")
            continue
        end

        X = hcat([e.surface_features for e in examples]...)
        Y = reshape(Float32[e.pnl for e in examples], 1, :)

        for seed in SEEDS
            Random.seed!(seed)

            model = Chain(
                Dense(input_dim => 32, relu),
                Dense(32 => 16, relu),
                Dense(16 => 1),
            )

            model, feat_means, feat_stds, _ = train_model!(
                model, X, Y;
                epochs=200, lr=1e-3, batch_size=32, val_fraction=0.2, patience=20
            )

            strategy = SizedIronCondorStrategy(
                test_schedule, EXPIRY_INTERVAL, baseline_selector,
                model, feat_means, feat_stds;
                surface_features=sf,
                sizing_policy=binary_sizing(; threshold=0.0, quantity=1.0),
                debug=false
            )

            result = backtest_strategy(strategy, source)
            margin = condor_max_loss_by_key(result.positions)
            metrics = performance_metrics(result.positions, result.pnl; margin_by_key=margin)

            push!(results, (symbol=symbol, lambda=lambda, seed=seed,
                base_sharpe=base_metrics.sharpe, bin_sharpe=metrics.sharpe,
                base_roi=base_metrics.total_roi, bin_roi=metrics.total_roi,
                base_trades=base_metrics.count, bin_trades=metrics.count))

            @printf("  seed=%d: binary trades=%d sharpe=%.2f roi=%s\n",
                seed, metrics.count, metrics.sharpe, fmt_metric(metrics.total_roi; pct=true))
        end
    end
end

# =============================================================================
# Summary table
# =============================================================================

println("\n\n", "=" ^ 95)
println("  SPREAD LAMBDA COMPARISON — Minimal (8d) Binary Sizing (avg over $(length(SEEDS)) seeds)")
println("=" ^ 95)
@printf("  %-6s  %6s  %8s  %8s  %8s  %8s  %8s  %8s\n",
    "Symbol", "Lambda", "B.Sharpe", "ML.Sharpe", "Lift", "B.ROI", "ML.ROI", "ML.Trades")
println("  ", "-" ^ 89)

for (symbol, _, _) in SYMBOLS
    for lambda in LAMBDAS
        rows = filter(r -> r.symbol == symbol && r.lambda == lambda, results)
        isempty(rows) && continue
        bs = rows[1].base_sharpe
        avg_ms = mean(r.bin_sharpe for r in rows)
        avg_roi = mean(r.bin_roi for r in rows)
        avg_trades = mean(r.bin_trades for r in rows)
        base_roi = rows[1].base_roi
        @printf("  %-6s  %6.1f  %8.2f  %8.2f  %+7.2f  %7.1f%%  %7.1f%%  %8.0f\n",
            symbol, lambda, bs, avg_ms, avg_ms - bs,
            base_roi * 100, avg_roi * 100, avg_trades)
    end
    println()
end
println("=" ^ 95)

println("\nDone.")
