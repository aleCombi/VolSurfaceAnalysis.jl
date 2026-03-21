using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Flux, Statistics, Random

# =============================================================================
# Parameters (same as comparison scripts)
# =============================================================================

ENTRY_TIME      = Time(10, 0)
TRAIN_ENTRY_TIMES = [Time(10, 0), Time(12, 0), Time(14, 0)]
EXPIRY_INTERVAL = Day(1)
SPREAD_LAMBDA   = 0.7

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

SEEDS = [42, 123, 7, 99, 256]  # 5 seeds for more stability

SYMBOLS = [
    ("SPY",  "SPY",  1.0),
    ("QQQ",  "QQQ",  1.0),
    ("IWM",  "IWM",  1.0),
    ("SPXW", "SPY", 10.0),
]

function build_features(; rate, div_yield)
    Feature[
        ATMImpliedVol(; rate, div_yield),
        DeltaSkew(0.25, :put; rate, div_yield),
        ATMSpread(),
        SpotLogSig(; lookback=20, depth=3),
    ]
end

"""Compute per-condor ROI = pnl / max_loss for each condor group."""
function condor_rois(positions, pnl)
    # Group PnL by condor key (same key as condor_max_loss_by_key)
    key_fn = pos -> (pos.entry_timestamp, pos.trade.expiry)
    pnl_by_key = Dict{Any, Float64}()
    for (pos, p) in zip(positions, pnl)
        ismissing(p) && continue
        k = key_fn(pos)
        pnl_by_key[k] = get(pnl_by_key, k, 0.0) + Float64(p)
    end

    margin_by_key = condor_max_loss_by_key(positions)

    rois = Float64[]
    for (k, condor_pnl) in pnl_by_key
        margin = get(margin_by_key, k, NaN)
        if margin > 0 && isfinite(margin)
            push!(rois, condor_pnl / margin)
        end
    end
    return rois
end

function tail_stats(pnls)
    n = length(pnls)
    n == 0 && return (;n=0, mean=NaN, std=NaN, p5=NaN, p10=NaN, p25=NaN,
        max_loss=NaN, loss_pct=NaN, avg_loss=NaN, skew=NaN)
    sorted = sort(pnls)
    losses = filter(x -> x < 0, pnls)
    m = mean(pnls)
    s = std(pnls)
    sk = length(pnls) < 3 ? NaN : mean(((pnls .- m) ./ s) .^ 3)
    (
        n = n,
        mean = m,
        std = s,
        p5 = sorted[max(1, ceil(Int, 0.05 * n))],
        p10 = sorted[max(1, ceil(Int, 0.10 * n))],
        p25 = sorted[max(1, ceil(Int, 0.25 * n))],
        max_loss = isempty(losses) ? 0.0 : minimum(losses),
        loss_pct = length(losses) / n,
        avg_loss = isempty(losses) ? 0.0 : mean(losses),
        skew = sk,
    )
end

# =============================================================================
# Run
# =============================================================================

store = DEFAULT_STORE

println("=" ^ 110)
println("  LEFT TAIL ANALYSIS — Minimal (8d) Binary Sizing vs Baseline (λ=$SPREAD_LAMBDA, $(length(SEEDS)) seeds)")
println("=" ^ 110)

for (symbol, spot_symbol, spot_multiplier) in SYMBOLS
    println("\n", "#" ^ 80)
    println("  $symbol")
    println("#" ^ 80)

    scaled_max_loss = BASE_MAX_LOSS * spot_multiplier

    all_dates = available_polygon_dates(store, symbol)
    filtered_dates = filter(d -> d >= TRAIN_START && d <= TEST_END, all_dates)
    length(filtered_dates) < 50 && continue

    all_entry_times = sort(unique([ENTRY_TIME; TRAIN_ENTRY_TIMES]))
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
        min_volume=0, warn=false, spread_lambda=SPREAD_LAMBDA
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
    base_rois = condor_rois(base_result.positions, base_result.pnl)
    base_ts = tail_stats(base_rois)

    # Binary across seeds
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

    # Collect all binary rois across seeds (pooled)
    all_binary_rois = Float64[]
    per_seed_stats = []

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
        rois = condor_rois(result.positions, result.pnl)
        append!(all_binary_rois, rois)
        push!(per_seed_stats, tail_stats(rois))
    end

    bin_ts = tail_stats(all_binary_rois)

    # Averaged per-seed stats
    avg_seed(field) = mean(getfield(s, field) for s in per_seed_stats)

    # Print table
    println()
    @printf("  %-20s  %10s  %10s\n", "", "Baseline", "Binary(avg)")
    println("  ", "-" ^ 42)
    @printf("  %-20s  %10d  %10.0f\n", "Trades", base_ts.n, avg_seed(:n))
    @printf("  %-20s  %9.1f%%  %9.1f%%\n", "Mean ROI", base_ts.mean * 100, avg_seed(:mean) * 100)
    @printf("  %-20s  %9.1f%%  %9.1f%%\n", "Std ROI", base_ts.std * 100, avg_seed(:std) * 100)
    @printf("  %-20s  %10.2f  %10.2f\n", "Skewness", base_ts.skew, avg_seed(:skew))
    println("  ", "-" ^ 42)
    @printf("  %-20s  %9.1f%%  %9.1f%%\n", "5th percentile", base_ts.p5 * 100, avg_seed(:p5) * 100)
    @printf("  %-20s  %9.1f%%  %9.1f%%\n", "10th percentile", base_ts.p10 * 100, avg_seed(:p10) * 100)
    @printf("  %-20s  %9.1f%%  %9.1f%%\n", "25th percentile", base_ts.p25 * 100, avg_seed(:p25) * 100)
    @printf("  %-20s  %9.1f%%  %9.1f%%\n", "Max loss", base_ts.max_loss * 100, avg_seed(:max_loss) * 100)
    println("  ", "-" ^ 42)
    @printf("  %-20s  %9.1f%%  %9.1f%%\n", "Loss frequency", base_ts.loss_pct * 100, avg_seed(:loss_pct) * 100)
    @printf("  %-20s  %9.1f%%  %9.1f%%\n", "Avg loss (when loss)", base_ts.avg_loss * 100, avg_seed(:avg_loss) * 100)
    @printf("  %-20s  %9.1f%%  %9.1f%%\n", "CVaR 5%",
        mean(sort(base_rois)[1:max(1, floor(Int, 0.05 * length(base_rois)))]) * 100,
        mean(sort(all_binary_rois)[1:max(1, floor(Int, 0.05 * length(all_binary_rois)))]) * 100)
end

println("\n\nDone.")
