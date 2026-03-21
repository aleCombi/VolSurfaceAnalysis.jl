using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Flux, Statistics, Plots, Random

# =============================================================================
# Parameters
# =============================================================================

ENTRY_TIME      = Time(10, 0)
TRAIN_ENTRY_TIMES = [Time(10, 0), Time(12, 0), Time(14, 0)]
EXPIRY_INTERVAL = Day(1)
SPREAD_LAMBDA   = 0.0   # conservative: widest synthetic spread

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

# (options_symbol, spot_symbol, spot_multiplier)
SYMBOLS = [
    ("SPY",  "SPY",  1.0),
    ("QQQ",  "QQQ",  1.0),
    ("IWM",  "IWM",  1.0),
    ("SPXW", "SPY", 10.0),
]

# =============================================================================
# Output directory
# =============================================================================

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "sizing_all_symbols_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")

# =============================================================================
# Feature set configs (built per-symbol due to rate/div params)
# =============================================================================

function build_feature_configs(; rate, div_yield)
    [
        ("Minimal (8d)" => Feature[
            ATMImpliedVol(; rate, div_yield),
            DeltaSkew(0.25, :put; rate, div_yield),
            ATMSpread(),
            SpotLogSig(; lookback=20, depth=3),
        ]),
        ("Full + daily (20d)" => Feature[
            ATMImpliedVol(; rate, div_yield),
            DeltaSkew(0.25, :put; rate, div_yield),
            RiskReversal(0.25; rate, div_yield),
            Butterfly(0.25; rate, div_yield),
            ATMSpread(),
            TotalVolume(),
            PutCallVolumeRatio(),
            HourOfDay(),
            DayOfWeek(),
            RealizedVol(; lookback=20),
            VarianceRiskPremium(; lookback=20, rate, div_yield),
            SpotMomentum(; lookback=5),
            SpotMomentum(; lookback=20),
            IVChange(; lookback=5, rate, div_yield),
            IVPercentile(; lookback=20, rate, div_yield),
            SpotLogSig(; lookback=20, depth=3),
        ]),
        ("Full + minute (25d)" => Feature[
            ATMImpliedVol(; rate, div_yield),
            DeltaSkew(0.25, :put; rate, div_yield),
            RiskReversal(0.25; rate, div_yield),
            Butterfly(0.25; rate, div_yield),
            ATMSpread(),
            TotalVolume(),
            PutCallVolumeRatio(),
            HourOfDay(),
            DayOfWeek(),
            RealizedVol(; lookback=20),
            VarianceRiskPremium(; lookback=20, rate, div_yield),
            SpotMomentum(; lookback=5),
            SpotMomentum(; lookback=20),
            IVChange(; lookback=5, rate, div_yield),
            IVPercentile(; lookback=20, rate, div_yield),
            SpotMinuteLogSig(; lookback_hours=1, depth=3),
            SpotMinuteLogSig(; lookback_hours=6, depth=3),
        ]),
        ("Full + all (30d)" => Feature[
            ATMImpliedVol(; rate, div_yield),
            DeltaSkew(0.25, :put; rate, div_yield),
            RiskReversal(0.25; rate, div_yield),
            Butterfly(0.25; rate, div_yield),
            ATMSpread(),
            TotalVolume(),
            PutCallVolumeRatio(),
            HourOfDay(),
            DayOfWeek(),
            RealizedVol(; lookback=20),
            VarianceRiskPremium(; lookback=20, rate, div_yield),
            SpotMomentum(; lookback=5),
            SpotMomentum(; lookback=20),
            IVChange(; lookback=5, rate, div_yield),
            IVPercentile(; lookback=20, rate, div_yield),
            SpotLogSig(; lookback=20, depth=3),
            SpotMinuteLogSig(; lookback_hours=1, depth=3),
            SpotMinuteLogSig(; lookback_hours=6, depth=3),
        ]),
    ]
end

# =============================================================================
# Helpers
# =============================================================================

function condor_pnl_curve(positions, pnl)
    by_date = Dict{Date, Float64}()
    for (pos, p) in zip(positions, pnl)
        ismissing(p) && continue
        d = Date(pos.entry_timestamp)
        by_date[d] = get(by_date, d, 0.0) + Float64(p)
    end
    dates = sort(collect(keys(by_date)))
    pnls = [by_date[d] for d in dates]
    return dates, pnls
end

function plot_cdf!(p, values, label; kwargs...)
    sorted = sort(values)
    n = length(sorted)
    ys = (1:n) ./ n
    plot!(p, sorted, ys; label=label, linewidth=2, kwargs...)
end

# =============================================================================
# Global summary collection
# =============================================================================

struct SymbolResult
    symbol::String
    feature_name::String
    seed::Int
    sharpe::Float64
    sortino::Float64
    win_rate::Float64
    total_roi::Float64
    total_pnl::Float64
    trades::Int
end

global_results = SymbolResult[]
baseline_results = Dict{String, PerformanceMetrics}()

# =============================================================================
# Main loop over symbols
# =============================================================================

store = DEFAULT_STORE

for (symbol, spot_symbol, spot_multiplier) in SYMBOLS
    println("\n", "#" ^ 70)
    println("  SYMBOL: $symbol (spot via $spot_symbol × $spot_multiplier)")
    println("#" ^ 70)

    scaled_max_loss = BASE_MAX_LOSS * spot_multiplier

    all_dates = available_polygon_dates(store, symbol)
    filtered_dates = filter(d -> d >= TRAIN_START && d <= TEST_END, all_dates)
    if length(filtered_dates) < 50
        println("  SKIP: only $(length(filtered_dates)) dates")
        continue
    end
    println("  $(length(filtered_dates)) trading days")

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
    println("  Train: $(length(train_schedule)) | Test: $(length(test_schedule))")

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
    baseline_results[symbol] = base_metrics

    push!(global_results, SymbolResult(symbol, "Baseline", 0,
        base_metrics.sharpe, base_metrics.sortino, base_metrics.win_rate,
        base_metrics.total_roi, base_metrics.total_pnl, base_metrics.count))

    @printf("  Baseline: trades=%d sharpe=%.2f sortino=%.2f roi=%s\n",
        base_metrics.count, base_metrics.sharpe, base_metrics.sortino,
        fmt_metric(base_metrics.total_roi; pct=true))

    feature_configs = build_feature_configs(; rate=RATE, div_yield=DIV_YIELD)

    for (name, sf) in feature_configs
        input_dim = surface_feature_dim(sf)

        examples = generate_sizing_training_data(
            source, EXPIRY_INTERVAL, train_schedule, baseline_selector;
            rate=RATE, div_yield=DIV_YIELD,
            surface_features=sf
        )

        if isempty(examples)
            println("  $name: SKIP (no data)")
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

            push!(global_results, SymbolResult(symbol, name, seed,
                metrics.sharpe, metrics.sortino, metrics.win_rate,
                metrics.total_roi, metrics.total_pnl, metrics.count))

            @printf("  %s seed=%d: trades=%d sharpe=%.2f roi=%s\n",
                name, seed, metrics.count, metrics.sharpe,
                fmt_metric(metrics.total_roi; pct=true))
        end
    end
end

# =============================================================================
# Cross-symbol summary table
# =============================================================================

println("\n\n", "=" ^ 100)
println("  CROSS-SYMBOL BINARY SIZING COMPARISON (avg over $(length(SEEDS)) seeds)")
println("=" ^ 100)

feature_names = ["Baseline"; [name for (name, _) in build_feature_configs(; rate=RATE, div_yield=DIV_YIELD)]]
symbols_run = unique([r.symbol for r in global_results])

# Header
let h = @sprintf("  %-22s", "Feature Set")
    for sym in symbols_run
        h *= @sprintf("  %18s", "$sym Sharpe")
    end
    println(h)
end
println("  ", "-" ^ (22 + 20 * length(symbols_run)))

for fname in feature_names
    row = @sprintf("  %-22s", fname)
    for sym in symbols_run
        rows = filter(r -> r.symbol == sym && r.feature_name == fname, global_results)
        if isempty(rows)
            row *= @sprintf("  %18s", "-")
        elseif fname == "Baseline"
            row *= @sprintf("  %18.2f", rows[1].sharpe)
        else
            avg = mean(r.sharpe for r in rows)
            sd = std([r.sharpe for r in rows])
            row *= @sprintf("  %10.2f ± %.2f", avg, sd)
        end
    end
    println(row)
end
println("=" ^ 100)

# Sharpe lift table
println("\n  Sharpe LIFT vs Baseline (avg):")
let h = @sprintf("  %-22s", "Feature Set")
    for sym in symbols_run
        h *= @sprintf("  %12s", sym)
    end
    println(h)
end
println("  ", "-" ^ (22 + 14 * length(symbols_run)))

for fname in feature_names
    fname == "Baseline" && continue
    row = @sprintf("  %-22s", fname)
    for sym in symbols_run
        rows = filter(r -> r.symbol == sym && r.feature_name == fname, global_results)
        base_rows = filter(r -> r.symbol == sym && r.feature_name == "Baseline", global_results)
        if isempty(rows) || isempty(base_rows)
            row *= @sprintf("  %12s", "-")
        else
            lift = mean(r.sharpe for r in rows) - base_rows[1].sharpe
            row *= @sprintf("  %+11.2f", lift)
        end
    end
    println(row)
end
println("=" ^ 100)

# =============================================================================
# Per-symbol CDF plots (baseline + best feature set)
# =============================================================================

println("\n--- Generating per-symbol CDF plots ---")
colors = Dict("Minimal (8d)" => :blue, "Full + daily (20d)" => :red,
    "Full + minute (25d)" => :green, "Full + all (30d)" => :orange)

# Combined CDF grid
n_sym = length(symbols_run)
p_grid = plot(layout=(1, n_sym), size=(350 * n_sym, 450))

for (si, sym) in enumerate(symbols_run)
    sym_results = filter(r -> r.symbol == sym, global_results)
    base = filter(r -> r.feature_name == "Baseline", sym_results)

    plot!(p_grid[si]; title="$sym", xlabel="P&L/condor", ylabel=si == 1 ? "CDF" : "")

    # We don't have the raw positions in global_results, so just show metrics bar
    # Instead make a simple bar comparison
    fnames = unique([r.feature_name for r in sym_results])
    sharpes = Float64[]
    labels = String[]
    bar_cols = []

    for fn in ["Baseline"; [name for (name, _) in build_feature_configs(; rate=RATE, div_yield=DIV_YIELD)]]
        rows = filter(r -> r.feature_name == fn, sym_results)
        isempty(rows) && continue
        push!(sharpes, mean(r.sharpe for r in rows))
        push!(labels, fn == "Baseline" ? "Base" : split(fn, " (")[1])
        push!(bar_cols, fn == "Baseline" ? :black : get(colors, fn, :gray))
    end

    bar!(p_grid[si], labels, sharpes; color=bar_cols, label="", xrotation=20,
        ylabel=si == 1 ? "Sharpe" : "")
end

grid_path = joinpath(run_dir, "sharpe_by_symbol.png")
savefig(p_grid, grid_path)
println("  Saved: $grid_path")

println("\nDone.")
