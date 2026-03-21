using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Flux, Statistics, Random

# =============================================================================
# Parameters
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

SEEDS = [42, 123, 7, 99, 256]

SYMBOLS = [
    ("SPY",  "SPY",  1.0),
    ("QQQ",  "QQQ",  1.0),
    ("IWM",  "IWM",  1.0),
    ("SPXW", "SPY", 10.0),
]

sf = Feature[
    ATMImpliedVol(; rate=RATE, div_yield=DIV_YIELD),
    DeltaSkew(0.25, :put; rate=RATE, div_yield=DIV_YIELD),
    ATMSpread(),
    SpotLogSig(; lookback=20, depth=3),
]
input_dim = surface_feature_dim(sf)

# =============================================================================

store = DEFAULT_STORE

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "classifier_vs_regressor_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")

function safe_metrics(r)
    isempty(r.positions) && return nothing
    margin = condor_max_loss_by_key(r.positions)
    isempty(margin) && return nothing
    try
        performance_metrics(r.positions, r.pnl; margin_by_key=margin)
    catch
        nothing
    end
end

# Collect results
results = []  # (symbol, method, seed, sharpe, roi, trades, win_rate)

for (symbol, spot_sym, mult) in SYMBOLS
    println("\n", "=" ^ 60)
    println("  $symbol")
    println("=" ^ 60)

    scaled_ml = BASE_MAX_LOSS * mult
    all_dates = available_polygon_dates(store, symbol)
    filtered = filter(d -> d >= TRAIN_START && d <= TEST_END, all_dates)
    length(filtered) < 50 && continue

    all_entry_times = sort(unique([ENTRY_TIME; TRAIN_ENTRY_TIMES]))
    entry_ts = build_entry_timestamps(filtered, all_entry_times)
    entry_spots = read_polygon_spot_prices_for_timestamps(polygon_spot_root(store), entry_ts; symbol=spot_sym)
    if mult != 1.0; for (k,v) in entry_spots; entry_spots[k] = v * mult; end; end

    source = ParquetDataSource(entry_ts;
        path_for_timestamp=ts -> polygon_options_path(store, Date(ts), symbol),
        read_records=(path; where="") -> read_polygon_option_records(path, entry_spots; where, min_volume=0, warn=false, spread_lambda=SPREAD_LAMBDA),
        spot_root=polygon_spot_root(store), spot_symbol=spot_sym, spot_multiplier=mult)

    all_ts = available_timestamps(source)
    train_schedule = filter(t -> Date(t) <= TRAIN_END, all_ts)

    test_dates = filter(d -> d >= TEST_START, all_dates)
    test_sched = filter(t -> t in Set(all_ts), build_entry_timestamps(test_dates, ENTRY_TIME))

    sel = constrained_delta_selector(PUT_DELTA, CALL_DELTA; rate=RATE, div_yield=DIV_YIELD,
        max_loss=scaled_ml, max_spread_rel=MAX_SPREAD_REL, min_delta_gap=MIN_DELTA_GAP)

    # Baseline
    bm = safe_metrics(backtest_strategy(IronCondorStrategy(test_sched, Day(1), sel), source))
    if bm !== nothing
        push!(results, (symbol=symbol, method="Baseline", seed=0,
            sharpe=bm.sharpe, roi=bm.total_roi, trades=bm.count, win_rate=bm.win_rate))
        @printf("  Baseline: trades=%d sharpe=%.2f roi=%s\n",
            bm.count, bm.sharpe, fmt_metric(bm.total_roi; pct=true))
    end

    # Generate training data
    examples = generate_sizing_training_data(source, Day(1), train_schedule, sel;
        rate=RATE, div_yield=DIV_YIELD, surface_features=sf)
    isempty(examples) && continue

    X = hcat([e.surface_features for e in examples]...)
    Y_pnl = reshape(Float32[e.pnl for e in examples], 1, :)

    # Loss labels at different severity thresholds
    # Y_loss = 1 means "this is a bad entry", model predicts P(loss)
    # Skip entry when P(loss) > skip_threshold
    loss_configs = [
        ("Loss>0",     Float32.(Y_pnl .< 0),    0.3),  # any loss
        ("Loss>25%",   Float32.(Y_pnl .< -0.5),  0.2),  # >25% of typical max_loss
        ("Loss>50%",   Float32.(Y_pnl .< -1.0),  0.2),  # >50% of typical max_loss
    ]

    loss_rate = mean(Y_pnl .< 0)
    big_loss_rate = mean(Y_pnl .< -1.0)
    @printf("  Training: %d examples, %.1f%% any loss, %.1f%% big loss (>50%% max_loss)\n",
        length(examples), loss_rate * 100, big_loss_rate * 100)

    for seed in SEEDS
        # --- Regressor + binary_sizing (existing approach) ---
        Random.seed!(seed)
        reg_model = Chain(Dense(input_dim => 32, relu), Dense(32 => 16, relu), Dense(16 => 1))
        reg_model, reg_fm, reg_fs, _ = train_model!(reg_model, X, Y_pnl;
            epochs=200, lr=1e-3, batch_size=32, val_fraction=0.2, patience=20)

        reg_strat = SizedIronCondorStrategy(test_sched, Day(1), sel, reg_model, reg_fm, reg_fs;
            surface_features=sf, sizing_policy=binary_sizing(; threshold=0.0, quantity=1.0))
        reg_m = safe_metrics(backtest_strategy(reg_strat, source))
        if reg_m !== nothing
            push!(results, (symbol=symbol, method="Regressor", seed=seed,
                sharpe=reg_m.sharpe, roi=reg_m.total_roi, trades=reg_m.count, win_rate=reg_m.win_rate))
        end

        # --- Loss classifiers at different thresholds ---
        for (label, Y_loss, skip_thresh) in loss_configs
            Random.seed!(seed)
            loss_model = Chain(Dense(input_dim => 32, relu), Dense(32 => 16, relu), Dense(16 => 1))

            # Upweight the rare loss class so the model pays attention to it
            loss_frac = mean(Y_loss)
            pw = loss_frac < 0.5 ? (1.0 - loss_frac) / loss_frac : 1.0  # balance classes

            loss_model, loss_fm, loss_fs, _ = train_classifier!(loss_model, X, Y_loss;
                epochs=200, lr=1e-3, batch_size=32, val_fraction=0.2, patience=20,
                pos_weight=pw)

            # Skip when P(loss) > threshold — invert: trade when P(loss) <= threshold
            # probability_sizing applies sigmoid and trades when prob > threshold
            # We want to SKIP when prob > skip_thresh, so trade when prob < skip_thresh
            # Easiest: negate the logit, then trade when P(-logit) > (1-skip_thresh)
            # Or just use a custom policy:
            skip_policy = let st = skip_thresh
                function(logit::Float64)
                    p_loss = 1.0 / (1.0 + exp(-logit))
                    return p_loss < st ? 1.0 : 0.0
                end
            end

            loss_strat = SizedIronCondorStrategy(test_sched, Day(1), sel, loss_model, loss_fm, loss_fs;
                surface_features=sf, sizing_policy=skip_policy)
            loss_m = safe_metrics(backtest_strategy(loss_strat, source))
            if loss_m !== nothing
                push!(results, (symbol=symbol, method=label, seed=seed,
                    sharpe=loss_m.sharpe, roi=loss_m.total_roi, trades=loss_m.count, win_rate=loss_m.win_rate))
            end
        end

        # Print seed summary
        let line = @sprintf("  seed=%d  reg=%.2f", seed, reg_m === nothing ? NaN : reg_m.sharpe)
            for (label, _, _) in loss_configs
                rows = filter(r -> r.symbol == symbol && r.method == label && r.seed == seed, results)
                sh = isempty(rows) ? NaN : rows[end].sharpe
                line *= @sprintf("  %s=%.2f", label, sh)
            end
            println(line)
        end
    end
end

# =============================================================================
# Summary
# =============================================================================

println("\n\n", "=" ^ 100)
println("  CLASSIFIER vs REGRESSOR — Minimal (8d), Binary Sizing (avg $(length(SEEDS)) seeds)")
println("=" ^ 100)

methods = ["Baseline", "Regressor", "Loss>0", "Loss>25%", "Loss>50%"]

# Header
let h = @sprintf("  %-12s", "Method")
    for (sym, _, _) in SYMBOLS
        h *= @sprintf("  %20s", "$sym Sharpe")
    end
    println(h)
end
println("  ", "-" ^ (12 + 22 * length(SYMBOLS)))

for meth in methods
    let row = @sprintf("  %-12s", meth)
        for (sym, _, _) in SYMBOLS
            rows = filter(r -> r.symbol == sym && r.method == meth, results)
            if isempty(rows)
                row *= @sprintf("  %20s", "-")
            elseif meth == "Baseline"
                row *= @sprintf("  %20.2f", rows[1].sharpe)
            else
                avg = mean(r.sharpe for r in rows)
                sd = length(rows) > 1 ? std([r.sharpe for r in rows]) : 0.0
                row *= @sprintf("  %12.2f ± %.2f", avg, sd)
            end
        end
        println(row)
    end
end

println()

# ROI table
let h = @sprintf("  %-12s", "Method")
    for (sym, _, _) in SYMBOLS
        h *= @sprintf("  %20s", "$sym ROI")
    end
    println(h)
end
println("  ", "-" ^ (12 + 22 * length(SYMBOLS)))

for meth in methods
    let row = @sprintf("  %-12s", meth)
        for (sym, _, _) in SYMBOLS
            rows = filter(r -> r.symbol == sym && r.method == meth, results)
            if isempty(rows)
                row *= @sprintf("  %20s", "-")
            elseif meth == "Baseline"
                row *= @sprintf("  %19.1f%%", rows[1].roi * 100)
            else
                avg = mean(r.roi for r in rows) * 100
                sd = length(rows) > 1 ? std([r.roi for r in rows]) * 100 : 0.0
                row *= @sprintf("  %11.1f%% ± %.1f%%", avg, sd)
            end
        end
        println(row)
    end
end

println()

# Trades table
let h = @sprintf("  %-12s", "Method")
    for (sym, _, _) in SYMBOLS
        h *= @sprintf("  %20s", "$sym Trades")
    end
    println(h)
end
println("  ", "-" ^ (12 + 22 * length(SYMBOLS)))

for meth in methods
    let row = @sprintf("  %-12s", meth)
        for (sym, _, _) in SYMBOLS
            rows = filter(r -> r.symbol == sym && r.method == meth, results)
            if isempty(rows)
                row *= @sprintf("  %20s", "-")
            elseif meth == "Baseline"
                row *= @sprintf("  %20d", rows[1].trades)
            else
                avg = mean(r.trades for r in rows)
                row *= @sprintf("  %20.0f", avg)
            end
        end
        println(row)
    end
end

println("=" ^ 100)
println("\nDone.")
