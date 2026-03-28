using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Flux, Statistics, Random, DataFrames

include("lib/experiment.jl")

# =============================================================================
# Tail risk analysis
# =============================================================================

function tail_analysis(label, result; baseline_result=nothing)
    df = condor_trade_table(result.positions, result.pnl)
    pnls = Float64[r.PnL for r in eachrow(df) if !ismissing(r.PnL) && !ismissing(r.MaxLoss) && r.MaxLoss > 0]
    rors = Float64[r.ReturnOnRisk for r in eachrow(df) if !ismissing(r.ReturnOnRisk)]
    n = length(pnls)
    n < 2 && return

    wins = filter(>(0), pnls)
    losses = filter(<=(0), pnls)
    s = std(pnls)
    skew = s > 0 ? mean((pnls .- mean(pnls)).^3) / s^3 : 0.0

    println("\n  ── $label ($n trades) ──")
    @printf("  Mean=\$%.4f  Std=\$%.4f  Skew=%.2f\n", mean(pnls), s, skew)
    @printf("  WinRate=%.1f%%  AvgWin=\$%.4f  AvgLoss=\$%.4f  W/L=%.2f\n",
        length(wins)/n*100,
        isempty(wins) ? 0.0 : mean(wins),
        isempty(losses) ? 0.0 : mean(losses),
        isempty(losses) ? Inf : mean(wins)/abs(mean(losses)))

    # Tail percentiles
    println("\n  Tail percentiles (PnL):")
    for p in [0.01, 0.02, 0.05, 0.10, 0.25]
        @printf("    p%-4.0f = \$%+.4f\n", p*100, quantile(pnls, p))
    end

    # CVaR (Expected Shortfall) at various levels
    println("\n  CVaR / Expected Shortfall:")
    for alpha in [0.05, 0.10, 0.20]
        cutoff = quantile(pnls, alpha)
        tail = filter(<=(cutoff), pnls)
        cvar = isempty(tail) ? 0.0 : mean(tail)
        @printf("    ES(%.0f%%) = \$%.4f  (avg of worst %.0f%% = %d trades)\n",
            alpha*100, cvar, alpha*100, length(tail))
    end

    # Loss severity buckets (by ROI)
    println("\n  Loss severity (by ReturnOnRisk):")
    buckets = [
        ("ROI > 0 (wins)",       r -> r > 0),
        ("ROI ∈ [-25%, 0]",      r -> -0.25 <= r <= 0),
        ("ROI ∈ [-50%, -25%)",   r -> -0.50 <= r < -0.25),
        ("ROI ∈ [-75%, -50%)",   r -> -0.75 <= r < -0.50),
        ("ROI < -75%",           r -> r < -0.75),
    ]
    for (lbl, pred) in buckets
        cnt = count(pred, rors)
        avg_pnl = cnt > 0 ? mean(pnls[findall(pred, rors)]) : 0.0
        @printf("    %-25s  %3d (%5.1f%%)  avgPnL=\$%+.4f\n", lbl, cnt, cnt/n*100, avg_pnl)
    end

    # Max drawdown (cumulative PnL)
    cum = cumsum(pnls)
    peak = accumulate(max, cum)
    dd = cum .- peak
    max_dd = minimum(dd)
    @printf("\n  Max drawdown: \$%.4f\n", max_dd)
    @printf("  Total PnL: \$%.4f  Min single: \$%.4f  Max single: \$%.4f\n",
        sum(pnls), minimum(pnls), maximum(pnls))

    # If baseline provided, show which losses the filter avoided
    if baseline_result !== nothing
        bdf = condor_trade_table(baseline_result.positions, baseline_result.pnl)
        b_entries = Set(r.EntryTimestamp for r in eachrow(bdf) if !ismissing(r.PnL))
        f_entries = Set(r.EntryTimestamp for r in eachrow(df) if !ismissing(r.PnL))
        skipped = setdiff(b_entries, f_entries)

        if !isempty(skipped)
            skipped_rows = [r for r in eachrow(bdf) if r.EntryTimestamp in skipped && !ismissing(r.PnL)]
            skipped_pnls = Float64[r.PnL for r in skipped_rows if !ismissing(r.MaxLoss) && r.MaxLoss > 0]
            skipped_rors = Float64[r.ReturnOnRisk for r in skipped_rows if !ismissing(r.ReturnOnRisk)]
            if !isempty(skipped_pnls)
                s_wins = count(>(0), skipped_pnls)
                s_losses = count(<=(0), skipped_pnls)
                s_big = count(r -> r < -0.5, skipped_rors)
                println("\n  Filter skipped $(length(skipped_pnls)) trades:")
                @printf("    Wins=%d  Losses=%d  BigLosses(ROI<-50%%)=%d\n", s_wins, s_losses, s_big)
                @printf("    AvgPnL=\$%+.4f  SumPnL=\$%+.4f\n", mean(skipped_pnls), sum(skipped_pnls))
                @printf("    Worst skipped: \$%.4f  Best skipped: \$%.4f\n",
                    minimum(skipped_pnls), maximum(skipped_pnls))
                # Were the skipped trades net negative?
                @printf("    → Filter value: skipped trades had %s avg PnL\n",
                    mean(skipped_pnls) < 0 ? "NEGATIVE" : "POSITIVE")
            end
        end
    end
end

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "cross_symbol_filter"

# SPY only, 1 month train / 1 month test
TRAIN_SYMBOLS = [
    ("SPY",  "SPY",  1.0),
]
TEST_SYMBOL = ("SPY", "SPY", 1.0)

SPREAD_LAMBDA   = 0.7
SEEDS           = [42, 123, 7]

TRAIN_START = Date(2024, 2, 1)
TRAIN_END   = Date(2024, 12, 31)
TEST_START  = Date(2025, 1, 1)
TEST_END    = Date(2025, 12, 31)

ENTRY_TIME        = Time(10, 0)
TRAIN_ENTRY_TIMES = [Time(10, 0), Time(12, 0)]
EXPIRY_INTERVAL   = Day(1)

RATE           = 0.045
DIV_YIELD      = 0.013
BASE_MAX_LOSS  = 5.0
MAX_SPREAD_REL = 0.50
PUT_DELTA      = 0.16
CALL_DELTA     = 0.16

FEATURES = Feature[
    IntradayLogSig(; depth=3, rate=RATE, div_yield=DIV_YIELD),
]
FEATURE_NAME = "intraday_logsig"

# Classifier thresholds (P(loss) > thresh → skip)
CLASSIFIER_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
# Regressor thresholds (predicted PnL < thresh → skip)
REGRESSOR_THRESHOLDS  = [-0.5, -0.25, 0.0, 0.1, 0.2, 0.3, 0.5]
# Loss threshold for classifier labels
LOSS_CUTOFF = -0.5   # ROI < -50% = "big loss"

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
# Build data sources for all symbols
# =============================================================================

function build_source(symbol, spot_sym, mult)
    all_dates = available_polygon_dates(store, symbol)
    filtered = filter(d -> d >= TRAIN_START && d <= TEST_END, all_dates)
    if length(filtered) < 10
        println("  SKIP $symbol: only $(length(filtered)) dates")
        return nothing, DateTime[], DateTime[]
    end

    all_entry_times = sort(unique([ENTRY_TIME; TRAIN_ENTRY_TIMES]))
    entry_ts = build_entry_timestamps(filtered, all_entry_times)

    # Lazy spot cache — loads per-date from parquet on demand
    _spot_cache = Dict{Date, Dict{DateTime, Float64}}()
    function _lazy_spots(d::Date)
        if !haskey(_spot_cache, d)
            sp = polygon_spot_path(store, d, spot_sym)
            _spot_cache[d] = isfile(sp) ? read_polygon_spot_prices(sp; underlying=spot_sym) : Dict{DateTime,Float64}()
            if mult != 1.0
                for (k, v) in _spot_cache[d]; _spot_cache[d][k] = v * mult; end
            end
        end
        return _spot_cache[d]
    end

    source = ParquetDataSource(entry_ts;
        path_for_timestamp=ts -> polygon_options_path(store, Date(ts), symbol),
        read_records=(path; where="") -> begin
            bars = read_polygon_parquet(path; where=where, min_volume=0)
            records = OptionRecord[]
            for bar in bars
                spot = get(_lazy_spots(Date(bar.timestamp)), bar.timestamp, missing)
                ismissing(spot) && continue
                push!(records, to_option_record(bar, spot; warn=false, spread_lambda=SPREAD_LAMBDA))
            end
            return records
        end,
        spot_root=polygon_spot_root(store),
        spot_symbol=spot_sym,
        spot_multiplier=mult)

    all_ts = available_timestamps(source)
    train_sched = filter(t -> Date(t) <= TRAIN_END, all_ts)
    test_sched = filter(t -> t in Set(build_entry_timestamps(
        filter(d -> d >= TEST_START, filtered), ENTRY_TIME)), all_ts)

    return source, train_sched, test_sched
end

# =============================================================================
# Gather training data from ALL symbols
# =============================================================================

println("\n", "=" ^ 60)
println("  Gathering cross-symbol training data")
println("=" ^ 60)

all_examples = VolSurfaceAnalysis.SizingTrainingExample[]
train_counts = Dict{String,Int}()

for (symbol, spot_sym, mult) in TRAIN_SYMBOLS
    scaled_ml = BASE_MAX_LOSS * mult
    source, train_sched, _ = build_source(symbol, spot_sym, mult)
    source === nothing && continue

    sel = constrained_delta_selector(PUT_DELTA, CALL_DELTA;
        rate=RATE, div_yield=DIV_YIELD, max_loss=scaled_ml,
        max_spread_rel=MAX_SPREAD_REL)

    examples = generate_sizing_training_data(source, EXPIRY_INTERVAL,
        train_sched, sel; rate=RATE, div_yield=DIV_YIELD, surface_features=FEATURES)

    train_counts[symbol] = length(examples)
    append!(all_examples, examples)
    @printf("  %s: %d training examples\n", symbol, length(examples))
end

if isempty(all_examples)
    println("No training data!")
    exit(1)
end

X_all = hcat([e.surface_features for e in all_examples]...)
Y_pnl = reshape(Float32[e.pnl for e in all_examples], 1, :)
@printf("\nTotal: %d training examples, %d dims\n", length(all_examples), input_dim)
for (sym, cnt) in sort(collect(train_counts))
    @printf("  %s: %d (%.0f%%)\n", sym, cnt, cnt/length(all_examples)*100)
end

# Classifier labels: 1 = big loss, 0 = not
# We need per-example max_loss to compute ROI. Use PnL sign as proxy if max_loss
# isn't available, but for iron condors PnL is already in dollar terms.
# For the classifier, use raw PnL threshold relative to typical max_loss.
# Since max_loss ~ BASE_MAX_LOSS for unit symbols, use PnL/max_loss-like cutoff.
# Actually, the examples store raw PnL (in price-fraction terms).
# A "big loss" = the trade lost more than LOSS_CUTOFF fraction of max_loss.
# Since we don't have max_loss per example, use a PnL < 0 as loss, and
# PnL < LOSS_CUTOFF * approx_max_loss as big loss.
# Simpler: just use PnL < 0 as the loss label (any loss).
# Or: classify by quantile of the training PnL distribution.

# Use a simple approach: label = 1 if PnL < pnl_threshold (bottom N%)
pnl_vec = vec(Y_pnl)
pnl_p25 = quantile(pnl_vec, 0.25)
pnl_p10 = quantile(pnl_vec, 0.10)
loss_frac = count(<(0), pnl_vec) / length(pnl_vec)
big_loss_frac = count(<(pnl_p10), pnl_vec) / length(pnl_vec)
@printf("\nPnL distribution: mean=%.4f  p10=%.4f  p25=%.4f\n", mean(pnl_vec), pnl_p10, pnl_p25)
@printf("Loss rate=%.1f%%  Bottom-10%% cutoff=%.4f\n", loss_frac*100, pnl_p10)

# For classifier: label = 1 if PnL < 0 (any loss)
Y_cls = reshape(Float32[p < 0 ? 1.0f0 : 0.0f0 for p in pnl_vec], 1, :)
pos_rate = mean(Y_cls)
pos_weight = (1.0 - pos_rate) / max(pos_rate, 0.01)
@printf("Classifier: %.1f%% positive (losses), pos_weight=%.2f\n", pos_rate*100, pos_weight)

# =============================================================================
# Build test source (SPY only)
# =============================================================================

println("\n", "=" ^ 60)
println("  Testing on $(TEST_SYMBOL[1])")
println("=" ^ 60)

test_sym, test_spot, test_mult = TEST_SYMBOL
test_scaled_ml = BASE_MAX_LOSS * test_mult
test_source, _, test_sched = build_source(test_sym, test_spot, test_mult)

test_sel = constrained_delta_selector(PUT_DELTA, CALL_DELTA;
    rate=RATE, div_yield=DIV_YIELD, max_loss=test_scaled_ml,
    max_spread_rel=MAX_SPREAD_REL)

println("  Test schedule: $(length(test_sched)) timestamps")

# Baseline
baseline_result = backtest_strategy(
    IronCondorStrategy(test_sched, EXPIRY_INTERVAL, test_sel), test_source)
bm = performance_metrics(baseline_result)
push!(results, (symbol=test_sym, features="—", variant="Baseline", seed=0,
    sharpe=bm.sharpe, sortino=bm.sortino, roi=bm.total_roi,
    trades=bm.count, win_rate=bm.win_rate, pnl=bm.total_pnl))
@printf("  Baseline: trades=%d sharpe=%.2f roi=%s\n",
    bm.count, bm.sharpe, fmt_metric(bm.total_roi; pct=true))
tail_analysis("Baseline", baseline_result)

# =============================================================================
# Train and evaluate — both regressor and classifier
# =============================================================================

for seed in SEEDS
    println("\n", "-" ^ 60)
    println("  Seed $seed")
    println("-" ^ 60)

    # --- Regressor ---
    Random.seed!(seed)
    reg_model = Chain(Dense(input_dim => 32, relu), Dense(32 => 16, relu), Dense(16 => 1))
    reg_model, reg_means, reg_stds, _ = train_model!(reg_model, X_all, Y_pnl;
        epochs=200, lr=1e-3, batch_size=32, val_fraction=0.2, patience=20)

    for thresh in REGRESSOR_THRESHOLDS
        policy = binary_sizing(; threshold=thresh, quantity=1.0)
        variant = @sprintf("Reg t=%+.2f", thresh)

        strategy = IronCondorStrategy(test_sched, EXPIRY_INTERVAL, test_sel;
            sizer=MLSizer(reg_model, reg_means, reg_stds; surface_features=FEATURES, policy=policy))
        ml_result = backtest_strategy(strategy, test_source)
        m = performance_metrics(ml_result)
        if m === nothing
            @printf("  Regressor seed=%d thresh=%+.2f: no trades\n", seed, thresh)
            continue
        end
        push!(results, (symbol=test_sym, features=FEATURE_NAME, variant=variant, seed=seed,
            sharpe=m.sharpe, sortino=coalesce(m.sortino, 0.0), roi=m.total_roi,
            trades=m.count, win_rate=m.win_rate, pnl=m.total_pnl))

        df = condor_trade_table(ml_result.positions, ml_result.pnl)
        rors = Float64[r.ReturnOnRisk for r in eachrow(df) if !ismissing(r.ReturnOnRisk)]
        big_l = count(r -> r < -0.5, rors)
        @printf("  Reg  s=%d t=%+.2f: trades=%3d sharpe=%5.2f roi=%6.2f%% win=%.1f%% bigL=%d/%d\n",
            seed, thresh, m.count, m.sharpe, m.total_roi*100, m.win_rate*100, big_l, m.count)
    end

    # --- Classifier ---
    Random.seed!(seed)
    cls_model = Chain(Dense(input_dim => 32, relu), Dense(32 => 16, relu), Dense(16 => 1))
    cls_model, cls_means, cls_stds, _ = train_classifier!(cls_model, X_all, Y_cls;
        pos_weight=pos_weight, epochs=200, lr=1e-3, batch_size=32,
        val_fraction=0.2, patience=20)

    for thresh in CLASSIFIER_THRESHOLDS
        # probability_sizing: trade when P(loss) < thresh, i.e. skip when P(loss) > thresh
        # The model predicts P(loss) as logit. We want to SKIP high-loss-probability.
        # probability_sizing trades when sigmoid(logit) > thresh → this is P(win) > thresh.
        # But our labels are 1=loss. So sigmoid(logit) = P(loss).
        # We want: trade when P(loss) < thresh, i.e. when 1-sigmoid(logit) > 1-thresh.
        # Equivalently: skip when sigmoid(logit) > thresh.
        # Custom policy:
        skip_thresh = thresh
        policy = function(logit::Float64)
            p_loss = 1.0 / (1.0 + exp(-logit))
            return p_loss < skip_thresh ? 1.0 : 0.0
        end

        variant = @sprintf("Cls p<%.2f", thresh)

        strategy = IronCondorStrategy(test_sched, EXPIRY_INTERVAL, test_sel;
            sizer=MLSizer(cls_model, cls_means, cls_stds; surface_features=FEATURES, policy=policy))
        ml_result = backtest_strategy(strategy, test_source)
        m = performance_metrics(ml_result)
        if m === nothing
            @printf("  Classifier seed=%d thresh=%.2f: no trades\n", seed, thresh)
            continue
        end
        push!(results, (symbol=test_sym, features=FEATURE_NAME, variant=variant, seed=seed,
            sharpe=m.sharpe, sortino=coalesce(m.sortino, 0.0), roi=m.total_roi,
            trades=m.count, win_rate=m.win_rate, pnl=m.total_pnl))

        df = condor_trade_table(ml_result.positions, ml_result.pnl)
        rors = Float64[r.ReturnOnRisk for r in eachrow(df) if !ismissing(r.ReturnOnRisk)]
        big_l = count(r -> r < -0.5, rors)
        @printf("  Cls  s=%d p<%.2f: trades=%3d sharpe=%5.2f roi=%6.2f%% win=%.1f%% bigL=%d/%d\n",
            seed, thresh, m.count, m.sharpe, m.total_roi*100, m.win_rate*100, big_l, m.count)
    end
end

# =============================================================================
# Detailed tail analysis for best variants
# =============================================================================

println("\n", "=" ^ 60)
println("  Detailed tail analysis (seed=$(SEEDS[1]))")
println("=" ^ 60)

# Re-run best regressor and classifier with seed[1] for detailed analysis
Random.seed!(SEEDS[1])
reg_model = Chain(Dense(input_dim => 32, relu), Dense(32 => 16, relu), Dense(16 => 1))
reg_model, reg_means, reg_stds, _ = train_model!(reg_model, X_all, Y_pnl;
    epochs=200, lr=1e-3, batch_size=32, val_fraction=0.2, patience=20)

Random.seed!(SEEDS[1])
cls_model = Chain(Dense(input_dim => 32, relu), Dense(32 => 16, relu), Dense(16 => 1))
cls_model, cls_means, cls_stds, _ = train_classifier!(cls_model, X_all, Y_cls;
    pos_weight=pos_weight, epochs=200, lr=1e-3, batch_size=32,
    val_fraction=0.2, patience=20)

# Regressor at t=0.0 (skip negative predicted PnL)
reg_result = backtest_strategy(
    IronCondorStrategy(test_sched, EXPIRY_INTERVAL, test_sel;
        sizer=MLSizer(reg_model, reg_means, reg_stds; surface_features=FEATURES,
            policy=binary_sizing(; threshold=0.0))),
    test_source)
tail_analysis("Regressor (t=0.00)", reg_result; baseline_result=baseline_result)

# Classifier at p<0.5 (skip when P(loss) >= 50%)
cls_result_50 = backtest_strategy(
    IronCondorStrategy(test_sched, EXPIRY_INTERVAL, test_sel;
        sizer=MLSizer(cls_model, cls_means, cls_stds; surface_features=FEATURES,
            policy=logit -> (1.0/(1.0+exp(-logit))) < 0.5 ? 1.0 : 0.0)),
    test_source)
tail_analysis("Classifier (p<0.50)", cls_result_50; baseline_result=baseline_result)

# Classifier at p<0.3 (aggressive filter)
cls_result_30 = backtest_strategy(
    IronCondorStrategy(test_sched, EXPIRY_INTERVAL, test_sel;
        sizer=MLSizer(cls_model, cls_means, cls_stds; surface_features=FEATURES,
            policy=logit -> (1.0/(1.0+exp(-logit))) < 0.3 ? 1.0 : 0.0)),
    test_source)
tail_analysis("Classifier (p<0.30)", cls_result_30; baseline_result=baseline_result)

# =============================================================================
# Summary
# =============================================================================

print_summary(results, [test_sym])

# Also show SPY-only training for comparison
println("\n", "=" ^ 60)
println("  Comparison: SPY-only training vs cross-symbol")
println("=" ^ 60)

spy_source, spy_train, _ = build_source("SPY", "SPY", 1.0)
spy_examples = generate_sizing_training_data(spy_source, EXPIRY_INTERVAL,
    spy_train, test_sel; rate=RATE, div_yield=DIV_YIELD, surface_features=FEATURES)
X_spy = hcat([e.surface_features for e in spy_examples]...)
Y_spy_pnl = reshape(Float32[e.pnl for e in spy_examples], 1, :)
Y_spy_cls = reshape(Float32[p < 0 ? 1.0f0 : 0.0f0 for p in vec(Y_spy_pnl)], 1, :)
spy_pos_rate = mean(Y_spy_cls)
spy_pos_weight = (1.0 - spy_pos_rate) / max(spy_pos_rate, 0.01)

@printf("SPY-only training: %d examples (vs %d cross-symbol)\n",
    length(spy_examples), length(all_examples))

for seed in SEEDS[1:1]
    Random.seed!(seed)
    spy_reg = Chain(Dense(input_dim => 32, relu), Dense(32 => 16, relu), Dense(16 => 1))
    spy_reg, spy_reg_m, spy_reg_s, _ = train_model!(spy_reg, X_spy, Y_spy_pnl;
        epochs=200, lr=1e-3, batch_size=32, val_fraction=0.2, patience=20)

    r = backtest_strategy(
        IronCondorStrategy(test_sched, EXPIRY_INTERVAL, test_sel;
            sizer=MLSizer(spy_reg, spy_reg_m, spy_reg_s; surface_features=FEATURES,
                policy=binary_sizing(; threshold=0.0))),
        test_source)
    m = performance_metrics(r)
    m !== nothing && @printf("  SPY-only Reg t=0.00: trades=%d sharpe=%.2f roi=%.2f%%\n",
        m.count, m.sharpe, m.total_roi*100)
    tail_analysis("SPY-only Regressor (t=0.00)", r; baseline_result=baseline_result)

    Random.seed!(seed)
    spy_cls = Chain(Dense(input_dim => 32, relu), Dense(32 => 16, relu), Dense(16 => 1))
    spy_cls, spy_cls_m, spy_cls_s, _ = train_classifier!(spy_cls, X_spy, Y_spy_cls;
        pos_weight=spy_pos_weight, epochs=200, lr=1e-3, batch_size=32,
        val_fraction=0.2, patience=20)

    r = backtest_strategy(
        IronCondorStrategy(test_sched, EXPIRY_INTERVAL, test_sel;
            sizer=MLSizer(spy_cls, spy_cls_m, spy_cls_s; surface_features=FEATURES,
                policy=logit -> (1.0/(1.0+exp(-logit))) < 0.5 ? 1.0 : 0.0)),
        test_source)
    m = performance_metrics(r)
    m !== nothing && @printf("  SPY-only Cls p<0.50: trades=%d sharpe=%.2f roi=%.2f%%\n",
        m.count, m.sharpe, m.total_roi*100)
    tail_analysis("SPY-only Classifier (p<0.50)", r; baseline_result=baseline_result)
end

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
