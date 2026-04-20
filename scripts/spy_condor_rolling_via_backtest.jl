# scripts/spy_condor_rolling_via_backtest.jl
#
# Runs the SPY 2pm/2h iron condor with rolling-wing selection via the
# standard `IronCondorStrategy + backtest_strategy` machinery, using the new
# `RollingWingCondorSelector`. Dumps OOS PnLs for cross-comparison with
# `condor_rolling_wing.jl` ground truth.

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics

SYMBOL              = get(ENV, "SYM", "SPY")
START_DATE          = Date(parse(Int, get(ENV, "START_YEAR", "2017")), 1, 1)
END_DATE            = Date(2024, 1, 31)
ENTRY_TIME          = Time(14, 0)
EXPIRY_INTERVAL     = Hour(2)
SPREAD_LAMBDA       = 0.7
RATE                = 0.045
DIV_YIELD           = parse(Float64, get(ENV, "DIV", "0.013"))

PUT_DELTA           = 0.20
CALL_DELTA          = 0.05
WING_WIDTHS         = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
TRAIN_DAYS          = 90
TEST_DAYS           = 30

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "spy_condor_rolling_via_backtest_$(SYMBOL)_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")

# --- Data source --------------------------------------------------------------
store = DEFAULT_STORE
all_dates = available_polygon_dates(store, SYMBOL)
filtered = filter(d -> START_DATE <= d <= END_DATE, all_dates)
println("\nLoading $SYMBOL  ($(length(filtered)) trading days)...")

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

# --- Build strategy + backtest ------------------------------------------------
selector = RollingWingCondorSelector(;
    put_delta = PUT_DELTA, call_delta = CALL_DELTA,
    wing_widths = WING_WIDTHS,
    train_days = TRAIN_DAYS, test_days = TEST_DAYS,
    rate = RATE, div_yield = DIV_YIELD,
    max_tau_days = 0.5,   # match ground truth's intraday filter
)
strategy = IronCondorStrategy(sched, EXPIRY_INTERVAL, selector)

println("\nRunning backtest_strategy...")
result = backtest_strategy(strategy, source)

# --- Build per-trade PnL via condor_trade_table -------------------------------
table = condor_trade_table(result.positions, result.pnl)
println("\nTrades: $(size(table, 1))")

metrics = performance_metrics(result)
println("\nPerformance metrics:")
println(metrics)

# --- Dump CSV in same format as ground truth ----------------------------------
# Trade-table gives PnL as fraction of spot (Position-pricing convention).
# Multiply by spot at entry to get USD-per-share equivalent.
# We'll join the table to the selector's history (entry_ts → spot_at_entry, chosen_wing).
hist_by_ts = Dict(e.entry_ts => e for e in selector.state.history)

# OOS window matches ground truth: skip dates before first training window completes.
oos_start_date = START_DATE + Day(TRAIN_DAYS)

csv_path = joinpath(run_dir, "oos_series_via_backtest.csv")
n_written = Ref(0)
open(csv_path, "w") do io
    println(io, "date,chosen_wing,oos_pnl_usd")
    for row in eachrow(table)
        ismissing(row.PnL) && continue
        d = Date(row.EntryTimestamp)
        d < oos_start_date && continue
        h = get(hist_by_ts, row.EntryTimestamp, nothing)
        h === nothing && continue
        pnl_usd = row.PnL * h.spot_at_entry
        @printf io "%s,%.1f,%.6f\n" d h.chosen_wing pnl_usd
        n_written[] += 1
    end
end
println("\n  Saved: $csv_path  ($(n_written[]) rows)")
