using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf

# =============================================================================
# Configuration
# =============================================================================

SYMBOLS = [
    # (option_symbol, spot_symbol, spot_multiplier)
    ("SPY",  "SPY",  1.0),
    ("QQQ",  "QQQ",  1.0),
    ("IWM",  "IWM",  1.0),
    ("SPXW", "SPY", 10.0),
]

YEARS = [2024, 2025]
SPREAD_LAMBDA   = 0.7
ENTRY_TIME      = Time(10, 0)
EXPIRY_INTERVAL = Day(1)

RATE           = 0.045
DIV_YIELD      = 0.013
BASE_MAX_LOSS  = 5.0
MAX_SPREAD_REL = 0.50
PUT_DELTA      = 0.16
CALL_DELTA     = 0.16

store = DEFAULT_STORE

# =============================================================================
# Run
# =============================================================================

println("\n16-delta iron condor baseline")
println("=" ^ 80)
@printf("  %-6s  %-6s  %6s  %8s  %8s  %8s  %8s  %8s\n",
    "Symbol", "Year", "Trades", "PnL", "Avg PnL", "WinRate", "Sharpe", "ROI")
println("-" ^ 80)

for (symbol, spot_sym, mult) in SYMBOLS
    scaled_ml = BASE_MAX_LOSS * mult

    all_dates = available_polygon_dates(store, symbol)

    for year in YEARS
        dates = filter(d -> Dates.year(d) == year, all_dates)
        isempty(dates) && continue

        entry_ts = build_entry_timestamps(dates, [ENTRY_TIME])
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

        schedule = available_timestamps(source)

        selector = constrained_delta_selector(PUT_DELTA, CALL_DELTA;
            rate=RATE, div_yield=DIV_YIELD, max_loss=scaled_ml,
            max_spread_rel=MAX_SPREAD_REL)

        result = backtest_strategy(
            IronCondorStrategy(schedule, EXPIRY_INTERVAL, selector), source)
        m = performance_metrics(result)

        @printf("  %-6s  %-6d  %6d  %8s  %8s  %8s  %8s  %8s\n",
            symbol, year, m.count,
            fmt_currency(m.total_pnl),
            fmt_currency(m.avg_pnl),
            fmt_pct(m.win_rate),
            fmt_ratio(m.sharpe),
            fmt_pct(m.total_roi))
    end
end

println("=" ^ 80)
