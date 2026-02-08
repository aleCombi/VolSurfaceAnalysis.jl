# Compare DictDataSource vs ParquetDataSource for correctness and speed.
# Runs the same iron condor backtest through both data sources and verifies
# identical positions and PnL, then reports timing.

using Pkg
Pkg.activate(@__DIR__)

using VolSurfaceAnalysis
using Dates

# ── Configuration (matches backtest_polygon_iron_condor.jl) ──────────────────
const SYMBOL = "SPY"
const END_DATE_CUTOFF = Date(2025, 8, 1)
const ENTRY_TIME_ET = Time(10, 0)
const EXPIRY_INTERVAL = Day(1)
const SHORT_DELTA_ABS = 0.16
const MIN_DELTA_GAP = 0.08
const RISK_FREE_RATE = 0.045
const DIV_YIELD = 0.013
const QUANTITY = 1.0
const TAU_TOL = 1e-6
const MIN_VOLUME = 0
const SPREAD_LAMBDA = 0.0
const WING_OBJECTIVE = :roi
const CONDOR_MAX_LOSS_MIN = 5.0
const CONDOR_MAX_LOSS_MAX = 30.0
const CONDOR_MIN_CREDIT = 0.10

# ── Shared setup ─────────────────────────────────────────────────────────────

function build_entry_timestamps(dates::Vector{Date})::Vector{DateTime}
    [et_to_utc(date, ENTRY_TIME_ET) for date in dates]
end

function make_strike_selector()
    ctx -> begin
        shorts = VolSurfaceAnalysis._delta_strangle_strikes_asymmetric(
            ctx, SHORT_DELTA_ABS, SHORT_DELTA_ABS;
            rate=RISK_FREE_RATE, div_yield=DIV_YIELD
        )
        shorts === nothing && return nothing
        short_put_K, short_call_K = shorts

        wings = VolSurfaceAnalysis._condor_wings_by_objective(
            ctx, short_put_K, short_call_K;
            objective=WING_OBJECTIVE,
            max_loss_min=CONDOR_MAX_LOSS_MIN,
            max_loss_max=CONDOR_MAX_LOSS_MAX,
            min_credit=CONDOR_MIN_CREDIT,
            min_delta_gap=MIN_DELTA_GAP,
            prefer_symmetric=false,
            rate=RISK_FREE_RATE,
            div_yield=DIV_YIELD,
            debug=false
        )
        wings === nothing && return nothing
        long_put_K, long_call_K = wings
        return (short_put_K, short_call_K, long_put_K, long_call_K)
    end
end

# ── Path / reader closures ──────────────────────────────────────────────────

all_dates = available_polygon_dates(DEFAULT_STORE, SYMBOL)
filtered_dates = filter(d -> d < END_DATE_CUTOFF, all_dates)
entry_ts = build_entry_timestamps(filtered_dates)

println("Symbol: $SYMBOL")
println("Dates:  $(length(filtered_dates)) ($(first(filtered_dates)) to $(last(filtered_dates)))")
println()

# Entry spots are needed by read_records to normalise option prices
entry_spots = read_polygon_spot_prices_for_timestamps(
    polygon_spot_root(DEFAULT_STORE), entry_ts; symbol=SYMBOL
)

path_for_ts = ts -> polygon_options_path(DEFAULT_STORE, Date(ts), SYMBOL)
read_records = (path; where="") -> read_polygon_option_records(
    path, entry_spots;
    where=where, min_volume=MIN_VOLUME, warn=false, spread_lambda=SPREAD_LAMBDA
)

# ══════════════════════════════════════════════════════════════════════════════
# Run 1: DictDataSource (current eager approach)
# ══════════════════════════════════════════════════════════════════════════════

println("=" ^ 60)
println("RUN 1: DictDataSource (eager loading)")
println("=" ^ 60)

t_dict = @elapsed begin
    surfaces = build_surfaces_for_timestamps(
        entry_ts; path_for_timestamp=path_for_ts, read_records=read_records
    )
    schedule = sort(collect(keys(surfaces)))

    strategy = IronCondorStrategy(
        schedule, EXPIRY_INTERVAL, 0.7, 1.5;
        rate=RISK_FREE_RATE, div_yield=DIV_YIELD, quantity=QUANTITY,
        tau_tol=TAU_TOL, debug=false, strike_selector=make_strike_selector()
    )

    # Dry run to discover expiry timestamps
    expiry_ts = DateTime[]
    for ts in schedule
        for pos in entry_positions(strategy, surfaces[ts])
            push!(expiry_ts, pos.trade.expiry)
        end
    end
    expiry_ts = unique(expiry_ts)

    settlement_spots = read_polygon_spot_prices_for_timestamps(
        polygon_spot_root(DEFAULT_STORE), expiry_ts; symbol=SYMBOL
    )

    pos_dict, pnl_dict = backtest_strategy(strategy, surfaces, settlement_spots)
end

println("  Positions: $(length(pos_dict))")
println("  Time:      $(round(t_dict, digits=2))s")
println()

# ══════════════════════════════════════════════════════════════════════════════
# Run 2: ParquetDataSource (lazy loading)
# ══════════════════════════════════════════════════════════════════════════════

println("=" ^ 60)
println("RUN 2: ParquetDataSource (lazy loading)")
println("=" ^ 60)

t_parquet = @elapsed begin
    source = ParquetDataSource(
        entry_ts;
        path_for_timestamp=path_for_ts,
        read_records=read_records,
        spot_root=polygon_spot_root(DEFAULT_STORE),
        spot_symbol=SYMBOL,
        spot_multiplier=1.0
    )

    # ParquetDataSource needs the same schedule; derive from available_timestamps
    pq_timestamps = available_timestamps(source)
    # Build strategy using timestamps where surfaces actually exist
    pq_schedule = DateTime[]
    for ts in pq_timestamps
        if get_surface(source, ts) !== nothing
            push!(pq_schedule, ts)
        end
    end

    strategy_pq = IronCondorStrategy(
        pq_schedule, EXPIRY_INTERVAL, 0.7, 1.5;
        rate=RISK_FREE_RATE, div_yield=DIV_YIELD, quantity=QUANTITY,
        tau_tol=TAU_TOL, debug=false, strike_selector=make_strike_selector()
    )

    pos_pq, pnl_pq = backtest_strategy(strategy_pq, source)
end

println("  Positions: $(length(pos_pq))")
println("  Time:      $(round(t_parquet, digits=2))s")
println()

# ══════════════════════════════════════════════════════════════════════════════
# Comparison
# ══════════════════════════════════════════════════════════════════════════════

println("=" ^ 60)
println("CORRECTNESS COMPARISON")
println("=" ^ 60)

n_match = length(pos_dict) == length(pos_pq)
println("  Position count:  Dict=$(length(pos_dict))  Parquet=$(length(pos_pq))  $(n_match ? "MATCH" : "MISMATCH")")

if n_match && !isempty(pos_dict)
    strikes_match = all(
        pos_dict[i].trade.strike == pos_pq[i].trade.strike
        for i in eachindex(pos_dict)
    )
    prices_match = all(
        pos_dict[i].entry_price == pos_pq[i].entry_price
        for i in eachindex(pos_dict)
    )

    # Compare PnL (handling missing)
    pnl_match = all(
        (ismissing(pnl_dict[i]) && ismissing(pnl_pq[i])) ||
        (!ismissing(pnl_dict[i]) && !ismissing(pnl_pq[i]) && pnl_dict[i] ≈ pnl_pq[i])
        for i in eachindex(pnl_dict)
    )

    n_missing_dict = count(ismissing, pnl_dict)
    n_missing_pq   = count(ismissing, pnl_pq)

    total_pnl_dict = sum(skipmissing(pnl_dict))
    total_pnl_pq   = sum(skipmissing(pnl_pq))

    println("  Strikes match:   $(strikes_match ? "YES" : "NO")")
    println("  Prices match:    $(prices_match ? "YES" : "NO")")
    println("  PnL match:       $(pnl_match ? "YES" : "NO")")
    println("  Missing PnL:     Dict=$n_missing_dict  Parquet=$n_missing_pq")
    println("  Total PnL:       Dict=\$$(round(total_pnl_dict, digits=2))  Parquet=\$$(round(total_pnl_pq, digits=2))")

    all_pass = strikes_match && prices_match && pnl_match
    println()
    println("  OVERALL: $(all_pass ? "ALL CHECKS PASSED" : "MISMATCH DETECTED")")
end

println()
println("=" ^ 60)
println("TIMING COMPARISON")
println("=" ^ 60)
println("  DictDataSource:    $(round(t_dict, digits=2))s")
println("  ParquetDataSource: $(round(t_parquet, digits=2))s")
ratio = t_parquet / t_dict
println("  Ratio (Parquet/Dict): $(round(ratio, digits=2))x")
println()
