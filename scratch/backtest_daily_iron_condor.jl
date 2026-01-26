# Daily Iron Condor Backtest (minimal, event-driven)
# Uses ScheduledStrategy + backtest_strategy

using Pkg
Pkg.activate(@__DIR__)

using VolSurfaceAnalysis
using Dates

const DEFAULT_HISTORY_PATH = raw"C:\repos\DeribitVols\data\local_db\history"
const DEFAULT_ENTRY_TIME = Time(12, 0)  # UTC

# Short strikes (sell) - inner legs
const DEFAULT_SHORT_PUT_PCT = 0.97
const DEFAULT_SHORT_CALL_PCT = 1.03

# Long strikes (buy) - outer legs
const DEFAULT_LONG_PUT_PCT = 0.90
const DEFAULT_LONG_CALL_PCT = 1.10

const DEFAULT_TAU_TOL = 1e-6
const DEFAULT_QUANTITY = 1.0

function parse_underlying_arg(s::AbstractString)::Underlying
    u = uppercase(strip(s))
    u == "BTC" && return BTC
    u == "ETH" && return ETH
    error("Unknown underlying: $s (expected BTC or ETH)")
end

function pick_data_path(arg_path::Union{Nothing,String})::String
    if arg_path !== nothing
        return arg_path
    end
    return isdir(DEFAULT_HISTORY_PATH) ? DEFAULT_HISTORY_PATH : joinpath(@__DIR__, "..", "data")
end

function build_schedule(dates::Vector{Date}; entry_time::Time=DEFAULT_ENTRY_TIME)
    return [DateTime(d) + Hour(Dates.hour(entry_time)) + Minute(Dates.minute(entry_time)) for d in dates]
end

function main()
    arg_path = length(ARGS) >= 1 ? ARGS[1] : nothing
    n_days = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 7
    underlying = length(ARGS) >= 3 ? parse_underlying_arg(ARGS[3]) : BTC
    short_put_pct = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : DEFAULT_SHORT_PUT_PCT
    short_call_pct = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : DEFAULT_SHORT_CALL_PCT
    long_put_pct = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : DEFAULT_LONG_PUT_PCT
    long_call_pct = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : DEFAULT_LONG_CALL_PCT

    data_path = pick_data_path(arg_path)
    store = LocalDataStore(data_path)

    dates = available_dates(store; underlying=underlying)
    isempty(dates) && error("No dates found in data store")

    last_n = dates[max(1, length(dates) - (n_days - 1)):end]
    schedule = build_schedule(last_n; entry_time=DEFAULT_ENTRY_TIME)

    strategy = IronCondorStrategy(
        schedule,
        Day(1),
        short_put_pct,
        short_call_pct,
        long_put_pct,
        long_call_pct,
        DEFAULT_QUANTITY,
        DEFAULT_TAU_TOL
    )

    start_date = Date(first(last_n))
    end_date = Date(last(last_n)) + Day(1)
    iter = SurfaceIterator(store, underlying; start_date=start_date, end_date=end_date, resolution=Minute(1))

    positions, pnl = backtest_strategy(strategy, iter)

    realized = filter(!ismissing, pnl)
    total_pnl = isempty(realized) ? 0.0 : sum(realized)
    avg_pnl = isempty(realized) ? 0.0 : total_pnl / length(realized)
    missing_n = count(ismissing, pnl)

    println("DAILY IRON CONDOR (minimal)")
    println("Underlying: $underlying | Entry time: $(DEFAULT_ENTRY_TIME)")
    println("Short: Put $(short_put_pct*100)% | Call $(short_call_pct*100)%")
    println("Long : Put $(long_put_pct*100)% | Call $(long_call_pct*100)%")
    println("Entries: $(length(schedule)) | Positions: $(length(positions))")
    println("PnL: total=$(round(total_pnl, digits=2)) | avg/position=$(round(avg_pnl, digits=2))")
    missing_n > 0 && println("Missing settlements: $missing_n")
end

main()
