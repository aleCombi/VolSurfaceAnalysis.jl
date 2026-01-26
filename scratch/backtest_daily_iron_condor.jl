# Daily Iron Condor Backtest (minimal, event-driven)
# Uses ScheduledStrategy + backtest_strategy

using Pkg
Pkg.activate(@__DIR__)

using VolSurfaceAnalysis
using Dates

const DEFAULT_HISTORY_PATH = raw"C:\repos\DeribitVols\data\local_db\history"
const DEFAULT_ENTRY_TIME = Time(12, 0)  # UTC
const DEFAULT_ENTRY_INTERVAL = Hour(4)
const DEFAULT_EXPIRY_INTERVAL = Hour(4)

# Short strikes (sell) - inner legs
const DEFAULT_SHORT_PUT_PCT = 0.98
const DEFAULT_SHORT_CALL_PCT = 1.02

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

function build_schedule(dates::Vector{Date};
                        entry_time::Time=DEFAULT_ENTRY_TIME,
                        entry_interval::Period=DEFAULT_ENTRY_INTERVAL)
    start_dt = DateTime(first(dates)) + Hour(Dates.hour(entry_time)) + Minute(Dates.minute(entry_time))
    end_dt = DateTime(last(dates) + Day(1))
    return collect(start_dt:entry_interval:end_dt)
end

function write_results(lines::Vector{String}, path::String)
    open(path, "w") do io
        for line in lines
            println(io, line)
        end
    end
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
    date_set = Set(dates)
    sched_dates = filter(d -> (d + Day(1)) in date_set, last_n)
    schedule = build_schedule(sched_dates; entry_time=DEFAULT_ENTRY_TIME, entry_interval=DEFAULT_ENTRY_INTERVAL)

    strategy = IronCondorStrategy(
        schedule,
        DEFAULT_EXPIRY_INTERVAL,
        short_put_pct,
        short_call_pct,
        long_put_pct,
        long_call_pct;
        quantity=DEFAULT_QUANTITY,
        tau_tol=DEFAULT_TAU_TOL,
        debug=true
    )

    start_date = Date(first(last_n))
    end_date = Date(last(last_n)) + Day(1)
    iter = SurfaceIterator(store, underlying; start_date=start_date, end_date=end_date, resolution=Minute(1))

    positions, pnl = backtest_strategy(strategy, iter)

    # Per-condor aggregation (group by entry timestamp + expiry)
    grouped = Dict{Tuple{DateTime, DateTime}, Vector{Int}}()
    for (i, pos) in enumerate(positions)
        key = (pos.entry_timestamp, pos.trade.expiry)
        if !haskey(grouped, key)
            grouped[key] = Int[]
        end
        push!(grouped[key], i)
    end

    condor_keys = sort(collect(keys(grouped)); by=k -> k[1])
    condor_pnls = Union{Missing, Float64}[]

    for key in condor_keys
        idxs = grouped[key]
        leg_pnls = pnl[idxs]
        if any(ismissing, leg_pnls)
            push!(condor_pnls, missing)
        else
            push!(condor_pnls, sum(skipmissing(leg_pnls)))
        end
    end

    realized = filter(!ismissing, condor_pnls)
    total_pnl = isempty(realized) ? 0.0 : sum(realized)
    avg_pnl = isempty(realized) ? 0.0 : total_pnl / length(realized)
    missing_n = count(ismissing, condor_pnls)

    # Worst single-position loss
    worst_idx = nothing
    worst_pnl = Inf
    for (i, p) in enumerate(pnl)
        if ismissing(p)
            continue
        end
        if p < worst_pnl
            worst_pnl = p
            worst_idx = i
        end
    end

    lines = String[]

    push!(lines, "DAILY IRON CONDOR (minimal)")
    push!(lines, "Underlying: $underlying | Entry time: $(DEFAULT_ENTRY_TIME)")
    push!(lines, "Short: Put $(short_put_pct*100)% | Call $(short_call_pct*100)%")
    push!(lines, "Long : Put $(long_put_pct*100)% | Call $(long_call_pct*100)%")
    push!(lines, "Expiry interval: $(DEFAULT_EXPIRY_INTERVAL)")
    push!(lines, "Entries: $(length(schedule)) | Positions: $(length(positions)) | Condors: $(length(condor_keys))")
    push!(lines, "PnL: total=$(round(total_pnl, digits=2)) | avg/condor=$(round(avg_pnl, digits=2))")
    missing_n > 0 && push!(lines, "Missing settlements: $missing_n")
    if worst_idx !== nothing
        wp = positions[worst_idx]
        side = wp.trade.direction > 0 ? "Long" : "Short"
        opt = wp.trade.option_type == Call ? "C" : "P"
        push!(lines, "Worst position pnl=$(round(worst_pnl, digits=2)) | $(side) $(opt) K=$(round(wp.trade.strike, digits=2)) expiry=$(wp.trade.expiry) entry_ts=$(wp.entry_timestamp)")
    end

    push!(lines, "")
    push!(lines, "PER-CONDOR P&L")
    for (i, key) in enumerate(condor_keys)
        entry_ts, expiry = key
        idxs = grouped[key]
        entry_spot = positions[idxs[1]].entry_spot
        expiry_surface = surface_at(iter, expiry)
        settle_spot = expiry_surface === nothing ? missing : expiry_surface.spot
        pnl_i = condor_pnls[i]
        strikes = sort(positions[idxs], by=p -> p.trade.strike)
        strikes_str = join((string(round(p.trade.strike, digits=2)) for p in strikes), " | ")
        push!(lines, "entry=$(entry_ts) expiry=$(expiry) entry_spot=$(round(entry_spot, digits=2)) settle_spot=$(settle_spot) pnl=$(pnl_i)")
        push!(lines, "  strikes: $(strikes_str)")

        if settle_spot === missing
            continue
        end

        entry_costs = [entry_cost(p) for p in strikes]
        payoffs = [payoff(p.trade, settle_spot) for p in strikes]
        net_entry = sum(entry_costs)
        net_payoff = sum(payoffs)
        net_pnl = net_payoff - net_entry
        qty = positions[idxs[1]].trade.quantity
        put_width = strikes[2].trade.strike - strikes[1].trade.strike
        call_width = strikes[4].trade.strike - strikes[3].trade.strike
        max_loss = max(put_width, call_width) * qty + net_entry
        push!(lines, "  net_entry=$(round(net_entry, digits=2)) net_payoff=$(round(net_payoff, digits=2)) pnl=$(round(net_pnl, digits=2)) max_loss=$(round(max_loss, digits=2))")

        for (p, ec, pf) in zip(strikes, entry_costs, payoffs)
            side = p.trade.direction > 0 ? "Long" : "Short"
            opt = p.trade.option_type == Call ? "C" : "P"
            entry_px = p.entry_price * p.entry_spot
            entry_frac = p.entry_price
            rec = find_record(surface_at(iter, entry_ts), p.trade.strike, p.trade.expiry, p.trade.option_type)
            if rec === missing
                push!(lines, "    $(side) $(opt) K=$(round(p.trade.strike, digits=2)) entry_px=$(round(entry_px, digits=6)) entry_frac=$(round(entry_frac, digits=8)) entry_cost=$(round(ec, digits=6)) payoff=$(round(pf, digits=6)) rec=missing")
            else
                bid = rec.bid_price
                ask = rec.ask_price
                mark = rec.mark_price
                push!(lines, "    $(side) $(opt) K=$(round(p.trade.strike, digits=2)) entry_px=$(round(entry_px, digits=6)) entry_frac=$(round(entry_frac, digits=8)) entry_cost=$(round(ec, digits=6)) payoff=$(round(pf, digits=6)) bid=$(bid) ask=$(ask) mark=$(mark)")
            end
        end
    end

    output_file = joinpath(@__DIR__, "iron_condor_backtest_results.txt")
    write_results(lines, output_file)
    for line in lines
        println(line)
    end
end

main()
