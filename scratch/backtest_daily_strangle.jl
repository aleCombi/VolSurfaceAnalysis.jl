# Backtest: Sell Daily OTM Strangles (multi-day)
# Strategy: Sell OTM put + OTM call (strangle) targeting 1-day expiry, hold to settlement
#
# Usage:
#   julia --project=. backtest_daily_strangle.jl [data_path] [n_days] [BTC|ETH] [put_pct] [call_pct]
#
# Examples:
#   julia --project=. backtest_daily_strangle.jl                     # defaults: 7 days, BTC, 97% put, 103% call
#   julia --project=. backtest_daily_strangle.jl path 14 BTC 0.95 1.05  # 5% OTM each side

using Pkg
Pkg.activate(@__DIR__)

using VolSurfaceAnalysis
using Dates
using Printf

const DEFAULT_HISTORY_PATH = raw"C:\repos\DeribitVols\data\local_db\history"
const DEFAULT_ENTRY_TIME = Time(12, 0)  # UTC
const DEFAULT_PUT_PCT = 0.97   # Put strike at 97% of spot (3% OTM)
const DEFAULT_CALL_PCT = 1.03  # Call strike at 103% of spot (3% OTM)

output_file = joinpath(@__DIR__, "strangle_backtest_results.txt")

function write_results(results::Vector{String}, filepath::String)
    open(filepath, "w") do f
        for line in results
            println(f, line)
        end
    end
end

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

function nearest_timestamp(ts::Vector{DateTime}, target::DateTime)::DateTime
    distances = [abs(Dates.value(t - target)) for t in ts]
    return ts[argmin(distances)]
end

function pick_entry_timestamp(ts::Vector{DateTime}, date::Date; entry_time::Time=DEFAULT_ENTRY_TIME)::DateTime
    target = DateTime(date) + Hour(Dates.hour(entry_time)) + Minute(Dates.minute(entry_time))
    after = filter(t -> t >= target, ts)
    return isempty(after) ? first(ts) : first(after)
end

function snapshot_map(records::Vector{VolRecord}; resolution::Period=Minute(1))
    by_ts = split_by_timestamp(records, resolution)
    ts = sort(collect(keys(by_ts)))
    return by_ts, ts
end

"""
    find_nearest_strike(strikes, target) -> Float64

Find the strike in `strikes` closest to `target`.
"""
function find_nearest_strike(strikes, target::Float64)::Float64
    distances = [abs(s - target) for s in strikes]
    return strikes[argmin(distances)]
end

"""
    find_otm_strikes(strikes, spot, put_pct, call_pct) -> (put_strike, call_strike)

Find OTM strikes for a strangle:
- put_strike: nearest available strike to `spot * put_pct` (should be < spot)
- call_strike: nearest available strike to `spot * call_pct` (should be > spot)
"""
function find_otm_strikes(strikes, spot::Float64, put_pct::Float64, call_pct::Float64)
    target_put = spot * put_pct
    target_call = spot * call_pct

    # Filter to actual OTM strikes if possible
    otm_puts = filter(s -> s < spot, strikes)
    otm_calls = filter(s -> s > spot, strikes)

    put_strike = if !isempty(otm_puts)
        find_nearest_strike(otm_puts, target_put)
    else
        find_nearest_strike(strikes, target_put)
    end

    call_strike = if !isempty(otm_calls)
        find_nearest_strike(otm_calls, target_call)
    else
        find_nearest_strike(strikes, target_call)
    end

    return (put_strike, call_strike)
end

"""
    daily_short_strangle(store, underlying, trade_date; kwargs...) -> NamedTuple or nothing

Execute a single-day short strangle trade:
- Entry: Sell OTM put + OTM call at specified % of spot
- Exit: Cash settlement at next-day 08:00 UTC
"""
function daily_short_strangle(store::LocalDataStore, underlying::Underlying, trade_date::Date;
                               entry_time::Time=DEFAULT_ENTRY_TIME,
                               put_pct::Float64=DEFAULT_PUT_PCT,
                               call_pct::Float64=DEFAULT_CALL_PCT)
    # Load entry day data
    day_records = load_date(store, trade_date; underlying=underlying)
    isempty(day_records) && return nothing

    by_ts, ts = snapshot_map(day_records; resolution=Minute(1))
    isempty(ts) && return nothing

    entry_ts = pick_entry_timestamp(ts, trade_date; entry_time=entry_time)
    entry_recs = by_ts[entry_ts]
    entry_surface = build_surface(entry_recs)
    spot_entry = entry_surface.spot

    # Target next-day 08:00 expiry
    expiry = DateTime(trade_date + Day(1)) + Hour(8)
    T = time_to_expiry(expiry, entry_surface.timestamp)
    T <= 0 && return nothing

    # Filter to options with target expiry
    expiry_recs = filter(r -> r.expiry == expiry, entry_recs)
    isempty(expiry_recs) && return nothing

    # Find available strikes
    strikes = sort(unique(r.strike for r in expiry_recs))

    # Find OTM strikes for strangle
    put_strike, call_strike = find_otm_strikes(strikes, spot_entry, put_pct, call_pct)

    # Ensure we have both options at chosen strikes
    put_idx = findfirst(r -> r.strike == put_strike && r.option_type == Put, expiry_recs)
    call_idx = findfirst(r -> r.strike == call_strike && r.option_type == Call, expiry_recs)
    (put_idx === nothing || call_idx === nothing) && return nothing

    put_rec = expiry_recs[put_idx]
    call_rec = expiry_recs[call_idx]

    # Get bid IVs for selling
    put_bid_iv = coalesce(bid_iv(put_rec), put_rec.mark_iv / 100)
    call_bid_iv = coalesce(bid_iv(call_rec), call_rec.mark_iv / 100)

    # Entry price: sell at bid
    put_entry_px = coalesce(put_rec.bid_price, vol_to_price(put_bid_iv, spot_entry, put_strike, T, Put))
    call_entry_px = coalesce(call_rec.bid_price, vol_to_price(call_bid_iv, spot_entry, call_strike, T, Call))
    premium_usd = (put_entry_px + call_entry_px) * spot_entry

    # Load settlement day data
    settle_date = trade_date + Day(1)
    settle_records = load_date(store, settle_date; underlying=underlying)
    isempty(settle_records) && return nothing

    settle_by_ts, settle_ts = snapshot_map(settle_records; resolution=Minute(1))
    isempty(settle_ts) && return nothing

    settle_target = DateTime(settle_date) + Hour(8)
    settle_pick = nearest_timestamp(settle_ts, settle_target)
    settle_recs = settle_by_ts[settle_pick]
    settle_surface = build_surface(settle_recs)
    spot_settle = settle_surface.spot

    # Settlement payoff (intrinsic value)
    put_exit_usd = max(put_strike - spot_settle, 0.0)
    call_exit_usd = max(spot_settle - call_strike, 0.0)
    put_exit_px = put_exit_usd / spot_settle
    call_exit_px = call_exit_usd / spot_settle

    payoff_usd = put_exit_usd + call_exit_usd
    pnl_usd = premium_usd - payoff_usd

    # Calculate actual % OTM achieved
    put_otm_pct = (spot_entry - put_strike) / spot_entry * 100
    call_otm_pct = (call_strike - spot_entry) / spot_entry * 100

    return (
        trade_date=trade_date,
        entry_ts=entry_ts,
        expiry=expiry,
        settle_ts=settle_pick,
        spot_entry=spot_entry,
        spot_settle=spot_settle,
        put_strike=put_strike,
        call_strike=call_strike,
        put_otm_pct=put_otm_pct,
        call_otm_pct=call_otm_pct,
        put_entry_px=put_entry_px,
        call_entry_px=call_entry_px,
        put_exit_px=put_exit_px,
        call_exit_px=call_exit_px,
        put_exit_usd=put_exit_usd,
        call_exit_usd=call_exit_usd,
        premium_usd=premium_usd,
        payoff_usd=payoff_usd,
        pnl_usd=pnl_usd,
    )
end

function main()
    # Parse arguments
    arg_path = length(ARGS) >= 1 ? ARGS[1] : nothing
    n_days = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 7
    underlying = length(ARGS) >= 3 ? parse_underlying_arg(ARGS[3]) : BTC
    put_pct = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : DEFAULT_PUT_PCT
    call_pct = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : DEFAULT_CALL_PCT

    data_path = pick_data_path(arg_path)
    store = LocalDataStore(data_path)

    dates = available_dates(store; underlying=underlying)
    date_set = Set(dates)

    results = String[]
    push!(results, "=" ^ 82)
    push!(results, "DAILY OTM STRANGLE BACKTEST (multi-day)")
    push!(results, "Sell OTM put + OTM call, hold to next-day 08:00 settlement")
    push!(results, "Underlying: $(underlying) | Entry time: $(DEFAULT_ENTRY_TIME) UTC")
    push!(results, "Target strikes: Put @ $(@sprintf("%.1f", put_pct*100))% | Call @ $(@sprintf("%.1f", call_pct*100))% of spot")
    push!(results, "Data source: $data_path")
    push!(results, "=" ^ 82)

    if isempty(dates)
        push!(results, "ERROR: No dates found in data store")
        write_results(results, output_file)
        for line in results; println(line); end
        return
    end

    last_n = dates[max(1, length(dates) - (n_days - 1)):end]
    push!(results, "Dates: $(first(last_n)) -> $(last(last_n)) (requested $(n_days))")
    push!(results, "")

    day_results = []

    for d in last_n
        (d + Day(1)) in date_set || continue
        r = daily_short_strangle(store, underlying, d;
                                  entry_time=DEFAULT_ENTRY_TIME,
                                  put_pct=put_pct,
                                  call_pct=call_pct)
        r === nothing && continue
        push!(day_results, r)
    end

    if isempty(day_results)
        push!(results, "ERROR: No days produced a strangle (missing entry/expiry/settlement snapshots).")
        write_results(results, output_file)
        for line in results; println(line); end
        return
    end

    push!(results, "-" ^ 82)
    push!(results, "PER-DAY RESULTS")
    push!(results, "-" ^ 82)

    total = 0.0
    wins = 0
    max_pnl = -Inf
    min_pnl = Inf

    for r in day_results
        pnl = r.pnl_usd
        total += pnl
        wins += pnl > 0 ? 1 : 0
        max_pnl = max(max_pnl, pnl)
        min_pnl = min(min_pnl, pnl)

        entry_hhmm = Dates.format(r.entry_ts, "HH:MM")
        settle_fmt = Dates.format(r.settle_ts, "yyyy-mm-dd HH:MM")

        entry_put_usd = r.put_entry_px * r.spot_entry
        entry_call_usd = r.call_entry_px * r.spot_entry

        # Header line with spot and strikes
        push!(results,
              "$(r.trade_date) | entry $(entry_hhmm) spot=$(@sprintf("%.0f", r.spot_entry)) | " *
              "settle $(settle_fmt) spot=$(@sprintf("%.0f", r.spot_settle))")

        # Strikes info
        push!(results,
              "  STRIKES: Put K=$(@sprintf("%.0f", r.put_strike)) ($(@sprintf("%.1f", r.put_otm_pct))% OTM) | " *
              "Call K=$(@sprintf("%.0f", r.call_strike)) ($(@sprintf("%.1f", r.call_otm_pct))% OTM)")

        # Put leg
        push!(results,
              "  PUT : entry=$(@sprintf("%.6f", r.put_entry_px)) ($(@sprintf("%.2f", entry_put_usd)) USD) | " *
              "exit=$(@sprintf("%.6f", r.put_exit_px)) ($(@sprintf("%.2f", r.put_exit_usd)) USD)")

        # Call leg
        push!(results,
              "  CALL: entry=$(@sprintf("%.6f", r.call_entry_px)) ($(@sprintf("%.2f", entry_call_usd)) USD) | " *
              "exit=$(@sprintf("%.6f", r.call_exit_px)) ($(@sprintf("%.2f", r.call_exit_usd)) USD)")

        # Totals
        push!(results,
              "  TOTAL: prem=$(@sprintf("%.2f", r.premium_usd)) USD | payoff=$(@sprintf("%.2f", r.payoff_usd)) USD | " *
              "pnl=$(@sprintf("%+.2f", pnl)) USD")
        push!(results, "")
    end

    avg = total / length(day_results)
    win_rate = wins / length(day_results)

    # Breakeven analysis
    push!(results, "")
    push!(results, "=" ^ 82)
    push!(results, "SUMMARY")
    push!(results, "=" ^ 82)
    push!(results,
          "Days: $(length(day_results)) | Total P&L: $(@sprintf("%+.2f", total)) USD | " *
          "Avg/day: $(@sprintf("%+.2f", avg)) USD")
    push!(results,
          "Win rate: $(@sprintf("%.1f", win_rate * 100))% | " *
          "Best: $(@sprintf("%+.2f", max_pnl)) USD | Worst: $(@sprintf("%+.2f", min_pnl)) USD")
    push!(results, "")
    push!(results, "Strategy: Short strangle profits when spot stays between put and call strikes.")
    push!(results, "          Loss occurs only if spot breaches a strike by more than premium collected.")
    push!(results, "")
    push!(results, "Generated: $(Dates.now())")

    write_results(results, output_file)
    println("Results written to: $output_file")

    for line in results
        println(line)
    end
end

main()
