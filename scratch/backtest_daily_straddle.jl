# Backtest: Sell Daily ATM Straddles (multi-day)
# Strategy: Sell ATM put + call (straddle) targeting 1-day expiry, hold to settlement
#
# Usage:
#   julia --project=. backtest_daily_straddle.jl [data_path] [n_days] [BTC|ETH]
# If data_path is omitted, defaults to:
#   C:\repos\DeribitVols\data\local_db\history

using Pkg
Pkg.activate(@__DIR__)

using VolSurfaceAnalysis
using Dates
using Printf

# Output file
output_file = joinpath(@__DIR__, "straddle_backtest_results.txt")

function main_single_day()
    results = String[]
    push!(results, "=" ^ 70)
    push!(results, "DAILY ATM STRADDLE BACKTEST")
    push!(results, "Strategy: Sell ATM Put + Call, hold to expiry")
    push!(results, "=" ^ 70)
    push!(results, "")

    # Load data
    data_path = joinpath(@__DIR__, "..", "data")
    store = LocalDataStore(data_path)

    push!(results, "Data source: $data_path")
    push!(results, "Available dates: $(available_dates(store))")
    push!(results, "")

    # Load BTC records
    records = load_all(store; underlying=BTC)
    push!(results, "Loaded $(length(records)) BTC records")

    # Get unique timestamps (rounded to minute for cleaner iteration)
    all_ts = sort(unique([floor(r.timestamp, Minute(1)) for r in records]))
    push!(results, "Timestamp range: $(first(all_ts)) to $(last(all_ts))")
    push!(results, "")

    # Find available expiries
    expiries = sort(unique([r.expiry for r in records]))
    push!(results, "Available expiries:")
    for exp in expiries[1:min(5, length(expiries))]
        push!(results, "  $exp")
    end
    push!(results, "")

    # Target the nearest 1-day expiry (2026-01-18 08:00)
    target_expiry = DateTime(2026, 1, 18, 8, 0, 0)
    push!(results, "Target expiry: $target_expiry")
    push!(results, "")

    # Use first available timestamp as entry point
    entry_ts = first(all_ts)
    entry_records = filter(r -> floor(r.timestamp, Minute(1)) == entry_ts, records)

    if isempty(entry_records)
        push!(results, "ERROR: No records at entry timestamp")
        write_results(results, output_file)
        return
    end

    # Build surface at entry
    entry_surface = build_surface(entry_records)
    spot = entry_surface.spot
    push!(results, "-" ^ 70)
    push!(results, "ENTRY: $(entry_surface.timestamp)")
    push!(results, "Spot: $(@sprintf("%.2f", spot))")
    push!(results, "-" ^ 70)

    # Find ATM strikes for target expiry
    atm_records = filter(entry_records) do r
        r.expiry == target_expiry && abs(r.strike - spot) / spot < 0.05
    end

    if isempty(atm_records)
        push!(results, "ERROR: No ATM options found for target expiry")
        write_results(results, output_file)
        return
    end

    # Find the strike closest to spot
    strikes = unique([r.strike for r in atm_records])
    atm_strike = strikes[argmin([abs(s - spot) for s in strikes])]
    push!(results, "ATM Strike: $(@sprintf("%.0f", atm_strike)) (spot: $(@sprintf("%.2f", spot)))")
    push!(results, "")

    # Get put and call records at ATM strike
    put_rec = findfirst(r -> r.strike == atm_strike && r.option_type == Put && r.expiry == target_expiry, atm_records)
    call_rec = findfirst(r -> r.strike == atm_strike && r.option_type == Call && r.expiry == target_expiry, atm_records)

    if isnothing(put_rec) || isnothing(call_rec)
        push!(results, "ERROR: Could not find both put and call at ATM strike")
        write_results(results, output_file)
        return
    end

    put_data = atm_records[put_rec]
    call_data = atm_records[call_rec]

    # Calculate entry prices (we SELL, so we receive bid)
    T = time_to_expiry(target_expiry, entry_surface.timestamp)

    # Use bid IV for selling
    put_bid_iv = coalesce(bid_iv(put_data), put_data.mark_iv / 100)
    call_bid_iv = coalesce(bid_iv(call_data), call_data.mark_iv / 100)

    put_price = vol_to_price(put_bid_iv, spot, atm_strike, T, Put)
    call_price = vol_to_price(call_bid_iv, spot, atm_strike, T, Call)

    straddle_premium = put_price + call_price

    push!(results, "STRADDLE ENTRY (SELL):")
    push!(results, "  Put  @ $(@sprintf("%.0f", atm_strike)): IV=$(@sprintf("%.1f", put_bid_iv*100))%, Price=$(@sprintf("%.4f", put_price)) ($(@sprintf("%.2f", put_price * spot)) USD)")
    push!(results, "  Call @ $(@sprintf("%.0f", atm_strike)): IV=$(@sprintf("%.1f", call_bid_iv*100))%, Price=$(@sprintf("%.4f", call_price)) ($(@sprintf("%.2f", call_price * spot)) USD)")
    push!(results, "  Total Premium Received: $(@sprintf("%.4f", straddle_premium)) ($(@sprintf("%.2f", straddle_premium * spot)) USD)")
    push!(results, "  Time to Expiry: $(@sprintf("%.2f", T * 365.25)) days")
    push!(results, "")

    # Track P&L through the day
    push!(results, "-" ^ 70)
    push!(results, "INTRADAY MARK-TO-MARKET")
    push!(results, "-" ^ 70)

    # Sample every hour
    hourly_ts = filter(t -> Dates.minute(t) == 0, all_ts)
    if isempty(hourly_ts)
        hourly_ts = all_ts[1:min(12, length(all_ts)):end]  # Sample ~12 points
    end

    pnl_history = []

    for ts in hourly_ts
        ts_records = filter(r -> floor(r.timestamp, Minute(1)) == ts, records)
        isempty(ts_records) && continue

        surface = build_surface(ts_records)
        current_spot = surface.spot
        T_now = time_to_expiry(target_expiry, surface.timestamp)

        if T_now <= 0
            push!(results, "  $(ts): EXPIRED")
            continue
        end

        # Find current ask IV (to buy back = close short)
        put_vol = find_vol(surface, atm_strike, target_expiry; side=:ask)
        call_vol = find_vol(surface, atm_strike, target_expiry; side=:ask)

        if ismissing(put_vol) || ismissing(call_vol)
            continue
        end

        put_price_now = vol_to_price(put_vol, current_spot, atm_strike, T_now, Put)
        call_price_now = vol_to_price(call_vol, current_spot, atm_strike, T_now, Call)
        straddle_now = put_price_now + call_price_now

        # P&L = premium received - cost to close (we sold, so profit if price dropped)
        pnl = (straddle_premium - straddle_now) * current_spot

        push!(pnl_history, (ts, current_spot, straddle_now, pnl))
        push!(results, "  $(Dates.format(ts, "HH:MM")): Spot=$(@sprintf("%.0f", current_spot)), Straddle=$(@sprintf("%.4f", straddle_now)), P&L=$(@sprintf("%+.2f", pnl)) USD")
    end

    push!(results, "")

    # Summary
    push!(results, "=" ^ 70)
    push!(results, "SUMMARY")
    push!(results, "=" ^ 70)

    if !isempty(pnl_history)
        final_ts, final_spot, final_straddle, final_pnl = last(pnl_history)
        max_pnl = maximum(p[4] for p in pnl_history)
        min_pnl = minimum(p[4] for p in pnl_history)

        push!(results, "Entry Premium: $(@sprintf("%.2f", straddle_premium * spot)) USD")
        push!(results, "Final MTM P&L: $(@sprintf("%+.2f", final_pnl)) USD")
        push!(results, "Max P&L (intraday): $(@sprintf("%+.2f", max_pnl)) USD")
        push!(results, "Min P&L (intraday): $(@sprintf("%+.2f", min_pnl)) USD")
        push!(results, "Spot move: $(@sprintf("%.0f", spot)) → $(@sprintf("%.0f", final_spot)) ($(@sprintf("%+.1f", (final_spot/spot - 1)*100))%)")
        push!(results, "")
        push!(results, "NOTE: Data ends before expiry. Final settlement P&L depends on")
        push!(results, "      spot at 2026-01-18 08:00 UTC. Breakeven range:")
        push!(results, "      $(@sprintf("%.0f", atm_strike - straddle_premium * spot)) - $(@sprintf("%.0f", atm_strike + straddle_premium * spot))")
    end

    push!(results, "")
    push!(results, "Generated: $(Dates.now())")

    write_results(results, output_file)
    println("Results written to: $output_file")

    # Also print to console
    for line in results
        println(line)
    end
end

function write_results(results::Vector{String}, filepath::String)
    open(filepath, "w") do f
        for line in results
            println(f, line)
        end
    end
end

const DEFAULT_HISTORY_PATH = raw"C:\repos\DeribitVols\data\local_db\history"
const DEFAULT_ENTRY_TIME = Time(12, 0)  # UTC

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

function daily_short_straddle(store::LocalDataStore, underlying::Underlying, trade_date::Date;
                              entry_time::Time=DEFAULT_ENTRY_TIME)
    day_records = load_date(store, trade_date; underlying=underlying)
    isempty(day_records) && return nothing

    by_ts, ts = snapshot_map(day_records; resolution=Minute(1))
    isempty(ts) && return nothing

    entry_ts = pick_entry_timestamp(ts, trade_date; entry_time=entry_time)
    entry_recs = by_ts[entry_ts]
    entry_surface = build_surface(entry_recs)
    spot_entry = entry_surface.spot

    expiry = DateTime(trade_date + Day(1)) + Hour(8)
    T = time_to_expiry(expiry, entry_surface.timestamp)
    T <= 0 && return nothing

    expiry_recs = filter(r -> r.expiry == expiry, entry_recs)
    isempty(expiry_recs) && return nothing

    strikes = unique(r.strike for r in expiry_recs)
    strike_distances = [abs(s - spot_entry) for s in strikes]
    atm_strike = strikes[argmin(strike_distances)]

    put = findfirst(r -> r.strike == atm_strike && r.option_type == Put, expiry_recs)
    call = findfirst(r -> r.strike == atm_strike && r.option_type == Call, expiry_recs)
    (put === nothing || call === nothing) && return nothing

    put_rec = expiry_recs[put]
    call_rec = expiry_recs[call]

    put_bid_iv = coalesce(bid_iv(put_rec), put_rec.mark_iv / 100)
    call_bid_iv = coalesce(bid_iv(call_rec), call_rec.mark_iv / 100)

    # Entry price: sell at bid. If bid is missing, fall back to IV-derived price.
    put_entry_px = coalesce(put_rec.bid_price, vol_to_price(put_bid_iv, spot_entry, atm_strike, T, Put))
    call_entry_px = coalesce(call_rec.bid_price, vol_to_price(call_bid_iv, spot_entry, atm_strike, T, Call))
    premium_usd = (put_entry_px + call_entry_px) * spot_entry

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

    # Exit price: cash settlement at expiry using settlement spot (intrinsic), not mark/IV.
    put_exit_usd = max(atm_strike - spot_settle, 0.0)
    call_exit_usd = max(spot_settle - atm_strike, 0.0)
    put_exit_px = put_exit_usd / spot_settle
    call_exit_px = call_exit_usd / spot_settle

    payoff_usd = put_exit_usd + call_exit_usd
    pnl_usd = premium_usd - payoff_usd

    return (
        trade_date=trade_date,
        entry_ts=entry_ts,
        expiry=expiry,
        settle_ts=settle_pick,
        spot_entry=spot_entry,
        spot_settle=spot_settle,
        strike=atm_strike,
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

function main_multi()
    arg_path = length(ARGS) >= 1 ? ARGS[1] : nothing
    n_days = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 7
    underlying = length(ARGS) >= 3 ? parse_underlying_arg(ARGS[3]) : BTC

    data_path = pick_data_path(arg_path)
    store = LocalDataStore(data_path)

    dates = available_dates(store; underlying=underlying)
    date_set = Set(dates)

    results = String[]
    push!(results, "=" ^ 78)
    push!(results, "DAILY ATM STRADDLE BACKTEST (multi-day)")
    push!(results, "Sell ATM put + call, hold to next-day 08:00 settlement")
    push!(results, "Underlying: $(underlying) | Entry time: $(DEFAULT_ENTRY_TIME) UTC")
    push!(results, "Data source: $data_path")
    push!(results, "=" ^ 78)

    if isempty(dates)
        push!(results, "ERROR: No dates found in data store")
        write_results(results, output_file)
        for line in results
            println(line)
        end
        return
    end

    last_n = dates[max(1, length(dates) - (n_days - 1)):end]
    push!(results, "Dates: $(first(last_n)) -> $(last(last_n)) (requested $(n_days))")
    push!(results, "")

    day_results = []

    for d in last_n
        (d + Day(1)) in date_set || continue
        r = daily_short_straddle(store, underlying, d; entry_time=DEFAULT_ENTRY_TIME)
        r === nothing && continue
        push!(day_results, r)
    end

    if isempty(day_results)
        push!(results, "ERROR: No days produced a straddle (missing entry/expiry/settlement snapshots).")
        write_results(results, output_file)
        for line in results
            println(line)
        end
        return
    end

    push!(results, "-" ^ 78)
    push!(results, "PER-DAY RESULTS")
    push!(results, "-" ^ 78)

    total = 0.0
    wins = 0
    for r in day_results
        pnl = r.pnl_usd
        total += pnl
        wins += pnl > 0 ? 1 : 0

        K = round(Int, r.strike)
        entry_hhmm = Dates.format(r.entry_ts, "HH:MM")
        settle_fmt = Dates.format(r.settle_ts, "yyyy-mm-dd HH:MM")
        entry_put_usd = r.put_entry_px * r.spot_entry
        entry_call_usd = r.call_entry_px * r.spot_entry
        exit_put_usd = r.put_exit_usd
        exit_call_usd = r.call_exit_usd

        put_entry_px_str = @sprintf("%.6f", r.put_entry_px)
        call_entry_px_str = @sprintf("%.6f", r.call_entry_px)
        put_exit_px_str = @sprintf("%.6f", r.put_exit_px)
        call_exit_px_str = @sprintf("%.6f", r.call_exit_px)
        entry_put_usd_str = @sprintf("%.2f", entry_put_usd)
        entry_call_usd_str = @sprintf("%.2f", entry_call_usd)
        exit_put_usd_str = @sprintf("%.2f", exit_put_usd)
        exit_call_usd_str = @sprintf("%.2f", exit_call_usd)
        prem_str = @sprintf("%.2f", r.premium_usd)
        payoff_str = @sprintf("%.2f", r.payoff_usd)
        pnl_str = @sprintf("%+.2f", pnl)

        push!(results,
              "$(r.trade_date) | entry $(entry_hhmm) spot=$(round(Int, r.spot_entry)) | " *
              "settle $(settle_fmt) spot=$(round(Int, r.spot_settle)) | K=$K")
        push!(results,
              "  PUT : entry(bid)=$(put_entry_px_str) ($(entry_put_usd_str) USD) | " *
              "exit(settle)=$(put_exit_px_str) ($(exit_put_usd_str) USD)")
        push!(results,
              "  CALL: entry(bid)=$(call_entry_px_str) ($(entry_call_usd_str) USD) | " *
              "exit(settle)=$(call_exit_px_str) ($(exit_call_usd_str) USD)")
        push!(results,
              "  TOTAL: prem=$(prem_str) USD | payoff=$(payoff_str) USD | pnl=$(pnl_str) USD")
        push!(results, "")
    end

    avg = total / length(day_results)
    win_rate = wins / length(day_results)

    push!(results, "")
    push!(results, "=" ^ 78)
    push!(results, "SUMMARY")
    push!(results, "=" ^ 78)
    push!(results,
          "Days: $(length(day_results)) | Total P&L: $(@sprintf("%+.2f", total)) USD | " *
          "Avg/day: $(@sprintf("%+.2f", avg)) USD | Win rate: $(@sprintf("%.1f", win_rate * 100))%")
    push!(results, "Generated: $(Dates.now())")

    write_results(results, output_file)
    println("Results written to: $output_file")

    for line in results
        println(line)
    end
end

main_multi()
