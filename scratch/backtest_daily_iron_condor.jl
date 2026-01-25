# Backtest: Sell Daily Iron Condors (multi-day)
# Strategy: Short strangle + long wings for defined risk
#   - Sell OTM put + call (inner strikes)
#   - Buy further OTM put + call (outer strikes / wings)
#
# Usage:
#   julia --project=. backtest_daily_iron_condor.jl [data_path] [n_days] [BTC|ETH] [short_put%] [short_call%] [long_put%] [long_call%]
#
# Examples:
#   julia --project=. backtest_daily_iron_condor.jl                              # defaults
#   julia --project=. backtest_daily_iron_condor.jl path 14 BTC 0.97 1.03 0.90 1.10  # 3% short, 10% long wings

using Pkg
Pkg.activate(@__DIR__)

using VolSurfaceAnalysis
using Dates
using Printf

const DEFAULT_HISTORY_PATH = raw"C:\repos\DeribitVols\data\local_db\history"
const DEFAULT_ENTRY_TIME = Time(12, 0)  # UTC

# Short strikes (sell) - inner legs
const DEFAULT_SHORT_PUT_PCT = 0.97   # Sell put at 97% of spot (3% OTM)
const DEFAULT_SHORT_CALL_PCT = 1.03  # Sell call at 103% of spot (3% OTM)

# Long strikes (buy) - outer legs / wings
const DEFAULT_LONG_PUT_PCT = 0.90    # Buy put at 90% of spot (10% OTM)
const DEFAULT_LONG_CALL_PCT = 1.10   # Buy call at 110% of spot (10% OTM)

output_file = joinpath(@__DIR__, "iron_condor_backtest_results.txt")

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

function find_nearest_strike(strikes, target::Float64)::Float64
    distances = [abs(s - target) for s in strikes]
    return strikes[argmin(distances)]
end

"""
    find_condor_strikes(strikes, spot, short_put_pct, short_call_pct, long_put_pct, long_call_pct)

Find all four strikes for an iron condor:
- short_put: sell put (inner, closer to spot)
- short_call: sell call (inner, closer to spot)
- long_put: buy put (outer wing, further OTM)
- long_call: buy call (outer wing, further OTM)
"""
function find_condor_strikes(strikes, spot::Float64,
                              short_put_pct::Float64, short_call_pct::Float64,
                              long_put_pct::Float64, long_call_pct::Float64)
    # Target strikes
    target_short_put = spot * short_put_pct
    target_short_call = spot * short_call_pct
    target_long_put = spot * long_put_pct
    target_long_call = spot * long_call_pct

    # Filter to OTM strikes
    otm_puts = filter(s -> s < spot, strikes)
    otm_calls = filter(s -> s > spot, strikes)

    # Short strikes (closer to spot)
    short_put = !isempty(otm_puts) ? find_nearest_strike(otm_puts, target_short_put) : find_nearest_strike(strikes, target_short_put)
    short_call = !isempty(otm_calls) ? find_nearest_strike(otm_calls, target_short_call) : find_nearest_strike(strikes, target_short_call)

    # Long strikes (further from spot) - must be more OTM than short strikes
    far_otm_puts = filter(s -> s < short_put, strikes)
    far_otm_calls = filter(s -> s > short_call, strikes)

    long_put = if !isempty(far_otm_puts)
        find_nearest_strike(far_otm_puts, target_long_put)
    else
        # Fallback: use the most OTM put available
        !isempty(otm_puts) ? minimum(otm_puts) : find_nearest_strike(strikes, target_long_put)
    end

    long_call = if !isempty(far_otm_calls)
        find_nearest_strike(far_otm_calls, target_long_call)
    else
        # Fallback: use the most OTM call available
        !isempty(otm_calls) ? maximum(otm_calls) : find_nearest_strike(strikes, target_long_call)
    end

    return (short_put, short_call, long_put, long_call)
end

"""
    daily_iron_condor(store, underlying, trade_date; kwargs...) -> NamedTuple or nothing

Execute a single-day iron condor:
- Entry: Sell inner strangle + buy outer wings
- Exit: Cash settlement at next-day 08:00 UTC
"""
function daily_iron_condor(store::LocalDataStore, underlying::Underlying, trade_date::Date;
                            entry_time::Time=DEFAULT_ENTRY_TIME,
                            short_put_pct::Float64=DEFAULT_SHORT_PUT_PCT,
                            short_call_pct::Float64=DEFAULT_SHORT_CALL_PCT,
                            long_put_pct::Float64=DEFAULT_LONG_PUT_PCT,
                            long_call_pct::Float64=DEFAULT_LONG_CALL_PCT)
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

    # Find all four strikes
    short_put_K, short_call_K, long_put_K, long_call_K = find_condor_strikes(
        strikes, spot_entry,
        short_put_pct, short_call_pct,
        long_put_pct, long_call_pct
    )

    # Validate strike ordering
    if !(long_put_K < short_put_K < spot_entry < short_call_K < long_call_K)
        # Can't form a valid condor, skip
        return nothing
    end

    # Find all four option records
    short_put_idx = findfirst(r -> r.strike == short_put_K && r.option_type == Put, expiry_recs)
    short_call_idx = findfirst(r -> r.strike == short_call_K && r.option_type == Call, expiry_recs)
    long_put_idx = findfirst(r -> r.strike == long_put_K && r.option_type == Put, expiry_recs)
    long_call_idx = findfirst(r -> r.strike == long_call_K && r.option_type == Call, expiry_recs)

    (short_put_idx === nothing || short_call_idx === nothing ||
     long_put_idx === nothing || long_call_idx === nothing) && return nothing

    short_put_rec = expiry_recs[short_put_idx]
    short_call_rec = expiry_recs[short_call_idx]
    long_put_rec = expiry_recs[long_put_idx]
    long_call_rec = expiry_recs[long_call_idx]

    # Get IVs (bid for selling, ask for buying)
    short_put_iv = coalesce(bid_iv(short_put_rec), short_put_rec.mark_iv / 100)
    short_call_iv = coalesce(bid_iv(short_call_rec), short_call_rec.mark_iv / 100)
    long_put_iv = coalesce(ask_iv(long_put_rec), long_put_rec.mark_iv / 100)
    long_call_iv = coalesce(ask_iv(long_call_rec), long_call_rec.mark_iv / 100)

    # Entry prices
    short_put_px = coalesce(short_put_rec.bid_price, vol_to_price(short_put_iv, spot_entry, short_put_K, T, Put))
    short_call_px = coalesce(short_call_rec.bid_price, vol_to_price(short_call_iv, spot_entry, short_call_K, T, Call))
    long_put_px = coalesce(long_put_rec.ask_price, vol_to_price(long_put_iv, spot_entry, long_put_K, T, Put))
    long_call_px = coalesce(long_call_rec.ask_price, vol_to_price(long_call_iv, spot_entry, long_call_K, T, Call))

    # Net premium = sold - bought
    net_premium_btc = (short_put_px + short_call_px) - (long_put_px + long_call_px)
    net_premium_usd = net_premium_btc * spot_entry

    # Calculate max loss (width of spread - net premium)
    put_spread_width = short_put_K - long_put_K
    call_spread_width = long_call_K - short_call_K
    max_loss_usd = max(put_spread_width, call_spread_width) - net_premium_usd

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

    # Settlement payoffs (intrinsic values)
    # Short positions: we owe the payoff
    short_put_payoff = max(short_put_K - spot_settle, 0.0)
    short_call_payoff = max(spot_settle - short_call_K, 0.0)
    # Long positions: we receive the payoff
    long_put_payoff = max(long_put_K - spot_settle, 0.0)
    long_call_payoff = max(spot_settle - long_call_K, 0.0)

    # Net payoff (positive = we owe, negative = we receive)
    net_payoff_usd = (short_put_payoff + short_call_payoff) - (long_put_payoff + long_call_payoff)

    # P&L = premium received - net payoff owed
    pnl_usd = net_premium_usd - net_payoff_usd

    # Calculate actual % OTM achieved
    short_put_otm = (spot_entry - short_put_K) / spot_entry * 100
    short_call_otm = (short_call_K - spot_entry) / spot_entry * 100
    long_put_otm = (spot_entry - long_put_K) / spot_entry * 100
    long_call_otm = (long_call_K - spot_entry) / spot_entry * 100

    return (
        trade_date=trade_date,
        entry_ts=entry_ts,
        expiry=expiry,
        settle_ts=settle_pick,
        spot_entry=spot_entry,
        spot_settle=spot_settle,
        # Strikes
        short_put_K=short_put_K,
        short_call_K=short_call_K,
        long_put_K=long_put_K,
        long_call_K=long_call_K,
        # OTM percentages
        short_put_otm=short_put_otm,
        short_call_otm=short_call_otm,
        long_put_otm=long_put_otm,
        long_call_otm=long_call_otm,
        # Entry prices (BTC fractions)
        short_put_px=short_put_px,
        short_call_px=short_call_px,
        long_put_px=long_put_px,
        long_call_px=long_call_px,
        # Settlement payoffs
        short_put_payoff=short_put_payoff,
        short_call_payoff=short_call_payoff,
        long_put_payoff=long_put_payoff,
        long_call_payoff=long_call_payoff,
        net_payoff_usd=net_payoff_usd,
        # Summary
        net_premium_usd=net_premium_usd,
        max_loss_usd=max_loss_usd,
        pnl_usd=pnl_usd,
    )
end

function main()
    # Parse arguments
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
    date_set = Set(dates)

    results = String[]
    push!(results, "=" ^ 90)
    push!(results, "DAILY IRON CONDOR BACKTEST (multi-day)")
    push!(results, "Sell inner strangle + buy outer wings for capped risk")
    push!(results, "Underlying: $(underlying) | Entry time: $(DEFAULT_ENTRY_TIME) UTC")
    push!(results, "Short strikes: Put @ $(@sprintf("%.1f", short_put_pct*100))% | Call @ $(@sprintf("%.1f", short_call_pct*100))% of spot")
    push!(results, "Long strikes:  Put @ $(@sprintf("%.1f", long_put_pct*100))% | Call @ $(@sprintf("%.1f", long_call_pct*100))% of spot (wings)")
    push!(results, "Data source: $data_path")
    push!(results, "=" ^ 90)

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
        r = daily_iron_condor(store, underlying, d;
                               entry_time=DEFAULT_ENTRY_TIME,
                               short_put_pct=short_put_pct,
                               short_call_pct=short_call_pct,
                               long_put_pct=long_put_pct,
                               long_call_pct=long_call_pct)
        r === nothing && continue
        push!(day_results, r)
    end

    if isempty(day_results)
        push!(results, "ERROR: No days produced an iron condor (missing strikes or data).")
        write_results(results, output_file)
        for line in results; println(line); end
        return
    end

    push!(results, "-" ^ 90)
    push!(results, "PER-DAY RESULTS")
    push!(results, "-" ^ 90)

    total = 0.0
    total_max_loss = 0.0
    wins = 0
    max_pnl = -Inf
    min_pnl = Inf

    for r in day_results
        pnl = r.pnl_usd
        total += pnl
        total_max_loss += r.max_loss_usd
        wins += pnl > 0 ? 1 : 0
        max_pnl = max(max_pnl, pnl)
        min_pnl = min(min_pnl, pnl)

        entry_hhmm = Dates.format(r.entry_ts, "HH:MM")
        settle_fmt = Dates.format(r.settle_ts, "yyyy-mm-dd HH:MM")

        # Header
        push!(results,
              "$(r.trade_date) | entry $(entry_hhmm) spot=$(@sprintf("%.0f", r.spot_entry)) | " *
              "settle $(settle_fmt) spot=$(@sprintf("%.0f", r.spot_settle))")

        # Strikes layout
        push!(results,
              "  STRIKES: Long P=$(@sprintf("%.0f", r.long_put_K)) ($(@sprintf("%.1f", r.long_put_otm))%) | " *
              "Short P=$(@sprintf("%.0f", r.short_put_K)) ($(@sprintf("%.1f", r.short_put_otm))%) | " *
              "[SPOT] | " *
              "Short C=$(@sprintf("%.0f", r.short_call_K)) ($(@sprintf("%.1f", r.short_call_otm))%) | " *
              "Long C=$(@sprintf("%.0f", r.long_call_K)) ($(@sprintf("%.1f", r.long_call_otm))%)")

        # Entry premiums
        short_credit = (r.short_put_px + r.short_call_px) * r.spot_entry
        long_debit = (r.long_put_px + r.long_call_px) * r.spot_entry
        push!(results,
              "  ENTRY: Sold=$(@sprintf("%.2f", short_credit)) USD | Bought=$(@sprintf("%.2f", long_debit)) USD | " *
              "Net credit=$(@sprintf("%.2f", r.net_premium_usd)) USD | Max loss=$(@sprintf("%.2f", r.max_loss_usd)) USD")

        # Settlement
        if r.net_payoff_usd > 0
            push!(results,
                  "  SETTLE: Short payoff=$(@sprintf("%.2f", r.short_put_payoff + r.short_call_payoff)) USD | " *
                  "Long payoff=$(@sprintf("%.2f", r.long_put_payoff + r.long_call_payoff)) USD | " *
                  "Net owed=$(@sprintf("%.2f", r.net_payoff_usd)) USD")
        else
            push!(results, "  SETTLE: All legs expired worthless (max profit)")
        end

        # Return on capital at risk
        roc = r.max_loss_usd > 0 ? (pnl / r.max_loss_usd) * 100 : 0.0
        push!(results, "  P&L: $(@sprintf("%+.2f", pnl)) USD | ROC: $(@sprintf("%+.2f", roc))% (P&L / max loss)")
        push!(results, "")
    end

    avg = total / length(day_results)
    win_rate = wins / length(day_results)

    push!(results, "")
    push!(results, "=" ^ 90)
    push!(results, "SUMMARY")
    push!(results, "=" ^ 90)
    push!(results,
          "Days: $(length(day_results)) | Total P&L: $(@sprintf("%+.2f", total)) USD | " *
          "Avg/day: $(@sprintf("%+.2f", avg)) USD")
    push!(results,
          "Win rate: $(@sprintf("%.1f", win_rate * 100))% | " *
          "Best: $(@sprintf("%+.2f", max_pnl)) USD | Worst: $(@sprintf("%+.2f", min_pnl)) USD")

    avg_max_loss = total_max_loss / length(day_results)
    total_roc = total_max_loss > 0 ? (total / total_max_loss) * 100 : 0.0
    push!(results,
          "Avg max loss (capital at risk): $(@sprintf("%.2f", avg_max_loss)) USD | " *
          "Total ROC: $(@sprintf("%+.2f", total_roc))%")
    push!(results, "")
    push!(results, "Iron Condor structure:")
    push!(results, "  Long Put (wing) < Short Put < [SPOT] < Short Call < Long Call (wing)")
    push!(results, "  Max profit = net premium (if spot stays between short strikes)")
    push!(results, "  Max loss = spread width - premium (if spot breaches beyond long strikes)")
    push!(results, "")
    push!(results, "Generated: $(Dates.now())")

    write_results(results, output_file)
    println("Results written to: $output_file")

    for line in results
        println(line)
    end
end

main()
