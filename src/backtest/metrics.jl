# Backtest metrics
# Aggregations and summary statistics from backtest outputs

using Dates
using Statistics
using DataFrames

"""
    BacktestMetrics

Summary metrics computed from aggregated P&L.

# Fields
- `count::Int`: number of aggregated trades
- `missing::Int`: number of missing P&L entries skipped
- `total_pnl::Float64`: total P&L
- `avg_pnl::Float64`: average P&L per aggregated trade
- `min_pnl::Union{Float64,Missing}`: minimum P&L or missing if none
- `max_pnl::Union{Float64,Missing}`: maximum P&L or missing if none
- `win_rate::Union{Float64,Missing}`: fraction of wins or missing if none
"""
struct BacktestMetrics
    count::Int
    missing::Int
    total_pnl::Float64
    avg_pnl::Float64
    min_pnl::Union{Float64,Missing}
    max_pnl::Union{Float64,Missing}
    win_rate::Union{Float64,Missing}
end

"""
    PerformanceMetrics

Extended performance metrics computed from aggregated P&L and optional margin.

# Fields
- `count::Int`: number of aggregated trades
- `missing::Int`: number of missing P&L entries skipped
- `total_pnl::Float64`: total P&L
- `avg_pnl::Float64`: average P&L per aggregated trade
- `min_pnl::Union{Float64,Missing}`: minimum P&L or missing if none
- `max_pnl::Union{Float64,Missing}`: maximum P&L or missing if none
- `win_rate::Union{Float64,Missing}`: fraction of wins or missing if none
- `total_roi::Union{Float64,Missing}`: total ROI on margin (if provided)
- `annualized_roi_simple::Union{Float64,Missing}`: simple annualized ROI
- `annualized_roi_cagr::Union{Float64,Missing}`: CAGR annualized ROI
- `avg_return::Union{Float64,Missing}`: average per-trade return
- `volatility::Union{Float64,Missing}`: per-trade return volatility
- `sharpe::Union{Float64,Missing}`: annualized Sharpe ratio
- `sortino::Union{Float64,Missing}`: annualized Sortino ratio
- `duration_days::Union{Int,Missing}`: span in days between first and last trade
- `duration_years::Union{Float64,Missing}`: span in years between first and last trade
"""
struct PerformanceMetrics
    count::Int
    missing::Int
    total_pnl::Float64
    avg_pnl::Float64
    min_pnl::Union{Float64,Missing}
    max_pnl::Union{Float64,Missing}
    win_rate::Union{Float64,Missing}
    total_roi::Union{Float64,Missing}
    annualized_roi_simple::Union{Float64,Missing}
    annualized_roi_cagr::Union{Float64,Missing}
    avg_return::Union{Float64,Missing}
    volatility::Union{Float64,Missing}
    sharpe::Union{Float64,Missing}
    sortino::Union{Float64,Missing}
    duration_days::Union{Int,Missing}
    duration_years::Union{Float64,Missing}
end

"""
    aggregate_pnl(positions, pnls; key=default_key) -> (Dict, missing_count)

Aggregate per-position P&L into grouped trades using `key(position)`.
The default key groups by `(entry_timestamp, trade.expiry)`.
Missing P&L entries are skipped and counted.
"""
function aggregate_pnl(
    positions::Vector{Position},
    pnls::Vector{Union{Missing,Float64}};
    key::Function = pos -> (pos.entry_timestamp, pos.trade.expiry)
)
    pnl_by_key = Dict{Any,Float64}()
    missing_count = 0

    for (pos, pnl) in zip(positions, pnls)
        if ismissing(pnl)
            missing_count += 1
            continue
        end
        k = key(pos)
        pnl_by_key[k] = get(pnl_by_key, k, 0.0) + pnl
    end

    return pnl_by_key, missing_count
end

"""
    backtest_metrics(positions, pnls; key=default_key) -> BacktestMetrics

Compute summary statistics from backtest outputs. Aggregates per-position P&L
using `aggregate_pnl`, then computes totals, averages, extrema, and win rate.
"""
function backtest_metrics(
    positions::Vector{Position},
    pnls::Vector{Union{Missing,Float64}};
    key::Function = pos -> (pos.entry_timestamp, pos.trade.expiry)
)::BacktestMetrics
    pnl_by_key, missing_count = aggregate_pnl(positions, pnls; key=key)
    pnl_values = collect(values(pnl_by_key))

    trade_count = length(pnl_values)
    total = trade_count == 0 ? 0.0 : sum(pnl_values)
    avg = trade_count == 0 ? 0.0 : total / trade_count
    min_pnl = trade_count == 0 ? missing : minimum(pnl_values)
    max_pnl = trade_count == 0 ? missing : maximum(pnl_values)
    win_rate = trade_count == 0 ? missing : count(x -> x > 0, pnl_values) / trade_count

    return BacktestMetrics(
        trade_count,
        missing_count,
        total,
        avg,
        min_pnl,
        max_pnl,
        win_rate
    )
end

"""
    performance_metrics(positions, pnls; margin_per_trade=nothing, annualization=252, key=default_key)

Compute extended performance metrics. If `margin_per_trade` is provided, ROI and
return-based metrics (Sharpe/Sortino) are computed; otherwise those fields are
returned as `missing`.
"""
function performance_metrics(
    positions::Vector{Position},
    pnls::Vector{Union{Missing,Float64}};
    margin_per_trade::Union{Nothing,Float64}=nothing,
    annualization::Int=252,
    key::Function = pos -> (pos.entry_timestamp, pos.trade.expiry)
)::PerformanceMetrics
    pnl_by_key = Dict{Any,Float64}()
    date_by_key = Dict{Any,Date}()
    missing_count = 0

    for (pos, pnl) in zip(positions, pnls)
        if ismissing(pnl)
            missing_count += 1
            continue
        end
        k = key(pos)
        pnl_by_key[k] = get(pnl_by_key, k, 0.0) + pnl
        if !haskey(date_by_key, k)
            date_by_key[k] = Date(pos.entry_timestamp)
        end
    end

    pnl_values = collect(values(pnl_by_key))
    trade_count = length(pnl_values)
    total = trade_count == 0 ? 0.0 : sum(pnl_values)
    avg = trade_count == 0 ? 0.0 : total / trade_count
    min_pnl = trade_count == 0 ? missing : minimum(pnl_values)
    max_pnl = trade_count == 0 ? missing : maximum(pnl_values)
    win_rate = trade_count == 0 ? missing : count(x -> x > 0, pnl_values) / trade_count

    duration_days = missing
    duration_years = missing
    if !isempty(date_by_key)
        dates = collect(values(date_by_key))
        start_date = minimum(dates)
        end_date = maximum(dates)
        duration_days = Dates.value(end_date - start_date)
        duration_years = duration_days / 365.25
    end

    total_roi = missing
    annualized_roi_simple = missing
    annualized_roi_cagr = missing
    avg_return = missing
    volatility = missing
    sharpe = missing
    sortino = missing

    if margin_per_trade !== nothing && margin_per_trade > 0 && trade_count > 0
        returns = [v / margin_per_trade for v in pnl_values]
        avg_return = mean(returns)
        volatility = std(returns)

        if volatility > 0
            sharpe = (avg_return / volatility) * sqrt(annualization)
        end

        semi_dev = sqrt(mean(min.(0, returns) .^ 2))
        if semi_dev > 0
            sortino = (avg_return / semi_dev) * sqrt(annualization)
        end

        total_roi = total / margin_per_trade

        if duration_years !== missing && duration_years > 0
            annualized_roi_simple = total_roi / duration_years
            if 1 + total_roi > 0
                annualized_roi_cagr = (1 + total_roi)^(1 / duration_years) - 1
            end
        end
    end

    return PerformanceMetrics(
        trade_count,
        missing_count,
        total,
        avg,
        min_pnl,
        max_pnl,
        win_rate,
        total_roi,
        annualized_roi_simple,
        annualized_roi_cagr,
        avg_return,
        volatility,
        sharpe,
        sortino,
        duration_days,
        duration_years
    )
end

"""
    profit_curve(dates, pnls) -> (Vector{Date}, Vector{Float64})

Prepare a per-trade profit curve by pairing dates and P&L values, filtering
missing/non-finite entries, and sorting by date. Returns dates and P&L values
in matching order (non-cumulative).
"""
function profit_curve(dates, pnls)::Tuple{Vector{Date}, Vector{Float64}}
    pairs = Tuple{Date,Float64}[]
    for (d, v) in zip(dates, pnls)
        d === missing && continue
        v === missing && continue
        fv = Float64(v)
        isfinite(fv) || continue
        push!(pairs, (Date(d), fv))
    end
    sort!(pairs, by=first)
    xs = [p[1] for p in pairs]
    ys = [p[2] for p in pairs]
    return xs, ys
end

"""
    settlement_zone_analysis(positions, settlement_spots; first_year_only=true) -> DataFrame

Analyze where the underlying settled relative to the option strikes for each trade group.
Groups positions by entry date (all legs on the same day = one condor/strangle).

# Arguments
- `positions::Vector{Position}`: All positions from the backtest
- `settlement_spots::AbstractDict{DateTime,Float64}`: Settlement prices by expiry timestamp
- `first_year_only::Bool`: If true, only analyze the first year of data (default: true)

# Returns
DataFrame with columns:
- `entry_date`: Date of entry
- `expiry`: Expiry timestamp
- `settlement_spot`: Where the underlying settled
- `zone`: Settlement zone (String)
- `short_put`, `long_put`, `short_call`, `long_call`: Strike prices (missing if not applicable)
"""
function settlement_zone_analysis(
    positions::Vector{Position},
    settlement_spots::AbstractDict{DateTime,Float64};
    first_year_only::Bool=true
)::DataFrame
    # Group positions by (entry_date, expiry)
    groups = Dict{Tuple{Date,DateTime}, Vector{Position}}()
    for pos in positions
        key = (Date(pos.entry_timestamp), pos.trade.expiry)
        push!(get!(groups, key, Position[]), pos)
    end

    # Filter to first year if requested
    all_dates = [k[1] for k in keys(groups)]
    isempty(all_dates) && return DataFrame()

    min_date = minimum(all_dates)
    if first_year_only
        cutoff = min_date + Year(1)
        groups = filter(kv -> kv[1][1] < cutoff, groups)
    end

    rows = NamedTuple[]

    for ((entry_date, expiry), group_positions) in sort(collect(groups); by=first)
        settlement = get(settlement_spots, expiry, missing)
        ismissing(settlement) && continue

        # Extract strikes by option type and direction
        short_puts = Float64[]
        long_puts = Float64[]
        short_calls = Float64[]
        long_calls = Float64[]

        for pos in group_positions
            t = pos.trade
            if t.option_type == Put
                if t.direction < 0
                    push!(short_puts, t.strike)
                else
                    push!(long_puts, t.strike)
                end
            else  # Call
                if t.direction < 0
                    push!(short_calls, t.strike)
                else
                    push!(long_calls, t.strike)
                end
            end
        end

        # Determine zone based on structure
        short_put = isempty(short_puts) ? missing : maximum(short_puts)
        long_put = isempty(long_puts) ? missing : minimum(long_puts)
        short_call = isempty(short_calls) ? missing : minimum(short_calls)
        long_call = isempty(long_calls) ? missing : maximum(long_calls)

        zone = _determine_zone(settlement, short_put, long_put, short_call, long_call)

        push!(rows, (
            entry_date = entry_date,
            expiry = expiry,
            settlement_spot = settlement,
            zone = zone,
            short_put = short_put,
            long_put = long_put,
            short_call = short_call,
            long_call = long_call
        ))
    end

    return DataFrame(rows)
end

"""Determine the settlement zone based on spot and strikes."""
function _determine_zone(
    spot::Float64,
    short_put::Union{Float64,Missing},
    long_put::Union{Float64,Missing},
    short_call::Union{Float64,Missing},
    long_call::Union{Float64,Missing}
)::String
    # Iron condor (all 4 legs)
    if !ismissing(short_put) && !ismissing(long_put) && !ismissing(short_call) && !ismissing(long_call)
        if spot < long_put
            return "below_long_put"
        elseif spot < short_put
            return "between_puts"
        elseif spot <= short_call
            return "max_profit"
        elseif spot <= long_call
            return "between_calls"
        else
            return "above_long_call"
        end
    end

    # Strangle (short put + short call, no longs)
    if !ismissing(short_put) && !ismissing(short_call) && ismissing(long_put) && ismissing(long_call)
        if spot < short_put
            return "below_short_put"
        elseif spot <= short_call
            return "max_profit"
        else
            return "above_short_call"
        end
    end

    return "unknown"
end

"""
    settlement_zone_summary(zone_df::DataFrame) -> DataFrame

Summarize settlement zone frequencies from zone analysis.

# Returns
DataFrame with columns: zone, count, percentage
"""
function settlement_zone_summary(zone_df::DataFrame)::DataFrame
    isempty(zone_df) && return DataFrame(zone=String[], count=Int[], percentage=Float64[])

    total = nrow(zone_df)
    zone_counts = combine(groupby(zone_df, :zone), nrow => :count)
    zone_counts.percentage = round.(zone_counts.count ./ total .* 100, digits=1)

    # Sort by a logical order
    zone_order = ["below_long_put", "between_puts", "below_short_put",
                  "max_profit",
                  "between_calls", "above_short_call", "above_long_call", "unknown"]
    sort!(zone_counts, :zone, by=z -> findfirst(==(z), zone_order))

    return zone_counts
end
