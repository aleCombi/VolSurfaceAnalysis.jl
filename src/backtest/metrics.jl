# Backtest metrics
# Aggregations and summary statistics from backtest outputs

using Dates
using Statistics

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
    values = collect(values(pnl_by_key))

    count = length(values)
    total = count == 0 ? 0.0 : sum(values)
    avg = count == 0 ? 0.0 : total / count
    min_pnl = count == 0 ? missing : minimum(values)
    max_pnl = count == 0 ? missing : maximum(values)
    win_rate = count == 0 ? missing : count(x -> x > 0, values) / count

    return BacktestMetrics(
        count,
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

    values = collect(values(pnl_by_key))
    count = length(values)
    total = count == 0 ? 0.0 : sum(values)
    avg = count == 0 ? 0.0 : total / count
    min_pnl = count == 0 ? missing : minimum(values)
    max_pnl = count == 0 ? missing : maximum(values)
    win_rate = count == 0 ? missing : count(x -> x > 0, values) / count

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

    if margin_per_trade !== nothing && margin_per_trade > 0 && count > 0
        returns = [v / margin_per_trade for v in values]
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
        count,
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
