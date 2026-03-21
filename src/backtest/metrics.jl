# Backtest metrics
# Aggregations and summary statistics from backtest outputs

using Dates
using Statistics
using DataFrames
using Printf

# =============================================================================
# Struct
# =============================================================================

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
- `avg_bid_ask_spread_usd::Union{Float64,Missing}`: mean entry bid-ask spread (USD)
- `avg_bid_ask_spread_rel::Union{Float64,Missing}`: mean entry bid-ask spread / mid (unitless)
- `total_roi::Union{Float64,Missing}`: total ROI on margin basis (if provided)
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
    avg_bid_ask_spread_usd::Union{Float64,Missing}
    avg_bid_ask_spread_rel::Union{Float64,Missing}
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

# =============================================================================
# Core aggregation
# =============================================================================

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
    condor_group_stats(group_positions::Vector{Position}) -> Union{NamedTuple, Missing}

Compute iron condor stats for a single grouped trade (all 4 legs).

Returns a NamedTuple with:
- `credit`: net credit (positive) or debit (negative)
- `max_loss`: max loss in USD
- `width_put`: put spread width
- `width_call`: call spread width
"""
function condor_group_stats(group_positions::Vector{Position})::Union{NamedTuple,Missing}
    short_puts = Float64[]
    long_puts = Float64[]
    short_calls = Float64[]
    long_calls = Float64[]
    total_entry_cost = 0.0

    for pos in group_positions
        total_entry_cost += entry_cost(pos)
        t = pos.trade
        if t.option_type == Put
            if t.direction < 0
                push!(short_puts, t.strike)
            else
                push!(long_puts, t.strike)
            end
        else
            if t.direction < 0
                push!(short_calls, t.strike)
            else
                push!(long_calls, t.strike)
            end
        end
    end

    if isempty(short_puts) || isempty(long_puts) || isempty(short_calls) || isempty(long_calls)
        return missing
    end

    if length(short_puts) != 1 || length(long_puts) != 1 || length(short_calls) != 1 || length(long_calls) != 1
        return missing
    end

    short_put = maximum(short_puts)
    long_put = minimum(long_puts)
    short_call = minimum(short_calls)
    long_call = maximum(long_calls)

    width_put = abs(short_put - long_put)
    width_call = abs(long_call - short_call)
    max_width = max(width_put, width_call)

    credit = -total_entry_cost
    max_loss = if total_entry_cost <= 0
        max_width - credit  # credit > 0 for short condor
    else
        total_entry_cost     # debit for long condor
    end
    max_loss = max(max_loss, 0.0)

    return (
        credit=credit,
        max_loss=max_loss,
        width_put=width_put,
        width_call=width_call
    )
end

"""
    condor_max_loss_by_key(positions; key=default_key) -> Dict{Any,Float64}

Build a dictionary of per-condor max loss by grouped trade key.
Only groups that can be parsed as condors and have positive max loss are included.
"""
function condor_max_loss_by_key(
    positions::Vector{Position};
    key::Function = pos -> (pos.entry_timestamp, pos.trade.expiry)
)::Dict{Any,Float64}
    groups = Dict{Any,Vector{Position}}()
    for pos in positions
        k = key(pos)
        push!(get!(groups, k, Position[]), pos)
    end

    margin_by_key = Dict{Any,Float64}()
    for (k, group_positions) in groups
        stats = condor_group_stats(group_positions)
        stats === missing && continue
        stats.max_loss > 0 || continue
        margin_by_key[k] = stats.max_loss
    end

    return margin_by_key
end

# =============================================================================
# Metrics computation
# =============================================================================

"""
    _return_metrics(returns, pnl_total, margin_total, duration_years, annualization)

Internal helper: compute ROI, Sharpe, Sortino from a returns vector.
Returns a NamedTuple with all return-based metric fields.
"""
function _return_metrics(returns::Vector{Float64}, pnl_total::Float64, margin_total::Float64,
                         duration_years, annualization::Int)
    avg_return = mean(returns)
    vol = std(returns)

    sharpe = missing
    if vol > 0
        sharpe = (avg_return / vol) * sqrt(annualization)
    end

    semi_dev = sqrt(mean(min.(0, returns) .^ 2))
    sortino = missing
    if semi_dev > 0
        sortino = (avg_return / semi_dev) * sqrt(annualization)
    end

    total_roi = pnl_total / margin_total

    annualized_roi_simple = missing
    annualized_roi_cagr = missing
    if duration_years !== missing && duration_years > 0
        annualized_roi_simple = total_roi / duration_years
        if 1 + total_roi > 0
            annualized_roi_cagr = (1 + total_roi)^(1 / duration_years) - 1
        end
    end

    return (
        total_roi=total_roi,
        annualized_roi_simple=annualized_roi_simple,
        annualized_roi_cagr=annualized_roi_cagr,
        avg_return=avg_return,
        volatility=vol,
        sharpe=sharpe,
        sortino=sortino
    )
end

"""
    performance_metrics(positions, pnls; margin_per_trade=nothing, margin_by_key=nothing,
                        annualization=252, key=default_key)

Compute extended performance metrics.

- If `margin_per_trade` is provided, fixed-margin return metrics are computed.
- If `margin_by_key` is provided, per-trade margin metrics are computed from
  each grouped trade key.
- If neither is provided, ROI/return fields are returned as `missing`.
"""
function performance_metrics(
    positions::Vector{Position},
    pnls::Vector{Union{Missing,Float64}};
    margin_per_trade::Union{Nothing,Float64}=nothing,
    margin_by_key::Union{Nothing,AbstractDict}=nothing,
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

    avg_spread_usd = average_entry_spread(positions; unit=:usd)
    avg_spread_rel = average_entry_spread(positions; unit=:relative)

    if margin_per_trade !== nothing && margin_by_key !== nothing
        throw(ArgumentError("Provide either margin_per_trade or margin_by_key, not both"))
    end

    # Compute return-based metrics
    rm = nothing
    if margin_by_key !== nothing && trade_count > 0
        returns = Float64[]
        pnl_for_returns = 0.0
        margin_total = 0.0

        for (k, pnl) in pnl_by_key
            margin = get(margin_by_key, k, nothing)
            if margin === nothing
                continue
            end
            margin = Float64(margin)
            if !isfinite(margin) || margin <= 0
                continue
            end

            push!(returns, pnl / margin)
            pnl_for_returns += pnl
            margin_total += margin
        end

        if !isempty(returns)
            rm = _return_metrics(returns, pnl_for_returns, margin_total, duration_years, annualization)
        end
    elseif margin_per_trade !== nothing && margin_per_trade > 0 && trade_count > 0
        returns = [v / margin_per_trade for v in pnl_values]
        rm = _return_metrics(returns, total, margin_per_trade, duration_years, annualization)
    end

    return PerformanceMetrics(
        trade_count,
        missing_count,
        total,
        avg,
        min_pnl,
        max_pnl,
        win_rate,
        avg_spread_usd,
        avg_spread_rel,
        rm === nothing ? missing : rm.total_roi,
        rm === nothing ? missing : rm.annualized_roi_simple,
        rm === nothing ? missing : rm.annualized_roi_cagr,
        rm === nothing ? missing : rm.avg_return,
        rm === nothing ? missing : rm.volatility,
        rm === nothing ? missing : rm.sharpe,
        rm === nothing ? missing : rm.sortino,
        duration_days,
        duration_years
    )
end

"""
    performance_metrics(result::BacktestResult; annualization=252)

Convenience method that auto-computes condor max loss margin from a BacktestResult.
Returns `nothing` if the result is empty or has no valid condor groups.
"""
function performance_metrics(result::BacktestResult; annualization::Int=252)
    isempty(result.positions) && return nothing
    margin = condor_max_loss_by_key(result.positions)
    isempty(margin) && return nothing
    try
        performance_metrics(result.positions, result.pnl; margin_by_key=margin, annualization=annualization)
    catch
        nothing
    end
end

# =============================================================================
# Data preparation
# =============================================================================

"""
    average_entry_spread(positions; unit=:usd) -> Union{Float64,Missing}

Compute the mean entry bid-ask spread across positions. Uses entry bid/ask
when available. If `unit=:usd`, converts spread to USD via entry spot.
If `unit=:relative`, returns (ask - bid) / mid (unitless).
"""
function average_entry_spread(
    positions::Vector{Position};
    unit::Symbol=:usd
)::Union{Float64,Missing}
    spreads = Float64[]
    for pos in positions
        bid = pos.entry_bid
        ask = pos.entry_ask
        if ismissing(bid) || ismissing(ask)
            continue
        end
        spread = Float64(ask) - Float64(bid)
        isfinite(spread) || continue
        if unit == :usd
            spread *= pos.entry_spot
        elseif unit == :relative
            mid = (Float64(ask) + Float64(bid)) / 2.0
            mid <= 0.0 && continue
            spread /= mid
        end
        push!(spreads, spread)
    end
    return isempty(spreads) ? missing : mean(spreads)
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

# =============================================================================
# Formatting helpers
# =============================================================================

"""Format a P&L value for display (e.g., "\$1234")."""
fmt_pnl(v) = ismissing(v) ? "n/a" : @sprintf("\$%.0f", v)

"""Format a ratio (Sharpe, Sortino) for display (e.g., "1.23")."""
fmt_ratio(v) = ismissing(v) ? "n/a" : @sprintf("%.2f", v)

"""Format a fraction as a percentage for display (e.g., "65.0%")."""
fmt_pct(v) = ismissing(v) ? "n/a" : @sprintf("%.1f%%", v * 100)

"""Format a currency value with cents for display (e.g., "\$1.23")."""
fmt_currency(v) = ismissing(v) ? "n/a" : @sprintf("\$%.2f", v)

"""Format a metric value, optionally as percentage (e.g., "12.34%" or "1.23")."""
fmt_metric(v; pct::Bool=false) = ismissing(v) ? "n/a" : pct ? "$(round(v * 100, digits=2))%" : "$(round(v, digits=2))"

# =============================================================================
# DataFrame serialization
# =============================================================================

"""
    condor_trade_table(positions, pnls; key=default_key) -> DataFrame

Build a per-condor trade table with P&L and max loss metrics.
"""
function condor_trade_table(
    positions::Vector{Position},
    pnls::Vector{Union{Missing,Float64}};
    key::Function = pos -> (pos.entry_timestamp, pos.trade.expiry)
)::DataFrame
    pnl_by_key, _ = aggregate_pnl(positions, pnls; key=key)

    groups = Dict{Any,Vector{Position}}()
    for pos in positions
        k = key(pos)
        push!(get!(groups, k, Position[]), pos)
    end

    rows = DataFrame(
        EntryTimestamp=DateTime[],
        Expiry=DateTime[],
        PnL=Union{Missing,Float64}[],
        Credit=Union{Missing,Float64}[],
        MaxLoss=Union{Missing,Float64}[],
        WidthPut=Union{Missing,Float64}[],
        WidthCall=Union{Missing,Float64}[],
        ReturnOnRisk=Union{Missing,Float64}[]
    )

    for k in sort(collect(keys(groups)); by=x -> x[1])
        group_positions = groups[k]
        stats = condor_group_stats(group_positions)
        pnl = get(pnl_by_key, k, missing)

        if stats === missing
            push!(rows, (k[1], k[2], pnl, missing, missing, missing, missing, missing))
            continue
        end

        ror = if ismissing(pnl) || stats.max_loss <= 0
            missing
        else
            pnl / stats.max_loss
        end

        push!(rows, (
            k[1],
            k[2],
            pnl,
            stats.credit,
            stats.max_loss,
            stats.width_put,
            stats.width_call,
            ror
        ))
    end

    return rows
end

"""
    metrics_to_dataframe(m::PerformanceMetrics) -> DataFrame

Serialize a PerformanceMetrics struct into a two-column DataFrame
with columns `Metric::String` and `Value::Any`. Suitable for CSV export.
"""
function metrics_to_dataframe(m::PerformanceMetrics)::DataFrame
    return DataFrame(
        Metric = [
            "count", "missing", "total_pnl", "avg_pnl", "min_pnl", "max_pnl", "win_rate",
            "avg_bid_ask_spread_rel",
            "total_roi", "annualized_roi_simple", "annualized_roi_cagr",
            "avg_return", "volatility", "sharpe", "sortino",
            "duration_days", "duration_years"
        ],
        Value = Any[
            m.count, m.missing, m.total_pnl, m.avg_pnl, m.min_pnl, m.max_pnl, m.win_rate,
            m.avg_bid_ask_spread_rel,
            m.total_roi, m.annualized_roi_simple, m.annualized_roi_cagr,
            m.avg_return, m.volatility, m.sharpe, m.sortino,
            m.duration_days, m.duration_years
        ]
    )
end

"""
    pnl_results_dataframe(positions, pnls; key=default_key) -> DataFrame

Aggregate per-position P&L into grouped trades using `aggregate_pnl`,
then return a DataFrame with columns `EntryDate`, `PnL`, `Result` ("Win"/"Loss")
sorted by entry date.
"""
function pnl_results_dataframe(
    positions::Vector{Position},
    pnls::Vector{Union{Missing,Float64}};
    key::Function = pos -> (pos.entry_timestamp, pos.trade.expiry)
)::DataFrame
    pnl_by_key, _ = aggregate_pnl(positions, pnls; key=key)
    realized_keys = sort(collect(keys(pnl_by_key)); by=k -> k[1])

    results = DataFrame(EntryDate=Date[], PnL=Float64[], Result=String[])
    for k in realized_keys
        pnl_val = pnl_by_key[k]
        push!(results, (Date(k[1]), pnl_val, pnl_val > 0 ? "Win" : "Loss"))
    end
    return results
end

# =============================================================================
# Report generation
# =============================================================================

"""
    format_backtest_report(metrics; title, ...) -> Vector{String}

Generate a standardized text report from backtest results. Returns a vector of
lines. Strategy-specific parameters are passed as `params` key-value pairs.
"""
function format_backtest_report(
    metrics::PerformanceMetrics;
    title::String,
    subtitle::String="",
    params::Vector{Pair{String,String}}=Pair{String,String}[],
    realized_pnls::Vector{Float64}=Float64[],
    n_scheduled::Int=0,
    n_attempted::Int=0,
    n_positions::Int=0,
    n_missing::Int=0,
    margin_description::String="per-condor max loss"
)::Vector{String}
    lines = String[]
    push!(lines, "=" ^ 80)
    push!(lines, title)
    push!(lines, "=" ^ 80)
    push!(lines, "")

    if !isempty(subtitle)
        for sub_line in split(subtitle, "\n")
            push!(lines, sub_line)
        end
        push!(lines, "")
    end

    if !isempty(params)
        for (k, v) in params
            push!(lines, "  $k: $v")
        end
        push!(lines, "")
    end

    push!(lines, "Results:")
    n_scheduled > 0 && push!(lines, "  Scheduled entries: $n_scheduled")
    n_attempted > 0 && push!(lines, "  Actual trades: $n_attempted")
    n_positions > 0 && push!(lines, "  Total positions: $n_positions")
    n_missing > 0 && push!(lines, "  Missing settlements: $n_missing")
    push!(lines, "")

    total_pnl = isempty(realized_pnls) ? 0.0 : sum(realized_pnls)
    push!(lines, "P&L:")
    push!(lines, "  Total: \$$(round(total_pnl, digits=2))")
    push!(lines, "  Count: $(length(realized_pnls))")

    if !isempty(realized_pnls)
        avg_pnl = total_pnl / length(realized_pnls)
        push!(lines, "  Average: \$$(round(avg_pnl, digits=2))")
        push!(lines, "  Min: \$$(round(minimum(realized_pnls), digits=2))")
        push!(lines, "  Max: \$$(round(maximum(realized_pnls), digits=2))")
        winners = count(x -> x > 0, realized_pnls)
        win_rate = winners / length(realized_pnls) * 100
        push!(lines, "  Winners: $winners / $(length(realized_pnls)) ($(round(win_rate, digits=1))%)")
    end

    push!(lines, "")
    push!(lines, "Performance Metrics (return basis: $margin_description):")
    push!(lines, "  Total ROI: $(fmt_metric(metrics.total_roi; pct=true))")
    push!(lines, "  Annualized ROI (CAGR): $(fmt_metric(metrics.annualized_roi_cagr; pct=true))")
    push!(lines, "  Sharpe: $(fmt_ratio(metrics.sharpe))")
    push!(lines, "  Sortino: $(fmt_ratio(metrics.sortino))")
    push!(lines, "  Win Rate: $(fmt_pct(metrics.win_rate))")
    push!(lines, "")
    push!(lines, "=" ^ 80)

    return lines
end
