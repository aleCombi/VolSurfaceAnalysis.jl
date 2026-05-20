# Optional symbol-addressable metrics. Each takes a `PnLSeries` plus
# kwargs; the dispatch table in `dispatch.jl` carries the defaults so
# every symbol has a complete, callable contract on its own.
#
# Sampling convention: each round trip is one observation. Sharpe /
# Sortino / volatility annualize by multiplying by `sqrt(periods_per_year)`
# under the standard assumption that `periods_per_year` round trips
# occur per year. Callers whose trade cadence differs override the
# default 252 to match their strategy.

using Statistics: mean, std

"""
    sharpe(series::PnLSeries; periods_per_year::Real=252, risk_free::Real=0.0) -> Float64

Annualized Sharpe ratio over the per-round-trip PnL series. Treats
each round trip as one observation, subtracts `risk_free / periods_per_year`
per observation, divides excess mean by sample std, and scales by
`sqrt(periods_per_year)`.

Returns `NaN` when fewer than two round trips exist or when realized
PnL has zero variance (Sharpe is undefined in both cases).
"""
function sharpe(s::PnLSeries; periods_per_year::Real=252, risk_free::Real=0.0)::Float64
    length(s.pnl) < 2 && return NaN
    excess = s.pnl .- risk_free / periods_per_year
    sigma  = std(excess; corrected=true)
    sigma == 0 && return NaN
    return (mean(excess) / sigma) * sqrt(periods_per_year)
end

"""
    sortino(series::PnLSeries; periods_per_year::Real=252, risk_free::Real=0.0) -> Float64

Annualized Sortino ratio: like [`sharpe`](@ref) but the denominator
is the downside deviation (RMS of strictly-negative excess returns,
divided by the full sample size). Returns `NaN` when fewer than two
round trips, when no downside observations exist, or when the
downside deviation is zero.
"""
function sortino(s::PnLSeries; periods_per_year::Real=252, risk_free::Real=0.0)::Float64
    length(s.pnl) < 2 && return NaN
    excess   = s.pnl .- risk_free / periods_per_year
    downside = filter(<(0.0), excess)
    isempty(downside) && return NaN
    dd = sqrt(sum(x^2 for x in downside) / length(excess))
    dd == 0 && return NaN
    return (mean(excess) / dd) * sqrt(periods_per_year)
end

"""
    max_drawdown(series::PnLSeries) -> Float64

Peak-to-trough drop on the equity curve, expressed in cash (USD).
Always non-negative. Returns `0.0` on an empty series (no curve, no
drawdown).
"""
function max_drawdown(s::PnLSeries)::Float64
    isempty(s.pnl) && return 0.0
    eq = equity_curve(s)
    peak   = eq[1]
    max_dd = 0.0
    for v in eq
        peak   = max(peak, v)
        dd     = peak - v
        dd > max_dd && (max_dd = dd)
    end
    return max_dd
end

"""
    volatility(series::PnLSeries; periods_per_year::Real=252) -> Float64

Annualized standard deviation of per-round-trip PnL. Returns `NaN`
when fewer than two round trips exist.
"""
function volatility(s::PnLSeries; periods_per_year::Real=252)::Float64
    length(s.pnl) < 2 && return NaN
    return std(s.pnl; corrected=true) * sqrt(periods_per_year)
end

"""
    profit_factor(series::PnLSeries) -> Float64

Ratio of gross wins to gross losses (absolute value). Returns `Inf`
when there are wins but no losses, and `NaN` when both are zero
(empty series or all breakevens).
"""
function profit_factor(s::PnLSeries)::Float64
    gross_win  = sum(x for x in s.pnl if x > 0; init=0.0)
    gross_loss = -sum(x for x in s.pnl if x < 0; init=0.0)
    if gross_loss == 0
        return gross_win == 0 ? NaN : Inf
    end
    return gross_win / gross_loss
end
