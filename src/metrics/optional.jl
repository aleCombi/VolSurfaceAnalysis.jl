# The optional metrics, requested by symbol. Each takes both inputs and
# reads whichever is its sample unit; capital is 1 and cancels.

using Statistics: mean, std

"""
    sharpe(trades, curve::MarkedCurve; periods_per_year::Real=252, risk_free::Real=0.0) -> Float64

Annualized Sharpe ratio of `curve`'s session-to-session marked-profit
changes ([`session_changes`](@ref)). Subtracts `risk_free / periods_per_year`
per session, divides the excess mean by the sample standard deviation and
scales by `sqrt(periods_per_year)`.

Returns `NaN` when fewer than two session changes exist or the changes
have zero variance -- Sharpe is undefined in both cases. That is the
"computed, and undefined" answer; a curve that could not be built at all
is a different answer, and `compute_metrics` omits the key rather than
reporting `NaN` for it.
"""
function sharpe(::AbstractVector{<:Real}, c::MarkedCurve;
                periods_per_year::Real=252, risk_free::Real=0.0)::Float64
    x = session_changes(c)
    length(x) < 2 && return NaN
    excess = x .- risk_free / periods_per_year
    sigma  = std(excess; corrected=true)
    sigma == 0 && return NaN
    return (mean(excess) / sigma) * sqrt(periods_per_year)
end

"""
    sortino(trades, curve::MarkedCurve; periods_per_year::Real=252, risk_free::Real=0.0) -> Float64

Annualized Sortino ratio: like [`sharpe`](@ref) but the denominator is the
downside deviation (RMS of strictly-negative excess session changes over
the full sample size). Returns `NaN` with fewer than two session changes,
with no downside session, or when the downside deviation is zero.
"""
function sortino(::AbstractVector{<:Real}, c::MarkedCurve;
                 periods_per_year::Real=252, risk_free::Real=0.0)::Float64
    x = session_changes(c)
    length(x) < 2 && return NaN
    excess   = x .- risk_free / periods_per_year
    downside = filter(<(0.0), excess)
    isempty(downside) && return NaN
    dd = sqrt(sum(v^2 for v in downside) / length(excess))
    dd == 0 && return NaN
    return (mean(excess) / dd) * sqrt(periods_per_year)
end

"""
    volatility(trades, curve::MarkedCurve; periods_per_year::Real=252) -> Float64

Annualized standard deviation of the session-to-session marked-profit
changes, in USD. Returns `NaN` with fewer than two session changes.
"""
function volatility(::AbstractVector{<:Real}, c::MarkedCurve;
                    periods_per_year::Real=252)::Float64
    x = session_changes(c)
    length(x) < 2 && return NaN
    return std(x; corrected=true) * sqrt(periods_per_year)
end

"""
    max_drawdown(trades, curve::MarkedCurve) -> Float64

Largest peak-to-trough drop in marked profit, in USD; always
non-negative, `0.0` on a curve with no marked point.

Walks the marked *levels*, so a position that moves against the book and
recovers before it closes produces a drawdown -- which a curve of closed
trades cannot, because it is flat for as long as anything is open. Marked
profit is measured from zero rather than reset per segment, so an
unmarked session costs the drawdown one observation but does not restart
the peak; the answer over a curve with unmarked sessions is therefore a
lower bound on the true depth, and `n_unmarked` says how many points it
could not see.
"""
function max_drawdown(::AbstractVector{<:Real}, c::MarkedCurve)::Float64
    isempty(c.profit) && return 0.0
    peak   = c.profit[1]
    max_dd = 0.0
    for v in c.profit
        peak = max(peak, v)
        dd   = peak - v
        dd > max_dd && (max_dd = dd)
    end
    return max_dd
end

"""
    profit_factor(trades::AbstractVector{<:Real}, curve) -> Float64

Ratio of gross wins to gross losses (absolute value) over closed trades.
Returns `Inf` when there are wins but no losses, and `NaN` when both are
zero (nothing closed, or all breakevens). A trade metric: its sample unit
is a trade, so it stays on per-trade dollars.
"""
function profit_factor(trades::AbstractVector{<:Real}, ::Any)::Float64
    gross_win  = sum(x for x in trades if x > 0; init=0.0)
    gross_loss = -sum(x for x in trades if x < 0; init=0.0)
    if gross_loss == 0
        return gross_win == 0 ? NaN : Inf
    end
    return gross_win / gross_loss
end
