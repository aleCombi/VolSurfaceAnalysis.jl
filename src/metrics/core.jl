# Always-on core metrics: cheap, unparameterized, universally
# interesting. `compute_metrics` (added with the optional set)
# computes these unconditionally on every call; they are not listed
# in `Experiment.metrics` because they cost nothing to ask for.

"""
    total_pnl(series::PnLSeries) -> Float64

Sum of realized PnL across every round trip in `series`. Returns
`0.0` on an empty series.
"""
total_pnl(s::PnLSeries)::Float64 = sum(s.pnl; init=0.0)

"""
    n_round_trips(series::PnLSeries) -> Int

Number of round-trip PnL entries in `series` -- matched chunks plus
still-open residuals. Equal to `length(series.pnl)`.
"""
n_round_trips(s::PnLSeries)::Int = length(s.pnl)

"""
    hit_rate(series::PnLSeries) -> Float64

Fraction of round trips with strictly positive PnL. Returns `NaN` on
an empty series -- hit rate is genuinely undefined with no trades,
and `NaN` propagates through downstream math rather than silently
reading as "0% wins."
"""
function hit_rate(s::PnLSeries)::Float64
    n = length(s.pnl)
    n == 0 && return NaN
    return count(>(0.0), s.pnl) / n
end
