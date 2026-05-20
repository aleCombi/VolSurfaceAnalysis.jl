# Symbol-addressable dispatch over the optional metric set.
#
# `_METRIC_TABLE` maps each public symbol to (a) the metric function
# and (b) the default kwargs that symbol carries. Every entry is
# self-contained: an experiment that requests `:sharpe` gets the
# baked-in `(periods_per_year=252, risk_free=0.0)` unless it passes
# an override map. This mirrors the dispatch-by-symbol-with-defaults
# pattern used by Optim.jl, MLJ.jl, and the rest of the Julia
# ecosystem for backend selection.

const _METRIC_TABLE = Dict{Symbol, NamedTuple{(:fn, :defaults)}}(
    :sharpe        => (fn=sharpe,        defaults=(periods_per_year=252, risk_free=0.0)),
    :sortino       => (fn=sortino,       defaults=(periods_per_year=252, risk_free=0.0)),
    :max_drawdown  => (fn=max_drawdown,  defaults=NamedTuple()),
    :volatility    => (fn=volatility,    defaults=(periods_per_year=252,)),
    :profit_factor => (fn=profit_factor, defaults=NamedTuple()),
)

"""
    compute_metrics(series::PnLSeries, requested::Vector{Symbol}=Symbol[];
                    kwargs::AbstractDict{Symbol,<:NamedTuple}=Dict{Symbol,NamedTuple}())
        -> NamedTuple

Compute the always-on core metrics (`total_pnl`, `n_round_trips`,
`n_opens`, `n_closes`, `hit_rate`) plus any optional metrics named in
`requested`. The result is a `NamedTuple` whose keys are the always-on
names first (in fixed order), followed by `requested` symbols in the
order given.

Optional metrics carry their own default kwargs (see the
`_METRIC_TABLE`). The `kwargs` argument is a per-metric override map:
`Dict(:sharpe => (periods_per_year=12,))` swaps just the keys you
provide and leaves the rest of that metric's defaults untouched.

Errors loudly if any requested symbol is not in the dispatch table.
"""
function compute_metrics(
    series::PnLSeries,
    requested::Vector{Symbol}=Symbol[];
    kwargs::AbstractDict{Symbol,<:NamedTuple}=Dict{Symbol,NamedTuple}(),
)::NamedTuple
    out = (
        total_pnl     = total_pnl(series),
        n_round_trips = n_round_trips(series),
        n_opens       = series.n_opens,
        n_closes      = series.n_closes,
        hit_rate      = hit_rate(series),
    )
    for sym in requested
        entry = get(_METRIC_TABLE, sym) do
            error("compute_metrics: unknown metric symbol :$sym. " *
                  "Known: $(sort(collect(keys(_METRIC_TABLE))))")
        end
        per_kw = merge(entry.defaults, get(kwargs, sym, NamedTuple()))
        val    = entry.fn(series; per_kw...)
        out    = merge(out, NamedTuple{(sym,)}((val,)))
    end
    return out
end
