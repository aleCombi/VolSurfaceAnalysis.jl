# `_METRIC_TABLE` and `compute_metrics`: the optional metrics by symbol,
# each carrying its own defaults.

const _METRIC_TABLE = Dict{Symbol, NamedTuple{(:fn, :defaults)}}(
    :sharpe        => (fn=sharpe,        defaults=(periods_per_year=252, risk_free=0.0)),
    :sortino       => (fn=sortino,       defaults=(periods_per_year=252, risk_free=0.0)),
    :max_drawdown  => (fn=max_drawdown,  defaults=NamedTuple()),
    :volatility    => (fn=volatility,    defaults=(periods_per_year=252,)),
    :profit_factor => (fn=profit_factor, defaults=NamedTuple()),
)

"""
    compute_metrics(L::Ledger, curve::Union{MarkedCurve,Nothing},
                    requested::Vector{Symbol}=Symbol[];
                    kwargs::AbstractDict{Symbol,<:NamedTuple}=Dict{Symbol,NamedTuple}())
        -> NamedTuple

Compute the always-on core metrics (`total_pnl`, `n_round_trips`,
`n_opens`, `n_closes`, `hit_rate`) plus any optional metrics named in
`requested`. The result is a `NamedTuple` whose keys are the always-on
names first (in fixed order), followed by `requested` symbols in the order
given.

`trade_pnl(L)` is derived here, once; the curve is passed in because
building it needs market data. With `curve === nothing` every optional
metric is omitted rather than reported as `NaN`: an absent key means
not computed. The omission is wholesale because every metric takes both
inputs, so `:profit_factor` goes with the rest.

`kwargs` is a per-metric override map: `Dict(:sharpe =>
(periods_per_year=12,))` replaces just the keys given and keeps the rest
of that metric's defaults.

Errors loudly if any requested symbol is not in the dispatch table.
"""
function compute_metrics(
    L::Ledger,
    curve::Union{MarkedCurve,Nothing},
    requested::Vector{Symbol}=Symbol[];
    kwargs::AbstractDict{Symbol,<:NamedTuple}=Dict{Symbol,NamedTuple}(),
)::NamedTuple
    trades = trade_pnl(L)
    out = (
        total_pnl     = total_pnl(trades),
        n_round_trips = n_round_trips(trades),
        n_opens       = n_opens(L),
        n_closes      = n_closes(L),
        hit_rate      = hit_rate(trades),
    )
    for sym in requested
        entry = get(_METRIC_TABLE, sym) do
            error("compute_metrics: unknown metric symbol :$sym. " *
                  "Known: $(sort(collect(keys(_METRIC_TABLE))))")
        end
        curve === nothing && continue
        per_kw  = merge(entry.defaults, get(kwargs, sym, NamedTuple()))
        val     = entry.fn(trades, curve; per_kw...)
        out     = merge(out, NamedTuple{(sym,)}((val,)))
    end
    return out
end
