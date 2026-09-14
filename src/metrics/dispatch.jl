# Symbol-addressable dispatch over the optional metric set.
#
# `_METRIC_TABLE` maps each public symbol to (a) the metric function and
# (b) the default kwargs that symbol carries. Every entry is self-contained: an experiment
# that requests `:sharpe` gets the baked-in
# `(periods_per_year=252, risk_free=0.0)` unless it passes an override
# map. This mirrors the dispatch-by-symbol-with-defaults pattern used by
# Optim.jl, MLJ.jl, and the rest of the Julia ecosystem for backend
# selection.
#
# Every metric takes both inputs -- per-trade dollars and the marked curve
# -- and reads whichever is its sample unit. The table therefore records
# no per-metric input, and adding a metric is one row and one function.
# The consequence to know: the dispatcher cannot tell a curve-reading
# metric from a trade-reading one, so when there is no curve it omits the
# optional set wholesale rather than part of it.

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

**The two inputs.** The ledger is the authority for everything about
trades: `trade_pnl(L)` is derived here, once, and the two fill counts are
read straight off it. The marked curve is passed in because it is the one
result that is *not* a function of the ledger -- marking an open lot needs
market data, so whoever has the data open builds it.

`curve === nothing` is that dependence made visible: the always-on core is
computed as always, and **every** optional metric is omitted from the
result rather than reported as `NaN`. An absent key says "not computed";
`NaN` would say "computed, undefined", which is a different and false
claim (design rule 7). `load_run` takes that path when a run's market data
is not on the machine. The omission is wholesale rather than per-metric
because every metric takes both inputs, so the table does not record which
one a metric reads; a trade metric such as `:profit_factor` is therefore
dropped too, even though it needs no curve.

Optional metrics carry their own default kwargs in the `_METRIC_TABLE`. The `kwargs` argument is a per-metric override map:
`Dict(:sharpe => (periods_per_year=12,))` swaps just the keys you provide
and leaves the rest of that metric's defaults untouched.

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
