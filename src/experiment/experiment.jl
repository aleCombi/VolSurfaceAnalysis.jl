# Experiment: the one-shot orchestrator that wires
# (Agent, ModelDataSource, time window, requested metrics) into a
# single rerunnable record. `run_experiment(exp)` does the backtest,
# builds the canonical `PnLSeries` (marked to spot at the window
# end), and returns an `ExperimentResult` carrying the originating
# `Experiment` for provenance and rerun.
#
# Train/val/test splits, refit cadence, and learning live inside the
# `Agent`; the `Experiment` only sees the evaluation window.

# --- Output spec ---------------------------------------------------------

# Default optional-metric set: every metric in the dispatch table, in
# canonical (sorted) order. Metrics are cheap reductions over a
# `PnLSeries`, so "all of them" is the sensible default when a config
# does not name them -- and it keeps which-metrics out of every header.
_default_metrics() = sort!(collect(keys(_METRIC_TABLE)))

# Default artifacts rendered when a run is materialized. The renderer
# registry lives in the viz layer; identity only needs the symbol here.
_default_artifacts() = [:equity_curve]

"""
    OutputSpec(; metrics, metric_params, artifacts)

Declarative spec of an experiment's *outputs*: the optional metrics to
compute, per-metric kwarg overrides, and the artifacts (plots, ...) to
render when the run is materialized.

Outputs are part of an experiment's *full* identity but not its backtest
(*core*) identity -- two specs differing only here describe the same
backtest viewed differently (see [`full_hash`](@ref) / [`core_hash`](@ref)).

Any field omitted defaults to: all registered metrics at their default
parameters, no overrides, and the default artifact set.

# Fields
- `metrics::Vector{Symbol}`                -- optional metrics to compute.
- `metric_params::Dict{Symbol,NamedTuple}` -- per-metric kwarg overrides.
- `artifacts::Vector{Symbol}`              -- renderer ids to materialize.
"""
struct OutputSpec
    metrics       :: Vector{Symbol}
    metric_params :: Dict{Symbol,NamedTuple}
    artifacts     :: Vector{Symbol}
end

OutputSpec(; metrics=_default_metrics(),
           metric_params=Dict{Symbol,NamedTuple}(),
           artifacts=_default_artifacts()) =
    OutputSpec(Symbol[Symbol(m) for m in metrics],
               Dict{Symbol,NamedTuple}(Symbol(k) => v for (k, v) in metric_params),
               Symbol[Symbol(a) for a in artifacts])

"""
    Experiment

One-shot configuration record for a backtest + its outputs.

# Fields
- `name::String`            -- short human label; carried into `ExperimentResult`.
- `agent::Agent`            -- the [`Agent`](@ref) that produces the policy per tick.
- `source::ModelDataSource` -- the data the engine ticks through.
- `from::DateTime`          -- evaluation window start (inclusive).
- `to::DateTime`            -- evaluation window end (inclusive).
- `outputs::OutputSpec`     -- the metrics + artifacts the run produces
                               (see [`OutputSpec`](@ref)).

A kwarg constructor is provided; `outputs` defaults to all registered
metrics and the default artifact set.
"""
struct Experiment
    name    :: String
    agent   :: Agent
    source  :: ModelDataSource
    from    :: DateTime
    to      :: DateTime
    outputs :: OutputSpec
end

Experiment(; name::AbstractString, agent::Agent, source::ModelDataSource,
           from::DateTime, to::DateTime,
           outputs::OutputSpec=OutputSpec()) =
    Experiment(String(name), agent, source, from, to, outputs)

"""
    ExperimentResult

Output of [`run_experiment`](@ref): the ledger, the canonical PnL
intermediate, the computed metrics, and the originating `Experiment`
itself so the run can be reproduced via
`run_experiment(result.experiment)`.

# Fields
- `experiment::Experiment`
- `positions::Vector{Position}`
- `pnl_series::PnLSeries`
- `metrics::NamedTuple`
"""
struct ExperimentResult
    experiment :: Experiment
    positions  :: Vector{Position}
    pnl_series :: PnLSeries
    metrics    :: NamedTuple
end

# Build the per-leg settle closure for `run_experiment`.
#
# Policy:
# - Case 1: `expiry > window_end`. The leg is genuinely still open past
#   the test window. Conventional open-residual mark using `window_end_spot`.
# - Case 2: `expiry <= window_end`. Looks up `get_spot(source, expiry)`.
#   If present, that's the leg's expiration spot -- held-to-expiry settles
#   honestly. If missing, returns `missing` so `pnl_series` counts the lot
#   in `n_unmarked` rather than silently substituting the wrong number.
#
# TODO: case 2 should fall back to a surface-based theoretical mark
# (`price(get_surface(source, expiry_proxy_ts), K, otype)`) when the spot
# at exact expiry is unavailable but a surface near it is. Today's
# `missing` behaviour exposes the data-gap visibly until that lands.
function _build_settle(source::ModelDataSource, window_end::DateTime,
                       window_end_spot::Float64)
    function settle(expiry::DateTime)::Union{Float64,Missing}
        expiry > window_end && return window_end_spot
        s = get_spot(source, expiry)
        return ismissing(s) ? missing : Float64(s)
    end
    return settle
end

"""
    run_experiment(exp::Experiment) -> ExperimentResult

Run the backtest defined by `exp`, build the canonical [`PnLSeries`](@ref)
with per-leg settlement (each residual lot marked at its own `trade.expiry`
via `get_spot`; legs whose expiry is past the window are marked at the
window-end spot; legs whose expiry-time spot is unavailable inside the
window are counted as `n_unmarked` and skipped from the realized PnL),
compute always-on metrics plus any metrics requested by symbol, and
return the result.

Errors loudly if the time window is empty, the window-end spot is
missing, or any requested metric symbol is unknown.
"""
function run_experiment(exp::Experiment)::ExperimentResult
    positions = run_backtest(exp.agent, exp.source, exp.from, exp.to)
    ts_list = available_timestamps(exp.source, exp.from, exp.to)
    isempty(ts_list) && error(
        "run_experiment: no available timestamps in [$(exp.from), $(exp.to)] " *
        "for experiment $(exp.name)")
    window_end = last(ts_list)
    window_end_spot = get_spot(exp.source, window_end)
    ismissing(window_end_spot) && error(
        "run_experiment: window-end spot missing at $(window_end) for experiment $(exp.name)")
    settle = _build_settle(exp.source, window_end, Float64(window_end_spot))
    series = pnl_series(positions; settle=settle, window_end_spot=window_end_spot)
    metrics = compute_metrics(series, exp.outputs.metrics; kwargs=exp.outputs.metric_params)
    return ExperimentResult(exp, positions, series, metrics)
end
