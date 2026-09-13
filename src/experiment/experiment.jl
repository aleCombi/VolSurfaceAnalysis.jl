# Experiment: the one-shot orchestrator that wires
# (Agent, MarketData, Clock, time window, requested metrics) into a
# single rerunnable record. `run_experiment(exp)` opens the data, runs
# the backtest to a `Ledger`, builds the canonical `PnLSeries` from the
# ledger's round trips, and returns an `ExperimentResult` carrying the
# originating `Experiment` for provenance and rerun.
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
- `name::String`        -- short human label; carried into `ExperimentResult`.
- `agent::Agent`        -- the [`Agent`](@ref) that produces the policy per tick.
- `data::MarketData`    -- the provider *specs*, one per kind; opened per run.
- `clock::Clock`        -- the tick grid (kind + selector); part of core identity.
- `from::DateTime`      -- evaluation window start (inclusive).
- `to::DateTime`        -- evaluation window end (inclusive).
- `outputs::OutputSpec` -- the metrics + artifacts the run produces
                           (see [`OutputSpec`](@ref)).

A kwarg constructor is provided; `outputs` defaults to all registered
metrics and the default artifact set.
"""
struct Experiment
    name    :: String
    agent   :: Agent
    data    :: MarketData
    clock   :: Clock
    from    :: DateTime
    to      :: DateTime
    outputs :: OutputSpec
end

Experiment(; name::AbstractString, agent::Agent, data::MarketData, clock::Clock,
           from::DateTime, to::DateTime,
           outputs::OutputSpec=OutputSpec()) =
    Experiment(String(name), agent, data, clock, from, to, outputs)

"""
    ExperimentResult

Output of [`run_experiment`](@ref): the ledger (events and the order
journal), the canonical PnL intermediate built from it, the computed
metrics, and the originating `Experiment` itself so the run can be
reproduced via `run_experiment(result.experiment)`.

# Fields
- `experiment::Experiment`
- `ledger::Ledger`
- `pnl_series::PnLSeries`
- `metrics::NamedTuple`
"""
struct ExperimentResult
    experiment :: Experiment
    ledger     :: Ledger
    pnl_series :: PnLSeries
    metrics    :: NamedTuple
end

"""
    run_experiment(exp::Experiment) -> ExperimentResult

Open `exp.data`, run the backtest on `exp.clock` with the venue's
defaults (`run_backtest`'s `fill_rule`, `cost_model` and `tick_cents`;
there is deliberately no keyword here, since a value that changes
results must be in the run id, which slice 4 arranges), build the
canonical [`PnLSeries`](@ref) from the ledger's round trips, compute
always-on metrics plus any metrics requested by symbol, close the data,
and return the result.

Open lots at the window end stay open and contribute nothing to the
series until the equity curve of slice 5 marks them; nothing is
force-settled, and expiries inside the window are booked by the
lifecycle of slice 3. Errors loudly if there is no clock tick in the
window, if the clock's selector is not an `Underlying` (an experiment
ticks on an underlying's grid), or if any requested metric symbol is
unknown.
"""
function run_experiment(exp::Experiment)::ExperimentResult
    u = exp.clock.sel
    u isa Underlying || error(
        "run_experiment: the clock selector must be an Underlying (an experiment ticks " *
        "on an underlying's grid), got $(typeof(u)) for experiment $(exp.name)")
    with_data(exp.data) do d
        ledger = run_backtest(exp.agent, d, exp.from, exp.to, exp.clock)
        last_block = asof(d, kind(exp.clock), u, exp.to)
        (isempty(last_block) || first(last_block).timestamp < exp.from) && error(
            "run_experiment: no clock ticks in [$(exp.from), $(exp.to)] " *
            "for experiment $(exp.name)")
        series = pnl_series(ledger)
        metrics = compute_metrics(series, exp.outputs.metrics; kwargs=exp.outputs.metric_params)
        ExperimentResult(exp, ledger, series, metrics)
    end
end
