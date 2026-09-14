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

# The venue as the engine has always run it. Both are choices rather than
# facts about the underlying, so they are fields below and enter
# `core_hash`; these are only the values a config that says nothing about
# the venue takes. The loader reads the same two constants, so an omitted
# `[venue]` and an explicitly-default one are one experiment and one id.
const _DEFAULT_FILL_RULE  = :cross_spread
const _DEFAULT_COST_MODEL = :ibkr_pro_us_options

"""
    Experiment

One-shot configuration record for a backtest + its outputs.

# Fields
- `name::String`         -- short human label; carried into `ExperimentResult`.
- `agent::Agent`         -- the [`Agent`](@ref) that produces the policy per tick.
- `data::MarketData`     -- the provider *specs*, one per kind; opened per run.
- `clock::Clock`         -- the tick grid (kind + selector); part of core identity.
- `from::DateTime`       -- evaluation window start (inclusive).
- `to::DateTime`         -- evaluation window end (inclusive).
- `fill_rule::Symbol`    -- how a quote becomes a leg price ([`fill_price`](@ref)).
- `cost_model::Symbol`   -- what an order costs ([`commission`](@ref)).
- `outputs::OutputSpec`  -- the metrics + artifacts the run produces
                            (see [`OutputSpec`](@ref)).

The two venue fields change what the backtest produces, so they are part
of `core_hash`. The class's tick and the settlement rule are not fields:
the tick is the constant `TICK_CENTS`, and settlement style is a contract
fact read per lot from `contract_spec`.

A kwarg constructor is provided; `fill_rule` defaults to `:cross_spread`,
`cost_model` to `:ibkr_pro_us_options`, and `outputs` to all registered
metrics and the default artifact set.
"""
struct Experiment
    name       :: String
    agent      :: Agent
    data       :: MarketData
    clock      :: Clock
    from       :: DateTime
    to         :: DateTime
    fill_rule  :: Symbol
    cost_model :: Symbol
    outputs    :: OutputSpec
end

Experiment(; name::AbstractString, agent::Agent, data::MarketData, clock::Clock,
           from::DateTime, to::DateTime,
           fill_rule::Symbol=_DEFAULT_FILL_RULE,
           cost_model::Symbol=_DEFAULT_COST_MODEL,
           outputs::OutputSpec=OutputSpec()) =
    Experiment(String(name), agent, data, clock, from, to, fill_rule, cost_model, outputs)

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

# The underlying an experiment ticks and trades on, from its clock. One
# experiment is one underlying (`load_experiment` asserts a declared
# policy underlying agrees), so this is also the underlying whose contract
# facts identity projects. Identity and the runner share the check so that
# a mis-specified clock reads the same either way, rather than surfacing as
# a `MethodError` out of `contract_spec`.
function _experiment_underlying(exp::Experiment)::Underlying
    u = exp.clock.sel
    u isa Underlying || error(
        "run_experiment: the clock selector must be an Underlying (an experiment ticks " *
        "on an underlying's grid), got $(typeof(u)) for experiment $(exp.name)")
    return u
end

"""
    run_experiment(exp::Experiment) -> ExperimentResult

Open `exp.data`, run the backtest on `exp.clock` through the
experiment's own venue (`exp.fill_rule` / `exp.cost_model`, both in its
`core_hash`), build the canonical [`PnLSeries`](@ref) from the ledger's
round trips, compute always-on metrics plus any metrics requested by
symbol, close the data, and return the result. There is deliberately no
keyword here: a value that changes results is a field, so that the run id
sees it.

Open lots at the window end stay open and contribute nothing to the
series until the equity curve of slice 5 marks them; nothing is
force-settled, and expiries inside the window are booked by the
engine's lifecycle step. Errors loudly if there is no clock tick in the
window, if the clock's selector is not an `Underlying` (an experiment
ticks on an underlying's grid), or if any requested metric symbol is
unknown.
"""
function run_experiment(exp::Experiment)::ExperimentResult
    u = _experiment_underlying(exp)
    with_data(exp.data) do d
        ledger = run_backtest(exp.agent, d, exp.from, exp.to, exp.clock;
                              fill_rule = exp.fill_rule, cost_model = exp.cost_model)
        last_block = asof(d, kind(exp.clock), u, exp.to)
        (isempty(last_block) || first(last_block).timestamp < exp.from) && error(
            "run_experiment: no clock ticks in [$(exp.from), $(exp.to)] " *
            "for experiment $(exp.name)")
        series = pnl_series(ledger)
        metrics = compute_metrics(series, exp.outputs.metrics; kwargs=exp.outputs.metric_params)
        ExperimentResult(exp, ledger, series, metrics)
    end
end
