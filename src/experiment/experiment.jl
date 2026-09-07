# Experiment: the one-shot orchestrator that wires
# (Agent, MarketData, Clock, time window, requested metrics) into a
# single rerunnable record. `run_experiment(exp)` opens the data, does
# the backtest, builds the canonical `PnLSeries` (marked to spot at the
# window end), and returns an `ExperimentResult` carrying the originating
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

# Build the per-leg settle closure for `run_experiment`, over the opened
# reader map `d`.
#
# Policy:
# - Case 1: `expiry > window_end`. The leg is genuinely still open past
#   the test window. Conventional open-residual mark using `window_end_spot`.
# - Case 2: `expiry <= window_end`. Looks up the spot at `expiry`. If
#   present, that's the leg's expiration spot -- held-to-expiry settles
#   honestly. If absent, returns `missing` so `pnl_series` counts the lot
#   in `n_unmarked` rather than silently substituting the wrong number.
#
# The spot is the clock selector's, i.e. the experiment's one underlying;
# a `spot_for` remap on the surface provider does not apply to
# settlement, the same simplification as at fill time.
#
# TODO: case 2 should fall back to a surface-based theoretical mark when
# the spot at exact expiry is unavailable but a surface near it is.
function _build_settle(d::MarketData, u::Underlying, window_end::DateTime,
                       window_end_spot::Float64)
    function settle(expiry::DateTime)::Union{Float64,Missing}
        expiry > window_end && return window_end_spot
        s = only_or_missing(at(d, SpotPrice, u, expiry))
        return ismissing(s) ? missing : s.price
    end
    return settle
end

"""
    run_experiment(exp::Experiment) -> ExperimentResult

Open `exp.data`, run the backtest on `exp.clock`, build the canonical
[`PnLSeries`](@ref) with per-leg settlement (each residual lot marked at
its own `trade.expiry` via the spot at that instant; legs whose expiry
is past the window are marked at the window-end spot; legs whose
expiry-time spot is unavailable inside the window are counted as
`n_unmarked` and skipped from the realized PnL), compute always-on
metrics plus any metrics requested by symbol, close the data, and
return the result.

The **window end is the last clock tick** at or before `exp.to`: the
timestamp of `asof` on the clock's kind and selector, one partition
walk and no scan. The settle spot is the spot at that tick. Errors
loudly if there is no clock tick in the window, the window-end spot is
missing, or any requested metric symbol is unknown.
"""
function run_experiment(exp::Experiment)::ExperimentResult
    u = exp.clock.sel
    u isa Underlying || error(
        "run_experiment: the clock selector must be an Underlying (window-end spot), " *
        "got $(typeof(u)) for experiment $(exp.name)")
    with_data(exp.data) do d
        positions = run_backtest(exp.agent, d, exp.from, exp.to, exp.clock)
        last_block = asof(d, kind(exp.clock), u, exp.to)
        (isempty(last_block) || first(last_block).timestamp < exp.from) && error(
            "run_experiment: no clock ticks in [$(exp.from), $(exp.to)] " *
            "for experiment $(exp.name)")
        window_end = first(last_block).timestamp
        spot = only_or_missing(at(d, SpotPrice, u, window_end))
        ismissing(spot) && error(
            "run_experiment: window-end spot missing at $(window_end) for experiment $(exp.name)")
        settle = _build_settle(d, u, window_end, spot.price)
        series = pnl_series(positions; settle=settle, window_end_spot=spot.price)
        metrics = compute_metrics(series, exp.outputs.metrics; kwargs=exp.outputs.metric_params)
        ExperimentResult(exp, positions, series, metrics)
    end
end
