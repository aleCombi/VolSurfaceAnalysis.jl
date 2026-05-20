# Config-file loading: TOML -> Experiment.
#
# Stdlib TOML only. Each dispatched sum-type (DataSource, Curve, Policy,
# Agent) has its own builder registry keyed by a string discriminator
# (`type = "..."` in the config); the rest of that table is forwarded
# as the builder's kwargs. New concrete types register themselves by
# adding one entry to the relevant table -- same shape as the
# `_METRIC_TABLE` pattern in `src/metrics/dispatch.jl`.

using TOML

# ---- helpers ------------------------------------------------------------

_require(d::AbstractDict, key::AbstractString, where_::AbstractString) =
    haskey(d, key) ? d[key] :
        error("load_experiment: missing required key \"$key\" in $where_")

function _pop_type!(d::AbstractDict, where_::AbstractString)::String
    haskey(d, "type") || error(
        "load_experiment: $where_ table must have a \"type\" key " *
        "(got keys: $(sort(collect(keys(d)))))")
    t = d["type"]
    t isa AbstractString || error(
        "load_experiment: $where_ \"type\" must be a string, got $(typeof(t))")
    return String(t)
end

# Look up `key` in `table`; if missing, error with the known keys.
function _dispatch(table::AbstractDict, key::AbstractString, where_::AbstractString)
    haskey(table, key) || error(
        "load_experiment: unknown $where_ type \"$key\". " *
        "Known: $(sort(collect(keys(table))))")
    return table[key]
end

# ---- Curve builders -----------------------------------------------------

function _build_flat_curve(d::AbstractDict)::Curve
    v = _require(d, "value", "curve(flat)")
    return FlatCurve(Float64(v))
end

function _build_pc_curve(d::AbstractDict)::Curve
    knots_raw  = _require(d, "knots",  "curve(pc)")
    values_raw = _require(d, "values", "curve(pc)")
    knots  = DateTime[DateTime(k) for k in knots_raw]
    values = Float64[Float64(v) for v in values_raw]
    return PCCurve(knots, values)
end

const _CURVE_BUILDERS = Dict{String, Function}(
    "flat" => _build_flat_curve,
    "pc"   => _build_pc_curve,
)

function build_curve(d::AbstractDict)::Curve
    t = _pop_type!(d, "curve")
    return _dispatch(_CURVE_BUILDERS, t, "curve")(d)
end

# ---- DataSource builders ------------------------------------------------

function _build_parquet_source(d::AbstractDict)::DataSource
    underlying = _require(d, "underlying", "source(parquet)")
    max_days = haskey(d, "max_days_cached") ? Int(d["max_days_cached"]) : 3
    if haskey(d, "root")
        return ParquetDataSource(
            String(underlying), String(d["root"]);
            max_days_cached = max_days,
        )
    end
    options_root = _require(d, "options_root", "source(parquet) without \"root\"")
    spot_root    = _require(d, "spot_root",    "source(parquet) without \"root\"")
    return ParquetDataSource(
        String(underlying);
        options_root    = String(options_root),
        spot_root       = String(spot_root),
        max_days_cached = max_days,
    )
end

const _DATA_SOURCE_BUILDERS = Dict{String, Function}(
    "parquet" => _build_parquet_source,
)

function build_data_source(d::AbstractDict)::DataSource
    t = _pop_type!(d, "source")
    return _dispatch(_DATA_SOURCE_BUILDERS, t, "source")(d)
end

# ---- ModelDataSource ----------------------------------------------------

function _build_model_data_source(d::AbstractDict)::ModelDataSource
    rate_tbl = _require(d, "rate", "source")
    div_tbl  = _require(d, "div",  "source")
    rate = build_curve(Dict{String,Any}(rate_tbl))
    div_ = build_curve(Dict{String,Any}(div_tbl))
    # `build_data_source` reads its own "type"/fields from the top-level
    # source table; strip the curve sub-tables so they don't leak into
    # the source kwargs.
    source_only = Dict{String,Any}(k => v for (k, v) in d
                                   if k != "rate" && k != "div")
    chain_src = build_data_source(source_only)
    return ModelDataSource(chain_src; rate=rate, div=div_)
end

# ---- Policy builders ----------------------------------------------------

_build_noop_policy(::AbstractDict)::Policy = NoOpPolicy()

const _POLICY_BUILDERS = Dict{String, Function}(
    "noop" => _build_noop_policy,
)

function build_policy(d::AbstractDict)::Policy
    t = _pop_type!(d, "policy")
    return _dispatch(_POLICY_BUILDERS, t, "policy")(d)
end

# ---- Agent builders -----------------------------------------------------

function _build_static_agent(d::AbstractDict)::Agent
    pol_tbl = _require(d, "policy", "agent(static)")
    return StaticAgent(build_policy(Dict{String,Any}(pol_tbl)))
end

const _AGENT_BUILDERS = Dict{String, Function}(
    "static" => _build_static_agent,
)

function build_agent(d::AbstractDict)::Agent
    t = _pop_type!(d, "agent")
    return _dispatch(_AGENT_BUILDERS, t, "agent")(d)
end

# ---- Top-level loader ---------------------------------------------------

"""
    load_experiment(path::AbstractString) -> Experiment

Parse a TOML file and construct the [`Experiment`](@ref) it describes.

The schema is a flat header (`name`, `from`, `to`, optional `metrics`)
plus two nested tables (`[source]`, `[agent]`). Every dispatched
sum-type (data source, curve, policy, agent) is keyed by a `type`
discriminator; the rest of that table is forwarded to the matching
builder.

# Example

```toml
name = "noop_smoke"
from = 2024-01-15T15:30:00
to   = 2024-01-15T15:32:00
metrics = ["sharpe", "max_drawdown"]

[source]
type       = "parquet"
underlying = "SPY"
root       = "C:/data/polygon"

[source.rate]
type = "flat"
value = 0.04

[source.div]
type = "flat"
value = 0.015

[agent]
type = "static"

[agent.policy]
type = "noop"
```

Errors loudly on missing required keys or unknown `type` discriminators.
"""
function load_experiment(path::AbstractString)::Experiment
    cfg = TOML.parsefile(String(path))
    name = String(_require(cfg, "name", "config"))
    from = DateTime(_require(cfg, "from", "config"))
    to   = DateTime(_require(cfg, "to",   "config"))
    metrics = Symbol[Symbol(m) for m in get(cfg, "metrics", String[])]
    source_tbl = _require(cfg, "source", "config")
    agent_tbl  = _require(cfg, "agent",  "config")
    source = _build_model_data_source(Dict{String,Any}(source_tbl))
    agent  = build_agent(Dict{String,Any}(agent_tbl))
    return Experiment(; name=name, agent=agent, source=source,
                       from=from, to=to, metrics=metrics)
end
