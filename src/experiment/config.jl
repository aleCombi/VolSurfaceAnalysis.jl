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

# ---- QuoteSynthesizer builders -----------------------------------------

function _build_ohlcv_spread(d::AbstractDict)::QuoteSynthesizer
    λ = _require(d, "lambda", "synthesizer(ohlcv_spread)")
    return SpreadFromOHLCV(Float64(λ))
end

const _SYNTHESIZER_BUILDERS = Dict{String, Function}(
    "ohlcv_spread" => _build_ohlcv_spread,
)

function build_synthesizer(d::AbstractDict)::QuoteSynthesizer
    t = _pop_type!(d, "synthesizer")
    return _dispatch(_SYNTHESIZER_BUILDERS, t, "synthesizer")(d)
end

# ---- DataSource builders ------------------------------------------------

function _build_parquet_source(d::AbstractDict)::DataSource
    underlying = _require(d, "underlying", "source(parquet)")
    synth_tbl  = _require(d, "synthesizer", "source(parquet)")
    synth = build_synthesizer(Dict{String,Any}(synth_tbl))
    max_days = haskey(d, "max_days_cached") ? Int(d["max_days_cached"]) : 3
    if haskey(d, "root")
        return ParquetDataSource(
            String(underlying), String(d["root"]);
            synthesizer     = synth,
            max_days_cached = max_days,
        )
    end
    options_root = _require(d, "options_root", "source(parquet) without \"root\"")
    spot_root    = _require(d, "spot_root",    "source(parquet) without \"root\"")
    return ParquetDataSource(
        String(underlying);
        options_root    = String(options_root),
        spot_root       = String(spot_root),
        synthesizer     = synth,
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
    # the source kwargs. (The synthesizer sub-table is consumed by the
    # source builder itself, so it stays in.)
    source_only = Dict{String,Any}(k => v for (k, v) in d
                                   if k != "rate" && k != "div")
    chain_src = build_data_source(source_only)
    return ModelDataSource(chain_src; rate=rate, div=div_)
end

# ---- Policy builders ----------------------------------------------------

_build_noop_policy(::AbstractDict)::Policy = NoOpPolicy()

# Parse `entry_time` from either a TOML local-time literal (stdlib TOML
# returns it as a `Dates.Time` directly) or an `"HH:MM:SS"` string.
function _parse_entry_time(v)::Time
    v isa Time             && return v
    v isa AbstractString   && return Time(String(v))
    error("policy(daily_short_strangle): entry_time must be a local-time " *
          "literal or \"HH:MM:SS\" string, got $(typeof(v))")
end

function _build_daily_short_strangle(d::AbstractDict)::Policy
    underlying  = _require(d, "underlying",  "policy(daily_short_strangle)")
    entry_raw   = _require(d, "entry_time",  "policy(daily_short_strangle)")
    expiry_days = _require(d, "expiry_days", "policy(daily_short_strangle)")
    put_delta   = _require(d, "put_delta",   "policy(daily_short_strangle)")
    call_delta  = _require(d, "call_delta",  "policy(daily_short_strangle)")
    quantity    = get(d, "quantity", 1.0)
    return DailyShortStrangle(
        Underlying(String(underlying)),
        _parse_entry_time(entry_raw),
        Day(Int(expiry_days)),
        Float64(put_delta),
        Float64(call_delta),
        Float64(quantity),
    )
end

const _POLICY_BUILDERS = Dict{String, Function}(
    "noop"                 => _build_noop_policy,
    "daily_short_strangle" => _build_daily_short_strangle,
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

[source.synthesizer]
type   = "ohlcv_spread"
lambda = 0.7

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

A real trading policy looks like:

```toml
[agent.policy]
type        = "daily_short_strangle"
underlying  = "SPY"
entry_time  = 15:45:00
expiry_days = 1
put_delta   = 0.20
call_delta  = 0.20
quantity    = 1.0    # optional, defaults to 1.0
```

Errors loudly on missing required keys or unknown `type` discriminators.
"""
function load_experiment(path::AbstractString)::Experiment
    return _experiment_from_cfg(TOML.parsefile(String(path)))
end

"""
    load_experiment_str(toml::AbstractString) -> Experiment

Parse `toml` content (not a path) and construct the [`Experiment`](@ref).
Same schema and validation as [`load_experiment`](@ref); useful for
rehydrating from a config string stored alongside a persisted run.
"""
function load_experiment_str(toml::AbstractString)::Experiment
    return _experiment_from_cfg(TOML.parse(String(toml)))
end

function _experiment_from_cfg(cfg::AbstractDict)::Experiment
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
