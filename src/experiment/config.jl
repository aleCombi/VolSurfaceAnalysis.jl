# Config-file loading: TOML -> Experiment.
#
# Stdlib TOML only. Each dispatched sum-type (data provider, Curve, Policy,
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

# ---- Kind names ---------------------------------------------------------
# The one string <-> type table. Kinds are keyed by type everywhere on the
# runtime path; only config and identity use these names.

const _KINDS = Dict{String,Type}(
    "option_bar"   => OptionBar,
    "option_quote" => OptionQuote,
    "spot_price"   => SpotPrice,
    "rate_curve"   => RateCurve,
    "div_curve"    => DivCurve,
    "vol_surface"  => VolatilitySurface,
)
const _KIND_NAMES = Dict{Type,String}(v => k for (k, v) in _KINDS)

"""
    kind_name(::Type) -> String

The config / identity name of a kind (`OptionQuote` -> `"option_quote"`).
"""
kind_name(T::Type) = get(_KIND_NAMES, T) do
    error("kind_name: no config name for kind $T (known: $(sort(collect(keys(_KINDS)))))")
end

# ---- Provider builders (`[data.<kind>]` tables) ------------------------
# One table per kind. The table name selects the kind; `type` selects the
# builder; the builder gets the rest of the table and the kind and returns
# a spec. Every builder has the signature `(d, R) -> spec`.

_selector_key(::Type{Underlying}) = "underlying"
_selector_key(::Type{Currency})   = "currency"
_parse_selector(::Type{Underlying}, s) = Underlying(String(s))
_parse_selector(::Type{Currency}, s)   = Currency(String(s))

# The selector of kind `R`, read from the key its type is named by.
function _selector_from(d::AbstractDict, ::Type{R}, where_::AbstractString) where {R}
    S = selector_type(R)
    key = _selector_key(S)
    return _parse_selector(S, _require(d, key, where_))
end

_build_parquet_option_bars(d::AbstractDict, ::Type) =
    ParquetOptionBars(String(_require(d, "root", "data(parquet_option_bars)")))

_build_parquet_spots(d::AbstractDict, ::Type) =
    ParquetSpots(String(_require(d, "root", "data(parquet_spots)")))

_build_from_bars(d::AbstractDict, ::Type) =
    QuotesFromBars(build_synthesizer(Dict{String,Any}(_require(d, "synthesizer", "data(from_bars)"))))

# `value` is a flat curve; `curve = { type = ... }` any curve builder.
function _curve_from(d::AbstractDict, where_::AbstractString)::Curve
    if haskey(d, "curve")
        return build_curve(Dict{String,Any}(d["curve"]))
    elseif haskey(d, "value")
        return FlatCurve(Float64(d["value"]))
    end
    error("load_experiment: $where_ needs \"value\" (flat) or a \"curve\" table")
end

_constant_record(::Type{RateCurve}, sel::Currency,  c::Curve) = RateCurve(sel, c)
_constant_record(::Type{DivCurve},  sel::Underlying, c::Curve) = DivCurve(sel, c)
_constant_record(::Type{R}, sel, ::Curve) where {R} = error(
    "load_experiment: data(constant) supports rate_curve and div_curve, not $(kind_name(R))")

function _build_constant(d::AbstractDict, ::Type{R}) where {R}
    sel = _selector_from(d, R, "data(constant)")
    return Constant(_constant_record(R, sel, _curve_from(d, "data(constant)")))
end

function _build_surface_from(d::AbstractDict, ::Type)
    currency = Currency(String(_require(d, "currency", "data(surface_from)")))
    spot_for = Dict{Underlying,Underlying}()
    if haskey(d, "spot_for")
        for (k, v) in d["spot_for"]
            spot_for[Underlying(String(k))] = Underlying(String(v))
        end
    end
    return SurfaceFrom(currency=currency, spot_for=spot_for)
end

# Every key other than `type` is `selector = { sub-table }`.
function _build_by_selector(d::AbstractDict, ::Type{R}) where {R}
    parts = Pair[]
    for (k, v) in d
        k == "type" && continue
        v isa AbstractDict || error(
            "load_experiment: data(by_selector) entry \"$k\" must be a table with a \"type\"")
        sel = _parse_selector(selector_type(R), k)
        push!(parts, sel => _build_provider(Dict{String,Any}(v), R, "data(by_selector).$k"))
    end
    isempty(parts) && error("load_experiment: data(by_selector) has no parts")
    sort!(parts; by = p -> string(first(p)))
    return BySelector{R}(parts...)
end

const _PROVIDER_BUILDERS = Dict{String, Function}(
    "parquet_option_bars" => _build_parquet_option_bars,
    "parquet_spots"       => _build_parquet_spots,
    "from_bars"           => _build_from_bars,
    "constant"            => _build_constant,
    "surface_from"        => _build_surface_from,
    "by_selector"         => _build_by_selector,
)

function _build_provider(d::AbstractDict, ::Type{R}, where_::AbstractString) where {R}
    t = _pop_type!(d, where_)
    spec = _dispatch(_PROVIDER_BUILDERS, t, "data provider")(d, R)
    kind(spec) === R || error(
        "load_experiment: $where_ has type \"$t\", which serves " *
        "$(kind_name(kind(spec))), not $(kind_name(R))")
    return spec
end

"""
    build_market_data(d::AbstractDict) -> MarketData

Build the provider map from a `[data]` table: one sub-table per kind,
keyed by kind name, each with a `type` discriminator. Load-time checks:
every table name is a known kind; the built spec serves that kind;
every derived spec's input kinds are present; every spec has a
lifecycle pair.
"""
function build_market_data(d::AbstractDict)::MarketData
    isempty(d) && error("load_experiment: [data] has no entries")
    specs = Any[]
    for name in sort!(collect(keys(d)))
        haskey(_KINDS, name) || error(
            "load_experiment: unknown data kind \"$name\". Known: $(sort(collect(keys(_KINDS))))")
        R = _KINDS[name]
        tbl = d[name]
        tbl isa AbstractDict || error("load_experiment: [data.$name] must be a table")
        push!(specs, _build_provider(Dict{String,Any}(tbl), R, "data.$name"))
    end
    m = MarketData(Tuple(specs))
    present = Set(kind(s) for s in m.entries)
    for s in m.entries, need in inputs(s)
        need in present || error(
            "load_experiment: data.$(kind_name(kind(s))) needs $(kind_name(need)), " *
            "which no [data.*] table provides")
        has_lifecycle(s) || error(
            "load_experiment: data.$(kind_name(kind(s))) has no open_data method")
    end
    for s in m.entries
        has_lifecycle(s) || error(
            "load_experiment: data.$(kind_name(kind(s))) has no open_data method")
    end
    return m
end

"""
    build_clock(d::AbstractDict) -> Clock

Build the tick grid from a `clock = { kind = "...", <selector> = "..." }`
table; the selector key is `underlying` or `currency` per the kind.
"""
function build_clock(d::AbstractDict)::Clock
    name = String(_require(d, "kind", "clock"))
    haskey(_KINDS, name) || error(
        "load_experiment: unknown clock kind \"$name\". Known: $(sort(collect(keys(_KINDS))))")
    R = _KINDS[name]
    return Clock{R}(_selector_from(d, R, "clock"))
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

# ---- OutputSpec builder -------------------------------------------------

# A flat TOML table -> NamedTuple (keys -> symbols). Values pass through as
# parsed (Int / Float / String); metric functions accept `Real` kwargs.
function _table_to_namedtuple(d::AbstractDict)::NamedTuple
    isempty(d) && return NamedTuple()
    ks = collect(keys(d))
    return NamedTuple{Tuple(Symbol(k) for k in ks)}(Tuple(d[k] for k in ks))
end

"""
    build_output_spec(d::AbstractDict) -> OutputSpec

Build an [`OutputSpec`](@ref) from an `[outputs]` config table. Each field
is optional; an omitted field takes the `OutputSpec` default (all metrics
/ default artifacts). `[outputs.metric_params.<metric>]` sub-tables become
per-metric kwarg overrides.
"""
function build_output_spec(d::AbstractDict)::OutputSpec
    metrics = haskey(d, "metrics") ?
        Symbol[Symbol(m) for m in d["metrics"]] : _default_metrics()
    artifacts = haskey(d, "artifacts") ?
        Symbol[Symbol(a) for a in d["artifacts"]] : _default_artifacts()
    metric_params = Dict{Symbol,NamedTuple}()
    if haskey(d, "metric_params")
        for (mname, params) in d["metric_params"]
            metric_params[Symbol(mname)] = _table_to_namedtuple(Dict{String,Any}(params))
        end
    end
    return OutputSpec(; metrics=metrics, metric_params=metric_params, artifacts=artifacts)
end

# ---- Top-level loader ---------------------------------------------------

"""
    load_experiment(path::AbstractString) -> Experiment

Parse a TOML file and construct the [`Experiment`](@ref) it describes.

The schema is a flat header (`name`, `from`, `to`, `clock`) plus nested
tables (`[outputs]`, `[data.<kind>]`, `[agent]`). Every dispatched
sum-type (data provider, synthesizer, curve, policy, agent) is keyed by
a `type` discriminator; the rest of that table is forwarded to the
matching builder. Optional metrics live under `[outputs]`; top-level
`metrics` and the old `[source]` table are rejected with a pointer.

# Example

```toml
name  = "noop_smoke"
from  = 2024-01-15T15:30:00
to    = 2024-01-15T15:32:00
clock = { kind = "option_quote", underlying = "SPY" }

[outputs]
metrics = ["sharpe", "max_drawdown"]

[data.option_bar]
type = "parquet_option_bars"
root = "C:/data/polygon/options_1min"

[data.option_quote]
type = "from_bars"
synthesizer = { type = "ohlcv_spread", lambda = 0.7 }

[data.spot_price]
type = "parquet_spots"
root = "C:/data/polygon/spots_1min"

[data.rate_curve]
type = "constant"
currency = "USD"
value = 0.04

[data.div_curve]
type = "constant"
underlying = "SPY"
value = 0.015

[data.vol_surface]
type = "surface_from"
currency = "USD"

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
    haskey(cfg, "metrics") && error(
        "load_experiment: top-level \"metrics\" is no longer supported; " *
        "move it under [outputs] as metrics = [...]")
    haskey(cfg, "source") && error(
        "load_experiment: the [source] table is no longer supported; " *
        "use [data.<kind>] tables and a top-level clock (see docs/modules/experiment.md)")
    data_tbl  = _require(cfg, "data",  "config")
    clock_tbl = _require(cfg, "clock", "config")
    agent_tbl = _require(cfg, "agent", "config")
    data  = build_market_data(Dict{String,Any}(data_tbl))
    clock = build_clock(Dict{String,Any}(clock_tbl))
    any(kind(s) === kind(clock) for s in data.entries) || error(
        "load_experiment: clock kind \"$(kind_name(kind(clock)))\" has no [data.*] table")
    agent = build_agent(Dict{String,Any}(agent_tbl))
    outputs = haskey(cfg, "outputs") ?
        build_output_spec(Dict{String,Any}(cfg["outputs"])) : OutputSpec()
    return Experiment(; name=name, agent=agent, data=data, clock=clock,
                       from=from, to=to, outputs=outputs)
end
