# Canonical, layered, content-addressed identity for experiments.
#
# An experiment's identity is computed from its *resolved* form (defaults
# filled in), not from raw config bytes -- so whitespace, key order, the
# human `name`, omitted-vs-explicit defaults, and machine cache knobs do
# not change identity. Two hashes are produced:
#
#   core_hash -- everything that determines the backtest result
#                (positions / pnl_series): data, clock, agent, window.
#   full_hash -- core plus outputs (metrics + params, artifacts).
#
# Same core_hash, different full_hash => an output/artifact variation of a
# backtest already run. `name` is in neither hash (label only).
#
# `to_dict` is an identity *projection*, not a faithful serializer: it
# emits only result-relevant fields, collapsing primitives to strings so
# the canonicaliser stays tiny. It is NOT used to rebuild experiments
# (load_run rebuilds from the verbatim config.toml); it exists only to be
# hashed.

using SHA

const _IDENTITY_HEX_LEN = 16

# --- canonical stringifier ----------------------------------------------
# Deterministic string for a value built from String / Real / Bool /
# Vector / Dict{String}. Dict keys are sorted; numbers normalise through
# Float64 so 1 and 1.0 collapse. This ordering is the only guarantee the
# hash relies on.
_canonical(x::Bool) = x ? "true" : "false"
_canonical(x::Real) = string(Float64(x))
function _canonical(x::AbstractString)
    s = replace(String(x), "\\" => "\\\\", "\"" => "\\\"")
    return "\"" * s * "\""
end
_canonical(x::AbstractVector) = "[" * join((_canonical(v) for v in x), ",") * "]"
function _canonical(d::AbstractDict)
    ks = sort!(collect(keys(d)))
    return "{" * join(("\"$(k)\":" * _canonical(d[k]) for k in ks), ",") * "}"
end

# --- to_dict: identity projection per sum type --------------------------
# Each method mirrors the matching `build_*` in config.jl, emitting the
# fields that define the experiment and omitting everything that does not
# affect its result (cache sizes, DuckDB handles, surface caches).

to_dict(c::FlatCurve) = Dict{String,Any}("type" => "flat", "value" => c.value)
to_dict(c::PCCurve) = Dict{String,Any}(
    "type"   => "pc",
    "knots"  => String[string(k) for k in c.knots],
    "values" => collect(Float64, c.values),
)

to_dict(s::SpreadFromOHLCV) =
    Dict{String,Any}("type" => "ohlcv_spread", "lambda" => s.lambda)

# --- market_data specs --------------------------------------------------
# One entry per kind, keyed by the loader's kind name; each spec emits
# only what determines the records it serves. Readers never appear (they
# are not on specs) and cache sizes are open_data kwargs, never identity.
# The `dataset` slot on the parquet specs is reserved for a logical
# dataset id / version; today it carries the root path.

to_dict(s::ParquetOptionBars) = Dict{String,Any}(
    "type" => "parquet_option_bars", "dataset" => Dict{String,Any}("root" => s.root))
to_dict(s::ParquetSpots) = Dict{String,Any}(
    "type" => "parquet_spots", "dataset" => Dict{String,Any}("root" => s.root))
to_dict(s::QuotesFromBars) = Dict{String,Any}(
    "type" => "from_bars", "synthesizer" => to_dict(s.synthesizer))

_constant_payload(r::RateCurve) = Dict{String,Any}("curve" => to_dict(r.curve))
_constant_payload(r::DivCurve)  = Dict{String,Any}("curve" => to_dict(r.curve))
_constant_payload(r) = error(
    "Constant{$(typeof(r))} has no identity projection; only curve kinds are config-buildable")

# The visibility stamp only appears when it is not the "always known"
# default, so the common flat-curve case stays minimal.
function to_dict(c::Constant)
    d = _constant_payload(c.record)
    d["type"] = "constant"
    d["selector"] = string(selector(c.record))
    c.record.timestamp == typemin(DateTime) || (d["timestamp"] = string(c.record.timestamp))
    return d
end

# spot_for as a sorted vector of [from, to] pairs, so map order never
# forks the hash.
to_dict(s::SurfaceFrom) = Dict{String,Any}(
    "type"           => "surface_from",
    "currency"       => s.currency.code,
    "spot_for"       => sort!([String[string(k), string(v)] for (k, v) in s.spot_for]),
    "lookback_ticks" => s.lookback_ticks,
)

# Parts sorted by selector string for the same reason.
function to_dict(b::BySelector)
    parts = [Dict{String,Any}("selector" => string(first(p)), "provider" => to_dict(last(p)))
             for p in b.parts]
    sort!(parts; by = p -> p["selector"])
    return Dict{String,Any}("type" => "by_selector", "parts" => parts)
end

# InMemory is a dev/test provider (not config-buildable) and is not
# identity-stable: a faithful projection would have to digest every
# record, and a shape-only projection would let fixtures that differ in
# prices collide. Rather than risk an unsafe hash, reject it -- build
# experiments from a config to hash or save them.
to_dict(::InMemory) = error(
    "InMemory is not identity-stable and cannot be hashed or saved; " *
    "build the experiment from a config to use the knowledge base.")

to_dict(m::MarketData) = Dict{String,Any}(
    "entries" => Dict{String,Any}(kind_name(kind(s)) => to_dict(s) for s in m.entries))

to_dict(c::Clock) = Dict{String,Any}(
    "kind" => kind_name(kind(c)), "selector" => string(c.sel))

to_dict(::NoOpPolicy) = Dict{String,Any}("type" => "noop")

# `expiry_interval` is any `Period`; stringify it so the unit is part of
# identity -- `Day(1)` and `Hour(1)` must not collide (they pick different
# expiries), and `Dates.value` alone would lose the unit.
to_dict(p::DailyShortStrangle) = Dict{String,Any}(
    "type"            => "daily_short_strangle",
    "underlying"      => ticker(p.underlying),
    "entry_time"      => string(p.entry_time),
    "expiry_interval" => string(p.expiry_interval),
    "put_delta"       => p.put_delta,
    "call_delta"      => p.call_delta,
    "quantity"        => p.quantity,
)

to_dict(a::StaticAgent) = Dict{String,Any}("type" => "static", "policy" => to_dict(a.policy))

function to_dict(o::OutputSpec)
    mp = Dict{String,Any}()
    for (k, v) in o.metric_params
        mp[string(k)] = Dict{String,Any}(string(pk) => pv for (pk, pv) in pairs(v))
    end
    return Dict{String,Any}(
        "metrics"       => sort!(String[string(m) for m in o.metrics]),
        "metric_params" => mp,
        "artifacts"     => sort!(String[string(a) for a in o.artifacts]),
    )
end

# --- experiment-level identity ------------------------------------------

# Core identity: everything that determines the backtest result.
_core_dict(exp::Experiment) = Dict{String,Any}(
    "from"  => string(exp.from),
    "to"    => string(exp.to),
    "data"  => to_dict(exp.data),
    "clock" => to_dict(exp.clock),
    "agent" => to_dict(exp.agent),
)

# Full identity: core plus outputs. `name` is excluded from both -- it is
# a human label, not part of what the experiment is.
function _full_dict(exp::Experiment)
    d = _core_dict(exp)
    d["outputs"] = to_dict(exp.outputs)
    return d
end

_hash16(s::AbstractString) = bytes2hex(sha2_256(codeunits(s)))[1:_IDENTITY_HEX_LEN]

"""
    core_hash(exp::Experiment) -> String

16-hex content hash of the backtest-determining inputs (data, clock,
agent, window). Identical across experiments that differ only in outputs
(metrics / artifacts) or in the human `name`. This is the key for
recognising that two experiments share a backtest.
"""
core_hash(exp::Experiment)::String = _hash16(_canonical(_core_dict(exp)))

"""
    full_hash(exp::Experiment) -> String

16-hex content hash of the complete experiment: core inputs plus the
output spec. This is the run's identity in the knowledge base.
"""
full_hash(exp::Experiment)::String = _hash16(_canonical(_full_dict(exp)))
