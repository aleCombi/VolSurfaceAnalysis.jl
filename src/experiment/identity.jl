# Canonical, layered, content-addressed identity for experiments.
#
# An experiment's identity is computed from its *resolved* form (defaults
# filled in), not from raw config bytes -- so whitespace, key order, the
# human `name`, omitted-vs-explicit defaults, and machine cache knobs do
# not change identity. Two hashes are produced:
#
#   core_hash -- everything that determines the backtest result
#                (positions / pnl_series): source, agent, window.
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

# ParquetDataSource: identity-relevant fields only. Cache knobs
# (`max_*_cached`), the DuckDB handle, and the caches are omitted -- they
# do not affect the backtest result. `options_root`/`spot_root` are the
# resolved paths, so the `root` shorthand and the explicit form collapse
# to the same identity.
to_dict(ds::ParquetDataSource) = Dict{String,Any}(
    "type"         => "parquet",
    "underlying"   => ticker(ds.underlying),
    "options_root" => ds.options_root,
    "spot_root"    => ds.spot_root,
    "synthesizer"  => to_dict(ds.synthesizer),
)

# InMemoryDataSource is a dev/test source (not config-buildable) and is
# not identity-stable: a faithful projection would have to digest every
# option quote, and a shape-only projection would let chains that differ
# in prices collide. Rather than risk an unsafe hash, reject it -- build
# experiments from a config (ParquetDataSource) to hash or save them.
to_dict(::InMemoryDataSource) = error(
    "InMemoryDataSource is not identity-stable and cannot be hashed or " *
    "saved; build the experiment from a config to use the knowledge base.")

# ModelDataSource: the chain source plus rate/div curves. `spot_source`
# is only distinct from `chain_source` in split-vendor setups the config
# loader cannot express; when they differ it is recorded so identity
# stays faithful.
function to_dict(m::ModelDataSource)
    d = to_dict(m.chain_source)
    d["rate"] = to_dict(m.rate)
    d["div"]  = to_dict(m.div)
    if m.spot_source !== m.chain_source
        d["spot_source"] = to_dict(m.spot_source)
    end
    return d
end

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
    "from"   => string(exp.from),
    "to"     => string(exp.to),
    "source" => to_dict(exp.source),
    "agent"  => to_dict(exp.agent),
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

16-hex content hash of the backtest-determining inputs (source, agent,
window). Identical across experiments that differ only in outputs
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
