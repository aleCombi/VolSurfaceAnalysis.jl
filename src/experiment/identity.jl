# Canonical, layered, content-addressed identity for experiments.
#
# An experiment's identity is computed from its *resolved* form (defaults
# filled in), not from raw config bytes -- so whitespace, key order, the
# human `name`, omitted-vs-explicit defaults, and machine cache knobs do
# not change identity. Two hashes are produced:
#
#   core_hash -- everything that determines the backtest result
#                (the ledger): data, clock, agent, window,
#                the venue's two choices, and the contract facts resolved
#                for the experiment's underlying.
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

# Both parquet readers map a vendor minute bar to a record visible at bar
# END (`bar_visible_at`, data/polygon.jl). That convention determines every
# record they serve -- which minute a decision reads, and therefore every
# fill and every settlement price -- so it belongs in the projection. It is
# a fixed constant, not a field: there is no setting to vary and none may
# be added, and it is not an output, so it goes here rather than in
# `OutputSpec`. Its job in the hash is to fork every id away from the runs
# made under bar-open visibility, which the corrected code cannot
# reproduce.
const _BAR_STAMP = "bar_end"

to_dict(s::ParquetOptionBars) = Dict{String,Any}(
    "type" => "parquet_option_bars", "stamp" => _BAR_STAMP,
    "dataset" => Dict{String,Any}("root" => s.root))
to_dict(s::ParquetSpots) = Dict{String,Any}(
    "type" => "parquet_spots", "stamp" => _BAR_STAMP,
    "dataset" => Dict{String,Any}("root" => s.root))
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

# --- contract facts -----------------------------------------------------
# `_CONTRACT_TABLE`'s entries reach cash through `contract_spec`, so a
# correction there must be a new run id rather than a silent change to old
# results. The enums project as their own names: the projection is read by
# people comparing two manifests, and `Int` codes would renumber whenever
# a variant is inserted.

to_dict(e::ExerciseStyle)   = string(e)
to_dict(e::SettlementStyle) = string(e)
to_dict(e::Delivery)        = string(e)

to_dict(c::ContractSpec) = Dict{String,Any}(
    "multiplier" => c.multiplier,
    "exercise"   => to_dict(c.exercise),
    "settlement" => to_dict(c.settlement),
    "delivery"   => to_dict(c.delivery),
)

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
#
# `contract` is the resolved spec for the experiment's one underlying, not
# the whole table: projecting the table would fork every id on an entry the
# run never touches. One experiment is one underlying, which is what
# `_experiment_underlying` stands on -- and it is what reports a clock that
# names something else, in the runner's words, rather than letting
# `contract_spec` fail on the selector type.
function _core_dict(exp::Experiment)
    return Dict{String,Any}(
        "from"     => string(exp.from),
        "to"       => string(exp.to),
        "data"     => to_dict(exp.data),
        "clock"    => to_dict(exp.clock),
        "agent"    => to_dict(exp.agent),
        "venue"    => Dict{String,Any}("fill_rule"  => String(exp.fill_rule),
                                       "cost_model" => String(exp.cost_model)),
        "contract" => to_dict(contract_spec(_experiment_underlying(exp))),
    )
end

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
agent, window, the venue's fill rule and cost model, and the contract
facts of the experiment's underlying). Identical across experiments that differ only in outputs
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
