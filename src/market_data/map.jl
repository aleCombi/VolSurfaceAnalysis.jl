# `market_data` module: the map.
#
# `MarketData` is an immutable tuple of providers, one per kind, looked up
# by type. It is what an `Experiment` stores (specs) and what a run reads
# through (readers, after `open_data`). Type lookup over a concrete tuple
# folds at compile time, so `entry(m, R)` costs nothing on the hot path.

"""
    MarketData(providers...)

One provider per kind. Construction rejects an empty tuple and two
providers of the same kind. Every map-level shape looks the provider up
by kind and passes the map itself down as the context, so raw providers
ignore it, derived providers read their inputs through it, and
composition forwards it.
"""
struct MarketData{P<:Tuple}
    entries::P
    function MarketData(entries::Tuple)
        isempty(entries) && throw(ArgumentError("MarketData: no providers"))
        ks = map(kind, entries)
        allunique(ks) || throw(ArgumentError(
            "MarketData: duplicate kind $(first(k for k in ks if count(==(k), ks) > 1))"))
        new{typeof(entries)}(entries)
    end
end
MarketData(providers...) = MarketData(providers)

"""
    entry(m, ::Type{R}) -> provider

The provider serving kind `R` in `m`; errors when `m` has none.
"""
entry(m::MarketData, ::Type{R}) where {R} = _entry(R, m.entries...)

@inline _entry(::Type{R}, p, rest...) where {R} = kind(p) === R ? p : _entry(R, rest...)
_entry(::Type{R}) where {R} = error("MarketData has no provider for $R")

serves(m::MarketData, ::Type{R}, sel) where {R} = serves(entry(m, R), m, R, sel)

# Structural absence throws, and the check lives here rather than inside
# each provider's four shapes: that would be sixteen call sites, and it
# would also fire on the provider-level delegation `BySelector` and
# `QuotesFromBars` already do. Two consequences, both deliberate:
# provider-level calls (`at(p, ctx, R, sel, ts)`) are unchecked, which is
# the arity tests and internal delegation use; and a provider with no
# `serves` method (`missing`) opts out.
@inline function _require_served(m, ::Type{R}, sel) where {R}
    serves(m, R, sel) === false &&
        throw(UnservedSelector(R, sel, served_description(entry(m, R))))
    return nothing
end

function at(m::MarketData, ::Type{R}, sel, ts::DateTime) where {R}
    _require_served(m, R, sel)
    at(entry(m, R), m, R, sel, ts)
end
function between(m::MarketData, ::Type{R}, sel, from::DateTime, to::DateTime) where {R}
    _require_served(m, R, sel)
    between(entry(m, R), m, R, sel, from, to)
end
function asof(m::MarketData, ::Type{R}, sel, ts::DateTime) where {R}
    _require_served(m, R, sel)
    asof(entry(m, R), m, R, sel, ts)
end
function timestamps(m::MarketData, ::Type{R}, sel, from::DateTime, to::DateTime) where {R}
    _require_served(m, R, sel)
    timestamps(entry(m, R), m, R, sel, from, to)
end
