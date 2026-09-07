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

at(m::MarketData, ::Type{R}, sel, ts::DateTime) where {R} =
    at(entry(m, R), m, R, sel, ts)
between(m::MarketData, ::Type{R}, sel, from::DateTime, to::DateTime) where {R} =
    between(entry(m, R), m, R, sel, from, to)
asof(m::MarketData, ::Type{R}, sel, ts::DateTime) where {R} =
    asof(entry(m, R), m, R, sel, ts)
timestamps(m::MarketData, ::Type{R}, sel, from::DateTime, to::DateTime) where {R} =
    timestamps(entry(m, R), m, R, sel, from, to)
