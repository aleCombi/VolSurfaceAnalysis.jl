# `surfaces` module: the derived provider that serves `VolatilitySurface`.
#
# `SurfaceFrom` reads its inputs (quotes, spot, rate and div curves) through
# the map it is called from, builds the surface with `build_surface`, and
# caches the result per (underlying, timestamp) in a bounded LRU. The cache
# is cut-independent by the invariant that every input read is at or before
# the requested `ts`, so an entry keyed on (sel, ts) is valid under any
# cutoff >= ts: the same surface object comes back through the bare map and
# through any cut at or after its timestamp.

selector(s::VolatilitySurface) = s.underlying
selector_type(::Type{<:VolatilitySurface}) = Underlying

"""
    SurfaceFrom(; currency, spot_for = Dict())

Derived provider spec for `VolatilitySurface`. `currency` selects the
`RateCurve`; `spot_for` remaps the underlying whose `SpotPrice` is used
(e.g. `SPY => SPX`), defaulting to the surface's own underlying. Holds
no inputs: they come from the map (`inputs` lists their kinds).
"""
struct SurfaceFrom
    spot_for::Dict{Underlying,Underlying}
    currency::Currency
end
SurfaceFrom(; currency::Currency, spot_for=Dict{Underlying,Underlying}()) =
    SurfaceFrom(Dict{Underlying,Underlying}(spot_for), currency)

# Value semantics: the Dict field would otherwise make == an identity test.
Base.:(==)(a::SurfaceFrom, b::SurfaceFrom) = a.currency == b.currency && a.spot_for == b.spot_for
Base.hash(s::SurfaceFrom, h::UInt) = hash(s.spot_for, hash(s.currency, hash(:SurfaceFrom, h)))

kind(::SurfaceFrom) = VolatilitySurface
inputs(::SurfaceFrom) = (OptionQuote, SpotPrice, RateCurve, DivCurve)

struct SurfaceReader
    spec::SurfaceFrom
    cache::LRU{Tuple{Underlying,DateTime},Vector{VolatilitySurface}}
end

kind(::SurfaceReader) = VolatilitySurface
inputs(r::SurfaceReader) = inputs(r.spec)

"""
    open_data(s::SurfaceFrom; max_surfaces=64)

The reader with a bounded surface cache; nothing to close.
"""
open_data(s::SurfaceFrom; max_surfaces::Int=64) =
    SurfaceReader(s, LRU{Tuple{Underlying,DateTime},Vector{VolatilitySurface}}(max_surfaces))
close_data!(::SurfaceReader) = nothing

# Empty when any input is absent or when no expiry survives the build;
# the empty result is cached too, so absence is not retried.
function at(r::SurfaceReader, m, ::Type{VolatilitySurface}, u::Underlying, ts::DateTime)
    get!(r.cache, (u, ts)) do
        chain = at(m, OptionQuote, u, ts)
        isempty(chain) && return VolatilitySurface[]
        spot = only_or_missing(at(m, SpotPrice, get(r.spec.spot_for, u, u), ts))
        ismissing(spot) && return VolatilitySurface[]
        rate = only_or_missing(asof(m, RateCurve, r.spec.currency, ts))
        div  = only_or_missing(asof(m, DivCurve, u, ts))
        (ismissing(rate) || ismissing(div)) && return VolatilitySurface[]
        s = build_surface(chain, spot.price, rate.curve(ts), div.curve(ts))
        s === nothing ? VolatilitySurface[] : VolatilitySurface[s]
    end
end

between(r::SurfaceReader, m, ::Type{VolatilitySurface}, u::Underlying, from::DateTime, to::DateTime) =
    Iterators.flatten(at(r, m, VolatilitySurface, u, ts) for ts in timestamps(m, OptionQuote, u, from, to))

function asof(r::SurfaceReader, m, ::Type{VolatilitySurface}, u::Underlying, ts::DateTime)
    q = asof(m, OptionQuote, u, ts)
    isempty(q) ? VolatilitySurface[] : at(r, m, VolatilitySurface, u, first(q).timestamp)
end

timestamps(::SurfaceReader, m, ::Type{VolatilitySurface}, u::Underlying, from::DateTime, to::DateTime) =
    timestamps(m, OptionQuote, u, from, to)
