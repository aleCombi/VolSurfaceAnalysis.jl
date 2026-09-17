# `SurfaceFrom`: the derived provider that serves `VolatilitySurface`, built
# from an option chain, a spot and the rate and dividend curves, all read back
# through the map the call arrived through.

"""
    SurfaceFrom(; currency, spot_for = Dict(), lookback_ticks = 3)

Derived provider spec for `VolatilitySurface`. `currency` selects the
`RateCurve`; `spot_for` remaps the underlying whose `SpotPrice` is used
(e.g. `SPY => SPX`), defaulting to the surface's own underlying. Holds
no inputs: they come from the map (`inputs` lists their kinds).

`lookback_ticks` bounds `asof`: the number of input timestamps it will
*examine*, so `1` tries only the newest quote timestamp. Rejected at
construction below `1`.
"""
struct SurfaceFrom
    spot_for::Dict{Underlying,Underlying}
    currency::Currency
    lookback_ticks::Int
    function SurfaceFrom(spot_for::Dict{Underlying,Underlying}, currency::Currency,
                         lookback_ticks::Int)
        lookback_ticks >= 1 || throw(ArgumentError(
            "SurfaceFrom: lookback_ticks must be >= 1, got $lookback_ticks"))
        new(spot_for, currency, lookback_ticks)
    end
end
SurfaceFrom(; currency::Currency, spot_for=Dict{Underlying,Underlying}(),
              lookback_ticks::Int=3) =
    SurfaceFrom(Dict{Underlying,Underlying}(spot_for), currency, lookback_ticks)

# Value semantics: the Dict field would otherwise make == an identity test.
# Every field must appear here -- these are hand-written, so a new one is
# silently dropped from identity otherwise.
Base.:(==)(a::SurfaceFrom, b::SurfaceFrom) =
    a.currency == b.currency && a.spot_for == b.spot_for &&
    a.lookback_ticks == b.lookback_ticks
Base.hash(s::SurfaceFrom, h::UInt) =
    hash(s.lookback_ticks, hash(s.spot_for, hash(s.currency, hash(:SurfaceFrom, h))))

kind(::SurfaceFrom) = VolatilitySurface
inputs(::SurfaceFrom) = (OptionQuote, SpotPrice, RateCurve, DivCurve)

# The two selectors a SurfaceFrom names in its own configuration, for the
# load-time fast path. The chain and the dividend curve follow the surface's
# own underlying, which is a query argument, so they cannot be checked here.
demands(s::SurfaceFrom) = ((RateCurve, s.currency),
                           ((SpotPrice, v) for v in values(s.spot_for))...)

struct SurfaceReader
    spec::SurfaceFrom
    # No cutoff in the key -- the `data` module doc says what must stay true
    # of the derivation for that to be safe.
    cache::LRU{Tuple{Underlying,DateTime},Vector{VolatilitySurface}}
end

kind(::SurfaceReader) = VolatilitySurface
inputs(r::SurfaceReader) = inputs(r.spec)

# Derived: spec and reader both delegate, so an unserved input surfaces
# as that input's own error.
serves(::Union{SurfaceFrom,SurfaceReader}, ::Any, ::Type{VolatilitySurface}, ::Any) = missing

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

# Walks back from the newest chain instant, because a chain can exist
# where a surface does not (same-day contracts at the expiry instant, a
# minute of unusable marks, a missing spot). Each benign case is one tick
# wide, which is what makes a small bound honest.
function asof(r::SurfaceReader, m, ::Type{VolatilitySurface}, u::Underlying, ts::DateTime)
    cursor = ts
    oldest = ts
    for _ in 1:r.spec.lookback_ticks
        q = asof(m, OptionQuote, u, cursor)
        isempty(q) && return VolatilitySurface[]
        win = first(q).timestamp
        s = at(r, m, VolatilitySurface, u, win)
        isempty(s) || return s
        oldest = win                       # the instant tried, not the cursor after it
        cursor = win - Millisecond(1)
    end
    throw(DerivationExhausted(VolatilitySurface, u, ts, oldest, r.spec.lookback_ticks))
end

# The input grid, an over-estimate by design (the `data` module doc).
timestamps(::SurfaceReader, m, ::Type{VolatilitySurface}, u::Underlying, from::DateTime, to::DateTime) =
    timestamps(m, OptionQuote, u, from, to)
