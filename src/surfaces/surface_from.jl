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
snapshot(::Type{<:VolatilitySurface}) = true

"""
    SurfaceFrom(; currency, spot_for = Dict(), lookback_ticks = 3)

Derived provider spec for `VolatilitySurface`. `currency` selects the
`RateCurve`; `spot_for` remaps the underlying whose `SpotPrice` is used
(e.g. `SPY => SPX`), defaulting to the surface's own underlying. Holds
no inputs: they come from the map (`inputs` lists their kinds).

`lookback_ticks` bounds `asof`: the number of input timestamps it will
*examine*, not the number of steps it takes, so `1` tries only the newest
quote timestamp. It changes which surface a policy sees, so it changes
results, so it is part of identity and of the config surface. Rejected at
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
    cache::LRU{Tuple{Underlying,DateTime},Vector{VolatilitySurface}}
end

kind(::SurfaceReader) = VolatilitySurface
inputs(r::SurfaceReader) = inputs(r.spec)

# Derived: spec and reader both delegate rather than answering, so an
# unserved input surfaces as that input's own error and names the real
# cause -- a SurfaceFrom asked for SPX reports OptionBar/SPX unserved,
# not "no surface".
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

# `asof` must return the newest instant at which a SURFACE exists, not the
# newest at which a chain does. The two differ exactly when derivation
# fails: a chain of same-day contracts evaluated at the expiry instant, a
# minute of unusable marks, a missing spot. Each benign case is one tick
# wide, so the walk is bounded by `lookback_ticks` and exhausting it
# throws: many consecutive failures mean a truncated dataset or a broken
# feed, and reporting that as absence is the mistake findings 2 and 3
# exist to correct. Three outcomes, one per state: no chain at all is
# empty (temporal), a chain that builds is the surface, and chains that
# never build within the bound throw.
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

# An over-estimate for a derived kind, deliberately: making it exact would
# mean building every surface in the range. `timestamps` and `between`
# report the input grid, so they can name instants where no surface
# exists. That is a property of derived kinds, not a defect of this one.
timestamps(::SurfaceReader, m, ::Type{VolatilitySurface}, u::Underlying, from::DateTime, to::DateTime) =
    timestamps(m, OptionQuote, u, from, to)
