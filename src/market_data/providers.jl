# `market_data` module: resource-free provider specs and the first derived
# provider.
#
# A spec is an immutable value describing where records of one kind come
# from. It holds no resources: config builds it, identity hashes it. The
# specs here need nothing at run time, so they are their own readers.

"""
    InMemory{R}(rows)

Fixture provider: every record of kind `R` in `rows`, kept sorted by
`timestamp` (stable, so input order is preserved within one instant).
Serves every selector present in `rows`.
"""
struct InMemory{R}
    rows::Vector{R}
    ts::Vector{DateTime}                  # rows[i].timestamp, for searchsorted
    function InMemory{R}(rows) where {R}
        sorted = sort(collect(R, rows); by = r -> r.timestamp)   # default sort is stable
        new{R}(sorted, DateTime[r.timestamp for r in sorted])
    end
end
InMemory(rows::AbstractVector{R}) where {R} = InMemory{R}(rows)

kind(::InMemory{R}) where {R} = R

function between(p::InMemory{R}, ::Any, ::Type{R}, sel, from::DateTime, to::DateTime) where {R}
    lo = searchsortedfirst(p.ts, from)
    hi = searchsortedlast(p.ts, to)
    R[r for r in view(p.rows, lo:hi) if selector(r) == sel]
end

function asof(p::InMemory{R}, ::Any, ::Type{R}, sel, ts::DateTime) where {R}
    i = searchsortedlast(p.ts, ts)
    j = findlast(r -> selector(r) == sel, view(p.rows, 1:i))
    j === nothing && return R[]
    win = p.ts[j]
    R[r for r in view(p.rows, searchsorted(p.ts, win)) if selector(r) == sel]
end

timestamps(p::InMemory{R}, ctx, ::Type{R}, sel, from::DateTime, to::DateTime) where {R} =
    unique!(DateTime[r.timestamp for r in between(p, ctx, R, sel, from, to)])

"""
    Constant{R}(record)

One record, visible from the start of time: `asof` returns it for its
own selector (`selector(record) == sel`) and nothing for any other;
`between` and `timestamps` contain it only when the selector matches
and its timestamp lies in the range, so over any real window they are
empty. The natural spec for a flat rate or dividend curve.
"""
struct Constant{R}
    record::R
end

kind(::Constant{R}) where {R} = R

asof(c::Constant{R}, ::Any, ::Type{R}, sel, ::DateTime) where {R} =
    selector(c.record) == sel ? R[c.record] : R[]

between(c::Constant{R}, ::Any, ::Type{R}, sel, from::DateTime, to::DateTime) where {R} =
    (selector(c.record) == sel && from <= c.record.timestamp <= to) ? R[c.record] : R[]

timestamps(c::Constant{R}, ctx, ::Type{R}, sel, from::DateTime, to::DateTime) where {R} =
    DateTime[r.timestamp for r in between(c, ctx, R, sel, from, to)]

"""
    inputs(spec) -> Tuple of kinds

The kinds a derived provider reads through the map. `()` for raw
providers. The config loader checks every input kind is present.
"""
inputs(::Any) = ()

"""
    QuotesFromBars(synthesizer)

Derived provider: serves `OptionQuote` by reading `OptionBar` through
the map it is called from and projecting each bar through the
`QuoteSynthesizer`. Holds only the synthesizer, never its input, so the
one `OptionBar` entry (and its reader) is shared by every consumer, and
a time cut handed as the context bounds what it can see.
"""
struct QuotesFromBars{Q<:QuoteSynthesizer}
    synthesizer::Q
end

kind(::QuotesFromBars) = OptionQuote
inputs(::QuotesFromBars) = (OptionBar,)

at(p::QuotesFromBars, m, ::Type{OptionQuote}, u, ts::DateTime) =
    OptionQuote[synthesize(p.synthesizer, b) for b in at(m, OptionBar, u, ts)]
between(p::QuotesFromBars, m, ::Type{OptionQuote}, u, from::DateTime, to::DateTime) =
    Iterators.map(b -> synthesize(p.synthesizer, b), between(m, OptionBar, u, from, to))
asof(p::QuotesFromBars, m, ::Type{OptionQuote}, u, ts::DateTime) =
    OptionQuote[synthesize(p.synthesizer, b) for b in asof(m, OptionBar, u, ts)]
timestamps(::QuotesFromBars, m, ::Type{OptionQuote}, u, from::DateTime, to::DateTime) =
    timestamps(m, OptionBar, u, from, to)
