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
Serves every selector present in `rows`. For a `snapshot` kind the rows
obey the same rule as the parquet spot reader: two rows for one selector
at one instant collapse when equal and throw `ConflictingRecords` when
they differ, so a fixture cannot hold a state the real reader aborts on.
"""
struct InMemory{R}
    rows::Vector{R}
    ts::Vector{DateTime}                  # rows[i].timestamp, for searchsorted
    sels::Set{Any}                        # the selectors present, for `serves`
    function InMemory{R}(rows) where {R}
        sorted = sort(collect(R, rows); by = r -> r.timestamp)   # default sort is stable
        snapshot(R) && _collapse_snapshot!(sorted, R)
        new{R}(sorted, DateTime[r.timestamp for r in sorted],
               Set{Any}(selector(r) for r in sorted))
    end
end

# One record per selector per instant, on rows already sorted by
# timestamp: an exact duplicate is dropped, a disagreement throws naming
# both records. Mirrors `_collapse_duplicates!` in the parquet spot
# reader, generic over the kind via `==` on the whole record.
function _collapse_snapshot!(sorted::Vector{R}, ::Type{R}) where {R}
    length(sorted) < 2 && return sorted
    keep = trues(length(sorted))
    seen = Dict{Any,R}()                  # selector => record, within one instant
    run_ts = sorted[1].timestamp
    for (i, r) in enumerate(sorted)
        if r.timestamp != run_ts
            empty!(seen)
            run_ts = r.timestamp
        end
        sel = selector(r)
        prev = get(seen, sel, nothing)
        if prev === nothing
            seen[sel] = r
        elseif prev == r
            keep[i] = false
        else
            throw(ConflictingRecords(R, sel, r.timestamp, prev, r))
        end
    end
    all(keep) || deleteat!(sorted, findall(!, keep))
    return sorted
end
InMemory(rows::AbstractVector{R}) where {R} = InMemory{R}(rows)

kind(::InMemory{R}) where {R} = R

# The rows are the whole world, so a selector with no row is structurally
# absent, not temporally. An empty InMemory therefore serves nothing.
serves(p::InMemory{R}, ::Any, ::Type{R}, sel) where {R} = sel in p.sels

function served_description(p::InMemory)
    isempty(p.sels) && return "InMemory with no rows"
    ss = sort!(String[string(s) for s in p.sels])
    length(ss) <= 8 || return "InMemory rows for " * join(ss[1:8], ", ") *
                              " and $(length(ss) - 8) more"
    "InMemory rows for " * join(ss, ", ")
end

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

One record, visible from its own timestamp -- which the two-argument
curve constructors stamp at the start of time, so the flat-curve case
reads "always known". `asof` returns it only for its own selector
(`selector(record) == sel`) and only at or after its stamp; `between`
and `timestamps` contain it only when the selector matches and its
timestamp lies in the range, so over any real window they are empty.
The natural spec for a flat rate or dividend curve.
"""
struct Constant{R}
    record::R
end

kind(::Constant{R}) where {R} = R

asof(c::Constant{R}, ::Any, ::Type{R}, sel, ts::DateTime) where {R} =
    (selector(c.record) == sel && c.record.timestamp <= ts) ? R[c.record] : R[]

between(c::Constant{R}, ::Any, ::Type{R}, sel, from::DateTime, to::DateTime) where {R} =
    (selector(c.record) == sel && from <= c.record.timestamp <= to) ? R[c.record] : R[]

timestamps(c::Constant{R}, ctx, ::Type{R}, sel, from::DateTime, to::DateTime) where {R} =
    DateTime[r.timestamp for r in between(c, ctx, R, sel, from, to)]

# A constant for SPY says nothing about SPX, so it serves exactly one
# selector; before its stamp it is temporally, not structurally, absent.
serves(c::Constant{R}, ::Any, ::Type{R}, sel) where {R} = selector(c.record) == sel
served_description(c::Constant) = "Constant for $(selector(c.record))"

"""
    inputs(spec) -> Tuple of kinds

The kinds a derived provider reads through the map. `()` for raw
providers. The config loader checks every input kind is present.
"""
inputs(::Any) = ()

"""
    demands(spec) -> iterable of (kind, selector)

The selectors a derived spec needs *statically*, known without a query.
`()` for everything else. `build_market_data` uses it as a load-time
fast path: a mistyped currency fails in a second rather than after a
backtest has been running. It is only the fast path -- the mechanism is
`serves` in the four map-level shapes -- so a spec that cannot name its
selectors ahead of time simply demands nothing.
"""
demands(::Any) = ()

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

# Explicit, not the default: a derived provider does not answer the
# structural question, it delegates. The map-level check waves the quote
# read through, the read reaches the OptionBar entry through the map, and
# that entry's own check throws naming OptionBar and the selector -- the
# real cause, rather than "no quote".
serves(::QuotesFromBars, ::Any, ::Type{OptionQuote}, ::Any) = missing
