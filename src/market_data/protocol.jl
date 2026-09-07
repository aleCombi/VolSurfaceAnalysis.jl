# `market_data` module: the protocol.
#
# Four shapes, one per question a consumer can ask about a kind `R` and a
# selector `sel`. Two arities exist for each:
#
# - map-level, called by consumers:      at(m, R, sel, ts)
# - provider-level, implemented by specs
#   and readers, context first:          at(p, ctx, R, sel, ts)
#
# `ctx` is the map (or time cut) the call came through. Raw providers ignore
# it; derived providers read their inputs through it; composition forwards
# it untouched. That is what makes no-lookahead through derived data
# structural: a provider handed a cut can only see what the cut shows.
#
# Rules (docs/modules/market_data.md):
# - results are sorted by `timestamp`; `at`/`between` return only records
#   in range; `asof` returns every record at the largest visible timestamp
#   <= ts; EMPTY MEANS ABSENT for all four shapes, no shape returns
#   `missing`;
# - `between` promises an iterable, not a container, valid only while its
#   reader is open;
# - `asof` has no default: each provider implements it with what its
#   storage does well;
# - ranges are always bounded; there is no discovery verb.

"""
    at(m, ::Type{R}, sel, ts) -> Vector{R}
    at(p, ctx, ::Type{R}, sel, ts) -> Vector{R}

Every record of kind `R` for selector `sel` with `timestamp == ts`,
sorted. Empty when absent. The provider-level default is
`collect(R, between(p, ctx, R, sel, ts, ts))`; providers override it
when they have a faster path.
"""
function at end

"""
    between(m, ::Type{R}, sel, from, to) -> iterable of R
    between(p, ctx, ::Type{R}, sel, from, to) -> iterable of R

Every record with `from <= timestamp <= to`, sorted. An iterable, not
necessarily a container: large providers yield lazily, one partition in
memory at a time, and the iterator is valid only while the reader is
open.
"""
function between end

"""
    asof(m, ::Type{R}, sel, ts) -> Vector{R}
    asof(p, ctx, ::Type{R}, sel, ts) -> Vector{R}

Every record at the largest visible timestamp `<= ts` (a whole chain for
grid kinds, one record for snapshots), empty when nothing is visible.
No default: each provider implements it with what its storage does well.
"""
function asof end

"""
    timestamps(m, ::Type{R}, sel, from, to) -> Vector{DateTime}
    timestamps(p, ctx, ::Type{R}, sel, from, to) -> Vector{DateTime}

The distinct timestamps at which records of `R` for `sel` exist in
`[from, to]`, sorted.
"""
function timestamps end

"""
    kind(p) -> Type

The kind a provider spec or reader serves.
"""
function kind end

at(p, ctx, ::Type{R}, sel, ts::DateTime) where {R} =
    collect(R, between(p, ctx, R, sel, ts, ts))
