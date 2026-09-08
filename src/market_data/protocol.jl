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
#   <= ts; no shape returns `missing`;
# - EMPTY MEANS TEMPORAL ABSENCE ONLY -- "this selector is served, and has
#   nothing at this instant". STRUCTURAL absence, "nothing here serves this
#   selector at all", is a named error (`UnservedSelector`), because a
#   consumer that cannot tell "not yet" from "not ever" correctly concludes
#   it has nothing to do and the run dies silently. `serves` is the shape
#   that answers the structural question, and the four map-level shapes
#   check it;
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

Read strictly, that is the largest timestamp at which a record of `R`
*exists* -- which for a derived kind is not necessarily where its input
exists. A derived provider whose build fails at the newest input instant
must keep walking back, under a bound; exhausting the bound throws
[`DerivationExhausted`](@ref) rather than reporting a broken feed through
the same channel as ordinary missing data.
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

"""
    serves(m, ::Type{R}, sel) -> Union{Bool,Missing}
    serves(p, ctx, ::Type{R}, sel) -> Union{Bool,Missing}

Whether anything here serves selector `sel` for kind `R`. Static by
definition, so there is no timestamp argument: the answer cannot vary
with one.

Three-valued, and the third value is required. `true` and `false` are
answers; `missing` is "cannot say", returned by

- a **parquet spec**, which would have to walk a tree it has not opened
  (and `build_market_data` holds specs, not readers); and
- every **derived provider**, by decision. It delegates rather than
  answering, so its input's error propagates and the failure names the
  real cause: a `SurfaceFrom` asked for SPX reports `OptionBar`/SPX
  unserved, not "no surface".

The default is `missing`, so a provider with no method opts out of the
check rather than breaking. `Union{Bool,Missing}` is deliberate: Julia's
three-valued `&` already gives `missing & false === false`, so a future
provider delegating conjunctively over several inputs gets Kleene
semantics for free.
"""
function serves end

serves(::Any, ::Any, ::Type, ::Any) = missing

"""
    served_description(p) -> String

What `p` does serve, in one phrase, for the message on an
[`UnservedSelector`](@ref). A bare key error naming only the selector
does not say enough to fix a config, which is the point of the named
error.
"""
served_description(p) = string(nameof(typeof(p)))

at(p, ctx, ::Type{R}, sel, ts::DateTime) where {R} =
    collect(R, between(p, ctx, R, sel, ts, ts))

# --- Errors ---------------------------------------------------------------
#
# Named failures for the questions the protocol cannot answer with an
# empty result. Empty is reserved for "served, and nothing at this
# instant"; everything else has a type and says enough to fix the cause.

"""
    UnservedSelector(kind, selector, served)

Nothing in this configuration serves `selector` for `kind`. Structural,
not temporal: it does not vary with the instant asked for, so it is a
configuration-grade problem and gets a name rather than an empty vector.
`served` says what the entry does serve.
"""
struct UnservedSelector <: Exception
    kind     :: Type
    selector :: Any
    served   :: String
end

Base.showerror(io::IO, e::UnservedSelector) = print(io,
    "UnservedSelector: nothing serves $(e.kind) for $(e.selector). ",
    "The entry: $(e.served)")

"""
    DerivationExhausted(kind, selector, requested, oldest, bound)

A derived provider walked back `bound` input timestamps from `requested`,
as far as `oldest`, and never produced a record of `kind`. The input is
present and the derivation keeps failing, which is a truncated dataset or
a broken feed rather than absence.
"""
struct DerivationExhausted <: Exception
    kind      :: Type
    selector  :: Any
    requested :: DateTime
    oldest    :: DateTime
    bound     :: Int
end

Base.showerror(io::IO, e::DerivationExhausted) = print(io,
    "DerivationExhausted: no $(e.kind) for $(e.selector) at or before $(e.requested) ",
    "within $(e.bound) input timestamp(s); the oldest tried was $(e.oldest). ",
    "The input is present and the derivation kept failing.")

"""
    ConflictingRecords(kind, selector, timestamp, a, b)

Two records of `kind` for `selector` at the same `timestamp` disagree.
Exact duplicates collapse silently; a store that disagrees with itself
about a value is worth stopping for, because taking the first is a
silent choice between two answers.
"""
struct ConflictingRecords <: Exception
    kind      :: Type
    selector  :: Any
    timestamp :: DateTime
    a         :: Any
    b         :: Any
end

Base.showerror(io::IO, e::ConflictingRecords) = print(io,
    "ConflictingRecords: two $(e.kind) records for $(e.selector) at $(e.timestamp) ",
    "disagree ($(e.a) vs $(e.b))")
