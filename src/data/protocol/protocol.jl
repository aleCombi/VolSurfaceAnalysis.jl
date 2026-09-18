# `data/protocol`: the four read shapes, the structural question `serves`,
# and the named failures.

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

Every record at the largest visible timestamp `<= ts`, empty when
nothing is visible. No default: each provider implements it with what
its storage does well. A derived provider whose input is present but
never builds throws [`DerivationExhausted`](@ref) at its bound; absent
input is an ordinary empty answer.
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

Whether anything here serves selector `sel` for kind `R`. Static, so
there is no timestamp argument. `true` and `false` are answers;
`missing` is "cannot say" and is also the default, so a provider with no
method opts out rather than breaking.
"""
function serves end

serves(::Any, ::Any, ::Type, ::Any) = missing

"""
    served_description(p) -> String

What `p` does serve, in one phrase, for the message on an
[`UnservedSelector`](@ref).
"""
served_description(p) = string(nameof(typeof(p)))

at(p, ctx, ::Type{R}, sel, ts::DateTime) where {R} =
    collect(R, between(p, ctx, R, sel, ts, ts))

"""
    UnservedSelector(kind, selector, served)

Nothing in this configuration serves `selector` for `kind`. `served`
says what the entry does serve.
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

A derived provider walked back `bound` input timestamps from
`requested`, as far as `oldest`, and never produced a record of `kind`.
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
Exact duplicates collapse silently and do not reach this.
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
