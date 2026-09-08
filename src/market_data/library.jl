# `market_data` module: library. Ordinary functions over the protocol's
# results, not part of the protocol.

"""
    only_or_missing(v) -> Union{eltype(v), Missing}

The one record of a singleton result (`at` on a one-per-timestamp kind,
`asof` on a snapshot kind), or `missing` when the result is empty.
Throws `ArgumentError` on more than one record, because a singleton kind
with two records at one timestamp is a data error, not a choice.
"""
only_or_missing(v) = isempty(v) ? missing : only(v)

"""
    by_timestamp(it) -> iterator of (ts::DateTime, Vector{R})

Lazy run-length grouping of a timestamp-sorted iterable of records into
one `(timestamp, records)` pair per distinct timestamp. Consumes `it`
exactly once and one group ahead, so it composes with a lazy `between`
without materializing the range. Throws `ArgumentError` if a later
record carries an earlier timestamp (the input was not sorted).
"""
by_timestamp(it) = ByTimestamp(it)

struct ByTimestamp{I}
    inner::I
end

Base.IteratorSize(::Type{<:ByTimestamp}) = Base.SizeUnknown()
Base.eltype(::Type{ByTimestamp{I}}) where {I} = Tuple{DateTime,Vector{eltype(I)}}

# Records of one kind are one concrete type; when the inner eltype is not
# concrete (a generator), type the group by its first record instead.
function _group_vector(inner, first_rec)
    T = eltype(inner)
    T = isconcretetype(T) ? T : typeof(first_rec)
    v = Vector{T}(undef, 1)
    v[1] = first_rec
    v
end

function Base.iterate(b::ByTimestamp)
    nx = iterate(b.inner)
    nx === nothing && return nothing
    r, st = nx
    _next_group(b, r, st)
end

function Base.iterate(b::ByTimestamp, state)
    pending, st = state
    pending === nothing && return nothing
    _next_group(b, pending, st)
end

# Collect the run of records sharing `first_rec.timestamp`; return it with
# the first record of the next run (or `nothing` at the end) as state.
function _next_group(b::ByTimestamp, first_rec, st)
    ts = first_rec.timestamp
    group = _group_vector(b.inner, first_rec)
    while true
        nx = iterate(b.inner, st)
        nx === nothing && return ((ts, group), (nothing, st))
        r, st = nx
        if r.timestamp == ts
            push!(group, r)
        elseif r.timestamp > ts
            return ((ts, group), (r, st))
        else
            throw(ArgumentError("by_timestamp: input not sorted ($(r.timestamp) after $ts)"))
        end
    end
end
