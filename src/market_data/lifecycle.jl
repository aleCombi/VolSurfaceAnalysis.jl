# `market_data` module: lifecycle.
#
# A reader is the opened form of a spec: it owns what the storage needs at
# run time (a connection, bounded caches, a partition list). Specs that
# need nothing are their own reader. The pair is project-owned, with NO
# fallback on Any: a spec without an explicit `open_data` method is a
# load-time error, never a silent no-op. The run opens and closes;
# `Experiment` holds the spec map only.

"""
    open_data(spec) -> reader

The opened form of `spec`. Resource-free specs return themselves. A
composite (`MarketData`, `BySelector`) opens its parts in order and, if
one fails, closes the ones already opened (quietly, so the original
error propagates) before rethrowing.
"""
function open_data end

"""
    close_data!(reader) -> nothing

Release what `open_data` acquired. On a composite every part is closed
in reverse order even if one throws; the first error is rethrown after
the loop. Use after close is the storage's own error.
"""
function close_data! end

# Explicit opt-in, one line per resource-free spec.
open_data(s::Union{InMemory,Constant,QuotesFromBars}) = s
close_data!(::Union{InMemory,Constant,QuotesFromBars}) = nothing

# Best-effort close during an unwind: warn and swallow, so the error that
# caused the unwind is the one the caller sees.
function _close_quietly(r)
    try
        close_data!(r)
    catch e
        @warn "close_data! failed during unwind" reader = typeof(r) exception = e
    end
    nothing
end

# Recursive, type-stable tuple open with unwind. Each level opens its head,
# then the tail; a failure in the tail closes the head and rethrows.
_open_all() = ()
function _open_all(s, rest...)
    r = open_data(s)
    tail = try
        _open_all(rest...)
    catch
        _close_quietly(r)
        rethrow()
    end
    (r, tail...)
end

# Reverse order, every close attempted, first error rethrown after the loop.
function _close_all_best_effort(readers)
    err = nothing
    for r in reverse(readers)
        try
            close_data!(r)
        catch e
            err === nothing && (err = e)
        end
    end
    err === nothing || throw(err)
    nothing
end

function open_data(b::BySelector{R}) where {R}
    readers = _open_all(map(last, b.parts)...)
    BySelector{R}(map(=>, map(first, b.parts), readers)...)
end
close_data!(b::BySelector) = _close_all_best_effort(map(last, b.parts))

open_data(m::MarketData) = MarketData(_open_all(m.entries...))
close_data!(m::MarketData) = _close_all_best_effort(m.entries)

"""
    with_data(f, m::MarketData)

Open `m`, call `f(readers)`, close. If `f` throws, the readers are
closed quietly and `f`'s error propagates; on success a close error
propagates normally.
"""
function with_data(f, m::MarketData)
    d = open_data(m)
    result = try
        f(d)
    catch
        _close_quietly(d)
        rethrow()
    end
    close_data!(d)
    result
end

"""
    has_lifecycle(spec) -> Bool

Whether `spec` has an explicit `open_data` method. Checked by the
config loader; the lifecycle test suite is the real guarantee that the
reader side closes.
"""
has_lifecycle(spec) = hasmethod(open_data, Tuple{typeof(spec)})
