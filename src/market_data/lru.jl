# `market_data` module: a bounded LRU cache for readers.
#
# OrderedDict-backed: insertion order is recency, a hit moves the key to the
# back, inserting past `max` pops the front. Small enough that a dependency
# (LRUCache.jl) would cost a resolve/precompile cycle for nothing.

using OrderedCollections

struct LRU{K,V}
    d::OrderedDict{K,V}
    max::Int
    function LRU{K,V}(max::Integer) where {K,V}
        max >= 1 || throw(ArgumentError("LRU: max must be >= 1, got $max"))
        new{K,V}(OrderedDict{K,V}(), Int(max))
    end
end

"""
    get!(f, cache::LRU, key)

The cached value for `key`, touching it as most recently used; otherwise
`f()`, inserted, evicting the least recently used entries past the
bound. Nothing is inserted if `f` throws.
"""
function Base.get!(f, c::LRU{K,V}, k) where {K,V}
    if haskey(c.d, k)
        v = c.d[k]
        delete!(c.d, k)
        c.d[k] = v
        return v
    end
    v = f()::V
    c.d[k] = v
    while length(c.d) > c.max
        popfirst!(c.d)
    end
    v
end

Base.haskey(c::LRU, k) = haskey(c.d, k)
Base.length(c::LRU) = length(c.d)
Base.keys(c::LRU) = keys(c.d)
Base.empty!(c::LRU) = (empty!(c.d); c)
