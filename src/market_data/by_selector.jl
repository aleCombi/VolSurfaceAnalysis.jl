# `market_data` module: composition by selector.
#
# "SPY spots from parquet, SPX spots from csv" inside the one SpotPrice
# entry. The kind is a type parameter, the parts are a tuple of
# `selector => provider` pairs, and routing on a runtime selector yields a
# small union of part types whose shapes all return the same record type,
# so call sites stay inferable (checked in test_by_selector.jl and recorded
# in proposal section 10).

"""
    BySelector{R}(sel => provider, ...)

One `R` entry composed of sub-providers routed by selector. Construction
rejects an empty list, a selector that is not a `selector_type(R)`, a
part whose `kind` is not `R`, and duplicate selectors. Every shape
routes on `sel` and forwards the context untouched; a selector with no
route is structurally absent and throws
[`UnservedSelector`](@ref) naming the routes there are.
"""
struct BySelector{R,P<:Tuple}
    parts::P
    function BySelector{R}(parts::Pair...) where {R}
        isempty(parts) && throw(ArgumentError("BySelector{$R}: no parts"))
        for p in parts
            first(p) isa selector_type(R) || throw(ArgumentError(
                "BySelector{$R}: selector $(repr(first(p))) is not a $(selector_type(R))"))
            kind(last(p)) === R || throw(ArgumentError(
                "BySelector{$R}: part for $(first(p)) serves $(kind(last(p))), not $R"))
        end
        allunique(first.(parts)) || throw(ArgumentError("BySelector{$R}: duplicate selector"))
        new{R,typeof(parts)}(parts)
    end
end

kind(::BySelector{R}) where {R} = R
inputs(b::BySelector) = Tuple(unique(Iterators.flatten(inputs(last(p)) for p in b.parts)))

@inline _route(sel, p::Pair, rest...) = first(p) == sel ? last(p) : _route(sel, rest...)
_route(sel) = nothing

# The routes are the whole world for this entry: no route is structural
# absence. A route that exists delegates -- routing to a parquet spec
# yields `missing`, because the spec cannot answer either.
function serves(b::BySelector{R}, m, ::Type{R}, sel) where {R}
    p = _route(sel, b.parts...)
    p === nothing ? false : serves(p, m, R, sel)
end

served_description(b::BySelector) =
    "BySelector routes for " * join(sort!(String[string(first(p)) for p in b.parts]), ", ")

# Unreachable from the map level (the map-level check throws first) but
# reachable provider-level, which tests and internal delegation use.
@inline function _part(b::BySelector{R}, sel) where {R}
    p = _route(sel, b.parts...)
    p === nothing && throw(UnservedSelector(R, sel, served_description(b)))
    return p
end

at(b::BySelector{R}, m, ::Type{R}, sel, ts::DateTime) where {R} =
    at(_part(b, sel), m, R, sel, ts)
between(b::BySelector{R}, m, ::Type{R}, sel, from::DateTime, to::DateTime) where {R} =
    between(_part(b, sel), m, R, sel, from, to)
asof(b::BySelector{R}, m, ::Type{R}, sel, ts::DateTime) where {R} =
    asof(_part(b, sel), m, R, sel, ts)
timestamps(b::BySelector{R}, m, ::Type{R}, sel, from::DateTime, to::DateTime) where {R} =
    timestamps(_part(b, sel), m, R, sel, from, to)
