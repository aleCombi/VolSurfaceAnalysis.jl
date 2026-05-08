using RecipesBase

@recipe function f(spots::Vector{SpotPrice})
    isempty(spots) && throw(ArgumentError("cannot plot empty spot series"))
    underlyings = unique(s.underlying for s in spots)
    length(underlyings) == 1 ||
        throw(ArgumentError("expected a single underlying, got $(length(underlyings))"))
    u = first(underlyings)
    title --> "$(ticker(u)) spot"
    ylabel --> "Price"
    legend --> false
    [s.timestamp for s in spots], [s.price for s in spots]
end
