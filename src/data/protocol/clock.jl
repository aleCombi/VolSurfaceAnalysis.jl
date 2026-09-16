# `data/protocol`: the tick grid, as a pure value over (kind, selector).

"""
    Clock{R}(sel)

The tick grid "timestamps of kind `R` for selector `sel`". Construction
checks `sel isa selector_type(R)`, so `Clock{OptionQuote}(Currency("USD"))`
fails at once. `timestamps(m, clock, from, to)` enumerates the grid.
"""
struct Clock{R,S}
    sel::S
    function Clock{R}(sel::S) where {R,S}
        sel isa selector_type(R) || throw(ArgumentError(
            "Clock{$R}: selector must be a $(selector_type(R)), got $(typeof(sel))"))
        new{R,S}(sel)
    end
end

kind(::Clock{R}) where {R} = R

timestamps(m, c::Clock{R}, from::DateTime, to::DateTime) where {R} =
    timestamps(m, R, c.sel, from, to)
