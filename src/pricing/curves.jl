"""
    Curve

A function of time: callable with a `DateTime`, returns `Float64`.
Concrete subtypes carry the representation.
"""
abstract type Curve end

(c::Curve)(::DateTime) = error("Curve interface not implemented for $(typeof(c))")

"""
    FlatCurve(value)

Constant curve. `(c)(ts)` returns `c.value` for any `ts`.
"""
struct FlatCurve <: Curve
    value::Float64
end

(c::FlatCurve)(::DateTime) = c.value

"""
    PCCurve(knots, values)

Piecewise-constant curve: `(c)(ts)` is the value at the last knot at or
before `ts`, and `values[1]` before the first knot, so both ends
flat-extrapolate. Throws `ArgumentError` unless `knots` is non-empty,
sorted, unique, and the same length as `values`.
"""
struct PCCurve <: Curve
    knots::Vector{DateTime}
    values::Vector{Float64}

    function PCCurve(knots::AbstractVector{DateTime}, values::AbstractVector{<:Real})
        length(knots) == length(values) ||
            throw(ArgumentError("knots and values must have equal length"))
        isempty(knots) &&
            throw(ArgumentError("PCCurve must have at least one knot"))
        issorted(knots) ||
            throw(ArgumentError("knots must be sorted"))
        allunique(knots) ||
            throw(ArgumentError("knots must be unique"))
        new(collect(DateTime, knots), collect(Float64, values))
    end
end

function (c::PCCurve)(ts::DateTime)
    i = searchsortedlast(c.knots, ts)
    i == 0 && return c.values[1]
    return c.values[i]
end
