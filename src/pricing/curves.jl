"""
    Curve

Math object representing a function in time. Callable with a
`DateTime`, returns `Float64`. Concrete subtypes carry the
representation (constant, piecewise constant, parametric, ...).
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

Piecewise-constant curve. `knots` must be sorted, non-empty,
unique, and have the same length as `values`.

Evaluation at `ts`:
- before the first knot: returns `values[1]`
- at or after knot `i` (and before `i+1`): returns `values[i]`
- at or after the last knot: returns `values[end]`

Out-of-range behavior is flat-extrapolation by construction
(`searchsortedlast` returns 0 below the range; we clamp to 1).
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
